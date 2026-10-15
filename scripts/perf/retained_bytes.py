"""Measure what ChunkedMultiPassProcessor._load_images retains per photo (SC4).

Port of D:\\facet-scratch\\chunk\\measure_load.py. Builds the processor with __new__, sets load_workers
and on_error, calls the repo's own _load_images on a stub scorer and walks the returned dict bucket by
bucket. No database, no GPU, no photo is written.

usage: retained_bytes.py LABEL WORKERS SELECTOR [tracemalloc] [--photos DIR] [--out DIR]
       retained_bytes.py LABEL --saliency-live-masks N [--out DIR]

env:   PERF_REPO   (required) checkout whose multi_pass.py is measured
SELECTOR  'first:N'      first N files by name (the profile's chunk 0 is first:32)
          'mp:LO-HI:N'   first N files whose megapixels are in [LO, HI] (header sizes only)
--photos  photo directory (default D:\\facet-scratch\\perf2\\photos); inventory.json is generated from it
          (PIL header read, no decode) into --out
--out     output directory (default D:\\facet-scratch\\perf2\\retained)

Output (stdout and OUT/out_<LABEL>.json): n, mp_total, retained_wset_mb, bytes_per_px_retained_wset (working-set
delta over the pixels loaded), buckets_bytes_per_px (per bucket of the returned dict: 'cv:ndarray',
'cache.gray:ndarray', 'pil:PIL(RGB)', ...), peak/transient figures, after-release figures.

Gates (exit 1 on failure, reason on stderr): --expect LO-HI (bytes_per_px_retained_wset), --has-bucket NAME and
--no-bucket NAME (substring of a bucket key, repeatable), --max-live K (live-masks mode). A photo that
failed to load is also exit 1.

--saliency-live-masks N: no photo decode. Runs the repo's real SaliencyScorer.score_batch /
_pass_saliency over N small synthetic images with only the model forward (`_run_chunk`) stubbed to
return tracked masks, at a fixed sub-batch of 8, and reports max_live_masks, the largest number of
masks alive at once. Before S9 that is N (the whole list is held); after it must be <= one sub-batch.
It drives `_pass_saliency` with the image dict shape of the tree under test; a later step that changes
that shape updates this mode in the same change.

Exit 0 on success, 2 on a usage error, 3 when PERF_REPO is not a checkout or the module imported is not
under it.
"""
import gc
import json
import os
import sys
import threading
import time
import weakref

REPO = os.environ.get('PERF_REPO', '')
if not REPO or not os.path.isfile(os.path.join(REPO, 'processing', 'multi_pass.py')):
    print('retained_bytes: PERF_REPO=%r has no processing/multi_pass.py' % REPO, file=sys.stderr)
    sys.exit(3)
REPO = os.path.realpath(REPO)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import psutil  # noqa: E402

DEFAULT_PHOTOS = r'D:\facet-scratch\perf2\photos'
DEFAULT_OUT = r'D:\facet-scratch\perf2\retained'


def usage():
    print(__doc__)
    return 2


def parse(argv):
    opts = {'photos': DEFAULT_PHOTOS, 'out': DEFAULT_OUT, 'live': None, 'tm': False,
            'expect': None, 'has': [], 'no': [], 'max_live': None}
    pos = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == '--photos':
            opts['photos'] = argv[i + 1]
            i += 2
        elif a == '--out':
            opts['out'] = argv[i + 1]
            i += 2
        elif a == '--saliency-live-masks':
            opts['live'] = int(argv[i + 1])
            i += 2
        elif a == '--expect':
            opts['expect'] = tuple(float(x) for x in argv[i + 1].split('-'))
            i += 2
        elif a == '--has-bucket':
            opts['has'].append(argv[i + 1])
            i += 2
        elif a == '--no-bucket':
            opts['no'].append(argv[i + 1])
            i += 2
        elif a == '--max-live':
            opts['max_live'] = int(argv[i + 1])
            i += 2
        elif a == 'tracemalloc':
            opts['tm'] = True
            i += 1
        else:
            pos.append(a)
            i += 1
    return opts, pos


def import_checked():
    import processing.multi_pass as mp
    f = os.path.realpath(mp.__file__)
    if os.path.commonpath([f, REPO]) != REPO:
        print('retained_bytes: processing.multi_pass imported from %s, not from %s' % (f, REPO), file=sys.stderr)
        sys.exit(3)
    return mp


def inventory(photos, out):
    from PIL import Image
    inv = []
    for name in sorted(os.listdir(photos)):
        p = os.path.join(photos, name)
        if not os.path.isfile(p) or not name.lower().endswith(('.jpg', '.jpeg')):
            continue
        with Image.open(p) as im:
            w, h = im.size
            mode = im.mode
        inv.append({'name': name, 'w': w, 'h': h, 'mp': w * h / 1e6, 'mode': mode,
                    'file_mb': os.path.getsize(p) / 1e6})
    os.makedirs(out, exist_ok=True)
    json.dump(inv, open(os.path.join(out, 'inventory.json'), 'w'), indent=1)
    return inv


def select(inv, selector):
    if selector.startswith('first:'):
        return sorted(inv, key=lambda r: r['name'])[:int(selector.split(':')[1])]
    _, rng, n = selector.split(':')
    lo, hi = [float(x) for x in rng.split('-')]
    return [r for r in sorted(inv, key=lambda r: r['name']) if lo <= r['mp'] <= hi][:int(n)]


def real_config():
    from config.scoring_config import ScoringConfig
    return ScoringConfig(config_path=os.path.join(REPO, 'config', 'scoring_config.default.json'), validate=False)


class _Scorer:
    """Stub scorer: a real ScoringConfig (the composition hoist reads it) and no EXIF."""

    def __init__(self, config):
        self.config = config

    def get_exif_data(self, path):
        return {}


def sizeof(o):
    """Bytes held by o, by type bucket."""
    from PIL import Image
    out = {}

    def add(k, v):
        out[k] = out.get(k, 0) + v

    def rec(name, x):
        if isinstance(x, Image.Image):
            add(name + ':PIL(%s)' % x.mode, x.width * x.height * len(x.getbands()))
        elif isinstance(x, np.ndarray):
            add(name + ':ndarray', x.nbytes)
        elif isinstance(x, (bytes, bytearray)):
            add(name + ':bytes', len(x))
        elif isinstance(x, dict):
            for kk, vv in x.items():
                rec(name + '.' + str(kk), vv)
        elif isinstance(x, (list, tuple)):
            for vv in x:
                rec(name, vv)
        elif hasattr(x, '__slots__'):
            for s in x.__slots__:
                rec(name + '.' + s, getattr(x, s, None))
        else:
            add(name + ':py', sys.getsizeof(x))
    rec('', o)
    return out


def measure(label, workers, selector, opts):
    mp = import_checked()
    inv = inventory(opts['photos'], opts['out'])
    sel = select(inv, selector)
    if not sel:
        print('retained_bytes: selector %r matched no photo in %s' % (selector, opts['photos']), file=sys.stderr)
        return 2
    paths = [os.path.join(opts['photos'], r['name']) for r in sel]
    total_px = sum(r['w'] * r['h'] for r in sel)
    os.chdir(REPO)   # exiftool.exe is found next to the repo, as in a real scan

    p = mp.ChunkedMultiPassProcessor.__new__(mp.ChunkedMultiPassProcessor)
    p.scorer = _Scorer(real_config())
    p.load_workers = workers
    p.restricted = False
    errors = []
    p.on_error = lambda *a: errors.append(a) or print('ON_ERROR', a)

    proc = psutil.Process()

    def snap():
        m = proc.memory_info()
        return {'wset': m.wset, 'private': m.private}

    # warm: import heavy modules and decode one photo so lazy imports do not pollute the delta
    mp._ensure_imports()
    from analyzers.form_facet import warmup
    warmup()
    warm = p._load_images([paths[0]])
    del warm
    gc.collect()
    time.sleep(0.5)

    base = snap()
    peak = dict(base)
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            s = snap()
            peak['wset'] = max(peak['wset'], s['wset'])
            peak['private'] = max(peak['private'], s['private'])
            time.sleep(0.02)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    if opts['tm']:
        import tracemalloc
        tracemalloc.start(8)
    t0 = time.time()
    images = p._load_images(paths)
    dt = time.time() - t0
    stop.set()
    th.join()
    time.sleep(0.3)
    after2 = snap()

    buckets = {}
    for _, d in images.items():
        for k, v in d.items():
            for bk, bv in sizeof(v).items():
                buckets[k + bk] = buckets.get(k + bk, 0) + bv
    n = len(images)
    retained_wset = after2['wset'] - base['wset']
    retained_priv = after2['private'] - base['private']
    res = {
        'label': label, 'repo': REPO, 'workers': workers, 'selector': selector, 'n': n, 'n_requested': len(paths),
        'on_errors': len(errors), 'mp_total': total_px / 1e6, 'mp_mean': total_px / 1e6 / max(n, 1),
        'load_s': dt, 's_per_photo': dt / max(n, 1), 'base_wset_mb': base['wset'] / 2 ** 20,
        'retained_wset_mb': retained_wset / 2 ** 20, 'retained_wset_mb_per_photo': retained_wset / 2 ** 20 / max(n, 1),
        'retained_private_mb': retained_priv / 2 ** 20,
        'retained_private_mb_per_photo': retained_priv / 2 ** 20 / max(n, 1),
        'peak_wset_over_base_mb': (peak['wset'] - base['wset']) / 2 ** 20,
        'peak_private_over_base_mb': (peak['private'] - base['private']) / 2 ** 20,
        'transient_above_retained_mb_wset': (peak['wset'] - after2['wset']) / 2 ** 20,
        'bytes_per_px_retained_wset': retained_wset / total_px,
        'buckets_mb_total': {k: v / 2 ** 20 for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])},
        'buckets_bytes_per_px': {k: v / total_px for k, v in sorted(buckets.items(), key=lambda kv: -kv[1])
                                 if v / total_px > 0.01},
    }
    if opts['tm']:
        import tracemalloc
        snapshot = tracemalloc.take_snapshot()
        cur, pk = tracemalloc.get_traced_memory()
        res['tracemalloc_current_mb'] = cur / 2 ** 20
        res['tracemalloc_peak_mb'] = pk / 2 ** 20
        res['tracemalloc_top'] = [(str(s.traceback[0]), s.size / 2 ** 20) for s in snapshot.statistics('lineno')[:12]]
    del images
    gc.collect()
    time.sleep(0.5)
    rel = snap()
    res['after_release_wset_over_base_mb'] = (rel['wset'] - base['wset']) / 2 ** 20
    res['after_release_private_over_base_mb'] = (rel['private'] - base['private']) / 2 ** 20
    rc = emit(res, label, opts)
    fails = []
    if opts['expect'] is not None:
        lo, hi = opts['expect']
        if not lo <= res['bytes_per_px_retained_wset'] <= hi:
            fails.append('bytes_per_px_retained_wset %.3f outside %s-%s' % (res['bytes_per_px_retained_wset'], lo, hi))
    keys = list(res['buckets_bytes_per_px'])
    for name in opts['has']:
        if not any(name in k for k in keys):
            fails.append('no bucket containing %r' % name)
    for name in opts['no']:
        if any(name in k for k in keys):
            fails.append('unexpected bucket containing %r' % name)
    if res['n'] != res['n_requested']:
        fails.append('only %d of %d photos loaded' % (res['n'], res['n_requested']))
    for f in fails:
        print('EXPECTATION FAILED: ' + f, file=sys.stderr)
    return 1 if fails else rc


class _TrackedMask(np.ndarray):
    """A mask the stub forward pass produced; instances are counted while alive."""


LIVE = {'n': 0, 'max': 0, 'created': 0}


def _released():
    LIVE['n'] -= 1


def _tracked_mask(h, w):
    m = np.zeros((h, w), np.uint8)
    m[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    t = m.view(_TrackedMask)
    LIVE['n'] += 1
    LIVE['created'] += 1
    LIVE['max'] = max(LIVE['max'], LIVE['n'])
    weakref.finalize(t, _released)
    return t


def live_masks(label, n, opts):
    mp = import_checked()
    from PIL import Image
    from models.saliency_scorer import SaliencyScorer
    sub_batch = 8
    scorer = SaliencyScorer(device='cpu')
    scorer._loaded = True
    scorer._affordable_batch_size = lambda requested: sub_batch
    scorer._run_chunk = lambda chunk, sizes: [_tracked_mask(h, w) for (w, h) in sizes]
    images, results = {}, {}
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        pil = Image.fromarray(arr, 'RGB')
        path = 'synthetic_%03d.jpg' % i
        images[path] = {'pil': pil, 'cv': arr[:, :, ::-1].copy(), 'path': path}
        results[path] = {}
    p = mp.ChunkedMultiPassProcessor.__new__(mp.ChunkedMultiPassProcessor)
    p._pass_saliency(scorer, images, results)
    gc.collect()
    failed = [k for k, v in results.items() if mp.SALIENCY_FAILED_KEY in v]
    res = {'label': label, 'repo': REPO, 'mode': 'saliency-live-masks', 'n': n, 'sub_batch': sub_batch,
           'masks_created': LIVE['created'], 'max_live_masks': LIVE['max'], 'live_after': LIVE['n'],
           'photos_scored': n - len(failed), 'photos_failed': len(failed)}
    rc = emit(res, label, opts)
    if opts['max_live'] is not None and LIVE['max'] > opts['max_live']:
        print('EXPECTATION FAILED: %d masks alive at once (> %d)' % (LIVE['max'], opts['max_live']), file=sys.stderr)
        return 1
    return rc


def emit(res, label, opts):
    os.makedirs(opts['out'], exist_ok=True)
    text = json.dumps(res, indent=1)
    print(text)
    json.dump(res, open(os.path.join(opts['out'], 'out_%s.json' % label), 'w'), indent=1)
    return 0


def main(argv):
    opts, pos = parse(argv)
    if opts['live'] is not None:
        if len(pos) != 1:
            return usage()
        return live_masks(pos[0], opts['live'], opts)
    if len(pos) != 3:
        return usage()
    return measure(pos[0], int(pos[1]), pos[2], opts)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
