"""Check the S3 replacements in analyzers.technical / analyzers.image_cache on real photos (spec S3, SC5).

usage: verify_technical_exact.py [--photos DIR] [--expect N] [--reps N]
env:   PERF_REPO   (required) checkout whose analyzers are checked

For every photo in DIR (default D:\\facet-scratch\\perf2\\photos) decodes it with the repo's
load_image_from_path and compares, against the frozen reference in tests/_technical_reference.py:

* get_dynamic_range, get_noise_estimate, get_contrast_score (cache and no-cache paths): the result
  dicts must be EXACTLY equal (same keys, same types, same bit pattern for every float);
* ImageCache.laplacian_variance: relative difference <= 1e-12 (the one tolerance, spec R5), and the
  get_sharpness_data normalised score equal at 2 decimals.

Then times the four computations together (Laplacian variance + dynamic range + noise + contrast,
grey/HSV already built in both arms) as new/reference over --reps repetitions (default 3, medians of
per-repetition means) and requires ratio <= 0.25.

Exit codes: 0 only when `compared N photos` was printed with N >= --expect (default 49), nothing
differed and the ratio is <= 0.25; 1 on any difference or a ratio above 0.25; 2 when fewer than
--expect photos could be compared; 3 when PERF_REPO is not a checkout or the analyzers module is not
under it.
"""
import os
import sys

REPO = os.environ.get('PERF_REPO', '')
if not REPO or not os.path.isfile(os.path.join(REPO, 'processing', 'multi_pass.py')):
    print('verify_technical_exact: PERF_REPO=%r has no processing/multi_pass.py' % REPO, file=sys.stderr)
    sys.exit(3)
REPO = os.path.realpath(REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

import statistics  # noqa: E402
import struct  # noqa: E402
import time  # noqa: E402

RATIO_LIMIT = 0.25
LAPLACIAN_REL_TOL = 1e-12


def _identical(new, old):
    """Return None when the dicts are exactly equal, else a one-line reason."""
    if list(new) != list(old):
        return 'keys %s vs %s' % (list(new), list(old))
    for key, b in old.items():
        a = new[key]
        if type(a) is not type(b):
            return '%s: type %s vs %s' % (key, type(a).__name__, type(b).__name__)
        if isinstance(b, float):
            if struct.pack('<d', float(a)) != struct.pack('<d', float(b)):
                return '%s: %r vs %r' % (key, a, b)
        elif a != b:
            return '%s: %r vs %r' % (key, a, b)
    return None


def main(argv):
    photos = r'D:\facet-scratch\perf2\photos'
    expect = 49
    reps = 3
    i = 0
    while i < len(argv):
        if argv[i] == '--photos':
            photos = argv[i + 1]
        elif argv[i] == '--expect':
            expect = int(argv[i + 1])
        elif argv[i] == '--reps':
            reps = int(argv[i + 1])
        else:
            print(__doc__)
            return 2
        i += 2

    import analyzers.technical as technical_module
    f = os.path.realpath(technical_module.__file__)
    if os.path.commonpath([f, REPO]) != REPO:
        print('module %s is not under PERF_REPO %s' % (f, REPO), file=sys.stderr)
        return 3
    import cv2
    import numpy as np
    from analyzers.image_cache import ImageCache, laplacian_variance
    from analyzers.technical import TechnicalAnalyzer
    from tests import _technical_reference as ref
    from utils import load_image_from_path

    names = sorted(n for n in os.listdir(photos) if n.lower().endswith(('.jpg', '.jpeg')))
    tech = TechnicalAnalyzer()
    compared = 0
    failures = []
    inputs = []
    for name in names:
        _pil, img = load_image_from_path(os.path.join(photos, name))
        if img is None:
            print('could not decode %s' % name, file=sys.stderr)
            continue
        cache, old_cache = ImageCache(img), ref.OldImageCache(img)
        with np.errstate(all='ignore'):
            for label, new, old in (
                ('dynamic_range', tech.get_dynamic_range(img, cache=cache), ref.old_dynamic_range(img, cache=old_cache)),
                ('noise', tech.get_noise_estimate(img, cache=cache), ref.old_noise_estimate(img, cache=old_cache)),
                ('contrast', tech.get_contrast_score(img, cache=cache), ref.old_contrast_score(img, cache=old_cache)),
                ('dynamic_range[no cache]', tech.get_dynamic_range(img), ref.old_dynamic_range(img)),
                ('noise[no cache]', tech.get_noise_estimate(img), ref.old_noise_estimate(img)),
                ('contrast[no cache]', tech.get_contrast_score(img), ref.old_contrast_score(img)),
            ):
                why = _identical(new, old)
                if why:
                    failures.append('%s %s: %s' % (name, label, why))
        new_v, old_v = cache.laplacian_variance, old_cache.laplacian_variance
        rel = abs(new_v - old_v) / abs(old_v) if old_v else abs(new_v - old_v)
        if not isinstance(new_v, float) or rel > LAPLACIAN_REL_TOL:
            failures.append('%s laplacian_variance: %r vs %r (rel %.3g)' % (name, new_v, old_v, rel))
        s_new = round(tech.get_sharpness_data(img, cache=cache)['normalized'], 2)
        s_old = round(ref.old_sharpness_data(img, cache=old_cache)['normalized'], 2)
        if s_new != s_old:
            failures.append('%s sharpness normalized: %r vs %r' % (name, s_new, s_old))
        compared += 1
        inputs.append((img, cache, old_cache, rel))

    print('compared %d photos' % compared)
    if inputs:
        print('max laplacian_variance relative difference: %.3g' % max(r for *_x, r in inputs))
    for line in failures[:20]:
        print('DIFF', line)

    def timed(run):
        means = []
        for _ in range(reps):
            t0 = time.perf_counter()
            for img, cache, old_cache, _rel in inputs:
                run(img, cache, old_cache)
            means.append((time.perf_counter() - t0) / max(len(inputs), 1))
        return statistics.median(means)

    def run_new(img, cache, _old):
        laplacian_variance(cache.gray)
        tech.get_dynamic_range(img, cache=cache)
        tech.get_noise_estimate(img, cache=cache)
        tech.get_contrast_score(img, cache=cache)

    def run_old(img, _cache, old_cache):
        cv2.Laplacian(old_cache.gray, cv2.CV_64F).var()
        ref.old_dynamic_range(img, cache=old_cache)
        ref.old_noise_estimate(img, cache=old_cache)
        ref.old_contrast_score(img, cache=old_cache)

    ratio = None
    if inputs:
        # Warm both arms once so allocator and thread-pool start-up do not land in one of them.
        run_new(*inputs[0][:3])
        run_old(*inputs[0][:3])
        t_old = timed(run_old)
        t_new = timed(run_new)
        ratio = t_new / t_old
        print('technical metrics per photo: reference %.1f ms, new %.1f ms, ratio %.3f (limit %.2f)'
              % (t_old * 1000, t_new * 1000, ratio, RATIO_LIMIT))

    if compared < expect:
        print('only %d of the expected %d photos were compared' % (compared, expect), file=sys.stderr)
        return 2
    if failures:
        print('%d difference(s)' % len(failures))
        return 1
    if ratio is None or ratio > RATIO_LIMIT:
        print('ratio above %.2f' % RATIO_LIMIT)
        return 1
    print('exact')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
