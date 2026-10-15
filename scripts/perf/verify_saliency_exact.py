"""Verify the S2 saliency rewrite on REAL BiRefNet masks (spec R11, SC5).

usage: verify_saliency_exact.py [--photos DIR] [--masks DIR] [--reps N] [--max-ratio R] [--expect N]
env:   PERF_REPO   (required) checkout whose models/saliency_scorer.py and tests/_saliency_reference.py are used

For every photo that has a mask PNG (written by capture_saliency_masks.py) it decodes the photo with the
repo's load_image_from_path, then scores (mask, image) with the verbatim OLD code
(tests/_saliency_reference.py) and with the repo's SaliencyScorer._score_from_mask, as a BGR array and as the
gray plane the S9 rebuilder passes, and compares the returned dicts: same keys, same Python types (an np.float64
fails against a float) and the same float bit patterns. bbox_from_mask is compared the same way. Timing is the
best of --reps runs per photo; the verdict ratio is total new time / total reference time.

Also prints the largest relative deviation of the UNROUNDED subject / background Laplacian variances (the only
place the two formulations can differ), and which photos had which mask coverage.

Exit codes: 0 only when N/N dicts are identical (N == --expect, default 49) and the time ratio is
<= --max-ratio (default 0.25); 1 on any mismatch or a ratio above the limit; 2 when fewer than --expect
photo/mask pairs exist or a mask does not match its photo's size; 3 when PERF_REPO is not a checkout.
Defaults: --photos D:\\facet-scratch\\perf2\\photos  --masks D:\\facet-scratch\\perf2\\masks  --reps 3
No GPU is used.
"""
import os
import struct
import sys
import time

REPO = os.environ.get('PERF_REPO', '')
if not REPO or not os.path.isfile(os.path.join(REPO, 'models', 'saliency_scorer.py')):
    print('verify_saliency_exact: PERF_REPO=%r has no models/saliency_scorer.py' % REPO, file=sys.stderr)
    sys.exit(3)
REPO = os.path.realpath(REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import json  # noqa: E402


def _bits(x):
    return struct.pack('<d', x)


def diffs(a, b, where=''):
    """Strict structural differences between two result values (type identity, float bit patterns)."""
    if type(a) is not type(b):
        return ['%s: type %s != %s (%r vs %r)' % (where, type(a).__name__, type(b).__name__, a, b)]
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            return ['%s: keys %s != %s' % (where, list(a), list(b))]
        out = []
        for k in a:
            out += diffs(a[k], b[k], '%s.%s' % (where, k))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return ['%s: len %d != %d' % (where, len(a), len(b))]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diffs(x, y, '%s[%d]' % (where, i))
        return out
    if isinstance(a, float):
        return [] if _bits(a) == _bits(b) else ['%s: %r != %r' % (where, a, b)]
    return [] if a == b else ['%s: %r != %r' % (where, a, b)]


def main(argv):
    photos = r'D:\facet-scratch\perf2\photos'
    masks_dir = r'D:\facet-scratch\perf2\masks'
    reps = 3
    max_ratio = 0.25
    expect = 49
    i = 0
    while i < len(argv):
        opt = argv[i]
        if opt not in ('--photos', '--masks', '--reps', '--max-ratio', '--expect') or i + 1 >= len(argv):
            print(__doc__)
            return 2
        val = argv[i + 1]
        if opt == '--photos':
            photos = val
        elif opt == '--masks':
            masks_dir = val
        elif opt == '--reps':
            reps = int(val)
        elif opt == '--max-ratio':
            max_ratio = float(val)
        else:
            expect = int(val)
        i += 2

    import cv2
    import numpy as np
    from PIL import Image
    import models.saliency_scorer as ss
    from models.saliency_scorer import SaliencyScorer, bbox_from_mask
    from tests._saliency_reference import ReferenceSaliency, reference_bbox_from_mask
    from utils import load_image_from_path

    f = os.path.realpath(ss.__file__)
    if os.path.commonpath([f, REPO]) != REPO:
        print('module %s is not under PERF_REPO %s' % (f, REPO), file=sys.stderr)
        return 3

    cfg_path = os.path.join(REPO, 'config', 'scoring_config.default.json')
    min_px = json.load(open(cfg_path))['models']['saliency']['min_subject_pixels']
    ref = ReferenceSaliency(min_px)
    new = SaliencyScorer.__new__(SaliencyScorer)
    new.min_subject_pixels = min_px
    new.mask_threshold = 0.3
    ss._ensure_imports()

    names = sorted(n for n in os.listdir(photos) if n.lower().endswith(('.jpg', '.jpeg')))
    pairs = [(n, os.path.join(masks_dir, os.path.splitext(n)[0] + '.png')) for n in names]
    pairs = [(n, m) for n, m in pairs if os.path.isfile(m)]
    print('pairs: %d photos with a mask (expect %d)' % (len(pairs), expect))
    if len(pairs) < expect:
        print('FEWER THAN %d photo/mask pairs: run capture_saliency_masks.py first' % expect, file=sys.stderr)
        return 2

    identical = 0
    failures = []
    t_ref = t_new = 0.0
    max_rel_s = max_rel_b = 0.0
    for name, mpath in pairs:
        pil, bgr = load_image_from_path(os.path.join(photos, name))
        mask = np.array(Image.open(mpath))
        if pil is None or mask.shape != bgr.shape[:2] or mask.dtype != np.uint8:
            print('SIZE/DTYPE MISMATCH %s: mask %s %s vs image %s' % (
                name, mask.shape, mask.dtype, None if pil is None else bgr.shape[:2]), file=sys.stderr)
            return 2
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        want = ref._score_from_mask(mask, bgr)
        problems = diffs(want, new._score_from_mask(mask, bgr), 'bgr')
        problems += diffs(want, new._score_from_mask(mask, gray), 'gray')
        problems += diffs(reference_bbox_from_mask(mask, min_px), bbox_from_mask(mask, min_px), 'bbox')
        if problems:
            failures.append((name, problems))
        else:
            identical += 1

        # unrounded variance deviation (the one place the formulations differ)
        sub = mask > 128
        if sub.any() and not sub.all():
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            v_s, v_b = float(np.var(lap[sub])), float(np.var(lap[~sub]))
            n_s, n_b = new._region_variances(gray, cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)[1], mask.size)
            if v_s > 0:
                max_rel_s = max(max_rel_s, abs(n_s - v_s) / v_s)
            if v_b > 0:
                max_rel_b = max(max_rel_b, abs(n_b - v_b) / v_b)

        best_ref = best_new = float('inf')
        for _ in range(reps):
            t = time.perf_counter()
            ref._score_from_mask(mask, bgr)
            best_ref = min(best_ref, time.perf_counter() - t)
            t = time.perf_counter()
            new._score_from_mask(mask, bgr)
            best_new = min(best_new, time.perf_counter() - t)
        t_ref += best_ref
        t_new += best_new
        print('%-44s %5.1f MP  cover %5.1f%%  ref %6.1f ms  new %6.1f ms  %s' % (
            name[:44], mask.size / 1e6, 100.0 * np.count_nonzero(mask) / mask.size,
            best_ref * 1000, best_new * 1000, 'identical' if not problems else 'DIFFERS'))

    ratio = t_new / t_ref if t_ref else float('inf')
    print('identical dicts: %d/%d' % (identical, len(pairs)))
    print('unrounded variance max relative deviation: subject %.2e, background %.2e' % (max_rel_s, max_rel_b))
    print('time: reference %.1f ms/photo, new %.1f ms/photo, ratio %.3f (limit %.2f)' % (
        t_ref / len(pairs) * 1000, t_new / len(pairs) * 1000, ratio, max_ratio))
    for name, problems in failures[:10]:
        print('MISMATCH %s' % name, file=sys.stderr)
        for p in problems[:5]:
            print('    ' + p, file=sys.stderr)
    if failures or identical != expect or len(pairs) != expect:
        return 1
    if ratio > max_ratio:
        print('RATIO %.3f above %.2f' % (ratio, max_ratio), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
