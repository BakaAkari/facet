"""Capture real BiRefNet saliency masks for the scratch photo set (spec R11).

usage: capture_saliency_masks.py [--photos DIR] [--out DIR] [--limit N] [--expect N]
env:   PERF_REPO   (required) checkout whose SaliencyScorer produces the masks (BASE for the baseline set)

Loads BiRefNet through the repo's SaliencyScorer on the GPU, decodes every photo with the repo's
load_image_from_path and runs get_saliency_masks ONCE over the whole list (as production hands it a
chunk; the scorer sub-batches by free VRAM), then writes each binary mask as OUT/<stem>.png.

Exit codes: 0 only when `captured N/N` was printed with N == --expect (default 49) and every mask's
(h, w) equals the (h, w) of its decoded photo; 2 on any None mask, a count different from --expect, or a
size mismatch; 3 when PERF_REPO is not a checkout.
Defaults: --photos D:\\facet-scratch\\perf2\\photos  --out D:\\facet-scratch\\perf2\\masks
The GPU must be otherwise idle (check nvidia-smi first); nothing here runs another GPU job.
"""
import os
import sys

REPO = os.environ.get('PERF_REPO', '')
if not REPO or not os.path.isfile(os.path.join(REPO, 'processing', 'multi_pass.py')):
    print('capture_saliency_masks: PERF_REPO=%r has no processing/multi_pass.py' % REPO, file=sys.stderr)
    sys.exit(3)
REPO = os.path.realpath(REPO)
sys.path.insert(0, REPO)
os.chdir(REPO)

import json  # noqa: E402
import time  # noqa: E402

EXPECT_DEFAULT = 49


def main(argv):
    photos = r'D:\facet-scratch\perf2\photos'
    out = r'D:\facet-scratch\perf2\masks'
    limit = None
    expect = EXPECT_DEFAULT
    i = 0
    while i < len(argv):
        if argv[i] == '--photos':
            photos = argv[i + 1]
        elif argv[i] == '--out':
            out = argv[i + 1]
        elif argv[i] == '--limit':
            limit = int(argv[i + 1])
        elif argv[i] == '--expect':
            expect = int(argv[i + 1])
        else:
            print(__doc__)
            return 2
        i += 2
    names = sorted(n for n in os.listdir(photos) if n.lower().endswith(('.jpg', '.jpeg')))
    if limit is not None:
        names = names[:limit]
    os.makedirs(out, exist_ok=True)

    import numpy as np
    from PIL import Image
    import processing.multi_pass as mp
    f = os.path.realpath(mp.__file__)
    if os.path.commonpath([f, REPO]) != REPO:
        print('module %s is not under PERF_REPO %s' % (f, REPO), file=sys.stderr)
        return 3
    from models.saliency_scorer import SaliencyScorer
    from utils import load_image_from_path
    import torch
    if not torch.cuda.is_available():
        print('CUDA is not available: the masks must come from the GPU', file=sys.stderr)
        return 2

    cfg = json.load(open(os.path.join(REPO, 'config', 'scoring_config.default.json')))['models']['saliency']
    t0 = time.time()
    pils = []
    for n in names:
        pil, _cv = load_image_from_path(os.path.join(photos, n))
        if pil is None:
            print('could not decode %s' % n, file=sys.stderr)
            return 2
        pils.append(pil)
    print('decoded %d photos in %.1fs' % (len(pils), time.time() - t0), flush=True)

    scorer = SaliencyScorer(device='cuda', model_name=cfg['model'], resolution=cfg['resolution'],
                            mask_threshold=cfg['mask_threshold'], min_subject_pixels=cfg['min_subject_pixels'])
    scorer.load()
    t0 = time.time()
    masks = scorer.get_saliency_masks(pils)
    torch.cuda.synchronize()
    print('masks in %.1fs (module %s)' % (time.time() - t0, f), flush=True)

    bad = []
    written = 0
    for n, pil, m in zip(names, pils, masks):
        if m is None:
            bad.append('%s: mask is None' % n)
            continue
        h, w = pil.size[1], pil.size[0]
        if m.shape != (h, w):
            bad.append('%s: mask %s != image (%d, %d)' % (n, m.shape, h, w))
            continue
        if m.dtype != np.uint8 or set(np.unique(m).tolist()) - {0, 255}:
            bad.append('%s: mask is not binary 0/255 uint8' % n)
            continue
        Image.fromarray(m, 'L').save(os.path.join(out, os.path.splitext(n)[0] + '.png'))
        written += 1
    for b in bad:
        print('BAD ' + b, file=sys.stderr)
    print('captured %d/%d' % (written, expect))
    if bad or written != expect or len(names) != expect:
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
