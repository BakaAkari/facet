"""CPU-only cost of the S9 plane rebuilds (R7), per photo.

S9 stops retaining the BGR array and the gray plane; the face pass rebuilds the BGR array and the
saliency pass rebuilds the gray plane (BGR then gray) from the retained PIL image. R7 says the two
rebuilds together must stay under 0.25 s/photo or the orchestrator falls back to keeping the gray
plane. This script measures exactly that on a photo directory, with no GPU and no model:

  1. decode each photo with the repo's own ``load_image_from_path`` (the PIL image is what is retained),
  2. time ``pil_to_bgr`` (face pass rebuild) and ``bgr_to_gray(pil_to_bgr(...))`` (saliency rebuild)
     best-of-REPS per photo, and
  3. run the real ``_pass_insightface`` and ``_pass_saliency`` over the whole set with stub models that
     only consume the planes, and read ``metrics['rebuild_time']`` (the figure ``_print_summary`` logs).

usage: rebuild_cost.py [--photos DIR] [--reps N] [--out FILE]
env:   PERF_REPO (required) checkout whose code is measured
Exit 0 when the per-photo total (face + saliency rebuilds, median over photos) is <= 0.25 s, 1 when it
is above, 2 on a usage error, 3 when PERF_REPO is wrong. Prints one JSON object.
"""
import gc
import json
import os
import statistics
import sys
import time

REPO = os.environ.get('PERF_REPO', '')
if not REPO or not os.path.isfile(os.path.join(REPO, 'processing', 'multi_pass.py')):
    print('rebuild_cost: PERF_REPO=%r has no processing/multi_pass.py' % REPO, file=sys.stderr)
    sys.exit(3)
REPO = os.path.realpath(REPO)
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, REPO)

LIMIT_S = 0.25


def main(argv):
    photos = r'D:\facet-scratch\perf2\photos'
    reps = 3
    out = r'D:\facet-scratch\perf2\retained\rebuild_cost.json'
    i = 0
    while i < len(argv):
        if argv[i] == '--photos':
            photos = argv[i + 1]
        elif argv[i] == '--reps':
            reps = int(argv[i + 1])
        elif argv[i] == '--out':
            out = argv[i + 1]
        else:
            print(__doc__)
            return 2
        i += 2

    import processing.multi_pass as mp
    f = os.path.realpath(mp.__file__)
    if os.path.commonpath([f, REPO]) != REPO:
        print('rebuild_cost: processing.multi_pass imported from %s, not %s' % (f, REPO), file=sys.stderr)
        return 3
    from utils.image_loading import bgr_to_gray, load_image_from_path, pil_to_bgr

    names = sorted(n for n in os.listdir(photos) if n.lower().endswith(('.jpg', '.jpeg')))
    if not names:
        print('rebuild_cost: no JPEG in %s' % photos, file=sys.stderr)
        return 2

    rows, pils = [], {}
    for name in names:
        pil, cv = load_image_from_path(os.path.join(photos, name))
        del cv
        pils[name] = pil
        best_bgr = best_chain = float('inf')
        for _ in range(reps):
            t0 = time.perf_counter()
            bgr = pil_to_bgr(pil)
            best_bgr = min(best_bgr, time.perf_counter() - t0)
            t0 = time.perf_counter()
            bgr_to_gray(pil_to_bgr(pil))
            best_chain = min(best_chain, time.perf_counter() - t0)
            del bgr
        rows.append({'name': name, 'mp': pil.width * pil.height / 1e6,
                     'bgr_s': best_bgr, 'gray_chain_s': best_chain,
                     'both_s': best_bgr + best_chain})
        gc.collect()

    # In-situ: the real passes with stub models, one rebuild of each kind per photo.
    proc = mp.ChunkedMultiPassProcessor.__new__(mp.ChunkedMultiPassProcessor)
    proc.metrics = {'rebuild_time': 0.0}
    images = {name: {'pil': pil, 'width': pil.width, 'height': pil.height, 'cache': None}
              for name, pil in pils.items()}

    class _Faces:
        def analyze_faces(self, img_cv):
            return {'face_count': 0, 'face_quality': None, 'eye_sharpness': None, 'face_sharpness': None,
                    'is_blink': None, 'is_group_portrait': 0, 'face_details': [], 'face_area': 0}

    class _Saliency:
        def score_batch(self, pil_images, rebuilders):
            for rebuild in rebuilders:
                rebuild()
            return [{'subject_sharpness': 5.0, 'subject_prominence': 0.0, 'subject_placement': 5.0,
                     'bg_separation': 5.0, 'subject_bbox': None} for _ in pil_images]

    results = {n: {} for n in images}
    t0 = time.perf_counter()
    proc._pass_insightface(_Faces(), images, results)
    face_wall = time.perf_counter() - t0
    face_metric = proc.metrics['rebuild_time']
    t0 = time.perf_counter()
    proc._pass_saliency(_Saliency(), images, results)
    sal_wall = time.perf_counter() - t0
    total_metric = proc.metrics['rebuild_time']

    n = len(rows)
    both = [r['both_s'] for r in rows]
    res = {
        'repo': REPO, 'photos': n, 'reps': reps, 'mp_total': sum(r['mp'] for r in rows),
        'mp_mean': sum(r['mp'] for r in rows) / n,
        'face_rebuild_s_median': statistics.median(r['bgr_s'] for r in rows),
        'saliency_rebuild_s_median': statistics.median(r['gray_chain_s'] for r in rows),
        'both_s_median': statistics.median(both), 'both_s_mean': statistics.mean(both), 'both_s_max': max(both),
        'both_s_p90': sorted(both)[int(0.9 * (n - 1))],
        'in_situ_face_pass_rebuild_s_per_photo': face_metric / n,
        'in_situ_saliency_pass_rebuild_s_per_photo': (total_metric - face_metric) / n,
        'in_situ_total_rebuild_s_per_photo': total_metric / n,
        'in_situ_face_pass_wall_s_per_photo': face_wall / n,
        'in_situ_saliency_pass_wall_s_per_photo': sal_wall / n,
        'limit_s_per_photo': LIMIT_S,
        'within_limit': statistics.median(both) <= LIMIT_S and total_metric / n <= LIMIT_S,
        'rows': rows,
    }
    os.makedirs(os.path.dirname(out), exist_ok=True)
    text = json.dumps(res, indent=1)
    with open(out, 'w') as fh:
        fh.write(text)
    summary = {k: v for k, v in res.items() if k != 'rows'}
    print(json.dumps(summary, indent=1))
    return 0 if res['within_limit'] else 1


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
