# Spec: Faster multi-pass scans with unchanged results

Branch `perf/scan-passes`, worktree `D:\photo-llm-perf`, written against master release 1.20.0 (HEAD
`21a25ef6`) and to be REBASED onto master after `fix/open-gaps-120` merges, before any builder starts (see
"Build order and rebase"). Wherever this spec says `21a25ef6`, read `BASE_SHA`, the commit the perf branch
sits on after that rebase. Written by the spec writer; nothing here has been built. Every `file:line` below
was read against this worktree at `21a25ef6`. Function-level line numbers are cited so a builder can find the
site; they drift as soon as an earlier step edits the file and shift again after the rebase, so a builder
re-greps the symbol, not the number.

Source reports (read them, do not re-derive): `D:\photo-llm\.claude\.scratch\scan-pass-profile.md`,
`cpuio-profile.md`, `chunk-size.md`. Prototype code to port lives under the session scratchpad
(`$SP` = `C:\Users\laure\AppData\Local\Temp\claude\D--photo-llm\01873979-9070-498f-a0cb-74e72a51ba83\scratchpad`).
Prototype wall numbers were taken on the working tree of `D:\photo-llm` (branch `fix/open-gaps-120`, which
carries uncommitted edits to `facet.py`, `processing/scorer.py`, `processing/scan_state.py`,
`processing/routing.py` and others), NOT on `21a25ef6`. They are references for expectations only; every gate
below is measured against a baseline built from `BASE_SHA` in the same session.

## Build order and rebase (read first)

The orchestrator rebases `perf/scan-passes` onto master AFTER `fix/open-gaps-120` merges and BEFORE W0 starts.
Consequences, all binding:
- `BASE_SHA` = the rebase target (master after the merge). `D:\photo-llm-base` exists today, clean, detached at
  `21a25ef6` (checked: `rev-parse --short HEAD` prints `21a25ef6`). It must be re-pointed to `BASE_SHA`
  (orchestrator, state-changing: `git -C D:\photo-llm-base checkout --detach <BASE_SHA>`) before S0. S0's first
  command asserts `git -C D:\photo-llm-base rev-parse HEAD` equals the merge-base of `perf/scan-passes` and
  master and stops otherwise. A baseline taken at `21a25ef6` against a rebased patch compares different code
  (the open-gaps branch changes scoring paths), so it is invalid.
- If more open-gaps commits land after the rebase, the perf branch is re-rebased only BETWEEN waves, BASE is
  re-pointed, and B16a/B16b/B8a (about 25 minutes) are regenerated before the next parity run.
- Branch state when this was written: `fix/open-gaps-120` = `bfa45272` + `e38e64e0` on top of `21a25ef6`, 20
  files, none of them `processing/multi_pass.py`. It edits `processing/scorer.py` (aesthetic batch, IQA
  recompute, `fix_thumbnail_rotation`), `processing/batch_processor.py` (error `'stage'` labels
  `'load'`/`'score'`, still to come), `facet.py`, `db/schema.py`, `models/pyiqa_scorer.py`,
  `models/qrealign_native.py`, `utils/image_bounds.py`, `processing/routing.py`, `processing/scan_state.py`.

Overlap with this spec (checked with `git diff -U0 master...fix/open-gaps-120`):

| Step | Shared file | Branch hunks | Conflict risk and action |
|---|---|---|---|
| S7 (D) | `facet.py` | `_run_scan` hunks at base lines 1563-1572, 1795, 1831, 1915-1919; step 6 itself is unmoved but sits about 30 lines lower (`resolve_scan_tagger` at 2009, `run_tagging` at 2011 on the branch) | MOST CONFLICT-PRONE: same function, nearby context. D edits only the step-6 block (from `emit_progress('tagging'` to the "No new tags" log); re-grep, do not trust line numbers. Textual conflict not expected, semantic re-read required. |
| S0 comparator | `db/schema.py` | adds the `iqa_declined` column to `photos` (27 lines) | Comparator derives columns from the schema and compares new columns by default, so nothing to edit; the EXCLUDED-list test must still equal the documented list. Cell counts in handoffs change. |
| S0 / SC1 | `processing/scan_state.py` | `scan_failures` counting | Comparator's `scan_failures` table check (G6) is built against the rebased `SCAN_FAILURES_COLUMNS`. |
| S1 (A1) | `utils/image_bounds.py` (read only) | adds `LIQE_MIN_SHORT_SIDE`, `liqe_input_too_small`; `bounded_size` and `bound_to_inference_size` untouched | No edit, no textual conflict. `liqe_input_too_small` bounds first, so it gives the same answer for the full and the bounded image; `test_shared_bounded_image` asserts that equality. D4 line numbers move. |
| S1 (A1) | `models/pyiqa_scorer.py`, `models/qrealign_native.py` (read only) | `last_declined` threading in `PyIQAScorer` (hunks 137-528, NOT `_preprocess_image` 251-278), class attribute in `QReAlignNativeScorer` | No edit. A1's tensor test builds `PyIQAScorer` via `__new__`; it must set whatever attributes the rebased `_preprocess_image` reads. |
| S3 (C), S8a (D) | `processing/scorer.py`, `processing/batch_processor.py` (read only) | scorer.py: `fix_thumbnail_rotation` (~383), aesthetic batch, IQA recompute (~2250-2470); batch_processor: error stage labels | No edit by any step. `ImageCache(` site (base 1381) and `generate_photo_thumbnail` callers (base 2530, 2667; branch 2624, 2761) move; C and D run `test_batch_processor*.py` / `test_scorer.py` / `test_thumbnails.py` on the REBASED tree, which carries the branch's new stage-label tests. |
| S5 (G) | `processing/routing.py` (read only) | `single_pass_unsupported_reason`, hint text in `route_models` | Plans do not change (hint text only), but G re-derives the default 16gb/8gb plans after the rebase before trusting `PASS_WRITE_KEYS` test (c). |
| A1 / A2 | `tests/test_licensing_gate.py` | branch edits L1107-1108 (`test_default_mode_without_a_head_has_no_aesthetic`) | Different hunks from A's edits (L708, L726, L1261+); adjacent file, no overlap expected. |
| DOC (W4) | `CHANGELOG.md` | not touched yet; the branch will add entries under `[Unreleased]` | DOC appends after the branch's lines at W4 time. |
| R8 / S1 | stage convention | the branch gives `batch_processor` errors a `'stage'` key (`'load'`/`'score'`) | `multi_pass.py` already passes a free-text stage to `on_error` (`'load'`, `'save'`, model names); S1 records composition errors under stage `'composition'`. |

Builders run `git log --oneline -1` first and report it; a SHA that is not on top of `BASE_SHA` means they were
started on the wrong tree.

## Objective + success criteria

Make `facet.py` multi-pass scans (`ChunkedMultiPassProcessor`, `processing/multi_pass.py`) faster on the
CPU and orchestration side with RESULTS UNCHANGED. "Unchanged" is defined, not assumed:

> At a pinned chunk size (`PERF_CHUNK=16`), a patched scan of the 49-photo set writes SQLite cells that
> are identical to the baseline scan's (same storage class, same bytes, BLOBs included) in `photos`,
> `faces`, `photo_tags` and `scan_failures`, with ONE closed tolerance list: `raw_sharpness_variance`,
> `isolation_bonus`, `aggregate` at <= 1e-12 relative error. Timestamps and run ids are the only excluded
> columns.

Auto-tuned runs are NOT comparable with each other (batch composition changes scores by up to 5e-3, see
`perf/runs/parity_b16_1_vs_2.json` in `$SP`), so parity is only ever claimed at a pinned chunk.

Success criteria (each independently testable; thresholds are gates unless marked "expectation"):

- SC1 Parity, 16gb profile: `scripts/perf/compare.py <baseline_db> <patched_db>` exits 0, prints per-table
  cell counts (`photos`, `faces` and `scan_failures` rows as the case may be, `photo_tags`) and 0 differing
  cells outside the tolerance list; the tables that must be non-empty are non-empty (rules in V4; the tool
  exits 2 otherwise). The saliency sub-batch sizes and pyiqa `reduced_to` halvings logged per chunk must be
  the same sequence in both arms (G4); a different sequence fails SC1 even if every cell matches, because it
  means the run is not comparable. Run after every wave (W1a, W1b, W2, W3) and at the final gate.
- SC2 Parity, 8gb profile (`PERF_PROFILE=8gb`, same pin): same bar (including the sub-batch sequences), run
  after W1b, after W3 (serpentine is the step most likely to move it) and at the final gate. The 8gb plan
  exercises `clip` + `qrealign` + `saliency` in one pass and `insightface` in another.
- SC3 The comparator is trustworthy: baseline vs baseline (two runs of `BASE_SHA`) exits 0 (control), and the
  comparator's own tests prove it fails on a one-byte BLOB change, a 1e-15 change in a non-tolerance float, a
  1e-9 change in a tolerance column, a storage-class change (INTEGER vs REAL), an empty `photos`, a changed
  `scan_failures` row, and a `photo_tags` set that differs.
- SC4 Retained bytes (S9): `scripts/perf/retained_bytes.py` on `mp:20-21:4` and `first:32`: working-set
  retained bytes per pixel drops from the baseline's ~8.05 to <= 4.30 (the 4.05 target plus 0.13 for the
  shared 1024 px image plus slack); the per-bucket walk shows NO `cv` entry and NO `cache.gray` entry; the
  saliency pass never holds more than one sub-batch of masks (unit test with a live-object counter).
- SC5 Targeted CPU phases, measured as new/reference ratios in the same process on the same inputs (ratios
  survive a busy box, absolute milliseconds do not): saliency `_score_from_mask` <= 0.25 of the reference
  (expectation 0.17: 302 -> 52 ms); technical metrics (dynamic range + noise + contrast + `ImageCache`
  Laplacian) <= 0.25 of the reference (expectation 0.08); thumbnail <= 0.75 of the reference (expectation
  0.60: 42.5 -> 25.6 ms). Harness phase `agg` per chunk (median, chunks >= 1) <= 25% of the baseline's.
- SC6 Whole scan, 16gb, PERF_CHUNK=16, timing protocol (all of it is part of the criterion):
  1. one warm-up run per arm, discarded; the OS file cache state is stated in the report: "warm" (photos and
     model files were read by the discarded warm-up; cold-cache numbers are NOT measured, the standby list is
     not purged because that needs elevation);
  2. then alternating arms B P B P B P, at least 3 measured runs per arm, never two runs of one arm in a row;
  3. every run passes the nvidia-smi gate (<= 2560 MiB used, no other compute process on the GPU) and logs
     `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,memory.used` at its start
     (into the run dir) so a thermally throttled run can be recognised and rerun;
  4. report median, min and max per arm and the median ratio;
  5. verdict on MEDIANS: PASS iff median(P) <= 0.80 x median(B) AND the ranges do not overlap
     (max(P) < min(B)); median ratio <= 0.80 with overlapping ranges is reported INCONCLUSIVE, never PASS; a
     median ratio > 0.80 with disjoint ranges is FAIL.
  Expectation, not a gate: 290-310 s patched vs ~453 s baseline (harness wall 493 s); the prototypes measured
  385.5 s (S1-hoist-S2 + S4) and 327.6 s (S1-hoist-S2 + S5) against a 493.3 s pinned baseline.
- SC7 RAM cache (S6): under a ballast that makes the baseline evict (`Evicted N model(s)` in its log), the
  patched run has total cache hits >= baseline's, total evicted models <= baseline's, swap-file growth <=
  baseline's + 128 MiB, and is byte-identical to the baseline. If no ballast level in three tries makes the
  baseline evict, the criterion is reported INCONCLUSIVE (not passed) and the fake-memory unit tests stand
  alone as evidence.
- SC8 Tests: every new/changed test passes; the full suite shows no failure that is not in the baseline
  failure list generated in S0 on `BASE_SHA`; `ruff check . --select E,F,W --ignore E501` (CI) and
  `python -m ruff check .` (project config) exit 0; mypy baseline (`mypy-baseline.txt`, `api/` only) untouched.
- SC9 Docs: every behaviour or config change is in `docs/` (EN + de, es, fr, it, pt, zh where the file is
  translated) and `CHANGELOG.md` `[Unreleased]` in the same change set; pure speedups carry one CHANGELOG line
  with the measured gain.
- SC10 Hygiene: `D:\photo-llm\pretrained_models` manifest identical before and after every run; the scratch
  photo set manifest (name, size, mtime, sha256) identical before and after every run; no file written into
  `D:\photo-llm` by any builder; no AI attribution in any commit or PR text.

## Files to create / change

Create (all new):
- `scripts/perf/harness.py`, `scripts/perf/run.sh`, `scripts/perf/compare.py`, `scripts/perf/chunks.py`,
  `scripts/perf/retained_bytes.py`, `scripts/perf/capture_saliency_masks.py`,
  `scripts/perf/verify_saliency_exact.py`, `scripts/perf/verify_technical_exact.py`,
  `scripts/perf/assets.sh`, `scripts/perf/ballast.py` (S0, S2, S3)
- `tests/test_perf_compare.py` (S0), `tests/test_shared_bounded_image.py`, `tests/test_composition_hoist.py`,
  `tests/test_scan_retained_bytes.py` (S1, S9), `tests/test_saliency_exact.py` + helper
  `tests/_saliency_reference.py` (S2), `tests/test_technical_exact.py` (S3), `tests/test_scan_decode_overlap.py`
  (S4), `tests/test_scan_serpentine.py` (S5), `tests/test_ram_cache_policy.py` (S6a),
  `tests/test_ram_cache_trim.py` (S6b), `tests/test_scan_tagging_skip.py` (S7), `tests/test_thumbnail_exact.py`
  (S8a)
- `.claude/.scratch/scan-perf-<step>.md` handoff files (git-ignored)

Change:
- `processing/multi_pass.py` (S1, S9, S8b, S4, S6b, S5, S8c; four owners in sequence)
- `utils/image_loading.py` (S1/S9: `pil_to_bgr`, `bgr_to_gray`; `load_image_from_path` calls the former)
- `models/saliency_scorer.py` (S2, S9 streaming)
- `analyzers/technical.py`, `analyzers/image_cache.py` (S3)
- `utils/image_transforms.py` (S8a)
- `tag_existing.py`, `facet.py` step 6 only (S7)
- `models/model_manager.py` (S6a)
- `processing/resource_monitor.py` (S6b)
- Existing tests adapted (listed per step)
- Docs (S-DOC): `docs/CONFIGURATION.md`, `docs/DEPLOYMENT.md`, `docs/{de,es,fr,it,pt,zh}/CONFIGURATION.md`,
  `docs/{de,es,fr,it,pt,zh}/DEPLOYMENT.md`, `CHANGELOG.md`; `docs/COMMANDS.md` only if S7 needs a sentence.

## Files touched (builder -> files)

Builders in the same wave have disjoint file sets. A file owned by several builders across waves is edited by
them strictly in sequence. No builder runs state-changing git; the orchestrator commits by explicit path
after each verified wave (never `git add -A`: worktree junctions show up as untracked).

| Builder | Wave | Source files | Test files |
|---|---|---|---|
| H (harness) | W0 | `scripts/perf/*` | `tests/test_perf_compare.py` |
| A1 (S1) | W1a | `processing/multi_pass.py`, `utils/image_loading.py` | `test_multi_pass.py`, `test_multi_pass_pyiqa.py`, `test_multi_pass_pass_failure.py`, `test_review_fixes_scan.py`, `test_image_loading.py` (L221-238 drive `_load_images` with a MagicMock scorer; owned here, NOT by A2), `test_multi_pass.py` (L282 fixture, `_compute_aggregates` tests; the `_pass_*` tests are A2's), `test_licensing_gate.py` (aggregate sites only), new `test_shared_bounded_image.py`, `test_composition_hoist.py` |
| B (S2) | W1a | `models/saliency_scorer.py` | `test_saliency.py`, `test_social_crop.py` (read-only check), new `test_saliency_exact.py`, `_saliency_reference.py` |
| C (S3) | W1a | `analyzers/technical.py`, `analyzers/image_cache.py` | `test_analyzers.py`, `test_batch_processor.py`, `test_batch_processor_e2e.py`, `test_scorer.py` (read-only check), new `test_technical_exact.py` |
| D (S7, S8a) | W1a | `tag_existing.py`, `facet.py` (step 6 only), `utils/image_transforms.py` | `test_thumbnails.py`, `test_render_version.py`, `test_tagging.py`, `test_manual_tags_import.py`, `test_retry_failed_cleanup.py`, new `test_scan_tagging_skip.py`, `test_thumbnail_exact.py` |
| E (S6a) | W1a | `models/model_manager.py` | `test_model_manager.py`, new `test_ram_cache_policy.py` |
| A2 (S9, S8b) | W1b | `processing/multi_pass.py`, `utils/image_loading.py` | `test_multi_pass.py` (every hit of `cv` / `.gray` / `img_cv`: L282, the recording analyzer L501-502, L557-568 `test_the_passed_analyzer_is_the_one_that_runs` whose assertion changes, L700-711 whose `gray is not None` assertion changes; re-grep, the line list is a floor), `test_multi_pass_pass_failure.py`, `test_saliency_failure.py`, `test_face_models.py`, `test_licensing_gate.py` (L708, L726, L1261+), `test_review_fixes_scan.py`, `test_image_loading.py` (read-only check: A1 owns edits), `test_extended_iqa_wiring.py`, `test_mps_memory.py`, new `test_scan_retained_bytes.py` |
| F (S4, S6b) | W2 | `processing/multi_pass.py`, `processing/resource_monitor.py` | `test_resource_monitor.py` (its `_FakeChunkProcessor` has `model_manager = None` and no `_next_use_order`: must stay green unedited, G10), `test_scan.py`, `test_scan_interrupt.py`, new `test_scan_decode_overlap.py`, `test_ram_cache_trim.py` |
| G (S5, S8c) | W3 | `processing/multi_pass.py` | `test_multi_pass.py`, new `test_scan_serpentine.py` |
| DOC | W4 | docs files above, `CHANGELOG.md` | none |

W1a file-set disjointness check: A1 {multi_pass.py, image_loading.py}, B {saliency_scorer.py}, C {technical.py,
image_cache.py}, D {tag_existing.py, facet.py, image_transforms.py}, E {model_manager.py}. No overlap. A1's tests
and E's tests do not share a file (`test_licensing_gate.py` is A-only; E must keep it green without editing it,
see S6a). Test files are disjoint across the five as well: `test_image_loading.py` is A1's; `test_saliency*.py`
B's; `test_analyzers.py`, `test_batch_processor*.py` C's; `test_thumbnails.py`, `test_render_version.py`,
`test_tagging.py` D's;
`test_model_manager.py` E's. A test file needed by two builders in one wave is handed to one and the other
sends the orchestrator the change it needs.

## Not changing

- Chunk-size defaults and thresholds (`ram_chunk_size` 32, `auto_tuning` 10/128, `memory_limit_percent` 85,
  the monitor's 65% grow gate and 85% shrink gate). S6b changes only what the monitor EVICTS.
- Downsized derivatives of the decoded image for any model, thumbnail or face stage (rejected in
  `chunk-size.md` section 4: SigLIP cosine 0.9977-0.9998, SAMP-Net and saliency inputs differ, thumbnail bytes
  differ, face metric ratio 0.27x-49.8x). The only shared reduced image is the existing LANCZOS-1024 that the
  pyiqa models and Q-ReAlign already compute for themselves.
- Qwen `vlm_batch_size` (changes greedy decoding, hence tags), SigLIP/Qwen/open_clip preprocessing sharing
  (Tier B in `cpuio-profile.md`; Qwen patch grid changed on 1 of 8 photos), autocast, channels_last,
  `cudnn.benchmark`, prefetching a second chunk, chunk-size balancing across chunks (not bit-identical).
- `aesthetic_v25`, `deqa`, SigLIP, Qwen, saliency inputs: they keep the full-resolution `pil`.
- `analyzers/composition.py` (the histogram-median micro-gain of 40 ms is not in scope; the hoist moves the
  call, not the function), `analyzers/technical.py::get_histogram_data` and `get_color_harmony_data`.
- No new CLI flags, no kill-switch config keys, no compatibility shims for old call signatures. Internal
  signatures change and every caller is updated.
- `scoring_config.default.json` gains NO new key (see R6: the config version hash covers the whole resolved
  config, so a new shipped default would make every existing library look config-drifted).
- Database schema, API, Angular client.

## Domain rules

Status: V = verified (source named), A = assumed (how the owner verifies).

| # | Rule | Status |
|---|---|---|
| D1 | `_process_chunk` is chunk-major: one `_load_images`, then per group `_load_pass_models` -> `_run_model_pass` per model -> `unload_model(reclaim=False)` + one `_reclaim_freed_memory`, then `_compute_aggregates` + `_save_results` (`processing/multi_pass.py` 401-501) | V, read |
| D2 | A pinned chunk gives bit-identical scans: `fb16_A` == `fb16_B`, `fb16_c16` == io-proto == serp-proto, 0 differing columns, 49/49 CLIP/thumbnail/histogram bytes identical (`$SP\perf\runs\parity_fb16A_vs_B.json`, `parity_c16_vs_io.json`, `parity_c16_vs_serp.json`) | V, read |
| D3 | `cv == cvtColor(np.array(pil), RGB2BGR)`; `gray == cvtColor(cv, BGR2GRAY)`; SAMP-Net tensor from `pil` is `torch.equal` to the one from BGR; `_score_from_mask(mask, gray) == _score_from_mask(mask, cv)` (`chunk-size.md` section 4, 3 photos at 12.8/20.1/39.7 MP) | V, measured; A re-proves in unit tests on RGB images of several sizes |
| D4 | pyiqa models and Q-ReAlign feed on the SAME LANCZOS-1024 of the full-res image: `PyIQAScorer._preprocess_image` (`models/pyiqa_scorer.py` 251-278) resizes only when long edge > 1024 with `int(w*scale)`; `QReAlignNativeScorer._prepare` (`models/qrealign_native.py` 78-80) is `bound_to_inference_size(pil.convert('RGB'))` (`utils/image_bounds.py` 22-37); `verify_c1.py` tensors equal 12/12 | V, read + measured |
| D5 | `aesthetic_v25`, `deqa`, SigLIP, Qwen and saliency preprocess differently (Qwen max abs diff 0.1176) and must not receive the bounded image | V, `cpuio-profile.md` Tier B |
| D6 | RAM-cacheable on 16gb CUDA: clip, topiq, saliency, topiq_iaa, topiq_nr_face, liqe, samp_net. NEVER cached: qwen taggers (not in `CPU_CACHEABLE_MODELS`, `models/model_manager.py` 112), insightface on CUDA (L526-527 gate). `qrealign` is cacheable ONLY in commercial mode (L521-523) | V, read |
| D7 | `ModelManager.load_model_only` returns an already-resident model (`if model_name in self.models`, L727) and `unload_model` is the only thing that removes it from `self.models` (L359) | V, read |
| D8 | Result keys written by more than one pass: `aesthetic` (clip with an MLP head; topiq; qrealign when primary), `quality_score` and `scoring_model` (topiq/pyiqa primary; qrealign primary; qrealign_quality), `face_quality_iqa` (topiq_nr_face; `faces_permissive` only, via OFIQ `analyzers/face_permissive.py:444`; InsightFace's `analyze_faces` never returns it). Sentinel keys `SCAN_FAILED_KEY`, `FACE_FAILED_KEY`, `SALIENCY_FAILED_KEY` are truthiness-only | V, read (`multi_pass.py` 799-998, grep of `face_quality_iqa`) |
| D9 | The monitor reads SYSTEM memory (`effective_memory().percent`), so on this box sibling processes (9-24 GiB measured) move it; at > 85% it shrinks the chunk 25% per tick and calls `evict_cpu_cache()` unconditionally (`processing/resource_monitor.py` 78-87); the growth gate (chunk peak < 65%) is effectively dead on busy boxes | V, read + `chunk-size.md` section 6 |
| D10 | `config_version` stored on every row is an md5 of the WHOLE resolved config (`config/scoring_config.py:370-380`); more than one value in a DB raises a priority-75 "scored with older config versions" recommendation (`config/percentile_normalizer.py:1203-1233`) | V, read |
| D11 | `ImageCache.laplacian_variance` is `np.float64` today (`cv2.Laplacian(gray, CV_64F).var()`, `analyzers/image_cache.py:31`; `np.float64` subclasses `float`, so there is no Python-`float` type to "preserve" and sqlite3 binds both identically); `CV_16S` + `meanStdDev` equals it to <= 6.1e-15 relative, NOT bit-identical; `raw_sharpness_variance` is stored unrounded (`multi_pass.py` `_save_results`), `tech_sharpness`, `noise_sigma`, `dynamic_range_stops`, `contrast_score` are rounded | V, read + `cpuio-profile.md` A4 |
| D12 | `face_bbox` is always `None` where `_compute_aggregates` calls `get_placement_data` (the `pass` at `multi_pass.py` ~1039-1045), so placement and leading lines are pure functions of the pixels + config and hoisting them is work-neutral | V, read |
| D13 | `run_single_pass` sets `processor.restricted = True` (L1447) and `_process_chunk` then calls `_save_results_restricted`, never `_compute_aggregates` | V, read |
| D14 | Post-scan step 6 (`facet.py` 1973-1986) loads the profile's SigLIP/CLIP model through `resolve_scan_tagger` -> `build_clip_tagger` (`tag_existing.py` 94-137) before `run_tagging` checks `tagging.enabled` or counts pending rows; steps 7/8 load `clip` through `model_manager.load_model_only('clip')` themselves (`facet.py` ~405-440), so they do not depend on step 6's model | V, read |
| D15 | `Image.thumbnail` applies `reducing_gap=2.0` and Pillow's `preserve_aspect_ratio` rounding; `resize(..., reducing_gap=2.0)` with the same target gives byte-identical JPEGs (147/147, `cpuio-profile.md` A5) | V, measured |
| D16 | The 49-photo set contains only 7 faces and no RAW/HEIC/PNG/CMYK files (`parity_*.json`: `faces_total` 7; `inventory.json` lists JPEGs) | V, read; coverage risk R10 |
| D17 | `D:\photo-llm-base` exists, clean, detached at `21a25ef6` (`git -C /d/photo-llm-base rev-parse --short HEAD` prints `21a25ef6`; an earlier draft of this spec said it did not exist). It must be re-pointed to `BASE_SHA` after the rebase (see "Build order and rebase") | V, checked |
| D19 | Saliency sub-batch size is `min(DEFAULT_BATCH_SIZE, int(spare_gb / per_image_gb))` with `spare_gb` from `torch.cuda.mem_get_info` free bytes plus torch-cached bytes at the moment the pass starts (`models/saliency_scorer.py` `_affordable_vram_batch_size`), i.e. it depends on the device state the PREVIOUS pass left behind; pyiqa halves its forward batch on a CUDA OOM (`reduced_to`, `models/pyiqa_scorer.py` ~400-450). Batch composition changes stored values (up to 5e-3 on aesthetic) | V, read |
| D20 | `form_facet.warmup()` (the issue-#55 KMeans/OpenMP initialisation) is called inside `_load_images` (`processing/multi_pass.py` ~681) and its docstring (`analyzers/form_facet.py` ~233-240) requires the MAIN thread, before a worker pool; `_WARMED` makes later calls no-ops, so the one real initialisation happens in chunk 0 | V, read |
| D21 | `on_error(path, stage, error)` takes a free-text stage (`'load'`, `'save'`, a model name); `ScanRun.record_failure` stores `(run_id, path, stage, str(error)[:500])` (`processing/scan_state.py:51-56`), so the exception CLASS is not stored, only its message text | V, read |
| D22 | `_score_from_mask` uses THREE different thresholds today: prominence and the `min_subject_pixels` gate use `np.count_nonzero(mask)` (any non-zero value); subject/background regions use `mask > 128`; `bbox_from_mask` and the centroid use `mask > 128` and `bbox_from_mask` gates on the COUNT of `> 128` pixels. A non-binary uint8 mask with values only in 1..128 therefore passes the gate with an EMPTY subject region (variance 0, bg_separation 0.0 or 10.0, placement 5.0, bbox None) | V, read (`models/saliency_scorer.py` 25-46, 387-504) |
| D18 | Pre-existing box-local test failures exist (POSIX-path tests broken by `Path.resolve()` on Windows, 401s from the local viewer password). `$SP\build\m.fails` (47 lines, mtime 2026-10-15 00:44) lists 10 files and does NOT include `tests/test_single_pass_preserves.py`, which an earlier note says fails 5 tests here | V, read; the authoritative list is regenerated in S0 |

## Open questions / risks

- R1 Low-RAM paging and sibling noise. Evidence: `b16_1` (auto-tuned) showed the box paging when siblings hold
  9-24 GiB. Wall timings on this box are noisy; therefore gains are reported per targeted phase and the
  whole-scan gate is loose (SC6). S6 must not make paging worse: a bigger cache or a bigger retained set under
  memory pressure is the failure mode. Mitigation: S6b trims before each chunk decode and evicts in
  Belady order only down to a reserve; S9 shrinks the retained set; SC7 checks swap growth.
- R2 S4 and S5 are anti-additive. S4 hides the chunk decode behind the first group's model loading; S5 makes
  the first group of every other chunk already resident, so there is no load window left to hide the decode in.
  Decision: implement S4 to overlap only when the first group has at least one non-resident model; after S5
  (W3) G re-measures and reports whether S4 still pays in the final configuration (S4 may then be reduced to
  chunk 0 plus every other chunk, which is what the condition already does).
- R3 S4 transient memory and RAM-budgeted hosts. The overlap runs the decode worker pool while a model is being
  staged (weights read from disk or moved CPU->GPU), and S5 keeps the boundary group resident across the next
  chunk's decode, aggregates and save. On CUDA hosts the weights live in VRAM so this is cheap; on CPU and MPS
  the "resident" group lives in RAM outside `_can_cache_to_ram` / `_fits_cpu_cache_budget`
  (`models/model_manager.py` 483-536) and outside monitor eviction, and it would ignore `keep_in_ram: "never"`,
  which today guarantees the RAM is returned; 8 GB containers have OOM-killed scans before. Ruling (G2): S4's
  overlap and S5's boundary retention run ONLY when the device is CUDA, `keep_in_ram != 'never'` and an
  available-bytes budget allows (S4/S5 text); otherwise the code takes today's path. S6b's pre-chunk trim is
  not called for chunk 0, which is exactly the chunk that always overlaps, so the S4 budget gate carries chunk 0.
  SC7's swap check covers the rest.
- R4 Multi-writer result keys (D8) make pass order observable. S5 is safe only when every key written by more
  than one model is written by models of the SAME group (intra-group order is preserved). Commercial mode puts
  `faces_permissive` (writes `face_quality_iqa`) after `topiq_nr_face`'s group and, when Q-ReAlign is primary,
  `qrealign` beside `clip` on `aesthetic`. S5 therefore computes safety from the actual plan and falls back to
  forward order (with an INFO log naming the key) when unsafe.
  Write-set independence is NOT input independence (G4, D19): the saliency and pyiqa sub-batch sizes come from
  the VRAM state the previous group left behind, and serpentine changes what precedes saliency (Qwen/torch on
  forward chunks, insightface/ONNX Runtime on reversed ones). S5 therefore ships only under the hard condition
  in S5; the residual risk (batch sizing that depends on the host's VRAM) stays recorded here: a W3 parity pass
  on this box does not prove the property on a smaller or busier GPU.
- R5 Float tolerance. S3 changes `ImageCache.laplacian_variance` and `get_contrast_score`'s std at the 1e-15
  level; the three unrounded stored columns inherit it (`raw_sharpness_variance`, `isolation_bonus` via
  `face_sharpness / (variance + 1)`, `aggregate`). Everything rounded matched 147/147. The tolerance list is
  closed; if the first parity run shows any other differing column, the owner of the responsible step must
  either remove the cause or add the column with evidence of its unrounded derivation. `category` (discrete)
  must stay identical; a flipped category is a failure, not tolerance. Accepted residual risk: "rounded outputs
  identical 147/147" (S3) and the <= 1e-15 variance differences (S2, S3) can still flip a `round(x, 2)` whose
  unrounded value sits exactly on a .xx5 boundary. The chance per value is negligible and nonzero over a
  127k-photo library; this is stated, not claimed away, and the 49-photo parity run cannot rule it out.
- R6 No new shipped default key. A key added to `scoring_config.default.json` changes `config_version` for all
  newly scanned rows (D10) and triggers the false "config drift, run --recompute-average" recommendation on
  every existing library. `models.ram_cache_reserve_gb` is therefore read with a code default (4.0) and
  documented as optional, absent from the shipped file. Alternative considered: ship the key in the defaults
  (matches how `keep_in_ram` is carried) and exclude `config_version` from parity; rejected for the user-facing
  drift side effect.
- R7 S9 rebuild cost. Dropping `cv` and `cache.gray` costs ~0.11 s/photo (BGR rebuild in the face pass) and
  ~0.10 s/photo (gray rebuild in the saliency pass) at 20 MP, serial on the main thread, against 0.40 s/photo of
  composition moved off it and 0.25 s saved by S2. Fallback "T1": keep `cache.gray` (retained 5.05 B/px) and
  rebuild only BGR in the face pass. Trigger: A2's `rebuild_time` metric plus the harness shows the two rebuilds
  cost more than 0.25 s/photo in total, or SC6 fails with S9 the only change in the wave. The orchestrator
  decides; the SC4 bound becomes 5.30 under T1.
- R8 Failure-path change from the composition hoist (S1). Today `_process_chunk` (401-501) has no try around
  `_compute_aggregates`, so a failure inside `get_placement_data` or `detect_leading_lines` aborts the whole
  scan, not just the chunk. After the hoist it is a per-photo failure recorded as
  `on_error(path, 'composition', exc)` (stage `'composition'`, NOT `'load'`: the open-gaps branch gives
  `batch_processor` errors a `'stage'` key, `'load'`/`'score'`, and a mislabelled stage would send a
  composition bug to the wrong triage bucket in `scan_failures`); the photo is skipped and the chunk continues.
  That is a behaviour change on an error path; it goes in CHANGELOG `### Changed` ("a composition error now
  fails one photo, recorded under stage `composition`, instead of aborting the scan") and has a test. The hoist
  is gated on `not self.restricted` (D13) so `--pass ...` scans do no extra work. A MagicMock config with
  real-content images would now raise inside the worker (tiny synthetic test images return from
  `get_placement_data` before reading the config, so the existing tests survive); S1's touched suites
  therefore assert `len(images) == len(paths)`, so a silent per-photo load failure cannot hide.
- R9 PIL must never be mutated after load. S9 rebuilds `cv`/`gray` from the retained `pil`, and S8a removes the
  thumbnail's `pil.copy()`. A pass that mutated `pil` in place would silently corrupt every later rebuild.
  Grep found no in-place `thumbnail/draft/paste/putpixel` on the retained image
  (`utils/image_transforms.py:65` and `:150` act on copies). Guards: a unit test hashing `pil.tobytes()` around
  every `_pass_*` with stub scorers, the harness `PERF_PIL_GUARD=1` mode (crc32 of each photo's pil before and
  after every `_run_model_pass`, one dedicated untimed run), and the byte-identical parity run.
- R10 Parity coverage is only as wide as the input set: 49 JPEGs, 7 faces, a few panoramas, no RAW/HEIC, no
  grayscale or CMYK source. Mitigation: synthetic edge cases in unit tests (tiny, 1xN, constant, saturated,
  extreme aspect, mode L/RGBA sources through `open_nonraw_image`), and the face-pass rebuild test on RGB
  arrays. Residual: RAW and HEIF paths are exercised only by existing tests, UNVERIFIED end to end, unless the
  PIL-guard run adds one HEIF/HIF and one RAW: S0 searches `D:\photo-llm\photos` read-only for one of each,
  copies them (never moves or deletes an original) into the scratch photo set as a separate 2-file micro-run
  compared B vs P at chunk 2, and states "RAW/HEIF not covered end to end" if none exists. Statically every
  non-RAW route ends in an RGB PIL (the HDR PQ path builds `Image.fromarray(out, 'RGB')`,
  `utils/image_loading.py` ~635; `_decode_raw` and the previews convert to RGB), so S9's `pil_to_bgr` rebuild
  sees RGB.
- R11 Real BiRefNet masks are required before S2 merges (the profile's masks were synthetic blobs). Dependency:
  S0 captures them on the GPU; `scripts/perf/verify_saliency_exact.py` must exit 0 with `49/49` before B's
  step is accepted. GPU scans and the capture are exclusive: nothing else runs on the GPU meanwhile.
- R12 `D:\photo-llm-base` exists at `21a25ef6` (D17) but must sit on `BASE_SHA`, the rebase target. Precondition
  for S0, owned by the orchestrator (state-changing git the spec writer does not run):
  `git -C D:\photo-llm-base checkout --detach <BASE_SHA>`. S0 asserts it and stops otherwise.
- R13 Box-local test failures (D18). The gate compares against a failure list regenerated in S0 on the baseline
  worktree, with `$SP\build\m.fails` as a cross-check; differences between the two are reported, not hidden.
- R14 Single-pass paths share the code S3 changes: `processing/batch_processor.py:310` and
  `processing/scorer.py:1381` build `ImageCache` too, and `facet.py` `_flush_saliency_batch` (L251, `score_batch` call at L258) calls
  `SaliencyScorer.score_batch(batch_pil, batch_cv)` with BGR ndarrays. S2/S3 stay drop-in for them and C runs
  `test_batch_processor*.py` and `test_scorer.py`; B keeps ndarray inputs working.
- R15 Worktree assets. Worktrees lack untracked runtime assets: `pretrained_models/` (D:\photo-llm has 11
  files, 228 MB, resolved against the repo root by `models/weights.py:25` and relative to cwd elsewhere),
  `exiftool.exe` + `exiftool_files/` (the `exiftool` binary is started by name, `exiftool/exiftool_batch.py:41`,
  and found via PATH or the current directory), and `scoring_config.json` (absent in both arms, so both use
  the shipped defaults symmetrically). A scan would otherwise download models or fall back to `exifread`
  (different EXIF, but symmetric in both arms - still a different run than production).
- R16 Windows junction deletion. `rm -r` or `git worktree remove` through a junction can recurse into the
  target. Teardown is `rmdir <junction>` (removes the link only) and `del <hardlink>`; never `rm -r`, never
  `git clean` in a worktree.
- R17 `ModelManager.evict_cpu_cache` runs on the monitor thread while `_restore_from_cache` pops on the scan
  thread (existing race handled by `pop(name, None)`). S6a's partial eviction must keep that discipline and
  S6b's pre-chunk trim runs on the scan thread, so both can run concurrently with the monitor.
- R18 Two parity-visible nondeterminisms exist outside this change: the saliency sub-batch size comes from free
  VRAM (`_affordable_batch_size`, D19), and Qwen's sub-batch of 4 interacts with chunk alignment. The nvidia-smi
  gate, the baseline-vs-baseline control (SC3) and the per-arm sub-batch logging (V3, SC1) are what separate
  them from a real regression.
- R19 Warm-up thread (G1, D20). S4 moves `_load_images` onto a worker thread, so chunk 0's real
  `form_facet.warmup()` would run on that worker while the main thread initialises torch/transformers in
  `_load_pass_models`, the concurrent native-pool initialisation class issue #55 deadlocked on (UNVERIFIED that
  it would hang; reproducing it is GPU- and timing-dependent, so the rule is prevention, with a test).
- R20 Cancellation (G8). A decode future that outlives the scan blocks Ctrl+C until the whole chunk decodes if
  the executor is shut down with `wait=True`. S4 shuts down with `wait=False, cancel_futures=True` and a stop
  flag checked per photo.

## Central verification protocol

Everything here is built once in S0 and reused by every later step. Principle: a step's own `verify:` proves
the step; the central protocol proves nothing else moved.

V1 Two clean trees, one scratch root.
- BASE = `D:\photo-llm-base` (detached at `BASE_SHA`, re-pointed by the orchestrator, R12). PERF =
  `D:\photo-llm-perf`.
- Scratch root `$SP\perf2\` holds all DBs, logs, masks and manifests. Nothing is written to `D:\photo-llm`.
- Photo set: `$SP\vram12\photos` (49 JPEGs; verified 49 files). Manifest `{name,size,mtime,sha256}` taken
  once, re-checked after every run (SC10).
- Interpreter: `D:\photo-llm\conda\python.exe` (the project's `venv/bin/python` does not exist on this box).
  Every harness and test command first proves it imports the intended tree: `python -c "import
  processing.multi_pass as m; print(m.__file__)"` from the worktree cwd must print a path under that worktree
  (an editable install pointing at `D:\photo-llm` would otherwise make a "patched" run silently use the wrong code).

V2 Assets without copying and without writing into `D:\photo-llm` (`scripts/perf/assets.sh`).
- `prepare <worktree>`: directory junctions `pretrained_models` and `exiftool_files` (via `cmd //c mklink //J`),
  hard link `exiftool.exe`, all pointing at `D:\photo-llm`. `HF_HUB_OFFLINE=1` in every run so nothing
  downloads; `HF_HOME` stays `D:\llm_cache` (user env).
- `manifest`: `{name,size,sha256}` of `D:\photo-llm\pretrained_models\*` and of `exiftool.exe`, stored in
  `$SP\perf2\assets.manifest.json`; `check` recomputes and exits 1 on any difference (a model that wrote a
  file through the junction shows up here).
- `teardown <worktree>`: `rmdir` the two junctions, `del` the hard link, then verify `check` still passes.
- The junctions/hard link appear as untracked files in `git status`; the orchestrator commits by explicit path.

V3 Harness (`scripts/perf/harness.py`, port of `$SP\perf\harness.py`, fixes below).
- Env: `PERF_REPO` (required; `sys.path.insert(0, PERF_REPO)`, `os.chdir(PERF_REPO)`, `runpy` of
  `<PERF_REPO>\facet.py`; the prototype hard-codes `D:\photo-llm` at its lines 17 and 277), `PERF_PROFILE`
  (`16gb` default | `8gb`), `PERF_CHUNK` (pin; port of prototype lines 144-155: forces `chunk_size` and
  neutralises `reduce_chunk_size`/`increase_chunk_size`; RAM-cache eviction deliberately stays live),
  `PERF_PIL_GUARD`.
- Hard guards: abort with exit 3 unless `processing.multi_pass.__file__` is under `PERF_REPO`; abort unless
  the nvidia-smi gate passed (done in `run.sh`).
- Outputs per run dir: `harness.log`, `events.json` (main-thread timed calls), `accum.json` (any-thread timers),
  `samples.csv` (GPU util/mem, CPU, rss/private, `sys_mem_pct`, phase, plus NEW `swap_used_gib`),
  `summary.json` (adds the processor `metrics` dict returned by `process_directory`: `io_time`,
  `model_load_time`, `inference_time`, `model_unload_time`, and later `io_decode_time`, `rebuild_time`),
  `batch_sizes.json`, `gpu_start.txt`.
- Batch-size ledger (G4, D19): the harness wraps `SaliencyScorer._affordable_batch_size` (record each return
  value, keyed by chunk index and pass-group name) and records every pyiqa `reduced_to` halving (wrap the
  logger call or parse `harness.log`); `batch_sizes.json` holds, per chunk, the ordered list of
  `(model, sub_batch_size, halved_to)`. No source edit is needed for this; `compare.py` takes the two run
  directories' `batch_sizes.json` and fails when the sequences differ (SC1/SC2).
- `gpu_start.txt`: written by `run.sh` at the start of every run:
  `nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,pstate,memory.used --format=csv` plus
  `nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv` (SC6 step 3).
- Fixes the prototype needs: (a) the `evict_cpu_cache` wrapper must be signature-agnostic (`*a, **k`) because
  S6a adds parameters; (b) `timed_phase` must only push the phase stack on the main thread, because S4 runs
  `_load_images` on a worker thread (other threads add to `accum` under a `thread:` name); (c) phase
  accounting must not depend on monkeypatching internals S4/S5 restructure, so wall per phase also comes from
  the `metrics` dict.
- `scripts/perf/run.sh LABEL REPO PHOTO_DIR [facet args]`: port of `$SP\perf\run.sh`: nvidia-smi gate at 2560 MiB
  (aborts with 99), `other_python.txt` snapshot, `HF_HUB_OFFLINE=1`, `timeout 1800`, `--db <rundir>\db.sqlite`,
  records start/end/exit.
- `scripts/perf/chunks.py`: port of `$SP\perf\chunks.py`/`analyze.py`/`compare.py`: per-chunk table (wall, io,
  load, infer per model, agg, save, unload, reclaim), per-chunk `load:*` hit/miss, evicted model names, max
  `sys_mem_pct`, max swap, and a two-run side-by-side. Port `$SP\perf\memtrace.py` and `others_mem.py` only as
  far as the swap and sibling-memory snapshot columns need.

V4 Comparator (`scripts/perf/compare.py A.db B.db [--profile-name]`), replacing `$SP\perf\parity.py`, whose
SKIP set hides `caption_embedding`, never compares `photo_tags`, compares faces only by count/cosine/bbox, has no
tolerance dict and returns no exit code.
- Opens both DBs read-only with a PLAIN path (`sqlite3.connect(path)` then `PRAGMA query_only = ON`), not a
  `file:` URI: a reviewer's probe on this box got "unable to open database file" from `file:...?mode=ro` (and
  `immutable=1`, and `Path.as_uri()`) on the prototype WAL databases while a plain-path open of the copied
  DB+wal+shm worked (root cause not established). The comparator self-test therefore opens a REAL run database
  (copied with the SQLite backup API into the scratch dir), not only synthetic `init_database` ones.
- Tables and keys: `photos` by `path`; `faces` by (`photo_path`, `face_index`); `photo_tags` as a SET of all
  its columns (it has no stable key); `scan_failures` as a SET of (`path`, `stage`, error text). The error text
  is the stored `str(error)[:500]` (D21: the exception class is not stored, only its message, which usually
  names it), with `0x...` addresses normalised. Every column of `photos` and `faces` is compared as a
  `(sqlite typeof, value)` pair, BLOBs by sha256. A column is skipped only if it is in the explicit EXCLUDED
  dict, each entry with a one-line reason: `photos.scanned_at` (timestamp); the run-id and timestamp columns of
  `scan_failures`; faces/photo_tags timestamp or autoincrement id columns if the schema has them (H derives them
  by reading `db/schema.py` `FACES_COLUMNS`/`PHOTO_TAGS_COLUMNS`/`SCAN_FAILURES_COLUMNS`). A unit test asserts
  the EXCLUDED set equals the documented list, so a new column is compared by default.
- TOLERANCE dict: `raw_sharpness_variance`, `isolation_bonus`, `aggregate` at 1e-12 relative (absolute
  floor 1e-12 when the baseline value is 0). Nothing else.
- Emptiness rules (per table, not global, so SC1 can pass and still cannot be vacuous):
  - `photos` MUST be non-empty in both DBs, with identical key sets: otherwise exit 2.
  - `faces`: row counts must be equal; if the baseline has faces they must also be present in the patched
    DB (a patched DB that lost them differs and exits 1). For the 49-photo pin the baseline must hold > 0 faces
    (7 are expected); S0 records the count, and a baseline with 0 faces makes the run invalid (exit 2), because
    the face path would be untested.
  - `photo_tags`: compared as a set; empty is allowed ONLY if it is empty in the baseline too, and the output
    says so in words ("photo_tags: 0 rows in both, expected-empty for a scan: a scan never writes it, only
    `--migrate-tags` and `tag_existing` do; per-photo `tags` is covered as a `photos` column"). If the baseline
    has rows the patched set must equal it.
  - `scan_failures`: compared as a set; empty in both is the normal case and is printed as such; a row present
    in one arm only is a difference (exit 1) and names the stage.
- Exit 0 only if every non-empty-required table is non-empty, `cells_compared > 0`, nothing differs and the
  batch-size ledgers match; exit 1 with the first 20 differing cells (table, column, path, both values)
  otherwise; exit 2 for a missing table, a required-non-empty table that is empty, or a row-set mismatch on
  `photos`/`faces` keys. Prints per-table cell and row counts.
- Value comparison for floats is by bit pattern (`struct.pack('<d')`) outside the tolerance dict, never by `==`
  alone, so `-0.0` vs `0.0` and NaN payloads cannot hide.
- Also prints the sets of tags per photo and counts of faces for the report.

V5 Run matrix and cadence (GPU scans are exclusive: one at a time, never while another GPU job runs).
- S0 (once, on BASE): B16a, B16b (16gb, `PERF_CHUNK=16`), B8a (8gb, same pin); `compare B16a B16b` is the control
  (SC3). Cost: ~8 + 8 + 5 min.
- After each wave W1a, W1b, W2, W3 (on PERF): one 16gb run `P16-<wave>`; `compare B16a P16-<wave>` must exit 0
  before the wave is committed. One 8gb run vs `B8a` after W1b, AFTER W3 (`P8-W3`, required: serpentine is the
  step that can change VRAM-dependent batch sizing on the 8gb plan, where `clip`, `qrealign` and `saliency`
  share one group) and at the final gate.
- Final gate: the SC6 protocol: a discarded warm-up per arm, then alternating B P B P B P (at least 3 measured
  runs per arm), the PIL-guard run, the ballast runs for SC7, the 8gb pair. Median with min/max; verdict on
  medians with overlapping ranges reported INCONCLUSIVE. Cost: about 8 runs x 8 minutes of GPU time.
- A parity failure at a wave is bisected by the wave's builder list (each builder's step is independently
  revertable), never by weakening the comparator.

V6 Unit-test protocol (no GPU; pytest).
- Per step: its new tests + the existing tests it touches + the test files in the same area, run with
  `D:\photo-llm\conda\python.exe -m pytest <files> -q -p no:cacheprovider` from the worktree. Builders in the
  same wave edit concurrently, so a builder runs only its own files; the FULL suite runs once per local commit
  (orchestrator) on a tree no builder is editing, written to a file, exit code read from the file or
  `${PIPESTATUS[0]}`, then diffed against the S0 baseline failure list. Each test file states its count.
- Test DBs are built with `db.schema.init_database`. No `mock.patch` on FastAPI auth dependencies. Monkeypatching
  a module attribute such as `processing.resource_monitor.effective_memory` is fine (not an auth dependency).

V7 Fault-injection hygiene (every `verify:` below names its fault). Before injecting, record
`git diff -- <file> | sha256sum` (the file carries uncommitted work, so `git status --porcelain` cannot be the
check). Inject by hand, run the check, see it FAIL, restore by retyping the original by hand (never
`git checkout`), then require the recorded hash to match again. The orchestrator's refuter, working on committed
code, uses `git status --porcelain <file>` empty.

## Steps

Each step lists: description, key decision, default, verify (including the fault that must make it fail),
affected files, owner. A builder writes `.claude/.scratch/scan-perf-<step>.md` when done: files changed, test
counts and exit codes, measured numbers, anything left undone.

### S0. Baseline tree, assets, harness, comparator, real masks, baselines (owner H, W0)

Precondition (orchestrator): the perf branch is rebased onto master after `fix/open-gaps-120` merged, and
`D:\photo-llm-base` is re-pointed (`git -C /d/photo-llm-base checkout --detach <BASE_SHA>`). H's first command:
`git -C /d/photo-llm-base rev-parse HEAD` must equal `git -C /d/photo-llm-perf merge-base HEAD master`; H stops
and reports if the tree is absent, dirty or not at that commit.

Description: create everything in the Central verification protocol (V1-V4), then produce the shared inputs and
baselines the other steps consume.
1. `scripts/perf/assets.sh` + manifests; prepare BASE and PERF; `check` passes.
2. `scripts/perf/harness.py`, `run.sh`, `chunks.py`, `ballast.py` (allocates and touches N GiB in a child process
   and holds it until a sentinel file appears; used only by SC7).
3. `scripts/perf/compare.py` + `tests/test_perf_compare.py`.
4. `scripts/perf/retained_bytes.py`: port of `D:\facet-scratch\chunk\measure_load.py` (builds
   `ChunkedMultiPassProcessor.__new__`, sets `load_workers`, `on_error`, `restricted=False`, calls `_load_images`,
   per-bucket `nbytes` walk, psutil working-set and private deltas, optional tracemalloc). Differences from the
   port source: `PERF_REPO` instead of the hard-coded path; the stub scorer wraps a real `ScoringConfig` built as
   `ScoringConfig(config_path=<repo>/config/scoring_config.default.json, validate=False)` because the S1 hoist
   calls `config.get_composition_weights()`; `inventory.json` is generated by the script from the photo dir
   (header-only sizes) instead of read from `D:\facet-scratch`; `--saliency-live-masks` mode that feeds a stub
   saliency model and reports the max number of simultaneously live masks (used by S9). Output JSON key names
   kept (`bytes_per_px_retained_wset`, `buckets_bytes_per_px`).
5. `scripts/perf/capture_saliency_masks.py`: loads BiRefNet through the repo's `SaliencyScorer` on the GPU, runs
   `get_saliency_masks` over the 49 photos exactly as production batches them (one call over the whole list),
   writes each binary mask as a PNG into `$SP\perf2\masks\<stem>.png`. Exit 0 only if it wrote 49 masks and
   each mask's (h, w) equals the (h, w) of `load_image_from_path(photo)[0]`; exit 2 on any `None`, count != 49
   or size mismatch.
6. Run on BASE: capture masks (GPU, exclusive), then B16a, B16b, B8a. Run `compare B16a B16b` (control).
   Record the per-arm `batch_sizes.json` ledgers and the face count of the baseline (> 0 expected: 7).
   Optionally (R10) search `D:\photo-llm\photos` read-only for one HEIF/HIF and one RAW, copy them into the
   scratch set, and run a 2-file baseline micro-run at chunk 2 for the later PIL-guard/parity micro-run.
7. Regenerate the baseline failure list: full pytest on BASE with the conda interpreter, failing node ids to
   `.claude/.scratch/scan-perf-baseline.fails`; diff against `$SP\build\m.fails` and report both directions.
8. Handoff `.claude/.scratch/scan-perf-S0.md`: exact commands, paths, B16a/B16b/B8a walls and per-chunk
   tables, `retained_bytes` baseline numbers for `mp:20-21:4` and `first:32`, the sibling-process snapshot.

Key decision: build the comparator and harness BEFORE any source step, from clean trees, so no step is
verified by a tool that was written to pass it.
Default: all artefacts under `$SP\perf2\`; scripts committed under `scripts/perf/`.
Verify (each must be able to fail):
- `assets.sh check` exits 0; then run it against a manifest copy with one sha256 altered: must exit 1.
- `harness.py` with `PERF_REPO` pointing at a directory without `processing/multi_pass.py`: exit 3 with a
  message; with the right repo `harness.log` records the module path under that repo.
- `tests/test_perf_compare.py` (CPU, DBs from `init_database` in `tmp_path`, N tests): identical DBs -> 0 diff;
  flip one thumbnail byte -> exit 1 naming `thumbnail`; change a non-tolerance REAL by 1e-15 -> exit 1; change
  `aggregate` by 1e-15 relative -> exit 0, by 1e-9 -> exit 1; write an INTEGER where the other has the same
  numeric value as REAL -> exit 1; empty `photos` -> exit 2; `faces` empty where the baseline has faces -> exit
  1 (or 2 for a zero-face baseline on the pinned set); `photo_tags` empty in both -> exit 0 with the
  "expected-empty" wording in the output, `photo_tags` rows in one arm only -> exit 1; a `scan_failures` row
  (path, stage, error) present in one arm -> exit 1, changed stage -> exit 1; differing `batch_sizes.json`
  sequences -> exit 1; the EXCLUDED set equals the documented list. Plus a self-test on a REAL run database
  (backup-API copy of a prototype or S0 run DB): identical copy -> exit 0, one flipped `thumbnail` byte ->
  exit 1 (NOTE: plain-path open, never a `file:` URI). Fault: delete `typeof` from the cell key and the
  storage-class test must fail; make the zero-rows rule global instead of per table and the `photo_tags`
  expected-empty test must fail.
- `retained_bytes.py` on BASE `mp:20-21:4` reports `bytes_per_px_retained_wset` within 7.5-8.6 (reference
  8.05) and bucket keys include `cv` and `cache.gray`. A value outside that range means the port is broken.
- `capture_saliency_masks.py --limit 3` exits 2 (expected 49); the full run prints `captured 49/49`.
- `compare B16a B16b` exits 0 with `cells_compared` printed; the same comparator on `B16a` vs a copy whose single
  `thumbnail` byte was changed (copy made with the SQLite backup API into `$SP\perf2\`, never touching a prod
  DB) exits 1.
Affected files: `scripts/perf/*`, `tests/test_perf_compare.py`, `.claude/.scratch/scan-perf-S0.md`,
`.claude/.scratch/scan-perf-baseline.fails`.

### S1. One shared LANCZOS-1024 image per photo + composition hoisted into the load workers (owner A1, W1a)

Description (all in `processing/multi_pass.py` `_load_images`/`_load_one` 571-672, plus consumers):
1. In `_load_one`, after decode, compute `bounded = bound_to_inference_size(pil_img)` (`utils/image_bounds.py`;
   `pil_img` is already RGB, so no `.convert('RGB')` copy) and store it as `'bounded'`. Wrap it in try/except
   that logs at DEBUG and stores `None`: a degenerate aspect (e.g. 20000x1: `int(1*scale)` = 0, `resize`
   raises) must leave each consumer to fail or succeed exactly as it does today on the full image. For a photo
   whose long edge is <= 1024 `bound_to_inference_size` returns the SAME object, so nothing extra is retained.
   Compute it only when the plan needs it (`_needs_bounded_image()`: any model in `self.pass_groups` that is a
   pyiqa model other than `aesthetic_v25`/`deqa`, or `qrealign`/`qrealign_quality`; `pass_groups` None -> True),
   so `--pass saliency` does not pay 145 ms/photo for nothing.
2. `_pass_pyiqa` (949): `pil_imgs = [img['pil'] if model_name in ('aesthetic_v25', 'deqa') else
   (img.get('bounded') if img.get('bounded') is not None else img['pil']) ...]`. `_pass_qrealign` (910) the same
   with the bounded image. Name the exclusion set once as a class constant next to `PYIQA_MODELS` (717).
3. Hoist composition: after the technical metrics and `cache.hsv = None` (643), when `not self.restricted`
   compute `get_placement_data(None, img_w, img_h, self.scorer.config, img_cv)` and
   `detect_leading_lines(img_cv, cache=cache)` and return
   `'composition': {'score', 'power_point_score', 'leading_lines_score'}` (the same `.get(..., 5.0)` /
   `.get(..., 0)` defaults `_compute_aggregates` uses today). The call is wrapped so that an exception is
   reported as `self.on_error(path, 'composition', exc)` and the photo is dropped from the chunk (R8); it is NOT
   labelled `'load'`.
4. `_compute_aggregates` (998-1126): replace the two in-place computations (1034-1063) by reads of
   `img_data['composition']`, keeping the branch structure exactly: when `comp_score` is absent/None take all
   three values from the hoisted dict; when SAMP-Net supplied `comp_score`, fill `power_point_score` and
   `leading_lines_score` only if absent. The read of `img_data['composition']` happens ONLY inside the branches
   that need it (so fixtures that preset all three values, e.g. `test_multi_pass.py:282`, keep working with an
   extra `'cv'` key). Remove the `img_cv = img_data['cv']` line (L1017) explicitly: it is the last reader of
   `'cv'` in `_compute_aggregates`, and S9 deletes the key from `_load_one`'s return value. Drop the dead
   `face_bbox` scaffolding. Correct the `_load_images` docstring (it already claims composition runs in load;
   after this step it does).
5. `utils/image_loading.py`: add `pil_to_bgr(pil)` = `cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)` and
   `bgr_to_gray(cv)` = `cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)`; `load_image_from_path` (L1435) calls `pil_to_bgr`
   so the load path and every later rebuild are one expression (S9 uses both).
6. Fix the false log line in `detect_and_configure` (L268, "images loaded Nx per chunk"): state what is true
   (images are decoded once per chunk and shared by all passes) or delete the clause.

Key decision: share the bounded image only with consumers that provably preprocess to it (D4, D5), and move
composition into the pool because it is a pure function of pixels + config (D12).
Default: bounded image on for pyiqa (except `aesthetic_v25`/`deqa`) and Q-ReAlign; composition hoisted when not
restricted; a bounded-resize failure degrades to the full-resolution `pil`.
Verify:
- `tests/test_shared_bounded_image.py`: load 3 generated JPEGs (4000x3000, 3000x4000, 1024x700) and one
  degenerate (20000x1) through the real `_load_images`; for each, `torch.equal(PyIQAScorer._preprocess_image(
  bounded), PyIQAScorer._preprocess_image(pil))` (instantiate via `__new__` with `device='cpu'`) and
  `np.array_equal(np.asarray(QReAlignNativeScorer._prepare(q, bounded)), np.asarray(q._prepare(pil)))`; the
  degenerate photo has `bounded is None` and the pass falls back to `pil`. Stub-scorer pass tests assert the
  object handed to `score_batch` `is` the bounded image for `topiq`/`topiq_iaa`/`topiq_nr_face`/`liqe`, `is` the
  full `pil` for `aesthetic_v25` and `deqa`, and the bounded image for `qrealign`. Also asserts
  `liqe_input_too_small(*bounded.size) == liqe_input_too_small(*pil.size)` once the rebased tree has that
  helper (the open-gaps branch adds it; it bounds first, so the answers agree). Fault: change the resample in
  the worker to `Image.BICUBIC` -> tensor test fails; delete the `deqa` exclusion -> pass test fails.
- `tests/test_composition_hoist.py`: `_compute_aggregates` fed a hoisted dict equals the old inline logic (a
  verbatim reference copy in the test) on random images for both branches (SAMP-Net `comp_score` present and
  absent); a restricted processor never calls `get_placement_data` (counter); a raising `get_placement_data`
  produces `on_error(path, 'composition', ...)` (the stage is asserted, not `'load'`), omits the photo from
  `images`, the other photos of the chunk are still returned and the chunk continues (R8). Fault: hoist
  unconditionally -> the restricted test fails; label the stage `'load'` -> the stage assertion fails.
- Existing tests green: `test_multi_pass.py`, `test_multi_pass_pyiqa.py`, `test_multi_pass_pass_failure.py`,
  `test_review_fixes_scan.py`, `test_licensing_gate.py`, `test_extended_iqa_wiring.py`, `test_image_loading.py`
  (A1 owns the edits to the last one: its L221-238 tests drive `_load_images` with a MagicMock scorer). In every
  touched suite that calls the real `_load_images`, add `assert len(images) == len(paths)` so a hoist that
  starts failing silently inside the worker (a MagicMock config with real-content images) cannot pass.
- Central: wave W1a parity (SC1); harness `agg` phase per chunk <= 25% of baseline.
Affected files: `processing/multi_pass.py`, `utils/image_loading.py`, tests listed in the builder table.

### S2. Saliency `_score_from_mask` rewrite, verified on real masks (owner B, W1a)

Description: replace the float64 numpy body of `SaliencyScorer._score_from_mask` (`models/saliency_scorer.py`
387-467), `_compute_placement_score` (469-504) and `bbox_from_mask` (25-46) with the cv2 integer formulation
prototyped in `$SP\cpuio\bench_saliency_cand.py` (`score_from_mask_fast`, `make_mask`): `cv2.threshold` for the
mask, `cv2.Laplacian(gray, CV_16S)` with `cv2.meanStdDev(lap, mask=...)` for subject and `bitwise_not` for
background (variances = std^2), `cv2.reduce(..., CV_32S)` column/row sums for the centroid (exact integer sums)
and for the bbox (first/last non-empty column/row). Contract details:
- Input: `img_cv` may be a 3-D BGR array (as today) OR a 2-D gray array (already supported at L418-421); S9 passes
  gray. When the gray plane's shape differs from the mask's, the existing `cv2.resize(gray, (w, h))` is kept.
- Routing, with ONE behaviour per metric that reproduces today's code exactly (D22; the three thresholds differ
  and are NOT unified). The fast path serves EVERY 2-D `uint8` mask, binary or not; only masks that are not 2-D
  `uint8` (floats, bool, 3-D) take the existing numpy code unchanged. In the fast path:
  - `subject_pixels` (prominence and the `min_subject_pixels` early return) = `cv2.countNonZero(mask)`, i.e.
    any non-zero value, exactly `np.count_nonzero(mask)`;
  - the subject region is `mask > 128` (`cv2.threshold(mask, 128, 255, THRESH_BINARY)`); an EMPTY subject region
    gives `subject_variance = 0` and the background is the whole frame (`~(mask > 128)`), as today;
  - `bbox_from_mask` and the centroid use the `> 128` region; `bbox_from_mask` gates on the count of `> 128`
    pixels (`< min_subject_pixels` -> `None`), the centroid returns `5.0` when that region is empty;
  - variances = `std**2` from `meanStdDev` over the region, with `bg_variance > 0` evaluated on that value
    (a constant region must give exactly 0.0, as `np.var` does: tested);
  - every returned number is a Python `float` (`float(...)` before `round`), and `subject_bbox` is a list of
    Python floats or `None`, because the reference returns Python scalars; rounding and the early-return dict
    are untouched.
- `score_batch` (506-545) accepts, per image, an `ndarray` OR a zero-argument callable returning the ndarray
  (A2 passes `functools.partial` rebuilders so the gray plane exists only while one photo is scored).
- Streaming (S9 part): add a generator over mask sub-batches (the loop at L283-286 of `get_saliency_masks` with
  `_masks_with_oom_retry`), keep `get_saliency_masks` as `list(...)` of it for `get_saliency_mask` (L172) and any
  other list caller, and have `score_batch` score each sub-batch's masks as they arrive, so at most one
  sub-batch of masks is alive. The sub-batch size comes from `_affordable_batch_size` computed ONCE up front,
  exactly as now, so the model forwards are unchanged. A mask-generation exception still returns an all-failed
  list for the whole call (as L525-528 does today); a `None` mask or a scoring exception still yields the
  failed-scores dict for that image only, with the same ERROR log.

Key decision: exactness first. The Laplacian of a uint8 image fits int16 (max |value| 2040) so `CV_16S` is the
same integers as `CV_64F`; only the variance summation order differs (<= 9.0e-16 subject / 1.6e-15 background
relative), and outputs are rounded to 2 decimals.
Default: ndarray or callable per image; list-returning `get_saliency_masks` kept as a thin wrapper.
Verify:
- `tests/_saliency_reference.py` holds the verbatim OLD `_score_from_mask`/`_compute_placement_score`/
  `bbox_from_mask`. `tests/test_saliency_exact.py` compares new vs reference with exact equality of the returned
  dict on hand-built masks: empty, below/at/above `min_subject_pixels`, full frame, 1-pixel-wide row and column,
  checkerboard, subject touching each border, mask size != image size, constant image (all-zero Laplacian in
  subject and/or background), 2-D gray vs 3-D BGR input giving the same dict, and the callable element form.
  Non-binary uint8 masks are asserted against the REFERENCE, not against a route: values only in 1..128 (gate
  passes, subject region empty: expects the reference's `bg_separation`, placement 5.0 and bbox `None`), values
  at exactly 128 and 129, a ramp 0..255, and a mask whose `> 128` count is below `min_subject_pixels` while its
  non-zero count is above. Non-uint8 (float32, bool) masks assert the numpy fallback gives the reference dict.
  The comparison is `type(a) is type(b) and a == b` per dict value and per `subject_bbox` element (floats by
  bit pattern), so an `np.float64` return fails against the reference's Python `float`. Plus a live-mask counter
  test: a stub model returns masks through `_run_chunk`; the peak number of simultaneously alive masks never
  exceeds the sub-batch size for a 7-image chunk with batch size 2. Existing `test_saliency.py` (345-597) and
  `test_social_crop.py` green unchanged.
- REAL masks: `scripts/perf/verify_saliency_exact.py` loads the 49 PNG masks from `$SP\perf2\masks` and the 49
  photos, runs reference and new on each (3 reps for timing), and exits 0 only if 49/49 dicts are equal under the
  same `type(a) is type(b)` + bit-pattern rule and the new/reference time ratio <= 0.25 (SC5); exit 2 if
  fewer than 49 pairs exist.
- Fault: in the new centroid code swap the row and column sums -> the real-mask script exits 1 with mismatches
  and the unit tests fail; change `CV_16S` to `CV_8U` (saturates) -> both fail; gate the early return on the
  `> 128` count instead of the non-zero count -> the values-in-1..128 case fails; drop a `float(...)` cast ->
  the type check fails. Restore by hand (V7).
- Central: wave W1a parity.
Affected files: `models/saliency_scorer.py`, `tests/test_saliency.py`, `tests/test_saliency_exact.py`,
`tests/_saliency_reference.py`, `scripts/perf/verify_saliency_exact.py` (created in S0's directory by B).

### S3. Exact replacements in `analyzers/technical.py` and `analyzers/image_cache.py` (owner C, W1a)

Description: port the candidates from `$SP\cpuio\bench_candidates.py` (functions `hist256`, `hist_percentile`,
`hist_std`, `dynamic_range_hist`, `contrast_hist`, `noise_i16`, `lap_var_i16`):
- `get_dynamic_range` (technical.py 274): percentiles 2 and 98 from a 256-bin histogram of the uint8 gray with
  numpy's `linear` method; stored value identical (147/147).
- `get_contrast_score` (337): percentiles 5/95 from the histogram (identical), `np.std` replaced by the exact
  histogram std (relative difference <= 2.9e-15; rounded outputs identical 147/147). No `astype(np.float64)`
  full-frame copy.
- `get_noise_estimate` (305): `cv2.filter2D(gray, CV_16S, kernel_f32)` + `cv2.norm(..., NORM_L1)` (integer sum
  identical 147/147). The final arithmetic MUST stay numpy-float64:
  `sigma = np.float64(s) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))`, so a 2-pixel-wide image yields `inf`
  (with a numpy warning) as today instead of raising `ZeroDivisionError` on Python floats.
- `ImageCache.__init__` (image_cache.py:31): `laplacian_variance = float(std)**2` from `cv2.Laplacian(gray,
  CV_16S)` + `cv2.meanStdDev` (relative difference <= 6.1e-15; NOT bit-identical; this is the one tolerance
  source, R5). The attribute is `np.float64` today (D11), not a Python `float`; the new value is a Python
  `float`. Both subclass/are `float`, bind identically in sqlite3 and format identically in `round(...)`, so
  there is no type to preserve; the test asserts `isinstance(v, float)` only.
- Leave unchanged: `get_histogram_data` (146), `get_color_harmony_data`, `detect_monochrome`, the `cache=None`
  paths' signatures, and `analyzers/composition.py`. Where a function has a no-cache path it gets the same
  replacement (one implementation, two entry points) so both stay consistent.
- Side effect to record: the ~154 MB float64 temporaries that raised `MemoryError` in an 8-worker run disappear.

Key decision: take only replacements that are bit-identical after rounding; the unrounded Laplacian variance is
the single declared tolerance.
Default: `CV_16S` Laplacian variance and histogram-based percentiles/std everywhere in these two files.
Verify:
- `tests/test_technical_exact.py`: a verbatim reference copy of each OLD function in the test; compare on
  synthetic arrays (random noise, gradient, constant 0 and 255, 1xN and Nx1, 2x2 (noise -> `inf` both), 3x3,
  checkerboard, saturated halves) and on RGB images of several sizes: dynamic range and noise dicts exactly
  equal; contrast dict exactly equal after the function's own rounding; `laplacian_variance` within 1e-12
  relative; `get_sharpness_data` normalised score equal after `float(min(10.0, v/50.0))` at 2 decimals.
  `scripts/perf/verify_technical_exact.py` repeats it on the 49 photos (exit 0 with `compared 49 photos`, exit 2
  if fewer) and prints the new/reference ratio (<= 0.25, SC5).
- Fault: make `hist_percentile` ignore the `gamma >= 0.5` branch -> the synthetic set fails; compute the noise
  tail with Python floats -> the 2x2 case fails (`ZeroDivisionError` vs `inf`). Restore by hand (V7).
- Existing: `test_analyzers.py`, `test_batch_processor.py`, `test_batch_processor_e2e.py`, `test_scorer.py`
  green (single-pass paths share this code, R14).
- Central: wave W1a parity with the tolerance dict; any other differing column is a failure (R5).
Affected files: `analyzers/technical.py`, `analyzers/image_cache.py`, tests listed.

### S4. Overlap chunk decode with pass-1 model loading (owner F, W2)

Description (`ChunkedMultiPassProcessor._process_chunk`, 401-501; `_load_pass_models`, 523): submit
`_load_images(paths)` to a single-worker `ThreadPoolExecutor`, run `_load_pass_models(first_group, ...)` on the
main thread meanwhile, then `future.result()` and continue into the group's inference with the images, ONLY
when `_overlap_allowed(paths, first_group)` is true; otherwise decode synchronously exactly as today. The gate:
1. Window: "cheaply restorable" is DEFINED, not guessed. A group-0 model name `n` has a load window iff
   `n` is neither resident (`n in model_manager.models`) nor held in the RAM cache (`n in
   model_manager._cpu_cache`). A RAM-cache restore is a host-to-device copy of seconds-or-less and counts as
   "ready": it leaves no window worth a thread. E adds `ModelManager.is_resident_or_cached(name)` (S6a, W1a) so
   F does not read two private attributes; a MagicMock model manager answers truthy, i.e. "no window", i.e.
   today's synchronous path.
2. Device and policy (G2): `model_manager.device == 'cuda'` AND `model_manager.keep_in_ram != 'never'`. CPU
   and MPS hosts, and `keep_in_ram: "never"`, always take today's synchronous path (R3).
3. Budget (G2): `_available_bytes() >= decode_estimate + staging_estimate + reserve`, where
   `decode_estimate = n_paths * mean_px * RETAINED_BYTES_PER_PX` (`mean_px` = previous chunk's mean, or 20e6 for
   chunk 0 when none is known: the 49-photo set spans 12-40 MP and the conservative figure only costs a missed
   overlap), `staging_estimate = sum(MODEL_RAM_REQUIREMENTS[n]` for group-0 names with a window`)` and
   `reserve = model_manager.ram_cache_reserve_gb` (S6a). `_available_bytes()` is the model manager's provider
   (S6a), so tests inject a fake. When the budget fails the log says so once at INFO and the chunk decodes
   synchronously.
Details:
- Warm-up on the main thread (G1, D20, R19): `form_facet.warmup()` is called ONCE on the main thread in
  `process_directory`, before the decode executor is created (and therefore before `_load_pass_models` can start
  torch/transformers pool initialisation); the call inside `_load_images` stays (it is a no-op once `_WARMED`)
  so the single-pass and test paths are unchanged.
- Exceptions: `future.result()` re-raises. A decode failure (`RuntimeError`, "decode abandonment budget
  exhausted", must still fail fast) after group 1's models are loaded must unload them (`unload_model(name,
  reclaim=False)` + `_reclaim_freed_memory`) in a `finally` so a failed chunk leaks no model. A model-load
  failure behaves as today (supplementary skipped, required recorded in `failed_stages`).
- Cancellation (G8, R20): `_load_images` takes a `stop_event` (`threading.Event`) checked before each photo is
  submitted and at the top of `_load_one`; `process_directory`'s `finally` sets it, then calls
  `executor.shutdown(wait=False, cancel_futures=True)`. A KeyboardInterrupt on the main thread (during
  `_load_pass_models` or anywhere else) therefore returns within one in-flight photo's decode per pool worker,
  not after the whole chunk. Pending futures are cancelled; the in-flight ones finish into a discarded result.
- The pool is created per scan and shut down in `process_directory`'s `finally` (384-394) before `unload_all`.
- Metrics: `io_time` keeps meaning "main-thread time blocked on the decode" (the part not hidden); new
  `io_decode_time` is the worker's wall. `_print_summary` (1316, io label at 1336) prints both.
- Thread-safety checked: `_load_one` uses only per-photo state, `self.scorer.get_exif_data` (already called
  from pool threads) and `exiftool` batch results; nothing in `load_model_only` touches it.
- The group-1 `_load_pass_models` call moves out of the per-group loop only for group 0 (its timing still goes to
  `model_load_time`).

Key decision: overlap only when it can hide something (non-resident first-group model), because after S5 the
boundary group stays resident and an unconditional overlap would add thread overhead for nothing.
Default: single decode worker (the pool inside `_load_images` already provides the parallelism).
Verify:
- `tests/test_scan_decode_overlap.py`, all DETERMINISTIC (events and barriers on stub loaders, no wall-clock
  assertion, no `sleep`-based timing):
  - overlap: a stub `_load_images` and a stub `_load_pass_models` each call `barrier.wait(timeout=5)` on a
    shared 2-party `threading.Barrier`; the test passes only if both are inside their calls at the same time
    (sequential execution breaks the barrier and fails with `BrokenBarrierError`); then event ordering is
    asserted: decode-start precedes load-end precedes `future.result()` returning, and the images exist before
    group-0 inference starts;
  - gate (G2, with the S6a fake provider): not overlapped when (a) every group-0 name is resident or in the RAM
    cache (window definition), (b) the device is `cpu` or `mps`, (c) `keep_in_ram == 'never'`, (d) the fake
    available bytes are below `decode_estimate + staging_estimate + reserve`; overlapped when all four allow it.
    Each negative case asserts the synchronous path ran (the stubs do NOT need the barrier);
  - warm-up (G1): a patched `form_facet.warmup` records `threading.current_thread()`; it is
    `threading.main_thread()`, it ran before the executor's first submit, and it ran before
    `_load_pass_models` was called. Fault: move the call into the submitted function -> fails;
  - exceptions: decode `RuntimeError` -> raised AND the loaded group-0 models were unloaded (recorded
    `unload_model` calls) AND the executor shut down; a model-load failure still skips/records as today;
  - cancellation (G8): a stub `_load_one` blocks on an event; the main thread raises `KeyboardInterrupt` inside
    the stub `_load_pass_models`; `process_directory` unwinds; the test asserts it returned while the decode
    was still blocked (so it did not wait for the chunk), that after the event is released no photo BEYOND the
    in-flight ones started (stop flag), and that pending futures are cancelled (`cancel_futures=True`
    asserted with a recording executor). Fault: `shutdown(wait=True)` -> the test hangs into its timeout and
    fails;
  - metrics: `io_time` (main-thread blocked time) + `io_decode_time` (worker wall) semantics; `restricted`
    single-group scans overlap when the gate allows it.
  Fault: call `future.result()` before `_load_pass_models` -> the barrier test fails.
  Existing `test_scan.py`, `test_scan_interrupt.py`, `test_multi_pass*.py` green.
- Central: wave W2 parity; harness `io` main-thread wait per chunk (median, chunks >= 1) <= 35% of the baseline's
  in the same measurement.
Affected files: `processing/multi_pass.py`, tests listed.

### S5. Serpentine pass order across chunks (owner G, W3)

Description: alternate the pass-group order every other chunk and keep the boundary group resident:
- Order is derived from a CARRY state, not from the parity of `chunk_idx` alone. A chunk runs `pass_groups`
  reversed iff the previous chunk ended by KEEPING its last group resident and ran forward; otherwise it runs
  forward. When the previous chunk kept its last group, that group is the first group of the next chunk's
  order by construction (forward ends with group N, reversed starts with group N; reversed ends with group 1,
  forward starts with group 1). `_process_chunk` skips `unload_model` + the per-group
  `_reclaim_freed_memory` for the LAST group of a chunk when the keep gate (below) allows it and the chunk is
  not final. "Final chunk" is DEFINED as `offset + len(chunk_paths) >= total` evaluated after the chunk is
  sliced, using the live (auto-tuned) chunk size; `num_chunks_est` is only an estimate (it is recomputed from
  a chunk size that can shrink mid-run) and is never used for this decision. The final chunk unloads
  everything as today, and the `process_directory` `finally` runs `unload_all` (384-394) for the abnormal-exit
  case. With one group (`run_single_pass`, any single-group plan) the order is trivially identical and nothing
  is kept resident: S5 is a defined no-op for `len(pass_groups) < 2`.
- Keep gate (G2, R3): the boundary group is kept resident only when `model_manager.device == 'cuda'`,
  `keep_in_ram != 'never'` and `_available_bytes() >= next_decode_estimate + reserve` (same estimate as S4's
  gate, with this chunk's mean pixels). When the gate says no for a chunk, that chunk unloads its last group as
  today and the NEXT chunk runs forward (carry reset), i.e. exactly today's behaviour. CPU and MPS hosts
  therefore never serpentine.
- Safety from the plan, not from a hope: a module-level table `PASS_WRITE_KEYS` maps every model name to the
  result keys its pass writes (D8; `faces_permissive` includes `face_quality_iqa`, `insightface` does not;
  `qrealign` includes `aesthetic`/`quality_score`/`scoring_model` only when `_qrealign_is_primary()`; sentinel
  keys excluded). `serpentine_safe(pass_groups)` is true iff every key written by more than one model has all its
  writers in the same group. When false (commercial plans, R4) run forward order every chunk and log once at
  INFO which key forced it.
- `_next_use_order()` (introduced by F in S6b as forward order) is updated to the serpentine order of the NEXT
  chunk so Belady eviction and the pre-chunk trim stay correct. This is the S5<->S6b contract.
- Resident-group VRAM: the kept group is exactly the group the next chunk runs first, so no two groups are ever
  co-resident; peak VRAM is unchanged.

- HARD CONDITION (G4, D19, R4): write-set independence does not imply input independence. Serpentine changes
  which group precedes saliency (Qwen/torch on forward chunks, insightface/ONNX Runtime on reversed ones) and
  therefore the free-VRAM reading that sizes the BiRefNet sub-batches and the pyiqa forward batches, and batch
  composition changes stored values. S5 SHIPS ONLY IF byte parity (SC1 and SC2) holds on BOTH the 16gb and the
  8gb profile with the per-arm sub-batch ledger (`batch_sizes.json`, V3) identical between baseline and
  patched. If either parity run fails, or the ledgers differ, G drops S5 (reverts its code and tests, keeps
  S8c if separately justified) and records why in the handoff; the orchestrator does not weaken the comparator
  to keep it.
- Preferred mitigation, tried first when the baseline ledger shows the sub-batch sizes already sit at their
  ceiling (`DEFAULT_BATCH_SIZE`): make the saliency and pyiqa batch sizing independent of pass order, e.g. size
  from a budget fixed at the start of the pass or from a measurement that does not depend on what ran before
  (free VRAM plus torch-cached bytes of the SAME process is exactly what depends on it). It is adopted ONLY if
  today's values stay byte-identical (the parity runs say so); a mitigation that changes any size relative to
  the baseline is not a mitigation, it is a result change, and S5 is dropped instead.
- Residual risk when S5 ships: the batch sizing still depends on the host's VRAM, so parity on this 16 GB card
  does not prove it on a smaller or busier GPU. R4 records this and the CHANGELOG line says it.

Key decision: reorder only where the write sets prove order cannot change a stored value, and only when the
sub-batch ledger proves the inputs do not move either.
Default: serpentine on for any plan with >= 2 groups whose multi-writer keys are intra-group, on CUDA, with
`keep_in_ram != 'never'` and a satisfied budget; forward otherwise.
Verify:
- `tests/test_scan_serpentine.py`: (a) group order for chunks 0..5 on a 3-group plan; kept-resident behaviour:
  `unload_model` not called for the last group except on the final chunk, and the next chunk's
  `load_model_only` for those names returns without a load; single-group no-op; final-chunk definition:
  a run whose chunk size SHRINKS mid-run (chunks of 16, 16, 8, 8, ...) unloads on exactly the chunk where
  `offset + len >= total`, and a run with `num_chunks_est` deliberately wrong still does; the keep gate with the
  S6a fake provider (device `cpu`/`mps`, `keep_in_ram == 'never'` and a too-small available figure each give
  forward order, unload as today and a reset carry on the next chunk); (b) `serpentine_safe` on the real
  default 16gb and 8gb plans (true) and on a commercial-shaped plan with `faces_permissive` and `topiq_nr_face`
  in different groups (false) with the INFO log; (c) write-set drift: run every `_pass_*` with stub scorers whose
  outputs are deterministic and assert the keys each writes equal the `PASS_WRITE_KEYS` entry (a new written
  key fails the test); (d) passes are write-only on `results` (a dict that raises on read proves no pass reads
  another pass's output); (e) order independence: run the default plans' groups forward and reversed on stub
  scorers and assert identical `results`, plus a NEGATIVE CONTROL that the commercial-shaped plan gives
  different `results` (proving the table, not the test, carries the conclusion); (f) `_next_use_order()` follows
  the serpentine order. Fault: drop `face_quality_iqa` from the `faces_permissive` entry -> (c) and (e) fail.
- Central: wave W3 parity (byte-identical, SC1) AND the 8gb run `P8-W3` vs `B8a` (SC2), both with identical
  `batch_sizes.json` ledgers per arm (G4 hard condition above); the `load:*` events are asserted from
  `chunks.py`, with the counts derived from the plan, not guessed: for n chunks on a forward-first plan the
  FIRST group (Qwen on 16gb) loads from disk `floor(n/2) + 1` times (n=4: chunks 0, 1, 3 = 3 loads, not 2) and
  the LAST group (insightface) `ceil(n/2)` times (n=4: chunks 0, 2 = 2), instead of n each. A run whose keep gate
  said no for some chunk is checked against its own gate log, not these formulas. Fault: unload the last group
  on every chunk -> both load counts return to n and the check fails.
- Coverage of the gate itself: every chunk logs `serpentine: keep=<bool> reason=<...>` at INFO, and the W3
  parity runs (16gb and 8gb) are valid evidence for S5 only if serpentine was ACTIVE on at least 2 chunks
  (a reversed chunk actually ran). Sibling processes on this box hold 9-24 GiB (D9), so a budget gate that
  refuses every chunk would make the parity run trivially "identical" because S5 never ran. Such a run is
  reported INVALID FOR S5 and repeated with the siblings stopped or the reserve lowered for the run; it is
  never counted as a pass.
- R2 measurement: G runs W3 with S4 present and with S4's overlap disabled for one alternating pair, reports
  whether S4 still pays; the result goes to the handoff and the CHANGELOG wording.
Affected files: `processing/multi_pass.py`, tests listed.

### S6. RAM-cache policy by available bytes, margin and floor (owners E in W1a, F in W2)

Monitor interaction first, because the orchestrator asked: S6 touches ONLY the monitor's eviction. The 85%
shrink trigger, the 25% shrink step, the 65% grow gate (`note_chunk_complete`, L92-109), `min/max_ram_chunk_size`
and `memory_limit_percent` are unchanged. After S6 the monitor tick at percent > 85 does (1) shrink the chunk
(unchanged) and (2) evict cached models ONLY while `available < reserve`, in Belady order and only as many as
needed (was: always all). The grow gate stays dead on busy boxes (D9) and is recorded as a non-goal/future
item (it needs a peak-vs-trough signal that survives sibling noise, and changing it changes chunk sizes, which
the task forbids).

**S6a (owner E, `models/model_manager.py`, W1a)**
- Memory-provider seam: `ModelManager.__init__(self, config, memory_provider=None)`; `self._memory_provider =
  memory_provider or utils.system_memory.effective_memory` (lazy import as today); one method
  `_available_bytes()` returns `self._memory_provider().available`. All reads of available memory in the cache
  code go through it so tests inject a fake without patching globals.
- `ram_cache_reserve_gb`: `float(model_config.get('ram_cache_reserve_gb', 4.0))` read in `__init__` next to
  `keep_in_ram` (L165); non-numeric or negative -> `ValueError` naming the key. Replace the class constant
  `_RAM_HEADROOM_GB` (L120) and its use at L536 by this value, same form `available_gb > model_ram +
  reserve_gb`, so the DEFAULT admission behaviour is identical. Update the docstring at L467.
  NOT added to `config/scoring_config.default.json` (R6).
- `evict_cpu_cache(self, needed_available_bytes=None, next_use=None) -> list[str]` (L679-710). `None` target =
  today's evict-everything. With a target: loop { if `_available_bytes() >= needed` stop; pick the victim by
  Belady (the cached model whose next use is farthest; names absent from `next_use` go first, cache insertion
  order breaking ties); `pop(name, None)` (race discipline, R17); report skips; `gc.collect()` +
  `release_freed_heap()`; re-sample }. Returns the evicted names; logs `Evicted N model(s) from RAM cache: ...`
  as before and the target in DEBUG. Eviction can free RAM on CPU devices too (`_move_to_cpu` is a no-op there but
  dropping the entry releases the model).
- Termination: with a target that can never be reached (the provider keeps reporting too little available
  memory, e.g. a sibling process holds it) the loop evicts every cached model once and returns; it never spins,
  and the return value is the full list (the caller can tell "reached" from "emptied" by re-reading
  `_available_bytes()`).
- `ModelManager.is_resident_or_cached(name) -> bool`: `name in self.models or name in self._cpu_cache`; the one
  predicate S4's gate uses (D: "cheaply restorable"). Reads only, no side effects.
- Keep every new read AFTER the `keep_in_ram == 'always'` return (L528): `tests/test_licensing_gate.py:1148-1161`
  calls `_can_cache_to_ram` on a `SimpleNamespace` stub that has no `ram_cache_reserve_gb` or provider.
- Docstrings: replace the "free-memory threshold" narrative with the new contract.

**S6b (owner F, `processing/resource_monitor.py` + `processing/multi_pass.py`, W2)**
- Monitor `run` (L60-90): replace the unconditional `evict()` by
  `evict(needed_available_bytes=<reserve in bytes>, next_use=<order>)`, where both values are read with
  `getattr` and a default, per tick: `mm = getattr(self.processor, 'model_manager', None)` (already guarded for
  None); `reserve_gb = getattr(mm, 'ram_cache_reserve_gb', None)`; `order_fn = getattr(self.processor,
  '_next_use_order', None)`. If `reserve_gb` is None the call is today's `evict()` (evict everything); if
  `order_fn` is None, `next_use=None`. This keeps `tests/test_resource_monitor.py`'s `_FakeChunkProcessor`
  (`model_manager = None`, no `_next_use_order`, L102-116) working unedited, and keeps the other stubs that
  predate the reserve working.
- The existing `except Exception: pass` in `run` (L89-90) stays as a crash guard for the thread but may no longer
  be silent about eviction: on the first exception the monitor logs ONE `logger.warning` with the exception
  (a `_warned_failure` flag; later exceptions stay quiet so a persistently failing tick cannot flood the log).
  A raise from the new reads must therefore show up in the log AND in `test_ram_cache_trim.py` instead of
  silently turning eviction off while "existing tests green" still passes.
- Pre-chunk trim: new `ChunkedMultiPassProcessor._trim_ram_cache_for_next_chunk(n_next)` called in
  `process_directory` before `_process_chunk` for chunks >= 1 (L360-382), on the scan thread. Estimate =
  `n_next * mean_px_of_previous_chunk * RETAINED_BYTES_PER_PX` (constant defined by A2 in S9 as the measured
  retained B/px rounded UP to the next 0.25; F reads it, never redefines it) plus the reserve; if
  `_available_bytes() < estimate`, call `evict_cpu_cache(needed_available_bytes=estimate,
  next_use=self._next_use_order())`. `_process_chunk` records `self._last_chunk_mean_px` from the loaded
  images. Log the decision at DEBUG, the eviction at INFO (existing line).
- `_next_use_order()`: names in the order the next chunk will first request them (forward order of
  `pass_groups` flattened). S5 (G, W3) replaces the body with the serpentine order of chunk `i+1`.
- `RETAINED_BYTES_PER_PX` exists after W1b; F's W2 follows it, so there is no forward reference.

Key decision: a floor expressed in absolute available bytes (`ram_cache_reserve_gb`, default = today's 4.0 GB
headroom) replaces the all-or-nothing percent-triggered wipe; the margin for the next chunk is the retained-bytes
estimate from S9. One knob, no second knob for the margin.
Default: reserve 4.0 GB; partial Belady eviction; trim only when the estimate does not fit.
Verify:
- `tests/test_ram_cache_policy.py` (E, fake provider = a closure over `manager._cpu_cache` that reports
  `base_available + sum(size of evicted models)`): admission boundary (available == model + reserve is refused,
  +1 byte admitted), `keep_in_ram='never'/'always'` unchanged, configured reserve 1.0 admits where 4.0 refuses,
  invalid value raises; `evict_cpu_cache()` with no target evicts all and logs as before; with a target it stops
  as soon as the fake says satisfied (evicts 1 of 3), uses Belady order for `next_use=['c','b','a']` (evicts
  `a` first), puts names missing from `next_use` first, tolerates a concurrent `_restore_from_cache` (reuse the
  existing race harness at `test_model_manager.py:1154-1296`), and returns the evicted list; an UNREACHABLE
  target (the fake never reports enough) evicts all three, terminates and returns all three names;
  `is_resident_or_cached` is true for a resident model, true for a cached one, false otherwise. Fault: make the
  loop ignore the target -> the "stops when satisfied" test fails; reverse the Belady key -> the order test
  fails; make the loop `while True` without the emptiness exit -> the unreachable-target test hangs into its
  timeout and fails.
  Existing `test_model_manager.py` (488-537, 600-645, 1154-1296, 1325-1366) and `test_licensing_gate.py` green.
- `tests/test_ram_cache_trim.py` (F): monitor tick with a stub processor/model manager and a monkeypatched
  `processing.resource_monitor.effective_memory`: percent 86 with available above the reserve shrinks the chunk
  but evicts nothing; percent 92 with available below evicts with the reserve target; the grow gate behaves as
  before (existing `test_resource_monitor.py` green unchanged); trim not called for chunk 0, called with the
  estimate for chunk >= 1, and skipped when it fits. Stub tolerance: a processor with `model_manager = None`,
  and one whose model manager lacks `ram_cache_reserve_gb`, tick without raising (the second falls back to
  `evict()` with no target). "Eviction really happens" test (G10): a tick at 92% with available below the
  reserve calls `evict_cpu_cache` on a recording manager with exactly `needed_available_bytes == reserve` and
  the processor's `_next_use_order()` result. Failure visibility: a processor whose `_next_use_order` raises
  makes the tick log exactly one WARNING (caplog) and `reduce_chunk_size` was still called before it. Fault:
  restore the unconditional `evict()` -> the "no eviction above reserve" test fails; wrap the eviction call in
  a bare `pass` again -> the failure-visibility test fails.
- SC7 run (orchestrator, after W2): `ballast.py` raises system memory until the BASELINE log shows `Evicted`
  lines; same ballast level for the patched run; compare hits (`chunks.py`), evicted counts, swap growth and
  parity. Honest outcome if no level evicts the baseline in three tries: INCONCLUSIVE.
- Low-RAM note (also goes into the docs, S-DOC): on a host with little free RAM the cache simply holds fewer
  models; correctness never depends on a hit (a miss reloads from disk). With `ram_cache_reserve_gb` raised the
  cache gives memory back earlier; with `keep_in_ram: "never"` it is off.
Affected files: E: `models/model_manager.py`, `tests/test_model_manager.py`, `tests/test_ram_cache_policy.py`.
F: `processing/resource_monitor.py`, `processing/multi_pass.py`, `tests/test_resource_monitor.py`,
`tests/test_ram_cache_trim.py`.

### S7. Skip the post-scan tagging model load when no photo needs tags (owner D, W1a)

Description: in `facet.py` step 6 (1973-1986) decide BEFORE `resolve_scan_tagger`: if `tagging.enabled` is false
or `count_untagged(scorer.db_path) == 0`, skip both `resolve_scan_tagger` and `run_tagging` and log the existing
"No new tags assigned" sentence (or a "tagging disabled" sentence). Today the model is loaded first and
`run_tagging` checks `enabled` afterwards (D14), so a disabled-tagging scan loads a SigLIP/CLIP model for
nothing. In `tag_existing.py` add `count_untagged(db_path)` and give the three SQL sites (the non-force query in
`tag_untagged_photos` 42-47, the count in `main` ~180-183, the new function) ONE shared predicate constant
`clip_embedding IS NOT NULL AND (tags IS NULL OR tags = '')`. `--tag-untagged` (facet.py ~3480-3510) already
skips the load when nothing is pending and is not touched. Steps 7/8 are independent of step 6's model (D14).

Key decision: gate on the exact predicate the tagging query uses, so skipping can never skip a photo the query
would have tagged.
Default: skip when disabled or pending == 0; otherwise unchanged.
Verify:
- `tests/test_scan_tagging_skip.py`: DB from `init_database` with rows (tags NULL, tags '', tagged, no embedding):
  `count_untagged` equals `len(rows)` of the real `tag_untagged_photos` query; pending 0 -> neither
  `resolve_scan_tagger` nor `run_tagging` is called (counters) and the DB is byte-identical afterwards;
  pending > 0 -> both called once; `tagging.enabled=false` -> neither called. Fault: drop `OR tags = ''` from
  the shared predicate -> the equality test fails.
- Existing `test_tagging.py`, `test_manual_tags_import.py`, `test_retry_failed_cleanup.py`, `test_scan.py` green.
- Central: harness event `post:tagging` of the 16gb run drops to ~0 when all 49 photos got Qwen tags; if the run
  leaves some untagged, the load legitimately happens and the report says so. Parity covers `tags`.
Affected files: `tag_existing.py`, `facet.py` (step 6 only), tests listed.

### S8. Small exact items (owners D for 8a, A2 for 8b, G for 8c)

- S8a Thumbnail without `pil.copy()` (`utils/image_transforms.py` 50-68, called from `processing/scorer.py`
  ~2530 and ~2667): compute the target size with Pillow's `preserve_aspect_ratio` rounding (port `thumb_nocopy`
  in `$SP\cpuio\bench_candidates.py`), return the JPEG of `pil_img` itself when it already fits, else
  `pil_img.resize(final, LANCZOS, reducing_gap=2.0)`; never `thumbnail()` on the original (R9). Verify:
  `tests/test_thumbnail_exact.py` compares bytes against a verbatim copy of the old function on a size matrix
  (1x1, 639x640, 640x640, 641x640, 640x1, 1x640, 4000x3000, 3000x4000, 20000x2000, 5494x3666) and on the 49
  photos when the env `PERF_PHOTOS` names them (else the synthetic matrix still runs, never skipped), ratio
  new/reference <= 0.75 (SC5). Fault: `floor`-only rounding in `round_aspect` -> a matrix case fails. Existing
  `test_thumbnails.py`, `test_render_version.py` green.
- S8b SAMP-Net input from PIL (`_pass_samp_net`, multi_pass.py 785): `scorer.score(images[path]['pil'])`;
  SAMP-Net's `preprocess` (`models/samp_net.py` 975-999) already accepts PIL and the tensor is `torch.equal`
  to the BGR route. Implemented by A2 inside S9 because S9 removes the BGR input. Verify: a test builds both
  inputs for RGB images of three sizes and asserts `torch.equal(scorer.preprocess(pil),
  scorer.preprocess(pil_to_bgr(pil)))` (instantiate the scorer without weights); fault: pass `pil.convert('L')`
  -> fails. Test file `test_scan_retained_bytes.py`.
- S8c `cv2.setNumThreads(1)` during the load phase only if measured beneficial (owner G, W3, after S4 because the
  decode then runs concurrently with model staging): CPU-only A/B with `retained_bytes.py`-style `_load_images`
  on `first:32`, 4 workers, 3 alternating pairs; adopt only if the median load wall improves >= 5% with no
  regression in any pair. If adopted: set before, restore in a `finally` (including on exceptions) to the prior
  `cv2.getNumThreads()` value; test proves restoration after a raising load (fault: remove the `finally`). If not
  adopted: record "measured, rejected" with the numbers in the handoff; no code. Weak prior evidence
  (3 reps, contended box) is why it is conditional.

### S9. Stop retaining `cv`, `cache.gray` and the saliency mask list (owner A2, W1b; LOCKED)

From `chunk-size.md` tiers T1 + T2 + T2b. Retained peak 9.05 -> ~4.05 B/px (-55%) with identical results.
Description (`processing/multi_pass.py`, with B's streaming from S2 and A1's helpers from S1):
- `_load_one` stops returning `'cv'`; after the composition hoist (S1) it sets `cache.gray = None`
  (`ImageCache` keeps `laplacian_variance`, a float, which the face pass reads at L836). `hsv` is already freed at
  L643.
- `_pass_samp_net`: PIL input (S8b).
- `_pass_insightface` (799): inside the per-photo `try`, `img_cv = pil_to_bgr(img_data['pil'])`; set it to `None`
  at the end of each iteration so at most one BGR copy (6 B/px transient for ONE photo) is alive. `img_h, img_w`
  come from `img_data['height']`/`['width']` (set from the same array shape at load). A rebuild failure is a face
  failure like an analyzer failure (same `FACE_FAILED_KEY` path).
- `_pass_saliency` (879): `scorer.score_batch(pil_imgs, [functools.partial(rebuild_gray, pil) ...])` with
  `rebuild_gray(pil) = bgr_to_gray(pil_to_bgr(pil))` (the exact chain D3 proved; never `RGB2GRAY` directly), so
  the gray plane exists only while one photo is scored. B's streaming (S2) removes the whole-chunk mask list.
- `_compute_aggregates` no longer touches `img_cv`: the `img_cv = img_data['cv']` line (L1017) is REMOVED by A1
  in S1 (it is the last reader of `'cv'` there), so by the time S9 deletes `'cv'` from `_load_one`'s return
  value nothing in `_compute_aggregates` can KeyError. A2 re-greps `\['cv'\]`, `.get('cv')`, `img_cv` and
  `.gray` across `processing/`, `models/`, `analyzers/` and `tests/` and lists every hit it changed or left.
- `RETAINED_BYTES_PER_PX`: class constant on `ChunkedMultiPassProcessor`, set to the measured
  `bytes_per_px_retained_wset` of this tree rounded UP to the next 0.25 (expected 4.25), with a docstring pointing
  at `scripts/perf/retained_bytes.py`. S6b consumes it.
- Metrics: `self.metrics['rebuild_time']` accumulates the two rebuilds; printed in `_print_summary`.
- Tests that encode today's retention are changed, each assertion change DISCLOSED by name in the handoff and
  justified: `tests/test_multi_pass.py` L700-711 (`assert entry['cache'].gray is not None` encodes retention,
  not behaviour -> asserts `gray is None` and `'cv' not in entry`); `tests/test_multi_pass.py` L557-568
  `test_the_passed_analyzer_is_the_one_that_runs` (puts a MagicMock in `images['cv']` with no `pil` and asserts
  `analyzer.seen == [img]`; after S9 the face pass rebuilds BGR from `pil`, so this is an ASSERTION change, not
  a signature adaptation: the fixture gets a real small PIL and the assertion becomes "the analyzer saw an array
  equal to `pil_to_bgr(pil)`" -- the analyzer-identity claim the test exists for is kept); `test_multi_pass.py`
  L282 (`'cv': None` fixture: drops the key, harmless); `test_licensing_gate.py` L708 and L726 (`{'cv': MagicMock()}` -> a real small PIL,
  because a MagicMock `cv` would now pass vacuously) and L1261+ (`'cv'` key in the `_compute_aggregates`
  fixture -> `'composition'`); `test_multi_pass_pass_failure.py`, `test_saliency_failure.py` (L41 passes
  `object()` pil and cv), `test_face_models.py`, `test_review_fixes_scan.py`: replace `'cv'` fixtures by real PIL.
- Explicitly NOT done: chunk-size defaults/thresholds, downsized derivatives (rejected), `pil` -> ndarray (T3).

Key decision: drop only what is a pure function of the retained `pil` (D3) and rebuild it exactly where a later
pass needs it; keep `pil` at full resolution because SigLIP, Qwen, saliency, the thumbnail and the face pass need it.
Default: T1+T2+T2b. Fallback T1 (keep `cache.gray`) if R7's trigger fires; the SC4 bound then becomes 5.30.
Verify:
- `scripts/perf/retained_bytes.py` on the PERF tree, same inputs as the S0 baseline numbers (`mp:20-21:4`,
  `first:32`): `bytes_per_px_retained_wset` <= 4.30 vs ~8.05; bucket keys contain neither `cv` nor `cache.gray`;
  the table before/after goes in the handoff. `--saliency-live-masks` reports max live masks <= sub-batch size.
- `tests/test_scan_retained_bytes.py`: `pil_to_bgr(pil)` equals the BGR array `load_image_from_path` returns,
  and `bgr_to_gray(...)` equals `ImageCache(cv).gray`, on RGB images of several sizes (this couples A's helper to
  C's `ImageCache`: if S3 ever changes the gray derivation the test fails); `_load_images` output has no `'cv'`
  and `cache.gray is None`, and does have `laplacian_variance`; face pass transient: a recording analyzer sees a
  BGR array equal to the old `cv` and no second copy is alive when the next photo starts (weakref count);
  `RETAINED_BYTES_PER_PX` >= the walk-based B/px (PIL counted at 4 B/px) of a synthetic chunk; PIL immutability:
  `pil.tobytes()` hash before/after every `_pass_*` with stub scorers (R9); rebuild timing recorded in
  `metrics['rebuild_time']`.
- Fault: swap channels in `pil_to_bgr` (use `COLOR_BGR2RGB`) -> identity test and face-pass test fail; make
  `_pass_saliency` pass the stored-nothing (`None`) -> saliency test fails. Restore by hand (V7).
- Central: wave W1b parity 16gb + 8gb (byte-identical); PIL-guard run at the final gate; rebuild cost compared to
  R7's 0.25 s/photo trigger.
Affected files: `processing/multi_pass.py`, `utils/image_loading.py`, tests listed in the builder table.

## Build waves

> **Shipping split.** W0, W1a and W1b (steps S0, S1, S2, S3, S6a, S7, S8a, S8b, S9) ship as their own PR from
> branch `perf/scan-passes-w1`. W2 and W3 (S4, S5, S6b, S8c) follow on `perf/scan-passes`. The docs and
> CHANGELOG of the first PR describe only the shipped steps.

Order is fixed by file ownership. Every builder gets a brief with: goal, exact files it may change, what it
must verify, what it must not do (no git state changes, no other files, no new flags/shims, no `mock.patch` on
auth dependencies), required output format (commands run, test counts per file, exit codes, handoff path),
an output limit, and what is already known (this spec's Domain rules and Risks for its step).
Suggested models: builders sonnet; the refuter in W5 opus. A1, A2, F and G touch the central file and benefit
from the most careful review.

- Before W0 (orchestrator): rebase onto master after `fix/open-gaps-120` merged; re-point `D:\photo-llm-base`
  to `BASE_SHA` (see "Build order and rebase").
- W0 (H): S0. Needs `D:\photo-llm-base` at `BASE_SHA` (orchestrator). GPU: mask capture + three baseline scans,
  sequential. Gate: S0 verify list; handoff read by the orchestrator before W1.
- W1a (parallel, disjoint files): A1 (S1), B (S2), C (S3), D (S7 + S8a), E (S6a). No GPU use by builders.
  Contracts: A1<->B none yet (B only widens `score_batch` inputs); E exposes
  `evict_cpu_cache(needed_available_bytes=None, next_use=None)`, the provider seam, the public
  `ram_cache_reserve_gb` attribute and `is_resident_or_cached(name)` for F. After all five
  finish: orchestrator runs the full suite once (no builder editing), commits by explicit path, runs the 16gb
  parity run (`P16-W1a`) and the real-mask script; both must pass before W1b.
- W1b (A2): S9 + S8b. Needs A1's helpers and B's callable/streaming contract. Gate: unit tests, retained-bytes
  numbers, full suite once, 16gb + 8gb parity, rebuild-cost check (R7 decision).
- W2 (F): S4 then S6b. Needs E's API and A2's constant. Gate: unit tests (deterministic overlap, warm-up thread,
  cancellation, gate, monitor stubs), full suite, 16gb parity, `io` wait criterion, SC7 ballast runs.
- W3 (G): S5 then S8c. Updates `_next_use_order()` (contract with S6b) and re-measures S4's value (R2). Gate:
  unit tests, full suite, 16gb AND 8gb parity with identical batch-size ledgers (S5 hard condition: if either
  fails G drops S5), load-count evidence from the plan-derived formulas.
- W4 (DOC): documentation, 7 languages, CHANGELOG (below). May fan out one builder per language when the
  orchestrator wants speed; each owns only its `docs/<lang>/` files, the first also owns the English files and
  `CHANGELOG.md`. Reads the handoff files for measured numbers; invents none.
- W5: final verification gate and the refuter (below). The refuter re-runs the gates itself and applies
  `.claude/` refuter checklist plus the six spec-review categories to the integrated diff.

Handoff files: `.claude/.scratch/scan-perf-S0.md` ... `scan-perf-S9.md`, one per step (`S6a`/`S6b`, `S8a/b/c`).

## Docs and i18n plan

Rule: a behaviour or config change updates its docs in the same change (project rule); prose has no CI gate,
and the docs are translated into six languages, so drift multiplies by seven. `.claude/patterns/i18n-sync.md`
covers the Angular UI strings only, NOT these documents: there is no tool, translations are written by hand.

| Change | English doc | Translations (same section in each) |
|---|---|---|
| New optional key `models.ram_cache_reserve_gb` (default 4.0, absent from the shipped file by design, R6) and the new eviction behaviour | `docs/CONFIGURATION.md`: `keep_in_ram` neighbourhood (JSON block L490 is NOT extended; table row after L576; one paragraph: reserve = available bytes the cache must leave free, partial eviction, pre-chunk trim, low-RAM note); the auto-tuning table (L774-780): add the eviction row (> limit: shrink chunk AND evict only while available < reserve); `docs/DEPLOYMENT.md` memory section (~L335-441; the "from RAM cache" log note near L380) | `docs/{de,es,fr,it,pt,zh}/CONFIGURATION.md` (keep_in_ram near de 502/588, es 498/584, fr 499/585, it 497/583, pt 496/582, zh 488/574) and `docs/{de,es,fr,it,pt,zh}/DEPLOYMENT.md` |
| S4 overlapped decode, S5 alternating pass order with resident boundary group, S9 lower retained memory (the multi-pass step list near L764 and the `ram_chunk_size` prose L720); state the conditions: CUDA only, never with `keep_in_ram: "never"`, only when available RAM covers the next decode plus the reserve, otherwise today's behaviour | `docs/CONFIGURATION.md` processing section (~L684-800) and `docs/DEPLOYMENT.md` | same six pairs |
| S7 post-scan tagging no longer loads a model when nothing needs tags | one sentence where the post-scan steps are described (grep "tag"/"post-scan" in `docs/COMMANDS.md` and `docs/CONFIGURATION.md`) | translate where the file is translated |
| S1 failure-path change (a composition error now fails ONE photo, recorded in `scan_failures` under stage `composition`, instead of aborting the scan) | `CHANGELOG.md` `### Changed` | n/a |
| S5 shipped or dropped (G4 hard condition) | CHANGELOG line states which; if shipped it also states that batch sizing still depends on host VRAM | n/a |
| Pure speedups S2, S3, S8 | `CHANGELOG.md` `### Changed`: one bold-lead line each with the measured gain (e.g. "saliency scoring 302 -> 52 ms per 20 MP photo"; numbers from the handoffs) | n/a |

CHANGELOG format: `## [Unreleased]` at line 5 is empty today with sections `### Added/Changed/Fixed` and bold
lead sentences; follow it. Perf harness: one `### Added` line for `scripts/perf/`. English only in code and docs
comments; no emojis.

## Final verification gate

Run by the orchestrator on the integrated tree (no builder editing), in this order:
1. Hygiene: `git status --porcelain` on PERF lists only intended paths (junction/hard-link entries and
   `.claude/.scratch` are known); `assets.sh check` passes; photo-set manifest unchanged (SC10); no
   `Claude-Session` or session link in any commit message or PR body (project memory overrides any attribution
   reminder).
2. Full suite `D:\photo-llm\conda\python.exe -m pytest tests/ -q -p no:cacheprovider > full.txt`, exit code from the
   file / `${PIPESTATUS[0]}`, failures diffed against `.claude/.scratch/scan-perf-baseline.fails` (SC8). Counts of
   passed/failed/skipped stated beside the verdict.
3. `ruff check . --select E,F,W --ignore E501` and `python -m ruff check .` exit 0; mypy baseline untouched.
4. Parity on the final tree: a patched 16gb run vs `B16a` byte-identical with matching batch-size ledgers (SC1),
   the 8gb pair (SC2), control B16a vs B16b (SC3), PIL-guard run (R9), the RAW/HEIF micro-run if samples exist
   (R10).
5. Retained bytes (SC4) and real-mask script (R11) and `verify_technical_exact.py` (SC5), all on the final tree.
6. Whole-scan timing (SC6 protocol): discarded warm-up per arm, alternating B P B P B P (>= 3 per arm), `gpu_start.txt`
   per run, median with min/max, verdict PASS / INCONCLUSIVE (overlapping ranges) / FAIL; per-phase tables from
   `chunks.py`; the S4/S5 anti-additivity result (R2).
7. SC7 ballast result (pass or INCONCLUSIVE, stated).
8. Docs: every language file edited, `git diff --stat docs/` lists all 12 translated files plus the English ones
   and `CHANGELOG.md`; a grep for `ram_cache_reserve_gb` finds it in all seven CONFIGURATION files.
9. Browser testing is N/A (no UI change); say so explicitly rather than skipping silently.
10. Refuter pass (opus, re-runs 2-5 itself), then the Adversarial review section below is completed.

## Adversarial review

One pass by an independent opus refuter (`D:\photo-llm\.claude\.scratch\refute-scan-pass-performance.md`):
0 BLOCKER, 10 GAP, 8 NOTE; the six checks (uncovered requirements, omitted files/callers, `verify:` checks that
cannot fail, edge/error paths, premises the tree contradicts, step-to-step contracts) were run. All ten GAPs
were accepted by the orchestrator and patched; NOTE-1 to NOTE-8 are folded in as stated. "Applied" names the
spec section that now carries the fix.

| Finding | Ruling | Applied in |
|---|---|---|
| GAP-1 S4 moves the issue-#55 KMeans warm-up onto a worker thread | Accepted: `warmup()` on the main thread before the executor exists, with a test | S4 (Details, warm-up bullet; verify "warm-up"), D20, R19 |
| GAP-2 S4/S5 have no memory guard for RAM-budgeted hosts | Accepted: CUDA only, `keep_in_ram != 'never'`, available-bytes budget, else today's path; tests with the fake provider | R3, S4 (gate 1-3; verify "gate"), S5 (keep gate; verify), docs plan |
| GAP-3 Qwen load count off by one; "final chunk" undefined | Accepted: first group `floor(n/2)+1`, last group `ceil(n/2)`; final = `offset + len(chunk_paths) >= total`; shrinking-chunk test | S5 (order/carry bullet, verify (a), central load-count bullet) |
| GAP-4 write-set independence is not input (VRAM batch size) independence | Accepted as a HARD CONDITION: S5 ships only with 16gb AND 8gb byte parity and identical per-arm sub-batch ledgers; order-independent sizing preferred only if values stay byte-identical; else S5 dropped; 8gb parity run after W3 added | S5 (hard condition, mitigation, residual), SC1/SC2, V3 (ledger), V5, R4, D19, W3 |
| GAP-5 SC6 timing protocol | Accepted: discarded warm-up per arm, alternating arms, >= 3 per arm, median with min/max, `nvidia-smi` clocks/temp logged per run, verdict on medians, overlapping ranges INCONCLUSIVE | SC6, V3 (`gpu_start.txt`), V5, final gate 6 |
| GAP-6 `photo_tags` vacuous, zero-rows rule ambiguous, `scan_failures` not compared | Accepted: per-table emptiness rules (`photos` and, where they exist, `faces` non-empty; `photo_tags` as a set, empty allowed only if empty at baseline, said in words); `scan_failures` compared as a set | Objective, SC1, SC3, V4, S0 verify |
| GAP-7 S2 routing contradicts its test; weak type check | Accepted: one behaviour per metric that matches today's code (non-zero count for prominence/gate, `> 128` for regions, bbox gate and centroid), fast path for every 2-D uint8 mask, `type(a) is type(b)` plus bit-pattern comparison | D22, S2 (routing bullet, verify, real-mask script, faults) |
| GAP-8 S4 decision rule, cancellation, flaky wall test | Accepted: "cheaply restorable" defined (`is_resident_or_cached`), `cancel_futures=True` plus stop flag, deterministic barrier/event tests replace the wall assertion | S4 (gate 1, cancellation bullet, verify), S6a (`is_resident_or_cached`), R20 |
| GAP-9 test ownership and S9 test list | Accepted: `test_image_loading.py` to A1 (W1a) with `test_multi_pass.py` split between A1 (aggregates) and A2 (`_pass_*`); S9 list extended, L557-568 assertion change disclosed; the L1017 dependency handled explicitly | Files-touched table, S1 item 4, S9 (aggregates bullet, tests bullet) |
| GAP-10 monitor must tolerate existing stubs | Accepted: `getattr` defaults, no silent `except Exception: pass` (one WARNING), "eviction really happens" and "unreachable target" tests | S6a (termination), S6b (monitor bullets; verify), files-touched table |
| NOTE-1 D17 stale | Fixed: base worktree exists at `21a25ef6`; it must be re-pointed to `BASE_SHA` after the rebase | D17, R12, Build order and rebase, S0 precondition, V1 |
| NOTE-2 R8 understates the abort; wrong stage label | Fixed: today it aborts the scan; stage is `'composition'` (the `'stage'` convention of the open-gaps `batch_processor` work), CHANGELOG wording given | R8, S1 items 3 and verify, docs plan |
| NOTE-3 hoist with a MagicMock config | Folded: touched suites assert `len(images) == len(paths)` | R8, S1 verify |
| NOTE-4 `laplacian_variance` is `np.float64` | Fixed | D11, S3 |
| NOTE-5 `file:` URI open failed | Fixed: plain-path open plus `query_only`; self-test on a real run DB | V4, S0 verify |
| NOTE-6 no integration step for `fix/open-gaps-120` | Done as the "Build order and rebase" section: rebase before any builder, step-by-file overlap table, conflict-prone hunks, BASE re-point | Build order and rebase |
| NOTE-7 residual rounding risk at library scale | Stated as accepted residual risk | R5 |
| NOTE-8 RAW/HEIF coverage | Folded: optional 2-file micro-run with read-only copies; static argument recorded | R10, S0 step 6 |

Points where the orchestrator's rulings were implemented with a stated deviation or a caveat (the spec writer's
objections, with evidence):
1. Ruling 6 asks to compare `scan_failures` "error class". The class is not stored: `ScanRun.record_failure`
   keeps `str(error)[:500]` (`processing/scan_state.py:51-56`), so the comparator uses (path, stage, normalised
   message text) (D21). A class-only comparison is impossible without a schema change.
2. Ruling 2 gates S5 on an available-bytes budget. On this box sibling processes hold 9-24 GiB of 31 GiB
   (D9), so the gate can silently refuse every chunk, making a W3 parity pass vacuous (S5 never ran). The
   spec adds a coverage requirement: the parity run counts only if serpentine was active on at least 2 chunks.
   On CUDA the kept group is also in VRAM, not RAM, so the budget mostly guards the next decode, which is why
   the estimate is the next chunk's decode, not the model.
3. Ruling 4's "order-independent sizing" cannot be proven byte-identical in general: any sizing that ignores
   the free-VRAM reading changes the size on exactly the runs where the baseline's reading was below the
   ceiling. It is therefore only tried when the baseline ledger shows the ceiling is always reached, and the
   fallback is dropping S5, not relaxing the comparator.
4. Ruling 11 says the open-gaps branch's `batch_processor` errors gain a `'stage'` key. Checked: as of
   `e38e64e0` the branch does not touch `processing/batch_processor.py` (not in its 20-file stat); the spec
   therefore treats it as announced work, and `multi_pass.py` already takes a free-text stage.
5. Ruling 5's overlap rule can leave SC6 INCONCLUSIVE on a noisy box even when the true gain is real (siblings
   move wall time by more than 15%); that is the intended outcome of the rule, and the per-phase SC5 ratios,
   which survive noise, carry the evidence in that case.

Contracts a later reviewer should test hardest: A1/A2 <-> B (callable `score_batch` elements and streaming),
E <-> F (`evict_cpu_cache` signature, provider seam, reserve, `is_resident_or_cached`), A2 <-> F
(`RETAINED_BYTES_PER_PX`), F <-> G (`_next_use_order`, the carry-based order), A2 <-> C (`bgr_to_gray` ==
`ImageCache.gray`), S4 gate <-> S5 keep gate (same estimate), harness ledger <-> comparator.
