# Spec: Fix the three #184 follow-ups (PyIQA silent 5.0, tag_existing fp32 SigLIP, planner CUDA margin)

Branch `fix/184-followups` (= master b877cbf8, PR #192 merged). Evidence base:
`.claude/.scratch/followups-184.md` (not redone) plus the re-reads cited below. Interpreter for
every command: `D:\photo-llm\conda\python.exe` (no venv on this box). Search with `git grep`, not
ripgrep. Run no interactive python: scratch scripts are files in the scratchpad, run by path.
No schema change, no API change, no new column, no config key.

The research report recommended AGAINST two of the locked decisions (it kept `tag_existing.py` at
fp32, and called the margin change "optional hygiene"). The decisions below are the user's and
override it; the report's measurements are used, its recommendations are not.

## Objective + success criteria

- S1. No PyIQA scorer stores a neutral 5.0 for an image it failed to score. `PyIQAScorer.score_batch`
  returns `list[Optional[float]]`; an image that fails (CUDA OOM that survives halving to 1,
  `topiq_nr_face` "No face detected", LIQE "Short side is less than 224", any other exception)
  yields `None`. Every writer leaves the column unset or NULL: never 5.0, never a crash on `round(None)`.
- S2. A CUDA OOM in a PyIQA forward is recovered like saliency: leave the `except` block before
  retrying, `clear_device_cache`, halve the forward batch down to 1, an image that still fails alone
  fails alone. Non-OOM errors keep the per-image fallback. Proven by a fake model with a call-size
  log (`[4, 2, 2]`) and by a clear-cache stub that asserts it runs outside any `except` block.
- S3. `topiq_nr_face` is not batchable (pyiqa asserts batch size 1); it is scored serially, one
  model call per image, never a call with batch > 1.
- S4. Unexpected scoring failures (OOM, anything not a known "expected skip") are reported at ERROR
  at the end of every `score_batch` call, with the count; they no longer wait for `unload()`, which a
  RAM-cached model never reaches. Expected skips stay counted and are totalled at `unload()` as today.
- S5. A required (primary) PyIQA model, i.e. TOPIQ on `16gb`/`24gb`, that fails for a photo sets
  `SCAN_FAILED_KEY` for that photo: it is recorded in `scan_failures`, excluded from the save, and
  retried by the next scan / `--retry-failed`. It no longer stores `aesthetic = quality_score = topiq_score = 5.0`
  and a complete scan. A chunk-wide exception from a primary model propagates to `_process_chunk`'s
  existing "Required ... inference failed, marking retryable" branch instead of being swallowed.
- S6. `aesthetic_v25` (same `_pass_pyiqa` writer) returns `None` instead of 5.0 on a per-image failure.
- S7. `--recompute-iqa` and `--score-topiq` skip a `None` score (no UPDATE, not counted), so a rerun
  retries it (`WHERE col IS NULL`). A multi-pass scan now leaves a faceless photo's `face_quality_iqa`
  NULL, matching `--recompute-iqa` (which already skips `face_count = 0`).
- S8. `tag_existing.build_clip_tagger` loads a transformers SigLIP model through
  `utils.siglip.load_siglip_model` (fp16 on CUDA, fp32 elsewhere); the open_clip branch is untouched.
  `api/routers/search.py` stays fp32 with a comment saying so on purpose.
  `tests/test_siglip_loader.py::test_all_siglip_sites_use_helper` covers the whole repo with the
  allow-list `{utils/siglip.py, api/routers/search.py}` and fails if either `tag_existing.py` or a new
  site loads `AutoModel.from_pretrained` directly.
- S9. `ModelManager._CUDA_PASS_MARGIN_GB = 1.3` replaces the inline `1.0` in `group_passes_by_vram`;
  the planner at 11.7 GiB plans 4 passes (insightface alone) for the `16gb` profile, at >= ~11.8 GiB 3 passes.
- S10. The "Dynamic Pass Grouping" table in `docs/CONFIGURATION.md` and its six translations is
  regenerated from the real planner output for the four shipped profiles at realistic totals; the
  stale `8GB` and `24GB+` rows are fixed; every place that quotes "three passes" for 12 GB is updated.
- S11. Behaviour changes are documented in the same change (EN + de/es/fr/it/pt/zh) and CHANGELOG
  `## [Unreleased]`: NULL instead of 5.0, the gallery range-filter effect, the retag change, the margin.
- S12. Every new test fails on the pre-change code for the reason it names (run it red first), builds
  its schema with `db.schema.init_database` / `tests/conftest.py` seeders, and is safe in CI's Python job
  (no torch / transformers / pyiqa): torch-needing tests `importorskip`; the writer-level contract
  tests live in files that need no torch so CI actually runs them.

## Files to create / change

Change:
- `models/pyiqa_scorer.py` (return type, OOM retry, serial `topiq_nr_face`, failure logging)
- `models/aesthetic_v25_scorer.py` (5.0 -> `None`, annotation)
- `processing/multi_pass.py` (`_pass_pyiqa`)
- `processing/scorer.py` (`_run_iqa_pass._flush`)
- `facet.py` (hoist `_flush_topiq_batch` to module level beside `_flush_saliency_batch`, skip `None`)
- `tag_existing.py` (`build_clip_tagger`), `utils/siglip.py` (lazy `import torch` on the CUDA branch only; docstring),
  `api/routers/search.py` (comment only)
- `models/model_manager.py` (`_CUDA_PASS_MARGIN_GB`, comment)
- `tests/test_pyiqa_failure.py` (new, needs torch), `tests/test_multi_pass_pyiqa.py`,
  `tests/test_siglip_loader.py`, `tests/test_tagging.py`, `tests/test_model_manager.py`,
  plus tests for the two flush sites (new file `tests/test_iqa_flush_none.py`)
- `docs/CONFIGURATION.md`, `docs/SCORING.md`, `docs/COMMANDS.md`, `docs/VIEWER.md` and the same four
  files under `docs/{de,es,fr,it,pt,zh}/`; `CHANGELOG.md`

Do not touch: `database.py`/`db/schema.py`, `api/models/*`, the client, `scoring_config*.json`,
`processing/scorer.py:IQA_PASS_VRAM_MARGIN_GB` (a different, `--recompute-iqa` only margin).

### Reader / writer inventory (every site that touches the affected columns)

Columns: `aesthetic_iaa`, `face_quality_iqa`, `liqe_score`, `aesthetic_v25` (new NULLs); `aesthetic`,
`quality_score`, `topiq_score`, `scoring_model` (primary: no new NULLs, the photo is not saved instead);
`deqa_score` (already None-returning). Verified by `git grep` this session; none of the reader
sites needs a code change.

Writers

| # | Site | Today | After |
|---|------|-------|-------|
| W1 | `models/pyiqa_scorer.py:score_batch` / `_score_batch_serial` | 5.0 on any exception (:337, :392) | `None` (S1) |
| W2 | `models/aesthetic_v25_scorer.py:score_batch` (:~240) | 5.0 per failed image; only caller is `_pass_pyiqa`; DeQA (`deqa_scorer.py:300-318`) already returns `None` and is the in-tree precedent | `None` (S6) |
| W3 | `processing/multi_pass.py:_pass_pyiqa` (:949-975) | writes whatever it gets; swallows every exception | skip `None`; primary -> `SCAN_FAILED_KEY` + `on_error`; primary exception re-raised (S5) |
| W4 | `multi_pass.py:_save_results` (:1208-1216) | `data.get(col)`: absent key = NULL | no change (NULL-in, NULL-out) |
| W5 | `multi_pass.py:_save_results_restricted` (:1255-1295) | absent key kept, present `None` key written as NULL | no change. The fix is to leave the key ABSENT; a stored good value then survives. Same contract as qrealign |
| W6 | `multi_pass.py:_pass_insightface` (:824) | writes `face_quality_iqa` only when the analyzer reports one (OFIQ, commercial) | no change. `face_quality_iqa` is in `FACE_PASS_COLUMNS`, so a failed face pass already skips it in a restricted save |
| W7 | `processing/scorer.py:_run_iqa_pass._flush` (:2335-2345) | `round(score, 2)` on every score; counter adds `len(scores)` | skip `None`, count only written rows (S7). qrealign native never returns `None`: unaffected |
| W8 | `facet.py:_flush_topiq_batch` (:2893-2901) | `round(score, 2)` on every score; `round(None)` would raise | hoist + skip `None` (S7) |
| W9 | `processing/scorer.py:2530,2553,2574`, `batch_processor.py:415,478-481`, `scorer.py:1500,1568,1574` single-pass save | `None` / `face_res.get` | no change: single-pass never runs these PyIQA models |
| W10 | `multi_pass.py:_pass_qrealign` | unset key + `SCAN_FAILED_KEY` when primary | no change; it is the pattern W3 copies |

Readers (all NULL-safe already; the aggregate scores a NULL as neutral 5.0 exactly as a stored 5.0 did, so aggregates, categories and rankings are numerically unchanged)

| # | Site | NULL handling |
|---|------|---------------|
| R1 | `processing/scorer.py:577-579` `build_metric_vector` (`safe_float(m.get(..), 5.0)`), also the recompute-average SELECT (:1636) | NULL -> 5.0, identical to the old stored value. Pin with a test |
| R2 | `scorer.py:624-641` `unavailable_for_config` / `renormalise_weights` | library-wide mechanism only; per-photo NULL deliberately does NOT renormalise. Do not extend it |
| R3 | `config/category_filter.py` (:44-52, :80-82) | these columns are weight keys only, not filter fields |
| R4 | `api/top_picks.py`, `config/percentile_normalizer.py` | do not reference the columns |
| R5 | `api/routers/gallery.py:488-492` range filters (`min_/max_face_quality_iqa`, `min_liqe`, `min_aesthetic_v25`...), `albums.py:1176-1179` | SQL comparison: a NULL row matches neither bound. **Visible change**: `min_face_quality_iqa=0` and `min_liqe=0` no longer return photos without a score (they used to return the stored 5.0). Document (Step 8), pin with a test |
| R6 | `api/routers/filter_options.py:540-588` sparkline/range | `WHERE col IS NOT NULL`: the 5.0 spike of faceless photos in the `face_quality_iqa` histogram disappears. Improvement |
| R7 | `api/routers/stats.py:72-73,876-893` | `IS NOT NULL` filters; category-metrics maps NULL -> 0 for display (pre-existing, commercial mode does the same) |
| R8 | `api/routers/comparison.py:71,667,899`, `critique.py:57-58,89-90,392` | metric dicts; NULL already occurs in commercial mode and pre-saliency rows. Confirm the critique builder skips `None` (read in Step 4; add a test only if it does not) |
| R9 | `api/models/gallery.py:94-97`, `api/db_helpers.py:406-407` | `Optional[float] = None`: NULL is a legal wire value; no `gen:api` diff |
| R10 | `optimization/insights_miner.py:26,112-134` NULL -> NaN, masked; `optimization/iqa_eval.py:26,84` `IS NOT NULL`; weight optimizer / ranker via `build_metric_vector` | NULL-safe |
| R11 | `calibrate.py:59,944-960` `float(r.get(col) or 5.0)` | NULL -> 5.0, same as before |
| R12 | `facet.py:3115-3145` distortion validation | `liqe_vals` NULL -> NaN, masked in `_spearman` |
| R13 | `build-tools/verify_scan_db.py:22-32,50-60` | `CORE` requires only `filled > 0`, not every row: a few NULL `liqe_score` rows do not fail it; `face_quality_iqa` is informational. No change |
| R14 | `topiq_score` readers (`api/models`, gallery `--min_topiq`, `scorer.py:1568` comment) | `--score-topiq` skipping a failure leaves the stored value or NULL; NULL `topiq_score` already exists on every non-TOPIQ profile |

## Open questions / risks

- Pre-existing libraries keep their stored 5.0 values (faceless photos' `face_quality_iqa`, short-side
  `liqe_score`). Only a new scan or a rescan of the photo writes NULL. No migration is added (not in the
  locked scope); the docs say so plainly. A later one-off `UPDATE ... SET face_quality_iqa = NULL WHERE face_count = 0`
  would be a separate decision: it cannot tell a genuine 5.0 from a stored fallback for photos with faces.
- `--recompute-iqa` re-attempts (and re-fails) a photo every run when its column stays NULL for a reason
  other than "no face" (LIQE short side; `face_count > 0` but pyiqa's own face detector finds none). Bounded
  cost, one thumbnail forward each; accepted, stated in the COMMANDS.md sentence.
- Expected-skip classification is by message substring (`No face detected`, `Short side is less than`),
  taken from pyiqa 0.1.x text (`topiq_arch.py:605`, LIQE). A future pyiqa that rewords them is treated as an
  unexpected failure: louder (ERROR), never silent. The safe direction.
- Expected skips are still totalled only at `unload()` (deliberate: a landscape-heavy scan skips ~92 % of
  `topiq_nr_face` images, a per-chunk WARNING was already judged noise in the existing comment at
  `pyiqa_scorer.py:~148`). What moves to "pass end" is the part that matters, failures. If the orchestrator
  wants expected skips at pass end too, make `_pass_pyiqa` log a DEBUG line from `scores.count(None)`.
  A cached model that is evicted (`evict_cpu_cache`) still loses the expected-skip total. Accepted.
- The forward batch, once halved by an OOM, stays halved for that scorer instance (reset in `load()` /
  `unload()`). A RAM-cached scorer therefore keeps the smaller batch for the rest of the scan. Throughput
  cost at batch 1 vs 4 is small for IQA models; the alternative (retry from 4 per slice) would OOM and flush
  the allocator up to 32 times per 128-image chunk. If this is unwanted, drop the attribute and pass the
  size down the recursion (the saliency shape).
- Not done (unmeasured, not in the locked decision): lazy per-shape-group preprocessing. `score_batch` still
  moves the whole chunk (up to 128 images) onto the device before any forward; the research could not
  verify the magnitude. A preprocess-time OOM is handled (falls back to serial outside the handler) but the
  residency is unchanged.
- The "~1.3 GiB measured residue" behind the margin is the user's measurement. The research could not
  reproduce a between-pass NVML reading (flagged UNVERIFIED there), and the only measured run (peak 9.06 GiB
  of an 11.7 GiB cap) had headroom at 1.0. Step 7's capped re-run records the between-pass residue so the
  constant's comment is backed by a number, not only by the decision.
- 12 GB cards straddle the new flip point. A planner total under ~11.8 GiB (an RTX 3060 12 GB reports about
  11.76 GiB total in widely quoted OOM messages: NOT verifiable on this 16 GB box) plans 4 passes; 4070-class
  12 GB cards (~11.99 GiB) keep 3. The table therefore shows the 11.7 plan (it matches the measured run and
  the conservative cards) and a one-line footnote for >= 11.8. The exact flip is produced by the sweep in Step 7,
  not by hand.
- The documented 12 GB measurement (per-pass peaks ~8.5 / ~9.1 / ~3.9 GiB) belongs to the 3-pass plan. With 4
  passes those per-pass numbers are stale. Step 7 re-measures; if the harness cannot run, the sentence is
  rewritten to say it was measured with the previous three-pass grouping, never left implying it describes
  the new one.
- CI scope: `tests/test_pyiqa_failure.py` skips in CI (no torch). The writer-level contract (Steps 2 and 3) is
  tested in torch-free files so CI exercises it.
- Windows line endings: translated docs have been rewritten whole-file by tools before. After each doc edit
  `git diff --stat` must show only the touched hunks.

## Steps

Each step: write its tests first and run them red on the unmodified code, then implement, then green.
Run tests as `D:\photo-llm\conda\python.exe -m pytest <files> -q > <scratch>\out.txt 2>&1` and read the file
(a pipe's exit code is not the suite's).

1. **PyIQAScorer failure contract, OOM recovery, serial `topiq_nr_face`, `aesthetic_v25`.**
   - description: In `models/pyiqa_scorer.py`:
     (a) `score_image` unchanged (still raises). `score_batch` and `_score_batch_serial` return
     `list[Optional[float]]`. A private `_score_one(image)` wraps `score_image`: on an exception it
     captures `oom = is_out_of_memory_error(e)` and the message, **leaves the handler**, runs
     `clear_device_cache(self.device)` if OOM, records the skip and returns `None`.
     (b) Remove `'topiq_nr_face'` from `_BATCHABLE_MODELS`; fix the comment above the set.
     (c) Batched path: an instance attribute `self.forward_batch = self.MAX_FORWARD_BATCH` (the class
     constant stays the ceiling and `tests/test_pyiqa_batching.py` keeps passing unchanged). For each
     same-shape group, slice by `self.forward_batch` and run a recursive helper in the shape of
     `saliency_scorer._masks_with_oom_retry` (:295-319): try the stacked forward; on exception record
     `oom` and the message and leave the handler; if OOM -> `clear_device_cache`, and if more than one
     image: halve, set `self.forward_batch` to the half, recurse on both halves; if one image: result
     `None`, skip key `'out of memory'`. If not OOM -> per-image fallback through `_score_one` (today's
     behaviour, minus the 5.0). The first extraction error (`_extract_batch_raw` `ValueError`) is a non-OOM
     error: unchanged fallback.
     (d) The up-front `tensors = [self._preprocess_image(...)]` try/except: do the serial fallback after
     leaving the handler (and `clear_device_cache` if the failure was an OOM) so the partial tensors are
     not pinned by the traceback.
     (e) Skip bookkeeping: normalise every OOM message to one key (`'out of memory'`, the raw text embeds
     allocator sizes that differ per run and would defeat the counting). `_record_skips` keeps counting
     into `_skipped_totals`. At the end of every `score_batch` call, every key that is not an expected skip
     (module constant `_EXPECTED_SKIPS = ('No face detected', 'Short side is less than')`, substring
     match) is logged once with `logger.error("  %s failed for %d image(s) (%s)", ...)`; expected skips
     stay DEBUG. A successful halving logs one WARNING per call, `"  %s forward batch reduced to %d after out-of-memory"`,
     WITHOUT the exception text. `unload()` keeps its totals report and also resets `forward_batch`.
     (f) `models/aesthetic_v25_scorer.py:~240`: `scores.append(5.0)` -> `None`; annotation
     `list[Optional[float]]`; docstring says a failed image yields `None`. No OOM retry there (deprecated,
     AGPL, off by default; out of scope).
   - key decision: how far the OOM handling goes. Default: exactly the saliency precedent (leave the
     handler, clear, halve to 1, lone failure is `None`), with the halved size sticky per instance. Why:
     it is the locked decision and the proven shape; sticky avoids repeated OOM + cache flushes per
     slice. Not doing lazy preprocessing (Open questions).
   - key decision: is `aesthetic_v25` in scope. Default: yes, the 5.0 -> `None` line only. Evidence: its
     only caller is `_pass_pyiqa` (`PYIQA_COLUMN_MAP['aesthetic_v25']`), so it is the same writer into a
     nullable REAL column that the same readers handle (R1-R12), and DeQA already sets the precedent. It
     gets no OOM machinery because it is a separate deprecated scorer.
   - verify: new `tests/test_pyiqa_failure.py`, `pytest.importorskip("torch")` at the top, scorers built
     like `tests/test_pyiqa_batching.py` (`PyIQAScorer(name, device='cpu')`, stub model, `_loaded = True`).
     Red-first cases:
     (1) `topiq_nr_face` with a stub that raises `AssertionError('No face detected in the input image.')`
         for chosen images and returns a score otherwise: result has `None` for those and floats for the
         rest, never 5.0; `supports_batching is False`; the stub's recorded batch sizes are all 1 (today it
         is called with batch > 1 first).
     (2) LIQE stub raising `'Short side is less than 224, try upsampling the original image'` -> `None`.
     (3) OOM halving: a 4-image same-shape group, stub that raises `RuntimeError("CUDA out of memory ...")`
         (recognised by message) above N images; assert call sizes `[4, 2, 2]`, all four scores are floats,
         order preserved. Repeat once with `torch.cuda.OutOfMemoryError` when `getattr(torch.cuda,
         'OutOfMemoryError', None)` exists.
     (4) Clear-outside-handler: monkeypatch `utils.device.clear_device_cache` with a stub that records
         `sys.exc_info()[1] is None`; assert it was called and every call was outside an exception. Red on
         any implementation that clears inside `except`.
     (5) OOM at every size -> all `None`, an ERROR record containing the count, no exception escapes.
     (6) Mixed: stub OOMs on any batch > 1 and on image index 2 even alone -> `[float, float, None, float]`.
     (7) Non-OOM error in a group falls back per image: call sizes `[4, 1, 1, 1, 1]`, scores floats, no
         halving, no `clear_device_cache` call.
     (8) Sticky: after (3)'s OOM the next group's first forward is 2; `unload()` resets it to 4.
     (9) Logging: an unexpected failure logs ERROR; an expected "No face detected" skip logs nothing above
         DEBUG during the call, and `unload()` still reports the total.
     (10) Non-batchable (`liqe`) serial OOM: image -> `None`, cache cleared outside the handler, the rest score.
     (11) `aesthetic_v25`: build with `AestheticV25Scorer.__new__`, `_loaded = True`, a `processor` that
          raises (forces the fallback) and a patched `score_image` that raises for one image -> that slot is
          `None`, not 5.0 (skip if importing the module needs a missing package).
     Existing `tests/test_pyiqa_batching.py` stays green unmodified (class constant 4, `[4, 4, 2]`).
   - affected files: models/pyiqa_scorer.py, models/aesthetic_v25_scorer.py, tests/test_pyiqa_failure.py

2. **`_pass_pyiqa`: skip `None`; a failed primary model marks the photo failed.**
   - description: In `processing/multi_pass.py:_pass_pyiqa`: iterate `enumerate(paths)`; a `None` score
     never writes a key. Supplementary (`model_name in PYIQA_COLUMN_MAP`): do nothing for that photo.
     Primary (`not in PYIQA_COLUMN_MAP`): do not write `aesthetic`, `scoring_model`, `quality_score`,
     `topiq_score` for it; set `results[path][SCAN_FAILED_KEY] = model_name` and call
     `getattr(self, 'on_error', None)(path, model_name, f"required model pass failed: {model_name}: no score")`,
     copying `_pass_qrealign`'s wording and its `getattr` guard (the existing test builds the processor with
     `__new__`). The outer `except`: keep `licensing.reraise_if_refused()`; then `if model_name not in
     self.PYIQA_COLUMN_MAP: raise` so `_process_chunk` (:~463-470) records the chunk retryable
     ("Required %s inference failed for chunk, marking retryable"); supplementary keeps the ERROR log.
     Update the docstring.
   - key decision: per-photo marking vs failing the whole chunk. Default: per-photo `SCAN_FAILED_KEY` for a
     `None`, whole-chunk re-raise only for an exception from `score_batch` itself. Why: the scorer now
     catches every per-image error, so a `None` is photo-specific and must not force the chunk's good photos
     to be rescanned; an exception that escapes `score_batch` is chunk-wide and the existing machinery
     already handles it. Interaction checked: `_iter_saveable` and `_compute_aggregates` already skip
     `SCAN_FAILED_KEY` rows, so a failed photo is not saved (a rescanned existing photo keeps its old row;
     a new photo gets no row and is picked up by the next scan or `--retry-failed`).
   - verify: extend `tests/test_multi_pass_pyiqa.py` (torch-free; its `_run_pass` helper already bypasses
     `__init__`). Red-first today:
     (a) `_run_pass("topiq_nr_face", [None, 8.0])` -> `p0 == {}`, `p1 == {"face_quality_iqa": 8.0}` (today
         `p0 == {"face_quality_iqa": None}`); same for `liqe` and `topiq_iaa`.
     (b) `_run_pass("topiq", [None, 6.0])`: p0 has `SCAN_FAILED_KEY == "topiq"` and none of `aesthetic`,
         `quality_score`, `topiq_score`, `scoring_model`; p1 is complete; the `on_error` callback received
         `(p0 path, "topiq", ...)` exactly once (set `proc.on_error`).
     (c) a stub whose `score_batch` raises: supplementary `liqe` is swallowed (no raise, results untouched);
         primary `topiq` raises out of `_pass_pyiqa`.
     (d) With `tests/test_single_pass_preserves.py` helpers (`_sp.seeded`, `_processor`, `_persist`,
         `_images`, imported not copied, as `tests/test_saliency_failure.py` does; `importorskip("imagehash")`,
         `importorskip("cv2")`): a restricted save of a `None`-yielding supplementary pass over the seeded
         row (`aesthetic_iaa 6.0`, `face_quality_iqa 7.0`, `liqe_score 3.7`) KEEPS all three stored values
         (red today: it writes NULL); a restricted save of a primary-failed photo leaves `aesthetic 8.2`,
         `quality_score 8.0`, `topiq_score 8.2` untouched; a full save (`restricted=False`) of a result with
         the keys absent stores NULL for the three supplementary columns, not 5.0.
     (e) Aggregate pin (passes today): `build_metric_vector` with the three columns `None` equals the vector
         with them `5.0` (documents that aggregates do not move).
   - affected files: processing/multi_pass.py, tests/test_multi_pass_pyiqa.py

3. **The two recompute writers skip `None`.**
   - description: `processing/scorer.py:_run_iqa_pass._flush` (:2335-2345): `if score is None: continue`
     before the UPDATE; `counters['updated'][column]` counts rows actually written. `facet.py`: move
     `_flush_topiq_batch` out of the `if args.score_topiq:` block to module level next to
     `_flush_saliency_batch` (:~251), `if score is None: continue`, return the written count. The
     in-block call sites (:2926, :2931) are unchanged. Update the `--score-topiq` closing log only if its
     wording becomes wrong.
   - key decision: hoist vs test through the CLI. Default: hoist, as PR #192 did for `_flush_saliency_batch`;
     the function has no closure state besides its four parameters.
   - verify: new torch-free `tests/test_iqa_flush_none.py`:
     (a) `_flush_topiq_batch(conn, fake, paths, imgs)` with `fake.score_batch -> [6.0, None, 7.0]` on a DB
         from `db.schema.init_database` (rows inserted through the shared seeder idiom): rows 0 and 2 get
         `round(x, 2)`, row 1 keeps its prior `topiq_score` (seed one non-NULL, one NULL), returns 2 (red
         today: `TypeError: type NoneType doesn't define __round__`).
     (b) `Facet._run_iqa_pass(['topiq_nr_face'], 8, counters)` via the `Facet.__new__` fixture pattern, with
         `facet._make_iqa_scorer = lambda name: fake` (so no pyiqa), three photos having decodable
         thumbnails (UPDATE a real JPEG BLOB in), NULL columns, `face_count` 1: a `None` score leaves that
         photo's column NULL, writes the others, counts only the written, and a second run selects the
         NULL photo again (`WHERE col IS NULL`). Faceless `face_count = 0` photos are still never selected
         for `face_quality_iqa` (existing behaviour pinned).
   - affected files: processing/scorer.py, facet.py, tests/test_iqa_flush_none.py

4. **Reader audit, no code change unless a check fails.**
   - description: Confirm the inventory rows R1-R14 by reading, and fix only what fails.
   - key decision: which unread sites to open. Default: `api/routers/critique.py` metric loop (:392) and
     the comparison preview (`comparison.py:~899-915`, which uses `metrics.get(k) or 0` where the aggregate
     uses 5.0: a pre-existing inconsistency, left alone, weights default 0, preview-only).
   - verify: `git grep -nE "aesthetic_iaa|face_quality_iqa|liqe_score|aesthetic_v25" -- '*.py' ':!tests'`
     lists no site not already in the inventory; add one pin test in the gallery API suite using the shared
     seeders from `tests/conftest.py` (follow an existing `/api/photos` range-filter test for the idiom):
     two photos, `face_quality_iqa` NULL and 5.0, `min_face_quality_iqa=0` returns only the 5.0 one. This
     pins the documented visible change; it passes today and exists so the change cannot regress unnoticed.
     Only if the critique builder does arithmetic on a `None` metric, fix and test it.
   - affected files: possibly api/routers/critique.py; one test file under tests/

5. **`tag_existing.py` loads SigLIP through the shared fp16 helper; widen the single-site test.**
   - description:
     (a) `tag_existing.build_clip_tagger` transformers branch: `from utils.siglip import load_siglip_model`
     and `model = load_siglip_model(clip_model_name, device)`; drop the direct `AutoModel` import and the
     `.to(device).eval()` (the helper does both). The `open_clip` branch is unchanged (CLIP ViT-L-14,
     `legacy`/`8gb`; its precision is not part of this decision).
     (b) `utils/siglip.py`: move `import torch` inside the `if device == 'cuda':` branch so the CPU/MPS path
     needs no torch (the existing `tests/test_tagging.py` case calls `build_clip_tagger(cfg, device="cpu")`
     with a fake `transformers`, and CI has no torch; without this it would raise `ImportError`). Rewrite the
     docstring sentence "Not covered: `api/routers/search.py` loads fp32 on purpose, and `tag_existing.py`
     does not use this loader" to name only `search.py`.
     (c) `api/routers/search.py:325`: add a comment that this text-only query encoder is deliberately left
     fp32 and off `utils.siglip.load_siglip_model` (it lives for the API process, outside the scan's VRAM
     planner), and must be re-measured against search ranking before it is ever routed through the helper.
     Do not assert a historical reason: the original commit left none and the research could not recover one.
     (d) `tests/test_siglip_loader.py::test_all_siglip_sites_use_helper`: `git grep --untracked -l
     "AutoModel.from_pretrained" -- '*.py' ':(exclude)tests' ':(exclude)conda'`, assert the sorted result
     equals `['api/routers/search.py', 'utils/siglip.py']` (the allow-list as a named constant with a comment
     per entry). Timed this session: 0.2 s, current hits are exactly `api/routers/search.py`,
     `tag_existing.py`, `utils/siglip.py`; `conda/` must be excluded because it is untracked and
     `--untracked` would scan it; `tests/` is excluded because `tests/test_tagging.py` and this file contain
     the string.
   - key decision: how the existing `test_tagging.py` case survives. Default: keep it, it still proves
     `tag_existing` loads `google/siglip2-so400m-patch16-naflex` and builds the tagger with the right
     `model_name`/`backend`; its fake `AutoModel.from_pretrained` is now reached through the helper. Add a
     second case: with `utils.siglip.load_siglip_model` monkeypatched, `build_clip_tagger(cfg, device="cuda")`
     calls it with `(name, "cuda")` and passes its return value to `CLIPTagger` (no torch needed).
   - accepted consequence (record, do not re-litigate): CUDA retagging encodes the vocabulary in fp16, as the
     single-pass scorer already did. Measured on the 16gb vocabulary (452 prompts) and 49 photos: text
     embeddings cosine >= 0.99997, similarity drift max 3.9e-4, tag SETS change for 2 of 49 photos (4 %),
     137 vs 138 tags in total. Affects `--recompute-tags --force`, `tag_existing.py` and the finalization
     auto-tag of a scan (which only tags untagged photos).
   - verify: red first: (d) fails today because `tag_existing.py` is an extra hit, and the new cuda case in
     `test_tagging.py` fails because the module calls `AutoModel` itself. Green after (a)-(d). Mutation
     check: temporarily reintroduce a direct `AutoModel.from_pretrained` in a scratch `.py` at the repo
     root (untracked) and confirm the test fails, then delete it and confirm `git status --porcelain <file>`
     is empty (retype by hand, never `git checkout`).
   - affected files: tag_existing.py, utils/siglip.py, api/routers/search.py, tests/test_siglip_loader.py, tests/test_tagging.py

6. **Planner: named CUDA margin of 1.3 GiB.**
   - description: `models/model_manager.py`: add `_CUDA_PASS_MARGIN_GB = 1.3` as a class attribute beside
     `_CGROUP_CAPACITY_CEILING_GB` (:123) with a comment (what it must cover: CUDA context, ONNX Runtime
     arena, non-torch residue, the desktop; `detect_vram()` returns the card's TOTAL, not free; GPU mode only,
     CPU mode uses `_cpu_pass_capacity_gb`). In `group_passes_by_vram` (:~1476-1478) replace the comment
     "1GB safety margin" and `capacity = available_vram - 1.0` with `capacity = available_vram - self._CUDA_PASS_MARGIN_GB`.
     Also fix the docstring of the method if it mentions 1 GB.
   - key decision: class attribute vs module constant vs config key. Default: class attribute, no config key.
     Why: matches the neighbouring constant, the user named it `_CUDA_PASS_MARGIN_GB`, and the repo rule is no
     speculative knobs. `empty_cache` is NOT added: `_process_chunk` already reclaims between passes
     (`multi_pass.py:482-484`, `_reclaim_freed_memory` :503-520), so the residue is non-torch.
   - verify: in `tests/test_model_manager.py::TestPassGrouping` add (red today) a roster pin using the real
     function: `['clip', 'saliency', 'topiq']` (declared 5 + 3 + 2.5 = 10.5) at 11.7 -> 2 bins, at 11.99 -> 1
     bin (today both are 1). Add one full-roster pin: with the `16gb` roster from the Step 7 script, 11.7 ->
     4 bins with `['insightface']` alone. Update the stale comments that quote the 1.0 margin (:216 "Capacity =
     vram - 1.0", :222 "cap = 9-1 = 8", :227 "capacity = 20-1 = 19") to the 1.3 arithmetic (18.7 / 7.7 / 18.7);
     the three existing assertions still hold (computed by hand in the research, now actually run).
     Run `tests/test_model_manager.py`, `tests/test_multi_pass.py` (:221 samp_net survives grouping) and any
     other suite that calls `group_passes_by_vram` (`git grep -l group_passes_by_vram tests`), fix only
     assertions whose expected grouping legitimately moved, and list each in the PR.
   - affected files: models/model_manager.py, tests/test_model_manager.py

7. **Regenerate the pass-grouping docs from the planner; re-measure the 12 GB run.**
   - description:
     (a) Scratch script (scratchpad, NOT committed; adapt `followups/plan_margin.py`): for each profile
     `8gb`, `16gb`, `24gb`, build `ChunkedMultiPassProcessor(scorer, ModelManager(config), config.config)` on a
     COPY of `config/scoring_config.default.json` (so the local override cannot skew it), `models =
     proc._select_models()`, then the REAL `mm.group_passes_by_vram(models, total)` (not a re-implemented
     bin-packer; this step runs after Step 6). Sweep totals in 0.01 steps over [7.9, 8.1], [11.6, 12.1],
     [15.9, 16.1], [23.6, 24.1] and also print 32 / 36 / 40 / 48; print the plan only where it changes, so
     knife-edges are visible. Print model names with declared GB per pass.
     (b) Table rows (one per card class, fixed column set Pass 1-Pass 4): `8GB` (`8gb` profile, 8.0),
     `12GB` (`16gb` profile, opt-in, 11.7), `16GB` (`16gb` profile, 15.99 as this box's 4070 Ti SUPER reports),
     `24GB` (`24gb` profile). Copy every model's display name from the language's existing cells (SigLIP vs
     CLIP, "Qwen3.5 tagger" vs the 4B tagger, Q-ReAlign, saliency, TOPIQ, TOPIQ IAA, TOPIQ NR-Face, LIQE,
     SAMP-Net, InsightFace). Replace the `24GB+ | All models together (single-pass)` row with the planner's
     two-pass `24gb` plan; add a `Nxx GB+ single pass` row only if the sweep proves one exists and at what
     total (the `24gb` roster declares ~33.5 GB, so only a ~35 GB+ total could be one pass).
     (c) Prose: a sentence under the table stating the planner holds back 1.3 GiB of the card's total VRAM
     for CUDA/ONNX Runtime overhead and the desktop; a footnote that a 12 GB card reporting about 11.8 GiB
     or more (value taken from the sweep) plans the three-pass grouping (Qwen3.5 tagger + Q-ReAlign; SigLIP +
     saliency + TOPIQ; TOPIQ IAA + TOPIQ NR-Face + LIQE + SAMP-Net + InsightFace) instead of four.
     (d) Update every "three passes" for 12 GB: `docs/CONFIGURATION.md:620` and its six translations (the
     "10-14GB can opt in" paragraph; find each with `git grep -n '16gb' docs/<lang>/CONFIGURATION.md` near the
     VRAM Auto-Detection section) and the `CHANGELOG.md:33` bullet ("splits them into three passes (...)").
     (e) Re-measure: copy the scratch harness `...\scratchpad\vram12\harness.py`, run it capped at 11.7 on a
     fresh copy of a test DB over `D:\photo-llm\photos` (never `photo_scores_pro.db`, never delete a photo
     file), plus the uncapped control. Record the exit code, OOM-line count (must be 0), NVML peak minus
     baseline (must be <= 11.7), tags 49/49, 1152-dim embeddings, face rows equal, saliency box count within 1
     of the control, per-pass peaks, and the NVML residue between passes. Rewrite the "Measured on a 49-photo
     run" paragraph (EN `docs/CONFIGURATION.md:797` + six translations) and the CHANGELOG sentence with the
     new numbers and the 4-pass grouping. If the harness cannot run, rewrite them to say the figures were
     measured with the previous three-pass grouping; never leave per-pass peaks attached to a plan they did
     not come from.
   - key decision: which total the `12GB` row uses. Default: 11.7. Evidence: it is the cap of the one measured
     run; the planner flips from 4 to 3 passes at roughly 11.8 GiB, i.e. between a 3060-class 12 GB card
     (~11.76 GiB reported, unverifiable here) and 4070-class cards (~11.99); the conservative plan is
     what the doc shows, with the footnote for the other. The `16GB` row stays at 15.99 because that is
     what this box's card reports and it is unchanged by the margin (3 passes at 1.0 and 1.3).
   - verify: the script's printed plan for each row equals the cell text (compare mechanically: model names
     per pass); `git grep -n "three passes\|3 passes\|tres pasadas\|drei Durchläufe\|trois passes\|tre passaggi\|três passagens\|三遍"
     -- docs CHANGELOG.md` shows no remaining 12 GB claim of three passes (adjust the pattern to what the
     translations actually say); `git diff --stat docs/` shows hunks, not whole files; all seven tables have
     the same column count and row count.
   - affected files: docs/CONFIGURATION.md, docs/{de,es,fr,it,pt,zh}/CONFIGURATION.md, CHANGELOG.md; scratch only otherwise

8. **Document the NULL behaviour in all seven languages; CHANGELOG.**
   - description:
     - `docs/SCORING.md` "Supplementary IQA Dimensions" (:~209-224): a paragraph "When a supplementary model
       cannot score a photo": TOPIQ NR-Face needs a face its own detector finds, LIQE needs a short side of at
       least 224 px; such a photo stores NULL (never a neutral 5.0), an out-of-memory error is retried at half
       the batch size down to one image before giving up and is logged as an ERROR; in the aggregate a NULL
       scores neutral (5.0), so scores do not move; **a gallery range filter on Face Quality (IQA) or LIQE
       (and Aesthetic IAA / Aesthetic V2.5) no longer matches a photo without a score** (before, a faceless
       photo carried 5.0 and matched `min_face_quality_iqa=0`); values stored by earlier versions stay until the
       photo is rescanned. Add the failed-primary sentence: if the primary TOPIQ model fails for a photo, the
       photo is recorded as failed and retried (`--retry-failed` / next scan) instead of being saved with 5.0.
     - `docs/COMMANDS.md` "Supplementary Quality Models" (:~157-163) and the `--recompute-iqa` row (:127): a
       photo a model cannot score stays NULL and is selected again by the next run; `--score-topiq` skips it.
     - `docs/VIEWER.md` (:~147 "Extended Quality" row area): one sentence that these range filters exclude
       photos with no score, with a pointer to the SCORING.md paragraph.
     - Translations: the same hunks in `docs/{de,es,fr,it,pt,zh}/SCORING.md`, `COMMANDS.md`, `VIEWER.md`
       (locate by the English anchor's translated heading and `git grep -n face_quality_iqa docs/<lang>/`;
       line numbers drift by a few lines per language).
     - `CHANGELOG.md` `## [Unreleased]`, in the file's existing bold-lead style:
       Fixed: "PyIQA scorers no longer store a neutral 5.0 for an image they failed to score (#184)" (NULL,
       the OOM retry, the faceless-photo case, the gallery-filter effect, existing 5.0s untouched);
       Fixed: "A failed TOPIQ is no longer saved as 5.0" (retried via scan_failures); Fixed: `topiq_nr_face`
       scored one image at a time; Changed: "`tag_existing.py` loads SigLIP in fp16 on CUDA" with the 2/49
       tag-set change and the rescan note (`--recompute-tags --force` to apply); Changed: "The pass planner
       holds back 1.3 GiB (was 1.0)" with the new 12 GB plan; and edit the existing #184 bullets (:33 three passes; :55 the SigLIP loader list, add tag_existing.py) so
       they do not contradict.
   - key decision: where the filter-effect sentence lives. Default: SCORING.md as the single full
     explanation, one-line pointers elsewhere. Why: avoids seven-language drift on a long paragraph.
   - verify: `git grep -n "5\.0" docs/SCORING.md docs/COMMANDS.md` shows no stale claim that a failed model
     stores 5.0; every English hunk has a counterpart in each of the six languages
     (`git diff --stat` file list is symmetric); links/anchors added resolve (`COMMANDS.md#...`, `SCORING.md#...`
     exist); `.claude/patterns/i18n-sync.md` is satisfied for the docs half (no client i18n keys change).
   - affected files: docs/{SCORING,COMMANDS,VIEWER}.md and the same three files in docs/{de,es,fr,it,pt,zh}/, CHANGELOG.md

## Final verification gate

1. Targeted suites, output redirected to a file and read:
   `D:\photo-llm\conda\python.exe -m pytest tests/test_pyiqa_failure.py tests/test_pyiqa_batching.py tests/test_multi_pass_pyiqa.py tests/test_iqa_flush_none.py tests/test_saliency_failure.py tests/test_single_pass_preserves.py tests/test_siglip_loader.py tests/test_tagging.py tests/test_model_manager.py tests/test_multi_pass.py tests/test_renormalisation.py -q`.
   Then the full suite `tests/ -q`; compare failures against a baseline taken before Step 1 (known box-local
   classes: 401s from the local viewer password, POSIX-path tests under Windows `Path.resolve()`); the
   change must add none.
2. CI-shape check: confirm the torch-free files import and pass with torch/transformers/pyiqa blocked
   (e.g. run them with `-p no:cacheprovider` after temporarily making `import torch` raise via a scratch
   `sitecustomize` on `PYTHONPATH`, or simply read their imports: no module-level `import torch`,
   `pyiqa`, `transformers`). `tests/test_pyiqa_failure.py` must start with `pytest.importorskip("torch")`.
3. `D:\photo-llm\conda\python.exe -m ruff check .` clean. No client change, so `npm run gen:api` is not needed
   (no API/schema change); state that in the PR.
4. Red-then-green evidence recorded per step (the failing assertion text before the fix).
5. Fault-injection of the two guards, each restored by hand-retyping and confirmed with
   `git status --porcelain <file>` empty: (a) put the `clear_device_cache` call back inside the `except` in
   `pyiqa_scorer.py` and see test (4) fail; (b) remove `if score is None: continue` from `_flush` and see the
   Step 3 test fail.
6. Step 7's capped end-to-end re-run: exit 0, zero OOM lines, NVML peak - baseline <= 11.7, 49/49 tags, 1152-dim,
   saliency boxes within 1 of the control, faces equal; the doc paragraph quotes these numbers. Also confirm the
   log shows the 4-pass plan and no ERROR from `facet.pyiqa` (the 49 test photos include no-face landscapes:
   those must produce NULL `face_quality_iqa` and no ERROR, only the DEBUG/`unload()` expected-skip total).
7. Browser check (project workflow) on a COPY of a test DB, never the prod DB: start the viewer, load the
   gallery with `min_face_quality_iqa=0`, confirm a photo with NULL `face_quality_iqa` is excluded and a
   scored one is shown, and the photo detail/critique dialog renders with no console errors for a NULL metric.
8. `git status --porcelain` lists only the files named above (plus scratch outside the repo); `git diff --stat`
   shows no whole-file rewrites in the 24 translated docs; `/review-all:review-all` on the diff before commit.

## Adversarial review

One independent pass (opus, read-only; probe `scratchpad/advreview/margin.py`, exit 0). All 17 findings
adjudicated below. **The amendments in this section are binding and supersede conflicting step text.**

| # | Class | Finding | Verdict | Amendment |
|---|---|---|---|---|
| 1 | BLOCKER | Expected-skip totals stay "at unload()", but a RAM-cached model never reaches `.unload()` (`unload_model` parks it, `unload_all`/`evict_cpu_cache` drop it) — so decision 1 ("pass end") was narrowed on a false premise | ACCEPTED | A1 |
| 2 | GAP | Ascending sort on Aesthetic (IAA) / LIQE / Face Quality (IQA) puts NULLs first (`gallery.py` bare `{col} ASC`) | ACCEPTED | A2 |
| 3 | GAP | `--recompute-iqa` counts faceless NULL `face_quality_iqa` as work and decodes the thumbnail before the `face_count = 0` skip — after this change it never reaches "nothing to do" | ACCEPTED | A3 |
| 4 | GAP | Sticky forward batch untested within one shape group | ACCEPTED | A4 |
| 5 | GAP | Non-OOM group fallback must also leave the `except` before per-image retries | ACCEPTED | A4 |
| 6 | GAP | Step 1(d) (preprocess failure / OOM) has no verify | ACCEPTED | A4 |
| 7 | GAP | `--pass quality` builds the processor without `on_error` (multi_pass.py:1424): no `scan_failures` row | ACCEPTED (text) | A5 |
| 8 | GAP | Planner sweep doesn't say how each profile is forced | ACCEPTED | A6 |
| 9 | GAP | Pins presented as red-first that pass today | ACCEPTED | A7 |
| 10 | GAP | `--score-topiq` does not filter `IS NULL` (facet.py:2884-2887) | ACCEPTED | A8 |
| 11 | GAP | Step 6 depends on a Step 7 output | ACCEPTED | A9 |
| 12 | GAP | Recursive forward must keep `**self._forward_kwargs` | ACCEPTED | A4 |
| 13 | NOTE | CPU allocator OOM is not recognised by `is_out_of_memory_error` | ACCEPTED (docs) | A10 |
| 14 | NOTE | A deterministic batch-1 TOPIQ OOM now skips the photo's whole save every scan | ACCEPTED (risk, by decision 2) | A11 |
| 15 | NOTE | Planner premises reproduced (4 passes at 11.7, flip to 3 at exactly 11.8) | CONFIRMED | — |
| 16 | NOTE | `stats.py:951-953` previews NULL as 0 in the weight-tuning preview while the aggregate uses 5.0 | DEFERRED — pre-existing mapping, already wrong for commercial mode; listed as a follow-up, not fixed here | — |
| 17 | NOTE | Step 5's red-first CUDA test could load the real 4.5 GB SigLIP | ACCEPTED | A12 |

### Amendments

- **A1 (skip totals, decision 1).** Expected skips (`No face detected`, `Short side is less than 224`) and
  failures are reported once per model **when the scorer leaves the process**, not only from
  `PyIQAScorer.unload()`: add `PyIQAScorer.report_skips()` (logs the accumulated totals at the level
  `unload()` uses today, then clears them; idempotent) and call it from `unload()` and from
  `ModelManager.unload_all` and `evict_cpu_cache` for every dropped cached entry that has it. `unload_model`'s
  RAM-park branch does NOT report (that would print once per chunk). Unexpected failures still log ERROR at
  the end of each call (Step 1). Tests: a cached scorer dropped by `unload_all`, and one by
  `evict_cpu_cache`, each log the "No face detected N" total exactly once; `unload_model` parking logs nothing.
  Affected: `models/model_manager.py` (+ `tests/test_model_manager.py`). Final gate 6 reads that line.
- **A2 (sort).** Add `aesthetic_iaa`, `liqe_score`, `face_quality_iqa`, `aesthetic_v25` to the NULL-guard
  tuple in `api/routers/gallery.py` (the `('learned_score', 'narrative_moment_confidence')` branch) so ASC
  puts NULLs last, as it already does for those two. Red-first test in the gallery suite with seeded rows
  (one NULL, two scored) asserting ASC order ends with the NULL row. Document alongside the filter effect
  (Step 8). Other sites the reviewer named (`api/config.py:1736`, `capsule_generator.py:1883`,
  `health.py:176`, `optimization/model_eval.py:148-151`) are added to the Step 4 inventory with a one-line
  NULL-safety verdict each; no code change unless one is unsafe.
- **A3 (`--recompute-iqa`).** The up-front NULL count for `face_quality_iqa` (scorer.py:2229-2238) gains the
  same `face_count` condition `_run_iqa_pass` applies, and `_run_iqa_pass` checks `face_count = 0` before
  decoding the thumbnail. Red-first test: a seeded faceless row with NULL `face_quality_iqa` is neither
  counted nor decoded, and a library where that is the only NULL exits "nothing to do".
- **A4 (Step 1 tests/contract).** (a) the per-group slice loop re-reads `self.forward_batch` per slice
  (while-loop, not a precomputed range); test: an 8-image same-shape group with an OOM on the first
  4-forward issues forwards of `[4, 2, 2, 2]`-then-done (exact sequence asserted). (b) Both the OOM and the
  non-OOM group fallbacks run after the `except` block has exited; test: group non-OOM error followed by a
  per-image OOM still clears the cache with no active exception (assert `sys.exc_info()` is empty inside the
  patched clear). (c) Step 1(d) gets a verify: `_preprocess_image` raising (OOM and non-OOM) yields the
  serial path's per-image results, `None` only where the image itself fails. (d) every forward, recursive
  or not, passes `**self._forward_kwargs`; `tests/test_extended_iqa_wiring.py` joins the targeted gate.
- **A5.** Step 2's text: under `--pass quality` a TOPIQ `None` keeps the stored value (restricted save) and
  is NOT recorded in `scan_failures` (no `on_error` there); only a full scan records and retries it.
- **A6.** Step 7's sweep script forces each profile by setting `config["models"]["vram_profile"]` (as
  `followups/plan_margin.py` does) and patches `processor.available_vram` to the swept total so
  `_profile_tagger_fits` sees it too.
- **A7.** S12 rewording: tests that pin unchanged behaviour (Step 2(e), the full-save leg of 2(d), the
  Step 4 filter pin) are labelled PIN in the build report and are exempt from red-first; every other new
  test must be shown failing on pre-change code.
- **A8.** `--score-topiq` text corrected: it re-scores every photo with a thumbnail (no NULL filter); a
  `None` now leaves the stored value.
- **A9.** Step 6's full-roster pin hardcodes the 11.7 GiB plan from finding 15:
  `[tagger, qrealign], [clip, saliency, topiq_iaa], [topiq, topiq_nr_face, liqe, samp_net], [insightface]`.
- **A10.** Docs say "a GPU out-of-memory error is retried at half the batch size"; CPU allocator failures
  take the ordinary per-image failure path.
- **A11.** Risk recorded: on a card where TOPIQ cannot fit even one image, decision 2 means the photo is not
  saved and is retried each scan. Final gate 6 (capped 11.7 GiB run: 0 OOM lines, every photo saved) is the
  guard; if it fails, stop and report rather than loosen decision 2.
- **A12.** Step 5's CUDA test injects a fake `transformers` module (as `tests/test_siglip_loader.py` does);
  no real model load in any new test.
