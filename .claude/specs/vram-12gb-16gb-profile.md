# Spec: Support the full 16gb feature set on a 12 GB VRAM card (issue #184)

Opt-in via `"vram_profile": "16gb"` (or `FACET_VRAM_PROFILE=16gb`). Auto-detect thresholds
(`ModelManager.get_recommended_profile`, models/model_manager.py:1085-1103: >=20 -> 24gb,
>=14 -> 16gb, >=6 -> 8gb) stay UNCHANGED (user decision). No silent degradation: a model
that cannot run stores NULL and logs ERROR, never a neutral number that looks like a result.

## Objective + success criteria
- S1. On a card capped to 11.7 GiB, `facet.py` with the 16gb profile exits 0, logs no OOM,
  and NVML peak minus baseline <= 11.7 GiB.
- S2. The capped run's results match the uncapped control: 49/49 tags, 1152-dim embeddings,
  Q-ReAlign present, saliency `subject_bbox` count within |capped - control| <= 1 and both
  >= 43 of 49 (pre-change baseline 44), face rows equal.
- S3. `_affordable_batch_size` on `cuda` sizes the batch from free VRAM, clamped
  `[MIN_BATCH_SIZE, ceiling]`; unit-tested with a mocked `mem_get_info`.
- S4. `get_saliency_masks` retries a CUDA-OOM sub-batch at half size down to 1 (unit-tested).
- S5. A final saliency failure logs ERROR with the batch size and never stores a neutral
  number: the photo's saliency keys are left unset so a restricted save / `--recompute-saliency`
  keeps stored values; a photo that never had saliency stays NULL. Every reader is NULL-safe.
- S6. SigLIP loads straight into fp16 on CUDA; embeddings match the old path (cosine >= 0.9999).
- S7. `topiq_nr_face` and InsightFace leave no avoidable residue after unload (measured).
- S8. `MODEL_VRAM_REQUIREMENTS`: `saliency` 2 -> 3, `topiq` 2 -> 2.5; nothing else touched.
- S9. docs (EN + de/es/fr/it/pt/zh CONFIGURATION.md) and CHANGELOG updated in the same change.

## Not changing
- Auto-detect thresholds, the profile table, `get_recommended_profile` (user decision).
- Any `MODEL_VRAM_REQUIREMENTS` entry other than `saliency` and `topiq` (the rest are
  overestimates; user decision - leave them).
- PyIQA's per-image OOM fallback then 5.0 (models/pyiqa_scorer.py:337, :392): same silent
  degradation class, OUT OF SCOPE - note as follow-up only.
- The empty-mask neutral result inside `_score_from_mask` (models/saliency_scorer.py:322-329):
  that is a successful run that found no subject, not a failure. Kept.
- `licensing.reraise_if_refused()` in `_pass_saliency` (tests/test_licensing_gate.py:436) stays.
- CPU / MPS branches of `_affordable_batch_size` and the SigLIP load (CPU/MPS unchanged).
- No new columns on `photos` (nothing sticky is added); no schema change.
- No back-compat shims: the neutral-default constants are removed, not aliased.

## Domain rules (verified = read in source this session; assumed = not yet proven)
- verified(models/saliency_scorer.py:65,210-220): `DEFAULT_BATCH_SIZE = 8`; on `cuda` the
  affordable batch is the ceiling, VRAM never consulted. `ACTIVATION_GB_PER_IMAGE = 2.4`,
  `USABLE_MEMORY_FRACTION = 0.8` already exist (RAM path uses them).
- verified(measured): batch curve alloc/reserved GiB b1 1.60/2.61, b2 2.76/5.02, b4 5.08/9.64,
  b8 9.70/19.59 -> marginal reserved ~2.4 GiB/image; the existing 2.4 constant is correct.
- verified(measured): under the 11.7 GiB cap b8 OOMs ("Tried to allocate 3.75 GiB"), score_batch
  swallows it and writes neutral defaults -> 0/49 bboxes vs 44/49. With batch 2 it passes.
- verified(models/saliency_scorer.py:425-446): `score_batch` catches `Exception`, logs WARNING,
  returns neutral dicts. Also a second neutral at :324 (empty mask, kept).
- verified(processing/scorer.py:578-583): the aggregate reads these metrics through
  `safe_float(m.get(..), 5.0)` - a NULL already scores as neutral 5.0. This is the existing
  "scanned before saliency existed" path; reuse it, add nothing.
- verified(config/scoring_config.default.json): every category ships all four
  `subject_*_percent` / `bg_separation_percent` weights at 0, so NULL changes no default aggregate.
- verified(config/category_filter.py:35-45,157-164): a numeric filter on a None value returns a
  mismatch (the photo does not match) - NULL is handled, no crash.
- verified(utils/device.py:439): `is_out_of_memory_error(ex)` already recognises
  `torch.cuda.OutOfMemoryError`, `torch.OutOfMemoryError` and MPS RuntimeError. Use it; do not
  hand-roll an except clause.
- verified(analyzers/face.py:46-70): `FaceAnalyzer` builds `FaceAnalysis(providers=[CUDA, CPU])`
  and has NO `unload()`. `ModelManager.unload_model` (:379-399) then falls to `del model`
  (no `.unload`, no `.cpu`, not a dict); `_can_cache_to_ram` refuses insightface on CUDA (:~500).
- verified(models/pyiqa_scorer.py:174-193): `PyIQAScorer.unload` does `model.cpu()`, `del`,
  `clear_device_cache`. It does not touch anything pyiqa holds outside `.model`.
- verified(models/model_manager.py:277-281): SigLIP loaded fp32, `.to(device)`, then `.half()`
  (the ~5 GB transient). Inference alone is 2.66 GiB reserved at chunk 32 (measured).
- verified(db/schema.py:145-149): all five columns are nullable (`REAL` x4, `TEXT`); the
  `subject_bbox` comment already says "NULL until saliency runs".
- assumed: the cause of the topiq_nr_face (~0.2 GiB) and InsightFace ORT (~0.44 GiB) residue.
  Candidates only; Step 5 measures before changing anything. Long-run growth is UNVERIFIED.
- assumed: reserved-but-unallocated torch cache is reusable by the next forward pass, so it
  should be added to `mem_get_info()` free. Confirmed or refuted by Step 1's harness run.

## Reader inventory for the five saliency columns - NULL handling
Columns: `subject_sharpness`, `subject_prominence`, `subject_placement`, `bg_separation`
(REAL) and `subject_bbox` (TEXT JSON). Writers and readers, one line each.

| # | Site | Today | NULL handling / action |
|---|------|-------|-----------------------|
| W1 | models/saliency_scorer.py:432-440 `score_batch` failure | returns 5.0/0.0/5.0/5.0/None | CHANGE: return `None` x5 + ERROR log with batch size (Step 3) |
| W2 | processing/multi_pass.py:881-884 `_pass_saliency` | `scores[i].get(key, 5.0)` | CHANGE (B1): a failed photo (all five values None from `score_batch`) gets `results[path][SALIENCY_FAILED_KEY] = True` and the five keys are NOT set (left unset). Successful photos keep writing their values (a legit no-subject photo still carries the neutral numbers and bbox None). No `5.0` default |
| W3 | processing/multi_pass.py:40-48 consts, :1244-1262 `_save_results_restricted` | `FACE_FAILED_KEY` / `FACE_PASS_COLUMNS` skip pattern; `subject_bbox` -> `json.dumps(v) if v else None` | CHANGE (B1): add `SALIENCY_FAILED_KEY` and `SALIENCY_PASS_COLUMNS` = the 5 columns beside the face pair; extend the skip at :1244-1262 and `_iter_saveable`/`SCAN_FAILED_KEY` handling exactly as for faces (`if key in (SCAN_FAILED_KEY, FACE_FAILED_KEY, SALIENCY_FAILED_KEY): continue` and `if saliency_failed and key in SALIENCY_PASS_COLUMNS: continue`). Effect: a restricted/partial save keeps the stored values. NULL is written only for a photo that never had saliency (fresh full save, where the base record leaves the keys NULL via batch_processor W5) |
| W4 | processing/multi_pass.py:1077-1080, :1209-1213 (record build) | `data.get(..)` / bbox `if data.get('subject_bbox') else None` | OK, NULL-in -> NULL-out |
| W5 | processing/batch_processor.py:482-485 | sets all four to None for non-saliency path | OK, already NULL |
| W6 | processing/scorer.py:2534,2557,2578 `save_photo*` (INSERT OR REPLACE) | `setdefault('subject_bbox', None)`; named params | OK; verify the four REALs accept None (named params, no coercion) |
| W7 | facet.py:2813-2818 `--recompute-saliency` UPDATE | `s.get(k, 5.0/0.0/5.0/5.0)` | CHANGE (B2): in `_flush_saliency_batch` skip the UPDATE for a failed result (all five None) and do not count it in `updated`/the return value; log it. No `5.0` defaults. A row that never had saliency keeps `subject_bbox IS NULL` and is re-selected next run (resumable); under `--force` an existing good value SURVIVES a failure (test below). A legit no-subject row is also re-selected - pre-existing, unchanged |
| R1 | processing/scorer.py:580-583 `build_metric_vector` | `safe_float(.., 5.0)` | OK - NULL scores neutral 5.0 (existing pre-saliency path). Add a test |
| R2 | processing/scorer.py:484 `SCORING_METRIC_KEYS`, :671-728 `calculate_aggregate_logic` args | None default | OK |
| R3 | processing/scorer.py:1640 recompute-average column list | SELECTs the 4 cols; rows -> R1 | OK via R1 |
| R4 | config/category_filter.py:91,157-164 | None -> filter mismatch | OK (photo does not match a saliency-range category filter); no change |
| R5 | config/percentile_normalizer.py, api/top_picks.py | git grep: no reference to these columns | N/A (verified none) |
| R6 | optimization/insights_miner.py:27 `METRIC_COLUMNS` | numpy arrays | VERIFY in Step 3: confirm rows with NULL are dropped before `np.asarray(..., float)`; if NULL reaches numpy it becomes nan/object. Fix only if the check fails (UNVERIFIED) |
| R7 | weight optimizer via `SCORING_METRIC_KEYS` (processing/scorer.py:484) | uses `build_metric_vector` | OK via R1 (training sees 5.0 for NULL, same as serving) |
| R8 | api/models/gallery.py:99-102 `Photo` | `Optional[float] = None` | OK, NULL is already a legal wire value; no model change |
| R9 | api/routers/gallery.py:494-497, albums.py:1191-1194 min/max range filters | SQL `col >= ?` | OK: SQL comparison with NULL is false, row excluded (same as pre-saliency rows) |
| R10 | api/routers/stats.py:75-78, api/routers/comparison.py:74-77,668,902-905 | column maps / SELECT | stats: AVG/histogram ignore NULL (VERIFY the histogram bucket code tolerates None rows). comparison `preview` (:~915) does `metrics.get(k) or 0` -> NULL is **0**, while aggregate uses **5.0**: a pre-existing inconsistency; DECISION: leave (weights default 0, preview-only) and list it under Open questions |
| R11 | api/routers/critique.py:59-62,91-94,118-120,393-394 | metric dicts | VERIFY the rule-critique builder skips None metrics (pre-saliency rows already hit this); add a test only if it does not |
| R12 | api/routers/burst_culling.py:1381-1405 | reads sharpness/prominence/bbox, emits None | OK, already NULL-aware (`subject_sharpness: None` defaults at :1396) |
| R13 | api/routers/saliency.py:203-204,312-315,352-363 | `parse_subject_bbox`, None defaults | OK, already NULL-aware |
| R14 | api/routers/social_crop.py:82-101 | `parse_subject_bbox(None)` -> falls to faces -> center | OK; this is the documented fallback chain |
| R15 | api/config.py:1738, api/db_helpers.py:408 (column allow-lists) | names only | N/A |
| R16 | client: schema.d.ts (all `number \| null`), burst-culling.pipes.ts (`number \| null`), burst-culling.component.ts:2781-2822, comparison.pipes.ts, i18n keys | typed nullable | OK, no client change; run `npm run gen:api` and expect NO diff (no API change) |
| R17 | build-tools/verify_scan_db.py:24,30 | lists the columns as scan-completeness checks | CHECK: if it asserts NOT NULL, a failed saliency now fails the check - correct behaviour; no change |

## Files to create / change
- models/saliency_scorer.py (batch sizing, OOM retry, NULL on failure, constants)
- models/model_manager.py (`MODEL_VRAM_REQUIREMENTS` saliency 3 / topiq 2.5; fp16 SigLIP load)
- processing/multi_pass.py (:881-884 default removal)
- facet.py (:2816-2818 default removal in `--recompute-saliency`)
- analyzers/face.py (new `FaceAnalyzer.unload()`), models/pyiqa_scorer.py (`unload`), only as Step 5 proves
- tests/test_saliency.py, tests/test_model_manager.py, tests/test_multi_pass.py, new tests as listed per step
- docs/CONFIGURATION.md, docs/SCORING.md, docs/COMMANDS.md (as needed) and docs/{de,es,fr,it,pt,zh}/CONFIGURATION.md (+ SCORING/COMMANDS where translated and edited), CHANGELOG.md
- Scratch (not committed): the vram12 harness, copied and extended in Step 7

## Open questions / risks
- Free-VRAM probing races with other processes (a viewer holding VRAM); mitigated by the OOM retry.
- Resolution scaling is unmeasured above/below 1024. Default: per-image cost
  `2.4 * max(1, resolution/1024)**2` (conservative upward, never below the measured figure).
- comparison preview uses `or 0` vs aggregate 5.0 for NULL (R10): pre-existing, not widened here.
- The leak causes (Step 5) are hypotheses until Step 5 (a) measures them; Step 5's verify has no
  fallback (N3).
- A very low free-VRAM reading yields batch 1 and still OOMs at 1: that is the genuine
  "does not fit" case and ends in ERROR + NULL (never a fake score).
- ruff and pytest: use `D:\photo-llm\conda\python.exe -m pytest` / `-m ruff` (no venv on this box).
  `git grep`, not ripgrep, on this tree.

## Steps

1. **Planner table: raise only the underestimates.**
   - key decision: which entries move. Default: `saliency` 2->3, `topiq` 2->2.5 (measured
     topiq 2.41 NVML / 2.19 reserved); every other entry untouched.
   - also: `SaliencyScorer.vram_gb` property (saliency_scorer.py:~447, returns 2) -> 3 so the two stay consistent (verify nothing else reads it: `git grep vram_gb`).
   - also (N1): update the existing assertion at tests/test_model_manager.py:155 (`topiq == 2` -> `2.5`) and change `get_model_vram`'s return annotation from `int` to `float` (grep its callers for int-only use).
   - verify: `git diff models/model_manager.py` shows the two table lines plus the annotation; add an assertion in
     tests/test_model_manager.py that `MODEL_VRAM_REQUIREMENTS['saliency'] == 3` and `['topiq'] == 2.5`; run the existing `group_passes_by_vram` tests and fix any whose expected grouping legitimately shifts (record which and why in the PR). Run the pass planner for 11.7 GiB and note the passes for the doc (Step 8).
   - affected files: models/model_manager.py, models/saliency_scorer.py, tests/test_model_manager.py

2. **VRAM-sized saliency batch on CUDA.**
   - key decision: formula and what counts as free. Default:
     `free = torch.cuda.mem_get_info()[0] + (memory_reserved - memory_allocated)` (cache the
     process can reuse), `per_image = ACTIVATION_GB_PER_IMAGE * max(1, resolution/1024)**2`,
     `affordable = int(free * USABLE_MEMORY_FRACTION / BYTES_PER_GB / per_image)`, result
     `max(MIN_BATCH_SIZE, min(ceiling, affordable))`. Replace the early `return ceiling` for
     `DEDICATED_VRAM_DEVICE`; fall back to the ceiling only when `mem_get_info` itself raises
     (log WARNING once - not silent). Update the method docstring (it currently claims CUDA
     ignores memory) and the class constant comment.
   - G5: call `mem_get_info(self.device)`, `memory_reserved(self.device)`, `memory_allocated(self.device)` - always pass `self.device`, never the current-device default.
   - verify: new tests in tests/test_saliency.py with `torch.cuda.mem_get_info`,
     `memory_reserved`, `memory_allocated` monkeypatched (assert the device argument is passed): (a) 24 GiB free -> 8 (ceiling) and 16 GiB free -> 5 (int(16*0.8/2.4)); (b) ~9 GiB
     free -> 3; (c) 1 GiB free -> `MIN_BATCH_SIZE`; (d) explicit `requested=2` stays a ceiling;
     (e) `mem_get_info` raising -> ceiling plus one WARNING; (f) resolution 1536 lowers the batch.
     REWRITE the two existing tests that assert CUDA ignores memory:
     `test_cuda_batch_ignores_host_memory` (:285) and the cuda half of
     `test_cpu_and_cuda_batch_sizing_unaffected_by_mps_cap` (:339-349). CPU and MPS tests must
     stay green unmodified.
   - affected files: models/saliency_scorer.py, tests/test_saliency.py

3. **OOM retry in `get_saliency_masks` + NULL on final failure.**
   - key decision: retry granularity and the failure contract. Default: wrap the per-chunk
     forward (:247-254) in a helper. On `is_out_of_memory_error` it sets a flag and DROPS the
     exception reference; the `clear_device_cache` (`empty_cache`) and the half-size retry run
     OUTSIDE the `except` block (comment why: the traceback held by the live exception pins the
     failed activations, so freeing inside the handler frees nothing). It splits the chunk in
     half and recurses on each half, down to one image, preserving output order. Applies on MPS
     too (`is_out_of_memory_error` covers its RuntimeError) - state in the docstring.
     G3: at batch size 1 an image that still OOMs yields `None` for THAT image only (the mask
     list stays aligned, other images in the call keep their masks); `get_saliency_masks`
     returns a list with `None` holes and `score_batch` marks only those photos failed (all-None
     dict), scoring the rest normally. A *successful* halving logs one WARNING per call
     ("saliency batch reduced to N after OOM") WITHOUT the exception text (N2: it embeds
     allocator sizes and varies per run).
     `score_batch`: the whole-call except and the per-image `_score_from_mask` except log
     `logger.error("... batch size %d: %s", n, exc)` and return all-`None` dicts (no
     `default_scores`; delete it). The empty-mask neutral stays. A failed photo is what W2 turns
     into `SALIENCY_FAILED_KEY`. Keep `licensing.reraise_if_refused()` in `_pass_saliency`.
     Then apply W2/W3/W7 (remove the `5.0`/`0.0` defaults in multi_pass.py:882 and facet.py:2816-2818).
   - verify: tests (tests/test_saliency.py): (a) fake model raising `torch.cuda.OutOfMemoryError`
     once on a batch of 4 -> 4 masks returned, order preserved, one retry at 2 (assert call
     sizes `[4, 2, 2]`); (b) OOM at every size -> raises, `score_batch` returns 4 dicts of
     all-`None` and caplog has an ERROR record containing the batch size; (b2) MIXED batch:
     fake model that OOMs on any batch >1 and on one specific image even alone -> that image is
     `None`, the others have masks, order aligned, `score_batch` fails only that photo;
     (c) non-OOM exception is NOT retried; the WARNING text does not contain the exception
     message; (d) in tests/test_multi_pass.py `_pass_saliency` with a failed photo sets
     `SALIENCY_FAILED_KEY` and leaves the five keys unset (use `seeded_photos`; schema via
     `db.schema.init_database`, never a handwritten CREATE TABLE); a restricted save over a
     previously populated row KEEPS its stored five values; a fresh row (never scored) ends NULL;
     (e) aggregate: `build_metric_vector({...saliency cols None...})` yields 5.0 for the four and
     `calculate_aggregate_logic` with a nonzero saliency weight is finite and equals the
     explicit-5.0 result; (f) `--recompute-saliency` flush with a failing scorer issues no UPDATE for the failed
     photo, excludes it from the `updated` count, and under `--force` an existing populated row
     survives unchanged (monkeypatch `score_batch`); a never-scored row stays NULL and is
     re-selected by `subject_bbox IS NULL`.
     Also resolve R6/R10/R11 by reading insights_miner / stats / critique and add a test only
     where a NULL reaches arithmetic.
   - affected files: models/saliency_scorer.py, processing/multi_pass.py, facet.py, tests/test_saliency.py, tests/test_multi_pass.py, possibly optimization/insights_miner.py, api/routers/critique.py, api/routers/stats.py

4. **SigLIP direct-to-fp16 load on CUDA.**
   - key decision: where. Default: ONE shared loader helper (new, e.g. in utils/ beside
     device helpers) used by all three SigLIP load sites: models/model_manager.py
     `_load_clip_transformers` (:277-281), processing/scorer.py:878-887, and
     scripts/compute_aesthetic_clip.py:61-63. It passes `dtype=torch.float16` to
     `from_pretrained` when the device is cuda (transformers 5.15.0 is installed: use `dtype=`,
     not `torch_dtype`), then `.to(device).eval()`; no trailing `.half()`. CPU/MPS keep fp32.
     The sites must agree or an embedding made by one differs from another.
   - verify: (a) measure load transient with `torch.cuda.max_memory_allocated` before/after on
     the real model (expect ~5 GB -> ~2.2 GB; record both); (b) embedding parity: encode the
     49 test photos in D:\photo-llm\photos with the old and new load, assert per-photo cosine
     >= 0.9999 and all-finite (script in scratchpad, results quoted in the PR); (c) a unit test
     with a fake `AutoModel` asserting `dtype=float16` is requested only for cuda, and that
     all three sites go through the helper (`git grep "AutoModel.from_pretrained"` finds only it).
   - affected files: models/model_manager.py, processing/scorer.py, scripts/compute_aesthetic_clip.py, the new helper module, tests/test_model_manager.py

5. **Account for the two post-pass residues (reproduce first, smallest correct change).**
   - key decision: which path leaves the residue, and whether it is a leak or a deliberate
     cache. Facts (orchestrator-verified, models/model_manager.py:110-117, :380-388):
     `topiq_nr_face` and `insightface` are both in `CPU_CACHEABLE_MODELS`; `_can_cache_to_ram`
     refuses insightface only on CUDA (and `unload_model` then does `del model`), while
     insightface "retained rather than moved" is by design off-CUDA.
   - (a) Reproduce in a scratch script (3 cycles: load, one batch, `unload_model(name,
     reclaim=False)`, `_reclaim_freed_memory`; print `memory_allocated`, `memory_reserved`, NVML
     used) under BOTH `keep_in_ram='never'` and the default; record which path leaves the
     residue for each model and whether it grows per cycle.
   - (b) topiq_nr_face: fix whichever path leaves ~0.2 GiB. If it is the cache path
     (`_move_to_cpu` moves only `model.model`), move every CUDA tensor the wrapper holds
     (the whole `PyIQAScorer`, incl. helper modules); if it is the full-unload path, drop the
     extra references in `PyIQAScorer.unload` (`del`/`None`) and `gc.collect()` before
     `clear_device_cache`. Do only the one the measurement shows.
   - (c) insightface: its ORT CUDA arena while the analyzer object is alive is a deliberate
     trade-off, NOT a leak; do NOT redesign the cache. Account for it instead. Default (smaller
     change than a code path): add `FaceAnalyzer.unload()` ONLY IF (a) shows the session is
     retained when `_can_cache_to_ram` says no (the full-unload path has no `.unload`, so
     `del model` is all that happens - confirm the sessions actually die via weakref); and
     subtract a measured constant from planner capacity (`group_passes_by_vram`,
     models/model_manager.py:1407, `capacity = vram - 1.0`) of ~0.5 GiB
     (`INSIGHTFACE_RETAINED_RESIDUE_GB`, set from the measurement, rounded up) whenever
     insightface is retained AND is not in the pass being planned. If the residue is instead
     released by (c)'s unload, no constant is needed - then document nothing. Pick the smaller.
   - verify (can fail): scripted assertion in the Step 8 run: for each pass boundary, NVML
     used between passes minus baseline <= the budgeted residue (0.2 for topiq_nr_face after
     fix target <= 0.05; the insightface constant if added), and no growth across 3 cycles
     (cycle-3 residue <= cycle-1 + 0.05 GiB). A planner unit test asserts capacity drops by the
     constant only in the retained-and-absent case. If the measured residue exceeds the budget
     the step is NOT done - there is no document-and-move-on fallback.
   - affected files: models/pyiqa_scorer.py, models/model_manager.py (only if the constant is
     needed), analyzers/face.py (only if unload proves needed), tests/ (planner test in
     tests/test_model_manager.py; FaceAnalyzer test beside existing ones: `git grep -l FaceAnalyzer tests/`)

6. **Lint and unit gate.**
   - key decision: none; this merges the unit work. Default: run only touched suites first.
   - verify: `D:\photo-llm\conda\python.exe -m pytest tests/test_saliency.py tests/test_model_manager.py tests/test_multi_pass.py tests/test_single_pass_preserves.py tests/test_key_subject.py tests/test_social_crop.py tests/test_mps_memory.py -q`, then the full suite; `D:\photo-llm\conda\python.exe -m ruff check .`. Known box-local failures (401s from the local viewer password, POSIX-path tests under Windows `Path.resolve()`) are pre-existing; compare to a baseline run rather than chasing them. Redirect output to a file (do not trust a pipe's exit code).
   - affected files: none

7. **Docs + CHANGELOG, same change.**
   - key decision: what to state. Default text:
     - docs/CONFIGURATION.md: near the profile docs (~:575 and :611-620) an opt-in note: cards of
       10-14 GB can run the full feature set with `"vram_profile": "16gb"` (or
       `FACET_VRAM_PROFILE=16gb`); auto-detect still picks 8gb below 14 GB. Embedding-dim caveat
       (8gb = CLIP 768-dim, 16gb = SigLIP 1152-dim): link the existing section at ~:602-604 and
       state that switching profile on an existing library mixes dimensions - re-embed or rescan.
     - Fix the 12GB row of "Dynamic Pass Grouping" (~:786-791) using the passes the planner
       really produces after Step 1 and Step 8 measured peaks (the current row claims a single
       pass with TOPIQ; measured is 3 passes, peaks 7.86 / 7.38 / 2.60 GiB). Re-measure rather than copy these numbers if Step 1 shifts the grouping.
     - Saliency failure semantics: docs/SCORING.md "Subject Saliency Dimensions" (:198) and
       docs/COMMANDS.md `--recompute-saliency` (:124, :172-): a photo whose saliency fails now
       stores NULL (not 5.0) and logs ERROR; NULL scores neutral in the aggregate as for
       pre-saliency photos; `--recompute-saliency` re-selects it. Batch size is VRAM-derived on CUDA.
     - Update the "~2 GB" saliency figure in docs/COMMANDS.md:61 to ~3 GB.
     - Translations: mirror every edited section in docs/{de,es,fr,it,pt,zh}/CONFIGURATION.md
       (the 12-GB row exists in each: fr:801 is "12 Go"), plus the matching SCORING.md and
       COMMANDS.md sections - all six languages have CONFIGURATION/SCORING/COMMANDS.md (N4;
       verify with `ls docs/<lang>`), including the "~2 GB" saliency figure (COMMANDS.md:61 and
       its translations -> ~3 GB).
       Follow .claude/patterns/i18n-sync.md for the docs half. NO client i18n keys change
       (no new user-facing string in the UI).
     - CHANGELOG.md `## [Unreleased]`: "### Fixed" (saliency OOM no longer silently stores
       neutral scores) and "### Added" (12 GB opt-in) entries in the file's existing style.
   - verify: `git grep -n "12 *GB\|12 Go\|12 GB" docs/` shows all seven rows agree; `git grep -n "5.0" docs/COMMANDS.md docs/SCORING.md` finds no stale neutral-default claim for saliency; link anchors resolve (`#...` exist).
   - affected files: docs/CONFIGURATION.md, docs/SCORING.md, docs/COMMANDS.md, docs/{de,es,fr,it,pt,zh}/{CONFIGURATION,SCORING,COMMANDS}.md where present, CHANGELOG.md

8. **Capped end-to-end verification with the REAL repo code.**
   - key decision: harness fidelity. Default: copy the scratch harness
     (C:\Users\laure\AppData\Local\Temp\claude\D--photo-llm\01873979-9070-498f-a0cb-74e72a51ba83\scratchpad\vram12\harness.py)
     to a new file and ADD one patch in its `if cap:` block:
     `torch.cuda.mem_get_info = lambda *a, **k: (max(0, int((CAP_GIB - (pynvml_used_gib() - base)) * 2**30)), int(CAP_GIB*2**30))`
     (free = cap - (NVML used - baseline), N2: NOT cap - `memory_reserved`, which ignores ORT and
     the CUDA context), otherwise the new sizing sees the physical 16 GiB. Keep
     `set_per_process_memory_fraction`, the ORT `gpu_mem_limit` patch and the `detect_vram` patch.
     NO scratch patch of the saliency batch size or table. Sanity-check that
     the stand-in agrees with `torch.cuda.mem_get_info` uncapped (within 5%).
   - run order (each on a fresh copy of the DB per the no-prod-DB rule; test photos only,
     never delete any photo file; never touch photo_scores_pro.db):
     1. uncapped control: `harness.py <rundir> <args...>` without `--cap`.
     2. capped: same args with `--cap`.
     3. rerun the uncapped control to show repeatability.
   - PASS (all must hold, scripted in a compare script, exit code printed):
     exit 0; zero lines matching `OutOfMemory|out of memory|Tried to allocate|CUDA error` in the log;
     `nvml_peak_minus_base <= 11.7`; photos tagged 49/49; embedding dim 1152 on every row;
     Q-ReAlign column non-NULL where the control's is; capped `subject_bbox IS NOT NULL` count
     within |capped - control| <= 1 and both >= 43 of 49 (baseline 44; the allowance exists
     because the batch size now differs between runs - uncapped 16 GiB yields batch 5, not 8 -
     and fp16 batch-composition changes can flip a borderline mask); the four REAL columns have
     no NULL where the control has values; the Step 5 residue assertions hold;
     face row count equal; a "saliency batch reduced" WARNING is acceptable but any ERROR line
     from saliency fails the run. Also record per-pass peaks and the batch size chosen
     (DEBUG line at saliency_scorer.py:~243) and the post-unload residuals from Step 5.
   - verify: the compare script's JSON summary (both control runs + capped run) pasted in the PR;
     a failure of any criterion blocks completion.
   - affected files: scratch harness copy only (not committed)

## Final verification gate
- Steps 1-7 unit/lint gate green (Step 6), docs diff reviewed across all seven languages (Step 7).
- Step 8 capped run passes every criterion with unmodified repo code. The uncapped control is
  NO LONGER batch 8: the formula gives 5 at 16 GiB free (8 only at >= ~24 GiB). Its pass
  criterion is bbox count within 1 of the pre-change baseline (44), >= 43 of 49, 1152-dim,
  49/49 tags - a no-regression check, not byte equality.
- `git status --porcelain` shows only the files listed above; no new untracked repo files
  besides tests. Run `/review-all:review-all` on the diff before commit.
- Browser check (project workflow): start the viewer on the test DB, open a photo whose
  saliency columns are NULL; confirm the gallery, the critique dialog and the social-crop
  menu render (bbox NULL -> faces/center fallback) with no console errors.

## Adversarial review
Reviewer findings and verdicts (orchestrator-adjudicated). Spec sections changed are named.

| ID | Finding | Verdict | How the spec now addresses it |
|----|---------|---------|-------------------------------|
| B1 | A failed saliency photo would be saved as neutral/NULL over stored values on a restricted save | accepted | W2/W3: `SALIENCY_FAILED_KEY` + `SALIENCY_PASS_COLUMNS` (5 cols) skipped in `_save_results_restricted` like `FACE_FAILED_KEY`/`FACE_PASS_COLUMNS`; NULL only for never-scored rows; S5 and Step 3 test (d) rewritten |
| B2 | `--recompute-saliency` UPDATE would overwrite good values on failure and count it as updated | accepted | W7: skip UPDATE for failed results, exclude from `updated`; Step 3 test (f) proves `--force` keeps existing values |
| B3 | Formula gives 5 at 16 GiB free, so "control still batch 8" and "bbox == 44" are wrong | accepted | Step 2 test (a): 24 GiB -> 8, 16 GiB -> 5; Final gate and Step 8 criterion: |capped - control| <= 1 and both >= 43/49, reason stated |
| G1 | topiq_nr_face / insightface are CPU-cacheable; unload path may not be the one in use | accepted-modified | Step 5 rewritten: reproduce under `keep_in_ram='never'` and default, fix only the path that leaves residue; insightface ORT arena is a deliberate cache trade-off, accounted for by a measured planner-capacity constant (or nothing if `unload` releases it) |
| G2 | SigLIP is loaded at three sites; kwarg name | accepted | Step 4: one shared loader helper for model_manager, processing/scorer.py:878-887, scripts/compute_aesthetic_clip.py:61-63; `dtype=` (transformers 5.15.0) |
| G3 | A single image that still OOMs at batch 1 must not fail the whole call | accepted | Step 3: `None` for that image only, aligned masks, `score_batch` fails only that photo; mixed-batch test (b2) |
| G4 | Retry inside `except` keeps the traceback and the activations alive | accepted | Step 3: flag, drop the exception reference, `empty_cache` + retry outside the handler, with a comment saying why |
| G5 | `mem_get_info` default device may be wrong | accepted | Step 2: pass `self.device` to `mem_get_info`, `memory_reserved`, `memory_allocated`; test asserts it |
| N1 | tests/test_model_manager.py:155 asserts topiq == 2; `get_model_vram` annotated int | accepted | Step 1 updates the assertion to 2.5 and the annotation to float |
| N2 | WARNING must not embed exception text; harness stand-in must not use `memory_reserved` | accepted | Step 3 WARNING text; Step 8 stand-in = cap - (NVML used - baseline) |
| N3 | The "document the floor" fallback in Step 5 cannot fail | accepted-modified | Removed: Step 5 verify asserts post-pass NVML residue <= budgeted value, step not done otherwise |
| N4 | Step 7 claimed some languages lack COMMANDS/SCORING | accepted | Step 7: all six languages have the three files; COMMANDS.md:61 "~2 GB" and translations included |
| N5 | Retry on MPS; keep licence re-raise | accepted | Step 3: retry applies on MPS via `is_out_of_memory_error`; `licensing.reraise_if_refused()` kept (Not changing; tests/test_licensing_gate.py:436) |

Checks the reviewer ran: requirement coverage, omitted consumers, can-fail verify checks, error
paths, contradictions with the tree, step-to-step contracts. Remaining UNVERIFIED items are
listed in Open questions / Step 3 (R6, R10, R11) and Step 5 (a).
