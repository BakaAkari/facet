# Issue #185 — commercial-use mode (P2 "gate + swap")

## Context
bsantero (#185) asked for three things: clarity on commercial permissions, the restrictions written down, and a reliable reduced pipeline with no InsightFace weights and no PyIQA code or weights.

Facet is MIT, but two hard dependencies are non-commercial:
- **pyiqa:** PolyForm-NC code, CC BY-NC-SA weights.
- **InsightFace:** the buffalo_l weights are licensed for non-commercial research only.

The 13 decisions locked on 2026-10-13 stand. This plan maps them onto the code at `d7beae3`, on branch `feat/commercial-mode`. Every site below was re-verified at that commit.

Two invariants:
1. **Default mode is unchanged.** It must produce golden aggregates identical to a run before this change.
2. **Commercial mode avoids every restricted component.** With `licensing.mode: commercial`, Facet never imports, queues or downloads pyiqa or insightface, by any path. The same holds for SAMP-Net / U2-Net-P and the LAION CLIP-MLP head unless `allow_unclear` is set. Every slot is routed to a permissive replacement, and there is no silent fallback.

## Findings that shape the design
- **Broad catches swallow load errors.** Almost every load is wrapped in `except Exception → return None` or `available=False`:
  - `model_manager.py:303/670/698/771`, `analyzers/face.py:71`
  - `scorer.py:784/803/2065`, `multi_pass.py:320/476`, `samp_net.py:446/1133`
  - `FaceAnalyzer` never raises; the scan just writes zeros.
  - So `LicenceError` needs an explicit `except LicenceError: raise` placed before each of these catches. A test enumerates the sites.
- **The policy can't be passed as a flag.** The viewer spawns scans without `--config` (`api/routers/scan.py:178`), and `--upgrade-db` re-execs itself (`facet.py:2597`). The policy therefore comes from the **resolved config**:
  - Entry points (`facet.main` after parse at `:2134`, `api.create_app`) set it explicitly.
  - `licensing.policy()` lazily resolves it from `load_resolved` when it was never set, so scripts and children can't bypass it.
  - Tests reset it through a conftest fixture.
- **Restricted-component sites (complete list):**
  - **pyiqa:** `models/pyiqa_scorer.py:_ensure_pyiqa` (only import), called from `model_manager._load_pyiqa`, `scorer.recompute_iqa_from_thumbnails` (`--recompute-iqa`), `facet.py:2747` (`--score-topiq`), `scripts/qrealign_ab.py`.
  - **insightface:** `analyzers/face.py:42` (only import). Its single builder, `model_manager.py:37 build_face_analyzer`, is reached from `_load_insightface`, `scorer.py:910` (lazy property ← `batch_processor.py:345`, dry-run `scorer.py:1301`, `faces/clusterer.py:777`).
  - **SAMP-Net:** `model_manager.py:651`, `scorer.py:790`, `scorer.py:1952`, `samp_net.py:431/905/1110`.
  - **LAION CLIP-MLP head:** loaded eagerly in every non-lightweight `Facet` (`scorer.py:844`) and in `model_manager.py:292`.
- **Queueing:**
  - `multi_pass._select_models` (276-355) appends insightface unconditionally at :352.
  - Required vs supplementary is decided at :457-461.
  - A missing required model leads to `_record_chunk_pass_failure`.
  - `run_single_pass` uses the pass map at 1301-1312.
- **Unknown face values are stored as 0:** `multi_pass._save_results` 1127-1138 and `_compute_aggregates` 1016-1024 default face fields to 0.
- **Aggregation treats NULL as neutral:** `build_metric_vector` (`scorer.py:482`) turns NULL into a neutral 5.0 that keeps its weight. `calculate_aggregate_logic` is at `scorer.py:1157`.
- **Face storage:**
  - `FACES_UPSERT_SQL` / `face_upsert_row` (`db/schema.py:246-267`) never write `embedding_model`.
  - `persons.centroid` carries no model tag.
  - 512 is hard-coded at `faces/clusterer.py:144/421/701` and `api/db_helpers.py:948`.
  - Dot products that don't check dimensions: `clusterer.py:337/719`, `faces/merge_analyzer.py:31/120`, `api/routers/merge_suggestions.py:84-98`, `api/routers/gallery.py:1507-1575`.
  - Readers of the 106-point landmarks: `scorer.py:1754-1907` (the blink reset at :1828 would zero SFace rows), `burst_culling.py:1281`, `saliency.py:142`, `analyzers/skin_tone.py:167/208`, and `face.py:178` (a `hasattr` bug).
- **Reuse:**
  - `models/weights.download_weights(url, dest, sha256)`: atomic and SHA-checked. Used for YuNet, SFace, and to pin MediaPipe, whose `face_blendshapes.py:98` currently uses an unpinned "latest" URL with a raw urlretrieve.
  - `analyzers/face_blendshapes.BlendshapeScorer` for the MediaPipe pass.
  - `api.config.map_disk_path` (lazy import, precedent `facet.py:935`).
  - `scripts/qrealign_ab.py` helpers: `ro_connection`, `stratified_sample_rows`, partial/resume, `_bound_to_inference_size`, the timing/VRAM block in `score_model`.
  - `optimization/personal_ranker._baseline_accuracy`.
  - The bootstrap pattern in `weight_optimizer.compute_weight_confidence`.
  - `/api/filter_options/metric_ranges`, which already omits all-NULL columns.

## Shared design

**`licensing.py`** (top-level, stdlib only)
- `COMPONENTS`: a registry of `Component(id, kind, code_licence, weights_licence, commercial: yes|no|unclear, source_url, slot, replacement)`.
- `LicenceError(RuntimeError)`.
- `policy()` / `set_policy(mode, allow_unclear)` / `reset_policy()`.
- `allowed(id)`, `require(id)`.
- `unavailable_metrics(cfg)` returns `{'liqe','aesthetic_iaa'}`, plus `'face_quality_iqa'` when OFIQ is not importable. It returns an empty set in default mode.
- `render_markdown(lang)` generates the docs table.

**Three gate layers.** Each one is independently tested.
1. **Queue.** `_select_models`, `run_single_pass`, `Facet.__init__` (CLIP-MLP and SAMP), the `face_analyzer` property and the CLI recompute flags route to replacements or refuse.
2. **Loader.** `ModelManager._load_*` call `require()` first.
3. **Leaf.** `require()` is the first statement in `_ensure_pyiqa`, `FaceAnalyzer.__init__` (before its try), `samp_net` downloads and `load_aesthetic_head`.

**Commercial routing (multi-pass and forced single-pass alike)**

| Slot | Default | Commercial |
|---|---|---|
| aesthetic (16/24gb) | topiq | `qrealign_native` (aesthetic task) → `aesthetic` + `qrealign_score` (one inference) |
| quality_score | topiq | `qrealign_native` quality task |
| aesthetic (legacy/8gb) | clip-mlp | `qrealign_native` Mini (or clip-mlp if `allow_unclear`) |
| supplementary IAA/NR-Face/LIQE | pyiqa | dropped; `face_quality_iqa` from OFIQ if installed |
| composition | samp-net | rule-based CPU (or samp-net if `allow_unclear`) |
| faces | `insightface` pass | `faces_permissive` pass (required) |

## Waves
Each wave is built by a sonnet builder and reviewed by me. I own git. Each wave is one green commit, and nothing is pushed or turned into a PR without you.

**W0 — Evaluation harness** (read-only)
- `optimization/eval_common.py`:
  - Move `ro_connection`, `stratified_sample_rows`, the partial/resume helpers and `bound_to_inference_size` out of `scripts/qrealign_ab.py`. The script imports them back, with no aliases.
  - Add `load_original(path)`: `map_disk_path` → `load_image_from_path`.
  - Add `pairwise_agreement(pairs, scores)`, where a tie counts as 0.5 and a missing score counts against coverage.
  - Add `bootstrap_ci(per_pair, n=2000, seed)`, and `measure()` (latency, peak VRAM via `torch.cuda.max_memory_allocated`, RSS via psutil).
- `optimization/model_eval.py`:
  - Slots: aesthetic, quality, faces-detect, faces-recognize, face-quality, end-to-end.
  - A candidate registry, open for W3/W5/W6 to add entries.
  - Pairs come from decisive `comparisons` (`winner IN ('a','b')`, ties reported separately).
  - Slot level uses the candidate's own score. End-to-end recomputes the aggregate through `calculate_aggregate_logic`, with the candidate's columns substituted.
  - Reference figures (face match-rate and person agreement vs InsightFace) are only allowed in default mode.
  - Output: a JSON report plus a logged table.
- CLI: `--evaluate-models SLOT` and `--candidates`.
  - Register in `cli_args.py` (weight group, next to `--eval-iqa-srcc`) and dispatch in `facet.py` next to `:2315`.
  - Add to `_READ_ONLY_LIBRARY_COMMANDS` (`tests/test_scan.py:1829`).
- **First measurement, before W3 is built:** in default mode, run the aesthetic and end-to-end slots for stored `topiq_score`/`aesthetic` vs Q-ReAlign Mini and Lite (pyiqa path, `task_='aesthetic'` and `'quality'`) on the 515 decisive pairs, using a copy of the production DB (never the original). Nobody has measured Q-ReAlign vs TOPIQ against the user's own picks. The 1.12.0 A/B only compared it with Q-Align and measured correlation with stored columns (SRCC 0.42 vs `topiq_score`), which shows the two disagree but not which is right. The result doesn't change *whether* commercial mode uses Q-ReAlign (it is the only permissive option), but it settles Mini vs Lite, and whether its weights must be re-calibrated, before W3/W4.
- verify:
  - Stub scorers give exactly 1.0 / 0.0 / 0.5.
  - The bootstrap is deterministic per seed.
  - The DB sha256 is unchanged after a run.
  - `tests/test_qrealign_ab.py` stays green with the same count.
  - `docs/COMMANDS.md` ×7 updated.

**W1 — Registry, docs and `--doctor`**
- `licensing.py` registry, covering every row of the verified licence table.
- `scripts/gen_third_party_models.py` writes `docs/THIRD_PARTY_MODELS.md` plus the 6 translations (table generated; prose translated). It includes the training-dataset and no-warranty disclaimer.
- A README licensing notice ×7.
- `diagnostics.run_doctor` gets:
  - a "Model licences" section: the active mode, plus each component in use with its commercial status;
  - a warning when stored `faces.embedding_model` values differ from the active model.
- verify:
  - A registry↔loader test: every `PYIQA_MODELS` key and every `_load_*` name maps to a registry id.
  - Docs equal the generator output.
  - A doctor mismatch test.
  - Each test is proved to fail by fault injection, with the restore checked by `git status --porcelain`.

**W2 — Config block, gate, null-honest writes and packaging**
- Config:
  - Add `licensing: {mode: "default", allow_unclear: false}` to `scoring_config.default.json` and a typed property in `config/scoring_config.schema.json`.
  - Document it in `docs/CONFIGURATION.md` ×7.
  - `ScoringConfig.get_licensing()`.
- Gate: the three layers above, plus `except LicenceError: raise` at every swallower listed. In commercial mode the CLI refuses outright `--score-topiq`, `--recompute-iqa` for pyiqa models, `--recompute-composition-gpu` (unless `allow_unclear`) and `--extract-faces-gpu-*`, which routes to the permissive extractor once W5 lands; until then it refuses.
- Writes:
  - When the face pass did not run, `face_quality`, `eye_sharpness`, `face_sharpness` and the blink/eye fields are stored NULL, while `face_count` and `face_ratio` stay 0 (`_save_results`, `_compute_aggregates`).
  - A full save no longer wipes faces when no face pass ran.
  - A missing aesthetic is stored NULL.
- Packaging:
  - `insightface` → `[faces-insightface]` and `pyiqa` → `[iqa]`; `[commercial]` = mediapipe deps (absl-py, flatbuffers) plus `python-ofiq` (optional, W6); `all` includes both.
  - Split `requirements.txt` and the lock files into base plus `requirements.restricted.txt`.
  - Add a Dockerfile `ARG FACET_LICENSING=default`. With `commercial`, it skips restricted installs and sets the config.
  - Add `install.sh --commercial`, which installs mediapipe with `--no-deps`.
  - Extend `scripts/check_packaging.py` to assert the restricted packages are absent from base deps.
- verify:
  - A **poison test**: `sys.modules['pyiqa'] = sys.modules['insightface'] = Poison` (import raises), then drive every entry path in commercial mode: multi-pass, auto→single-pass, forced `--single-pass`, `--dry-run`, `--pass`, `--retry-failed`, `--recompute-iqa`, `--score-topiq`, `--recompute-composition-gpu`, `--extract-faces-*`, `--upgrade-db` child.
  - A **download tripwire**: monkeypatch `urlretrieve`, `download_weights` and `hf_hub_download` to fail on restricted URLs or repos.
  - A swallower-enumeration test.
  - Default-mode golden aggregates unchanged.

**W3 — Native Q-ReAlign** (`models/qrealign_native.py`)
- Loads with `AutoModelForImageTextToText` plus the processor; the level-token softmax mirrors `pyiqa/archs/qrealign_arch.py:115-139` (LEVELS/WEIGHTS, first token of `" "+word`).
- Tasks: `aesthetic` and `quality`. Sizes: Mini and Lite. Inputs are bounded to 1024 via `eval_common`.
- It replaces the `qrealign` entry in `PYIQA_MODELS` in **both** modes, so `qrealign` no longer needs pyiqa.
- `model_manager` routes `qrealign` and the new `qrealign_quality` to it.
- In commercial mode, multi-pass makes it the required primary model: one aesthetic forward fills `aesthetic` and `qrealign_score`, and the quality forward fills `quality_score`.
- Registered as W0 harness candidates.
- verify:
  - Stub-logit tests of the 5-token softmax.
  - A local parity test against pyiqa within 1e-3, marked `requires_gpu` (skipped in CI; counted).
  - The legacy/CPU path runs Mini on CPU.

**W4 — Renormalisation at scoring time only**
- In `calculate_aggregate_logic`, drop the weights for `licensing.unavailable_metrics(cfg)` and divide by the remaining sum.
- In commercial mode `build_metric_vector` leaves those metrics out instead of using 5.0.
- `get_weights`, snapshots, `weight_optimizer`, `calibrate.py` and the weight editor keep the raw `*_percent` values.
- A mode change alters `version_hash`, so the existing `config_version` drift detection (`percentile_normalizer.py:1206`, `scan_state.py:184`) flags `--recompute-average`. Add the mode to the doctor message.
- verify:
  - Default-mode golden aggregates are byte-identical.
  - In commercial mode the effective weights sum to 1.
  - A snapshot round-trip keeps `liqe_percent`.

**W5 — Permissive face pipeline** (`analyzers/face_permissive.py`)
- Pipeline:
  - `PermissiveFaceAnalyzer` keeps the same `analyze_faces` contract as `FaceAnalyzer`.
  - YuNet (`cv2.FaceDetectorYN`) runs on a downscaled image.
  - For each padded crop, upscaled to at least 192, a MediaPipe landmarker produces the 478-point mesh and blendshapes. `BlendshapeScorer` is extended to return the mesh too.
  - SFace `alignCrop` + `feature` gives a 128-d embedding.
  - YuNet and SFace ONNX files come from opencv_zoo through `download_weights` with SHA pins. The MediaPipe bundle is pinned to a versioned URL with a SHA.
- Per-face output:
  - Eyes-open and blink come from `eyeBlinkL/R`; smile from the blendshapes.
  - Eye sharpness uses mesh eye points; quality uses the YuNet score.
  - `landmark_2d_106=None`; `embedding_model='sface'`.
- Model awareness:
  - `face_upsert_row` gains `embedding_model`, written explicitly by every writer.
  - `ACTIVE_EMBEDDING_MODEL` becomes `active_embedding_model(cfg)`.
  - Every 512 check becomes `expected_dim(model)`.
  - Every dot-product site filters by `embedding_model`.
  - A new `persons.centroid_model` column (schema migration).
  - `face_clustering.thresholds_by_model: {arcface_buffalo_l: {...}, sface: {...}}`, read through `get_face_clustering_settings`. The merge-suggestion default also comes from here.
- Readers:
  - Every 106-point reader skips NULL.
  - The blink reset (`scorer.py:1828`) is scoped to faces that have landmarks.
  - Fix `face.py:178`.
- Re-extraction: before the force-delete, snapshot `(photo_path, bbox, person_id)` for named persons. Re-attach them by one-to-one IoU ≥ 0.5, then recompute the centroids per model.
- `--doctor` and the scan warn when the stored model differs from the active one.
- verify:
  - A mixed 512/128 library never mixes dimensions (clusterer, merge suggestions, similar-person, centroid).
  - A person-carry test.
  - The poison test still passes, now with real faces populated through stub detectors.
  - Uses the `seed_photos_prefix` fixture conventions from `.claude/patterns/test-fixtures.md`.

**W6 — OFIQ optional extra**
- `analyzers/face_quality_ofiq.py` runs a lazy `import ofiq` per face crop in commercial mode and fills `face_quality_iqa`. It fetches models on demand and never bundles them.
- When the module is absent, the column is NULL and `unavailable_metrics` includes it.
- verify: a stub-module test and an absent-module test.

**W7 — Viewer**
- `ViewerConfigResponse.unavailable_metrics: list[str]` (`api/models/discovery.py:82`), filled in `gallery.py:1724 api_config`.
- `api/types._build_sort_options` drops those sorts.
- Client:
  - `gallery.store.ts` ViewerConfig.
  - `gallery-filter-sidebar.component.ts` filters `ADDITIONAL_FILTERS` through a pipe or computed, never a template method.
  - `app.ts RANGE_CHIPS`.
  - The comparison weight editor shows unavailable weights as disabled, with a tooltip.
- i18n ×7 via `.claude/patterns/i18n-sync.md`. Run `npm run gen:api`.
- verify:
  - An API contract test.
  - Vitest, tsc on app and spec, `ng lint`, and a clean `gen:api` diff.
  - A Chrome snapshot of a scratch viewer in commercial mode.

**W8 — Measure and pick the defaults** (needs you)
- Fix the decision rules in `.claude/specs/issue-185-commercial-mode.md` **before** running:
  - Mini vs Lite: Lite only if its agreement CI lower bound beats Mini's point estimate and it fits the 16gb budget.
  - YuNet vs BlazeFace: higher detection match-rate vs InsightFace at equal-or-better latency.
  - SFace thresholds: the grid point maximising person agreement with the InsightFace reference.
- Run on a SQLite-backup copy (`photo_scores_eval.db`) — never on `photo_scores_pro.db` itself — with originals over `\\Sweet\home\Photos` mapped through `viewer.path_mapping`.
- verify: the same seed gives the same numbers, and a test pins the shipped defaults to the report.

**W9 — Wrap-up**
- Docs ×7 (FACE_RECOGNITION, CONFIGURATION, COMMANDS, DEPLOYMENT, README).
- A CHANGELOG entry with the **breaking change**: the base install no longer includes insightface or pyiqa, so use `.[all]`.
- Draft the #185 reply in `.claude/specs/` (not posted).
- `/review-all` over the whole branch diff.

## Final verification
- `venv/bin/python -m pytest tests/ -q` (count the skips), ruff, and the mypy ratchet ≤ 153.
- `scripts/check_packaging.py`, `scripts/audit_i18n.py`, and the docs parity/xlinks tools.
- Client: tsc on app and spec, `ng lint`, `npm run test`, `npm run gen:api` with a clean diff.
- **Commercial end-to-end:** a fresh venv without pyiqa or insightface, scanning the E2E harness in commercial mode. Faces must be populated, `--doctor` clean, and the viewer filters hidden.
- **Default regression:** golden aggregates identical to a pre-change run on the same harness.
- Every fault injection is restored by retyping, then confirmed with `git status --porcelain`.

## Not built
- Legal review of training-dataset licences (disclaimed in the docs).
- An About/licences panel in the viewer.
- dlib, unless W8 shows SFace failing.
- A named-person recognition ground truth.
- Posting the #185 reply.
