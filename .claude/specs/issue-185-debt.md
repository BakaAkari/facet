# Issue #185 debt items: implementation spec

Branch `feat/commercial-mode`, HEAD `e9aa4344`. Read-only research; nothing in the tree was changed.
Source of the debt list: D36 "Left open (appendix DEBT)" plus D33 ("No head-pose gate (DEBT)").
`UNVERIFIED` marks anything not confirmed by reading code or running a command.

Docs rule: every doc below exists in 7 languages (`docs/` and `docs/{de,es,fr,it,pt,zh}/`). A prose change in
one language is a change in seven; the translations are edited by hand (see `.claude/patterns/i18n-sync.md`).
Config keys must never gain legacy aliases (CLAUDE.md "No backward-compatibility fallbacks").

---------------------------------------------------------------------------------------------------------

## Item 1. 27 hand-placed `except LicenceError: raise` guards

### Current sites (27 handlers, 8 files)
`grep -rn "except LicenceError"`:
- `analyzers/face.py:75`; `faces/processor.py:139`
- `models/model_manager.py:334, 604, 716, 747, 782, 811, 887, 905, 930`
- `models/samp_net.py:450, 916, 1144`
- `processing/batch_processor.py:293, 493`
- `processing/multi_pass.py:543, 882, 931, 962, 981, 1024, 1060`
- `processing/scorer.py:837, 858, 1581, 2867`
- `facet.py:1697` (this one is `except licensing.LicenceError:` in `_run_scan`'s dry-run loop)

Every one has the shape
```python
        except LicenceError:
            raise
        except Exception as e:
            logger.warning("InsightFace not available: %s", e)      # analyzers/face.py:75-78
```
Definition: `licensing.py:36  class LicenceError(RuntimeError): """A component was requested that the active licensing policy forbids."""`; raised only by `licensing.require()` (`licensing.py:297`) and `processing/scorer.py:2221`.

### The AST test
`tests/test_licensing_gate.py:409-500`: `SWALLOW_SITES` (25 `(file, function)` pairs), `SWALLOW_MARKERS` (restrict two big functions to
try blocks mentioning `SAMPNetScorer`/`create_composition_analyzer`, `process_single_photo`), `_unprotected_broad_handlers()` walks each
function's `ast.Try` nodes; for each handler it sets `guarded` when an EARLIER handler names `LicenceError` and its body contains a bare `raise`,
and records a `bad` entry for any handler naming `Exception`/`BaseException`/bare that comes with `guarded` False.
`TestSwallowers` has three tests: per-site `bad == []`, "the scan finds broad handlers" (total >= 15), and `test_scan_detects_a_missing_guard`.
The test is order-dependent (guard must precede) and function-list driven: a NEW function with a broad handler around a restricted load is
invisible until someone adds it to `SWALLOW_SITES` by hand. That is the real weakness of the current mechanism.

### Option evaluated: make LicenceError a BaseException (REJECTED)
`class LicenceError(BaseException)` would remove the need for any guard, but:
- Sites that catch `BaseException` or bare `except:` in non-vendored code (would now see it; all re-raise, so harmless):
  `api/config.py:1550, 1604` (stats single-flight; publishes then re-raises), `api/routers/webdav.py:243` (unlink then raise),
  `db/maintenance.py:212` (cleanup then raise), `sync/immich.py:317`, plus the `except BaseException: conn.rollback(); raise` context manager.
  Vendored `build-tools/**/wfastcgi.py` and `client/node_modules` hits are irrelevant.
- The danger is the opposite direction: every `except Exception` that is a deliberate containment boundary would stop containing it:
  worker threads in `processing/batch_processor.py` and `faces/processor.py` (the thread dies, producer/consumer queues then block: a hang, not an error),
  Starlette/uvicorn request handling (`ServerErrorMiddleware` catches `Exception`; a BaseException escapes to the server loop), `tqdm`, `contextlib.suppress(Exception)`.
  Python reserves BaseException for `SystemExit`/`KeyboardInterrupt`/`GeneratorExit`; a licence refusal is an ordinary operational error.
- It is also RuntimeError today, so ~17 `except RuntimeError` sites could swallow it; I read the four in the load paths
  (`processing/multi_pass.py:780, 794`, `models/aesthetic_head.py:77`, `models/vlm_tagger.py:412/420`): all re-raise or re-wrap. No present bug, but the AST test does not cover them.
- Explicit catchers of LicenceError for reporting: none. `facet.py:1697` only re-raises; the CLI relies on an uncaught traceback or the preflight `parser.error` at `facet.py:2232`. How an uncaught LicenceError is rendered to the operator is UNVERIFIED.

### Recommendation (ONE approach): keep `LicenceError(RuntimeError)`, replace the two-line guard with one shared helper called first in every broad handler
Add to `licensing.py`:
```python
def reraise_if_refused() -> None:
    """Call as the FIRST statement of a broad ``except Exception`` handler: re-raises the active
    exception when it is, or is caused/contextualised by, a LicenceError."""
    exc = sys.exc_info()[1]
    seen = set()
    while exc is not None and id(exc) not in seen:
        if isinstance(exc, LicenceError):
            raise exc
        seen.add(id(exc))
        exc = exc.__cause__ or exc.__context__
```
Each site becomes
```python
        except Exception as e:
            licensing.reraise_if_refused()
            logger.warning(...)
```
Justification: (a) one definition of "a refusal passes through", including refusals wrapped by `raise X from LicenceError` (the 27 literal guards miss those, a latent
bypass); (b) order independence (no earlier-handler requirement); (c) mechanical, no semantic change to non-licence paths; (d) the BaseException route is a behaviour change with hang risk (above).
It does NOT remove the manual step, and that is honest: Python cannot make an `except Exception` skip a subclass. The gain is correctness and a stronger test.

### Proposed change
1. `licensing.py`: add `reraise_if_refused()` (+ `import sys`). Chosen default: follow `__cause__`/`__context__` chain, depth-guarded as above.
2. Replace the 27 two-line guards (the `except LicenceError:` + `raise` lines, and the `from licensing import LicenceError` imports that become unused in `analyzers/face.py`, `faces/processor.py`, `models/model_manager.py`, `models/samp_net.py`, `processing/batch_processor.py`, `processing/multi_pass.py`, `processing/scorer.py`; `scorer.py:2221` still raises it so it keeps its import). Sites whose handler already is `except Exception as e:` get the call on the first line. `facet.py:1697` the same.
   Where a handler is the only catcher (e.g. `multi_pass.py:543` is followed by `except Exception as ex:` that distinguishes supplementary vs required) the call goes first, before that branching.
3. `tests/test_licensing_gate.py`: rewrite `_unprotected_broad_handlers` so a broad handler (`Exception`/`BaseException`/bare) is guarded iff its FIRST body statement is an `ast.Expr` calling `reraise_if_refused` / `licensing.reraise_if_refused` (also accept the legacy shape ONLY if you keep any, but there should be none: no compat). Replace the per-function `SWALLOW_SITES` list by a walk of the whole file set so a new function cannot hide: scan every `.py` under `models/ processing/ analyzers/ faces/ facet.py` for `ast.Try` whose body contains a call to a restricted loader or `licensing.require(...)`, using the existing marker idea (`SWALLOW_MARKERS`) generalised to names `{'require', 'loader_allowed'-guarded loads, 'build_face_analyzer', 'build_permissive_face_analyzer', 'create_samp_scorer', '_load_*'}`. Keep the explicit site list as a floor (`assert set(listed) <= found`) so the scan cannot silently find zero.
4. Add `tests/test_licensing.py::test_reraise_if_refused_*`: re-raises a direct LicenceError, a `RuntimeError` raised `from LicenceError`, and a context-chained one; passes through an unrelated `ValueError`; cycles in `__context__` terminate.

### verify (can fail)
- Fault injection (retype by hand, never `git checkout`; then `git status --porcelain <file>` must be empty): delete the `licensing.reraise_if_refused()` line in `models/model_manager.py::_load_insightface`; `venv/bin/python -m pytest tests/test_licensing_gate.py -k Swallowers -q` MUST fail on that site. Also add a fresh function with a bare broad handler around a `licensing.require(...)` in a scratch copy: the whole-tree scan must fail.
- Behavioural: existing `test_licensing_gate.py:854-867` (`side_effect=LicenceError(...)` + `pytest.raises`) still passes.
- `grep -rn "except LicenceError" --include=*.py . | grep -v "tests/\|conda"` returns nothing.

### Affected files
Code: the 8 files above + `licensing.py`. Tests: `tests/test_licensing_gate.py`, `tests/test_licensing.py`. Docs: none user-facing (`.claude/patterns/` has no entry; optionally mention in D-log). 

### Risks
A site where the guard was deliberately AFTER a narrower handler (none found; all 27 are the first handler). `sys.exc_info()` is empty when called outside an except block: the helper then does nothing (document it). Chain-following could re-raise a LicenceError that was intentionally logged-and-converted (grep found none).

---------------------------------------------------------------------------------------------------------

## Item 2. Per-batch centroid recompute for named persons on rescan

### Current sites
`db/person_carry_over.py:137 reattach_named_persons`, lines 172-177:
```python
    touched = {entry[2] for entry in snapshot}
    for person_id in touched:
        recompute_person_centroid(conn, person_id)
    refresh_person_counts(conn, touched | set(extra_person_ids))
```
`replace_photo_faces` (181-202) calls it once per call. Callers of `replace_photo_faces` (only two):
- `processing/scorer.py:2589` inside `save_photos_batch` (reached from `processing/batch_processor.py:563, 592, 721, 750` and `processing/multi_pass.py:1332`)
- `processing/scorer.py:2662` inside `save_photos_partial` (`processing/multi_pass.py:1374`, `refresh_faces` only)
Other `reattach_named_persons` caller: `replay_persisted_snapshot` (`person_carry_over.py:245`, used by `faces/clusterer.py:774`) — runs once per clustering, keep as is.
`recompute_person_centroid` (`db/person_centroids.py`) loads every face embedding BLOB of the person and averages: O(faces of person) per call. A named person with 2,000 faces re-read for each of ~hundreds of batches.

### Proposed change
- `reattach_named_persons(..., recompute_centroids=True)` gets a keyword. `replace_photo_faces(..., defer_centroids=None)` takes an optional `set` collector: when given, the function adds `touched` to it and SKIPS `recompute_person_centroid` but STILL calls `refresh_person_counts` (counts and representative must stay consistent per batch, they are cheap and the representative must not dangle).
- `Facet` (scorer) owns `self._pending_centroid_persons = set()`; `save_photos_batch`/`save_photos_partial` pass it. New `Facet.flush_person_centroids()` opens one connection, calls `recompute_person_centroid` per id, commits, clears the set.
- Call `flush_person_centroids()` in the `finally` blocks that end a scan: `processing/multi_pass.py:474` (`process_directory`), `processing/batch_processor.py:604` (`process_files`) and `:762` (`process_stream`). In `finally` so a Ctrl-C or exception still flushes. Chosen default: defer ON for scans; the default argument stays "recompute now" so the clusterer replay and API callers are unchanged.
- Person with no faces left: deferred flush gives NULL centroid exactly as today (D36 "intended").

### Crash behaviour / readers
- Kill -9 / power loss mid-scan: faces, counts, representatives are already committed per batch; only centroids are stale (they still describe the pre-scan face set, a valid unit vector). Acceptable: centroids are advisory (clusterer treats a stale or NULL one as "match by history", see `faces/clusterer.py:318-335`; merge suggestions use it as a similarity hint). The next clustering run or the next scan flush recomputes. To make that self-healing without a persisted set: `--cluster-faces-incremental` already calls `recompute_person_centroid` for preserved persons (`faces/clusterer.py:19` import). UNVERIFIED that every incremental path recomputes all named persons, not just touched ones; the builder must confirm or add a `--repair-person-centroids` pass.
- Readers during a scan: viewer `/api/merge_suggestions` (`api/routers/merge_suggestions.py:72`, `load_person_centroids`) and the clusterer. Both tolerate stale/NULL. During a long scan a merge-suggestion for a re-attached person is computed from the older centroid: cosmetic.
- Model switch (ArcFace to SFace re-extract): `centroid_model` is set by recompute; between batches and the flush a person's `centroid_model` is the old model while some faces are new-model. `person_centroid_model()` reads face tags, not the column, so the final flush is correct; `load_person_centroids(embedding_model=...)` filters by the stored column, so mid-scan a re-extracted person is simply absent from the new-model list. Fine.

### verify
New test in `tests/test_face_models.py` (the existing carry-over tests live there): two batches re-attach the same named person; spy `recompute_person_centroid` via `monkeypatch.setattr(db.person_carry_over, 'recompute_person_centroid', counting)`; assert it is called 0 times during `save_photos_batch` x2 and exactly 1 time after `flush_person_centroids()`, and that the final `persons.centroid` bytes equal the per-batch-recompute result (compute both on copies). Mutation check: remove the flush call from `process_directory`'s `finally`; a scan-level test (`tests/test_multi_pass.py` style with a stub scorer) must fail because centroid stays at the pre-scan value.

### Affected
`db/person_carry_over.py`, `processing/scorer.py`, `processing/multi_pass.py`, `processing/batch_processor.py`. Tests: `tests/test_face_models.py`, `tests/test_multi_pass.py`. Docs: `docs/FACE_RECOGNITION.md` (the re-extract/carry-over paragraph, 7 languages) one sentence "centroids are rebuilt once at the end of a scan".

### Risks
A scan aborted before `finally` (hard kill) leaves stale centroids (above). Threading: `BatchProcessor` saves from the GPU thread; the set is only touched there and flushed after the thread join (`process_files` joins at 604-606 before `finally` completes: put the flush AFTER the joins).

---------------------------------------------------------------------------------------------------------

## Item 3. OFIQ first-use fetch has no socket timeout

### Current sites
`analyzers/face_quality_ofiq.py:44-51 _fetch_models` calls `download_weights(OFIQ_CONFIG_URL, ...)` and `download_weights(OFIQ_MODELS_URL, archive, ...)` (about 400 MB) under `_fetch_lock`;
`_get_engine` (78-91) wraps it in `except Exception` and sets `_setup_failed`.
Root cause is not OFIQ-specific: `models/weights.py:57  urllib.request.urlretrieve(url, temp_path)` has NO timeout, and `download_weights` is the single funnel for YuNet, SFace, MediaPipe (`analyzers/face_blendshapes.py:100`), BlazeFace (`optimization/model_eval.py:577`), `models/aesthetic_head.py:71`, SAMP-Net (`models/samp_net.py:913`), `analyzers/face_permissive.py:123`, OFIQ. A stalled connection holds `_fetch_lock` forever, blocking every face worker thread (they all wait on the lock).
Also bypassing the funnel: `models/samp_net.py:444  urllib.request.urlretrieve(U2NETP_WEIGHTS_URL, self.weights_path)` (non-atomic, no sha, no timeout).
The task text says "consistent with models/weights.download_weights": there is currently NO timeout there to be consistent with, so the fix goes IN `download_weights`, and OFIQ inherits it.

### Proposed change
In `models/weights.py` add `DOWNLOAD_TIMEOUT_SECONDS = 60` (per socket operation, NOT total; a 400 MB fetch on a slow link is legitimate) and replace `urlretrieve` by
```python
with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as resp, open(temp_path, 'wb') as out:
    shutil.copyfileobj(resp, out, 1 << 20)
```
Chosen default 60 s (matches the order of the repo's other fetches: `api/updates.py` `REQUEST_TIMEOUT_SECONDS`, `models/vlm_backend.py`, plugins use 10 s; weights are big so 60). No config key (YAGNI; a module constant). `samp_net.py:444` switches to `download_weights(U2NETP_WEIGHTS_URL, Path(self.weights_path))` (no sha: none is pinned for u2netp; UNVERIFIED whether one exists upstream).
On timeout `socket.timeout`/`URLError` propagates: `_get_engine` already catches and logs "OFIQ setup failed" and sets `_setup_failed`, so the column stays NULL (D35: weight-fetch failure aborts the preflight for YuNet/SFace; OFIQ stays optional).

### verify
`tests/test_model_manager.py:572-590` and `tests/test_face_blendshapes.py:169-191` patch `urllib.request.urlretrieve`; `tests/test_licensing_gate.py:323` installs a tripwire on `urlretrieve`. All must be rewritten to patch `urllib.request.urlopen` with a fake returning a BytesIO context manager. New test: fake `urlopen` that records `timeout=` and assert it equals `weights.DOWNLOAD_TIMEOUT_SECONDS`; a fake raising `socket.timeout` leaves no `.part` file and no destination (the `finally` already removes temp). Mutation: revert to `urlretrieve`; the recording test fails.

### Affected
`models/weights.py`, `models/samp_net.py`, tests `test_model_manager.py`, `test_face_blendshapes.py`, `test_licensing_gate.py` (tripwire at 323 must also trip on `urlopen`), `test_face_quality_ofiq.py` (mocks `download_weights`, unchanged). Docs: none.
Risk: mocks elsewhere that import `urlretrieve` by name (grep showed only the files above). A server that sends data slower than 60 s per chunk never trips; acceptable.

---------------------------------------------------------------------------------------------------------

## Item 4. Merge-suggestions slider hidden until first response

### Current sites
`client/src/app/features/persons/merge-suggestions.component.ts`:
- line 62 `@if (threshold(); as value) {` wraps the whole `mat-slider` (63-70) and the percentage label; `threshold = signal<number | null>(null)` (196).
- `loadSuggestions()` (230-246): sends no threshold until the user moves the slider (`userThreshold === null`), then `if (this.userThreshold === null) this.threshold.set(res.threshold);` (240). Slider range `[min]=0.3 [max]=0.9 [step]=0.05` (63).
Server: `api/routers/merge_suggestions.py:122  threshold = server_scoring_config().get_face_clustering_settings()['merge_threshold']` (active model, `config/scoring_config.py:883-912`, ARCFACE 0.6 / SFACE 0.65).
Client has no source for the default before the first response: `store.config()` (`/api/config`, `api/routers/gallery.py:1734`, typed `GalleryStore` config at `client/src/app/features/gallery/gallery.store.ts:185`) carries `features.*` but no face threshold.

### Proposed change
1. Server: add `face_merge_threshold: float` to `/api/config` (set next to `'unavailable_metrics': unavailable_columns()` at `api/routers/gallery.py:1772`; type in `api/models/discovery.py` beside `unavailable_metrics` at line 102). Value = `server_scoring_config().get_face_clustering_settings()['merge_threshold']`; wrap in try/except `ValueError` (a model with no entry raises) and omit the field then. Regenerate `client/src/app/core/api/schema.d.ts` (`cd client && npm run gen:api`; CI fails if stale).
2. Client: `GalleryStore` config type gets `face_merge_threshold?: number`; the component injects the store (it is already provided at root for the persons page; confirm, UNVERIFIED) and initialises `threshold` from `store.config()?.face_merge_threshold ?? null` in `ngOnInit` before `await this.loadSuggestions()`. The slider renders immediately; `loadSuggestions` still overwrites with `res.threshold` when `userThreshold === null` (same value, so no visible jump).
   Chosen default if the config field is absent: keep today's behaviour (hidden until response) rather than inventing 0.6, because a hard-coded client constant is exactly the wrong-for-SFace bug D34 fixed.
3. While the first load is in flight the slider is shown enabled but changes only schedule the debounced reload (existing).

### verify
`merge-suggestions.component.spec.ts` 'per-model default threshold' block (285-307) currently asserts `component.threshold()` is `null` before the response (line 291): change to assert it equals the store's `face_merge_threshold` (0.65 in the SFace case) BEFORE the mocked `api.get` resolves (use a deferred promise), and that the first request still sends no `threshold`. Mutation: drop the seeding line; the new assertion fails. Server: `tests/test_api_contract.py` + a test in `tests/test_merge_suggestions.py` that `/api/config` returns the active model's threshold (SFace commercial vs ArcFace default). Gates: `cd client && npm run test && npx tsc --noEmit -p tsconfig.app.json && npx tsc --noEmit -p tsconfig.spec.json && npx ng lint`, `npm run gen:api`.

### Affected
`api/routers/gallery.py`, `api/models/discovery.py`, `client/src/app/core/api/schema.d.ts`, `client/src/app/features/gallery/gallery.store.ts`, `client/src/app/features/persons/merge-suggestions.component.ts` (+spec), `tests/test_merge_suggestions.py`, `tests/test_api_contract.py`. Docs: `docs/VIEWER.md` `/api/config` field list only if it enumerates fields (UNVERIFIED). Browser check required (project rule: browser test at end of every change): open `/persons/merge`, throttle network, confirm slider visible before the response.
Risk: `response_model_exclude_unset=True` on `/api/config`; field must be set explicitly. Template rule: no method calls in templates (only the existing `fixed` pipe).

---------------------------------------------------------------------------------------------------------

## Item 5. `--doctor` lists configured, not routed, components in commercial mode

### Current sites
`diagnostics.py:75-123 _components_in_use(cfg, profile_name)` is a hand-rolled mirror of the profile config: adds `profile['aesthetic_model']` (e.g. `topiq`), `insightface`, supplementary pyiqa names, `birefnet`, `qrealign_native` etc., with NO routing. `_report_model_licences` (127-160) then prints each, and for refused ones prints `not permitted in commercial mode ... use <replacement>`. So in commercial mode a 16gb profile shows TOPIQ, TOPIQ IAA, LIQE, InsightFace as warnings although the scan never loads them, and omits what IS loaded (YuNet, SFace, OFIQ, qrealign as primary).
The routing truth lives in `processing/multi_pass.py`: `_select_models` (278-360), `_resolve_selected_quality` (static, 373-396, takes `(config dict, model_manager)`; `model_manager` is only used for `.get_active_profile()`), `_commercial_primary_quality` (406), `_face_pass_name` (402, `'insightface' if licensing.loader_allowed('insightface') else 'faces_permissive'`). `_select_models` needs `self._selected_quality()`, `self.scorer.config.get_extended_iqa_settings()`, `self.model_manager.get_active_profile()` and `self._profile_tagger_fits(vram)`.

### Proposed change
Extract the pure core of `_select_models` into a module-level function in `processing/multi_pass.py` (or a new `processing/routing.py` that multi_pass imports; the new file is cleaner and importable without torch: `multi_pass._ensure_imports()` is lazy so check, UNVERIFIED):
`route_models(config: dict, profile: dict, extended: dict, policy, tagger_fits: Callable[[float], bool]) -> list[str]` returning the same loader names `_select_models` returns today; `_select_models` becomes a thin wrapper (so no behaviour change; the existing `tests/test_multi_pass*.py` selection tests are the guard).
`diagnostics._components_in_use` then: resolve policy (`licensing.resolve_policy(cfg)`), build `ScoringConfig`-free inputs from `cfg` via `config_resolve.load_resolved` (already done in `_report_model_licences`), call `route_models(..., tagger_fits=lambda gb: True)` (doctor reports the configured profile, VRAM fit is reported elsewhere in the doctor VRAM section; chosen default), and map each loader name to a registry id with `licensing.component_for_loader` (`licensing.py:157`; aliases for `clip_aesthetic`, `faces_permissive` need registry entries: `faces_permissive` expands to `yunet`, `sface`, plus `mediapipe_face_landmarker` and `ofiq` when installed/enabled). Keep default-mode output byte-identical for the `insightface` path (add to the golden check).
No model is loaded: only config dicts and `importlib.util.find_spec('ofiq')`.

### verify
`tests/test_diagnostics.py`: with `licensing.mode: commercial` and profile `16gb`, captured doctor lines MUST contain `yunet`, `sface`, `qrealign_native`, and MUST NOT contain `topiq`, `insightface`, `liqe`; default mode output unchanged (snapshot of the id list for each of the four profiles). Mutation: make the wrapper return the configured list; the commercial assertion fails.

### Affected
`diagnostics.py`, `processing/multi_pass.py` (+ optional `processing/routing.py`), `licensing.py` only if a `faces_permissive` loader-to-ids mapping is added there (conflicts with item 1 / 10 edits: same work package). Tests: `tests/test_diagnostics.py`, `tests/test_multi_pass.py`. Docs: `docs/FACE_RECOGNITION.md` line ~487 (doctor warnings paragraph) and `docs/INSTALLATION.md` doctor section, 7 languages, one sentence.
Risk: two routing implementations drifting if the wrapper is not the single source (hence the extraction). Doctor currently tolerates unknown profile (`profile_name is None`): keep.

---------------------------------------------------------------------------------------------------------

## Item 6. No blink head-pose gate for the permissive (MediaPipe) blink

### Current sites
- Landmarker creation: `analyzers/face_blendshapes.py:118-124`
  ```python
  options = vision.FaceLandmarkerOptions(base_options=..., running_mode=vision.RunningMode.IMAGE, num_faces=1,
      output_face_blendshapes=True, output_facial_transformation_matrixes=False,)
  ```
  So the matrix is NOT requested. `analyze_face_crop` (131-162) returns `(eyes_open, smile, names, landmarks)`; `result.face_landmarks` (478 mesh points) is read, `result.facial_transformation_matrixes` is not.
- Pass-through: `analyzers/face_permissive.py:225-257 _mesh_analysis` returns `(eyes_open, smile, names, mesh_in_full_res)`.
- Blink verdict: `analyzers/face_permissive.py:365-372`
  ```python
  mesh_result = self._mesh_analysis(img_cv, bbox)
  ...
  if max(names['eyeBlinkLeft'], names['eyeBlinkRight']) >= self.blink_threshold: any_blink = True
  ```
  and `:403  'is_blink': 1 if any_blink else (0 if blink_known else None)`; `blink_known = False` when any face has no mesh (so NULL = unknown). Threshold `DEFAULT_BLINK_THRESHOLD` 0.55 (D33).
- InsightFace path: `analyzers/face.py:320-323  POSE_BLINK_GATE_DEG = 35.0`; `is_blinking` (475-500) returns **False** (not NULL) when `|yaw|` or `|pitch|` > 35, `_head_turned` (341-360) also falls back to eye-width asymmetry < 0.45 for the eyes-open/smile scores (those become None = "NULL on turned heads", schema comment `db/schema.py:232`). BUT the pose gate in `is_blinking` only works when `enable_3d_landmarks` is True; its default is False (`analyzers/face.py:26`, wired from `face_detection.enable_3d_landmarks` at `models/model_manager.py:70`), so by default InsightFace gates blinks by nothing at all (is_blinking has no asymmetry fallback). Worth a note in the D-log; not changed here.
- Storage: the `faces` table has NO pose columns (`db/schema.py:228-250` has `landmark_2d_106`, `eyes_open_score`, `smile_score`, no yaw/pitch). `photos` has none either. `analyzers/face.py:257-259` sets `detail['pose_yaw'|'pose_pitch'|'pose_roll']` in the in-memory face dict only; never persisted (UNVERIFIED that `face_upsert_row` ignores them: grep found no column).

### Proposed change
Store nothing new (sticky-state rule does not even arise: no new column). Transient only.
1. `face_blendshapes.py`: set `output_facial_transformation_matrixes=True`; `analyze_face_crop` additionally returns `pose = (yaw_deg, pitch_deg, roll_deg) | None` derived from `result.facial_transformation_matrixes[0]` (4x4): take `R = M[:3,:3]`, divide each column by its norm (the matrix includes scale), then `yaw = degrees(atan2(R[0,2], R[2,2]))`, `pitch = degrees(asin(-R[1,2]))`, `roll = degrees(atan2(R[1,0], R[1,1]))`. AXIS CONVENTION AND SIGN ARE UNVERIFIED (MediaPipe's canonical-face-to-camera frame; Y axis may be flipped). Only absolute values are used, so sign is irrelevant, but axis ASSIGNMENT (which angle is yaw vs pitch) must be proven by a unit test with a synthetic rotation matrix and by an empirical check on 5 known profile and 5 known frontal photos before shipping. Fallback if the matrix proves unreliable: geometric yaw proxy from the mesh: ratio of nose-tip (index 1) distances to the two face contour points (234, 454).
   Return arity changes for `analyze_face_crop` (4 to 5 values): update `face_permissive._mesh_analysis`, `optimization/model_eval.py:1202-1213` (`_blink_scorer_factory` uses `res[2]`), and tests `tests/test_face_blendshapes.py`, `tests/test_face_permissive.py`. Prefer returning a small dataclass/dict instead of a longer tuple.
2. `face_permissive.py`: constant `BLINK_POSE_GATE_DEG` (value decided by the measurement below; ship as `None` = no gate until measured). In the loop at 365-372: if `pose is not None and max(abs(yaw), abs(pitch)) > BLINK_POSE_GATE_DEG`, treat the face like a mesh failure for blink purposes: `blink_known = False`, do NOT set `any_blink`. The eyes_open/smile scores for that face: also None when gated (consistent with `_head_turned` semantics); sharpness etc. unchanged. Result: photo `is_blink` is NULL (unknown), never a false 1, never a false 0.
   Config: `face_detection.blendshapes.blink_pose_gate_deg` (default per measurement, `null` = off) in `config/scoring_config.default.json`, `config/scoring_config.schema.json`, `docs/CONFIGURATION.md` (+6 languages). No alias for anything.
3. Default until measured: gate OFF (current behaviour), shipping only the instrumentation.

### Measurement (extend `--evaluate-models blink`, `optimization/model_eval.py`)
Functions: `sample_blink_faces` (1134), `_blink_scorer_factory` (1202), `_run_blink` (1216), `sweep_blink` (1176), `blink_grid` (1159).
- `_blink_scorer_factory.score` returns `(left, right, yaw, pitch)`; `_run_blink` keeps parallel `yaw`, `pitch` arrays (None allowed) and writes `faces: [{path, label, max_blink, mean_blink, yaw, pitch}]` into the report JSON so any later cutoff analysis needs no re-run.
- New report section `pose_gate`: for each cutoff c in grid `{15,20,25,30,35,40,45,60,90}` (degrees, applied to `max(|yaw|,|pitch|)`): share of faces gated, and at the shipped t = 0.55 the library-weighted precision/recall/F1 where gated faces count as unflagged (a gated true positive is a lost recall), library blink rate 0.0074 (read it from the DB, as D33 did: 263/35,386, do not hard-code), negatives reweighted by the same factor as D33.
- PRE-REGISTERED RULE (write it into the D-log before the run): choose the smallest c whose library-weighted F1 at t = 0.55 on split A (item 7) is within 0.01 of the maximum over the grid; adopt a gate only if (i) held-out split B F1 improves over the ungated F1 by >= 0.02 absolute AND (ii) the gated share of faces on B is <= 15%. Otherwise ship NO gate and record the null result. Ties go to the larger c (fewer NULLs).
- Run on a COPY: `D:\photo-llm\photo_scores_eval.db` (never `photo_scores_pro.db`). Command shape: `python facet.py --evaluate-models blink --db D:\photo-llm\photo_scores_eval.db` (exact flag names UNVERIFIED, check `cli_args.py`). Needs the photo files readable (UNVERIFIED whether paths in the eval copy point at NAS `\\Sweet`, usually offline, or local `D:\`). MediaPipe + YuNet are CPU, so no GPU is required; a GPU only speeds the loader.

### verify
Unit: synthetic 4x4 matrices with known yaw/pitch give the right angles and ordering (this test fails if the axes are swapped). `tests/test_face_permissive.py`: with a stub scorer returning yaw 50 and `blink_pose_gate_deg=35`, a face with eyeBlink 0.9 yields `is_blink is None`; with yaw 10 it yields 1; gate `None` yields 1 for yaw 50 (off). Mutation: remove the gate condition; the yaw-50 test fails. Measurement artefact: the report JSON has `pose_gate` and per-face yaw/pitch.

### Affected
`analyzers/face_blendshapes.py`, `analyzers/face_permissive.py`, `optimization/model_eval.py`, `config/scoring_config.default.json`, `config/scoring_config.schema.json`, tests (`test_face_blendshapes.py`, `test_face_permissive.py`, `test_model_eval.py`). Docs: `docs/CONFIGURATION.md` and `docs/FACE_RECOGNITION.md` (blink paragraph near line 250), 7 languages, plus a D37 line in `.claude/specs/issue-185-decisions.md`.
Risks: after enabling the matrix output, a library already scanned has `is_blink` values from the ungated rule; they change only on rescan or `--recompute-blinks` (which for permissive photos needs the mesh and therefore the images; D36 exempts all-SFace photos from the reset). Document.

---------------------------------------------------------------------------------------------------------

## Item 7. Threshold-selection bias in the faces-centroid and blink slots

### Current sites (`optimization/model_eval.py`)
- faces-centroid: `_run_faces_centroid` (1070-1123) builds ONE embedding set, then `centroid_sweep(e, lab, grid, seed)` (994) draws half-centroids over ALL persons with >= 4 faces (`half_centroids` 971) and returns the grid; `choose_centroid_threshold` (1031) and `choose_auto_merge_distance` (1045) then pick the threshold FROM those same rows, and the reported `false_merge_rate` / `same_recall` for the chosen row come from the same pairs. D34's "rests on one pair" is this bias made visible. No CI on the centroid result at all (only `same_p5` etc.).
- blink: `sweep_blink` (1176) picks `best` by max F1 on the sample and the bootstrap CI (`f1_ci_low/high`) is the F1 AT THAT SAME best threshold on resamples of the same data (optimistic: the argmax is re-taken only for `threshold_ci`). D33 then used a different (library-weighted) criterion by hand.
- `DEFAULT_SEED = 42` (line 46); `bootstrap_ci` and `stratified_sample_rows` in `optimization/eval_common.py:163, 21`.

### Proposed change
Add a seeded 50/50 split and report on the held-out half:
- `split_halves(keys, seed, stratify=None) -> (A, B)`: `np.random.default_rng(seed + 7)` permutation; for blink stratify by label so each half keeps the positive share; for centroid split BY PERSON (never by face: a person's faces on both sides would leak identity) stratified by `n_faces` tercile.
- faces-centroid: run `choose_centroid_threshold` / `choose_auto_merge_distance` on persons in A only (ArcFace target rate also from A at the ref threshold), then recompute `centroid_sweep` / `face_distance_sweep` rows on B at the chosen threshold and attach a person-level bootstrap CI (resample B persons, n_boot=1000, same seed) of `same_recall` and `false_merge_rate`. If no grid point on A qualifies, report `chosen: None` (as today).
- blink: `sweep_blink` is run on A for selection; the report's headline metric is `grid` row at A's chosen threshold evaluated on B with a bootstrap CI over B (F1, precision, recall, and the library-weighted F1 with the base-rate read from the DB). Keep the full-sample sweep in the JSON under `"full_sample_selection_biased"` so D33/D34's numbers remain reproducible and are visibly labelled.
- Report JSON gains `selection: {split:'A', n, seed, chosen}` and `report: {split:'B', n, metric, ci_low, ci_high, n_boot}`; the shipped numbers (0.65 / 27.5 / 0.55) are not changed by this item, they are re-validated: if the held-out figures disagree materially, record it in the D-log and DO NOT silently change config.

### verify
`tests/test_model_eval.py`: synthetic embeddings where the best A threshold is poor on B (construct B from a shifted distribution) must report the B number, not A's; assert the split is disjoint, covers all keys, is deterministic per seed, differs across seeds, and (centroid) that no person id appears in both halves. Mutation: evaluate the chosen threshold on A instead of B; the synthetic-shift test fails.

### Affected
`optimization/model_eval.py` (+`optimization/eval_common.py` if the split helper lives there), `tests/test_model_eval.py`. Docs: the evaluation paragraph of `docs/COMMANDS.md` for `--evaluate-models` only if it describes the report shape (UNVERIFIED); a D-log line.
Risk: halving shrinks the sample (blink: 446 faces to 223 per half, about 108 positives, with the library-weighted estimate resting on single-digit FP counts): CIs will be wide; that is the honest result. Same package/file as items 6 and 12.

---------------------------------------------------------------------------------------------------------

## Item 8. "saliency ignores stored eyes scores"

### Current code (the earlier note's `saliency.py:142` is `api/routers/saliency.py:136-157`, `api_face_markers`)
```python
rows = conn.execute("SELECT bbox_x1, bbox_y1, bbox_x2, bbox_y2, landmark_2d_106 FROM faces WHERE photo_path = ? ORDER BY face_index", ...)
...
blob = r["landmark_2d_106"]
if blob is not None:
    lm = np.frombuffer(blob, dtype=np.float32).reshape(106, 2)
    eyes_score = FaceAnalyzer.compute_eyes_open_score(lm)          # recomputed from landmarks, line 143
    ... eye_points from lm[LEFT_EYE_INDICES]/[RIGHT_EYE_INDICES]
faces.append({"bbox": bbox, "eyes": eye_points, "eyes_open_score": eyes_score,
              "is_blink": eyes_score is not None and eyes_score <= FaceAnalyzer.EYES_CLOSED_MAX})   # 157
```
Meaning: the viewer's face-marker overlay never reads the stored `faces.eyes_open_score`. (a) For SFace/permissive faces `landmark_2d_106` is NULL (`face_permissive.py` writes `'landmark_2d_106': None`), so `eyes_open_score` is None and `is_blink` False for every commercial-mode face, although the stored column holds the MediaPipe-derived score (0-10, same scale and same closed cutoff `EYES_CLOSED_MAX = 4.0`, `analyzers/face.py:301`; `face_blendshapes.py:59-61`). (b) For InsightFace faces scanned with blendshapes ON, the stored score is the appearance-based one but the overlay shows the landmark-geometry one: two numbers for one eye. The sibling endpoint already does it right: `api/routers/burst_culling.py:1250-1296` selects `eyes_open_score` and falls back to the landmark computation only when the stored value is NULL.

### Proposed change
In `api_face_markers`: add `eyes_open_score` to the SELECT; `eyes_score = r["eyes_open_score"]` when not NULL, else the landmark-derived value when a blob exists (same precedence as burst_culling); keep eye POINTS from landmarks (when the blob exists) and return `eyes: []` for SFace (no stored eye positions; the mesh is not persisted; acceptable and already the contract for a bbox-less face). `is_blink` derived from the chosen score. Note: only the MediaPipe `is_blink` differs from "eyes_open <= 4.0" (0.55 vs 0.6 cutoffs, D33); the overlay's per-face blink is a score-based display and stays so. Chosen default: stored first, landmark fallback.

### verify
`tests/test_saliency.py` (face_markers cases at 122-175): add a row with `landmark_2d_106 = NULL`, `eyes_open_score = 2.0`: response has `eyes_open_score == 2.0`, `is_blink is True`, `eyes == []`; and a row with a blob AND stored 8.0 where the landmarks compute 3.0: response reports 8.0. Mutation: revert to blob-only; both fail. Per CLAUDE.md the seeding must use `db.schema.init_database`/`seeded_photos` fixtures, not a hand-rolled table.

### Affected
`api/routers/saliency.py`, `tests/test_saliency.py`. Docs: none (VIEWER.md overlay description unchanged). Client types unchanged (`MediaFaceMarkersResponse`).
Ambiguity note: "saliency" here is the face-marker overlay in the saliency router, not BiRefNet; if the original note meant `analyzers/*saliency*` there is no such file (`ls analyzers` has none).

---------------------------------------------------------------------------------------------------------

## Item 9. face_count = 0 after a face-pass exception

### Current sites
`processing/multi_pass.py:928-944` (`_pass_insightface` except branch; also serves `faces_permissive` since the pass name maps to the same function, UNVERIFIED which handler):
```python
logger.error("Face detection failed for %s: %s", path, e)
results[path]['face_count'] = 0          # 938   <- stores "no face"
results[path]['face_ratio'] = 0
results[path]['is_group_portrait'] = 0
results[path]['isolation_bonus'] = 1.0   # these three are neutral for scoring
results[path]['face_quality'] = None ... face_details = None
```
Everything but `face_count`/`face_ratio`/`is_group_portrait` is already NULL-honest (D36). Then `_save_results` (1228-1330) stores `'face_count': data.get('face_count', 0)` (1256), `'face_ratio': data.get('face_ratio', 0)` (1260), `'is_group_portrait': data.get('is_group_portrait', 0)` (1262); in-memory uses at `:1134` (`detect_silhouette(..., data.get('face_count', 0))`) and `:1145` (`build_scoring_metrics(face_count=...)`). `save_photos_batch` is `INSERT OR REPLACE` of the whole row (`processing/scorer.py:2496-2590`), so a stored previous `face_count` CANNOT be kept by this path; only `face_details: None` keeps the `faces` rows (2589-2592 filter).
`save_photos_partial` (`scorer.py:2628`) writes only `computed_columns`, so a `--pass`/partial run that fails the face pass and leaves the keys out of `computed` keeps the prior stored values; there, check that the except branch does not put `face_count` into the computed keys (UNVERIFIED: `multi_pass.py:1300-1360` builds `computed` from the per-pass result keys; the except branch's explicit `face_count = 0` WOULD be included).
Schema: `face_count INTEGER DEFAULT 0 CHECK (face_count >= 0)` (`db/schema.py:49`): NULL passes the CHECK; the DEFAULT only applies to omitted columns.

### Proposed change
- Except branch: set `face_count = None`, `face_ratio = None`, `is_group_portrait = None` (isolation_bonus stays 1.0, neutral and not user-visible... or None for symmetry; keep 1.0, it feeds the aggregate).
- `_save_results`: store `data.get('face_count')`, `data.get('face_ratio')`, `data.get('is_group_portrait')` with NO default (absent = no face pass produced them = unknown, same reasoning as the existing comment at 1253). BUT the scoring inputs at 1134/1145 must keep a numeric fallback: `data.get('face_count') or 0` so the aggregate and category computation behave as before (NULL face_count is read as 0 by `category_filter` already: `config/category_filter.py:153 (pd.get("face_count") or 0) > 0`, and `scoring_config.py:1252 int(safe_float(m.get('face_count'), 0))`). Result: category/aggregate unchanged, stored column honest.
- Partial path: ensure the except branch's None values are in `computed` so... decision: for the partial path a failed face pass should leave the key OUT of `computed_columns` so the old value survives. Implement by having `_pass_insightface` mark failures in `results[path]['_face_failed'] = True` and `_save_results_partial` skip face keys for flagged rows. (Needs the builder to read 1300-1375 first.)
- `--recompute-average` / `get_face_quality`-style readers: unaffected (they read stored columns, coalescing).

### Readers of face_count NULL (checked)
Safe, coalesce or treat NULL as no-face: `config/category_filter.py:153` (has_face) and numeric filters 138-143 (a `face_count_min` rule then fails with actual None: the photo is not placed in a face category, correct); `scoring_config.py:1252`; `processing/scorer.py:1723` (`row_dict.get('face_count') or 0`), `:1825 or 0`, `:2356 not (row['face_count'] or 0)`; `api/routers/burst_culling.py:355, 425`; `comparison.py:970, 1069`; `utils/selection.py:42`. Need attention (SQL comparisons where NULL silently drops the row): `scorer.py:1954  UPDATE photos SET is_blink = 0 WHERE face_count >= 1 AND ...` (NULL excluded, fine); `config/percentile_normalizer.py:825-827, 1378-1380, 1722-1726` (CASE `face_count > 1` etc.: NULL falls to the ELSE bucket, i.e. counted as "no face" in category stats; line 975 already says `face_count = 0 OR face_count IS NULL`); `api/routers/gallery.py:481` min/max_face_count filter (NULL excluded from any range, correct); `api/routers/stats.py:381 AVG(face_count)` (AVG ignores NULL: the average rises slightly, correct by definition); `api/models/gallery.py:56 face_count: CoercedInt = None` (already Optional); `api/models/albums.py:134 face_count: int` and `api/models/culling.py:33 face_count: int` (NON-optional: UNVERIFIED whether those queries select photos.face_count or person counts; the albums one at line 134 appears to be persons; check). `api/top_picks.py` does not read face_count (uses `COALESCE(face_ratio, 0)`), so a NULL face_ratio is safe there. Top picks and silhouette: `utils/detection.py:26 face_count > 0 or` with None raises TypeError: callers pass `data.get('face_count') or 0` per above.

### verify
`tests/test_multi_pass_pass_failure.py`: make the analyzer raise for one path; assert stored `face_count IS NULL`, `face_ratio IS NULL`, `is_group_portrait IS NULL`, existing `faces` rows kept, `aggregate`/`category` identical to a run where the pass reported 0 faces. Mutation: restore `= 0`; assertion fails. Add a contract assertion in `tests/test_api_contract.py` that a NULL face_count photo serialises as null. A rescan with a working analyzer afterwards restores the integers.

### Affected
`processing/multi_pass.py`, `processing/scorer.py` (only if the partial path needs a guard), tests above. Docs: `docs/FACE_RECOGNITION.md` mid-scan failure sentence if any (D35 wording), CHANGELOG line.
Risk: gallery shows `face_count` null for failed photos until rescan; `CoercedInt` already allows None. Default-mode golden invariant: only the except branch changes.

---------------------------------------------------------------------------------------------------------

## Item 10. The stale `'pro'` Q-ReAlign entry

### Findings
`grep` for a 'pro' VRAM profile entry found none in `ModelManager`/config. The only stale `pro` is the Q-ReAlign SIZE:
- `models/qrealign_native.py:13-17  MODELS = {'mini': 'q-future/Q-ReAlign-Mini-0.8B', 'lite': 'q-future/Q-ReAlign-Lite-4B', 'pro': 'q-future/Q-ReAlign-Pro-9B'}`
- It is unreachable: `config/scoring_config.schema.json:104  "qrealign_size": { "enum": ["mini", "lite"] }`, `config/scoring_config.py:880 """... "mini" | "lite"."""`, `docs/CONFIGURATION.md:1291` ("`mini` (0.8B) or `lite` (4B)"), and no VRAM/RAM budget exists for it (`models/model_manager.py:1136, 1159` size only 'qrealign' = mini; `QREALIGN_LITE_VRAM_GB = 10` at 1181; `_qrealign_is_lite` 1186). A 9B model would need ~20+ GB: it would pass `QReAlignNativeScorer(size='pro')` validation (`qrealign_native.py:~48 if size not in MODELS`) and then be budgeted as Mini. It is also referenced in the registry display name `licensing.py:85  'Q-ReAlign Mini/Lite/Pro (transformers)'` and in all 7 `docs/**/THIRD_PARTY_MODELS.md` (grep "Mini/Lite/Pro"). Licence of the Pro weights was never verified in the registry note (UNVERIFIED).

### Proposed change
Remove `'pro'` from `MODELS`; rename the registry component to `'Q-ReAlign Mini/Lite (transformers)'`; update the 7 THIRD_PARTY_MODELS tables. No alias. `test` guard: a test that `set(MODELS) == set(schema enum for qrealign_size)` and that every size has a budget in `ModelManager` (this prevents the same drift). The registry rendering test (`licensing.render_markdown`) may snapshot the name: update (UNVERIFIED which test).

### verify
New test above fails if someone re-adds `pro` without schema and budget; `grep -rn "Mini/Lite/Pro\|Pro-9B" --include=*.py --include=*.md .` returns nothing outside CHANGELOG.

### Affected
`models/qrealign_native.py`, `licensing.py` (conflicts with item 1: same package), `docs/THIRD_PARTY_MODELS.md` + 6 translations (the table is generated from the registry by `licensing.render_markdown`; regenerate rather than hand-edit if a generator/CI check exists, UNVERIFIED), a test in `tests/test_extended_iqa_wiring.py`.

---------------------------------------------------------------------------------------------------------

## Item 11. mypy ratchet and the branch's extra error

### How it runs
`.github/workflows/ci.yml:102-111` job `lint`: `pip install ruff mypy`, then `python scripts/check_mypy_ratchet.py`, which runs `<interpreter> -m mypy api/` (`DEFAULT_TARGET = "api/"`) and fails if `Found N error` exceeds the integer in `mypy-baseline.txt` (currently `153`). Config in `pyproject.toml [tool.mypy]` (`python_version = "3.12"`, `ignore_missing_imports`). The count depends on the environment (CI has no project dependencies).

### What I ran
`conda/python.exe` has no mypy, so I created a throwaway venv in the scratchpad (`pip install mypy`, mypy 2.4.0; the CI version is UNVERIFIED and differs, so absolute counts differ from CI by 2). Same tool on `git archive master` and on HEAD, `--cache-dir` in the scratchpad, repo untouched:
- master: `Found 155 errors in 32 files`
- HEAD (this branch): `Found 156 errors in 32 files`
Normalised diff (line numbers stripped) shows exactly:
```
processing/scorer.py: error: "Callable[[Facet], Any]" has no attribute "setter"  [attr-defined]    (new)
processing/scorer.py: error: Name "face_analyzer" already defined on line 973  [no-redef]           (new)
processing/scorer.py: error: Generator has incompatible item type "object"; expected "bool"  [misc]  (gone from master)
```
Net +1 (two new, one disappeared because the code around it was rewritten). Against CI's baseline 153 the branch is at 153 + 1 = 154 expected in CI: over by one. UNVERIFIED in CI's exact environment.

### Cause and fix
`processing/scorer.py:973-1019`: the `face_analyzer` property (`@property` at 973) is followed by `preflight_models` (997) and only THEN `@face_analyzer.setter` (1017). mypy requires the setter immediately after the property. Fix: move
```python
    @face_analyzer.setter
    def face_analyzer(self, analyzer):
        self._face_analyzer = analyzer
```
to directly after the property's `return self._face_analyzer` (line 995), before `preflight_models`. I verified on a `git archive HEAD` copy with exactly that move: `Found 154 errors` (1 below master's 155 here), both new errors gone. Then lower `mypy-baseline.txt` by the CI-measured delta (expected 152; the builder must confirm in a CI-like venv: `python -m venv x && x/bin/pip install mypy && x/bin/python scripts/check_mypy_ratchet.py`, per the script docstring), never raise it.

### verify
`<ci-like venv>/python scripts/check_mypy_ratchet.py` exits 0 and prints a count <= baseline; mutation: move the setter back, the count rises by 2 and the ratchet fails (use `--baseline 152`).

### Affected
`processing/scorer.py` (moves 3 lines; same file as items 1, 2, 9: same work package), `mypy-baseline.txt`.

---------------------------------------------------------------------------------------------------------

## Item 12. "python-ofiq not on PyPI" wording

### Current text
`optimization/model_eval.py:539-545 _ofiq_unavailable_message`:
```python
return ("face-quality candidate 'ofiq' skipped: python-ofiq is not installed "
        "(it is not on PyPI; see the commercial install notes)")
```
It is raised as a `ValueError` at `model_eval.py:225`. No test asserts the string (`grep tests` found nothing). It contradicts the docs: `docs/INSTALLATION.md:352` ("is published on PyPI but ships wheels for Linux x86-64 and macOS arm64 only, so it is not part of the extra; install it by hand with `pip install python-ofiq` where a wheel exists"), `docs/FACE_RECOGNITION.md:234` (same, adds "Windows users: WSL2 or Docker"), `analyzers/face_quality_ofiq.py:3-7`, `pyproject.toml:163`. The sentence "see the commercial install notes" is also vague.

### Proposed wording
```python
return ("face-quality candidate 'ofiq' skipped: python-ofiq is not installed. It is on PyPI "
        "(`pip install python-ofiq`) but ships wheels for Linux x86-64 and macOS arm64 only; "
        "on Windows use WSL2 or Docker (docs/FACE_RECOGNITION.md, commercial mode)")
```
Also fix the adjacent `(UNVERIFIED)` message at 541 only if the adapter has since been wired (check `register_candidate('face-quality', ...)`). Sweep: `grep -rn -i "not on pypi" --include=*.py --include=*.md .` returned only this site (excluding CHANGELOG, which I did not scan; builder to check it too).

### verify
Add a test in `tests/test_model_eval.py` that, with `importlib.util.find_spec` patched to return None for `ofiq`, the message contains `pip install python-ofiq` and does NOT contain `not on PyPI`. Mutation: restore the old string; the test fails.

### Affected
`optimization/model_eval.py`, `tests/test_model_eval.py`. Docs: none.

---------------------------------------------------------------------------------------------------------

## Work packages (disjoint file sets)

| WP | Items | Files (exclusive) | GPU / NAS / measurement |
|----|-------|-------------------|--------------------------|
| WP-A "licensing + processing core" | 1, 2, 3, 5, 9, 10, 11 | `licensing.py`, `diagnostics.py`, `models/model_manager.py`, `models/samp_net.py`, `models/weights.py`, `models/qrealign_native.py`, `analyzers/face.py`, `faces/processor.py`, `processing/*.py` (multi_pass, scorer, batch_processor), `db/person_carry_over.py`, `facet.py`, `mypy-baseline.txt`, tests `test_licensing_gate.py`, `test_licensing.py`, `test_model_manager.py`, `test_face_blendshapes.py` (only the `urlretrieve` patches), `test_diagnostics.py`, `test_multi_pass*.py`, `test_face_models.py`, `test_extended_iqa_wiring.py`; docs `FACE_RECOGNITION.md`/`INSTALLATION.md` doctor and carry-over sentences, `THIRD_PARTY_MODELS.md` (7 languages) | None needed. Code + unit tests only. Do items sequentially inside the package (they share `scorer.py`/`multi_pass.py`): 11 first (3-line move), then 1, 9, 2, 5, 3, 10. |
| WP-B "face analyzers + evaluation" | 6, 7, 12 | `analyzers/face_blendshapes.py`, `analyzers/face_permissive.py`, `optimization/model_eval.py`, `optimization/eval_common.py`, config default+schema (`blink_pose_gate_deg`), tests `test_face_permissive.py`, `test_model_eval.py`, `test_face_blendshapes.py` (arity only: conflicts with WP-A's `urlretrieve` edit in the same file: assign that file to WP-A, and WP-B edits only `analyze_face_crop` tests in `test_face_permissive.py`; or serialise), `docs/CONFIGURATION.md` + `FACE_RECOGNITION.md` (7 languages) | Code is CPU-only (MediaPipe, YuNet, SFace run on CPU). The MEASUREMENT RUN (item 6 pose cutoff, item 7 held-out re-validation) needs the photo files of `D:\photo-llm\photo_scores_eval.db` to be readable: UNVERIFIED whether they are local or on the NAS `\\Sweet` (usually offline). GPU optional. Never run against `photo_scores_pro.db`. The pose-cutoff decision follows the pre-registered rule and is made AFTER the code lands. |
| WP-C "viewer" | 4, 8 | `api/routers/gallery.py`, `api/models/discovery.py`, `api/routers/saliency.py`, `client/src/app/**` (schema.d.ts, gallery.store.ts, merge-suggestions component + spec), tests `test_merge_suggestions.py`, `test_api_contract.py`, `test_saliency.py`, `docs/VIEWER.md` if needed | No GPU/NAS. Needs the viewer + a browser check for item 4 (`feedback_browser_test`), against a copied DB, not the prod one. |

Overlap audit: `test_face_blendshapes.py` is the only file touched by two packages (WP-A urlretrieve patches in tests 169-191, WP-B return-arity change): resolve by giving WP-A the `urlretrieve` fixtures and WP-B the `analyze_face_crop` tests inside it, merged by running WP-B after WP-A, or by moving WP-A's patch to `urlopen` in a shared helper first. Everything else is disjoint. `licensing.py` is only in WP-A (items 1, 5, 10). `processing/` is only in WP-A. `optimization/` is only in WP-B.

Run order: WP-A and WP-C in parallel with WP-B's code; the WP-B measurement run is last and writes `.facet_cache/eval/blink-report.json`.

## Commands I ran (all read-only except the scratchpad venv)
See the final message for exit codes.
