# Spec: Viewer API speed on a 130k-photo library, identical results (V1-V9)

Source of truth: master at HEAD `e38e64e0` (1.20.0 plus open-gaps-120 Steps 1-5, committed). Every `file:line` below is
re-based on that commit; `db/schema.py` moved by +27 lines (B5) and `api/routers/stats.py` lost the dead
`categories/metrics` handler (below line 925, so nothing cited here moved). Builders still locate by the quoted text,
not the number. Measurement report: `.claude/.scratch/viewer-speed.md` ("VS report", Appendix A has the raw tables).
Review that shaped this revision: `.claude/.scratch/refute-viewer-api-speed.md` (rulings under "Adversarial review").
The library under test is a 129,737-photo, 16.3 GB DB (82,943 visible under the default hide toggles); the pristine
copy is `D:\facet-scratch\lib_copy.db`.

## Objective + success criteria

Make the viewer API fast on a 130k-photo library. Scope is locked to V1-V9 (V1 token, V2 visibility indexes, V3
category index, V4 `junk_kind=any`, V5 tags dropdown, V6 camera/lens/star/NULLS-last, V7 stats categories index +
warm-up decision, V8 `+Inf f_stop` 500s, V9 capsule 1152-vs-768 crash), plus the review's accepted safety work (index
build safety, shutdown checkpoint, deterministic tie order).

Global criteria (each independently testable):

- G1 Identical results. For the case matrix in "Final verification gate": the count, the hidden-summary row and the
  ordered `path` list of the page are equal before and after, through the real HTTP routes on a COPY of the library,
  for every query whose ORDER BY is total (all gallery routes: `_resolve_order_by` ends in `path ASC`,
  `gallery.py:882`). For the queries whose ORDER BY was NOT total before (album pages `albums.py:188,230`, the two map
  queries `map.py:60,93`, the per-category top camera/lens window `stats.py:318,331`) the claim is narrowed to
  IDENTICAL SETS: this spec makes them total (Step 10), so their tied rows now order deterministically and may differ
  from what the old plan happened to return. `/api/filter_options/tags` returns the same 89 `(tag, count)` pairs.
  `/api/stats/categories` rows equal the old statement's rows on every field except `avg_f_stop` of groups that
  contain a non-finite row, and `top_camera`/`top_lens` where the top count ties.
- G2 No contract change. No response model is edited, so `npm run gen:api` produces no diff, `tests/test_api_contract.py`
  stays green, and no existing test assertion changes (disclosed per file in the final report).
- G3 Speed, measured on cache MISSES. Each row of the timing table meets its acceptance column. A request served from
  the count/summary cache proves nothing about an index, so every timing is taken twice: HIT p50 (live HTTP) and MISS
  p50 (in-process harness that clears `api.config._count_cache` before each rep, "bench_miss.py" in the gate).
  Acceptance applies to both columns where both are given.
- G4 Bounded write cost. Total new index size <= 125 MB (estimate 106.9 MB for the final eight; the prototype sizes are
  measured, the `junk_kind` addition is estimated at +1.3 MB per index from 129,737 non-NULL values (every row) and must be re-
  measured after the real build); INSERT OR REPLACE <= +2.5 ms/row and a hide-column UPDATE <= +1.5 ms/row against the
  schema without the eight new indexes (bench in the gate; derivation in "Write-cost accounting").
- G5 Tests can fail. Every new test has a named fault injection that turns it red, restored by hand and proven with a
  byte-identical `git diff -- <file>` snapshot (CLAUDE.md rule), not `git checkout`.
- G6 Safe rollout. An upgrade start never blocks the writers of a running library job, loses no progress when
  interrupted, and is safe with `--workers N` (Step 11); a clean shutdown always leaves every committed row in the
  `.db` file (Step 12).

Per-step criteria are in the Steps (S1-S12).

### Before / after timing table

Copy of the library, warm OS cache, p50 ms (p95 in brackets, first-request "cold" in italics). "Before" = 1.20.0
(VS report). "After" = prototype indexes with the same column sets (the `junk_kind` column excepted), measured with the
token fix emulated by a held-open connection, so every "After" figure below is a cache HIT unless the MISS column says
otherwise. MISS figures marked "this pass" were measured while writing this revision with `INDEXED BY` on the
investigation copy (`D:\facet-scratch\viewer\lib.db`, read-only immutable open): default page count 24-27, hidden summary
36-40, page 2.4 (page 500: 24-34, 44-50, 17). The remaining MISS thresholds are bounds, not measurements.

| # | Scenario | Step | Before | After, HIT (prototype) | Acceptance HIT | Acceptance MISS |
|---|---|---|---|---|---|---|
| 1 | `/api/photos` default page 1 | V1+V2 | 1006 (1060) *1639* | 23 (24) *1027* | p50 <= 60, cold <= 1500 | <= 200 (about 70 this pass) |
| 2 | page 2 / 20 / 100 | V1+V2 | 1017 / 1556 / 1704 | 21 / 23 / 25 | each <= 80 | each <= 250 |
| 3 | page 500 (deep) | V2 | 3070 (3938) *38155* | 124 *406* | p50 <= 300 | <= 400 (about 95 this pass) |
| 4 | `/api/photos/count` | V1+V2 | 713 (762) *10988* | 7 (7) | <= 20 | <= 60 (24-27 this pass) |
| 5 | `sort=date_taken` DESC / ASC | V2 | 848 / 919 | 20 / 17 | each <= 60 | each <= 250 |
| 6 | `sort=aesthetic` DESC, `per_page=200` | V2 | 1011 ; 1199 | 21 ; 52 | <= 60 ; <= 120 | <= 250 ; <= 350 |
| 7 | `/api/folders` (side effect of V2) | V2 | 819 | 191 | <= 400 | n/a (not count-cached) |
| 8 | category `portrait` / `landscape` | V3 | 245 / 431 | 21 / 19 | each <= 60 | each <= 250 |
| 9 | `type=top_picks` / `type=portrait` | V3 | 975 / 344 | 20 / 18 | each <= 60 | each <= 250 |
| 10 | camera R6 / 600D | V6 | 411 / 329 | 70 / 53 | each <= 150 | each <= 300 |
| 11 | lens only | V6 | 1325 *up to 31000* | 71 *129* | <= 150 | <= 300 |
| 12 | `sort=star_rating` (this library has 129,736 NULL ratings: shows the plan, not a rated library) | V6 | 6876 (10932) *12235* | 79 *1580* | <= 200 | <= 400 |
| 13 | `sort=aesthetic_iaa` ASC | V6 | 6431 (6531) *22412* | 15 *89* | <= 60 | <= 250 |
| 14 | `junk_kind=any`, sort=aggregate or date_taken, library copy (26 junk rows, 24 visible under the default hides): count / hidden summary / page | V4 | about 330000 (count, 332 s) / n.m. / page interrupted at 90000 | to be measured on the real build (Wave 3) | each <= 200 warm, <= 3000 cold | each <= 200 warm |
| 14b | same, synthetic 130k rows, sparse (24 junk), this pass | V4 | 157 / 129 / 716 | 51 / 33 / 56 | - | - |
| 14c | same, synthetic 130k rows, DENSE (30 % junk), this pass | V4 | 111 / 105 / 0.8 | 40 / 33 / 0.6 | - | - |
| 15 | `junk_kind=any` with the VS report's `(junk_kind < ? OR junk_kind > ?)` form, and with the previous revision's `path IN (SELECT ...)` form | V4 | - | **both rejected**: OR page interrupted at 90000; IN-subquery page 3-23 sparse but 162 ms dense at 60k rows (about 350 ms extrapolated to 130k) | - | - |
| 16 | `/api/filter_options/tags` | V5 | 972 (999) *1648* | 38 | <= 100 | n/a (stat cache) |
| 17 | `/api/stats/categories` first request | V7+V8 | 85563 then **HTTP 500** | 2140 cold (index present) | status 200 and <= 5000 cold | same |
| 18 | categories GROUP BY alone | V7 | 59 s, 6,823 MB read | 0.38 s, 17.1 MB read | bytes read <= 50 MB | same |
| 19 | `/api/stats/gear` first request | V8 | 122997 then **HTTP 500** | 43323 cold (500 from Inf; time unchanged by this spec) | status 200 only | same |
| 20 | repeat of an identical gallery request (cache hit) | V1 | 1006 (every request misses) | 6-7 | second identical request <= 15 | n/a |
| 21 | default gallery page, hit vs aggregate-leading index choice (this pass): count / summary / page with `idx_p_gal_agg` (9.9 MB) vs `idx_p_gal_wide` (15.7 MB superset) | Step 2 | - | agg: 24 / 36 / 2.4; wide: 27 / 40 / 2.4 | - | decides the merge in Step 2 |

Combined: default gallery page p50 1006 -> 23 ms HIT (V1+V2). V1 alone 1006 -> 51; V2 alone 1006 -> 269 (VS report
0 and 3.2 columns, that is the all-miss cost with the indexes). Rows 14-15 correct the VS report, which timed only the
COUNT of the OR form. Row 14b/14c "Before" is the original predicate with no `junk_kind` in any index, "After" the
original predicate with `junk_kind` in the covering index (`junk_cover130.py`, plans: `SCAN photos USING COVERING INDEX
idx_gallery_date_desc` for count/summary, `SCAN photos USING INDEX idx_gallery_aggregate` for the page, whose junk test is
evaluated on the index entry before the table row is read).

## Files to create / change

### Files touched

| File | Step | Change |
|---|---|---|
| `api/change_token.py` (NEW) | 1 | `db_change_token(db_path)`, `close_change_token_sentinels()`: one lazily-opened read-only connection per DB path, token = `generation:device:inode:PRAGMA data_version`; `timeout=0.1`; inode change reopens |
| `api/db_helpers.py` | 1 | one `_count_cache_key(prefix, from_clause, where_str, sql_params)` helper used by the 3 call sites (`get_cached_count` 745, `get_cached_count_async` 758, `get_cached_hidden_aggregates_async` 780); it reads `api.database.DEFAULT_DB_PATH` AT CALL TIME and puts the path in the key; delete `_db_change_token` (724-740). `from db import DEFAULT_DB_PATH` (27) and `import os` stay (used at 175, 569, 1126, 1154) |
| `tests/conftest.py` | 1 | one autouse fixture: clear `api.config._count_cache` before each test, `close_change_token_sentinels()` after it |
| `api/schema_startup.py` (NEW) | 11 | `init_schema_at_startup(db_path)`: take `facet.LibraryLock` (kind `maintenance`), run `init_database` with the photos-index build only when the lock was won, log and return `'built' / 'deferred' / 'failed'` |
| `api/__init__.py` | 1, 11, 12 | lifespan: call `init_schema_at_startup` in place of the bare `init_database` try-block (191-196); after `_flush_pending_rating_syncs()` (297) call `close_change_token_sentinels()`, then run the final `wal_checkpoint(TRUNCATE)` block (308-318) UNCONDITIONALLY (drop the `if wal_minutes > 0:`) |
| `db/schema.py` | 2, 3, 4, 6, 7, 11 | `GALLERY_VISIBILITY_COLUMNS` and `GALLERY_COVER_COLUMNS` constants; eight `INDEXES` entries appended after `idx_shoot_type_evidence` (434); `init_database(db_path, build_photo_indexes=True)`: commit before the INDEXES loop (1197-1200), one autocommit `CREATE INDEX` per index with a log line before each missing one, ANALYZE gate (1207-1214) keyed on `idx_gallery_aggregate` as well and skipped on an empty `photos`; with `build_photo_indexes=False` the INDEXES loop and the gate are skipped |
| `api/routers/filter_options.py` | 5 | the no-`vis` photo_tags branch of `tags()` (153-165); the `vis` branch, the recursive-split fallback and `junk_kinds` (463-485, unchanged) stay |
| `api/routers/albums.py` | 10 | `tiebreak='photos.path ASC'` at the two `order_by_clause(safe_sort, sort_dir)` calls (188, 230) |
| `api/routers/map.py` | 10 | `, p2.path ASC` on the cluster representative (60), `, path ASC` on the individual-photo query (93) |
| `api/routers/stats.py` | 7, 8, 10 | one shared finite-average SQL constant used by the categories SELECT (303) and `_GEAR_AGGREGATES` (379); the categories averages statement moved verbatim into a module constant; `COUNT(*) DESC, camera_model` / `COUNT(*) DESC, lens_model` in the two window ORDER BYs (318, 331); optional one-line guard on the aperture histogram (526-527) |
| `analyzers/capsule_generator.py` | 9 | one shape check at 1220 (`_generate_seeded`) |
| `tests/test_change_token.py` (NEW) | 1 | see Step 1 |
| `tests/test_gallery_indexes.py` (NEW) | 2, 3, 4, 6, 11 | registry pins, ANALYZE gate, per-index commit, plan + equivalence per gallery case incl. junk sparse and dense |
| `tests/test_viewer_sql_rewrites.py` (NEW) | 5, 10 | tags (oracle, plan), album and map tie order |
| `tests/test_stats_nonfinite.py` (NEW) | 8, 10 | seeded +Inf rows through the real routes; top camera/lens tie |
| `tests/test_capsule_seed_dim.py` (NEW) | 9 | mixed-dimension seeds through `_generate_seeded` |
| `tests/test_lifespan_startup.py` (NEW, Wave 2) | 1, 11, 12 | sentinel closed before checkpoint, `wal_checkpoint_minutes=0`, lock-held deferral, concurrent starts |
| `tests/test_stats_category_plan.py` (NEW, Wave 2) | 7 | the categories statement is answered from `idx_stats_categories` |
| `CHANGELOG.md`, `CLAUDE.md`, `docs/{,de,es,fr,it,pt,zh}/{VIEWER,DEPLOYMENT,COMMANDS,CONFIGURATION}.md` | docs | Wave 4, see "Documentation plan" |

`api/routers/gallery.py` is NOT touched any more: the `junk_kind=any` predicate stays byte-for-byte (Step 4 is an
index change).

### Final index registry (the eight indexes this spec adds)

For the post-scan speed spec, which builds after this one and dedupes against these names. Identical to the
`INDEXES` entries Step 2-7 add. `V` = `is_blink, is_burst_lead, burst_group_id, is_sequence_lead, is_duplicate_lead,
duplicate_group_id, sequence_kind, sequence_ev_offset, is_rejected`; `C` = `V, junk_kind`. Sizes: prototype measured, plus
an estimated 1.3 MB for `junk_kind` where `C` is used.

| Name | Table | Column list (exact) | Est. size | Step | Written by |
|---|---|---|---|---|---|
| `idx_gallery_aggregate` | `photos` | `aggregate DESC, path, C, category, camera_model, lens_model` | 17.0 MB | 2 | scan, `--recompute-average`, hide-column passes, `--detect-junk`, category/camera/lens edits |
| `idx_gallery_date_desc` | `photos` | `date_taken DESC, path, C` | 12.8 MB | 2 | scan, hide-column passes, `--detect-junk`, date edits |
| `idx_gallery_date_asc` | `photos` | `date_taken ASC, path, C` | 12.8 MB | 2 | same as above |
| `idx_gallery_category` | `photos` | `category, aggregate DESC, path, C` | 12.5 MB | 3 | scan, `--recompute-average`/`-category`, hide-column passes, `--detect-junk` |
| `idx_gallery_camera` | `photos` | `camera_model, lens_model, aggregate DESC, path, C` | 15.8 MB | 6 | scan, `--recompute-average`, hide-column passes, `--detect-junk` |
| `idx_star_rating_path` | `photos` | `star_rating DESC, path` | 7.5 MB | 6 | scan, rating edits |
| `idx_gallery_iaa_asc` | `photos` | `(aesthetic_iaa IS NULL), aesthetic_iaa, path, C` | 11.4 MB | 6 | scan, IAA recompute, hide-column passes, `--detect-junk` |
| `idx_stats_categories` | `photos` | `category, aggregate, aesthetic, comp_score, tech_sharpness, color_score, exposure_score, iso, f_stop, focal_length_35mm, focal_length, face_quality, contrast_score, camera_model, lens_model` | 17.1 MB | 7 | scan, `--recompute-average`, score recomputes |

Total about 106.9 MB. Merge decision: `idx_gallery_aggregate` IS the aggregate-leading superset the VS report called
`gal_wide`/`idx_gallery_lens` (it serves the lens filter and the default scan), so there is no separate lens index and
no second aggregate-leading index; measured on the default statements it costs +3 to +4 ms on the cached count/summary
and nothing on the page (row 21). `idx_gallery_category` stays separate because its leading column is `category`.
Existing indexes kept untouched (Q3): `idx_aggregate`, `idx_date_taken`, `idx_date_taken_desc`, `idx_category_aggregate`,
`idx_star_rating`, `idx_camera_lens`, `idx_junk_kind`. No new index name collides with an existing one (`git grep` for
`idx_gallery|idx_stats|idx_star_rating_path` was empty at `21a25ef6` and the HEAD diff does not add any).

### Not changing

- Response models, OpenAPI, `client/` (so no `tsc`/`ng lint`/Vitest run; the only client check is the empty `gen:api` diff).
- Row/column layout of `photos` (VS candidate 11, a 16 GB table rebuild), `order_by_clause` / `NULLABLE_SORT_COLS`
  (only `aesthetic_iaa` ASC gets an index; the other five nullable ASC sorts stay slow), `HIDE_*_SQL` / `HIDDEN_*_SQL`
  semantics, `build_hide_clauses`, the `junk_kind=any` predicate in `gallery.py`.
- No existing index is dropped (Q3). No timeline index (VS candidate 10), no map index, no `/api/capsules` start-up cost.
- `_get_stats_cached_async`, `_sanitize_stats`, `CORRELATION_X_AXES['f_stop']` (cosmetic `inf` label, hot file),
  the stored `f_stop` rows (Q5), `scoring_config.json` (no new key), `database.py`, the Docker healthcheck.
- The other stats endpoints (overview 12-20 s, gear 43 s, correlations 119-163 s, overlap 33 s cold): no start-up
  warm-up (Step 7), they pay once per TTL as today.
- `tests/test_stats.py` (hand-rolled `CREATE TABLE photos`, and open-gaps B2 edited it): never extended here.
- The junk filter combined with a sort that has no `junk_kind`-carrying index (everything except aggregate, date,
  aesthetic_iaa ASC, category-led and camera-led plans): unchanged from today, see Q17.

## Domain rules verified / assumed

| Rule | Status | Source |
|---|---|---|
| `PRAGMA data_version` on a long-lived connection changes on every commit by ANY other connection or process, incl. the same process; does not change on pure reads; works without a checkpoint | verified | scratch `tok_exp.py`: viewer-style writes 0/600 missed, external writer 0/599 missed |
| The header change counter (bytes 24-27) is never updated in WAL mode; size+mtime of db and `-wal` (with or without ignoring an empty WAL) misses commits | verified | same run: header missed 599/599, size+mtime and "ignore empty WAL" each missed 169/599 |
| Today's token changes on every request, so the count and hidden-summary caches never hit | verified | `old_token_exp.py`: 6 pure-read requests, 6 distinct tokens (`-wal` mtime moves each time); with the sentinel the WAL persists and the token is stable |
| The token and the cache key read `db_helpers`' own `DEFAULT_DB_PATH` binding (line 27, used at 736), while connections read `api.database.DEFAULT_DB_PATH` (19, 33, 68), which is the global the tests patch | verified (static, review B1) | `api/db_helpers.py:27,736`, `api/database.py:19,33,68`; this is why Step 1 reads the path at call time and keys on it |
| `wal_checkpoint(TRUNCATE)` with the sentinel open succeeds and only bumps `data_version` (harmless), but it NEVER raises on contention: a pinned reader returns `(1, n, m)` | verified | `tok_exp.py` (c), (c2); review probe `probe_ties.py` |
| A read-only sentinel cannot delete the WAL at last close, so after it the `.db` alone can lack committed rows (0 of 3 in the probe); today the last per-request read-write connection checkpoints and deletes the WAL even with `wal_checkpoint_minutes = 0` | verified | review probe `probe_lastclose.py`; `api/__init__.py:308-318` (checkpoint only `if wal_minutes > 0`) |
| `init_database` holds ONE write transaction from its first DML (`UPDATE persons ...`) through every `CREATE INDEX` and `ANALYZE photos` to the final commit, and the viewer calls it without the library lock | verified | review probe `probe_init_txn.py` (`in_transaction=True` at `CREATE INDEX idx_aggregate` and `ANALYZE photos`); `api/__init__.py:191-196`; every CLI job locks first (`facet.py:2179`) |
| `PRAGMA busy_timeout = 5000` on every connection `apply_pragmas` opens, so a second writer waits 5 s and then raises `database is locked` | verified (static) | `db/connection.py:142` |
| `facet.library_job_holder` / `LibraryLock(db, kind=LIBRARY_JOB_MAINTENANCE)` is the existing host-local mutex the API already peeks (`api/routers/scan.py:57-71,461-472`); `import facet` is cheap here (no torch/transformers loaded) | verified | `facet.py:1277,1320,705`; import probe |
| `photos.path` is unique (`sqlite_autoindex_photos_1`) | verified | `db/schema.py` `PHOTOS_COLUMNS` |
| `junk_kind IS NOT NULL AND junk_kind != c` is evaluable from an index entry that contains `junk_kind`; with it in the covering indexes the planner scans the covering index for count/summary and the ordered index with the junk test applied before the table lookup for the page | verified (synthetic 130k, sparse and dense) | `junk_cover130.py` (row 14b/14c) |
| On this library `junk_kind` is non-NULL on every one of the 129,737 rows: `not_junk` 129,711, `document` 19, `screenshot` 5, `receipt` 2 (26 junk rows, 24 of them visible under the default hide toggles) | verified | `libdb_idx.py` on `lib.db` |
| The gallery `ORDER BY` always ends in `path ASC`; album pages, the map queries and the stats top-camera/lens windows do not | verified | `gallery.py:882`; `albums.py:188,230`; `map.py:60,93`; `stats.py:318,331`; review probe `probe_ties.py`: smart-album page 2 shares only 16 of 48 paths once `idx_gallery_category` exists |
| The hide OR-chains read exactly the nine V columns plus the single-user `is_rejected` | verified | `db_helpers.py:207-262`, `gallery.py:454-456` |
| `ANALYZE photos` is needed for the planner to take the new indexes on a small seeded DB; the production gate keys only on `idx_moment_confidence`; `ANALYZE` on an EMPTY table writes no `sqlite_stat1` rows, so the gate refires on every start of an empty DB | verified | `plan_exp2.py`/`plan_exp3.py`; review probe; `db/schema.py:1207-1214` |
| Only `f_stop` holds non-finite values on this library (144 rows, all +Inf). `focal_length_35mm` is NULL on every row and `focal_length` has no value beyond 61,440 and none above 1e300 or below -1e300 | verified (this pass; closes review NOTE 13) | `inf_cols.py`; `inf_focal.py` through the covering prototype `idx_p_stats_cat` (129,737 rows, 99 ms) |
| The EXIF writers already null non-finite values | verified | `exiftool_batch._safe_numeric` (~234), `processing/scorer.py:290` `_sanitize_exif_numeric`, commit `87a44806` (2026-06-22): the 144 rows predate it |
| `_sanitize_stats` turns inf into None, and `avg_f_stop: Union[int, float]` rejects None, so FastAPI answers 500 after the compute | verified | `api/config.py:1470`, `api/models/stats.py:55,78`; reproduced in about 1.3 s with seeded rows (`probe_inf/test_probe_inf.py`) |
| The seeded-capsule candidate set mixes dimensions: 1,940 rows at 768 and 60 at 1152 of the top 2,000; `filter_uniform_embeddings` keeps 768, the seed vector is never filtered | verified | `emb_dims.py`; `capsule_generator.py:1177`, `1219-1221`. About 26% of 10-seed draws hit a 1152 seed |
| Multi-user mode (`LEFT JOIN user_preferences up`) keeps `path` unambiguous (that table has `photo_path`) | verified by schema read; **UNVERIFIED** timing | `get_photos_from_clause` (db_helpers ~1294) |
| `init_database` runs synchronously in the lifespan before the port opens | verified | `api/__init__.py:194` |

## Open questions / risks

Defaults are what a builder does absent further input.

- Q1 ANALYZE after upgrade. Default: the gate fires when `idx_moment_confidence` OR `idx_gallery_aggregate` has no
  `sqlite_stat1` row AND `photos` has at least one row (review NOTE 6: an empty table cannot be analyzed, so the gate
  used to refire on every empty-DB init, which the test suite does constantly). It runs as its own autocommit statement
  after the index loop (7.1 s measured on the copy). Risk: ANALYZE can flip other plans; the equivalence run covers the
  case matrix and Q12 re-runs the full VS list once.
- Q2 Index build at first start (revised, review G-1). The nine-index build used to run inside ONE write transaction
  without the library lock: a concurrent scan's writes failed after the 5 s busy timeout, a killed start rolled back all
  progress, and `--workers 4` raced. Now (Step 11): the lifespan takes the library lock first (a held lock defers the
  build with a log line), each index is its own autocommit `CREATE INDEX` with an INFO line before it, and ANALYZE runs
  last. The build is still synchronous and still blocks the port: the VS report measured 102-106 s per gallery index and
  169 s for the stats index on a cold, contended cache (4-8 s warm), so up to about 15 minutes on a 130k library, but a
  kill now keeps every finished index and the next start resumes (`IF NOT EXISTS`). A background build was rejected:
  `CREATE INDEX` holds the write lock for its whole run, so viewer writes would turn into `database is locked`.
  Docker: `docker-compose.yml:64-69` has `start_period: 30s`, `retries: 3`; the container is reported unhealthy during a
  long build but `restart: unless-stopped` does not act on health. DEPLOYMENT.md says to pre-create with
  `python database.py` (takes the same lock).
- Q3 Drop the indexes the new ones supersede (`idx_aggregate`, `idx_date_taken`, `idx_date_taken_desc`,
  `idx_category_aggregate`, `idx_star_rating`). Default: no. Untested for the queries that still use them
  (`api/routers/frame.py` `_sample` relies on `idx_aggregate`). A follow-up would offset the write cost. (The review's
  NOTE 5 asked to drop the redundant aggregate index; Step 2 already removes the redundancy WITHIN this spec by shipping
  one aggregate-leading index instead of two; the pre-existing ones are not touched.)
- Q4 Value of `idx_gallery_iaa_asc` (11.4 MB, one of six nullable ASC sorts). Default: ship, it is measured (6431 -> 15 ms)
  and the scope lists it.
- Q5 Repair the 144 stored `+Inf` rows (`UPDATE photos SET f_stop = NULL WHERE f_stop > 1e300`). Default: no. It
  rewrites user data and 9+ indexes for rows the read guard already handles; the writer is fixed. Offer as a
  one-line `db/maintenance.py` follow-up.
- Q6 Aperture histogram in `/api/stats/settings` (`stats.py:526-527`) admits the same Inf rows and serializes one
  bucket with `value: null` (200, no crash). Default: include a one-line `AND f_stop < 1e308` guard and a test; it is
  the same defect class in the same hunk. The lead may strike it without touching anything else.
- Q7 Correlations `f_stop` axis `inf` label (`api/config.py:1642-1644`): not changed.
- Q8 Stats warm-up. Default: none. A start-up pass over the remaining slow endpoints is about four minutes of table
  scans that contend with the first gallery pages; categories drops to 2.1 s with its index, gear stays about 43 s once
  per hour (follow-up: a gear covering index, unprototyped). Filtered stats (`date_from`/`date_to`) read `date_taken`,
  which `idx_stats_categories` does not contain, so they stay slow (**UNVERIFIED** whether adding `date_taken` pays).
- Q9 The sentinel keeps a file handle on the DB for the life of the process. On Windows the DB cannot be renamed or
  deleted while the viewer runs (per-request connections did not hold it while idle); `-wal`/`-shm` also persist while
  the viewer runs. Accepted and documented ("stop the viewer before replacing or restoring the DB"). A UNC DB path
  (`\\host\share`) probably fails `Path.as_uri()` as a SQLite URI and falls back to the never-hit token (today's
  behaviour, plus one warning): **UNVERIFIED**, not a regression.
- Q10 Plan assertions are SQLite-version sensitive (verified on 3.51.0 only). Default: assert properties
  (`COVERING INDEX` present, no non-covering `SCAN photos`, `USING INDEX idx_gallery_*`, no `TEMP B-TREE FOR ORDER BY`
  for the cases that had none), never whole plan strings. If the CI SQLite disagrees on the temp-b-tree property for a
  page, the builder relaxes only that sub-assertion and records it.
- Q11 Multi-user mode (`vis` clauses, `up.is_rejected`) is not measured: the covering index cannot cover `up.*`, so
  those requests may stay at table-pass speed (still correct). The gate times one multi-user gallery page before and
  after (review NOTE 14) and requires "not worse than before + 10 %". **UNVERIFIED**: whether the copy's config can
  host a multi-user run; fallback is the captured SQL timed on a sync connection with `user_id` set.
- Q12 Planner flips on cheap queries. The VS report's first sweep showed `g_min_score_8` 65 -> 211, `g_date_2024`
  92 -> 298, `g_search_nomatch` 31 -> 258; the targeted re-runs gave 68, 126, 49, so mostly noise. The gate re-runs the
  VS report's FULL case list once (review NOTE 15, because the gate's ANALYZE can flip any plan) and fails on any case
  that is worse than before + 100 ms at MISS p50.
- Q13 (rewritten, review B1) V1 makes the count cache hit where it never did. The new tests DO patch
  `api.database.DEFAULT_DB_PATH` (the global connections read) and that is safe because (a) the cache key now includes
  the path read at call time from that module and (b) the autouse fixture in `tests/conftest.py` clears
  `api.config._count_cache` before every test, so no key from another test or DB survives. The existing tests that
  patch `mock.patch.dict("api.config._count_cache", {}, clear=True)` keep working. The full-suite diff in the gate
  catches any other test that relied on an always-missing cache.
- Q14 Float `AVG` over a different row order can differ in the last ulp; the endpoints `ROUND` to 1-2 decimals, so a
  mismatch would need an exact .5 boundary. The gate compares the rounded output and investigates any difference.
- Q15 Disk and copy plan. `D:` has 94 GB free now. One pristine `lib_copy.db` (15.2 GiB) exists;
  `D:\facet-scratch\viewer\lib.db` (15.3 GiB) is the earlier investigation's copy with prototype `idx_p_*` indexes and
  must NOT be used as a "before" (contaminated). Default: ONE working copy `D:\facet-scratch\viewer_spec\lib.db` made
  from `lib_copy.db` with the sqlite3 backup API, the BEFORE results saved to JSON, the indexes then built in place by
  the real first start, AFTER diffed against the JSON. No second 16 GB copy. A dense-junk run (gate item 5b) rewrites
  `junk_kind` on that working copy only, after AFTER is captured. Never open the production DB; never delete a photo
  file; no cull/apply runs.
- Q16 Sibling specs (scan-pass performance, scoring/tagging defects, post-scan step speed) were written in parallel.
  The post-scan spec (`.claude/specs/postscan-speed.md`) builds AFTER this one and dedupes its indexes against the
  registry table above; this spec does not change an index for it. Any sibling that edits `db/schema.py`,
  `api/__init__.py`, `CHANGELOG.md`, `CLAUDE.md` or the docs must be sequenced against Waves 1, 2 and 4 by file.
- Q17 (new) Residual of the junk decision. The ruling drops the subquery rewrite because it is slower on the DENSE
  distribution (row 15). The covering form fixes count, summary and the page for sorts whose index carries `junk_kind`
  (aggregate, date, aesthetic_iaa ASC, category- and camera-led plans). A sparse-junk request sorted by a column with no
  such index (`sort=tech_sharpness`, `face_quality`, `comp_score`, ...) still walks that sort's index with a table lookup
  per row to find the few junk rows, exactly the plan that ran for minutes on the copy; that is UNCHANGED from today, not
  a regression, and is left open. Evidence: row 14b "Before" (716 ms warm on narrow synthetic rows; interrupted at 90 s
  on the real rows). A follow-up could add a `junk_kind`-leading index (`junk_kind, aggregate DESC, path`) if sparse junk
  with arbitrary sorts matters.
- Q18 (new) Replacing the DB file under a running viewer. On POSIX (rsync-rename, restore) the sentinel would keep the old
  inode and `data_version` would freeze, so the token folds in `st_dev`/`st_ino` and reopens the sentinel when they change
  (one `os.stat`, about 5 us). On Windows the replace itself fails while the sentinel holds the file (Q9), so that test is
  POSIX-only and **UNVERIFIED on this box**. SMB multi-host: `data_version` rides the shared-memory WAL index and does not
  see another host's commits; WAL over SMB is already unsupported and CLAUDE.md says the token is host-local, like
  `LibraryLock`.
- Q19 (new) `wal_checkpoint_minutes = 0` now means "no periodic checkpoint thread" only; the shutdown checkpoint always
  runs (Step 12). The CONFIGURATION.md row is adjusted in all seven languages.
- Q20 (new) `conn.commit()` before the index loop makes the migration steps above it durable before the indexes are built
  (they used to roll back together with them). They are idempotent `ALTER ... ADD COLUMN IF MISSING` and backfills, so
  a re-run after an interruption is safe; stated so a reviewer does not read it as a regression.
- Q21 (new) A start that loses the library-lock race still runs `init_database` for the tables/columns (without the photos
  indexes). While another process is mid-`CREATE INDEX` that call can hit the 5 s busy timeout and fail; the helper catches
  it, logs it and returns `'failed'` exactly as the old lifespan try-block did, and the viewer serves. Acceptable because
  the holder ran the same schema step.
- Q22 (new) A running library job started by an OLDER build (no lock) is invisible to the lock peek; it is covered by the
  5 s busy timeout as before.

## Steps

### Step 1 (V1): replace the never-hitting `_db_change_token` (B1, G-2, G-6, NOTE 2, 4, 8, 16)

- description: add `api/change_token.py`. `db_change_token(db_path)` returns `f"{generation}:{st_dev}:{st_ino}:{data_version}"`
  from a per-path sentinel opened on first use with `sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro",
  uri=True, isolation_level=None, check_same_thread=False, timeout=0.1)`, guarded by one module lock. `generation` comes
  from a module `itertools.count()` and is bumped on every (re)open, so a reopened sentinel (whose `data_version`
  restarts) can never reproduce an earlier token. `PRAGMA data_version` is read with `fetchall()` so the cursor is
  finished and the sentinel never pins a read transaction. Every call stats the path (`os.stat`, about 5 us); if
  `(st_dev, st_ino)` differs from the values captured at open, the sentinel is closed and reopened (generation bump). A
  busy/locked error from the pragma returns a never-repeating miss token (`f"x{next(_misses)}"`), keeps the sentinel and
  logs at DEBUG only (the 0.1 s timeout stops a shm-lock wait from stalling the event loop, review NOTE 4). Any failure to
  open, stat or parse drops the sentinel, logs ONE warning per path per process, and returns a miss token, which is
  today's behaviour (no cache hit). `close_change_token_sentinels()` closes every sentinel and clears the registry; it is
  idempotent.
  In `api/db_helpers.py` replace the three duplicated f-string keys with one `_count_cache_key(prefix, from_clause,
  where_str, sql_params)` that does `from api import database as _database` (module-level import; `api.database` imports
  only `db`, no cycle) and reads `_database.DEFAULT_DB_PATH` at call time: key = sha256 of `path : from_clause : where :
  params : db_change_token(path)`. Delete `_db_change_token` and its docstring (no alias).
  In `tests/conftest.py` add the autouse fixture (clear `api.config._count_cache` before, `close_change_token_sentinels()`
  after).
  The `api/__init__.py` lifespan part (close the sentinels after `_flush_pending_rating_syncs()`, unconditional final
  checkpoint) is Step 12 and lands in Wave 2.
- key decision: `PRAGMA data_version` sentinel versus an `os.stat` token. The VS report suggested ignoring an empty
  `-wal`; the measurements contradict it (Domain rules): that token misses 169 of 599 commits from a long-lived external
  writer, the sentinel misses 0, and it costs 3.8 us per call against 9.4 us for two `stat`s. Rejected: a trigger counter
  (writes on every photo change), the header counter (frozen in WAL mode), reading `-shm`. The DB path is resolved at
  call time from the module the connections and the tests use, never from `db_helpers`' own import-time binding (review
  B1: the old shape would make a tmp-DB test compare the cache with itself).
- default: sentinel opened lazily at the first token request (works in tests that never run the lifespan); `mode=ro`;
  the key is computed BEFORE the query, which the three callers already do (the token is evaluated ahead of the
  `execute`), so a commit that lands mid-query can only store under an old key that no later request asks for.
- verify (`tests/test_change_token.py`, tmp DBs built by `init_database`, rows inserted with plain `INSERT (path,
  filename)`; every test patches `api.database.DEFAULT_DB_PATH` to its own tmp DB):
  1. six request-style open/read/close cycles with the token read while the request connection is open -> exactly one
     distinct token (the old function gave six);
  2. a commit from a second connection changes the token; 100 sequential commits from a child process (script file under
     `tmp_path`, run with `subprocess`, autocheckpoint on) each change it (100 distinct tokens; a size+mtime token fails);
  3. `PRAGMA wal_checkpoint(TRUNCATE)` on another connection with the sentinel open returns exactly `busy == 0` and
     `os.path.getsize(db + "-wal") == 0` afterwards (review G-6: the pragma never raises on contention, so "succeeds" proves
     nothing); fault: make the sentinel keep a statement open (`cur = sent.execute("SELECT 1 UNION ALL SELECT 2");
     cur.fetchone()`) and the test goes red on `busy == 1`;
  4. a nonexistent path returns two different tokens and logs one warning (`caplog`);
  5. the test reads at least one token BEFORE its first commit, then `close_change_token_sentinels()` twice is fine, and the
     next token differs from every earlier one including the pre-commit one (review NOTE 8: otherwise a generation reset is
     indistinguishable from a `data_version` bump); fault: drop the generation counter -> red;
  6. integration through the shared `edition_client`: two identical `/api/photos/count` requests call
     `api.db_helpers._count_cache_store` once (`mock.patch.object`, not an auth dependency); `seed_photos_prefix` then adds a
     row and the next request returns the larger count;
  7. path resolution (B1): two tmp DBs with different row counts, `api.database.DEFAULT_DB_PATH` patched to the first and then
     the second, the same `/api/photos/count` filter returns each DB's own count with NO cache clearing between them; fault:
     make `_count_cache_key` read `db_helpers.DEFAULT_DB_PATH` (the import-time binding) -> the second request returns the
     first DB's count and the test fails; this is also the control that the Step 2/3/6 equivalence tests cannot compare the
     cache with itself;
  8. inode change (POSIX only, `skipif sys.platform == "win32"`): `os.replace` a different DB over the path -> the next token
     differs and `/api/photos/count` reflects the new file;
  9. busy sentinel: patch the internal pragma reader to raise `sqlite3.OperationalError("database is locked")` -> the token is a
     miss token and the NEXT call (reader restored) returns the stable token again without a reopen (generation unchanged).
  Fault injections (each restored by hand, `git diff` snapshot compared): (a) put the old stat body back into
  `db_change_token` -> tests 1 and 2 fail; (c) per test 5 and 3 above; (d) path binding per test 7.
  Live (copy): `cache_probe.py 8` (scratch, no second connection held) shows about 6 ms from call 2, against 34-42 ms
  all-miss; the default page row of the timing table.
- affected files: `api/change_token.py` (NEW), `api/db_helpers.py`, `tests/conftest.py`, `tests/test_change_token.py` (NEW).
- docs (Wave 4): CLAUDE.md invariant (token source, host-local like LibraryLock, key includes the DB path), DEPLOYMENT.md
  (WAL file persists while the viewer runs; stop the viewer before replacing or restoring the DB), CHANGELOG `### Fixed`.

### Step 2 (V2): covering visibility indexes

- description: in `db/schema.py` add `GALLERY_VISIBILITY_COLUMNS` (the nine columns, one tuple) and `GALLERY_COVER_COLUMNS
  = GALLERY_VISIBILITY_COLUMNS + ("junk_kind",)`, and append `idx_gallery_aggregate`, `idx_gallery_date_desc`,
  `idx_gallery_date_asc` to `INDEXES` (column lists in the registry table), built from those tuples so the registry cannot
  drift from the tests that read it. Extend the ANALYZE gate (Q1).
- key decision: covering index with `path` right after the sort column. `path` is the tiebreak `_resolve_order_by`
  appends (`gallery.py:882`), and without it the old `idx_aggregate` forced a `TEMP B-TREE FOR LAST TERM OF ORDER BY`
  (10,246 ms vs 24 ms for the page-500 statement). All three gallery statements (count, hidden summary, page) read the
  same nine columns, which today costs 516 MB and 654 MB of table pages per request; the index reads 10 MB. The
  aggregate-leading index is the superset that also carries `category, camera_model, lens_model` (serves the lens
  filter, review NOTE 5, row 21): one index instead of two.
- default: as in the registry; `idx_aggregate`/`idx_date_taken*` stay (Q3).
- verify (`tests/test_gallery_indexes.py`; schema built with `init_database`; 2,000+ pseudo-random rows with random
  aggregate, dates, categories, flags and cameras, then `ANALYZE photos`, on its OWN tmp DB; read
  `.claude/patterns/test-fixtures.md` first; tests that go through routes patch `api.database.DEFAULT_DB_PATH` and rely on
  the Step 1 key/fixture):
  1. registry pin: every `photos` column named in `HIDE_*_SQL`, `HIDDEN_*_SQL` (imported from `api.db_helpers`) plus
     `is_rejected` is in `GALLERY_VISIBILITY_COLUMNS`; `junk_kind` is in `GALLERY_COVER_COLUMNS` and in every index built
     from it;
  2. plan: the count, hidden-summary and page statements are the ones the route issues, captured by routing the real
     endpoints through an `aiosqlite` connection with `set_trace_callback` and then `EXPLAIN QUERY PLAN`-ed on a sync
     connection; never a hand-copied statement. Count and aggregate show `COVERING INDEX`; the default/date DESC/date ASC
     pages show `USING INDEX idx_gallery_*` and no `TEMP B-TREE FOR ORDER BY`. Control inside the same test: `DROP INDEX` the
     new ones and the assertion flips to a non-covering `SCAN photos`, so it can fail;
  3. equivalence: count, aggregate row and ordered page paths are equal with and without the new indexes for the case matrix
     of Steps 2, 3, 6. Every comparison clears `api.config._count_cache` between the two runs (explicit `.clear()` in the
     test body, in addition to the fixture), and the control run asserts the cache was empty before each side;
  4. gate: build the DB, drop the eight new indexes and their `sqlite_stat1` rows after an `ANALYZE`, run `init_database`
     again -> `sqlite_stat1` has a row for `idx_gallery_aggregate`; an EMPTY-photos DB gets no ANALYZE (assert via the trace
     callback that no `ANALYZE` statement ran);
  5. `init_database` twice creates nothing the second time (existing `tests/test_a2_db_and_comparisons_migration.py:267`
     keeps `db.info`'s index count honest).
  Fault injections: remove `is_rejected` from the tuple -> test 1 fails (and a plan assertion on the single-user rejected
  filter); revert the gate -> test 4 fails; remove the empty-table condition -> test 4's empty-DB half fails.
  Live (copy): VS report scripts re-run, `verify_equiv.py` style comparison through the HTTP matrix (Final gate).
- affected files: `db/schema.py`, `tests/test_gallery_indexes.py` (NEW).
- docs (Wave 4): VIEWER.md Performance, DEPLOYMENT.md (first start builds indexes; pre-create), COMMANDS.md row,
  CLAUDE.md invariant "a new hide column must join `GALLERY_VISIBILITY_COLUMNS`", CHANGELOG `### Changed`.

### Step 3 (V3): category-leading index

- description: append `idx_gallery_category (category, aggregate DESC, path, C)`.
- key decision: one more index for the category/type dropdown, the most common filter. Category and type filters
  today read the table for the hide columns of every row in the category.
- default: ship (portrait 245 -> 21, landscape 431 -> 19, `type=top_picks` 975 -> 20).
- verify: in `tests/test_gallery_indexes.py`, the category-filter statements show `SEARCH ... USING COVERING INDEX
  idx_gallery_category (category=?)`; control drop flips it; equivalence for `category=portrait` and a `type=` case (cache
  cleared per side). Fault: drop the entry -> plan test fails.
- affected files: `db/schema.py`, `tests/test_gallery_indexes.py`.

### Step 4 (V4): `junk_kind=any` is fixed in the INDEX, the predicate stays (G-4)

- description: no change to `api/routers/gallery.py`. `junk_kind` is appended to the covering tuple (`GALLERY_COVER_COLUMNS`,
  Step 2) so every gallery covering index contains it. The existing predicate `junk_kind IS NOT NULL AND junk_kind != ?`
  (`gallery.py:259-261`) is then evaluated from the index entry: count and hidden summary become a covering index scan,
  the page an ordered index scan whose junk test runs before the table row is read.
- key decision (review G-4, coordinator ruling): keep the original predicate and add the column, rather than rewrite it.
  Identical by construction (no rewrite, no new equivalence argument). Both previous rewrites are rejected on evidence:
  the VS report's `(junk_kind < ? OR junk_kind > ?)` left the PAGE on an ordered table scan (interrupted at 90 s), and the
  previous revision's `path IN (SELECT path ... )` form is fast on a sparse library (page 3-23 ms) but 162 ms at 60k
  DENSE rows (about 350 ms at 130k, `probe_junk_cover.py`). Measured this pass at 130k synthetic rows with the real route
  SQL and default hide clauses (`junk_cover130.py`): sparse before 157/129/716 ms, after 51/33/56; dense before 111/105/0.8,
  after 40/33/0.6 (count/summary/page). Faster or equal on both distributions; the subquery form was faster on only one.
- default: ship the covering form; the copy is measured in Wave 3 (sparse, 26 junk rows) and in the optional dense run
  (gate 5b). Cost: +1.3 MB per covering index.
- verify (`tests/test_gallery_indexes.py`, `init_database` schema, own tmp DBs, `ANALYZE` first): two distributions of
  at least 20,000 rows each, SPARSE (24 junk rows) and DENSE (30 % junk), both with a second category and the default hide
  toggles; per distribution, through the captured route statements:
  1. `/api/photos?junk_kind=any` returns exactly the non-NULL, non-`not_junk` rows, `/api/photos/count` and
     `/api/photos/paths` agree, combined with a `category=` filter and the default hide toggles;
  2. plan: the count and summary statements show `COVERING INDEX idx_gallery_*` and no bare `SCAN photos`; the PAGE shows
     `USING INDEX idx_gallery_aggregate`, never a non-covering scan of another index; for the `sort=date_taken` page,
     `idx_gallery_date_desc`. Control: drop the junk-carrying indexes and re-create them WITHOUT `junk_kind` in the same test,
     and the sparse count flips to a bare `SCAN photos` (so the test can fail);
  3. `seed_photos_prefix` rows hold `junk_kind` NULL, `'not_junk'`, `'document'`, `'screenshot'` and a BLOB. (TEXT affinity
     converts an INTEGER to TEXT on insert, so the storage-class claim is dropped; review NOTE 9.)
  Fault injection: remove `junk_kind` from `GALLERY_COVER_COLUMNS` -> the registry pin and the plan test go red while the
  results test stays green (state in the report: only the plan tests guard the speed).
  Live (copy): BEFORE capture of the old SQL through the route (about 330 s for the count, in the background, saved to
  JSON); AFTER must be equal and each statement <= 200 ms warm.
- affected files: `db/schema.py`, `tests/test_gallery_indexes.py`.
- docs (Wave 4): CHANGELOG `### Fixed` ("the gallery's junk filter set to any no longer scans the library"), CLAUDE.md
  invariant "`junk_kind` rides in the gallery covering indexes (`GALLERY_COVER_COLUMNS`) on purpose".

### Step 5 (V5): tags dropdown

- description: in `filter_options.tags()` (121-195), the `photo_tags_ready and not vis` branch becomes
  `SELECT tag, SUM(n) AS cnt FROM (SELECT tag, COUNT(*) AS n FROM photo_tags GROUP BY tag UNION ALL SELECT m.tag,
  COUNT(*) FROM photo_manual_tags m WHERE NOT EXISTS (SELECT 1 FROM photo_tags t WHERE t.photo_path = m.photo_path AND
  t.tag = m.tag) GROUP BY m.tag) GROUP BY tag ORDER BY cnt DESC, tag ASC LIMIT ?` (without the manual half when
  `manual_ready` is false). The `vis` branch keeps the current `UNION` statement, because its `vis_sub` filter needs the
  pair rows. The recursive-split fallback and `get_cached_stat` read are untouched.
- key decision: rewrite only, no in-process cache. The `UNION` de-dupe builds a temp B-tree over all 344,113
  `(photo, tag)` pairs on every call after the 300 s stat window (972 ms p50); the rewrite is an index-only `GROUP BY` on
  `idx_photo_tags_tag` plus an anti-join of the manual tags: 929 -> 38 ms (`tags_rw.py`), 89 identical rows, also
  identical with duplicate/new/popular manual tags (`dedupe_chk.py`). A cache would add an invalidation path for 38 ms.
- default: as above; `max_tags` limit and the `cnt DESC, tag ASC` tiebreak unchanged.
- verify (`tests/test_viewer_sql_rewrites.py`): seed `photo_tags` and `photo_manual_tags` with an AI-only tag, a
  manual-only tag, a manual tag equal to an AI pair (must count once), two tags with equal counts (tiebreak by name); the
  test DELETES the `stats_cache` `'tags'` row (or patches `get_cached_stat` to return None) before every request, because
  `tags()` returns `get_cached_stat('tags')` for single-user/no-`vis` and an earlier test in the session may have filled it
  (review NOTE 10: otherwise the oracle compares the cache); the endpoint's list equals a count computed in Python from the
  raw rows (an independent oracle, not the old SQL); a `max_tags` cut keeps the same prefix; the plan of the captured
  statement shows `COVERING INDEX idx_photo_tags_tag` and no `SCAN photo_tags` without it (control: the old statement shows
  the temp B-tree).
  Fault injection: drop the `NOT EXISTS` -> the manual-equals-AI case over-counts and the test fails.
- affected files: `api/routers/filter_options.py`, `tests/test_viewer_sql_rewrites.py`.
- docs (Wave 4): CHANGELOG `### Changed` only.

### Step 6 (V6): camera / lens, star rating, NULLS-last ASC

- description: append `idx_gallery_camera`, `idx_star_rating_path`, `idx_gallery_iaa_asc` (registry table; the lens filter
  is served by the aggregate-leading `idx_gallery_aggregate`, Step 2). The expression index matches `order_by_clause`'s
  `(aesthetic_iaa IS NULL) ASC, aesthetic_iaa ASC, path ASC` text exactly (`types.py:110-111`).
- key decision: one expression index for `aesthetic_iaa` only. The other five nullable ASC sorts would each need one;
  rarely used, left slow and listed in "Not changing".
- default: ship all three (Q4).
- verify (`tests/test_gallery_indexes.py`): per case the captured statements: `camera=` count/summary covered by
  `idx_gallery_camera` (the page may keep `idx_camera_lens` plus a temp b-tree; the planner chose that in the seeded run,
  and the copy still measured 411 -> 70 ms); `lens=` count/summary covered by `idx_gallery_aggregate`; `sort=star_rating`
  page `USING INDEX idx_star_rating_path`; `sort=aesthetic_iaa&dir=ASC` page `USING INDEX idx_gallery_iaa_asc` with no
  temp b-tree. Equivalence includes rows with NULL `aesthetic_iaa` and equal values, so NULLs-last and the `path` tiebreak
  are proven identical with and without the index (cache cleared per side). Controls and faults as Step 2; extra fault:
  change the index expression to `(aesthetic_iaa IS NOT NULL)` -> the ASC plan test fails.
- affected files: `db/schema.py`, `tests/test_gallery_indexes.py`.
- docs: with Step 2.

### Step 7 (V7): stats categories covering index, warm-up decision

- description: append `idx_stats_categories` (registry table) and, in `stats.py`, move the categories averages statement
  (`SELECT category, COUNT(*) ... GROUP BY category ORDER BY cnt DESC`, 296-310) verbatim into a module constant with a
  `{vis}` placeholder that `api_stats_categories` formats. The top-camera and top-lens subqueries (312-337) read only
  `category, camera_model, lens_model`, which the same index contains.
- key decision: an index for the single request that costs 59 s and 6.8 GB (0.38 s and 17.1 MB with it), plus an
  explicit NO to a start-up warm-up (Q8). The index adds a write cost on every one of the scored columns it contains;
  that is part of G4.
- default: index yes, warm-up no; the constant exists only so the plan test reads the real text (no hand copy).
- verify (`tests/test_stats_category_plan.py`, Wave 2, after Steps 7 and 8 are both in the tree): the constant's
  statement, formatted with an empty `vis`, plans `USING COVERING INDEX idx_stats_categories`; control: drop the index and
  it plans `SCAN photos`. Pin: every `photos` column the statement names is in the index definition (the same silent-
  regression guard as Step 2). Fault: add a column to the SELECT without the index -> the pin fails.
  Live: `io_bytes.py stats_cat` reads <= 50 MB (17.1 MB prototype). The plan is robust without ANALYZE (review check
  `probe_cat_plan.py`).
- affected files: `db/schema.py`, `api/routers/stats.py`, `tests/test_stats_category_plan.py` (NEW).
- docs: CHANGELOG `### Changed`; VIEWER.md Performance mentions the first-start build.

### Step 8 (V8): the `+Inf f_stop` 500 on `/api/stats/categories` and `/api/stats/gear`

- description: in `stats.py` define one SQL fragment, `ROUND(AVG(CASE WHEN f_stop BETWEEN -1e308 AND 1e308 THEN f_stop
  END), 1)`, and use it in place of `ROUND(AVG(f_stop), 1)` in the categories statement (303) and, with its `AS
  avg_f_stop` alias, in `_GEAR_AGGREGATES` (379). Optional (Q6): add `AND f_stop < 1e308` to the aperture histogram query
  (526-527). `r[9] or 0` (352) and the gear zero-fill stay. The focal-length averages (304, 380) are NOT guarded: the
  library holds no non-finite focal value (Domain rules, closes review NOTE 13); a future writer is the EXIF layer's job.
- key decision: a SQL read guard that keeps only finite values (not `< 1000`, not `> 0`), so every group without a
  non-finite row averages byte-identically to today, and a group with one averages its finite rows. No stored-row repair
  (Q5) and no model change: widening `avg_f_stop` to Optional would change the OpenAPI wire type and fail G2. An all-
  non-finite group yields NULL, which the existing `or 0` turns into 0.
- default: as above. The guard uses `f_stop` only, so it stays inside `idx_stats_categories`.
- verify (`tests/test_stats_nonfinite.py`, `init_database` schema, `seed_photos_prefix`, `edition_client`,
  `invalidate_stats_cache()` first; start from `probe_inf/test_probe_inf.py`, which reproduced the 500 in about 1.3 s;
  the seeded category names carry a `uuid4().hex` suffix because the rows go into the shared session DB and other tests'
  rows would otherwise join the averages, review NOTE 11):
  1. seed three rows with `f_stop = float('inf')` in category A and three finite rows in category B: both endpoints
     answer 200, A's `avg_f_stop` is 0 (all non-finite), B's is the finite mean, every other field matches a Python
     oracle;
  2. a mixed category (two finite, one Inf): `avg_f_stop` is the mean of the two finite values;
  3. gear: the same through `/api/stats/gear` for a camera, a lens and a combo group;
  4. with Q6: `/api/stats/settings` aperture list has no `null` `value`.
  Fault injection: put `AVG(f_stop)` back in either place -> that endpoint's test returns 500 (the TestClient raises
  `ResponseValidationError`) and fails; restore by hand.
  Live: after the change both routes answer 200 on the copy; categories rows equal the old statement's rows on every
  field except `avg_f_stop` of the groups that contain the 144 Inf rows.
- affected files: `api/routers/stats.py`, `tests/test_stats_nonfinite.py` (NEW).
- docs (Wave 4): CHANGELOG `### Fixed`.

### Step 9 (V9): capsule `ValueError: matmul ... 1152 is different from 768`

- description: in `_generate_seeded` (1130), the seed vector is decoded at 1219 without the dimension filter that built
  `emb_matrix` (1177). Guard the product: `if seed_emb is not None and emb_matrix is not None and seed_emb.shape[0] ==
  emb_matrix.shape[1]:`. A seed whose dimension differs skips only its visual-similarity axis; the time and location axes
  still run. The other matmul sites (`260-286`, `863-867`) compare filtered rows with filtered rows and need no change.
- key decision: a shape check, not "filter the seeds up front" (which would silently reduce the seed count and change
  the RNG-stable seed selection) and not a catch around the matmul. The error surfaced as one aborted generator because
  the broad `except` in the capsule driver swallows it; the cause is a code defect, not the environment: 60 of the top
  2,000 candidates are 1152-d next to 1,940 at 768-d, so about one 10-seed draw in four hit it.
- default: shape check only.
- verify (`tests/test_capsule_seed_dim.py`, direct-generator pattern of `tests/test_capsule_generator.py:160-215`:
  `init_database(tmp_path / "cap.db")`, `monkeypatch.setattr("api.db_helpers.is_photo_tags_available", lambda *a, **k:
  False)`): insert 6 rows with a 768-float32 `clip_embedding` and 2 with 1152, dated within 7 days so the time axis has
  neighbours, config `{"seeded": {"num_seeds": 10, "min_photos": 4}}` (explicit, <= 6, so "non-empty" does not depend on
  the shipped default of 8 or on all 8 rows being gathered, review NOTE 7; `num_seeds` 10 makes all 8 rows seeds,
  deterministic: `min(num_seeds, len(rows))`), call `cg._generate_seeded(conn, {"seeded": {...}}, min_aggregate=...,
  vis=("1=1", []), user_id=None)`. Assert no exception and a non-empty result (so it cannot pass on an early
  `return []`). Fault injection: remove the shape condition -> `ValueError` and the test fails.
- affected files: `analyzers/capsule_generator.py`, `tests/test_capsule_seed_dim.py` (NEW).
- docs (Wave 4): CHANGELOG `### Fixed`.

### Step 10 (review G-3, NOTE 12): deterministic tie order on album pages, the map and the stats top camera/lens

- description: the new covering indexes change which tied row an `ORDER BY aggregate DESC LIMIT` returns when the ORDER
  BY is not total. Make those ORDER BYs total: pass `tiebreak='photos.path ASC'` to the two `order_by_clause(safe_sort,
  sort_dir)` calls in `albums.py` (188 smart album, 230 regular album; `photos.path` explicit because the regular album
  joins `album_photos`), add `, p2.path ASC` to the map cluster representative (`map.py:60`) and `, path ASC` to the
  individual-photo query (`map.py:93`), and add `, camera_model` / `, lens_model` to the window `ORDER BY COUNT(*) DESC`
  in `stats.py` (318, 331).
- key decision: fix the cause (non-total ORDER BY) rather than narrow the contract and accept a shuffle. Pagination of
  an album with ties was already unstable across plan changes; the review's probe showed page 2 of a smart album sharing
  only 16 of 48 paths once `idx_gallery_category` exists. The claim "identical results" is narrowed (G1) because this step
  changes the tied rows' order on purpose; CHANGELOG says so.
- default: as above; `idx_gallery_category` then serves the album page without a temp b-tree.
- verify (`tests/test_viewer_sql_rewrites.py` for albums/map, `tests/test_stats_nonfinite.py` for top camera/lens; seed
  ties with paths inserted in DESCENDING order so rowid order is the opposite of path order, which makes the un-tiebroken
  statement return rowid order): a smart album and a regular album with 10 rows of equal aggregate, paged 3 at a time,
  concatenate to `sorted(rows, key=(-aggregate, path))` with no duplicate and no gap; the map cluster representative and
  the individual-photo list are the lowest-path row among ties; two cameras with equal counts in one category give the
  alphabetically first as `top_camera` (same for lens). Pattern for albums: `tests/test_album_null_sort.py` (added by
  open-gaps B3). Fault injection: remove any one tiebreak -> its test fails (run each fault, confirm red).
  Live: the gate compares album and map results as SETS against BEFORE and checks the AFTER ordering rule (gate item 5).
- affected files: `api/routers/albums.py`, `api/routers/map.py`, `api/routers/stats.py`, `tests/test_viewer_sql_rewrites.py`,
  `tests/test_stats_nonfinite.py`.
- docs (Wave 4): CHANGELOG `### Changed` ("rows with equal scores now order by path on album pages, the map and the
  per-category top camera/lens, so paging is stable").

### Step 11 (review G-1): index build safety at viewer start

- description: (a) `db/schema.py`: `init_database(db_path, build_photo_indexes=True)`; `conn.commit()` before the photos
  INDEXES loop (1197), so each `CREATE INDEX` then runs in autocommit and releases the write lock when it finishes; for
  each index missing from `sqlite_master` log `logger.info("Creating index %s (a large library can take minutes)", name)`
  before it; the ANALYZE gate stays after the loop (Q1). With `build_photo_indexes=False` the INDEXES loop and the gate are
  skipped. (b) `api/schema_startup.py` `init_schema_at_startup(db_path)`: try `facet.LibraryLock(db_path,
  kind=LIBRARY_JOB_MAINTENANCE).acquire()` (non-forced, imported lazily like `api/routers/scan.py`); on success run
  `init_database(db_path)` and release in `finally`, return `'built'`; on `LibraryLockError` log a WARNING naming the
  holder (`library_job_conflict_message`), run `init_database(db_path, build_photo_indexes=False)` and return `'deferred'`
  (the next start, or `python database.py`, builds them); any other exception is logged and returns `'failed'`, as the old
  lifespan try-block did. (c) the lifespan calls it in place of the bare `init_database` try-block.
- key decision: take the real library lock rather than only peek at it. Every CLI job locks before its schema step
  (`facet.py:2179`); the viewer was the one writer of the library that did not, which is how a 15-minute transaction could
  run unseen under a scan. Holding the lock makes a scan started meanwhile fail fast with the clear "a maintenance is
  already running" message instead of timing out on `database is locked`, and makes `--workers N` safe by construction:
  exactly one worker holds the lock and builds, the others defer and serve (Q21). Skipping the build when a job holds the
  lock follows the coordinator's ruling.
- default: `kind=LIBRARY_JOB_MAINTENANCE` (existing kind, so `recompute_status` and the lock message need no new case).
- verify (`tests/test_gallery_indexes.py` for the schema half, `tests/test_lifespan_startup.py` for the helper; own tmp DBs
  with at least 200 rows):
  1. per-index commit (schema half): monkeypatch `db.schema.INDEXES` so the third entry names a nonexistent column
     (`CREATE INDEX` raises `OperationalError`); after the exception the first two indexes EXIST in `sqlite_master` (they
     were rolled back with everything else before). Fault: delete the new `conn.commit()` -> red;
  2. log line: `caplog` shows one "Creating index" INFO per missing index and none for existing ones;
  3. interrupted start resumes: after (1), restore `INDEXES` and run `init_database` again -> all indexes present;
  4. lock held: hold `LibraryLock(tmp, kind=maintenance)` in the test, call `init_schema_at_startup` -> returns `'deferred'`,
     the photos indexes are NOT created, the warning names the holder; release the lock, call again -> `'built'` and the
     indexes exist. Fault: ignore the lock result and always build -> red;
  5. concurrent starts: two threads call `init_schema_at_startup` on an upgraded DB (the eight indexes dropped) behind a
     `threading.Barrier`; no exception escapes either, at least one returns `'built'`, and after a final call the full index
     set exists. The helper's per-thread connections use the real busy timeout; the test DB is small so the overlap is
     short. **UNVERIFIED on this box** until run: two file descriptors of one process contending for `msvcrt.locking` (the
     builder reports the actual behaviour; fallback is a child process holding the lock);
  6. the lifespan itself still starts: `with TestClient(create_app()):` with `api.database.DEFAULT_DB_PATH` patched to a tmp
     DB reports `init_schema_at_startup` called once.
- affected files: `db/schema.py`, `tests/test_gallery_indexes.py` (Wave 1-B); `api/schema_startup.py`, `api/__init__.py`,
  `tests/test_lifespan_startup.py` (Wave 2-E1).
- docs (Wave 4): DEPLOYMENT.md (first start after an upgrade builds eight indexes in minutes, in their own transactions; a
  running scan defers them to the next start; run `python database.py` first; `--workers N`: one worker builds, the others
  serve), CHANGELOG `### Changed`.

### Step 12 (review G-2): the shutdown checkpoint no longer depends on `wal_checkpoint_minutes`

- description: in the lifespan, after `_flush_pending_rating_syncs()` (297) call `close_change_token_sentinels()`, then run
  the final `wal_checkpoint(TRUNCATE)` block (308-318) unconditionally: remove the `if wal_minutes > 0:` line and
  dedent. `wal_minutes` keeps governing only the periodic thread. The checkpoint connection is a read-write
  `get_db_connection()`, so it is the last closer and deletes the WAL.
- key decision: with `wal_checkpoint_minutes = 0` the read-only sentinel would otherwise become the last closer: probe
  `probe_lastclose.py` left a 12 KB WAL and the `.db` alone showed 0 of 3 committed rows, which breaks every file-level
  copy of the `.db` (rsync workflow in DEPLOYMENT.md, volume backups). Fixing it at the lifespan rather than inside
  `close_change_token_sentinels()` keeps the helper free of write connections.
- default: always checkpoint at shutdown (best-effort, logged on failure, never blocks shutdown, as now).
- verify (`tests/test_lifespan_startup.py`, tmp DB, `api.database.DEFAULT_DB_PATH` patched, `api.config._FULL_CONFIG`
  patched with `{"performance": {"wal_checkpoint_minutes": 0}, "viewer": {"features": {"show_capsules": False}}}` so the
  test never depends on the operator's `scoring_config.json` and no capsule thread holds a connection, review NOTE 16):
  1. with the app running, take one token (opens the sentinel), commit a row from a separate `sqlite3` connection and close
     it; after the `with TestClient(create_app())` block exits, poll up to 5 s for `db + "-wal"` to disappear AND assert the
     row is present when the `.db` is opened `?mode=ro&immutable=1` (the `.db` alone holds every committed row);
  2. the same with `wal_checkpoint_minutes = 30` (regression guard for the old path);
  3. ordering: patch `api.change_token.close_change_token_sentinels` to record a call and the final checkpoint helper to
     record another; assert the sentinels close first.
  Fault injection: restore the `if wal_minutes > 0:` guard -> test 1 goes red; remove the `close_change_token_sentinels()`
  call -> test 1 goes red (the sentinel keeps the WAL alive); restore both by hand.
  Live smoke (gate 8): run once with `wal_checkpoint_minutes = 0` and check `lib.db-wal` is absent after a clean stop.
- affected files: `api/__init__.py`, `tests/test_lifespan_startup.py` (Wave 2-E1).
- docs (Wave 4): CONFIGURATION.md row in 7 languages ("0 disables the periodic thread; the shutdown checkpoint always
  runs"), DEPLOYMENT.md, CHANGELOG `### Fixed`.

## Write-cost accounting

- Size: see the registry, about 106.9 MB (0.65% of 16.3 GB). The VS report's 122 MB figure also counted a timeline index
  that this spec does not ship; the previous revision's 109.0 MB counted a separate lens index and no `junk_kind`.
- Build: 102-106 s per gallery index cold and 4-8 s warm (three measured), 169 s for `idx_stats_categories`, 7.1 s
  ANALYZE; first start after upgrade about 2 minutes warm, up to about 15 minutes cold (Q2), now in separate
  transactions.
- INSERT OR REPLACE (scan, `save_photo`): VS `write_cost.py`, 50k rows: +0.15-0.2 ms per row per index, so at most
  +1.6 ms per photo for eight indexes against hundreds of ms to seconds of GPU time per photo (< 1%).
- `--recompute-average` rewrites `aggregate`, which four new indexes contain (`gallery_aggregate`, `gallery_category`,
  `gallery_camera`, `stats_categories`): about 20-25 s per index per 130k rows, +80-100 s.
- Burst/duplicate/sequence passes rewrite hide columns and now maintain six more indexes per rewritten row (aggregate,
  date desc, date asc, category, camera, iaa_asc): estimated <= 6 x 0.2 = 1.2 ms per row, **UNVERIFIED** (the VS bench
  measured INSERT/REPLACE only). The gate measures an UPDATE of `is_burst_lead` on 10k random rows with and without the
  eight indexes (G4).
- `--detect-junk` rewrites `junk_kind`, which seven new indexes now contain: a one-off pass of about 7 x 0.2 ms per
  evaluated row (about 3 minutes at 130k, estimated, **UNVERIFIED**); this is the price of Step 4.
- User `is_rejected` toggles touch six indexes: microseconds.

## Build waves

Every builder is pinned to `sonnet` (Builder role) by the orchestrator and told: "Do the work yourself; do NOT spawn
sub-agents or background agents." Builders run ONLY their own tests; the orchestrator owns the gates and every git verb;
builders never run state-changing git, never touch the production DB or any photo file, and never write
`.compact-gate`. The orchestrator commits locally after each verified wave with NO AI or session attribution in commit
messages (the user's standing rule and the repo's commit-hygiene CI override any default trailer). Each brief names its
exclusive file list, the steps' text above, and `.claude/patterns/test-fixtures.md`.
`PY=/d/photo-llm/conda/python.exe`.

- Wave 0 (orchestrator, no builder). Open-gaps-120 Steps 1-5 are committed at `e38e64e0`; any later open-gaps step that
  touches `db/schema.py`, `api/__init__.py`, `api/routers/stats.py`, `albums.py` or the docs must be committed first
  (working tree at spec time also carried uncommitted `db/maintenance.py` and three test files from another builder: not
  touched here). Record the baseline failing-test set on that tree (command in the gate). Confirm no scan holds the GPU or
  the DB (`nvidia-smi`, `tasklist`) and `D:` has >= 20 GB free. Make the working copy with the backup API
  (`D:\facet-scratch\viewer_spec\lib.db` from `lib_copy.db`), start the PRE-change viewer on it, capture the BEFORE JSON
  for the case matrix (about 15 minutes, `junk_any` is about 6 of them, plus the album, map and multi-user cases) and the
  old categories statement result (59 s), then stop that viewer. This runs in the background while Wave 1 builds.
- Wave 1 (four builders in PARALLEL, disjoint file sets):
  - A = Step 1: `api/change_token.py`, `api/db_helpers.py`, `tests/conftest.py`, `tests/test_change_token.py`.
  - B = Steps 2, 3, 4, 6, the Step 7 index entry and the schema half of Step 11: `db/schema.py`,
    `tests/test_gallery_indexes.py`.
  - C = Steps 5, 10 (albums, map): `api/routers/filter_options.py`, `api/routers/albums.py`, `api/routers/map.py`,
    `tests/test_viewer_sql_rewrites.py`.
  - D = Steps 7 (constant only), 8, 9, 10 (stats top camera/lens): `api/routers/stats.py`,
    `analyzers/capsule_generator.py`, `tests/test_stats_nonfinite.py`, `tests/test_capsule_seed_dim.py`.
  - Phase 1b (orchestrator, tree quiet): `ruff check` on the touched files, each builder's tests, then
    `cd client && npm run gen:api` and `git diff --stat client/src/app/core/api/schema.d.ts` must be empty; local commit.
- Wave 2 (two builders in PARALLEL, after Wave 1 is committed; both need Wave 1 code):
  - E1 = Steps 1 (lifespan half), 11 (helper + lifespan), 12: `api/__init__.py`, `api/schema_startup.py`,
    `tests/test_lifespan_startup.py`. It needs A's `close_change_token_sentinels` and B's `build_photo_indexes` argument,
    which is why it is not in Wave 1 (a lifespan calling a missing keyword would be swallowed by its broad `except` and
    pass silently).
  - E2 = Step 7 plan test: `tests/test_stats_category_plan.py`. It needs B's index and D's constant.
  Local commit after both.
- Wave 3 (orchestrator, no repo files): start the viewer on the working copy with the post-change tree; the first start
  runs `init_schema_at_startup` and builds the eight indexes (time it and capture the INFO lines: this is the number the
  docs quote); capture AFTER, diff against BEFORE, run the HIT and MISS timing suites (>= 5 reps on the headline rows, 1
  cold + warm), the write-cost bench and `io_bytes.py`. Results go to `D:\facet-scratch\viewer_spec\` (outside the repo).
- Wave 4 (ONE docs builder, after Wave 3, so every quoted number is measured): `CHANGELOG.md`, `CLAUDE.md`,
  `docs/{,de,es,fr,it,pt,zh}/{VIEWER,DEPLOYMENT,COMMANDS,CONFIGURATION}.md`. One builder, because the seven translations
  and the CHANGELOG must say the same thing. It rebases onto open-gaps-120's docs edit first (same `[Unreleased]` block).
- Final (orchestrator): the gate below, then an independent review of the integrated diff by a fresh opus Refuter that
  re-runs the gates itself (`workflow-kit/0.13.0/context/reference/refuter-checklist.md`).

### open-gaps-120 conflict notes

open-gaps-120 (`.claude/specs/open-gaps-120.md`) Steps 1-5 are committed at `e38e64e0` (B2 removed the stats handler, B3
edited `albums.py`, B5 added the `iqa_declined` hunks to `db/schema.py`), so the earlier "wait for B2/B5" prerequisites
are met. What remains is its later work and docs; this spec adds new test files only, so there are no test-file
collisions.

| File | open-gaps step | This spec | Resolution |
|---|---|---|---|
| `api/routers/stats.py` | B2 deleted the `categories/metrics` handler (done) | V7/V8/Step 10 edit 296-337, 371-410, 526-527 | disjoint hunks, lines above 925 unchanged |
| `db/schema.py` | B5 added `IQA_DECLINED_COLUMNS`, `IQA_SCAN_FINGERPRINT_SQL`, a `_MIGRATED_TABLES` entry and an `init_database` creation block (done, +27 lines) | `INDEXES` tail (434), the index loop and gate (1197-1214), the `init_database` signature | disjoint hunks; locate by quoted text |
| `api/routers/albums.py` | B3 (done): sort NULL handling at the same two `order_by_clause` calls | Step 10 adds the tiebreak argument to those two calls | same lines, different arguments; re-read the file at HEAD before editing |
| `tests/test_stats.py` | B2 | not touched | none |
| `CHANGELOG.md`, `CLAUDE.md`, `docs/*/VIEWER.md` | their docs wave | Wave 4 | mine after theirs; both write `[Unreleased]` and use `### Fixed/Changed` |
| `facet.py`, `cli_args.py`, `processing/*`, `db/maintenance.py` | B1, B4, B5, B6 | not touched (`facet.LibraryLock` is imported, not edited) | none |

## Documentation plan (Wave 4; CLAUDE.md: a behaviour change updates its docs in the same change, 7 languages)

Located by language-neutral tokens, the translated headings differ (`## Performance` / `## Performances`).

- `CHANGELOG.md` `[Unreleased]` (currently empty, line 5): `### Fixed` for V1 (the count and hidden-summary caches
  never hit), V4, V8, V9, and the shutdown checkpoint with `wal_checkpoint_minutes = 0` (Step 12); `### Changed` for the
  eight indexes (first start builds them, order of minutes on a 130k library, each in its own transaction, deferred while
  a library job runs, run `python database.py` first), the tags dropdown, and "rows with equal scores now order by path on
  album pages, the map and the per-category top camera/lens, so paging is stable" (Step 10).
- `CLAUDE.md`: Performance Optimizations -> "Query Optimizations in api/": replace "COUNT result caching (5 minute TTL)"
  with the sentinel-keyed wording; under "Invariants that will bite you" add (a) the change token comes from one read-only
  connection per DB, is host-local like `LibraryLock`, is keyed on the DB path read at call time, and the sentinel must close
  before the final checkpoint, (b) a new hide column must join `GALLERY_VISIBILITY_COLUMNS` or the gallery silently reverts to
  table passes, (c) `junk_kind` rides in the gallery covering indexes on purpose, (d) an ORDER BY that feeds a paged
  list must be total (end in `path`), because a new index can reorder ties, (e) the viewer takes `facet.LibraryLock`
  around its start-up schema step.
- `docs/VIEWER.md` `## Performance` (the block with `python database.py --migrate-tags`): a short "Gallery indexes and
  count cache" paragraph (what the covering indexes are for, first-start build cost, the cache is invalidated by any
  commit including the CLI), and the Statistics Cache paragraph keeps its 5-minute TTL wording.
- `docs/DEPLOYMENT.md` near `performance.wal_checkpoint_minutes` (line 758): the viewer keeps one read-only DB connection,
  so `-wal`/`-shm` persist while it runs and the file cannot be replaced under it (stop the viewer first); the shutdown
  checkpoint always runs; the upgrade note (pre-create with `python database.py`, Docker healthcheck `start_period`,
  `--workers N` builds once).
- `docs/COMMANDS.md` Database Maintenance row ``| `python database.py` |``: "also builds the gallery covering indexes".
- `docs/CONFIGURATION.md` row `wal_checkpoint_minutes` (line 2007 and the six translations): "Set to `0` to disable the
  periodic thread; the checkpoint on clean shutdown always runs."

## Final verification gate

Run after Wave 4, on a tree no builder is editing. Every command writes to a file and the exit code comes from
`${PIPESTATUS[0]}` or a redirect, never from a pipe; filtered output is capped.

1. New tests: `"$PY" -m pytest tests/test_change_token.py tests/test_gallery_indexes.py tests/test_viewer_sql_rewrites.py
   tests/test_stats_nonfinite.py tests/test_capsule_seed_dim.py tests/test_lifespan_startup.py
   tests/test_stats_category_plan.py -q -p no:cacheprovider`; report the file count, the test count and the exit code.
2. Full suite: `"$PY" -m pytest tests/ -q -p no:cacheprovider -rf > .claude/.scratch/viewer-after.txt 2>&1; echo
   ${PIPESTATUS[0]}`; the failing set must equal Wave 0's baseline (the two known box-local classes: viewer-password
   401s and Windows `Path.resolve()` POSIX-path tests). `tests/test_api_contract.py` green.
3. `"$PY" -m ruff check .` exit 0; `cd client && npm run gen:api` then `git diff --stat
   client/src/app/core/api/schema.d.ts` empty (G2).
4. Fault injections of Steps 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12 each shown red then restored, with a byte-identical
   `git diff -- <file>` snapshot compared after the hand restore. Disclose per test file whether any assertion was adapted
   (expected: none; all test files are new, `tests/conftest.py` only gains a fixture).
5. Equivalence (G1) on the working copy, through the real HTTP routes, BEFORE JSON vs AFTER JSON, case matrix: default,
   page 2, page 500, `date_taken` DESC/ASC, `aggregate` ASC, `aesthetic_iaa` ASC and DESC, `star_rating`,
   `category=portrait`, `category=landscape`, `camera=Canon EOS R6`, a second camera, `lens=<one>`, `junk_kind=any` (sorted
   by aggregate and by date_taken), `type=top_picks`, `tag=street`, `person=<one>`, `min_score=7`, `date_2024` sorted by
   date, and the no-hide variant; per case compare `total`, the hidden summary and the ordered `path` list. Plus: the 89
   tag pairs; the categories rows against the old statement (59 s direct SQL) with the stated `avg_f_stop` and top
   camera/lens (compared by count, not name, where the top count ties) exceptions; and the Step 10 cases, compared as SETS
   against BEFORE (a smart-album page 2 requested with `per_page` large enough to return the whole album, a regular album
   page, a map cluster list and the individual-photo list) plus the AFTER ordering rule `(aggregate DESC, path ASC)` checked
   on the paged album requests. Any difference blocks.
   5b (optional, working copy only, after AFTER is captured): `UPDATE photos SET junk_kind = 'screenshot' WHERE rowid % 3
   = 0` (about 43k rows), then the `junk_kind=any` count/summary/page for both sorts must stay <= 400 ms warm (dense
   case on the real rows; the synthetic 130k run measured 40/33/0.6).
6. Speed (G3): the timing table re-measured as HIT p50 (live HTTP) and MISS p50 (in-process `bench_miss.py` in
   `D:\facet-scratch\viewer_spec\`: create the app against the working copy, `api.config._count_cache.clear()` before every
   rep; >= 5 reps on the headline rows, p50 and p95, cold first request); one multi-user gallery page before and after (Q11);
   the VS report's FULL case list once (Q12). Slow-request log count for the run recorded.
7. Write cost (G4): INSERT OR REPLACE and the `is_burst_lead` UPDATE bench, with and without the eight indexes, bounds in
   G4; first-start build time and per-index INFO timestamps recorded (the number the docs quote); measured index sizes
   against the registry estimate.
8. Live smoke: `/api/stats/categories` and `/api/stats/gear` answer 200 on the copy; `/api/capsules` after start logs no
   `ValueError: matmul`; after a clean viewer shutdown `lib.db-wal` is absent, once with the default
   `wal_checkpoint_minutes` and once with `0` (Steps 1, 12); a start while a `python facet.py --recompute-average` style job
   holds the lock logs the deferral and builds nothing (Step 11).
9. Browser (user rule: test at the end of every change): gallery page, junk filter, a category filter, the tags dropdown, an
   album and the stats page against the copy, frozen build, screenshots saved outside the repo.
10. `git status --porcelain` lists only the files in "Files touched" plus whatever open-gaps-120 still has in flight.

## Evidence index (session scratchpad and scratch dirs; read-only use of the repo)

`C:\Users\laure\AppData\Local\Temp\claude\D--photo-llm\01873979-9070-498f-a0cb-74e72a51ba83\scratchpad\`: `old_token_exp.py`
(old token never repeats), `tok_exp.py` + `tok_child.py` (token candidates, collision tests), `plan_exp2.py` /
`plan_exp3.py` (plans with and without ANALYZE and the new indexes), `junk_page.py` / `junk_page2.py` / `junk_page3.py`
(junk forms on the copy), `inf_cols.py` (non-finite scan), `emb_dims.py` (embedding dimensions), `probe_inf/test_probe_inf.py`;
this revision: `junk_cover130.py` + `junk_cover130.out` (junk covering form at 130k, sparse and dense, real route SQL),
`wide_vs_agg.py` + `wide_vs_agg.out` (default statements on the two aggregate-leading indexes), `inf_focal.py` (focal-length
non-finite check), `libdb_idx.py` (prototype index definitions and junk distribution). Review probes (session scratchpad
`refute_viewer\`): `probe_ties.py`, `probe_init_txn.py`, `probe_cat_plan.py`, `probe_lastclose.py`, `probe_junk_cover.py`.
`D:\facet-scratch\viewer\`: the VS report scripts (`bench.py`, `suites.py`, `sqlgen.py`, `qplan.py`, `verify_equiv.py`,
`proto_idx.py`, `tags_rw.py`, `dedupe_chk.py`, `write_cost.py`, `io_bytes.py`, `cache_probe.py`, `hold_conn.py`).

## Adversarial review

Independent opus Refuter, report `.claude/.scratch/refute-viewer-api-speed.md`: 1 BLOCKER, 6 GAP, 16 NOTE, judged against
the pre-patch spec. Coordinator adjudication applied below; "applied" names the section of this revision.

### BLOCKER

| ID | Finding | Ruling | Applied |
|---|---|---|---|
| B1 | Token and cache key read `db_helpers`' import-time `DEFAULT_DB_PATH`; the tests patch `api.database.DEFAULT_DB_PATH`; count/summary equivalence and DROP-INDEX controls compared the cache with itself; Q13 contradicted Step 2 | Accepted | Step 1 (call-time path, key includes the path, autouse fixture, tests 6-7 with the path-binding fault); Steps 2, 3, 6 (explicit cache clear per side, empty-cache assertion); Q13 rewritten; Domain rules row |

### GAP

| ID | Finding | Ruling | Applied |
|---|---|---|---|
| G-1 | `init_database` builds all indexes and ANALYZE in one write transaction without the library lock | Accepted: per-index autocommit, log before each, ANALYZE after; skip the build when the library lock is held; `--workers N` safe; tested | Step 11, Q2, Q20-Q22, G6, wave E1, DEPLOYMENT docs. Implementation takes the real lock (not only a peek), reason in Step 11 |
| G-2 | `wal_checkpoint_minutes = 0` leaves the sentinel as last closer; WAL holds commits the `.db` lacks | Accepted: always close the sentinel then checkpoint at shutdown | Step 12, Q19, CONFIGURATION.md in 7 languages, tested with `wal_checkpoint_minutes=0` |
| G-3 | New indexes change tie order for non-total ORDER BYs (album pages, map) | Accepted: add path tiebreaks, narrow the claim to sets / total orders, add albums and map to the gate, CHANGELOG note | Step 10, G1, gate 5, CHANGELOG line |
| G-4 | The IN-subquery junk rewrite regresses the dense distribution and was measured on 24 rows | Accepted: covering index with `junk_kind`, original predicate kept, subquery dropped; measured at 130k | Step 4, rows 14-15, Q17 (residual for sorts without a junk-carrying index). `gallery.py` is no longer touched |
| G-5 | G3 timings pass on cache hits | Accepted: time misses | G3, timing table (HIT and MISS columns), gate 6 (`bench_miss.py`, in-process, cache cleared per rep) |
| G-6 | Step 1 test 3 (`wal_checkpoint` succeeds) cannot fail | Accepted: assert `busy == 0` and an empty WAL, pinned-reader fault | Step 1 test 3 |

### NOTE

| # | Finding | Ruling | Applied |
|---|---|---|---|
| 1 | Stale premise: open-gaps B1-B4 already committed, stats handler gone | Accepted; re-based on `e38e64e0` | Header, Wave 0, conflict table |
| 2 | DB replaced under a running viewer freezes `data_version` | Accepted: fold `st_dev`/`st_ino` into the token and reopen on change; POSIX-only test | Step 1, Q18, DEPLOYMENT docs |
| 3 | SMB multi-host: `data_version` is host-local | Accepted as documentation | Q18, CLAUDE.md invariant (a) |
| 4 | Sync token on the event loop can wait on a shm lock | Accepted: `timeout=0.1`, busy returns a miss token | Step 1 description, test 9 |
| 5 | `idx_gallery_lens` makes `idx_gallery_aggregate` redundant | Accepted and measured: one aggregate-leading superset index named `idx_gallery_aggregate`, no separate lens index (+3-4 ms on the cached count/summary, nothing on the page) | Registry, Step 2, row 21, Q3 |
| 6 | ANALYZE gate refires on every empty-DB init | Accepted: skip when `photos` is empty | Q1, Step 2 test 4 |
| 7 | Step 9 test depends on `min_photos` | Accepted: set `min_photos` explicitly (4) | Step 9 verify |
| 8 | Step 1 test 5 cannot distinguish a generation reset | Accepted: read a token before the first commit | Step 1 test 5 |
| 9 | Step 4 INTEGER seed becomes TEXT | Accepted: claim dropped, BLOB kept | Step 4 verify 3 |
| 10 | Step 5 tags may be served from `stats_cache` | Accepted: delete the `'tags'` row / patch `get_cached_stat` | Step 5 verify |
| 11 | Step 8 seeds share the session DB | Accepted: `uuid4` category suffix | Step 8 verify |
| 12 | Top camera/lens window has no tiebreak | Accepted: add `camera_model` / `lens_model` | Step 10, G1 |
| 13 | `AVG(COALESCE(focal_length_35mm, focal_length))` may hold Inf | Checked: `focal_length_35mm` is NULL on every row, `focal_length` max 61,440, no non-finite; not guarded | Step 8, Domain rules |
| 14 | Multi-user not timed | Accepted: time one multi-user page before/after | Q11, gate 6 |
| 15 | Q12 re-ran only 12 cases | Accepted: full VS list once | Q12, gate 6 |
| 16 | Step 1 test 6 reads the operator's `_FULL_CONFIG` | Accepted: patch `_FULL_CONFIG` | Step 12 verify |

### Adjudication caveats

- Q17: the junk ruling leaves a sparse-junk request sorted by a column with no `junk_kind`-carrying index on the old plan.
  This is unchanged from today, not a regression; the subquery form would have fixed it for every sort but loses on the
  dense case, so the ruling stands on its stated criterion. Offered as a follow-up index.
- Step 11 takes `LibraryLock` instead of only peeking at it; a peek would leave a scan that starts during a 15-minute build
  to time out on the write lock.
- The two-descriptor lock concurrency test (Step 11 test 5) is UNVERIFIED on Windows until it runs.
