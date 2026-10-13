# Issue #185 — decisions taken autonomously (feeds the final summary)

| # | When | Decision | Why |
|---|---|---|---|
| D1 | W0 setup | Evaluation runs on `photo_scores_eval.db`, a SQLite-backup copy of the prod DB (quick_check ok, 129,737 photos). The prod DB is never opened. | User rule. |
| D2 | W0 setup | Ground truth is **393** decisive pairs (203 a / 190 b), with 48 ties and 6 skips, not the 515 the plan assumed. | Counted on the copy. |
| D3 | loop | The oracle picks `venv/bin/python` if present, else `conda/python.exe`. tsc checks `tsconfig.app.json` + `tsconfig.spec.json`. The usage halt is at 80% and covers both windows, because the plugin has one floor setting. | Make the gate portable and real on this Windows box. |
| D4 | W0 | Live candidates are `qrealign-{mini,lite}:{aesthetic,quality}`, registered under both the aesthetic and quality slots. Stored candidates are `stored:<col>`. End-to-end candidates are named `<slot>:<cand>`. | The builder's naming; kept. |
| D5 | W0 | The temp-DB test fixture uses journal_mode=DELETE so the unchanged-sha256 check means something. A read-only WAL open can touch the -shm file. | Builder finding. |
| D6 | W0 | No `path_mapping` is needed on this box: the stored `\Sweet\home\Photos\…` UNC paths open directly. All 680 distinct photos in the decisive pairs are reachable. | Checked live. |
| D7 | loop | The test gate deselects the 47 tests that already fail or error on this Windows box at pristine `d7beae3` (`.claude/loop/known-env-failures.txt`, local only). Every other test must pass. | Baseline: 41 failed + 6 errors, 4437 passed, in a clean worktree. |
| D8 | loop | maxIterations is raised from 40 to 150. | Ten waves, and each Stop-hook block while waiting on background builders costs one iteration. |
| D9 | W0 | The end-to-end slot never constructs a full `Facet` (whose init migrates the DB and switches it to WAL). It rebuilds the percentile normalizer read-only from `stats_cache.percentiles_snapshot`. The baseline then reproduces stored aggregates 100% when the stored category is kept, and 78% under the current category rules (category drift). | Opus refuter found the harness was writing the DB it evaluated. |
| D10 | W0 | Baseline agreement with the user's 393 picks: stored `aesthetic` (CLIP-MLP, this is a CLIP-768 library) 0.636 [0.590–0.682], `aesthetic_iaa` 0.604, `aggregate` 0.589, `topiq_score` 0.578. | Smoke run on the eval copy. |
| D11 | W3 | W3 is split. W3a (the standalone pyiqa-free loader plus harness candidates) runs in parallel with W1 review. W3b (routing into model_manager/multi_pass) waits for W2. | Avoid edit conflicts in model_manager/multi_pass. |
| D12 | W3a | The native loader matches pyiqa exactly (max abs diff 0.0, CPU, 3 images, both tasks) with no pyiqa import. It is reviewed together with W3b rather than separately. | One review over the complete W3. |
