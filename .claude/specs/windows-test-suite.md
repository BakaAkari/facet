# Spec: full test suite green on the Windows dev box (47 box-local failures)

Evidence base: each of the ten failing files was run on its own on 2026-10-15 (Python 3.13.11, pytest 9.0.2,
`D:/photo-llm/conda/python.exe`); output kept in `C:\Users\laure\AppData\Local\Temp\claude\D--photo-llm\01873979-9070-498f-a0cb-74e72a51ba83\scratchpad\winsuite\*.txt`.
Every proposed fix below was prototyped in `scratchpad\winsuite\trial\` (copies of the test files, a lazy-patch plugin for the one
product change, wheels unzipped onto `PYTHONPATH` rather than installed) and re-run; results are quoted per step.

Revision 2 (2026-10-15): patched with the coordinator's rulings on the adversarial review (`.claude/.scratch/refute-windows-test-suite.md`); the finding-by-finding ledger is the last section.

## Objective + success criteria
- Run from the **worktree `D:\photo-llm-win`** (step W0), with the `pyt` function defined in W0 (TMP/TEMP/TMPDIR and `--basetemp` on `D:/facet-scratch/tmp`, and `PATH` carrying the main checkout so the untracked `exiftool.exe` is found): `pyt tests/ -q -p no:cacheprovider -rs` on this box: **0 failed, 0 errors**. Skips are allowed only if they are in the skip ledger of the final gate (count stated, every reason listed, diffed against master).
- No assertion is weakened. Two expectations are corrected because they were wrong on their face (see steps 4 and 6); both are called out where they occur. A test that mixes a portable assertion with a POSIX-only one is **split**, never skipped whole (steps 2, 4, 8).
- Linux CI behaviour changes in exactly two product places, both argued below: (1) the multi-user visibility clause, which on a case-sensitive host now matches case-sensitively (a *deliberate* fix of a cross-user leak, CHANGELOG-noted), and (2) the backup writer, which gains an identity check. Every test edit keeps the same assertion on POSIX.
- Three real product defects found while triaging are fixed, each with a regression test that fails-before on Linux too:
  1. `get_visibility_clause` never matches a Windows-stored path (multi-user isolation is empty on Windows), and over-matches by case on POSIX.
  2. `write_owner_only_backup` has no symlink refusal on Windows (`O_NOFOLLOW` is absent there) and, with `O_TRUNC` in its flags, would truncate a raced-in link target.
  3. CI itself does not install `jsonschema` or `pytest-timeout`, so 6 schema tests skip silently on Linux and the `timeout` marker is inert.
- Ships as **one separate PR**: branch `fix/windows-test-suite`, built in its own worktree, cut from `master` after `fix/open-gaps-120` has merged. Commits and the PR body carry **no attribution lines** (`.github/workflows/commit-hygiene.yml` fails CI on `Claude-Session:` and `claude.ai/code/session`; the project rule "No AI attribution in commits/PRs" outranks any harness reminder to add them).
- Gates run exactly as CI runs them: `ruff check . --select E,F,W --ignore E501` (tracked files only here) **and** the mypy ratchet.

## Triage result (47 tests)

Classes: (a) environment drift, (b) test not hermetic, (c) real Windows bug in product code, (d) POSIX-only feature / POSIX assumption in the test.

| Class | Count | Tests |
|---|---|---|
| (a) | 11 | `test_cull` x7 (`send2trash` missing), `test_raw_decode` `TestDngJxlPreviewFallback` x4 (`imagecodecs` missing) |
| (b) | 0 | The 401 hint did **not** reproduce: there is no `scoring_config.json` at the repo root right now (`ls` fails; only `scoring_config.local-backup*.json` / `scoring_config.discovered.json` exist and nothing reads them). Not one of the 47 failures is a 401. See step 10 for the latent risk and why `FACET_CONFIG` must not be pinned. |
| (c) | 2 | `test_auto_cull_cli` `TestRejectStandaloneBelow` multi-user x2: `api/db_helpers.py:1218` |
| (d) | 34 | see per-test table |

(c) also has one **latent** product defect with no currently-failing test: `api/config_writes.py:48-49,99` (symlink refusal depends on `O_NOFOLLOW`, absent on Windows; the 3 symlink tests only error out earlier, at `symlink_to`, because this box lacks the symlink privilege), and one **pre-existing POSIX-side** defect of the same function as the Windows failure: `get_visibility_clause`'s bare `LIKE` folds ASCII case, so on Linux a user scoped to `/mnt/photos` also sees `/mnt/Photos/a.jpg` (cross-user leak; reviewer note 4, ruling: fix, step 1).

### Per-test evidence

| File::test(s) | Exact error | Root cause (file:line) | Class |
|---|---|---|---|
| `test_cull.py::TestCullCapabilities::test_the_real_probe_finds_the_installed_package` | `assert gallery_module.HAS_SEND2TRASH is True` -> False | `send2trash` not installed in `conda/`; declared base dep `requirements.txt` + `pyproject.toml`, lock `Send2Trash==2.1.0` | a |
| `test_cull.py::TestPhotoDelete::{test_dry_run_reports_would_trash_and_writes_nothing, test_not_found_path_never_reaches_disk_resolution, test_not_visible_path_reported_and_untouched}`, `TestPhotoDeleteBracketLead::{test_bracket_lead_refused_without_flag, test_bracket_lead_written_by_the_real_pass_is_still_refused}`, `TestPhotoDeleteSequenceSiblings::test_dry_run_reports_sequence_siblings` | `assert 400 == 200` | `api/routers/export.py:971-982` `_require_trash_available()` raises 400 "send2trash ships with Facet" when the import fails; every one of these posts `/api/photo/delete` | a |
| `test_raw_decode.py::TestDngJxlPreviewFallback` x4 | `KeyError: "<COMPRESSION.JPEG: 7> requires the 'imagecodecs' package"` while building the fixture DNG (`test_raw_decode.py:1090`) | `imagecodecs` not installed; tifffile falls back to its stub `tifffile._imagecodecs` with no `jpeg_encode`. Declared base dep, lock `imagecodecs==2026.3.6` | a |
| `test_auto_cull_cli.py::TestRejectStandaloneBelow::{test_multi_user_writes_user_preferences_not_the_global_column, test_multi_user_only_shields_a_photo_in_the_same_users_culling_pair}` | `assert [] == ['C:\\Users\\...\\low.jpg']` | `api/db_helpers.py:1218` builds the visibility prefix as `d.rstrip('/\\') + '/'` and matches it with a bare `LIKE` against `photos.path`; the scanner stores `Path.resolve()` (backslashes on Windows, `api/db_helpers.py` docstring of `build_path_scope`), so `C:\x/` never prefixes `C:\x\low.jpg`. Result: every multi-user query on Windows sees no photos. `gallery.py:559` and `folders.py:57` already normalise with `REPLACE(path,'\\','/')`; this one does not | **c** |
| `test_auto_cull_cli.py::TestPathScope` x6 (`test_a_directory_prefix_stops_at_the_separator`, `test_wildcards_and_case_are_literal`, `test_sql_agrees_with_matches`, `test_a_file_is_an_exact_entry`, `test_a_denormalised_path_equals_the_clean_one`, `test_a_missing_entry_raises`) | `FileExistsError: [WinError 183]` in the `dirs` fixture (`test_auto_cull_cli.py:126`); reported as 5 FAILED/ERROR + 1 ERROR, all setup errors | fixture does `mkdir()` for both `shoot` and `Shoot`; NTFS is case-insensitive so the second is the same directory. Product is fine: `PathScope` is documented as plain case-sensitive string comparison on resolved paths (`api/db_helpers.py:1109-1120`) and `os.path.realpath` canonicalises the on-disk case on Windows (probe `case_probe.py`: `realpath(.../Shoot)` gives `.../shoot`). The property that verdict rests on is asserted by a NEW case-folding test (step 3) | d |
| `test_config_merge.py::TestShippedDefaults::test_they_live_inside_the_config_package` | `'D:\\photo-llm\\config\\scoring_config.default.json'.endswith('config/scoring_config.default.json')` False | test hard-codes `/`; `defaults_path()` (`config_resolve.py`) uses `os.path.join`. Wrong expectation, not a product bug | d |
| `test_config_merge.py::TestCompactConfig::test_it_leaves_an_owner_only_backup` (`:381-391`) | `438 == 384` (0o666 vs 0o600) at `:391` | NTFS has no group/other bits; `os.chmod` only toggles read-only. `api/config.py:172-182` already documents this and gates its own mode checks on `_POSIX_FILE_MODES`. Only `:391` is POSIX; `returncode == 0` and "exactly one backup is created" are portable and must keep running | d (split, step 4) |
| `test_config_merge.py::TestInPlaceCommitStillWritesTheDelta::test_a_denied_chown_still_persists_only_the_override` | `AttributeError: <module 'os' (frozen)> has no attribute 'chown'` at `monkeypatch.setattr(os, "chown", _refuse)` (`:489`) | `os.chown` is POSIX-only. Product degrades correctly: `config_resolve.py:462-467` documents that on Windows "the replace route is the only correct one" and `_adopt_owner` returns True when `not hasattr(os, 'chown')`, so the in-place route cannot run in production there; planting a fake `os.chown` would pin code the module says must not run | d (skip, step 4) |
| `test_config_path_env.py::TestAnUnreadableConfigStillStartsTheServer::test_a_config_it_cannot_open_boots_locked_rather_than_crashing` | `assert False is True` (`load_failed`) | `chmod(0o000)` does not make a file unreadable on NTFS, so the config loads. Windows DOES support an unreadable file (an ACL deny); only the mechanism is POSIX. Replayed with `icacls /deny *S-1-1-0:(RD)`: `open()` raises `PermissionError 13`, the boot subprocess returns rc 0 and `{"load_failed": true, "open_viewer": false, "any_feature_on": false}` (reviewer probe `unreadable_boot.py`) | d -> **ported, not skipped** (step 5) |
| `test_config_path_env.py::TestALooseConfigModeIsReported::{test_a_world_readable_config_is_reported, test_the_mode_is_left_exactly_as_the_operator_set_it}` | `'readable beyond its owner' in ''`; `438 == 420` | `api/config.py:528` `_group_or_other_readable_mode` returns 0 when `not _POSIX_FILE_MODES` (deliberate); `chmod(0o644)` stats back as 0o666 | d |
| `test_config_path_env.py::TestALooseConfigModeIsReported::test_a_symlinked_config_is_reported_through_the_link` | `OSError: [WinError 1314]` at `link.symlink_to(target)` (`:741`) | no `SeCreateSymbolicLinkPrivilege`; and the same POSIX mode gate would suppress the warning anyway | d |
| `test_config_writes.py::TestTheBackupWriterRefusesASymlink::{test_it_writes_nothing_through_the_link, test_it_does_not_re_mode_the_target, test_it_says_why_rather_than_failing_silently}` | `OSError: [WinError 1314]` at `backup.symlink_to(victim)` (`:1044,1060,1073`) | symlink privilege missing. The third assertion path would also expose the latent product gap (see "Triage result" and step 2) | d (+ latent c) |
| `test_config_writes.py::TestTheBackupWriterRefusesASymlink::test_an_ordinary_backup_is_unaffected` | `438 == 384` at `:1088` | the content half of the test is portable and passes; only the final `S_IMODE == 0o600` is POSIX | d |
| `test_manifest_builder.py::TestPendingCorrectionsCount::test_root_scope_excludes_applied_and_other_root_groups` | `assert 0 == 1` (`:203`) | test seeds `f"{lib_a}/g1.jpg"` (mixed separators); `processing/xmp_export.py:653-656` `build_root_filter` correctly prefixes with `os.path.join(abs_root, "")` = `...\lib_a\`. Production rows come from `Path.resolve()`, so the product is consistent; the test literal is the defect | d |
| `test_system_memory.py::TestSysconfTotalBytesItself::{test_it_reports_this_machines_real_physical_ram, test_it_agrees_with_psutil_to_within_a_page}` | `assert None is not None`; `TypeError: NoneType - int` (`os.sysconf('SC_PAGE_SIZE')` also absent) | `os.sysconf` is POSIX-only. `utils/system_memory.py:299-306` catches `AttributeError` and returns None (correct); psutil is the Windows reader (`_host_total_bytes`, `:309-312`) | d |
| `test_system_memory.py::TestSysconfTotalBytesItself::{test_a_platform_without_the_names_reports_nothing x3, test_a_nonsense_reading_is_no_reading x4}` | `AttributeError: module 'os' has no attribute 'sysconf'` at `monkeypatch.setattr(system_memory.os, 'sysconf', ...)` | `monkeypatch.setattr` defaults to `raising=True`; the fake can be installed on Windows with `raising=False` and the function body then runs for real | d |
| `test_weight_optimizer_sources.py::TestApplyPreservesConfigPermissions::{test_apply_preserves_owner_only_mode, test_apply_preserves_a_group_readable_mode}` (`:383-389` and below) | `438 == 384`; `438 == 416` | POSIX mode bits (the sibling at `:424` is already `skipif win32`). In the first test only `:389` is POSIX; `:388` ("`viewer.password` survives `apply_optimized_weights`") is portable. The second is mode-only | d (first split, second skipped whole; step 8) |
| `test_xmp_export.py::TestExiftoolVersionGuard::test_write_metadata_refuses_and_leaves_sidecar_untouched`, `TestPostWriteValidation::{test_malformed_output_restores_original_bytes, test_malformed_new_sidecar_is_removed, test_embedded_original_is_not_validated, test_export_sidecars_counts_and_logs_error}` | `OSError: [WinError 193] %1 n'est pas une application Win32 valide` from `subprocess.run` (`processing/xmp_export.py:588`) | `_fake_exiftool` (`test_xmp_export.py:419-432`) writes a `#!/bin/sh` script; CreateProcess cannot execute it. Product is correct (`OSError` surfaces to `export_sidecars`, which logs it) | d |

Counts: a=11, b=0, c=2, d=34 (6 PathScope + 1 sep-literal + 1 merge-mode + 1 chown + 4 path_env + 4 writes + 1 manifest + 9 sysconf + 2 weight-opt + 5 xmp) = 47.

## Dependency drift (installed `conda/` vs declared)

Checked with a floor/lock comparator (`scratchpad\winsuite\drift.py`) against `requirements.txt`, `requirements.restricted.txt`, `pyproject.toml` (base + `[dev]`) and the lock files.

| Package | Declared | Installed | Status |
|---|---|---|---|
| `send2trash` | `>=1.8.0` (base), lock `Send2Trash==2.1.0` | absent | **MISSING** -> fixes 7 tests |
| `imagecodecs` | `>=2024.1.1` (base), lock `==2026.3.6` | absent | **MISSING** -> fixes 4 tests |
| `pytest-timeout` | `>=2.3.0` (`[dev]`, `requirements.txt`), lock `==2.4.0` | absent | **MISSING** here AND in CI's pip line (`ci.yml:23`): the `@pytest.mark.timeout` in `test_batch_processor_e2e.py` is an unknown, inert mark on both. No test of the 47 needs it; `test_batch_processor_e2e.py` passes 9/9 either way |
| `jsonschema` | `>=4.18.0` (base), `pyproject.toml:87`, `requirements.txt:69` | absent | **MISSING** here AND in CI's pip line AND in all four lock files (so also in the Docker image, `Dockerfile:130` installs only from the lock, where `ScoringConfig.validate_schema` returns `[]` on ImportError). `tests/test_config_schema.py:10` does `pytest.importorskip("jsonschema")`, so 6 tests are silently skipped here and on CI. With it installed (4.26.0) they pass 6/6 |
| `pillow-heif` | `>=1.5.0` (base), lock `==1.5.0` | 1.3.0 | **BELOW FLOOR**. Not required for any of the 47, but it is a declared-floor violation: installed in Wave 0 by real pip, then `tests/test_image_loading.py` re-run (baseline 55 passed, 0 skipped). `pip install --dry-run pillow-heif==1.5.0` plans exactly `pillow_heif-1.5.0` and upgrades nothing else |
| `tifffile` | lock `==2026.3.3` | 2026.1.14 | Floor-satisfying but not at the lock version (`pytest` 9.0.2 vs lock 9.1.1 likewise). The DNG tests run with `imagecodecs` at the lock version and `tifffile` not; prototype passed. Recorded, not aligned (the lock is the Linux container's) |
| everything else with a floor | | | satisfied (`pillow` 12.0.0 >= 11.3, `tifffile` 2026.1.14, `transformers`, `torch` etc. all in range) |

Lock-vs-installed differences beyond that (`fastapi` 0.133 vs 0.141, `starlette`, `uvicorn`, `ruff` 0.15.1 vs 0.16.5, ~70 more) are floor-satisfying and irrelevant to these failures; the lock file is the Linux container's. Not touched.

CI parity note: this box runs Python 3.13.11, CI pins 3.12 (memory note: PEP 479 differences). Not a factor in the 47.

**exiftool:** `tests` reach `processing/xmp_export.py:416-424 _resolve_exiftool()`, which prefers the repo-root `D:\photo-llm\exiftool.exe` (untracked, 58 368 bytes) and only then `PATH`. `PATH` contains no installed exiftool (only `.`, which resolves to the same file). `D:\photo-llm\exiftool.exe -ver` = **13.45**, which meets the >= 13.28 floor (CLAUDE.md; also clear of the 13.23-13.27 defective window). The real-exiftool suites in `test_xmp_export.py` (`TestKeywordMergeIntegration` etc.) already pass **in the main checkout**. In a git worktree there is no `exiftool.exe` (it is untracked), so those suites skip with "exiftool not installed" (`tests/test_xmp_export.py:518-521`) unless `PATH` carries it: every gate in the worktree therefore runs with `PATH="/d/photo-llm:$PATH"` (step W0). The Docker pin is 13.55 (not relevant here).

## Files to create / change
All paths are relative to the worktree `D:\photo-llm-win` (step W0).
- `.github/workflows/ci.yml` (step C1: `python-tests` pip line + `windows-tests` job)
- `api/db_helpers.py` (product, step 1: `stored_path_form`, `get_visibility_clause`, one-line swap in `build_path_scope`)
- `tests/test_visibility_fail_open.py` (step 1: the new cases are added HERE, and its existing POSIX pin at `:54-62` is changed deliberately)
- `config_resolve.py` (product, step 2: extract `descriptor_matches_identity`, call it from `_rewrite_in_place`)
- `api/config_writes.py` (product, step 2)
- `tests/test_config_writes.py` (step 2)
- `tests/conftest.py` (step 2, optional: host `symlink_or_skip` so later symlink tests can adopt it; if skipped, it stays a local helper)
- `tests/test_auto_cull_cli.py` (step 3)
- `tests/test_config_merge.py` (step 4)
- `tests/test_config_path_env.py` (step 5)
- `tests/test_manifest_builder.py` (step 6)
- `tests/test_system_memory.py` (step 7)
- `tests/test_weight_optimizer_sources.py` (step 8)
- `tests/test_xmp_export.py` (step 9)
- `CHANGELOG.md` (step 11: three `### Fixed` entries under `## [Unreleased]`)
- `docs/CONFIGURATION.md`, `docs/VIEWER.md` and their 6 translations (step 11: one sentence on how `directories` entries are matched; anchors listed there)
- NOT changed: any lock file, `requirements*.txt`, `Dockerfile` (see "Not changing").
- No new test file: the visibility cases extend `tests/test_visibility_fail_open.py` so the POSIX control is not written twice.

## Not changing (decisions, with where the follow-up goes)
- **`jsonschema` in the four lock files and the Docker image.** It is a declared base dependency (`pyproject.toml:87`) that is in none of `requirements.lock.txt`, `requirements.legacy.lock.txt` and the restricted locks, so every container runs `ScoringConfig.validate_schema` as a no-op. Fixing it needs `jsonschema` + `referencing` + `rpds-py` + `jsonschema-specifications` pinned identically in both full lock files (CI's lock-parity check, `ci.yml:183-212`) and a Docker rebuild. That belongs to the later dependency / lock-file (Dependabot) PR; this PR only adds `jsonschema` and `pytest-timeout` to CI's `pip install` lines (step C1) and records the gap here.
- **`tests/conftest.py` does not pin `FACET_CONFIG`** (step 10).
- **Windows ACL analogue of the "world-readable config" warning.** `api/config.py:528` (`_group_or_other_readable_mode`) deliberately returns 0 off POSIX, so `test_a_world_readable_config_is_reported` and `test_a_symlinked_config_is_reported_through_the_link` skip. This is a product gap (Windows has no equivalent warning), not a test defect; recorded as a decision, not fixed here.
- **Root-drive directory file access (pre-existing, reviewer note 5).** Once step 1 lands, a Windows user scoped to `D:\` passes visibility but `api/path_validation.py:47-52` builds `realpath('D:\\') + os.sep`, which is `D:\\\\`, so file access 404s; `/` on POSIX has the same shape of bug. Out of scope; follow-up.
- **Misleading log (reviewer note 6).** `config/percentile_normalizer.py:2096-2098` logs "No existing config to back up" for any `None` return, including a symlink refusal. Out of scope; follow-up.
- **`tifffile`/`pytest` not at the lock versions** (drift table). Not aligned.

## Open questions / risks
- The backup-writer symlink fix (step 2) cannot be exercised on a real symlink on this box (WinError 1314). It is proven here by the identity-mismatch tests (a swapped-in regular file refused, a non-regular destination refused) and on Linux CI by a real symlink with `O_NOFOLLOW` stripped. GitHub's Windows runners run elevated, so `os.symlink` should work in the `windows-tests` job and exercise it for real (UNVERIFIED until that job runs).
- Installing `jsonschema` turns schema validation ON for every `ScoringConfig` built in the suite (`config/scoring_config.py:293`), not only `test_config_schema.py`. The specific risk: a `caplog` assertion of "no warnings" or "no records" now sees new output. The full-suite run in the worktree is the check; any such failure is fixed in the test, not by uninstalling.
- Wave 0 changes the shared `conda/` env while other jobs may run. The dry run is additive only (7 new packages, nothing upgraded; `numpy` 2.2.6 untouched), and `pillow-heif` upgrades one package. `imagecodecs` also changes live scan behaviour on this box: the DNG JPEG XL preview fallback becomes active (desirable).
- The range predicate on POSIX changes the SQL shape of the visibility clause from 1 to 2 bound params per directory. Callers splice `vis_params` generically (100 call sites, none inspects it), but only the LIKE form was run across the multi-user slice in the prototype; the range form is measured for speed and plan (step 1) and must be re-run through the slice (gate step 5).
- The Bash tool on this box unescapes one level of backslashes in heredocs, so any script containing Windows/regex backslashes must be written with the Write tool, not `cat <<EOF`.

## Steps

### Wave 0 — worktree, baseline, environment (orchestrator, before any build)
W0. **Create the worktree and capture the baseline.**
   - description: wait until `fix/open-gaps-120` has merged to `master` (it carries 4 commits not on master and HEAD was still moving; the tracked tree it leaves is what this PR must sit on). Then, from `D:\photo-llm`: `git fetch` and `git worktree add D:\photo-llm-win -b fix/windows-test-suite master`. Every builder edits in `D:\photo-llm-win` and every gate runs there; nobody commits in `D:\photo-llm`, so none of the open-gaps commits can enter this PR.
   - key decision: worktree off master vs a branch in place. Default: worktree. This PR must be separate, and the working branch is not master.
   - gate environment, used by EVERY pytest run in this spec (a shell function, written `pyt` below). C: is nearly full, so all test temp output goes on D:, and the worktree needs the main checkout's `exiftool.exe` on `PATH`:
     `pyt() { TMP='D:\facet-scratch\tmp' TEMP='D:\facet-scratch\tmp' TMPDIR='D:\facet-scratch\tmp' PATH="/d/photo-llm:$PATH" D:/photo-llm/conda/python.exe -m pytest --basetemp='D:\facet-scratch\tmp\pt-win' "$@"; }` (run from `D:\photo-llm-win`; `mkdir -p /d/facet-scratch/tmp` first; each `pyt` line below passes `-p no:cacheprovider` as written). From PowerShell, set `$env:TMP`, `$env:TEMP`, `$env:TMPDIR` to `D:\facet-scratch\tmp` and pass the same `--basetemp`. Probes the builders write themselves use the same three variables. The repo-root `exiftool.exe` is untracked, so a worktree has none; `processing/xmp_export.py:416-424` falls back to `PATH`, which finds the main checkout's. Without the `PATH` entry `TestKeywordMergeIntegration` and the other real-exiftool suites silently skip (`tests/test_xmp_export.py:518-521`).
   - baseline (on the fresh worktree, before any edit AND before step 0, so the `jsonschema` skips are still in it): `pyt tests/ -q -p no:cacheprovider -rs > <scratch>\base_rs.txt 2>&1`, exit code read from the command itself, and keep the `SKIPPED [n] file:line: reason` set plus the total skip count. This is what the final gate diffs against (G12).
   - verify: `git -C D:/photo-llm-win status --porcelain` empty and `git -C D:/photo-llm-win branch --show-current` is `fix/windows-test-suite`; `pyt tests/test_xmp_export.py -q -p no:cacheprovider -rs | grep -c "exiftool not installed"` is 0 on the baseline.
   - affected files: none.

0. **Install the declared-but-missing packages into the local conda env** (environment only; the main checkout's `conda/` is shared by the worktree because the interpreter path is absolute).
   - key decision: install the lock version where the lock has one; `jsonschema` has no lock entry, so pin the version that was verified, and pin its whole stack so "verified" stays true (G10/N11). Real pip, not an unzipped wheel.
   - run with `TMP`, `TEMP` and `TMPDIR` set to `D:/facet-scratch/tmp` (C: is nearly full; pip unpacks wheels into the temp dir). Default: one command (all wheels exist for cp313-win_amd64; `pip install --dry-run` reported exactly `Send2Trash-2.1.0 imagecodecs-2026.3.6 pytest-timeout-2.4.0 jsonschema-4.26.0 jsonschema-specifications-2025.9.1 referencing-0.37.0 rpds-py-2026.9.1`, nothing upgraded):
     `D:/photo-llm/conda/python.exe -m pip install send2trash==2.1.0 imagecodecs==2026.3.6 pytest-timeout==2.4.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==2026.9.1`
   - then, as a **separate** command so a failure cannot take the first down: `D:/photo-llm/conda/python.exe -m pip install "pillow-heif>=1.5.0"` (lock pin 1.5.0; dry run plans only `pillow_heif-1.5.0`, pillow 12.0.0 already satisfies `>=11.1.0`). If it fails, the step **reports the failure and continues**; it is an environment fix, not a repo change, and no test of the 47 depends on it. On success run `tests/test_image_loading.py` (baseline 55 passed, 0 skipped) and report the result; a regression is reported, not hidden.
   - verify: `pyt tests/test_cull.py tests/test_raw_decode.py tests/test_config_schema.py tests/test_batch_processor_e2e.py -q -p no:cacheprovider` -> 0 failed (prototype with the same wheels on `PYTHONPATH`: 160 passed). `pip list` shows all four jsonschema-stack packages at the pinned versions.
   - CI is NOT already equipped: `ci.yml:23` installs an unpinned list that has neither `jsonschema` nor `pytest-timeout` (the earlier claim "Linux CI already has all four (lock files)" was false: CI does not read the lock files). That is fixed in step C1.
   - affected files: none in the tree.

C1. **CI: install what the tests import, and make the Windows job count.** (`.github/workflows/ci.yml`)
   - description: (i) append `jsonschema pytest-timeout` to the `pip install` line of `python-tests` (`ci.yml:23`) at the pinned versions the lock/verification uses (`jsonschema==4.26.0 pytest-timeout==2.4.0`; the rest of that line stays unpinned as today, so only these two are pinned and a comment says why: the lock does not carry `jsonschema`). (ii) In `windows-tests` (`ci.yml:58-76`): make its `pip install` equal the Linux list plus the two packages (it currently lacks `tifffile imagecodecs lupa`, which the files below import); add the files this PR makes green to the pytest line, **keeping an explicit file list** (default; NOT the full suite): `tests/test_scan.py tests/test_device.py tests/test_auto_cull_cli.py tests/test_config_merge.py tests/test_config_path_env.py tests/test_config_writes.py tests/test_manifest_builder.py tests/test_system_memory.py tests/test_weight_optimizer_sources.py tests/test_xmp_export.py tests/test_visibility_fail_open.py tests/test_cull.py tests/test_raw_decode.py tests/test_config_schema.py`. (iii) Rewrite the comment at `:69-75` so it no longer says a full run is "permanently red and therefore ignored": say instead that the listed files are the ones kept green on Windows, that the rest of the suite is not yet run there, and that a file is added to the list in the PR that makes it green.
   - key decision: switch the job to the full suite vs extend the list. Default: extend the list. The full suite on Windows is unproven on a runner (it is green only on this box, and with exiftool on `PATH`), and a red required job nobody can fix quickly is the failure mode the old comment described.
   - what this buys: the 47 cannot silently come back; GitHub's elevated Windows runner should create real symlinks, which would run the backup-writer symlink tests (step 2) for real (UNVERIFIED until the job runs).
   - verify: `gh pr checks` after the PR is opened shows `python-tests`, `windows-tests`, `lint`, `commit-hygiene` all green; the `windows-tests` log lists every file above; `python-tests` shows `test_config_schema.py` passing 6 (not skipped) and no "unknown mark: timeout" warning.
   - affected files: `.github/workflows/ci.yml`

### Wave 1 — three builders in parallel (file sets are disjoint)
- Builder A: steps 1, 3 (both touch the multi-user path; step 3's two multi-user tests only turn green with step 1). Files: `api/db_helpers.py`, `tests/test_visibility_fail_open.py`, `tests/test_auto_cull_cli.py`.
- Builder B: steps 2, 4, 5, 8. Files: `config_resolve.py`, `api/config_writes.py`, `tests/test_config_writes.py`, `tests/conftest.py` (only if the optional move is taken), `tests/test_config_merge.py`, `tests/test_config_path_env.py`, `tests/test_weight_optimizer_sources.py`.
- Builder C: steps 6, 7, 9, C1. Files: `tests/test_manifest_builder.py`, `tests/test_system_memory.py`, `tests/test_xmp_export.py`, `.github/workflows/ci.yml`.
- Orchestrator after the builders: step 11 (CHANGELOG + docs).

1. **Fix multi-user directory visibility: Windows-stored paths, and the POSIX case leak** (class c, two faces of one function).
   - description: rewrite the per-directory body of `get_visibility_clause` (`api/db_helpers.py:1215-1228`). Today: `prefix = d.rstrip('/\\') + '/'` and a bare `LIKE`, so (Windows) `C:\x/` never prefixes the stored `C:\x\low.jpg`, and (POSIX) LIKE's ASCII case folding lets `/mnt/photos` see `/mnt/Photos/a.jpg` on a case-sensitive filesystem. The new body does three things per directory:
     1. **Normalise it to the form the scanner stores.** There is no single normaliser today: the scanner writes `str(Path(p).resolve())` inline at `facet.py:168, 1620, 1658/1666, 2670, 4070`, and `build_path_scope` (`api/db_helpers.py:1143`) re-derives it with `os.path.realpath` (identical to `Path.resolve` on Windows in 3.12 and 3.13: probe `probe_realpath.py`). Define the minimal shared one next to them, `stored_path_form(directory)` in `api/db_helpers.py`, `functools.lru_cache`d (an offline UNC costs about 130 ms on its first call), and replace `os.path.realpath(...)` in `build_path_scope` by it so the two cannot drift.
        ```python
        def _is_windows_shaped(directory):
            return ntpath.splitdrive(directory)[0] != ''  # `D:`, `D:\x`, `\\host\share`, `\\?\D:\x`

        @functools.lru_cache(maxsize=256)
        def stored_path_form(directory):
            """What the scanner would store for this directory: str(Path(p).resolve())."""
            if not _is_windows_shaped(directory):
                return directory
            text = directory
            if text.startswith('\\\\?\\UNC\\'):
                text = '\\\\' + text[8:]
            elif text.startswith('\\\\?\\'):
                text = text[4:]                       # resolve() does not strip the \\?\ prefix
            drive, rest = ntpath.splitdrive(text)
            if not rest:
                text = drive + '\\'                   # bare `D:` resolves to the drive's CWD, not its root
            if sys.platform == 'win32':
                try:
                    return str(Path(text).resolve())  # the scanner's own call: separators, case, mapped drive -> UNC
                except OSError:
                    pass
            return ntpath.normpath(text)              # a POSIX host holding a DB scanned on Windows: lexical only
        ```
        A POSIX-shaped directory is returned untouched (today's behaviour; resolving it would add a filesystem round trip and change behaviour for symlinked configured paths, which is out of scope). `import ntpath, functools, sys` and `from pathlib import Path` are added where missing (the earlier draft used `re` without importing it).
     2. **Emit ONE predicate in the stored separator form.** A Windows-shaped directory gets prefix `form.rstrip('/\\') + '\\'`; a POSIX-shaped one `form.rstrip('/\\') + '/'` (`D:\` rstrips to `D:` and becomes `D:\`). One form only, not two: measured on the 127k-class library copy (backup-API copy of `lib_copy.db`, deleted afterwards) the forward-slash form has no writer and no rows: **0 drive-letter paths, 0 drive paths containing `/`, 129,737 UNC backslash paths, 0 UNC paths containing `/`, 0 mixed-separator rows.** The ruling's rule (both counts 0 -> one LIKE) applies.
     3. **Match case the way the filesystem that wrote the path does.** `_CASE_INSENSITIVE_HOST = sys.platform in ('win32', 'darwin')` (APFS defaults to case-insensitive). A directory takes the existing `LIKE ... ESCAPE '\'` (ASCII case-insensitive; wildcard escaping unchanged) when the host is case-insensitive OR the directory is Windows-shaped (its rows were written by NTFS even if the viewer runs on Linux). Every other directory (a POSIX-shaped path on a case-sensitive host) takes a **case-sensitive range predicate**, `({alias}.path >= ? AND {alias}.path < ?)` with params `[prefix, prefix[:-1] + chr(ord(prefix[-1]) + 1)]` (for `/mnt/photos/` that is `['/mnt/photos/', '/mnt/photos0']`).
   - measurements (backup-API copy of `lib_copy.db`, `PRAGMA query_only`, `COUNT(*)`, two disjoint directories as a user with own + shared dirs has; `probe_db2.py`):

     | Variant | Rows | Median |
     |---|---|---|
     | today: 1 LIKE, forward-slash form | 0 (matches nothing, so it is "fast" only because it is wrong) | 19.3 ms |
     | **new Windows form: 1 native LIKE per directory** | the real count | **37.5 ms** |
     | rejected: two forms per directory (4 LIKE) | same | 46.8 ms (+25%) |
     | rejected: `REPLACE(path,'\','/') LIKE` (what `gallery.py:559` does) | same | 115 ms |
     | **new POSIX form: case-sensitive range x2** | | **0.6 ms**, a SEARCH on `sqlite_autoindex_photos_1` (the PK, BINARY collation, which LIKE can never use) |

     The reviewer's synthetic 127,463-row Windows table agrees: today 34.7 ms, two LIKEs 50.6 ms, native single LIKE 34.3 ms, `REPLACE` 116.3 ms. So the Windows fix costs one LIKE per directory (the same shape as today's), and POSIX gets about 50x faster as a side effect. **Re-measure after the edit** on a fresh backup-API copy (`D:\facet-scratch\<new dir>\`, never `lib_copy.db` itself, deleted afterwards) with the same probe and record the new numbers in the PR body; acceptance: Windows within 10% of 37.5 ms, POSIX range under 2 ms and still a SEARCH.
   - key decision: one separator form vs both vs `REPLACE`. Default: one stored form (measured above). `REPLACE` would also run per row and, on Linux, fold a legitimate backslash in a POSIX filename into a separator (`/a\b/x` visible to a user scoped to `/a/b`), loosening isolation.
   - tests (failing-first): written and run BEFORE the product edit, in `tests/test_visibility_fail_open.py` (extend it; do not create a second file, so the POSIX control exists once). The fragment is executed against `init_database(tmp_path / "v.db")` (the real schema; never `CREATE TABLE photos`; `INSERT INTO photos (path) VALUES (?)` is valid because every other column has a default), with `mock.patch("api.db_helpers.is_multi_user_enabled", return_value=True)` and `mock.patch("api.db_helpers.get_user_directories", return_value=[...])`.
     - Windows shapes, host-independent, one param each: `C:\photos\2024`, `C:/photos/2024`, `C:\photos\2024\`, `\\nas\share\2024`, `\\?\C:\photos\2024`, `\\?\UNC\nas\share\2024`, bare `D:` (rows `D:\a.jpg`). Stored rows in the backslash form are visible; `C:\photos\20245\a.jpg`, `C:\photos\2024_x\a.jpg` are not; a directory containing `_` and `%` matches only itself; `c:\PHOTOS\2024` sees `C:\photos\2024\a.jpg` (NTFS semantics) on every host.
     - POSIX case, **parametrised on `monkeypatch.setattr(db_helpers, "_CASE_INSENSITIVE_HOST", flag)`**: flag False -> `/mnt/photos` yields exactly `(photos.path >= ? AND photos.path < ?)` with `['/mnt/photos/', '/mnt/photos0']`, sees `/mnt/photos/a.jpg`, does NOT see `/mnt/Photos/a.jpg` or `/mnt/photos2/a.jpg`; flag True -> the old `LIKE` shape and `/mnt/Photos/a.jpg` IS visible (macOS/APFS and Windows keep case-insensitive matching). A backslash inside a POSIX name (`/mnt/we\ird`) yields one pair and does not match `/mnt/we/ird/a.jpg`.
     - `EXPLAIN QUERY PLAN SELECT 1 FROM photos WHERE <fragment>` for the flag-False fragment (1 and 2 directories) contains a `SEARCH` and no `SCAN` row.
     - `stored_path_form`: non-Windows host, lexical results (`C:/photos/2024/` -> `C:\photos\2024`, the `\\?\` forms, bare `D:` -> `D:\`); win32 host, a differently-cased spelling of an existing `tmp_path` directory resolves to its canonical on-disk case (skip with a reason on other hosts).
     - The existing `test_authenticated_multi_user_scoped_to_directories` (`:54-62`) pins `sql == "(photos.path LIKE ? ESCAPE '\\')"` and `params == ["/photos/alice/%"]`. That pin **changes deliberately**: it becomes the `_CASE_INSENSITIVE_HOST = True` case of the parametrised test above (same two assertions, kept verbatim), and the new default case pins the range. State both in the PR body as an intentional assertion move, not a drive-by edit.
   - verify: before the edit the new Windows-shape and case tests fail on Linux and on Windows (fail-before; the old code yields `C:\photos\2024/` and a case-folding LIKE); after: `pyt tests/test_visibility_fail_open.py tests/test_auto_cull_cli.py -q -p no:cacheprovider` passes. The 2 multi-user `TestRejectStandaloneBelow` tests turn green. Regression slice: `git grep -l -e is_multi_user_enabled -e get_user_directories -- tests/` (the explicit file list, reproducible; prototype baseline with the LIKE form: 40 files / 1123 passed, 12 skipped) must be identical before and after, now **including the range form**, which was not prototyped. Then the perf re-measure above.
   - known limitations (documented, not fixed): (i) **non-ASCII case**: SQLite's `LIKE` folds ASCII only, so a configured `C:\Élan` does not match stored `C:\élan` although NTFS treats them as one directory; (ii) a **mixed-separator stored path** (`C:\photos/2024\a.jpg`) matches nothing (count on the copy: 0); (iii) a **mapped-drive configured entry** (`Z:\...`) is resolved to its UNC target only on a Windows host that has the drive mapped; on any other host it cannot be (the user's NAS is `\\Sweet`, stored as UNC); (iv) a POSIX host treats a leading `//` as Windows-shaped via `ntpath.splitdrive`; no real config has one; (v) a symlinked POSIX configured directory is not resolved (unchanged). Items (i)-(iii) are one sentence in the docs (step 11).
   - Docs/changelog: step 11 (behaviour changes: a POSIX user whose configured directory differs in case from the stored paths loses those photos, and that is correct).
   - affected files: `api/db_helpers.py`, `tests/test_visibility_fail_open.py`

2. **Backup writer: refuse a symlinked backup name on Windows, close the check-then-open race, and make the symlink tests portable** (latent class c + 4 class d tests).
   - description (product): `write_owner_only_backup` (`api/config_writes.py:95-125`) currently opens with `O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW`. On Windows `O_NOFOLLOW` is absent, and a simple `os.path.islink` guard before the open would leave a window in which a link raced in is still truncated and written through (`O_TRUNC` empties the victim before anything can be checked). Instead mirror the race-free pattern `config_resolve._rewrite_in_place` already uses (`config_resolve.py:483-540`): pre-open `lstat` identity, open WITHOUT `O_TRUNC`, post-open `fstat` identity comparison, and only then `truncate`.
     ```python
     # config_resolve.py: extracted from the inline check in _rewrite_in_place, which now calls it
     def descriptor_matches_identity(fd, identity):
         opened = os.fstat(fd)
         return stat.S_ISREG(opened.st_mode) and (opened.st_ino, opened.st_dev) == (identity.st_ino, identity.st_dev)

     # api/config_writes.py
     _BACKUP_OPEN_FLAGS = os.O_WRONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)   # no O_TRUNC

     identity = config_resolve._destination_identity(backup_path)   # lstat if a regular file, else None (a link is None)
     if identity is None and os.path.lexists(backup_path):
         _log_refused_symlink(backup_path); return None              # link, directory, FIFO ...
     flags = _BACKUP_OPEN_FLAGS | (0 if identity else os.O_CREAT | os.O_EXCL)
     try:
         fd = os.open(backup_path, flags, BACKUP_FILE_MODE)
     except OSError as ex:
         if ex.errno not in (errno.ELOOP, errno.EEXIST, errno.ENOENT):
             raise
         _log_refused_symlink(backup_path); return None              # raced in between the check and the open
     if identity is not None and not config_resolve.descriptor_matches_identity(fd, identity):
         os.close(fd); _log_refused_symlink(backup_path); return None  # swapped for another file
     with os.fdopen(fd, 'wb') as destination:                        # 'wb' + O_BINARY: no LF -> CRLF on Windows
         destination.truncate(0)                                      # truncate FIRST, then re-mode while empty, then write
         if hasattr(os, 'fchmod'):
             os.fchmod(destination.fileno(), BACKUP_FILE_MODE)
         shutil.copyfileobj(source, destination)
     ```
     An absent backup is created with `O_CREAT|O_EXCL` at 0600, so a link raced in at that instant fails with `EEXIST`/`ELOOP` rather than being followed. The refusal message is factored into one module-level helper `_log_refused_symlink` used by every refusal branch so they cannot drift. `descriptor_matches_identity` is the one shared helper (DRY): both writers now compare descriptor to `lstat` through it. `config_resolve._destination_identity` stays private and keeps its name (two test modules monkeypatch it by name; renaming would force an alias, which the project forbids); `config_writes` calls it as `config_resolve._destination_identity(...)` (attribute lookup at call time, so those fixtures keep working: their wrapper keeps the real `st_ino`/`st_dev`). Update the stale docstring/comment: "O_NOFOLLOW is a no-op on Windows" now says the lstat/fstat identity check is what protects there, and the re-mode-while-empty sentence stays true (truncate precedes `fchmod`). Probe `trial\test_backup_v2_probe.py` (this exact logic): 4 passed on this box (absent-then-shorter overwrite leaves no stale tail and no CRLF; `lstat` and `fstat` identity agree on NTFS; a swapped-in file is refused; a non-regular destination is refused).
   - description (tests, `tests/test_config_writes.py`): a helper `symlink_or_skip(link, target)` replaces the three `backup.symlink_to(victim)` calls in `TestTheBackupWriterRefusesASymlink`: it calls `os.symlink` and, on `OSError`, re-raises unless `sys.platform == "win32"`, in which case it `pytest.skip`s with `f"cannot create a symlink here: {ex}"` (never a silent skip on Linux). Optionally host it in `tests/conftest.py` so `tests/test_api_config.py:34`'s `_WIN32_SYMLINK_REASON` and `tests/test_webdav.py:323,333` can adopt it later; default: put it in conftest, adopt nowhere else in this PR. `test_it_does_not_re_mode_the_target` gets `@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits do not apply on Windows")` (its assertion is `S_IMODE(victim) == 0o644`). `test_an_ordinary_backup_is_unaffected` is **split**: return value and content stay; the `S_IMODE == 0o600` assertion moves unchanged into `test_an_ordinary_backup_is_owner_only` with the same win32 skipif. New tests that run on Windows too and need no symlink privilege: (a) `test_a_destination_swapped_for_another_file_is_refused` (monkeypatch `config_resolve._destination_identity` to return the victim's `lstat`; assert `None`, backup and victim bytes unchanged, "Refusing" logged); (b) `test_a_non_regular_destination_is_refused` (a directory at the backup name); (c) `test_an_absent_backup_that_appears_before_the_create_is_refused` (patch `os.path.lexists` to False while the file exists, so `O_EXCL` raises `EEXIST`; assert `None` and the file untouched); (d) `test_a_shorter_backup_leaves_no_stale_tail_and_no_crlf` (the probe's first test); (e) `test_it_refuses_a_link_even_where_the_open_cannot`: create the link via `symlink_or_skip`, `monkeypatch.setattr(config_writes, "_BACKUP_OPEN_FLAGS", os.O_WRONLY | getattr(os, "O_BINARY", 0))` (the flag set Windows effectively has), assert `None`, victim untouched, an error mentioning "symlink" logged. Without the lstat guard (e) fails on Linux, so it proves the guard rather than `O_NOFOLLOW`.
   - key decision: guard vs skip the symlink tests on win32 and leave the product alone; and `islink` vs identity. Default: guard, race-free identity check (the reviewer's G7). Skipping would hide that `api/config_writes.py` writes the complete config (every `password_hash`, `viewer.password`, `frame.tokens`, `immich.api_key`) through a planted link on Windows.
   - verify: failing-first for (a), (b), (c), (e) on the unmodified writer (it follows the link, truncates the victim, and has no identity check); then `pyt tests/test_config_writes.py tests/test_config_merge.py -q -p no:cacheprovider -rs` (the second file exercises `_rewrite_in_place` through the extracted helper). Prototype of the test-side edit: 44 passed, 25 skipped (three of them "cannot create a symlink here: WinError 1314"); previous: 4 failed, 43 passed, 21 skipped. UNVERIFIED on a real Windows symlink (no privilege here; the `windows-tests` CI job is expected to run it for real). Callers all accept `None` (`api/auth.py:513`, `database.py:120`, `config/percentile_normalizer.py:2096`, `api/config_writes.py:181`). Linux CI: one extra `lstat`, and an `O_EXCL` create for the absent case; the symlink case returns the same `None` and the same log as the old `ELOOP` branch.
   - affected files: `api/config_writes.py`, `config_resolve.py`, `tests/test_config_writes.py`, `tests/conftest.py` (optional helper)

3. **`TestPathScope` fixture: tolerate a case-insensitive filesystem, and assert the Windows semantics instead of discarding them.**
   - description: in the `dirs` fixture (`tests/test_auto_cull_cli.py:125-127`) change `(tmp_path / name).mkdir()` to `(tmp_path / name).mkdir(exist_ok=True)` with a comment that on NTFS `Shoot` is `shoot`. Add `test_a_case_variant_scope_collapses_where_the_filesystem_folds_case`: skip with a reason ("filesystem is case-sensitive") unless `(tmp_path / "shoot").mkdir(); (tmp_path / "Shoot").exists()`; then assert `build_path_scope([dirs / "Shoot"]).prefixes == build_path_scope([dirs / "shoot"]).prefixes`.
   - key decision: tolerate the collision vs skip the tests that name `Shoot`. Default: tolerate, keep every assertion, and add the one test that states what the "product is fine" verdict depends on. Without it, with `exist_ok=True` the `'Shoot'` iteration of `test_sql_agrees_with_matches` (`:140-151`) silently repeats `'shoot'` and nothing asserts that `realpath` canonicalises case. On Linux the new test skips and the original distinct-directory check runs.
   - verify: `pyt tests/test_auto_cull_cli.py -q -p no:cacheprovider -rs` (needs step 1 for the two multi-user tests). Prototype: `TestPathScope` 7/7 pass with `exist_ok=True`. Fault injection that proves the new test can fail (per the project rule, restore by retyping, then `git status --porcelain api/db_helpers.py` shows only the intended step-1 diff): swap `stored_path_form`/`realpath` in `build_path_scope` for `os.path.abspath`, the new test fails, restore.
   - affected files: `tests/test_auto_cull_cli.py`

4. **`test_config_merge.py`: separator literal, chmod (split), chown (skip).**
   - description: (i) `test_they_live_inside_the_config_package`: replace `defaults_path().endswith("config/scoring_config.default.json")` with `Path(defaults_path()).parts[-2:] == ("config", "scoring_config.default.json")` (`Path` is already imported). This corrects a wrong expectation (it asserted a `/` that `os.path.join` never promised) and still pins both the directory and the file name. (ii) `test_it_leaves_an_owner_only_backup` (`:381-391`) is **split, not skipped**: `returncode == 0` and "exactly one backup is created" stay in the original test and run on Windows; the `S_IMODE == 0o600` assertion (`:391`) moves unchanged into a new `test_it_leaves_an_owner_only_backup_mode` marked `@pytest.mark.skipif(sys.platform == 'win32', reason="POSIX permission bits do not apply on Windows")` (the file already uses this form at `:435`). (iii) `test_a_denied_chown_still_persists_only_the_override`: `@pytest.mark.skipif(sys.platform == "win32", reason="POSIX ownership has no Windows analogue")`, the reason already used at `tests/test_weight_optimizer_sources.py:424`.
   - key decision for (iii): skip vs `raising=False`. Default: **skip**. `config_resolve.py:462-467` documents that on Windows "the replace route is the only correct one" and `_adopt_owner` returns True whenever `os.chown` is absent, so the in-place route cannot run in production there; `raising=False` would plant an `os.chown` so the route runs anyway and the test would pin behaviour of code the module says must not run on Windows. Ownership adoption really is POSIX-only, so the skip is honest.
   - verify: `pyt tests/test_config_merge.py -q -p no:cacheprovider -rs`. Prototype: (i) passes, (ii)'s portable half passes and the sibling skips, (iii) skips (the only other failures in the scratch copy were the CLI-subprocess tests that locate `database.py` relative to the copy's `__file__`, an artefact of testing a copy). Linux CI: (i) same assertion on POSIX; (ii) both tests run; (iii) not skipped.
   - affected files: `tests/test_config_merge.py`

5. **`test_config_path_env.py`: port the unreadable-config test, skip the three POSIX-mode tests.**
   - description: (a) **port** `test_a_config_it_cannot_open_boots_locked_rather_than_crashing` (`:662-669`; its `_start` helper at `:642-660` applies `config.chmod(0o000)`). Windows supports unreadable files; only `chmod(0o000)` is POSIX-specific. On win32 deny read to Everyone through the ACL instead and always undo it so pytest can clean `tmp_path`:
        ```python
        denied = mode == 0 and sys.platform == "win32"
        if denied:
            subprocess.run(["icacls", str(config), "/deny", "*S-1-1-0:(RD)"], check=True, capture_output=True)
        try:
            return subprocess.run(...)          # the existing boot subprocess, unchanged
        finally:
            if denied:
                subprocess.run(["icacls", str(config), "/remove:d", "*S-1-1-0"], check=True, capture_output=True)
                config.chmod(0o600)
        ```
        `*S-1-1-0` is the Everyone SID, so it is locale-independent. POSIX keeps `chmod(0o000)` untouched. Verified by the reviewer on this box without admin rights: after the deny `open()` raises `PermissionError 13`, and replaying the test's subprocess gives rc 0 and `{"load_failed": true, "open_viewer": false, "any_feature_on": false}` (`unreadable_boot.py`). The restore sits in `finally` (or fixture teardown), so a failed assertion cannot leave an undeletable file. (b) `@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits do not apply on Windows")` (the wording already used at `tests/test_config_writes.py:58`, `tests/test_auth.py:789`) on `test_a_world_readable_config_is_reported`, `test_the_mode_is_left_exactly_as_the_operator_set_it` and `test_a_symlinked_config_is_reported_through_the_link` (needs both modes and the symlink privilege; the warning is deliberately silent on Windows). Scope: the individual test, not the class; `test_an_owner_only_config_says_nothing` and `test_a_malformed_config_boots_locked_rather_than_crashing` are valid on Windows and stay.
   - key decision: skip vs port. (a) port (the locked decision: a POSIX assumption about something Windows supports is fixed, not skipped). (b) skip: there is no Windows analogue of "world-readable" and the product degrades on purpose (`api/config.py:172-182`, `:528`): no warning, no crash, no re-mode. Optional, not taken: `test_the_mode_is_left_exactly_as_the_operator_set_it` could instead record `S_IMODE` after `chmod` and assert it unchanged after boot (checks on Windows that boot never toggles the read-only bit); skipped to keep the diff to what the ruling asked.
   - verify: `pyt tests/test_config_path_env.py -q -p no:cacheprovider -rs` -> **42 passed, 3 skipped** (was 41/4 before the port). A leaked deny would show as a `PermissionError` when pytest removes `tmp_path`, so a clean teardown is part of the check. Linux CI: the `win32` branch is inert there.
   - affected files: `tests/test_config_path_env.py` (already imports `sys` and `pytest`; add `subprocess` if missing)

6. **`test_manifest_builder.py`: mixed-separator seed paths.**
   - description: in `test_root_scope_excludes_applied_and_other_root_groups` (`:170-203`) replace the five `f"{lib_a}/NAME.jpg"` literals (`g1`, `g2`, `g3`, `applied1`, `applied2`) with `os.path.join(lib_a, "NAME.jpg")`; add `import os` if missing. This corrects a wrong expectation (a `/`-joined path under a root that is queried with the native separator) without changing any asserted count (1 and 2).
   - key decision: fix the test vs make `build_root_filter` match both separators. Default: test. Production rows are always `Path.resolve()`d native paths and `build_root_filter` is shared with sidecar import/export that rely on native-separator prefixes; widening it would be a behaviour change with no failing real-world case.
   - verify: `pyt tests/test_manifest_builder.py -q -p no:cacheprovider`. Prototype: the target passes (only three CLI-subprocess tests fail in the scratch copy, same `__file__` artefact as step 4). Linux CI: `os.path.join` equals `f"{x}/..."` there.
   - affected files: `tests/test_manifest_builder.py`

7. **`test_system_memory.py`: sysconf is POSIX-only.**
   - description: define `_NEEDS_SYSCONF = pytest.mark.skipif(not hasattr(os, 'sysconf'), reason='os.sysconf is POSIX-only; psutil is the reader on Windows')` above `TestSysconfTotalBytesItself` and apply it to the two tests that call the real `os.sysconf` (`test_it_reports_this_machines_real_physical_ram`, `test_it_agrees_with_psutil_to_within_a_page`). For the seven tests that replace `os.sysconf` with a fake (`test_a_platform_without_the_names_reports_nothing` x3, `test_a_nonsense_reading_is_no_reading` x4) pass `raising=False` to `monkeypatch.setattr` so they run for real on Windows and exercise `_sysconf_total_bytes`'s own parsing.
   - key decision: skip the whole class vs split. Default: split — keyed on `hasattr(os, 'sysconf')`, which states the real precondition (also true for any other platform without it), rather than `sys.platform`.
   - verify: `pyt tests/test_system_memory.py -q -p no:cacheprovider -rs` -> prototype: 65 passed, 3 skipped (the 2 new skips plus the pre-existing one). Linux CI: `hasattr(os,'sysconf')` is True so nothing is skipped; `raising=False` on an existing attribute equals the default.
   - affected files: `tests/test_system_memory.py`

8. **`test_weight_optimizer_sources.py`: POSIX mode assertions.**
   - description: `test_apply_preserves_owner_only_mode` (`:383-389`) is **split**: the assertion that `viewer.password` survives `apply_optimized_weights` (`:388`) is portable and stays in the original test, which runs on Windows; the `S_IMODE` assertion (`:389`) moves unchanged into a new sibling `test_apply_preserves_owner_only_mode_bits` carrying `@pytest.mark.skipif(sys.platform == 'win32', reason="POSIX permission bits do not apply on Windows")` (`sys` is already imported; the same file already skips `:424` this way). `test_apply_preserves_a_group_readable_mode` is mode-only, so it is skipped whole with the same marker. `test_scratch_file_name_is_gitignored` passes and stays.
   - key decision: skip vs conditional assert vs split. Default: split where a portable assertion exists (a whole-test skip would drop an assertion that passes on Windows, contradicting "no assertion is weakened"), skip where the test is mode-only.
   - verify: `pyt tests/test_weight_optimizer_sources.py -q -p no:cacheprovider -rs` -> the previous prototype counted 22 passed, 3 skipped with both tests skipped whole; with the split expect one more passed and one more skipped than that on win32 (record the exact figures in the PR). Linux CI: inert, both halves run.
   - affected files: `tests/test_weight_optimizer_sources.py`

9. **`test_xmp_export.py`: a fake exiftool that Windows can execute.**
   - description: rewrite `_fake_exiftool` (`:419-432`) so the behaviour lives in a Python script run by `sys.executable` and only the launcher is platform-specific; add `import sys`. The version answer, the `[]` for `-json` reads, and the corrupt write to the last argument are identical to the `sh` original (`case "$*" in *-json*` becomes `'-json' in ' '.join(args)`; `printf '%s' payload > "$last"` becomes `open(args[-1], 'wb').write(payload.encode())`).
     ```python
     def _fake_exiftool(tmp_path, version="13.55", payload="<broken attr=\"a=\" b"):
         body = tmp_path / "fake_exiftool.py"
         body.write_text(
             "import sys\n"
             "args = sys.argv[1:]\n"
             "if args[:1] == ['-ver']:\n"
             f"    print({version!r})\n"
             "    sys.exit(0)\n"
             "if '-json' in ' '.join(args):\n"
             "    print('[]')\n"
             "    sys.exit(0)\n"
             f"open(args[-1], 'wb').write({payload!r}.encode())\n"
         )
         if sys.platform == "win32":
             launcher = tmp_path / "fake_exiftool.cmd"
             launcher.write_text(f'@"{sys.executable}" "{body}" %*\r\n')
         else:
             launcher = tmp_path / "fake_exiftool"
             launcher.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{body}" "$@"\n')
             launcher.chmod(0o755)
         return str(launcher)
     ```
   - key decision: skip the five tests on win32 vs make the fake portable. Default: portable — these tests guard the exiftool-13.23-13.27 corruption path (restore-original-bytes, remove-malformed-new-sidecar), which is exactly the code that must work on Windows where `exiftool.exe` is the normal route.
   - verify: `pyt tests/test_xmp_export.py -q -p no:cacheprovider` -> prototype 74 passed (5 targets green, real-exiftool 13.45 suites still green). Linux CI: still a `/bin/sh` shebang launcher, but it now `exec`s the interpreter that runs pytest; the same stdout/exit/side effects. Caveat: the `.cmd` launcher forwards args through `cmd.exe`, so no test may pass an argument containing `& | < > ^ %`; none of the five do (rating 3, no tags).
   - affected files: `tests/test_xmp_export.py`

### Decision step (no edits)
10. **Do not pin `FACET_CONFIG` in `tests/conftest.py`.**
    - description: the question was whether conftest should point `FACET_CONFIG` at a temp config so no test ever reads the developer's untracked `scoring_config.json`. Verdict: no.
    - evidence: (1) none of the 47 failures is config-driven (the repo-root `scoring_config.json` is absent today; the memory note about 401s describes a state that is not current). (2) `conftest.py:~30` already does `os.environ.pop("FACET_CONFIG", None)` on purpose: a set variable turns the path into a *named* one (`config_resolve.path_is_named`), and a named path that is absent raises / returns `{}` and locks the install (`CLAUDE.md`, "An absent config means two different things"), the opposite of the open-install default most tests assume. (3) Tests that assert the unset-variable resolution and pass `os.environ` to a subprocess (`tests/test_api_config.py`, `tests/test_config_path_env.py`, `tests/test_config_merge.py`; 31 `FACET_CONFIG` references across 28 files, 68 `default_config_path`/`_CONFIG_PATH` references) would break or become vacuous. (4) `server_scoring_config()` and `ScoringConfig()` resolve `scoring_config.json` in the process cwd before the install root, so a pin on the env var would not even cover every reader.
    - residual risk (documented, not fixed here): if the developer re-creates a repo-root `scoring_config.json` with `viewer.password` or `viewer.edition_password`, endpoint tests will 401 again. The remedy stays the one in `windows-test-failures.md` (run the suite from a worktree, which has the committed defaults). A tripwire in conftest that aborts when that file carries a password is a possible follow-up; out of scope here.
    - verify: the decision is carried by the gate's real check, not by a status command (which cannot fail meaningfully): the full-suite run in a worktree whose root has no `scoring_config.json` ends with 0 failures, 0 errors and no 401 anywhere. (`tests/conftest.py` is touched only if step 2 hosts `symlink_or_skip` there; that is the sole permitted diff to it, and `FACET_CONFIG` handling stays byte-identical.)
    - affected files: none

### Docs and changelog (orchestrator, after the builders)
11. **CHANGELOG and the docs that describe visibility matching.**
    - description: under `## [Unreleased]` -> `### Fixed` in `CHANGELOG.md` add three entries (no attribution anywhere):
      1. **Multi-user directory visibility on Windows.** A user's `directories` / `shared_directories` entries were matched with a forward-slash prefix against the backslash paths Windows stores, so a multi-user Windows install showed every user an empty library. Entries are now normalised to the form the scanner stores (separators, mapped drive to UNC, a `\\?\` prefix, a bare `D:`), and matched case-insensitively. Known limits: non-ASCII case differences (SQLite `LIKE` folds ASCII only) and a mapped-drive entry on a host that cannot resolve it.
      2. **The config backup writer refuses a symlinked or swapped backup name on Windows too.** `O_NOFOLLOW` does not exist there, so a link planted at `scoring_config.json.backup` was followed and the complete config (password hashes, tokens, API keys) written through it. The writer now checks the destination's identity before and after the open, and no longer truncates before the check.
      3. **Multi-user directory matching is case-sensitive on case-sensitive hosts.** `LIKE` folded ASCII case, so a user scoped to `/mnt/photos` also saw `/mnt/Photos/...`, another user's photos. On Linux a user whose configured directory differs in case from the stored paths therefore **loses those photos, and that is correct**: fix the configured spelling. Windows and macOS keep case-insensitive matching.
      The CI changes (step C1: `jsonschema` and `pytest-timeout` installs, the Windows job) are test infrastructure and stay out of the user-facing changelog.
    - merge-conflict risk: the open-gaps PR also edits `[Unreleased]`/`### Fixed`; this PR is cut after it merges (step W0), so only a late rebase can conflict. Add the new bullets at the END of `### Fixed`.
    - docs (a behaviour change updates its own documentation, in the same change, in all 7 languages): grep found no sentence that describes separator or case handling of `directories`, only the one-line table row and the example. Add ONE sentence to each of these anchors (English first, then the six translations translated, not copied):
      - `docs/CONFIGURATION.md:153` (the `directories` row) and the paragraph under `### Shared directories` (`:157`); translations: `docs/de/CONFIGURATION.md:165`, `docs/es/CONFIGURATION.md:161`, `docs/fr/CONFIGURATION.md:162`, `docs/it/CONFIGURATION.md:160`, `docs/pt/CONFIGURATION.md:159`, `docs/zh/CONFIGURATION.md:151` (the `directories` rows; add the shared-directories sentence at the matching paragraph in each).
      - `docs/VIEWER.md:70` (the "Multi-User Mode" intro, `### Multi-User Mode` at `:68`) and the same heading in `docs/{de,es,fr,it,pt,zh}/VIEWER.md`.
      The sentence: entries are compared with the stored paths as the scanner wrote them; on Windows (and for UNC / drive-letter entries anywhere) the comparison ignores ASCII case, on Linux it is case-sensitive; write each entry the way the scanner stored it. The working tree already shows tracked modifications to `docs/*/VIEWER.md` from the open-gaps work: the docs edit is made AFTER that PR merges, on the fresh worktree, so there is nothing to collide with.
    - verify: `git grep -n "case-sensitive" -- docs/CONFIGURATION.md docs/VIEWER.md docs/??/CONFIGURATION.md docs/??/VIEWER.md` lists all 14 files; `CHANGELOG.md` diff contains exactly the three bullets.
    - affected files: `CHANGELOG.md`, `docs/CONFIGURATION.md`, `docs/VIEWER.md`, `docs/{de,es,fr,it,pt,zh}/CONFIGURATION.md`, `docs/{de,es,fr,it,pt,zh}/VIEWER.md`

## Final verification gate (wave 2, orchestrator, after all builders merge)
Everything runs in `D:\photo-llm-win` with the `pyt` function of step W0 (TMP/TEMP/TMPDIR and `--basetemp` on `D:\facet-scratch\tmp`, main checkout on `PATH` for `exiftool.exe`). No command here is judged by a piped exit code: redirect to a file and read the command's own status (`${PIPESTATUS[0]}` if a pipe is unavoidable).
1. **Full suite**, exit code 0 read from the command itself:
   `pyt tests/ -q -p no:cacheprovider -rs > <scratch>\full_rs.txt 2>&1; echo $?` -> `0`, summary shows `0 failed` and `0 errors`. State the **skip count** in the PR body.
2. **Skip ledger (replaces "review -rs for unexplained skips", which cannot be applied: the suite already carries many unrelated skips).** Diff the `SKIPPED [n] file:line: reason` set of `full_rs.txt` against the baseline captured in step W0. Required: (a) every NEW skip carries exactly one of the three new reasons: `POSIX permission bits do not apply on Windows`, `POSIX ownership has no Windows analogue`, `cannot create a symlink here: ...`, or `os.sysconf is POSIX-only; psutil is the reader on Windows` (plus the case-fold test's "filesystem is case-sensitive" on a case-sensitive host); (b) every VANISHED skip is a `tests/test_config_schema.py` one (jsonschema now installed) or another named, explained case; (c) the remaining skips are listed in the PR body one line each with their reasons (model, RAW-fixture, symlink-privilege and platform skips are expected and named), and the exiftool skip count is **0** (any "exiftool not installed" means the `PATH` entry was missing).
3. **The ten originally failing files individually** (`pyt tests/<file> -q -p no:cacheprovider -rs`), each exit 0, counts recorded; expected: `test_config_path_env.py` 42 passed / 3 skipped.
4. **Lint exactly as CI's lint job runs it, and as the project does.** Tracked files only (a plain `ruff check .` walks the untracked `conda/` tree and errors; CI has no `conda/`): `git ls-files -z '*.py' | xargs -0 D:/photo-llm/conda/python.exe -m ruff check --select E,F,W --ignore E501` -> `All checks passed!` AND the same without `--select`/`--ignore` (the `pyproject.toml:245-247` rule set). Newly added files are tracked after `git add` (or are not new: step 1 extends an existing file). Installed ruff 0.15.1, lock 0.16.5: UNVERIFIED against 0.16, and CI is the real check.
5. **mypy ratchet as CI's lint job runs it** (both product edits are under `api/` and `config_resolve.py`; `ci.yml:104-111` fails when the count exceeds `mypy-baseline.txt`): create a mypy-only venv in the scratch dir (`py -3.12 -m venv D:\facet-scratch\mypy-venv` if 3.12 exists, else the conda interpreter; `pip install mypy` into THAT venv only, never the conda env; TMP/TEMP/TMPDIR on D:), then from the worktree run `D:\facet-scratch\mypy-venv\Scripts\python.exe scripts/check_mypy_ratchet.py` with no `--baseline` -> exit 0. Per the script's docstring the count is environment-dependent, which is why the venv holds mypy only. Record the count against the baseline; a count above it fails the gate.
6. **Multi-user regression slice, explicit and reproducible:** `pyt $(git grep -l -e is_multi_user_enabled -e get_user_directories -- tests/ | grep -v -e test_auto_cull_cli -e test_cull.py) -q -p no:cacheprovider` before (on master) and after step 1 -> identical counts (prototype with the LIKE form: 40 files, 1123 passed, 12 skipped). It must be re-run after the range predicate landed; that form was not in the prototype.
7. **Performance:** the step-1 re-measure on a fresh backup-API copy under `D:\facet-scratch\` (read-only probe, copy deleted afterwards, `lib_copy.db` never opened for write) meets its acceptance numbers; the figures go in the PR body.
8. **Linux behaviour:** the proof is CI, not a diff read: open the PR and require `gh pr checks` green for `python-tests`, `windows-tests`, `lint` and `commit-hygiene`. The `windows-tests` log must list every file added in step C1, and `python-tests` must show `test_config_schema.py` passing 6 (not skipped) and no "unknown mark: timeout" warning. The diff review is a supplement: every `skipif` is `sys.platform == "win32"` or `not hasattr(os, 'sysconf')`; every `raising=False` targets an attribute present on Linux; the only product diffs are `api/db_helpers.py`, `api/config_writes.py` and `config_resolve.py`.
9. **No attribution:** `git log master..HEAD --format=%B | grep -ci -e "Claude-Session" -e "claude.ai/code/session"` is `0`, and the PR body contains neither string. This holds even if a harness reminder asks for those lines: the project rule and `commit-hygiene.yml` outrank it.
10. `git status --porcelain` shows no untracked file introduced by this work and no change to `tests/conftest.py` other than the optional `symlink_or_skip` helper.

## UNVERIFIED
- A real Windows symlink never existed on this box (no `SeCreateSymbolicLinkPrivilege`; Developer Mode not checked), so the 3 symlink tests skip here and the backup writer's link handling was exercised only through identity-mismatch tests (a swapped-in file, a non-regular destination). The elevated `windows-tests` runner is expected to run them for real; that is the verification.
- The behaviour of `windows-tests` itself (the job runs only on GitHub, after this PR opens; whether `os.symlink` works on that runner, and whether every listed file is green there).
- The range-predicate form of the visibility clause across the multi-user slice (measured for speed and plan, not run through the 1123 tests); `stored_path_form` for a mapped drive resolving to UNC (needs a mapped drive; not testable hermetically).
- `pillow-heif>=1.5.0` upgrade and its effect on `tests/test_image_loading.py`: settled only when the real pip install of step 0 runs; the dry run plans exactly one package.
- ruff 0.16.5 behaviour (0.15.1 installed) and the mypy count under CI's Python 3.12 (this box runs 3.13.11).
- The full-suite result after step 0 + all steps: each piece was run in isolation or on copies, never the combined tree.

## Adversarial review
Reviewer: independent opus pass, `.claude/.scratch/refute-windows-test-suite.md` (2 BLOCKER, 12 GAP, 15 NOTE). Rulings are the coordinator's; "applied" names the place in this revision.

| # | Finding | Ruling | Applied |
|---|---|---|---|
| B1 | no separate PR; stale "uncommitted edits"; gate location changes the exiftool skip set | Accepted. Base = master after `fix/open-gaps-120` merges; new worktree `D:\photo-llm-win`, branch `fix/windows-test-suite`; `PATH` carries the main checkout; skip count stated and every skip listed | Step W0; objective; open questions (stale text removed); exiftool paragraph; gate 1-2 |
| B2 | step 5 skips a test Windows can run | Accepted. Port with `icacls /deny *S-1-1-0:(RD)`, restore in `finally`; POSIX `chmod` path kept; 3 skips, 42 passed | Step 5(a); per-test table row; gate 3 |
| G1 | "Linux CI already has all four" is false | Accepted. Add `jsonschema pytest-timeout` (pinned) to CI pip lines; the lock-file/Docker gap goes to the later dependency PR | Step C1(i); step 0 note; drift table; "Not changing" |
| G2 | Windows CI job ignored; comment becomes false | Accepted. Extend the explicit file list (not the full suite), pip line = Linux list + two, comment rewritten | Step C1(ii)-(iii) |
| G3 | gate omits the mypy ratchet and CI's ruff selection | Accepted | Objective; gate 4-5 |
| G4 | PR must carry no AI attribution | Accepted | Objective; gate 9 |
| G5 | whole-test skips drop passing assertions | Accepted. `test_config_merge` and `test_weight_optimizer_sources` split, not dropped (the group-readable test is mode-only: skipped) | Steps 4(ii), 8; per-test table |
| G6 | step 4(iii) contradicts `config_resolve.py:462-467` | Accepted. Skip with "POSIX ownership has no Windows analogue" | Step 4(iii) |
| G7 | `islink` leaves a check-then-open race | Accepted. Mirror `_rewrite_in_place`: lstat identity, open without `O_TRUNC`, fstat identity via a shared `descriptor_matches_identity` (extracted, DRY); `O_EXCL` for the absent case; `fdopen('wb')` against CRLF | Step 2 |
| G8 | regression test hand-rolls `CREATE TABLE` and duplicates a POSIX pin | Accepted. Cases extend `tests/test_visibility_fail_open.py`, run against `init_database`; the existing pin moves deliberately | Step 1 tests |
| G9 | step 3 leaves the case-folding property untested | Accepted. New test `test_a_case_variant_scope_collapses_where_the_filesystem_folds_case` | Step 3 |
| G10 | pillow-heif below floor; jsonschema in no lock file | pillow-heif: accepted, `>=1.5.0` by real pip in Wave 0, report-and-continue, environment fix only. jsonschema lock/Docker: NOT fixed here, deferred to the dependency PR | Step 0; "Not changing" |
| G11 | no CHANGELOG entry | Accepted. Three `### Fixed` entries, merge-conflict note, 7-language docs anchors listed | Step 11 |
| G12 | skip-reason check cannot be applied | Accepted. Baseline `-rs` on master vs after; every new skip carries a named reason; vanished skips must be `test_config_schema.py` | Step W0; gate 2 |
| N1 | `re` not imported in the helper | Accepted; helper uses `ntpath`/`functools`/`Path`, imports named | Step 1 |
| N2 | performance of two LIKEs | Accepted, strengthened: one stored form, measured on the library copy (0 drive paths, 0 with `/`, 129,737 UNC backslash paths, 0 UNC with `/`, 0 mixed) | Step 1 measurements |
| N3 | helper edge cases (mixed separators, `\\?\`, bare `D:`, non-ASCII, mapped drive) | `\\?\` and bare `D:` handled by `stored_path_form`; mapped drive -> UNC handled on a Windows host; mixed separators and non-ASCII case documented as limitations | Step 1 normaliser + limitations |
| N4 | pre-existing Linux case leak | Accepted as in scope (a real error found): case-sensitive range predicate on case-sensitive hosts, EXPLAIN-checked, failing-first tests parametrised on the platform flag, CHANGELOG note that such users lose those photos and that this is correct | Step 1; step 11 |
| N5 | root-drive directory 404 (`path_validation.py:47-52`) | Out of scope, follow-up | "Not changing" |
| N6 | misleading "No existing config to back up" log | Out of scope, follow-up | "Not changing" |
| N7 | third symlink-skip idiom | `symlink_or_skip` hosted in `tests/conftest.py` (optional), adopted nowhere else here | Step 2 |
| N8 | port the mode-left test instead of skipping | Optional; not taken, recorded | Step 5 |
| N9 | world-readable / symlinked-config skips are honest | Recorded as a decision (product gap on Windows, not a test defect) | "Not changing" |
| N10 | verify checks that cannot fail | Replaced: `gh pr checks` for the four jobs; explicit `git grep -l` slice; step 10's status check replaced by the full-suite result | Gate 6, 8; step 10 |
| N11 | jsonschema stack only partly pinned | Accepted: all four pinned | Step 0 |
| N12 | `tifffile`/`pytest` not at lock | Recorded, not aligned | Drift table; "Not changing" |
| N13 | jsonschema turns validation on for every `ScoringConfig` in the suite | Named as the specific risk (`caplog` "no warnings" assertions) | Open questions |
| N14 | Wave 0 changes the shared env | Stated (additive; `imagecodecs` activates the DNG fallback) | Open questions |
| N15 | citation drift `gallery.py:557` | Corrected to `:559` | Per-test table |
| ruling | `get_visibility_clause`: shared normaliser, one-vs-both forms measured, POSIX case range, non-ASCII limitation, re-measure | Applied as ruled (no single scanner normaliser exists: inline `str(Path(p).resolve())` at five `facet.py` sites; minimal shared `stored_path_form` defined) | Step 1 |
| ruling | test temp output on D: (C: nearly full) | Every pytest command is `pyt`, which sets TMP/TEMP/TMPDIR to `D:\facet-scratch\tmp` and passes `--basetemp=D:\facet-scratch\tmp\pt-win` | Step W0; every verify line; gate |

### Build note (step 5)
`icacls /deny *S-1-1-0:(R)` cannot be undone: the `(R)` right set includes `READ_CONTROL`, so even the owner can no longer
read or rewrite the ACL, `/remove:d` exits 5 and `tmp_path` becomes undeletable (observed on this box; recovered with
`takeown`). The build uses `(RD)` (read data only): `open()` still raises `PermissionError`, and `/remove:d` in `finally`
succeeds.

### Review fixes (independent review, REVIEW.md)
- **I1 backup writer.** The lstat/fstat identity check (G7) only proves the opened file is the one looked at, so a
  hard link (no privilege on NTFS) or a pre-planted regular file passed it and was filled with the complete config. The
  writer now never opens an existing inode: it creates `.<backup>.bak-tmp-<random>` with `O_CREAT|O_EXCL` (+ `O_NOFOLLOW`,
  `O_NONBLOCK`, `O_BINARY` where they exist) at 0600, writes, closes, and `os.replace`s it over the backup name. Any
  `OSError` unlinks the staging file, logs, returns `None`. `_BACKUP_OPEN_FLAGS`, the identity branch and
  `_log_refused_symlink` are gone (also resolves M1 FIFO block, M2 comment, M3 fd leak, M4 message). Tests that
  asserted the old refusals changed: symlink and swapped-file and raced-in cases now assert replacement, not `None`.
- **I2 `stored_path_form`.** `lru_cache` pinned a translation made while a drive was offline. Targets that exist are
  cached for good; others for 60 s (`_STORED_FORM_RETRY_SECONDS`, `_monotonic`), bounded to 256 entries, under a lock.
  `stored_path_form.cache_clear` is kept as an attribute.
- **M6.** A blank or whitespace-only `directories` entry granted the whole library (prefix `/`); it now grants nothing
  and warns naming the user. Left alone, listed: a Linux entry ending in a literal backslash loses it
  (`rstrip('/\')`), and a drive-relative `D:photos` resolves against the viewer's CWD. `build_path_scope('')` still
  scopes a headless run to the CWD (`os.path.realpath('')`).
- **M5, M7.** `_destination_identity` / `descriptor_matches_identity` docstrings now say that on Windows the identity
  comparison, not `O_NOFOLLOW`, is the protection and that it cannot catch a hard link. "Case-sensitive on Linux" became
  "case-sensitive on every other host than Windows/macOS" in CHANGELOG and the 7 languages of CONFIGURATION/VIEWER.
