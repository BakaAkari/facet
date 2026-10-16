# Protected capture sequences (Kari fork)

Development branch: `kari/sequence-protection`; `master` remains the upstream baseline.
Images: `ghcr.io/bakaakari/facet:latest-cuda` and immutable `sha-<commit>-cuda`.
The dedicated workflow tests before publishing a linux/amd64 CUDA image.

## Safety boundary

Validated Fuji PixelShiftShots (4/16) or Sony PixelShiftInfo source-frame evidence
protects even incomplete sets. Sony composed frames and disabled values are excluded.
Fuji offsets split consecutive runs; absent offsets produce protected singleton sets
rather than fabricated complete sets. Sony hardware group ids are scoped by body,
directory and date. Group identity includes sequence kind.

Timelapse detection requires at least 15 frames, 1–3600 second cadence, at most 15%
interval deviation, identical focal length and low perceptual-hash distance from the
first frame. Camera serial, model and directory separate candidates. This is a
heuristic, not proof of photographic intent; equal-cadence ordinary captures can match.
Sequences with moving scenes, missing metadata or large gaps can be missed. Before
15 frames are present there is no reliable timelapse classification: **do not cull
an incomplete import**. Missing serial numbers cannot distinguish identical bodies
shooting into the same directory. Preserve originals and verify detection previews.

Once recognised, protection is sticky. Rescanning cannot silently revoke it. Protected
frames are excluded from burst grouping, bracket and panorama candidate passes.
Reject, move/trash, delete and sequence-override API boundaries refuse protected paths
with HTTP 409, including indexed RAW companions. There is deliberately no ordinary
UI/API action to remove this protection. Existing rejection flags are not erased;
file-operation guards still block those stale decisions. Unindexed companions with
missing metadata are not guaranteed to be recognised: import/backfill the RAW library
before destructive workflows. This is not filesystem-level write protection.

## Existing library

Back up the database/config first. Pause scanning and editing for an apply operation.
The dry run clones the SQLite database and only reads source metadata:

```sh
python scripts/backfill_capture_sequences.py --database /path/library.db --report /path/preview.json
# Review groups and metadata_failures; exit 2 means incomplete metadata extraction.
python scripts/backfill_capture_sequences.py --database /path/library.db --report /path/applied.json --apply
```

`--min-frames` (minimum 15) and `--tolerance` configure this backfill. Automatic passes
currently use conservative module defaults. No backfill command moves, deletes or
writes a photo/XMP. An interrupted apply can be rerun, but may leave only part of the
library labelled until completed. The metadata fields use additive schema migration.

## Validation

```sh
venv/bin/python -m unittest discover -s tests_capture -v
venv/bin/python -m pytest tests/test_capture_sequences.py tests/test_capture_api.py tests/test_sequence_detection.py tests/test_burst_culling.py tests/test_cull.py tests/test_auto_cull.py -q
cd client
npm ci
npm run gen:i18n
PYTHON=/absolute/path/to/venv/bin/python npm run gen:api
npx tsc --noEmit -p tsconfig.app.json
npx tsc --noEmit -p tsconfig.spec.json
npm run build
```

## NAS rollout and rollback

Do not replace the live stack solely because CI is green. Record its current image
ID/digest and compose, back up config plus SQLite using a consistent backup, then run
the candidate with isolated writable data and a **read-only** photo mount. Verify UI,
CUDA computation and metadata extraction against actual source files. Backfill a copy
first. Change only Facet's image reference after this validation. Preserve the old
image and database backup; rollback restores both together if necessary. Never run
both versions against the same writable database.

## Upstream maintenance

```sh
git fetch upstream
git switch master && git merge --ff-only upstream/master
git switch kari/sequence-protection
git merge upstream/master
# rerun tests, review migration changes, then push the integration branch
```

Do not force-push shared history. The upstream publish workflow remains untouched;
this fork publishes through `.github/workflows/protected-cuda.yml`.
