# Image fixtures

Real camera files, kept small. Regenerating or replacing one means rechecking the
NCLX values the tests assert on — that metadata is the point of the fixture, not
the picture.

## `canon_eos_r8_hdr_pq.hif` — 75 KiB, 600×400, 10-bit

An unmodified in-camera HDR PQ still from a Canon EOS R8.

| field | value |
|---|---|
| `Make` / `Model` | Canon / Canon EOS R8 |
| `color_primaries` | 9 — BT.2020 |
| `transfer_characteristics` | 16 — SMPTE ST 2084 (PQ) |
| `matrix_coefficients` | 1 — BT.709 |
| `full_range_flag` | 1 |

Note the matrix is **1, not 9**: an HDR PQ HEIF does not have to carry BT.2020
non-constant luminance, so nothing may gate on that field.

Decoded as plain 8-bit sRGB the frame reads mean 92.6/255 with p99 153 and no
pure-white pixel anywhere — the "dark and washed out" bug this fixture exists to
pin. It is also the regression guard for the tone curve's white point: a curve
normalised at diffuse white instead of the signal peak turns 21 648 of its
240 000 pixels pure white, inventing clipping in a frame that has none.

Source: <https://en.bandisoft.com/bandiview/help/hdr-samples/> ("Sample files for
HDR support"), retrieved 2026-09-22. Published by Bandisoft for testing HDR
support; no redistribution licence is stated on that page. Vendored here for
tests only.

## `sony_a7sm3_sdr.hif` — 108 KiB, 400×267

A Sony ILCE-7SM3 `.HIF`, downscaled from 4240×2832 and re-encoded at quality 40
with its original NCLX values preserved.

| field | value |
|---|---|
| `color_primaries` | 1 — BT.709 |
| `transfer_characteristics` | 13 — sRGB / sYCC |
| `matrix_coefficients` | 5 — BT.470BG |

`.HIF` is **not** a Canon-only extension and does **not** imply HDR: Sony's A7S III
and A7 IV write SDR stills under it. That is exactly why the loader gates on the
NCLX transfer rather than on the file extension, and this fixture is the case that
would break an extension-based gate.

Source: `tests/images/heif_other/cat.hif` in
<https://github.com/bigcat88/pillow_heif> (BSD-3-Clause), retrieved 2026-09-22.

## `apple_iphone13pro_gainmap.heic` — 233 KiB, 1512×850, 8-bit

An unmodified iPhone 13 Pro HDR still (iOS 17.6.1). Vendored to pin the third
NCLX state, which no synthetic fixture covers honestly: **no NCLX box at all**.

| field | value |
|---|---|
| `nclx_profile` | absent — `img.info` has no such key |
| `icc_profile` | 536 bytes, Display P3 |
| `AuxiliaryImageType` | `urn:com:apple:photo:2020:aux:hdrgainmap` |
| `BitDepthLuma` | 8 |

Apple HDR stills are **not** PQ. They are an 8-bit SDR base image plus an
auxiliary gain map — a different HDR mechanism entirely — so the base image is
already the right thing to display and the loader must pass it through
untouched. The Sony fixture exercises `_heif_is_pq`'s "NCLX present, transfer is
not 16" path; this one exercises the `bool(nclx)` falsy path, and it is the
common case rather than an exotic one.

Source: `tests/data/hdr-sample.heic` in
<https://github.com/johncf/apple-hdr-heic> (MIT, © 2024 John Charankattu),
retrieved 2026-09-22. Fetch it through `media.githubusercontent.com/media/...`
— the file is Git LFS, so `raw.githubusercontent.com` serves the 131-byte
pointer instead.
sha256 `c690fa4ecc6ae71ee9a926de869a95835a5d41e764ccbc200647e2fb9d0800cc`

## `pq_gradient_ramp.hif` — 3 KiB, 1024×8, 10-bit — synthetic

The one deliberately synthetic fixture here, and the reason is the whole point of
it: **no real photograph can show this regression.** Sensor noise dithers across
the quantisation steps and fills the comb back in, so the Canon frame above scores
248 of 256 occupied luma bins whether it is decoded at 8-bit or at its native
10-bit. Smooth content — clear sky, a studio backdrop, heavy bokeh — has no noise
to dither with, and that is where a decode that throws away two bits before the PQ
EOTF expands them shows as a visible contour.

One 10-bit code per column, 0–1023 left to right, grey (R=G=B), encoded losslessly.

| field | value |
|---|---|
| `color_primaries` | 9 — BT.2020 |
| `transfer_characteristics` | 16 — SMPTE ST 2084 (PQ) |
| `matrix_coefficients` | 9 — BT.2020 non-constant luminance |
| `full_range_flag` | 1 |
| `bit_depth` | 10 |

Through the shipped tone map the two decode paths separate cleanly: the 8-bit path
reaches 170 distinct output levels with 86 empty interior luma bins, the native
10-bit path reaches all 256 with none. That gap is what
`test_pq_gradient_ramp_native_decode_has_no_posterisation_gaps` asserts.

Regenerate with (note `transfer_characteristics`, **plural** — the singular spelling
is accepted and silently ignored, and the file comes back as transfer 13, sRGB):

```python
import numpy as np, pillow_heif
h, w = 8, 1024
ramp = (np.arange(w, dtype=np.uint16) << 6)          # 10-bit codes, left-shifted to 16
rgb = np.repeat(np.stack([ramp] * 3, axis=-1)[None, :, :], h, axis=0).copy()
pillow_heif.from_bytes(mode='RGB;16', size=(w, h), data=rgb.tobytes()).save(
    'tests/fixtures/pq_gradient_ramp.hif', quality=-1, chroma=444,
    save_nclx_profile=True, color_primaries=9, transfer_characteristics=16,
    matrix_coefficients=9, full_range_flag=1)
```

The round trip is exact: reopening with `convert_hdr_to_8bit=False` returns the
same 1024 distinct values per channel.

## `faces_update_database_golden.json` — 146 KB — synthetic, produced by pre-change code

Not an image. The expected `faces.person_id` and `persons` rows after
`FaceClusterer._update_database` runs five cases (incremental, a stable second run,
incremental with new faces, named-only, force) on the synthetic library built by
`tests/test_cluster_update_database_parity.py::build_library`: about 300 unit-norm
embeddings in 8 blobs, 10 persons, a second embedding space, noise faces and legacy
faces with no stored thumbnail. Centroids are stored as base64 of the exact float32
bytes; person thumbnails as their sha256. No photo file is read (the decode is
stubbed and the paths do not exist).

It must come from the code BEFORE the owner write was made diff-only, because it is
the oracle for that change. Produced from commit `8a2bb440` (`git archive HEAD` into
a scratch directory, nothing checked out) with a driver that puts that tree first on
`sys.path` and calls the test module's own `run_all_cases()`:

```
git archive HEAD | tar -x -C <scratch>/head
PARITY_TEST_FILE=<repo>/tests/test_cluster_update_database_parity.py \
  python -I dump_golden.py <scratch>/head <scratch>/golden_scratch golden.json 8a2bb440...
```

`dump_golden.py` is ~20 lines: `os.chdir(tree)`, `sys.path.insert(0, tree)`, load the
test module by path, assert `faces.clusterer.__file__` lies under `tree`, call
`run_all_cases(scratch)`, `json.dumps(..., indent=1, sort_keys=True)`, and collapse each
`[face_id, person_id]` pair onto one line. Regenerate it only from a tree that predates
the change (or after deliberately changing the scenario in `build_library`, which
changes the expected rows). On that pre-change tree the parity tests pass and the
behaviour tests (atomicity, id order, no decode inside a transaction, compare-and-set)
fail, which is how the file is known to describe HEAD and not the new code.
