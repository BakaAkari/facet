# Face Recognition

> 🌐 **English** · [Français](fr/FACE_RECOGNITION.md) · [Deutsch](de/FACE_RECOGNITION.md) · [Italiano](it/FACE_RECOGNITION.md) · [Español](es/FACE_RECOGNITION.md) · [Português](pt/FACE_RECOGNITION.md) · [简体中文](zh/FACE_RECOGNITION.md)

Facet uses InsightFace for face detection and HDBSCAN for clustering faces into persons.

## Overview

1. **Detection** - InsightFace buffalo_l model detects faces and extracts 512-dim embeddings
2. **Clustering** - HDBSCAN groups similar embeddings into person clusters
3. **Management** - Web viewer for merging, renaming, and organizing persons

## Complete Workflow

### Step 1: Extract Faces

During photo scanning, faces are automatically extracted:

```bash
python facet.py /path/to/photos
```

For existing photos without faces:

```bash
python facet.py --extract-faces-gpu-incremental  # New photos only
python facet.py --extract-faces-gpu-force        # All photos (deletes existing)
```

### Step 2: Cluster Faces

Group similar faces into persons:

```bash
python facet.py --cluster-faces-incremental  # Preserves existing persons
```

**Clustering modes:**

| Command | Behavior |
|---------|----------|
| `--cluster-faces-incremental` | Preserves all persons, matches new to existing |
| `--cluster-faces-incremental-named` | Preserves only named persons |
| `--cluster-faces-force` | Deletes all persons, full re-cluster |

### Step 3: Review and Merge

Find duplicate person clusters:

```bash
python facet.py --suggest-person-merges
python facet.py --suggest-person-merges --merge-threshold 0.7  # Stricter
```

This opens the merge-suggestions page in the browser.

### Step 4: Manage in the Viewer

The remaining work happens in the web viewer, following the pipeline **Extract → Cluster → Merge → Manage**:

- **Merge** duplicate clusters on the Merge Suggestions page.
- **Manage** persons (merge, batch merge, split, hide, rename, delete) on the Manage Persons page.

See [Viewer Integration](#viewer-integration) for the full UI reference.

## Configuration

### Face Detection

```json
{
  "face_detection": {
    "min_confidence_percent": 65,
    "min_face_size": 20,
    "blink_ear_threshold": 0.28
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `min_confidence_percent` | `65` | Minimum detection confidence |
| `min_face_size` | `20` | Minimum face size in pixels — InsightFace only: the commercial-mode YuNet detector uses fixed measured values (0.7 confidence, 30 px) and ignores `min_confidence_percent` and this setting. |
| `blink_ear_threshold` | `0.28` | Eye Aspect Ratio for blink detection |

### Face Clustering

```json
{
  "face_clustering": {
    "enabled": true,
    "min_faces_per_person": 2,
    "min_samples": 2,
    "clustering_algorithm": "best",
    "leaf_size": 40,
    "use_gpu": "auto",
    "thresholds_by_model": {
      "arcface_buffalo_l": { "auto_merge_distance_percent": 15, "merge_threshold": 0.6 },
      "sface": { "auto_merge_distance_percent": 27.5, "merge_threshold": 0.65 }
    }
  }
}
```

| Setting | Default | Description |
|---------|---------|-------------|
| `min_faces_per_person` | `2` | Minimum photos to create a person |
| `min_samples` | `2` | HDBSCAN min_samples parameter |
| `thresholds_by_model` | see JSON | Per recognition model: `auto_merge_distance_percent` (auto-merge distance) and `merge_threshold` (centroid similarity for matching). Defaults: ArcFace `15` / `0.6`, SFace `27.5` / `0.65` |
| `use_gpu` | `"auto"` | GPU mode: `auto`, `always`, `never` |

### Face Processing

```json
{
  "face_processing": {
    "crop_padding": 0.3,
    "use_db_thumbnails": true,
    "face_thumbnail_size": 640,
    "face_thumbnail_quality": 90,
    "extract_workers": 2,
    "extract_batch_size": 16,
    "refill_workers": 4,
    "refill_batch_size": 100
  }
}
```

## Clustering Algorithms

For CPU clustering, choose the algorithm based on dataset size:

| Algorithm | Complexity | Best For |
|-----------|------------|----------|
| `boruvka_balltree` | O(n log n) | High-dimensional (recommended for 50K+ faces) |
| `boruvka_kdtree` | O(n log n) | Low-dimensional data |
| `prims_balltree` | O(n²) | Small datasets, memory-constrained |
| `prims_kdtree` | O(n²) | Small datasets |
| `best` | Auto | Let HDBSCAN decide |

**Performance note:** For large datasets, use `boruvka_balltree`. With 80K faces it completes in 2-5 minutes, where exact algorithms can hang.

## GPU Clustering (cuML)

For large datasets (80K+ faces), GPU clustering via RAPIDS cuML is faster than CPU.

### Installation

```bash
# Conda
conda install -c rapidsai -c conda-forge -c nvidia cuml cuda-version=12.0

# Pip
pip install --extra-index-url https://pypi.nvidia.com/ "cuml-cu12"
```

### Configuration

```json
{
  "face_clustering": {
    "use_gpu": "auto"
  }
}
```

| Mode | Behavior |
|------|----------|
| `"auto"` | Use GPU if cuML available, fallback to CPU |
| `"always"` | Try GPU, warn and fallback if unavailable |
| `"never"` | Always use CPU |

**Note:** cuML uses its own HDBSCAN implementation. The `algorithm` and `leaf_size` parameters only apply to CPU clustering.

### GPU and CPU do not produce identical clusters

Measured on a 145,677-face library, same embeddings and same parameters:

| Faces | cuML clusters | CPU clusters | Ratio |
|------:|--------------:|-------------:|------:|
| 5,000 | 202 | 181 | 1.12× |
| 20,000 | 1,285 | 1,039 | 1.24× |

cuML is slightly finer-grained, with comparable noise rates and an identical median cluster size (4). The difference is small enough not to change who ends up grouped with whom, but it does mean cluster *counts* will not match between a GPU and a CPU run.

**cuML is also not deterministic.** Two consecutive GPU runs over the same 145,677 faces with identical parameters agreed on only **41% of labels**, while producing near-identical cluster counts (14,809 vs 14,807) and identical downstream outcomes. The structure is stable; the labelling is not. Consequences worth knowing:

- **Person IDs are not reproducible across runs.** Do not assume a re-cluster preserves the numbering of auto-clustered persons — name the people you care about, which is what makes them stable.
- **Do not diff two clustering runs by label.** Compare face membership instead.

CPU clustering with a fixed `algorithm` is deterministic for the same input.

### Re-clustering and curated people

Every mode except `--cluster-faces-force` preserves existing person records, but all of them re-derive every face→person assignment from scratch. Only the faces whose person actually changes are written, all in one transaction, so a run interrupted before that commit leaves every face's person as it was. A run has two phases: loading and clustering the faces, the long part, and then a short database phase that writes the result (about 13 s on a copy of a 145,677-face library, measured). A face you reassign in the viewer during the database phase keeps your assignment: the run changes a face only if it still has the person the run read at the start of that phase, and logs how many faces it skipped. A save sent while the run is writing waits for it (2.8 s on the 145,677-face copy); one that waits more than 5 s is refused with "library busy" and must be repeated. An edit made earlier, while the faces are still being loaded or clustered, is overwritten, as it was before. A rebuilt cluster is returned to a person when either:

1. its mean embedding is within `merge_threshold` (default 0.6 cosine for ArcFace, 0.65 for SFace) of that person's stored centroid, **or**
2. at least half of its faces belonged to that person before the run.

The second rule matters more than it sounds. A person's centroid is a single averaged vector, so someone photographed across years spans a broad region — HDBSCAN correctly splits them into tight sub-clusters that each sit far from that average and fail rule 1. On the library above, rule 1 alone returned 38% of the faces attached to named people; adding rule 2 returns 96%.

Because rule 2 defers to the previous assignment, an earlier mis-assignment is carried forward rather than corrected. That is the intended contract for a mode whose purpose is preserving existing persons — use `--cluster-faces-force` to re-derive everything from scratch. When a cluster's faces split exactly evenly between two people, the tie is broken by face id order: the person who owns the cluster's lowest-numbered previously assigned face is kept. Embeddings are loaded `ORDER BY id`; older versions left that order unspecified, so in such a tie the kept person id can differ from an older run.

## Blink Detection

Uses Eye Aspect Ratio (EAR) from InsightFace 106-point landmarks.

### How It Works

EAR measures the ratio of eye height to width. When eyes close, EAR drops below the threshold.

### Configuration

```json
{
  "face_detection": {
    "blink_ear_threshold": 0.28
  }
}
```

Lower threshold = stricter detection (more photos flagged as blinks).

### Recompute After Threshold Change

```bash
python facet.py --recompute-blinks
```

Only processes photos with faces, no GPU needed.

## Optional OFIQ face quality (commercial mode)

In commercial mode the restricted face-quality model is unavailable. As an optional replacement, install `python-ofiq` (UNICEF's bindings over BSI's OFIQ, ISO/IEC 29794-5; MIT; it coexists with numpy 2 and OpenCV 4.8+) with `pip install python-ofiq`. It ships wheels for Linux x86-64 and macOS arm64 only (Windows users: WSL2 or Docker). The OFIQ models (about 400 MB, MIT/Apache-2.0) are never bundled: on first use Facet downloads them, SHA-256 pinned, into `pretrained_models/ofiq/`. The commercial face pipeline divides OFIQ's unified score (0-100) by 10 and stores it in `face_quality_iqa`; a photo's value is 0.7 x its worst face + 0.3 x the mean. When `python-ofiq` is absent the column stays NULL and its weight is renormalised away.

## Per-face Expression Signals (eyes-open + smile)

Each face row stores two continuous 0-10 signals used by the culling face panel
and the photo-level aggregates: `eyes_open_score` (10 = wide open, 0 = fully
closed) and `smile_score` (5 = neutral, 10 = broad smile, 0 = frown).

Two backends produce them, on the same 0-10 scale:

1. **Geometry (always available).** Derived from the stored InsightFace 106-point
   landmarks: Eye Aspect Ratio for eyes-open, mouth-corner lift for smile. Pure
   geometry, so `--recompute-face-signals` can backfill them from stored
   landmarks with no pixels and no GPU.
2. **MediaPipe blendshapes (optional, appearance-based).** During scanning /
   face extraction, a generous crop of each face is run through the MediaPipe
   Face Landmarker, whose ARKit-style blendshapes (`eyeBlink*`, `mouthSmile*`,
   `mouthFrown*`) map to the same scales. Appearance beats landmark geometry on
   closed eyes, subtle smiles and off-axis heads, so when a face scores through
   MediaPipe it **replaces** the geometric value. If MediaPipe or its model
   bundle is absent, or the face crop is too small / undetected, the geometric
   value is kept — behaviour is identical to a geometry-only install.

### Installing MediaPipe

MediaPipe is optional and **must** be installed without its bundled
`opencv-contrib-python`, which would install a second `cv2` namespace alongside
Facet's `opencv-python`:

```bash
pip install mediapipe==0.10.35 --no-deps
pip install absl-py flatbuffers
```

Never run a plain `pip install mediapipe`.

### Model bundle

The `face_landmarker.task` bundle (~3.6 MiB, Apache-2.0) is auto-downloaded on
first use to `pretrained_models/face_landmarker.task`. If the machine is
offline, download it manually from
`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
and place it at that path. A failed download logs a warning once and falls back
to the geometric scores.

### Configuration

```json
{
  "face_detection": {
    "blendshapes": {
      "enabled": true,
      "min_crop_size": 192,
      "blink_pose_gate_deg": null
    }
  }
}
```

- `enabled` (default `true`): use blendshape scores whenever MediaPipe and the
  model bundle are available; otherwise the geometry fallback runs automatically.
  Set `false` to force geometry-only.
- `min_crop_size` (default `192`): faces whose padded crop is smaller than this
  (px, shorter side) fall back to geometry rather than upscale a tiny face.
- `blink_pose_gate_deg` (default `null` = off): commercial-mode (MediaPipe) analyzer only. A face whose |yaw| or |pitch| (degrees, from the Face Landmarker's facial transformation matrix) exceeds this value gets no blink verdict and no eyes-open / smile score (NULL = unknown), because a turned head reads as closed eyes. It stays off by default: on a held-out half of a real library, no cutoff between 20° and 45° improved blink detection (each lost more true blinks than false ones), so none ships; set it only if your library has many strongly turned faces; it applies to newly scanned photos, existing `is_blink` values change only on rescan. Must be a positive number of degrees: 0 or a negative value is rejected at startup; `null` = off.

### Recompute

`--recompute-face-signals` recomputes the per-face signals from stored landmarks
only — it is **geometry-only** and does not run MediaPipe (no pixels are read).
To refresh the appearance-based scores, re-extract faces
(`--extract-faces-gpu-force`) so the full-resolution crops are re-analysed.

## Face Thumbnails

Thumbnails are stored in the database for fast display.

### Storage

- Generated during scanning from full-resolution images
- Stored in `faces.face_thumbnail` column as JPEG BLOBs (~5-10KB each)
- Used by clustering and viewer instead of regenerating

### Regeneration

```bash
# Generate missing thumbnails
python facet.py --refill-face-thumbnails-incremental

# Regenerate ALL thumbnails
python facet.py --refill-face-thumbnails-force
```

Both commands use parallel processing for speed.

## Database Schema

### faces Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `photo_path` | TEXT | Foreign key to photos |
| `face_index` | INTEGER | Index within photo |
| `embedding` | BLOB | 512-dim face embedding |
| `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` | INTEGER | Bounding box corners |
| `confidence` | REAL | Detection confidence |
| `person_id` | INTEGER | Foreign key to persons |
| `face_thumbnail` | BLOB | JPEG thumbnail |
| `landmark_2d_106` | BLOB | 106-point landmarks (blink detection) |
| `embedding_model` | TEXT | Recognition model tag (default `arcface_buffalo_l`) |

### persons Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `name` | TEXT | Person name (NULL = auto-clustered) |
| `representative_face_id` | INTEGER | Best face for avatar |
| `face_count` | INTEGER | Number of faces |
| `centroid` | BLOB | Cluster centroid embedding |
| `auto_clustered` | INTEGER | 1 if auto-generated |
| `face_thumbnail` | BLOB | Person avatar thumbnail |
| `is_hidden` | INTEGER | 1 = excluded from filters/suggestions |

## Incremental vs Force Modes

### Incremental Clustering

- Preserves all existing persons (named and auto-clustered)
- Clusters only new, unassigned faces
- Matches new clusters to existing persons via centroid similarity
- Updates centroids after merging

**Use when:** Adding new photos to existing collection

### Force Clustering

- Deletes ALL persons including named ones
- Full re-cluster from scratch

**Use when:** Starting fresh or major algorithm changes

### Incremental-Named Clustering

- Preserves only named persons
- Deletes auto-clustered persons
- Re-clusters all unnamed faces

**Use when:** Maintaining curated names while refreshing auto-detected clusters

## Viewer Integration

### Person Filter

- Dropdown shows persons with face thumbnails
- Filter gallery by person

### Person Gallery

- Click person in dropdown to view all their photos
- Clicking a person applies a `person_id` filter on the gallery (no dedicated per-person route)

### Manage Persons Page

Access via header button or `/persons`:

- **Grid View** - All recognized persons
- **Merge** - Select source, click target, confirm
- **Batch Merge** - Select multiple persons and merge into one target
- **Split** - Move selected faces into a new person
- **Hide** - Exclude a cluster from the list, filters, and merge suggestions
- **Delete** - Remove person cluster
- **Rename** - Click name to edit inline

### Create a Person

Persons no longer come only from clustering — you can name a face the clusterer
missed straight from the gallery:

1. On a photo card, open the person actions and pick an unassigned face.
2. In the person picker, choose **Create new person** and type a name.
3. The face is attached to the new (manually created, `auto_clustered = 0`)
   person in one call.

Endpoint: `POST /api/persons` (edition-gated), body
`{ "name": "<name>", "face_ids": [<id>, ...] }`. The name is required (non-empty
after trimming). Faces that already belong to another person are reassigned, and
any old person left with no faces is deleted — the same semantics as face
assignment. In multi-user mode the caller may only attach faces from photos
inside their own (or shared) directories; a face outside that scope is rejected
as not found.

### Needs Naming

The Manage Persons page surfaces auto-clustered persons that are worth naming in
a **Needs naming** section: unnamed clusters (`name IS NULL`,
`auto_clustered = 1`) with at least `viewer.persons.needs_naming_min_faces`
faces (default `5`), each with an inline name field so large clusters can be
named without hunting for them. Served by
`GET /api/persons/needs_naming?min_faces=N`.

### Merge Suggestions Page

Access via `/merge-suggestions` or the "Merge Suggestions" button on the Manage Persons page:

- Shows pairs of persons with similar face embeddings that may be the same individual
- **Threshold slider** — controls similarity cutoff (lower = more suggestions)
- **One-click merge** — merge a suggested pair instantly
- **Batch merge** — select multiple suggestions and merge them all at once

### Photo Cards

- Small face thumbnails (avatars) shown for recognized people
- Configurable via `viewer.face_thumbnails.output_size_px`

## Embedding-space marker (recognition-model safety)

Every face row carries an `embedding_model` tag (column on `faces`, default
`arcface_buffalo_l` — the current InsightFace `buffalo_l` / ArcFace `w600k_r50`
recognition model). Embeddings produced by **different** recognition models live
in **incompatible vector spaces** and must never be clustered together — doing so
silently produces garbage persons with no error.

`FaceClusterer.load_embeddings()` therefore loads only the **active** embedding
space (`active_embedding_model()` in `db/embedding_space.py`; a `NULL` tag is treated
as the legacy ArcFace space) and logs a loud warning if faces from any other
space are present and excluded. This is a forward-compatibility guard: it makes a
future recognition-model swap safe by construction.

### Swapping the recognition model (e.g. AdaFace) — deferred plan

A quality upgrade such as **AdaFace** (quality-adaptive margin, better clustering
of blurry/candid faces) is integrable as an opt-in 512-d backend (same storage
path, same HDBSCAN), but is **not yet implemented** because it cannot be
validated without real data. Doing it correctly requires:

1. **Weights + backbone** — an AdaFace checkpoint (e.g. `adaface_ir101_webface12m`)
   plus its IResNet backbone; a new model-cache download.
2. **Aligned crops** — compute the embedding from a `norm_crop(img, face.kps, 112)`
   aligned 112×112 crop at extraction time (the kps exist on the InsightFace
   `face` object but aren't persisted, so AdaFace cannot be back-filled offline —
   it must run during extraction). Verify BGR/normalization match the checkpoint.
3. **Config switch** — add `face_detection.recognition_model: arcface|adaface`
   and resolve `active_embedding_model()` from it; tag new faces accordingly.
4. **Full re-extraction + re-cluster** — `--extract-faces-gpu-force` then
   `--cluster-faces-force`, because ArcFace and AdaFace embeddings are not
   comparable. The embedding-space marker above prevents a half-migrated DB from
   silently clustering the two spaces together (it warns and excludes instead).
5. **Quality validation** — measure cluster quality against labelled identities;
   "runs and emits 512-d vectors" does not prove the preprocessing is correct.

### Mixed libraries, re-extraction and unknown values

- **Mixed libraries.** A library may hold ArcFace (512-d) and SFace (128-d) faces at once. The clusterer, merge suggestions, the similar-person search and person centroids each use one model at a time (`persons.centroid_model` records which one built a centroid); the incremental and named-only clustering modes leave faces and persons of the other model untouched, but `--cluster-faces-force` clears every face assignment and deletes every person, named ones included, whatever their model. To move a library to the active model, run `--extract-faces-gpu-force` (in commercial mode it uses the permissive extractor; named persons are carried over) and then `--cluster-faces-incremental`.
- **Warnings.** A scan logs a warning when stored faces belong to a model other than the active one, and `--doctor` reports the same mismatch against the active model. In commercial mode `--doctor` lists the components a scan would actually load (YuNet, SFace, Q-ReAlign, plus OFIQ when `python-ofiq` is installed), not those the VRAM profile merely names, so TOPIQ, LIQE and InsightFace no longer appear as warnings.
- **Named persons survive re-extraction.** Before `--extract-faces-gpu-force` deletes the face rows, Facet remembers the box of every face belonging to a named person; afterwards each re-detected face is given back to the person whose old box overlaps it by IoU >= 0.5 (one-to-one, best overlap first), and centroids are rebuilt per model. A face that matches no old box stays unassigned for the next clustering run. The remembered boxes are stored in the `face_reextract_snapshot` table in the same transaction as the delete, so an interrupted run is replayed by the next extraction. Rescanning a photo keeps its named persons the same way. Person centroids that a rescan left stale are rebuilt once when the scan ends (even if it is interrupted), not once per batch; counts and representative faces are still updated per batch.
- **Unknown is not zero.** Without MediaPipe the permissive analyzer cannot measure eyes, smile or blinks, so those values are stored as NULL (not 0, and not "blinking"); the blink filter and the blink penalty treat NULL as "not a blink". The 106-point landmark readers skip faces without landmarks. A photo whose face pass fails is stored with `face_count`, `face_ratio` and `is_group_portrait` as NULL (unknown), not 0, and its stored faces are kept; scoring treats it as having no face until it is rescanned.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Clustering hangs | Use `boruvka_balltree` algorithm |
| Too many small clusters | Increase `min_faces_per_person` |
| Faces not grouping | Decrease `merge_threshold` |
| GPU clustering fails | Check cuML installation, use `"never"` to force CPU |
| Thumbnails missing | Run `--refill-face-thumbnails-incremental` |
| Wrong blink detection | Adjust `blink_ear_threshold`, run `--recompute-blinks` |
| "Excluded N faces from non-active embedding space" warning | A recognition-model change left mixed embeddings — run `--extract-faces-gpu-force` then `--cluster-faces-incremental` |
