# Scoring System

> 🌐 **English** · [Français](fr/SCORING.md) · [Deutsch](de/SCORING.md) · [Italiano](it/SCORING.md) · [Español](es/SCORING.md) · [Português](pt/SCORING.md) · [简体中文](zh/SCORING.md)

Photos are classified into a category, then scored with that category's weights.

## How Scoring Works

1. **Category Detection** - Photo analyzed for content (faces, tags, EXIF data)
2. **Filter Evaluation** - Categories evaluated in priority order until one matches (a *scoring context* can promote/exclude categories per album or photo without changing the base order — see [Scoring Contexts](#scoring-contexts))
3. **Weight Application** - Category-specific weights applied to metrics
4. **Modifier Application** - Bonuses, penalties, and behavior flags applied
5. **Final Score** - Weighted sum clamped to 0-10 range

## Categories

`scoring_config.json` defines 34 categories (33 named plus `default`), evaluated in ascending priority order until one matches. Lower priority wins. The full list lives in the `categories` array; the main ones:

| Priority | Category | Detection Method |
|----------|----------|------------------|
| 8 | `art` | Tags: painting, statue, drawing, cartoon, anime |
| 10 | `astro` | Tags: aurora, astrophotography, stars, milky way, AND mean luminance ≤ 0.30 |
| 15 | `concert` | Tags: concert |
| 35 | `group_portrait` | Face ratio ≥ 5% AND is_group_portrait |
| 42 | `silhouette` | Has face AND `is_silhouette` (the largest face is dark against a brighter surround) |
| 45 | `portrait` | Face ratio ≥ 5%, not silhouette/group/mono |
| 46 | `portrait_bw` | Monochrome portrait (face ≥ 5%) |
| 55 | `macro` | Tags: macro, insect, butterfly, dewdrop, ... |
| 65 | `wildlife` | Tags: animal, bird, marine, reptile, primate |
| 80 | `long_exposure` | Shutter 1-10 seconds |
| 85 | `night` | Luminance < 0.15 |
| 88 | `monochrome` | is_monochrome (saturation < 5%) |
| 95 | `street` | Tags: street, urban_culture |
| 96 | `human_others` | Has face AND face ratio < 5% |
| 100 | `landscape` | Tags: landscape, mountain, beach, forest, ... |
| 999 | `default` | Fallback (no filter) |

Other tag-based categories include `aerial`, `food`, `sports`, `vehicle`, `travel`, `fashion`, `candid`, `product`, `architecture`, `urban`, `golden_hour`, `blue_hour`, `cinematic`, `vintage`, `abstract`, `minimalist`, `dramatic`, and `weather`.

On the CLIP-tagging profiles (`legacy`/`8gb`, and `16gb`/`24gb` when the VLM tagger is not scheduled) a multi-pass scan tags the photo during the embeddings pass, before its category and aggregate are computed, so the scan-time category sees the tags just as `--recompute-average` does; the post-scan tag step is only a fallback for photos still untagged. A single `--pass NAME` run does not tag.

**Silhouette.** `is_silhouette` is a face-aware pixel rule (see the filter table below), not a tag or a histogram test. The stored flag steers the aggregate as well as the category: a silhouette skips the clipping penalty in `aggregate` (`exposure.silhouette_detection`), while a scan stores `exposure_score` with the clipping penalty, but `--recompute-average` re-derives it without that penalty for any photo whose stored `is_silhouette` is set, whatever `exposure.silhouette_detection` says. Re-derive stored flags with `python facet.py --recompute-silhouette`, then `--recompute-average`.

**Astro.** A photo needs an astro tag *and* a mean luminance of at most 0.30 (`luminance_max`): a bright scene that CLIP tags `aurora` or `stars` is not astro, and a photo with no stored luminance never matches. A `scoring_config.json` with its own `categories` array keeps its own `astro` entry and does not get this guard until you add `"luminance_max": 0.30` to `astro.filters`; `python facet.py --doctor` says whether yours is missing it.

## Scoring Contexts

The priority order above is global — every photo evaluates against the same list. A **scoring context** is a named *delta* over that base order: it promotes a short list of categories to the front and excludes others outright, without renumbering anything. `default` (empty `promote`/`excluded`) is the no-op context, so nothing changes for a photo unless a context is explicitly assigned to it.

**Effective order** = `promote` (in the order given) → the global priority order with the promoted and excluded names removed → `default` last. A name listed in both `promote` and `excluded` is dropped entirely — `excluded` wins. `ScoringConfig.resolve_context_order()` (`config/scoring_config.py`) computes and memoizes this once per context name.

Shipped presets — editable from the viewer's **Scoring Context** tab (`PUT /api/config/scoring_contexts/{name}`, edition-gated) or directly in the JSON; see [Scoring Contexts](CONFIGURATION.md#scoring-contexts) for the full field reference:

| Context | Promotes | Excludes |
|---------|----------|----------|
| `default` | — | — |
| `action_stage` | `sports`, `concert`, `candid` | `silhouette` |
| `party_event` | `group_portrait`, `candid`, `food` | — |
| `portrait_session` | `portrait`, `portrait_bw`, `fashion` | — |
| `wildlife` | `wildlife` | — |
| `landscape` | `landscape`, `golden_hour`, `blue_hour` | — |
| `motorsport` | `sports`, `vehicle` | `silhouette` |

Only the *delta* is editable — drag the promoted head into order (or use the move-up/move-down buttons), toggle a category's exclusion — never a full standalone ordering per context: the non-promoted categories always keep the global priority order, so a category added later can never be silently missing from six separate lists. See [Editing a context](CONFIGURATION.md#editing-a-context) for the validation rules.

A context is assigned per album (`PUT /api/albums/{id}/scoring_context`, which materializes it onto every photo that is currently a member — a one-time snapshot, not a live subscription for a smart album, see [Assigning a context](CONFIGURATION.md#scoring-contexts)) or, for a single stubborn photo, applied as a sticky category override (`POST /api/comparison/override_category`). Both levers persist in a `photo_scoring_overrides` side table rather than as columns on `photos` — `save_photo`/`save_photos_batch` write photo rows with `INSERT OR REPLACE`, which would silently wipe a new column on that row at the next rescan. Setting one lever leaves the other untouched, and either can be cleared independently. **Neither takes effect on already-scored photos until a recompute** — `python facet.py --recompute-average`, or `POST /api/scan/recompute` from the viewer (guarded cross-process against a second scan/recompute running at once — see [Changing priorities requires a recompute](CONFIGURATION.md#reordering-the-global-priority)). If `normalization.per_category` is enabled, run the recompute twice — see [Normalization](CONFIGURATION.md#normalization) for why the first pass normalizes against each photo's old category.

### The missing-EXIF trap

Reordering — whether by editing global priority or by promoting via a context — only changes which category is *tried first*. It cannot make a category's filters match a photo they otherwise wouldn't. `config/category_filter.py:122-128` fails a numeric range filter outright whenever the photo's underlying value is missing or unparseable, rather than skipping just that bound — a missing value and an out-of-range value are treated identically, and the category is passed over either way.

Concretely: `sports` (priority 71) carries `shutter_speed_max: 0.02`. A dance frame shot slower than 1/50s, or with no readable EXIF shutter speed at all, fails that filter no matter where `sports` sits in the evaluation order — even promoted to the very front by a context like `action_stage`. The photo falls through to whatever matches next, typically `fashion` (priority 43, tagged `fashion`, has a face) or `silhouette` (priority 42, backlit with a face). **This is the single most useful thing to check when a photo lands in an unexpected category:** before reordering or promoting anything, confirm the target category's numeric filters can actually match the photo's stored EXIF, not just its tags.

### The missing-tag trap

The EXIF trap above assumes the tag layer already matched — `required_tags` has its own version of the same failure: a filter can't match a tag that doesn't exist in the CLIP vocabulary at all. Before commit `917dd94`, nothing in `scoring_config.json` covered dance: a frame the tagger described as dance, performance, or stage matched none of `sports`'s `required_tags` (`sports`, `motion`, `athlete`, `competition`, `action_sport`) and fell straight through to the `default` catch-all — promoting `sports` via `action_stage` changed nothing, because promotion only reorders *when* a category is tried, never whether its filters match.

`sports.tags.dance` now carries 11 CLIP prompts (`dance performance`, `dancer on stage`, `ballet dancer`, `contemporary dance`, `ballroom dancing`, `latin dance`, `hip hop dance`, `dance competition`, `dance troupe performing`, `dancer mid leap`, `dancer in motion`), and `dance` was added to `sports.filters.required_tags`, so a tagged dance photo can now clear that filter. This makes `sports` *reachable* for dance content — it doesn't waive the shutter-speed trap above: a slow-shutter or EXIF-less dance frame can now pass the tag check and still fail `shutter_speed_max: 0.02`, landing in `fashion` or `silhouette` exactly as described there.

**Existing photos keep whatever tags they already have** — the new vocabulary only changes what the tagger assigns from here on. Re-tag the library with `python facet.py --recompute-tags` to apply it retroactively.

### Reordering global priority

`GET/POST /api/config/category_priorities` (edition-gated) reads and rewrites the base order that every context deltas against. `POST` takes `{"order": [name, ...]}` — a set-equal permutation of every non-`default` category name — and **permutes the existing priority values onto the new order** rather than renumbering (10/20/30/…): the priority multiset is unchanged, so the numbers in the table above stay meaningful and uniqueness holds by construction. `default` (priority 999) is pinned last and excluded from reordering. Every write takes a timestamped `.backup.<timestamp>` copy of `scoring_config.json` first; this writer and the weight editor (`update_category_weights`) now share one lock, since the previous unguarded read-modify-write let a concurrent save from each silently drop the other's changes.

Reordering does not touch any photo's stored `category` by itself — run a recompute afterward (`--recompute-average` or `POST /api/scan/recompute`) to apply it.

**Known limitation:** `api/types.py` builds the gallery's type/filter dropdown list from `ScoringConfig.get_categories()` once at import time. A priority reorder is picked up immediately for actual category matching (every scoring and recompute call re-reads the config from disk), but the gallery's type dropdown keeps its old ordering until the viewer process restarts. Filtering itself is unaffected — reordering adds or removes no category names.

## Category Definition

Each category in `scoring_config.json` has these components:

```json
{
  "name": "portrait",
  "priority": 45,
  "filters": {
    "face_ratio_min": 0.05,
    "has_face": true,
    "is_silhouette": false,
    "is_group_portrait": false,
    "is_monochrome": false
  },
  "weights": {
    "aesthetic_percent": 32,
    "eye_sharpness_percent": 16,
    "face_quality_percent": 14,
    "composition_percent": 12,
    "liqe_percent": 8,
    "exposure_percent": 4,
    "tech_sharpness_percent": 4,
    "color_percent": 4,
    "contrast_percent": 4,
    "aesthetic_iaa_percent": 2
  },
  "modifiers": {
    "bonus": 0.419,
    "_apply_blink_penalty": true,
    "noise_tolerance_multiplier": 0.006,
    "_clipping_multiplier": 0.5
  },
  "tags": {}
}
```

## Filters Reference

### Numeric Range Filters

| Filter | Field | Description |
|--------|-------|-------------|
| `face_ratio_min` / `face_ratio_max` | `face_ratio` | Face area as fraction (0.0-1.0) |
| `face_count_min` / `face_count_max` | `face_count` | Number of faces |
| `iso_min` / `iso_max` | `ISO` | Camera ISO |
| `shutter_speed_min` / `shutter_speed_max` | `shutter_speed` | Exposure time (seconds) |
| `luminance_min` / `luminance_max` | `mean_luminance` | Brightness (0.0-1.0) |
| `focal_length_min` / `focal_length_max` | `focal_length` | Focal length (mm) |
| `f_stop_min` / `f_stop_max` | `f_stop` | Aperture f-number |

### Boolean Filters

| Filter | Description |
|--------|-------------|
| `has_face` | At least one face detected |
| `is_monochrome` | Saturation < 5% |
| `is_silhouette` | The largest face is dark and its surround is bright: the mean luma of the central 60% of the face box is below 0.12 and the band around the box (0.2 to 0.75 box-widths out, clipped to the frame) is brighter by more than 0.20. A face-aware pixel measurement; the CLIP `silhouette` tag and the whole-frame histogram no longer set it. A photo without a face is never a silhouette |
| `is_group_portrait` | face_count >= `min_faces_for_group` (configurable, default: 4) |

### Tag Filters

| Filter | Description |
|--------|-------------|
| `required_tags` | List of tags photo must have |
| `excluded_tags` | List of tags photo must NOT have |
| `tag_match_mode` | `"any"` (default) or `"all"` |

## Weight Keys

All weights use the `_percent` suffix. They are normalized by `get_weights()`, so totals need not equal exactly 100 — but keeping them at 100 keeps scores on the 0-10 scale.

| Key | Metric | Source | Best For |
|-----|--------|--------|----------|
| `aesthetic_percent` | Visual appeal | TOPIQ or CLIP+MLP | All |
| `quality_percent` | Legacy quality | Added to `aesthetic_percent` before scoring (no separate signal) | — |
| `face_quality_percent` | Face clarity | InsightFace | Portraits |
| `eye_sharpness_percent` | Eye sharpness | InsightFace landmarks | Portraits |
| `tech_sharpness_percent` | Overall sharpness | Laplacian variance | Landscapes |
| `composition_percent` | Composition | SAMP-Net or rule-based | All |
| `exposure_percent` | Exposure balance | Histogram analysis | All |
| `color_percent` | Color harmony | HSV analysis | Color photos |
| `contrast_percent` | Tonal contrast | Histogram spread | B&W |
| `dynamic_range_percent` | Tonal range | Histogram analysis | HDR, landscapes |
| `isolation_percent` | Subject separation | Face vs background | Portraits, wildlife |
| `leading_lines_percent` | Leading lines | Edge detection | Architecture |
| `power_point_percent` | Rule-of-thirds | Subject placement | All |
| `saturation_percent` | Color saturation | HSV analysis | Vibrant photos |
| `noise_percent` | Noise level | Noise estimation | Low-light |
| `face_sharpness_percent` | Face region sharpness | Face analysis | Portraits |
| `aesthetic_iaa_percent` | Artistic aesthetic merit | TOPIQ IAA (AVA-trained) | Art, creative |
| `face_quality_iqa_percent` | Face quality (IQA) | TOPIQ NR-Face | Portraits |
| `liqe_percent` | LIQE quality score | LIQE | Diagnostics |
| `subject_sharpness_percent` | Subject region sharpness | BiRefNet + Laplacian | Portraits, wildlife |
| `subject_prominence_percent` | Subject area ratio | BiRefNet | Macro, wildlife |
| `subject_placement_percent` | Subject rule-of-thirds | BiRefNet | All |
| `bg_separation_percent` | Background separation | BiRefNet | Portraits, macro |

**`quality_percent` counts toward `aesthetic`.** The quality metric has no score of its own, so a category's `quality_percent` is added to its `aesthetic_percent` before scoring: the effective aesthetic weight is `aesthetic_percent + quality_percent`, and the weights still sum to 100. Before this was fixed, the `quality` share scored a metric that was always 0 and so dropped out of the aggregate; a library aggregated by older code rises slightly (and changes no category) on its next `--recompute-average`.

## Modifiers

Adjust scoring behavior per category:

| Modifier | Type | Description |
|----------|------|-------------|
| `bonus` | float | Added to final score (e.g., 0.5) |
| `noise_tolerance_multiplier` | float | Scale noise penalty (0.5 = half) |
| `_skip_clipping_penalty` | bool | Skip exposure clipping penalty |
| `_skip_oversaturation_penalty` | bool | Skip oversaturation penalty |
| `_clipping_multiplier` | float | Scale clipping penalty |
| `_apply_blink_penalty` | bool | Apply blink detection penalty |

## Subject Saliency Dimensions

Four dimensions derived from BiRefNet subject segmentation:

| Weight Key | Metric | Description |
|-----------|--------|-------------|
| `subject_sharpness_percent` | Subject sharpness | Focus quality of the subject region vs the background. High = sharp subject, soft background. |
| `subject_prominence_percent` | Subject prominence | Subject area as a fraction of the frame. High for macro and tightly-framed subjects, low for wide scenes. |
| `subject_placement_percent` | Subject placement | Rule-of-thirds score for the subject's center of mass. |
| `bg_separation_percent` | Background separation | Edge gradient difference at the subject boundary (bokeh quality). |

Use `subject_sharpness_percent` and `bg_separation_percent` for portrait/wildlife; `subject_prominence_percent` for macro.

**No subject, no number.** When no subject is found, `subject_sharpness`, `subject_placement` and `bg_separation` are stored as NULL (as is `power_point_score` when there is no subject box), `subject_prominence` is 0.0 and `subject_bbox` is NULL; earlier versions stored a neutral 5.0. The gallery's min/max range filters on those metrics skip a NULL (`min_subject_sharpness=0` no longer returns a photo without a subject), SQL averages ignore it and the UI shows it as absent. The aggregate still scores a NULL as the neutral 5.0, so no aggregate moves. Existing rows keep their 5.0 until `--recompute-saliency` and `--recompute-composition-cpu` rewrite them.

**When saliency fails.** On CUDA the saliency batch size is derived from the free VRAM (about 2.4 GiB per image at 1024 px, between 1 and 8 images) and is halved after an out-of-memory error, down to one image. A photo whose saliency still fails logs an ERROR and stores nothing: a photo that was never scored keeps NULL in all five saliency columns (`subject_sharpness`, `subject_prominence`, `subject_placement`, `bg_separation`, `subject_bbox`), and `--recompute-saliency` and `--pass saliency` keep a photo's stored values when its saliency fails. A full rescan of the photo (for example `--force`) stores NULL for it, as for a never-scored photo. Facet never writes a neutral number in place of a failed result. In the aggregate a NULL scores neutral (5.0), exactly like a photo scanned before saliency existed, so it neither helps nor hurts the score. `--recompute-saliency` selects such a photo again (see [Subject Saliency](COMMANDS.md#subject-saliency)).

## Supplementary IQA Dimensions

Three additional quality models:

| Weight Key | Model | Description |
|-----------|-------|-------------|
| `aesthetic_iaa_percent` | TOPIQ IAA | AVA-trained aesthetic merit, distinct from the technical-quality aesthetic score. Best for art/creative categories. |
| `face_quality_iqa_percent` | TOPIQ NR-Face | Face-region quality assessment. Best for portrait categories. |
| `liqe_percent` | LIQE | Quality score plus a distortion diagnosis (motion blur, overexposure, noise). |

These models run as part of the default scoring pipeline on all GPU profiles (8gb/16gb/24gb) and share VRAM with TOPIQ; the CPU legacy profile skips them. Add their weight keys to any category where the assessment is useful.

**When a supplementary model cannot score a photo.** TOPIQ NR-Face needs a face its own detector finds, and LIQE needs a photo whose short side is at least 224 px; Aesthetic V2.5, when enabled, can fail on any image. A photo a model cannot score stores NULL in that model's column (`face_quality_iqa`, `liqe_score`, `aesthetic_iaa`, `aesthetic_v25`) instead of a neutral 5.0. A GPU out-of-memory error is retried at half the batch size, down to one image, before the photo gives up; a CPU memory error is not retried that way, it takes the ordinary per-image path, and a photo that still fails stores NULL. An unexpected failure is logged as an ERROR with the number of photos affected, while the two expected cases (no face found, short side under 224 px) are only totalled in the log when the model is released. `--recompute-iqa` does not retry either expected case on every run: the short-side case is decided from the stored thumbnail's size, and an NR-Face decline is remembered (in the `iqa_declined` table) until the photo is rescanned, whereas an out-of-memory or unexpected failure is retried by the next run. In the aggregate a NULL scores neutral (5.0), so aggregate scores, categories and rankings do not move. **A gallery range filter on Aesthetic (IAA), Face Quality (IQA), LIQE or Aesthetic V2.5 no longer matches a photo without a score**: a faceless photo used to carry a stored 5.0 and matched `min_face_quality_iqa=0`, and now it does not; an ascending sort on Aesthetic (IAA), Face Quality (IQA) or LIQE puts such photos last, in the gallery and in albums (smart and shared included). Values stored by earlier versions stay as they are until `--repair-neutral-iqa` clears the neutral 5.0 in `aesthetic_iaa`, `face_quality_iqa` and `liqe_score` (then run `--recompute-iqa` and `--recompute-average`). A rescan that rewrites the photo (for example `--force`) stores NULL for a score the model could not produce; `--recompute-iqa` only fills columns that are NULL, and `--pass quality-iaa`, `--pass quality-face` and `--pass quality-liqe` keep the stored value when a model fails, so none of them replaces an old 5.0. If the primary quality model (TOPIQ on the `16gb` and `24gb` profiles) fails for a photo, the photo is recorded as failed in `scan_failures`, instead of being saved with a neutral 5.0 aesthetic score. The next scan picks up a photo that is not yet in the database; a photo that is already stored (for example after a `--force` rescan) keeps its row, and is retried by `--retry-failed all` (a bare `--retry-failed` covers only the latest run that recorded failures) or by another `--force` rescan. Under `--pass quality` the stored value is kept and no failure row is recorded. See [Supplementary Quality Models](COMMANDS.md#supplementary-quality-models).

### Supplementary signals (not in default aggregate)

| Column | Source | Description |
|--------|--------|-------------|
| `aesthetic_clip` | `analyzers/aesthetic_clip.py` + cached CLIP/SigLIP embedding | A free supplementary aesthetic score (0-10) derived from cached image embeddings by projecting onto an "aesthetic axis" built from positive/negative text prompts. Zero extra image inference at scan time. **Not** part of the default `aggregate`. Populate with `python scripts/compute_aesthetic_clip.py --db <path>`. Benchmark with `python scripts/benchmark_aesthetic.py --db <path> --ava AVA.txt --photo-dir <dir>`. AVA SRCC ≈ 0.52 on the 500-photo `ava_test/` set (vs 0.94 for `aesthetic_iaa`) — useful as a cheap pre-filter or when TOPIQ-IAA is unavailable. |

## Category Tags (CLIP Vocabulary)

Tags trigger tag-based categories and are matched using CLIP similarity:

```json
{
  "tags": {
    "landscape": ["landscape", "scenic view", "nature scene"],
    "mountain": ["mountain", "alpine", "peaks"],
    "beach": ["beach", "ocean", "seaside", "coastal"]
  }
}
```

Each key is the canonical tag name, and the array contains synonyms for CLIP matching.

## Top Picks Scoring

The viewer's "Top Picks" filter uses a custom weighted score:

```json
"top_picks_weights": {
  "aggregate_percent": 30,
  "aesthetic_percent": 28,
  "composition_percent": 18,
  "face_quality_percent": 24
}
```

**Score computation:**
- With face (face_ratio ≥ 20%): All four metrics contribute
- Without face: `face_quality_percent` redistributed evenly (half each) to `aesthetic` and `composition` (with default weights: aesthetic 0.40, composition 0.30)

## VRAM Profile Considerations

Default weights are optimized for **TOPIQ** (0.93 SRCC), the aesthetic model for all profiles.

| Profile | Aesthetic Model | Embeddings | Tagger | Recommendations |
|---------|-----------------|-----------|--------|-----------------|
| `24gb` | TOPIQ (0.93 SRCC) | SigLIP 2 NaFlex SO400M | Qwen3.5-4B | Best accuracy, default weights |
| `16gb` | TOPIQ (0.93 SRCC) | SigLIP 2 NaFlex SO400M | Qwen3.5-2B | Default weights |
| `8gb` | CLIP+MLP (0.76 SRCC) | CLIP ViT-L-14 | CLIP similarity | Default weights work well |
| `legacy` | CLIP+MLP on CPU | CLIP ViT-L-14 | CLIP similarity | Default weights, slower |

All GPU profiles (8gb/16gb/24gb) additionally run supplementary PyIQA models (TOPIQ IAA, TOPIQ NR-Face, LIQE) and optionally BiRefNet_dynamic for subject saliency; the CPU legacy profile skips them.

Run `--compute-recommendations` after switching profiles to analyze score distributions.

## Weight Tuning Workflow

### Option A: Via Viewer (Recommended)

1. Open `/stats` → **Categories** tab → **Weights** sub-tab
2. Unlock edition mode
3. Select a category from the editor dropdown
4. Adjust sliders — the live **Score Distribution Preview** shows estimated impact
5. Click **Save** then **Recompute Scores** to apply

The viewer runs `--recompute-category` under the hood, updating only photos in that category.

### Option B: Via CLI

#### 1. Analyze Current Scores

```bash
python facet.py --compute-recommendations
```

Shows:
- Score distributions per category
- Weight correlation analysis
- Suggested adjustments

#### 2. Adjust Weights

Edit `scoring_config.json` category weights. Ensure they sum to 100.

#### 3. Recompute Scores

```bash
python facet.py --recompute-average               # All categories
python facet.py --recompute-category portrait      # Single category (faster)
```

Uses stored embeddings - no GPU needed.

#### 4. Validate Changes

```bash
python facet.py --compute-recommendations
```

Compare distributions before/after.

## Pairwise Comparison Mode

Train weights by comparing photo pairs:

### Setup

1. Set a non-empty `edition_password` in config: `"viewer": { "edition_password": "your-password" }`
2. Start viewer: `python viewer.py`
3. Click "Compare" button

### Comparison Interface

- Side-by-side photos
- Keyboard: ← (left wins), → (right wins), T (tie), S (skip). The on-screen buttons are still labelled **A** / **B** (the values submitted), but the keys are ArrowLeft/ArrowRight.
- Progress bar shows comparisons toward 50 minimum

### Comparison Sources

Comparisons carry a `source` marker so the optimizer can weight them by reliability:

- `vote` — explicit A/B votes from the comparison interface
- `culling` — derived automatically from burst/similar culling decisions: each
  rejected photo is paired against up to two kept photos from the same group
  (capped at 12 pairs per group). Kept photos win. Explicit votes on the same
  pair are never overwritten.
- `rating` — synthetic pairs generated from star ratings and favorites

Reviewing burst groups in the viewer therefore grows the training set for
weight optimization without any extra effort.

### Weight Optimization

```bash
# Check comparison stats
python facet.py --comparison-stats

# Optimize weights from comparisons (applied only if it generalizes)
python facet.py --optimize-weights --optimize-category portrait

# Restrict training data to specific sources
python facet.py --optimize-weights --optimize-category portrait --optimize-sources vote,culling

# Apply even if the held-out gate is not met
python facet.py --optimize-weights --optimize-category portrait --optimize-force

# Apply to all photos
python facet.py --recompute-average
```

### Label-to-Weight Pipeline

Beyond explicit A/B votes, two more label streams feed the optimizer:

1. **Culling decisions** are captured automatically on every burst/similar
   confirm (`source='culling'`).
2. **Star ratings, favorites and rejections** are materialized into synthetic
   pairs with `python facet.py --sync-label-comparisons` (`source='rating'`).
   Re-running re-syncs from the current labels, so retracted ratings disappear.

The optimizer weighs each source by reliability (vote 1.0, rating 0.7, culling 0.5) when maximizing the Bradley-Terry likelihood. A **tie** counts as half a win plus half a loss: evidence that two photos score alike, at the same strength as a decisive vote (`include_ties` still drops them). The fit is **anchored to the current weights** by an L2 prior (`l2_regularization`, strength 100), so with few comparisons the suggestion stays close to the weights you have and moves further as votes accumulate. It trains on each photo as the scorer serves it: the same 0-10 metric vector the aggregate reads (including `liqe`, `aesthetic_iaa`, `face_quality_iqa` and the subject-saliency metrics), with the percentile-normalised eye sharpness, technical sharpness and colour, the rebuilt exposure score and the group-portrait flag. The percentiles come from the snapshot `--recompute-average` persisted; before the first `--recompute-average` the stored columns are read instead. It starts from, and writes, the **served** weights: `quality_percent` is folded into `aesthetic` the way the scorer folds it, and applying writes `quality_percent: 0` for the categories that carry it.

Weights are **applied only if they generalize**. The final weights are fit on all comparisons, but the decision to write them is a held-out gate: 5-fold cross-validation with **photo-grouped folds** (no photo is in both the training pairs and the held-out pairs of a fold), pooled over the folds and compared with the current weights on the same held-out pairs by a **one-sided exact McNemar test**. They are applied only when `p < 0.05` **and** the held-out gain is at least 2 pp (`min_improvement`); otherwise the run logs the numbers and writes nothing — pass `--optimize-force` to override. If the gate cannot run (too few independent photo groups) the CLI does not fall back to training accuracy: it applies nothing unless forced. Optimization is per-category and needs **decisive** labelled comparisons (A or B won) **for that category**: `min_comparisons` counts those, ties and other categories do not satisfy it; categories with no votes cannot be tuned from data. The log shows, for example:

```
Held-out accuracy:      69.2% +/- 4.0 (5-fold, photo-grouped) - the honest estimate
Held-out gain:          +8.5 pp over the current weights on the same 389 pairs
McNemar (one-sided):    p = 0.0006 (only the fit right: 66, only the current weights right: 33)
NOT applying: the held-out gate needs p < 0.05 and a gain of at least 2.0 pp. ...   (logged when the gate fails)
```

Applying a result with `--optimize-weights` changes how the category's photos rank among themselves, not where the category sits against the others: in the same config write as the weights, `modifiers.bonus` is moved by minus the projected mean score shift of the category's photos (clamped to [-5, 5]; a clamp is logged as a residual), so the category's mean aggregate stays put. A category with no `categories[]` entry (`others`) has no bonus and is not re-centred. The weight snapshot recorded before the write carries the bonus, and restoring a snapshot restores weights **and** bonus (snapshots recorded before this change, by `calibrate.py` or for `others` carry none and leave the bonus untouched). Documented residual: the viewer writes weights through `POST /api/config/update_weights`, for its **Apply** button as for a hand edit in the Weights tab, and that route does not re-centre the bonus (only `--optimize-weights` does).

**After upgrading, retrain the personal ranker and the keeper heads, then run `--recompute-average`.** The optimizer, the ranker and the keeper heads share one feature vector, which now is the served one (normalised eye, technical and colour metrics); a saved model cannot detect the change.

Recommended cadence:

```bash
python facet.py --mine-insights          # what signal exists, drift, health
python facet.py --sync-label-comparisons # refresh rating-derived pairs
python facet.py --optimize-weights       # learn weights from all sources
python facet.py --recompute-average      # apply + persist percentile snapshot
```

### In-UI Weight Tuning

During comparison, the Weight Preview panel lets you adjust sliders for
real-time score changes and click "Suggest Weights" for optimized values.
This is the same in-viewer slider workflow described in
[Option A: Via Viewer](#option-a-via-viewer-recommended) above — see there
for the full save/recompute flow.

**Suggest Weights** also answers a narrower question than the held-out gate above: how well do this category's *current, live* weights already agree with your own comparisons? Clicking it returns `accuracy_before` — the percentage of that category's labelled pairs (A/B votes, culling, and rating-derived pairs) whose winner the live weights predict correctly — next to `accuracy_after`, the same figure for the suggested weights; both are in-sample figures. They are shown side by side in the Weight Suggestions tab and in the A/B Compare tab's sidebar every time you run it (`GET /api/comparison/learned_weights`, `optimization/weight_optimizer.py:optimize_weights_direct`). Whether **Apply** is offered follows the same held-out McNemar gate as the CLI, not the in-sample gain: `suggest_changes` is true only when the gate passes (the request runs the cross-validation only when the in-sample gain already reaches 2 pp; `?use_cv=true` runs it unconditionally). Like the CLI, this needs `min_comparisons_for_optimization` (shipped default 50) decisive A/B comparisons for that category — below that the button reports the shortfall ("Need at least N decisive comparisons (have K)") instead of a number.

## Adding Custom Categories

```json
{
  "name": "underwater",
  "priority": 62,
  "filters": {
    "required_tags": ["underwater"],
    "tag_match_mode": "any"
  },
  "weights": {
    "aesthetic_percent": 40,
    "color_percent": 25,
    "composition_percent": 20,
    "exposure_percent": 15
  },
  "modifiers": {
    "noise_tolerance_multiplier": 0.3,
    "bonus": 0.5
  },
  "tags": {
    "underwater": ["underwater", "scuba", "diving", "ocean"],
    "fish": ["fish", "coral", "reef"]
  }
}
```

Add to the `categories` array in `scoring_config.json`, then run `--recompute-average` (or `--recompute-category underwater` for just the new category).

## Workflow Examples

### Tune Concert Category

```bash
# Edit scoring_config.json:
# Find "concert" category, adjust:
#   "noise_tolerance_multiplier": 0.05
#   "exposure_percent": 5

python facet.py --recompute-category concert
```

Or use the viewer's weight editor at `/stats` → Categories → Weights for live preview and one-click recompute.

### Switch to 8gb Profile

```bash
# Edit: "vram_profile": "8gb"
python facet.py --compute-recommendations  # Analyze
# Reduce aesthetic_percent in categories if needed
python facet.py --recompute-average
```

### Add Underwater Category

1. Add category definition (see above)
2. Run `python facet.py --validate-categories`
3. Run `python facet.py --recompute-average`
