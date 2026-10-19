"""
Sort/type/filter definitions for the API server.

"""

import logging
import sqlite3
import time

from api.config import VIEWER_CONFIG, _photo_types_cache, _photo_types_lock, server_scoring_config
from api.top_picks import get_top_picks_score_sql
from api.db_helpers import build_hide_clauses
from api.exposure import exposure_seconds_sql

logger = logging.getLogger(__name__)

# Junk-sweep sentinels shared by the gallery filter and filter_options.
# NOTE: the value must match models.junk_classifier.NOT_JUNK; kept as a plain
# literal here so the API never imports the torch-heavy classifier module.
JUNK_NOT_JUNK = 'not_junk'   # junk_kind sentinel: photo evaluated and found clean
JUNK_ANY = 'any'             # gallery junk_kind filter value: match any junk kind

# Gallery `sequence_override` filter values. A correction is pending until the
# next detection run applies it, so listing them is how a user finds what is
# still waiting on that run.
SEQUENCE_OVERRIDE_ANY = 'any'                # any pending correction
SEQUENCE_OVERRIDE_SUPPRESSED = 'suppressed'  # "this is not a panorama"
SEQUENCE_OVERRIDE_FORCED = 'forced'          # "these frames are one"
SEQUENCE_OVERRIDE_VALUES = (
    SEQUENCE_OVERRIDE_ANY, SEQUENCE_OVERRIDE_SUPPRESSED, SEQUENCE_OVERRIDE_FORCED,
)


# --- SORT OPTIONS (loaded from config) ---
def _build_sort_options():
    """Build sort options from config - supports both flat and grouped formats."""
    sort_opts = VIEWER_CONFIG.get('sort_options', {})

    if isinstance(sort_opts, dict):
        flat = []
        for category, options in sort_opts.items():
            for opt in options:
                flat.append((opt['column'], opt['label']))
        return flat, sort_opts

    flat = [(opt['column'], opt['label']) for opt in sort_opts]
    return flat, None


SORT_OPTIONS, SORT_OPTIONS_GROUPED = _build_sort_options()

# licensing.unavailable_metrics() speaks in weight keys; the viewer sorts and
# filters by photos columns. The two differ only for LIQE ('liqe' -> 'liqe_score').
_METRIC_COLUMNS = {
    'liqe': 'liqe_score',
    'aesthetic_iaa': 'aesthetic_iaa',
    'face_quality_iqa': 'face_quality_iqa',
}


def unavailable_columns():
    """Column-level names of metrics the active licensing policy cannot compute.

    Evaluated per call (licensing mode is fixed at startup, but this keeps the
    import-time sort tables free of policy state). Empty in default mode.
    """
    import licensing
    return sorted(_METRIC_COLUMNS.get(m, m) for m in licensing.unavailable_metrics())


def available_sort_options():
    """``(SORT_OPTIONS, SORT_OPTIONS_GROUPED)`` minus unavailable-metric sorts."""
    hidden = set(unavailable_columns())
    if not hidden:
        return SORT_OPTIONS, SORT_OPTIONS_GROUPED
    flat = [opt for opt in SORT_OPTIONS if opt[0] not in hidden]
    grouped = None
    if SORT_OPTIONS_GROUPED is not None:
        grouped = {
            cat: [o for o in opts if o['column'] not in hidden]
            for cat, opts in SORT_OPTIONS_GROUPED.items()
        }
        grouped = {cat: opts for cat, opts in grouped.items() if opts}
    return flat, grouped
# 'top_picks_score' and 'learned_score' are computed/joined sort columns, not
# stored photos columns: top_picks_score is an on-the-fly weighted expression and
# learned_score comes from the personal-ranker learned_scores table (opt-in
# alternate sort; NULL when the ranker hasn't been trained).
VALID_SORT_COLS = [opt[0] for opt in SORT_OPTIONS] + ['top_picks_score', 'learned_score']

# Sort columns that can be NULL: learned_score / narrative_moment_confidence when
# untrained / unlabelled, and the supplementary IQA scores, where a photo the model
# could not score stores NULL, never 5.0. ``shutter_speed`` is not listed: it sorts
# through its own seconds expression (see ``order_by_clause``) and always sinks its
# missing and invalid values.
NULLABLE_SORT_COLS = ('learned_score', 'narrative_moment_confidence', 'aesthetic_iaa',
                      'liqe_score', 'face_quality_iqa', 'aesthetic_v25')


def order_by_clause(sort_col: str, sort_dir: str, tiebreak: str = '') -> str:
    """Build an ORDER BY body for ``sort_col`` / ``sort_dir`` that sinks NULLs.

    ``sort_col`` and ``sort_dir`` ('ASC' or 'DESC') must already be validated by the
    caller; ``tiebreak`` is appended verbatim (e.g. ``'path ASC'``).

    In DESC, SQLite already orders NULLs last, so the ``(col IS NULL)`` guard is
    omitted -- that lets a ``(... col DESC, path)`` index satisfy the sort with no
    temp B-tree (the default, common direction). In ASC the explicit guard stays,
    since NULLs would otherwise sort first. A column outside ``NULLABLE_SORT_COLS``
    (including a computed expression such as the top-picks score) is never guarded.
    ``shutter_speed`` is the exception: it sorts on ``exposure_seconds_sql()`` (text
    projected to positive seconds, NULL when missing or invalid), evaluated once per
    row, and sinks NULLs in both directions (``NULLS LAST`` in ASC, which needs
    SQLite 3.30+; the project floor is 3.31.1). Stored EXIF text is never changed.
    """
    tail = f", {tiebreak}" if tiebreak else ""
    if sort_col == 'shutter_speed':
        nulls = ' NULLS LAST' if sort_dir == 'ASC' else ''
        return f"{exposure_seconds_sql()} {sort_dir}{nulls}{tail}"
    if sort_col in NULLABLE_SORT_COLS and sort_dir == 'ASC':
        return f"({sort_col} IS NULL) ASC, {sort_col} ASC{tail}"
    return f"{sort_col} {sort_dir}{tail}"


# --- SEMANTIC FILTER MAPPINGS ---
def _build_quality_levels():
    """Build quality levels from config."""
    qt = VIEWER_CONFIG['quality_thresholds']
    return [
        ('', 'All'),
        ('good', f"Good ({qt['good']}+)"),
        ('great', f"Great ({qt['great']}+)"),
        ('excellent', f"Excellent ({qt['excellent']}+)"),
        ('best', f"Best ({qt['best']}+)"),
    ]


QUALITY_LEVELS = _build_quality_levels()

# Build type definitions and filters from scoring_config.json categories
_scoring_config = server_scoring_config()
_config_categories = _scoring_config.get_categories()


def _build_type_definitions(categories):
    """Build type definitions from config categories."""
    pt = VIEWER_CONFIG['photo_types']
    threshold = pt.get('top_picks_min_score', 7)
    top_picks_expr = get_top_picks_score_sql()

    types = [
        ('top_picks', 'Top Picks', f"({top_picks_expr}) >= {threshold}", []),
    ]

    # Auto-include all categories from config (label resolved via i18n on frontend)
    for cat in categories:
        cat_name = cat.get('name', '')
        if cat_name and cat_name != 'default':
            types.append((cat_name, cat_name, "category = ?", [cat_name]))

    return types


TYPE_DEFINITIONS = _build_type_definitions(_config_categories)


def _build_type_filters(categories):
    """Build type filters from config categories."""
    filters = {
        'top_picks': {'top_picks_filter': '1'},
    }
    for cat in categories:
        cat_name = cat.get('name', '')
        if cat_name:
            filters[cat_name] = {'category': cat_name}
    return filters


TYPE_FILTERS = _build_type_filters(_config_categories)
del _scoring_config, _config_categories

TYPE_LABELS = {type_id: label for type_id, label, *_ in TYPE_DEFINITIONS}
QUALITY_MAP = VIEWER_CONFIG['quality_thresholds']


def get_photo_types(hide_blinks=False, hide_bursts=False, hide_duplicates=False,
                    hide_brackets=False, hide_panoramas=False, user_id=None):
    """Build type list dynamically from database, showing only non-empty categories with counts."""
    from api.db_helpers import get_db_connection, get_existing_columns, get_visibility_clause

    cache_key = (hide_blinks, hide_bursts, hide_duplicates,
                 hide_brackets, hide_panoramas, user_id or '')
    with _photo_types_lock:
        if time.time() < _photo_types_cache['expires'] and cache_key in _photo_types_cache['data']:
            return _photo_types_cache['data'][cache_key]

    conn = get_db_connection()
    try:
        existing_cols = get_existing_columns(conn)

        base_filters = build_hide_clauses(
            '1' if hide_blinks else '0',
            '1' if hide_bursts else '0',
            '1' if hide_duplicates else '0',
            '1' if hide_brackets else '0',
            '1' if hide_panoramas else '0',
        )
        sql_params = []

        if user_id:
            vis_sql, vis_params = get_visibility_clause(user_id)
            base_filters.append(vis_sql)
            sql_params.extend(vis_params)

        base_where = " AND ".join(base_filters) if base_filters else ""

        valid_types = []
        for type_id, label, where_clause, type_params in TYPE_DEFINITIONS:
            if 'is_monochrome' in where_clause and 'is_monochrome' not in existing_cols:
                continue
            if 'mean_luminance' in where_clause and 'mean_luminance' not in existing_cols:
                continue
            if 'is_silhouette' in where_clause and 'is_silhouette' not in existing_cols:
                where_clause = "tags LIKE '%silhouette%'"
            if 'tags' in where_clause and 'tags' not in existing_cols:
                continue

            if base_where:
                combined_where = f"({where_clause}) AND {base_where}"
            else:
                combined_where = where_clause

            valid_types.append((type_id, label, combined_where, type_params))

        all_params = []
        query_parts = []
        if base_where:
            query_parts.append(f"SELECT '' as type_id, COUNT(*) as cnt FROM photos WHERE {base_where}")
            all_params.extend(sql_params)
        else:
            query_parts.append("SELECT '' as type_id, COUNT(*) as cnt FROM photos")

        for type_id, label, combined_where, type_params in valid_types:
            query_parts.append(f"SELECT ? as type_id, COUNT(*) as cnt FROM photos WHERE {combined_where}")
            all_params.append(type_id)
            all_params.extend(type_params)
            all_params.extend(sql_params)

        union_query = " UNION ALL ".join(query_parts)
        results = conn.execute(union_query, all_params).fetchall()
    except sqlite3.Error:
        logger.exception("Failed to compute photo type counts")
        return []
    finally:
        conn.close()

    types = []
    type_label_map = {type_id: label for type_id, label, *_ in TYPE_DEFINITIONS}
    type_label_map[''] = 'All Photos'

    for row in results:
        type_id, count = row[0], row[1]
        if count > 0:
            label = type_label_map.get(type_id, type_id)
            types.append({'id': type_id, 'label': label, 'count': count})

    with _photo_types_lock:
        _photo_types_cache['data'][cache_key] = types
        _photo_types_cache['expires'] = time.time() + VIEWER_CONFIG['cache_ttl_seconds']

    return types


def normalize_params(params):
    """Translate semantic params to legacy format while preserving originals."""
    result = dict(params)

    quality = params.get('quality', '')
    if quality and quality in QUALITY_MAP and not params.get('min_score'):
        result['min_score'] = str(QUALITY_MAP[quality])

    photo_type = params.get('type', '')
    if photo_type in TYPE_FILTERS:
        for key, value in TYPE_FILTERS[photo_type].items():
            if not params.get(key):
                result[key] = value

    return result
