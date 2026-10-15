"""
Top picks score SQL builder.

"""

from api.config import VIEWER_CONFIG


def get_top_picks_score_sql():
    """Build SQL expression for top_picks_score based on config weights."""
    pt = VIEWER_CONFIG['photo_types']
    weights = pt.get('top_picks_weights', {
        'aggregate_percent': 20,
        'aesthetic_percent': 32,
        'composition_percent': 24,
        'face_quality_percent': 24
    })

    min_face_ratio = pt.get('top_picks_min_face_ratio', 0.20)

    agg_w = weights.get('aggregate_percent', 20) / 100.0
    aesthetic_w = weights.get('aesthetic_percent', 32) / 100.0
    comp_w = weights.get('composition_percent', 24) / 100.0
    face_w = weights.get('face_quality_percent', 24) / 100.0

    no_face_aesthetic_w = aesthetic_w + (face_w / 2.0)
    no_face_comp_w = comp_w + (face_w / 2.0)

    return f"""CASE
        WHEN COALESCE(face_ratio, 0) >= {min_face_ratio} THEN
            (COALESCE(aggregate, 0) * {agg_w:.2f} + COALESCE(aesthetic, 0) * {aesthetic_w:.2f} + COALESCE(comp_score, 0) * {comp_w:.2f} + COALESCE(face_quality, 0) * {face_w:.2f})
        ELSE
            (COALESCE(aggregate, 0) * {agg_w:.2f} + COALESCE(aesthetic, 0) * {no_face_aesthetic_w:.2f} + COALESCE(comp_score, 0) * {no_face_comp_w:.2f})
    END"""


def with_top_picks_alias(select_cols, needed):
    """Return ``select_cols`` plus the ``top_picks_score`` SELECT alias when ``needed``.

    ``top_picks_score`` is a weighted expression, not a ``photos`` column, so
    ``ORDER BY top_picks_score`` only resolves when the SELECT list defines it. The
    alias is also what puts the score on each returned photo. ``needed`` is true for
    a top-picks sort or the top-picks filter. Returns a new list; ``select_cols`` is
    left untouched.
    """
    if not needed:
        return select_cols
    return [*select_cols, f"({get_top_picks_score_sql()}) as top_picks_score"]


def get_top_picks_threshold():
    """Get the minimum score threshold for top picks."""
    return VIEWER_CONFIG['photo_types'].get('top_picks_min_score', 7)
