"""
Merge suggestions API router -- face merge group analysis.

"""

import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from api.auth import CurrentUser, require_authenticated
from api.config import VIEWER_CONFIG, is_multi_user_enabled, server_scoring_config
from api.database import get_db
from api.db_helpers import get_visibility_clause
from api.models.albums import MergeSuggestionsResponse

router = APIRouter(tags=["merge_suggestions"])


def _visible_person_ids(user: CurrentUser):
    """Set of person ids the caller may see, or None when no scoping applies.

    Returns None outside multi-user mode (an authenticated user sees the whole
    library); in multi-user mode it restricts to persons with a face in a photo
    within the caller's directories.
    """
    if not is_multi_user_enabled():
        return None
    vis_sql, vis_params = get_visibility_clause(
        user.user_id if user else None, table_alias='p'
    )
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT f.person_id FROM faces f "
            f"JOIN photos p ON p.path = f.photo_path "
            f"WHERE f.person_id IS NOT NULL AND {vis_sql}",
            vis_params,
        ).fetchall()
    return {r[0] for r in rows}


def _load_rejected_pairs():
    """Return the set of (a, b) person pairs (a < b) the user has dismissed.

    Defensive: returns an empty set if the table is absent so suggestions are
    simply unfiltered rather than erroring.
    """
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT person_a_id, person_b_id FROM rejected_merge_suggestions"
            ).fetchall()
        return {(row[0], row[1]) for row in rows}
    except sqlite3.Error:
        return set()


def _pairwise_suggestions(threshold):
    """Merge suggestions as true pairwise person comparisons.

    Loads every non-hidden person's centroid and emits each person pair whose
    OWN centroid cosine similarity clears ``threshold``, labelled with that
    pair's real similarity.

    Replaces a union-find grouping that emitted the *adjacent* pairs of a
    transitively-connected group and stamped every one with the group's average
    similarity -- so a pair merely linked through a chain (A~B~C, with A and C
    themselves dissimilar) was shown as a confident match and an 'Accept all'
    then merged different people irreversibly.
    """
    import numpy as np
    from db.person_centroids import load_person_centroids

    with get_db() as conn:
        loaded = load_person_centroids(conn, include_hidden=False)

    persons = [{k: p[k] for k in ('id', 'name', 'face_count')} for p in loaded]
    centroids = [p['centroid'] for p in loaded]

    suggestions = []
    n = len(persons)
    for i in range(n):
        ci = centroids[i]
        for j in range(i + 1, n):
            similarity = float(np.dot(ci, centroids[j]))
            if similarity >= threshold:
                suggestions.append({
                    "person1": {
                        "id": persons[i]["id"],
                        "name": persons[i]["name"],
                        "face_count": persons[i]["face_count"],
                    },
                    "person2": {
                        "id": persons[j]["id"],
                        "name": persons[j]["name"],
                        "face_count": persons[j]["face_count"],
                    },
                    "similarity": similarity,
                })
    return suggestions


@router.get(
    "/api/merge_suggestions",
    response_model=MergeSuggestionsResponse,
    response_model_exclude_unset=True,
)
def get_merge_suggestions(
    threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    user: CurrentUser = Depends(require_authenticated),
):
    """Return merge suggestions as pairwise person comparisons.

    Without ``threshold`` the active embedding model's ``merge_threshold`` from
    ``face_clustering.thresholds_by_model`` applies.
    """
    # Check if feature is enabled
    if not VIEWER_CONFIG.get("features", {}).get("show_merge_suggestions", True):
        raise HTTPException(status_code=403, detail="Merge suggestions feature is disabled")

    if threshold is None:
        threshold = server_scoring_config().get_face_clustering_settings()['merge_threshold']
    suggestions = _pairwise_suggestions(threshold)

    # Drop pairs the user has already dismissed so they stop reappearing.
    rejected = _load_rejected_pairs()
    if rejected:
        suggestions = [
            s for s in suggestions
            if tuple(sorted((s["person1"]["id"], s["person2"]["id"]))) not in rejected
        ]

    visible = _visible_person_ids(user)
    if visible is not None:
        suggestions = [
            s for s in suggestions
            if s["person1"]["id"] in visible and s["person2"]["id"] in visible
        ]

    return {"suggestions": suggestions, "threshold": threshold}
