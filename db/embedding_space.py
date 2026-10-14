"""Face-embedding spaces: which recognition model is active and what it emits.

Embeddings from different recognition models live in incompatible vector spaces
(and different dimensions), so every reader that compares them must first
restrict itself to one model. The active model follows the licensing mode:
commercial mode never runs InsightFace, so it embeds with SFace (128-d); every
other install keeps ArcFace/buffalo_l (512-d).

A NULL ``embedding_model`` / ``centroid_model`` tag is the legacy ArcFace space:
rows written before the column existed are all ArcFace.
"""

import licensing

ARCFACE_MODEL = 'arcface_buffalo_l'
SFACE_MODEL = 'sface'

EMBEDDING_DIMS = {ARCFACE_MODEL: 512, SFACE_MODEL: 128}


def embedding_model_for_policy(policy):
    """The recognition model a licensing ``Policy`` runs."""
    return SFACE_MODEL if licensing.is_commercial(policy) else ARCFACE_MODEL


def active_embedding_model(cfg=None):
    """The recognition model the face pipeline runs under the licensing policy.

    ``cfg`` is a ``ScoringConfig``; without one the process-wide policy applies
    (which itself resolves from the config on first use).
    """
    policy = licensing.resolve_policy(cfg.config) if cfg is not None else licensing.policy()
    return embedding_model_for_policy(policy)


def expected_dim(model):
    """Embedding length of ``model``; an unknown model is an error, not a guess."""
    try:
        return EMBEDDING_DIMS[model]
    except KeyError:
        raise ValueError(f"unknown face embedding model {model!r}") from None


def model_filter_sql(model, column='embedding_model'):
    """``(sql, params)`` selecting rows whose ``column`` tag is ``model``.

    A NULL tag counts as ArcFace, so it matches only when ``model`` is ArcFace.
    """
    if model == ARCFACE_MODEL:
        return f"({column} = ? OR {column} IS NULL)", [model]
    return f"{column} = ?", [model]


def tag_matches(tag, model):
    """Python twin of :func:`model_filter_sql` for an already-loaded tag."""
    return (tag or ARCFACE_MODEL) == model


def model_mismatch_sql(model, column='embedding_model'):
    """``(sql, params)`` selecting rows tagged with a model OTHER than ``model``."""
    if model == ARCFACE_MODEL:
        return f"({column} IS NOT NULL AND {column} != ?)", [model]
    return f"({column} IS NULL OR {column} != ?)", [model]


def mismatched_face_models(conn, active_model):
    """``[(model, count)]`` of stored face embeddings NOT made by ``active_model``."""
    sql, params = model_mismatch_sql(active_model)
    return conn.execute(
        "SELECT COALESCE(embedding_model, ?), COUNT(*) FROM faces "
        f"WHERE embedding IS NOT NULL AND {sql} GROUP BY 1",
        [ARCFACE_MODEL] + params).fetchall()


def face_model_mismatch_message(model, count, active_model):
    """The one-line explanation shared by ``--doctor`` and the scan warning."""
    return (f"{count:,} stored faces use '{model}' but the active model is "
            f"'{active_model}'; re-extract faces (--extract-faces-gpu-force, which uses the "
            "active model's extractor and keeps named persons) so every face is comparable")
