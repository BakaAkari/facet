"""Which model passes a scan schedules: the pure routing decision.

Kept free of torch and of any processor instance so ``--doctor`` can ask the
same question the scan asks, from configuration alone, instead of keeping a
second hand-rolled copy of the profile that drifts from it.
"""

import logging
from typing import Callable, Dict, List, Optional

import licensing

logger = logging.getLogger("facet.multi_pass")

# Profile ``tagging_model`` string
# -> (model-manager tagger key, profile-activation min VRAM in GB). Every routing
# site (route_models, _run_model_pass, run_single_pass, facet recompute-tags)
# reads this map so adding a tagger is one entry, not ~6 duplicated string->key
# branches that silently fall through to CLIP when one is missed (commit 3cf0604).
# RAM++ is not here: it is a non-VLM tagger handled separately in facet.py.
TAGGING_MODELS = {
    'qwen2.5-vl-7b': ('vlm_tagger', 16),
    'qwen3-vl-2b': ('qwen3_vl_tagger', 4),
    'qwen3.5-2b': ('qwen3_5_tagger', 4),
    'qwen3.5-4b': ('qwen3_5_4b_tagger', 8),
}


def commercial_primary_quality(selected_quality: str, policy_=None) -> str:
    """Primary aesthetic/quality model under commercial mode.

    The CLIP+MLP head survives only when the registry allows it (it does
    under ``licensing.allow_unclear``); every pyiqa model is replaced by the
    native Q-ReAlign scorer, which fills aesthetic and quality in one pass.
    """
    if selected_quality == 'clip_aesthetic' and licensing.loader_allowed('clip_aesthetic', policy_):
        return 'clip_aesthetic'
    return 'qrealign'


def single_pass_unsupported_reason(aesthetic_model: str, policy_=None) -> Optional[str]:
    """Why single-pass cannot score this profile's aesthetics, or None when it can.

    A TOPIQ profile (the SigLIP ones) loads no CLIP+MLP aesthetic head and only
    the multi-pass scan runs TOPIQ, so a single-pass scan has nothing to score
    the aesthetic with. Commercial mode is the exception: native Q-ReAlign
    serves the head-less branch there.
    """
    if aesthetic_model == 'topiq' and not licensing.is_commercial(policy_):
        return ("this profile scores aesthetics with TOPIQ, which only the multi-pass "
                "scan runs (single-pass needs the CLIP+MLP head of the legacy/8gb profiles)")
    return None


def resolve_selected_quality(config: dict, profile: dict, policy_=None) -> str:
    """The primary aesthetic/quality model a run uses, licensing applied.

    Args:
        config: The scoring config dict (``scorer.config.config``)
        profile: The active VRAM profile's dict (source of ``aesthetic_model``)
    """
    quality_model = config.get('quality', {}).get('model', 'auto')

    if quality_model == 'auto':
        profile_aesthetic = profile.get('aesthetic_model', 'clip-mlp')
        # Map profile aesthetic_model to quality model name used in passes
        selected_quality = 'topiq' if profile_aesthetic == 'topiq' else 'clip_aesthetic'
    else:
        selected_quality = quality_model

    if licensing.is_commercial(policy_):
        selected_quality = commercial_primary_quality(selected_quality, policy_)
    return selected_quality


def face_pass_name(policy_=None) -> str:
    """Name of the face pass the licensing policy schedules."""
    return 'insightface' if licensing.loader_allowed('insightface', policy_) else 'faces_permissive'


def route_models(config: dict, profile: dict, extended: Dict[str, bool],
                 tagger_fits: Callable[[float], bool], policy_=None) -> List[str]:
    """The loader names a scan schedules, in queue order.

    Args:
        config: The scoring config dict
        profile: The active VRAM profile's dict
        extended: Resolved extended-IQA flags (``get_extended_iqa_settings()``)
        tagger_fits: Whether a VLM tagger needing this many GB may be loaded
        policy_: Licensing policy; the process-wide one when omitted
    """
    models = []

    # CLIP for embeddings (always needed)
    models.append('clip')

    # Quality/Aesthetic model selection
    selected_quality = resolve_selected_quality(config, profile, policy_)
    commercial = licensing.is_commercial(policy_)

    # PyIQA models: topiq, hyperiqa, dbcnn, musiq
    pyiqa_models = ['topiq', 'hyperiqa', 'dbcnn', 'musiq', 'musiq-koniq', 'clipiqa+']
    if selected_quality in pyiqa_models or selected_quality == 'qrealign':
        models.append(selected_quality)
    # else: clip_aesthetic uses same model as CLIP embeddings

    # Supplementary PyIQA models (per-profile). Commercial mode drops every
    # one the registry refuses; they are all pyiqa-backed.
    for supp_model in profile.get('supplementary_pyiqa', []):
        if supp_model in models:
            continue
        if commercial and not licensing.loader_allowed(supp_model, policy_):
            logger.info("Commercial mode: skipping supplementary model '%s'.", supp_model)
            continue
        models.append(supp_model)

    # Extended-IQA tier (config-gated: qrealign defaults to "auto" -> on for
    # every profile but legacy; aesthetic_v25/deqa are OFF by default). Each
    # scorer writes its own dedicated column (see PYIQA_COLUMN_MAP); they are
    # in the supplementary set so a load/VRAM failure skips the pass instead
    # of aborting the scan.
    for ext_model in ('qrealign', 'aesthetic_v25', 'deqa'):
        if not extended.get(ext_model) or ext_model in models:
            continue
        if commercial and not licensing.loader_allowed(ext_model, policy_):
            logger.info("Commercial mode: skipping extended-IQA model '%s'.", ext_model)
            continue
        models.append(ext_model)

    # Saliency model (BiRefNet) if enabled for this profile
    if profile.get('saliency_enabled', False):
        models.append('saliency')

    # Tagging model (from profile)
    tagging_entry = TAGGING_MODELS.get(profile.get('tagging_model', 'clip'))
    if tagging_entry and tagger_fits(tagging_entry[1]):
        models.append(tagging_entry[0])
    # else: CLIP tagging uses clip embeddings, no extra model needed

    # Composition model. SAMP-Net is the only one with a multi-pass pass;
    # qwen2-vl-2b runs a per-photo VLM generation and exists for
    # `processing.mode: "single-pass"` only. Say so instead of silently
    # leaving composition to the rule-based fallback, which is what made the
    # 24gb profile ship without any composition model on the default path.
    # On a profile single-pass cannot serve, the scan falls back to multi-pass,
    # so that advice would lead straight back here and is left out.
    composition_model = profile.get('composition_model', 'rule-based')
    if composition_model == 'samp-net' and commercial and not licensing.loader_allowed('samp_net', policy_):
        logger.info("Commercial mode: SAMP-Net is not permitted; composition uses the "
                    "rule-based analyzer.")
    elif composition_model == 'samp-net':
        models.append('samp_net')
    elif composition_model not in ('rule-based', 'none'):
        single_pass_hint = ""
        if single_pass_unsupported_reason(profile.get('aesthetic_model', 'clip-mlp'), policy_) is None:
            single_pass_hint = ", or processing.mode to 'single-pass'"
        logger.warning(
            "Composition model '%s' has no multi-pass pass; composition falls back to the "
            "rule-based analyzer. Set composition_model to 'samp-net'%s.",
            composition_model, single_pass_hint)

    # Face analysis. Commercial mode swaps InsightFace for the permissive stack.
    models.append(face_pass_name(policy_))

    return models
