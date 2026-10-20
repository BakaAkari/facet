"""Stored score columns shared by exports and gallery range filters.

Definitions only: importing this registry never imports a model or backend.
"""

# The seven legacy scores, in this order: first in --export-json, after the metadata columns in --export-csv.
JSON_LEGACY_SCORE_COLUMNS = (
    'aggregate', 'aesthetic', 'comp_score', 'face_quality', 'tech_sharpness',
    'exposure_score', 'color_score',
)

SCORE_RANGE_COLUMNS = [
    ("aggregate", "min_score", "max_score", True),
    ("aesthetic", "min_aesthetic", "max_aesthetic", True),
    ("quality_score", "min_quality_score", "max_quality_score", True),
    ("topiq_score", "min_topiq", "max_topiq", True),
    ("tech_sharpness", "min_sharpness", "max_sharpness", True),
    ("exposure_score", "min_exposure", "max_exposure", True),
    ("color_score", "min_color", "max_color", True),
    ("contrast_score", "min_contrast", "max_contrast", True),
    ("noise_sigma", "min_noise", "max_noise", True),
    ("mean_saturation", "min_saturation", "max_saturation", True),
    ("mean_luminance", "min_luminance", "max_luminance", True),
    ("histogram_spread", "min_histogram_spread", "max_histogram_spread", True),
    ("channel_clip_shadow_pct", "min_channel_clip_shadow", "max_channel_clip_shadow", True),
    ("channel_clip_highlight_pct", "min_channel_clip_highlight", "max_channel_clip_highlight", True),
    ("dynamic_range_stops", "min_dynamic_range", "max_dynamic_range", True),
    ("comp_score", "min_composition", "max_composition", True),
    ("power_point_score", "min_power_point", "max_power_point", True),
    ("leading_lines_score", "min_leading_lines", "max_leading_lines", True),
    ("isolation_bonus", "min_isolation", "max_isolation", True),
    ("face_count", "min_face_count", "max_face_count", False),
    ("face_quality", "min_face_quality", "max_face_quality", True),
    ("eye_sharpness", "min_eye_sharpness", "max_eye_sharpness", True),
    ("face_sharpness", "min_face_sharpness", "max_face_sharpness", True),
    ("face_ratio", "min_face_ratio", "max_face_ratio", True),
    ("face_confidence", "min_face_confidence", "max_face_confidence", True),
    ("star_rating", "min_star_rating", "max_star_rating", False),
    ("aesthetic_iaa", "min_aesthetic_iaa", "max_aesthetic_iaa", True),
    ("face_quality_iqa", "min_face_quality_iqa", "max_face_quality_iqa", True),
    ("liqe_score", "min_liqe", "max_liqe", True),
    ("qrealign_score", "min_qrealign", "max_qrealign", True),
    ("aesthetic_v25", "min_aesthetic_v25", "max_aesthetic_v25", True),
    ("deqa_score", "min_deqa", "max_deqa", True),
    ("subject_sharpness", "min_subject_sharpness", "max_subject_sharpness", True),
    ("subject_prominence", "min_subject_prominence", "max_subject_prominence", True),
    ("subject_placement", "min_subject_placement", "max_subject_placement", True),
    ("bg_separation", "min_bg_separation", "max_bg_separation", True),
    ("narrative_moment_confidence", "min_moment_confidence", "max_moment_confidence", True),
]

FORM_METRIC_COLUMNS = (
    'form_symmetry', 'form_balance', 'form_edge_entropy', 'form_fractal',
    'color_harmony',
)

# Stored metrics without numeric gallery range filters. Classifications remain
# their original text; exporting a stored value never computes it.
_OTHER_SCORE_COLUMNS = (
    'aesthetic_clip', 'eyes_open_score', 'expression_score',
    'composition_pattern', 'learned_score',
)

# A manual rating is not a computed score. Camera/EXIF fields have their own
# gallery registry and deliberately do not enter this score-only namespace.
JSON_SCORE_COLUMNS = tuple(dict.fromkeys(
    [column for column, *_ in SCORE_RANGE_COLUMNS if column != 'star_rating']
    + list(FORM_METRIC_COLUMNS) + list(_OTHER_SCORE_COLUMNS)
))
