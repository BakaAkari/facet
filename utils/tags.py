"""
Tag string conversion utilities for Facet.

Convert between tag lists and comma-separated strings for database storage.
"""


def tags_to_string(tags):
    """
    Convert list of tags to comma-separated string for database storage.

    Args:
        tags: List of tag strings

    Returns:
        str: Comma-separated tags, or None if empty
    """
    if not tags:
        return None
    return ','.join(tags)


def string_to_tags(tags_str):
    """
    Convert comma-separated string back to list of tags.

    Args:
        tags_str: Comma-separated tag string

    Returns:
        list: List of tag strings
    """
    if not tags_str:
        return []
    return [tag.strip() for tag in tags_str.split(',') if tag.strip()]


def get_tag_params(config):
    """
    Extract tagging parameters from config.

    The threshold is ``tag_threshold_percent`` of the ACTIVE profile's clip
    block (``get_clip_config()`` resolves 'auto'), so it always belongs to the
    model whose embeddings are being tagged. The old
    ``similarity_threshold_percent`` key is deliberately not read.

    Args:
        config: ScoringConfig instance

    Returns:
        tuple: (threshold, max_tags) for tag generation
    """
    clip_cfg = config.get_clip_config()
    tag_settings = config.get_tagging_settings()
    threshold = clip_cfg.get('tag_threshold_percent', 8) / 100
    max_tags = tag_settings.get('max_tags', 5)
    return threshold, max_tags
