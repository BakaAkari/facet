"""Strict exposure-seconds projection for sorting; stored EXIF is never changed."""
import math

SQL_FUNCTION = 'photo_exposure_seconds'


def exposure_seconds(value: object) -> float | None:
    """Return positive finite seconds for decimal or rational EXIF values."""
    if type(value) not in (str, int, float):
        return None
    try:
        if isinstance(value, str) and '/' in value:
            numerator, denominator = (float(part) for part in value.split('/'))
            if not math.isfinite(numerator) or not math.isfinite(denominator):
                return None
            if numerator <= 0 or denominator <= 0:
                return None
            seconds = numerator / denominator
        else:
            seconds = float(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None
