"""Pure-SQL projection of the TEXT ``shutter_speed`` column to positive seconds."""


def exposure_seconds_sql(column: str = 'shutter_speed') -> str:
    """Return a SQLite expression: positive finite seconds as REAL, else NULL.

    Reads decimal ('0.25', '1e-3') and rational ('1/250') EXIF text; missing,
    malformed, zero, negative and nonfinite values yield NULL. ``column`` is
    interpolated verbatim and must be a trusted identifier. Runs at C speed (no
    Python UDF, so no GIL contention) and never changes the stored value.
    """
    t = f"TRIM(CAST({column} AS TEXT))"
    slash = f"INSTR({t}, '/')"
    num = f"CAST(SUBSTR({t}, 1, {slash} - 1) AS REAL)"
    den = f"CAST(SUBSTR({t}, {slash} + 1) AS REAL)"
    dec = f"CAST({t} AS REAL)"
    return (
        f"(CASE WHEN {t} != '' AND {t} NOT GLOB '*[^0-9.eE+/ -]*' AND {t} NOT GLOB '*/*/*' "
        f"THEN CASE WHEN {slash} > 0 "
        f"THEN CASE WHEN {num} > 0 AND {den} > 0 AND {num} < 1e308 AND {den} < 1e308 "
        f"THEN {num} / {den} END "
        f"WHEN {dec} > 0 AND {dec} < 1e308 THEN {dec} END END)"
    )
