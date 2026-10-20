"""Pure-SQL projection of the TEXT ``shutter_speed`` column to positive seconds."""


def _decimal_ok(x: str) -> str:
    """SQL predicate: ``x`` is one well-formed unsigned decimal: digits with at most one '.', at
    least one digit, then an optional exponent (e/E, one optional sign, 1+ digits).

    ``CAST(x AS REAL)`` reads a numeric PREFIX ('1-2' -> 1.0), so the shape is
    checked first with GLOB (no REGEXP, no JSON1). Bare '.5' and '5.' are accepted.
    """
    return (
        f"({x} GLOB '*[0-9.]' AND ({x} GLOB '[0-9]*' OR {x} GLOB '.[0-9]*') "
        f"AND {x} NOT GLOB '*[^0-9.eE+-]*' AND {x} NOT GLOB '*.*.*' "
        f"AND {x} NOT GLOB '*[eE]*[eE]*' AND {x} NOT GLOB '*[eE]*.*' "
        f"AND {x} NOT GLOB '*[^eE][+-]*')"
    )


def exposure_seconds_sql(column: str = 'shutter_speed') -> str:
    """Return a SQLite expression: positive finite seconds as REAL, else NULL.

    Reads decimal ('0.25', '1e-3') and rational ('1/250') EXIF text. Each side of
    the slash (or the whole value) must be a well-formed unsigned decimal, with
    surrounding spaces trimmed; missing, malformed, zero, negative and nonfinite
    values yield NULL. ``column`` is interpolated verbatim and must be a trusted
    identifier. Runs at C speed (no Python UDF, so no GIL contention) and never
    changes the stored value.
    """
    c = f"CAST({column} AS TEXT)"
    t = f"TRIM({c})"
    slash = f"INSTR({t}, '/')"
    n = f"TRIM(SUBSTR({t}, 1, {slash} - 1))"
    d = f"TRIM(SUBSTR({t}, {slash} + 1))"
    # Fast paths for the overwhelmingly common shapes (plain decimal, plain integer
    # rational); anything else (exponent, spaces, odd text) takes the validated path.
    # On a pure digit/'.' or digit/'/' string the REAL prefix cast reads the whole
    # number, so it needs no SUBSTR for the numerator.
    return (
        f"(CASE WHEN {c} GLOB '[0-9]*' AND {c} NOT GLOB '*[^0-9.]*' AND {c} NOT GLOB '*.*.*' "
        f"THEN CASE WHEN CAST({c} AS REAL) > 0 AND CAST({c} AS REAL) < 1e308 "
        f"THEN CAST({c} AS REAL) END "
        f"WHEN {c} GLOB '[0-9]*/[0-9]*' AND {c} NOT GLOB '*[^0-9/]*' AND {c} NOT GLOB '*/*/*' "
        f"THEN CASE WHEN CAST({c} AS REAL) > 0 AND CAST({c} AS REAL) < 1e308 "
        f"AND CAST(SUBSTR({c}, INSTR({c}, '/') + 1) AS REAL) > 0 "
        f"AND CAST(SUBSTR({c}, INSTR({c}, '/') + 1) AS REAL) < 1e308 "
        f"THEN CAST({c} AS REAL) / CAST(SUBSTR({c}, INSTR({c}, '/') + 1) AS REAL) END "
        f"WHEN {slash} > 0 "
        f"THEN CASE WHEN {_decimal_ok(n)} AND {_decimal_ok(d)} "
        f"AND CAST({n} AS REAL) > 0 AND CAST({d} AS REAL) > 0 "
        f"AND CAST({n} AS REAL) < 1e308 AND CAST({d} AS REAL) < 1e308 "
        f"THEN CAST({n} AS REAL) / CAST({d} AS REAL) END "
        f"WHEN {_decimal_ok(t)} AND CAST({t} AS REAL) > 0 AND CAST({t} AS REAL) < 1e308 "
        f"THEN CAST({t} AS REAL) END)"
    )
