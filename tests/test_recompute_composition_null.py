"""``--recompute-composition-cpu`` stores NULL for a thumbnail with no subject box.

``recompute_composition_scores`` is the one writer that also replaces the legacy neutral
``power_point_score`` of 5.0, so a re-run over an existing library turns the placeholders into
NULL (a measured value is re-measured, and a row whose thumbnail cannot be decoded is left alone).
"""

import io

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")
pytest.importorskip("cv2")

from db import get_connection  # noqa: E402
from db.schema import init_database  # noqa: E402
from tests.test_composition_hoist import _structured_image  # noqa: E402


def _jpeg(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, 'JPEG', quality=95)
    return buf.getvalue()


@pytest.fixture()
def library(tmp_path):
    from config.scoring_config import ScoringConfig
    from config_resolve import defaults_path
    from processing.scorer import Facet

    db = str(tmp_path / 'recompute.db')
    init_database(db)
    facet = Facet.__new__(Facet)   # bypass the heavy __init__; only db_path/config are used
    facet.db_path = db
    facet.config = ScoringConfig(defaults_path(), validate=False)
    thumbs = {
        '/lib/flat.jpg': _jpeg(np.full((160, 160, 3), 128, dtype=np.uint8)),
        '/lib/structured.jpg': _jpeg(_structured_image(3)),
        '/lib/corrupt.jpg': b'not a jpeg',
    }
    with get_connection(db, row_factory=False) as conn:
        for path, blob in thumbs.items():
            conn.execute(
                "INSERT INTO photos (path, filename, thumbnail, power_point_score, comp_score, "
                "leading_lines_score) VALUES (?, ?, ?, 5.0, 5.0, -1.0)", (path, path.rsplit('/', 1)[-1], blob))
        conn.commit()
    return facet, db


def _power_point(db):
    with get_connection(db, row_factory=True) as conn:
        return {r['path']: r['power_point_score']
                for r in conn.execute("SELECT path, power_point_score FROM photos")}


def test_a_flat_thumbnail_goes_from_the_legacy_five_to_null(library):
    facet, db = library
    facet.recompute_composition_scores()
    assert _power_point(db)['/lib/flat.jpg'] is None


def test_a_structured_thumbnail_gets_a_measured_two_decimal_score(library):
    facet, db = library
    facet.recompute_composition_scores()
    value = _power_point(db)['/lib/structured.jpg']
    assert isinstance(value, float)
    assert value == round(value, 2)


def test_an_undecodable_thumbnail_keeps_its_stored_value(library):
    facet, db = library
    facet.recompute_composition_scores()
    assert _power_point(db)['/lib/corrupt.jpg'] == 5.0


def test_one_null_does_not_discard_the_other_rows_updates(library):
    """The old unguarded ``round(None, 2)`` raised before the single commit and lost every update."""
    facet, db = library
    facet.recompute_composition_scores()
    with get_connection(db, row_factory=True) as conn:
        rows = {r['path']: dict(r) for r in conn.execute(
            "SELECT path, comp_score, leading_lines_score FROM photos")}
    assert rows['/lib/flat.jpg']['comp_score'] == 7.0           # the no-box centre assumption
    assert rows['/lib/flat.jpg']['leading_lines_score'] >= 0    # rewritten from the -1.0 sentinel
    assert rows['/lib/structured.jpg']['leading_lines_score'] >= 0
    assert rows['/lib/corrupt.jpg']['leading_lines_score'] == -1.0
