"""Non-finite ``f_stop`` rows must not turn the stats routes into HTTP 500, and the
per-category top camera / lens must be deterministic when counts tie.

A real library holds ``f_stop = +Inf`` rows written by an older EXIF layer. SQLite's
``AVG`` over such a row is ``+Inf``; ``_sanitize_stats`` turns it into ``None``, and
``avg_f_stop: Union[int, float]`` rejects ``None``, so FastAPI answered 500 *after* the
(minutes-long) compute. Both ``/api/stats/categories`` and ``/api/stats/gear`` are
exercised through the real routes.

Rows go into the shared session database, so every category / camera / lens name carries
a ``uuid4`` suffix: another test's rows can never join these groups.
"""

import uuid

import pytest

from api.config import invalidate_stats_cache

INF = float("inf")


@pytest.fixture()
def _fresh_stats():
    invalidate_stats_cache()
    yield
    invalidate_stats_cache()


def _row(prefix, name, *, category, f_stop, camera="", lens="", aggregate=6.0, iso=100.0):
    row = {
        "path": f"{prefix}{name}.jpg", "filename": f"{name}.jpg", "aggregate": aggregate,
        "category": category, "f_stop": f_stop, "iso": iso,
        "date_taken": "2026:01:02 10:00:00",
    }
    if camera:
        row["camera_model"] = camera
    if lens:
        row["lens_model"] = lens
    return row


def _by_category(rows, category):
    matches = [r for r in rows if r["category"] == category]
    assert len(matches) == 1, f"expected one {category!r} row, got {len(matches)}"
    return matches[0]


class TestCategoriesNonFinite:
    def test_all_inf_group_is_zero_and_finite_group_is_the_mean(
            self, edition_client, seed_photos_prefix, _fresh_stats):
        tag = uuid.uuid4().hex
        prefix = f"/statsinf-{tag}/"
        cat_inf, cat_ok = f"inf_{tag}", f"ok_{tag}"
        seed_photos_prefix(prefix, [
            *(_row(prefix, f"inf{i}", category=cat_inf, f_stop=INF, aggregate=5.0 + i)
              for i in range(3)),
            *(_row(prefix, f"ok{i}", category=cat_ok, f_stop=v, aggregate=7.0 + i, iso=200.0)
              for i, v in enumerate((2.8, 4.0, 5.6))),
        ])

        resp = edition_client.get("/api/stats/categories")
        assert resp.status_code == 200, resp.text
        rows = resp.json()

        inf_row = _by_category(rows, cat_inf)
        assert inf_row["count"] == 3
        assert inf_row["avg_f_stop"] == 0
        assert inf_row["avg_score"] == 6.0  # (5 + 6 + 7) / 3: nothing else was dropped

        ok_row = _by_category(rows, cat_ok)
        assert ok_row["count"] == 3
        assert ok_row["avg_f_stop"] == round((2.8 + 4.0 + 5.6) / 3, 1)
        assert ok_row["avg_score"] == 8.0
        assert ok_row["avg_iso"] == 200

    def test_mixed_group_averages_only_the_finite_rows(
            self, edition_client, seed_photos_prefix, _fresh_stats):
        tag = uuid.uuid4().hex
        prefix = f"/statsmix-{tag}/"
        cat = f"mixed_{tag}"
        seed_photos_prefix(prefix, [
            _row(prefix, "a", category=cat, f_stop=2.0),
            _row(prefix, "b", category=cat, f_stop=4.0),
            _row(prefix, "c", category=cat, f_stop=INF),
            _row(prefix, "d", category=cat, f_stop=-INF),
        ])

        resp = edition_client.get("/api/stats/categories", params={"category": cat})
        assert resp.status_code == 200, resp.text
        (row,) = resp.json()
        assert row["count"] == 4  # the non-finite rows still count as photos
        assert row["avg_f_stop"] == 3.0


class TestGearNonFinite:
    def test_camera_lens_and_combo_groups_survive_an_inf_row(
            self, edition_client, seed_photos_prefix, _fresh_stats):
        tag = uuid.uuid4().hex
        prefix = f"/gearinf-{tag}/"
        cat = f"gear_{tag}"
        cam_mixed, cam_inf, cam_ok = f"CamMixed {tag}", f"CamInf {tag}", f"CamOk {tag}"
        lens_mixed, lens_inf, lens_ok = f"LensMixed {tag}", f"LensInf {tag}", f"LensOk {tag}"
        seed_photos_prefix(prefix, [
            _row(prefix, "m0", category=cat, f_stop=2.0, camera=cam_mixed, lens=lens_mixed),
            _row(prefix, "m1", category=cat, f_stop=4.0, camera=cam_mixed, lens=lens_mixed),
            _row(prefix, "m2", category=cat, f_stop=INF, camera=cam_mixed, lens=lens_mixed),
            _row(prefix, "i0", category=cat, f_stop=INF, camera=cam_inf, lens=lens_inf),
            _row(prefix, "i1", category=cat, f_stop=INF, camera=cam_inf, lens=lens_inf),
            _row(prefix, "o0", category=cat, f_stop=2.8, camera=cam_ok, lens=lens_ok),
        ])

        resp = edition_client.get("/api/stats/gear", params={"category": cat})
        assert resp.status_code == 200, resp.text
        body = resp.json()

        def item(group, name):
            (found,) = [r for r in body[group] if r["name"] == name]
            return found

        for group, mixed, only_inf, ok in (
                ("cameras", cam_mixed, cam_inf, cam_ok),
                ("lenses", lens_mixed, lens_inf, lens_ok),
                ("combos", f"{cam_mixed} + {lens_mixed}", f"{cam_inf} + {lens_inf}",
                 f"{cam_ok} + {lens_ok}")):
            assert item(group, mixed)["count"] == 3
            assert item(group, mixed)["avg_f_stop"] == 3.0
            assert item(group, only_inf)["count"] == 2
            assert item(group, only_inf)["avg_f_stop"] == 0
            assert item(group, ok)["avg_f_stop"] == 2.8


class TestTopCameraLensTieOrder:
    """Two cameras (and two lenses) with equal counts in one category: the window
    function's ORDER BY must be total, or which one is 'top' follows the plan."""

    @pytest.mark.parametrize("insert_order", ["ascending", "descending"])
    def test_equal_counts_resolve_to_the_alphabetically_first(
            self, edition_client, seed_photos_prefix, _fresh_stats, insert_order):
        tag = uuid.uuid4().hex
        prefix = f"/stattie-{tag}/"
        cat = f"tie_{tag}"
        names = [f"{letter}-{tag}" for letter in "abc"]
        if insert_order == "descending":
            names.reverse()
        rows = []
        for n, name in enumerate(names):
            for i in range(2):  # every camera / lens appears twice: a three-way tie
                rows.append(_row(prefix, f"{n}-{i}", category=cat, f_stop=2.8,
                                 camera=f"Cam {name}", lens=f"Lens {name}"))
        seed_photos_prefix(prefix, rows)

        resp = edition_client.get("/api/stats/categories", params={"category": cat})
        assert resp.status_code == 200, resp.text
        (row,) = resp.json()
        assert row["top_camera"] == f"Cam a-{tag}"
        assert row["top_lens"] == f"Lens a-{tag}"


class TestApertureHistogram:
    def test_inf_f_stop_is_not_a_null_bucket(
            self, edition_client, seed_photos_prefix, _fresh_stats):
        tag = uuid.uuid4().hex
        prefix = f"/statsap-{tag}/"
        cat = f"ap_{tag}"
        seed_photos_prefix(prefix, [
            _row(prefix, "a", category=cat, f_stop=2.8),
            _row(prefix, "b", category=cat, f_stop=INF),
        ])

        resp = edition_client.get("/api/stats/settings", params={"category": cat})
        assert resp.status_code == 200, resp.text
        aperture = resp.json()["aperture"]
        assert aperture == [{"value": 2.8, "count": 1}]
