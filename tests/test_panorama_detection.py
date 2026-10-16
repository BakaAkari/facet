"""Panorama detection.

Geometry is exercised on synthetic frames built by warping one image through a
known camera rotation (`H = K R K^-1`), so a "panorama" here is geometrically
exact and needs no committed photo corpus. The thresholds under test were
calibrated against 26 real panoramas and 8 real non-panoramas; the cases below
encode the failures that calibration exposed, each of which shipped as a bug in
an earlier revision of this detector.
"""

import sqlite3

import cv2
import numpy as np
import pytest

from db.sequence_overrides import clear_sequence_overrides, set_sequence_overrides
from tests.test_post_scan_indexes import MIN_COVERING_SQLITE_EXPRESSION
from utils.sequence import BRACKET
from utils.panorama import (
    DEFAULTS,
    HDR_PANORAMA,
    PANORAMA,
    classify_kind,
    detect_panoramas,
    find_segments,
    load_overrides,
    read_run_cache,
    resolve_segments,
    run_signature,
    split_runs,
    write_run_cache,
)


def _sequence_labels(db_path):
    """Every stored panorama label, as a comparable snapshot."""
    with sqlite3.connect(db_path) as conn:
        return sorted(conn.execute(
            "SELECT path, sequence_group_id, sequence_kind, is_sequence_lead, "
            "sequence_ev_offset FROM photos "
            "WHERE sequence_kind IS NOT NULL").fetchall())


def _settings(**overrides):
    settings = dict(DEFAULTS)
    settings.update(overrides)
    return settings


def _step(dx=0.0, dy=0.0, inliers=200):
    return {'dx': dx, 'dy': dy, 'inliers': inliers}


def _paths(count):
    return [f'/p{i}.jpg' for i in range(count)]


@pytest.fixture()
def pass_overrides(monkeypatch):
    """The panorama settings a pass sees: `DEFAULTS` plus whatever a test puts here.

    Replaces the config lookup so the pass neither reads the box's own
    `scoring_config.json` nor spawns a process pool (`workers=1`, the serial path)
    unless a test asks for one. A test that edits a threshold edits this dict.
    """
    overrides = {'workers': 1}
    monkeypatch.setattr('config.ScoringConfig.get_panorama_detection_settings',
                        lambda self: dict(overrides))
    return overrides


@pytest.fixture()
def measured_runs(monkeypatch):
    """Spy on `_analyse_runs`: the list of runs (as path lists) each pass measured.

    Calls through, so the pass behaves exactly as without it; `measured_runs` is
    cleared by the test between passes.
    """
    import utils.panorama as module
    measured = []
    original = module._analyse_runs

    def spy(db_path, runs, *args, **kwargs):
        measured.extend([photo['path'] for photo in run] for run in runs)
        return original(db_path, runs, *args, **kwargs)

    monkeypatch.setattr(module, '_analyse_runs', spy)
    return measured


def _pass_settings(**overrides):
    """What `detect_panoramas` resolves its settings to under `pass_overrides`."""
    settings = dict(DEFAULTS)
    settings.update({'workers': 1, **overrides})
    return settings


# ---------------------------------------------------------------------------
# Run-level acceptance
# ---------------------------------------------------------------------------

class TestFindSegments:
    def test_a_steady_sweep_is_one_segment(self):
        steps = [_step(dx=-0.07) for _ in range(12)]
        segments = find_segments(_paths(13), steps, _settings())
        assert len(segments) == 1
        assert len(segments[0]['paths']) == 13
        assert segments[0]['axis'] == 'x'
        assert segments[0]['drift'] == pytest.approx(0.84, abs=0.01)

    def test_a_burst_holding_still_is_not_a_panorama(self):
        """The discriminator is cumulative drift, not per-pair shift."""
        steps = [_step(dx=0.001 * (-1) ** i) for i in range(12)]
        assert find_segments(_paths(13), steps, _settings()) == []

    def test_an_hdr_sweep_with_static_bracket_steps_is_accepted(self):
        """Requiring every pair to shift would reject every HDR panorama."""
        steps = []
        for _ in range(8):
            steps += [_step(dx=0.0), _step(dx=0.0), _step(dx=-0.12)]
        segments = find_segments(_paths(len(steps) + 1), steps, _settings())
        assert len(segments) == 1
        assert len(segments[0]['paths']) == len(steps) + 1

    def test_the_axis_is_chosen_by_drift_not_by_run_length(self):
        """On the motionless axis the extension never breaks and runs to the end.

        Ranking candidates by length therefore always picked the degenerate axis:
        a real 57-frame sweep resolved to drift 0.03 on `y` instead of ~1.5 on `x`.
        """
        steps = [_step(dx=-0.09, dy=0.0) for _ in range(20)]
        segments = find_segments(_paths(21), steps, _settings())
        assert len(segments) == 1
        assert segments[0]['axis'] == 'x'

    def test_a_direction_reversal_splits_one_run_into_two_sweeps(self):
        steps = [_step(dx=-0.09) for _ in range(12)] + [_step(dx=+0.09) for _ in range(12)]
        segments = find_segments(_paths(25), steps, _settings())
        assert len(segments) == 2

    def test_a_recomposition_step_is_not_absorbed(self):
        """A step moving substantially in both axes belongs to neither sweep.

        Measured on a real set: the step into the panorama carried dy=+0.255
        against |dy|<=0.004 on every genuine step, and absorbing it blew the
        orthogonal budget and lost the whole set.
        """
        steps = ([_step(dx=+0.03) for _ in range(8)]
                 + [_step(dx=-0.165, dy=+0.255)]
                 + [_step(dx=-0.06) for _ in range(10)])
        segments = find_segments(_paths(len(steps) + 1), steps, _settings())
        assert len(segments) == 1
        assert segments[0]['paths'][0] == '/p9.jpg'

    def test_a_vertical_sweep_is_found_on_the_y_axis(self):
        steps = [_step(dy=+0.08) for _ in range(12)]
        segments = find_segments(_paths(13), steps, _settings())
        assert len(segments) == 1
        assert segments[0]['axis'] == 'y'

    def test_a_run_shorter_than_min_frames_is_rejected(self):
        steps = [_step(dx=-0.20) for _ in range(4)]
        assert find_segments(_paths(5), steps, _settings()) == []

    def test_an_unmatched_pair_splits_rather_than_kills_the_run(self):
        steps = ([_step(dx=-0.09) for _ in range(10)]
                 + [_step(dx=0.0, inliers=0)]
                 + [_step(dx=-0.09) for _ in range(10)])
        segments = find_segments(_paths(len(steps) + 1), steps, _settings())
        assert len(segments) == 2

    def test_a_scene_cut_sized_jump_is_not_a_step(self):
        steps = [_step(dx=-0.95) for _ in range(12)]
        assert find_segments(_paths(13), steps, _settings()) == []


# ---------------------------------------------------------------------------
# Geometry on synthetic frames
# ---------------------------------------------------------------------------

def _rotated(image, degrees, focal_ratio=0.9):
    """Frame `image` as seen after rotating the camera about its vertical axis."""
    height, width = image.shape[:2]
    focal = focal_ratio * width
    intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
    theta = np.deg2rad(degrees)
    rotation = np.array([[np.cos(theta), 0, np.sin(theta)],
                         [0, 1, 0],
                         [-np.sin(theta), 0, np.cos(theta)]])
    homography = intrinsics @ rotation @ np.linalg.inv(intrinsics)
    return cv2.warpPerspective(image, homography, (width, height))


def _textured_frame(width=640, height=427, seed=7):
    """A frame with enough structure for SIFT to key on."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (height // 4, width // 4, 3), dtype=np.uint8)
    return cv2.resize(noise, (width, height), interpolation=cv2.INTER_LINEAR)


class TestSyntheticGeometry:
    def test_a_rotated_pair_reports_a_horizontal_translation(self):
        from utils.panorama import _sift_and_matcher, _translation
        base = cv2.cvtColor(_textured_frame(), cv2.COLOR_BGR2GRAY)
        turned = cv2.cvtColor(_rotated(_textured_frame(), 6.0), cv2.COLOR_BGR2GRAY)
        settings = _settings()
        sift, matcher = _sift_and_matcher(settings)

        def features(image):
            keypoints, descriptors = sift.detectAndCompute(image, None)
            return keypoints, descriptors, image.shape

        step = _translation(features(base), features(turned), matcher, settings)
        assert step['inliers'] >= settings['min_inliers']
        assert abs(step['dx']) > 0.02
        assert abs(step['dy']) < 0.02

    def test_unrelated_frames_do_not_match(self):
        from utils.panorama import _sift_and_matcher, _translation
        settings = _settings()
        sift, matcher = _sift_and_matcher(settings)

        def features(image):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray, None)
            return keypoints, descriptors, gray.shape

        step = _translation(features(_textured_frame(seed=1)),
                            features(_textured_frame(seed=99)), matcher, settings)
        assert step['inliers'] < settings['min_inliers']


# ---------------------------------------------------------------------------
# Kind classification
# ---------------------------------------------------------------------------

class TestClassifyKind:
    def test_a_locked_exposure_sweep_is_a_plain_panorama(self):
        assert classify_kind([12.0] * 10, _settings()) == PANORAMA

    def test_a_bracketed_sweep_is_an_hdr_panorama(self):
        assert classify_kind([10.0, 12.0, 14.0] * 4, _settings()) == HDR_PANORAMA

    def test_a_small_exposure_drift_is_not_hdr(self):
        """Auto-exposure wanders across a pan; that is not a bracket.

        One confirmed plain panorama drifts 0.7 stops, and a confirmed non-HDR
        run reaches 1.3 -- both must stay below the threshold.
        """
        assert classify_kind([12.0, 12.4, 12.7], _settings()) == PANORAMA
        assert classify_kind([12.0, 13.3], _settings()) == PANORAMA

    def test_missing_exposure_data_falls_back_to_plain(self):
        assert classify_kind([None, None], _settings()) == PANORAMA


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

class TestResolveSegments:
    def _segment(self, paths, kind=PANORAMA):
        return {'paths': paths, 'axis': 'x', 'drift': 1.0, 'ortho': 0.01, 'kind': kind}

    def test_without_overrides_the_detector_stands(self):
        segments = [self._segment(['/a.jpg', '/b.jpg', '/c.jpg'])]
        assert len(resolve_segments(segments, set(), {})) == 1

    def test_a_suppressed_member_drops_the_detected_set(self):
        segments = [self._segment(['/a.jpg', '/b.jpg', '/c.jpg'])]
        assert resolve_segments(segments, {'/b.jpg'}, {}) == []

    def test_a_forced_set_is_added(self):
        forced = {'k1': [(PANORAMA, '/x.jpg'), (PANORAMA, '/y.jpg')]}
        resolved = resolve_segments([], set(), forced)
        assert len(resolved) == 1
        assert resolved[0]['paths'] == ['/x.jpg', '/y.jpg']
        assert resolved[0]['source'] == 'override'

    def test_two_forced_sets_stay_separate(self):
        """Grouping by kind alone would merge every forced set of one kind."""
        forced = {'k1': [(PANORAMA, '/x.jpg'), (PANORAMA, '/y.jpg')],
                  'k2': [(PANORAMA, '/m.jpg'), (PANORAMA, '/n.jpg')]}
        resolved = resolve_segments([], set(), forced)
        assert sorted(len(s['paths']) for s in resolved) == [2, 2]

    def test_a_forced_set_displaces_an_overlapping_detected_set(self):
        segments = [self._segment(['/a.jpg', '/b.jpg', '/c.jpg'])]
        forced = {'k1': [(HDR_PANORAMA, '/b.jpg'), (HDR_PANORAMA, '/z.jpg')]}
        resolved = resolve_segments(segments, set(), forced)
        assert len(resolved) == 1
        assert resolved[0]['kind'] == HDR_PANORAMA


# ---------------------------------------------------------------------------
# End-to-end over a real SQLite database
# ---------------------------------------------------------------------------

def _seed_sweep(db_path, frames=12, start_second=0, degrees_per_frame=5.0):
    """A synthetic panorama written into a real schema, thumbnails included."""
    from db.schema import init_database
    init_database(str(db_path))
    rows = []
    for index in range(frames):
        frame = _rotated(_textured_frame(), degrees_per_frame * index)
        ok, buffer = cv2.imencode('.jpg', frame)
        assert ok
        rows.append((
            f'/pano{index}.jpg', f'pano{index}.jpg',
            f'2025:04:15 12:00:{start_second + index:02d}',
            'Canon EOS R6', 24.0, 8.0, '0.005', 100, buffer.tobytes(),
        ))
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO photos (path, filename, date_taken, camera_model, focal_length, "
            "f_stop, shutter_speed, iso, thumbnail) VALUES (?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
    return [row[0] for row in rows]


class TestDetectPanoramasEndToEnd:
    def test_labels_a_synthetic_sweep(self, tmp_path):
        db = tmp_path / 'pano.db'
        _seed_sweep(db)
        result = detect_panoramas(str(db))
        assert result['sets'] == 1
        with sqlite3.connect(db) as conn:
            kinds = {row[0] for row in conn.execute(
                "SELECT sequence_kind FROM photos WHERE sequence_kind IS NOT NULL")}
        assert kinds == {PANORAMA}

    def test_is_idempotent(self, tmp_path):
        db = tmp_path / 'pano.db'
        _seed_sweep(db)
        assert detect_panoramas(str(db)) == detect_panoramas(str(db))

    def test_leaves_bracket_labels_alone(self, tmp_path):
        db = tmp_path / 'pano.db'
        _seed_sweep(db)
        with sqlite3.connect(db) as conn:
            conn.execute("INSERT INTO photos (path, filename, sequence_group_id, sequence_kind) "
                         "VALUES ('/b.jpg', 'b.jpg', 1, 'bracket')")
            conn.commit()

        detect_panoramas(str(db))

        with sqlite3.connect(db) as conn:
            kind = conn.execute(
                "SELECT sequence_kind FROM photos WHERE path = '/b.jpg'").fetchone()[0]
        assert kind == 'bracket'

    def test_disabled_in_config_writes_nothing(self, tmp_path, monkeypatch):
        db = tmp_path / 'pano.db'
        _seed_sweep(db)
        monkeypatch.setattr(
            'config.ScoringConfig.get_panorama_detection_settings',
            lambda self: {'enabled': False})
        assert detect_panoramas(str(db)) is None

    def test_an_override_survives_a_second_run(self, tmp_path):
        """The bug the side table exists to prevent.

        A correction written into `photos.sequence_*` would be erased by the
        clear-and-rewrite at the start of the very next pass.
        """
        db = tmp_path / 'pano.db'
        paths = _seed_sweep(db)
        detect_panoramas(str(db))
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO photo_sequence_overrides "
                "(photo_path, sequence_kind, override_group_key, source) VALUES (?,?,?,?)",
                [(path, None, None, 'user') for path in paths])
            conn.commit()

        detect_panoramas(str(db))
        detect_panoramas(str(db))

        with sqlite3.connect(db) as conn:
            labelled = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE sequence_kind IS NOT NULL").fetchone()[0]
        assert labelled == 0

    def test_a_forced_set_survives_a_second_run(self, tmp_path):
        db = tmp_path / 'pano.db'
        _seed_sweep(db, frames=4)
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO photo_sequence_overrides "
                "(photo_path, sequence_kind, override_group_key, source) VALUES (?,?,?,?)",
                [('/pano0.jpg', PANORAMA, 'g1', 'user'),
                 ('/pano1.jpg', PANORAMA, 'g1', 'user')])
            conn.commit()

        detect_panoramas(str(db))
        result = detect_panoramas(str(db))

        assert result['sets'] == 1
        with sqlite3.connect(db) as conn:
            labelled = {row[0] for row in conn.execute(
                "SELECT path FROM photos WHERE sequence_kind IS NOT NULL")}
        assert labelled == {'/pano0.jpg', '/pano1.jpg'}


class TestLoadOverrides:
    def test_splits_suppressions_from_forced_sets(self, tmp_path):
        from db.schema import init_database
        db = tmp_path / 'o.db'
        init_database(str(db))
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO photo_sequence_overrides "
                "(photo_path, sequence_kind, override_group_key, source) VALUES (?,?,?,?)",
                [('/a.jpg', None, None, 'user'),
                 ('/b.jpg', PANORAMA, 'g1', 'user'),
                 ('/c.jpg', PANORAMA, 'g1', 'user')])
            conn.commit()
            conn.row_factory = sqlite3.Row
            suppressed, forced = load_overrides(conn)

        assert suppressed == {'/a.jpg'}
        assert sorted(path for _, path in forced['g1']) == ['/b.jpg', '/c.jpg']

    def test_a_bracket_forced_path_joins_this_passs_suppression_set(self, tmp_path):
        """A path the user marked `bracket` must never become a panorama
        candidate -- it folds into `suppressed`, never into `forced`, which
        stays scoped to this pass's own two kinds (#162 finding, A3)."""
        from db.schema import init_database
        db = tmp_path / 'o2.db'
        init_database(str(db))
        with sqlite3.connect(db) as conn:
            conn.execute(
                "INSERT INTO photo_sequence_overrides "
                "(photo_path, sequence_kind, override_group_key, source) "
                "VALUES ('/br.jpg', ?, 'gbr', 'user')", (BRACKET,))
            conn.commit()
            conn.row_factory = sqlite3.Row
            suppressed, forced = load_overrides(conn)

        assert '/br.jpg' in suppressed
        assert forced == {}


class TestBracketOverridesSuppressPanorama:
    """Manual wins symmetrically: a bracket mark on a settled panorama's own
    members must drop that panorama -- the direction #162 adds on top of the
    reverse (a panorama mark can never resurrect bracket-marked frames)."""

    def test_a_bracket_mark_on_a_settled_panoramas_members_removes_it(self, tmp_path):
        db = tmp_path / 'pano.db'
        paths = _seed_sweep(db)
        detect_panoramas(str(db))
        with sqlite3.connect(db) as conn:
            set_sequence_overrides(conn, paths[:3], BRACKET)
            conn.commit()

        detect_panoramas(str(db))

        with sqlite3.connect(db) as conn:
            labelled = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE sequence_kind IN (?, ?)",
                (PANORAMA, HDR_PANORAMA)).fetchone()[0]
        assert labelled == 0


class TestParallelAnalysis:
    """The pool path is a different code path from the serial one.

    Every other end-to-end test seeds a single run, which takes the serial
    branch, so without this the worker initializer, the read-only connections it
    opens and the picklability of a candidate run were never executed at all.
    """

    def test_two_sweeps_are_found_with_a_worker_pool(self, tmp_path):
        db = tmp_path / 'pano.db'
        _seed_sweep(db, frames=12, start_second=0)
        rows = []
        for index in range(12):
            frame = _rotated(_textured_frame(seed=21), 5.0 * index)
            ok, buffer = cv2.imencode('.jpg', frame)
            assert ok
            rows.append((f'/second{index}.jpg', f'second{index}.jpg',
                         f'2025:04:15 12:05:{index:02d}', 'Canon EOS R6', 24.0,
                         8.0, '0.005', 100, buffer.tobytes()))
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO photos (path, filename, date_taken, camera_model, focal_length, "
                "f_stop, shutter_speed, iso, thumbnail) VALUES (?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()

        result = detect_panoramas(str(db))

        assert result['sets'] == 2
        with sqlite3.connect(db) as conn:
            leads = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE is_sequence_lead = 1").fetchone()[0]
        assert leads == 2


class TestStaticRunProbe:
    """The cheap gate that abandons a run before any geometry.

    Untested, this is the module's most dangerous failure: an over-eager probe
    discards every panorama and the rest of the suite still passes, because
    every other fixture here is a moving run that never reaches `return True`.
    Both `probe_stride` and `probe_min_drift` are editable through the config
    endpoint, so a bad value has a direct route in.
    """

    def _probe(self, db, paths, **overrides):
        from utils.panorama import _is_static_run, _sift_and_matcher
        settings = _settings(**overrides)
        sift, matcher = _sift_and_matcher(settings)
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        try:
            return _is_static_run(conn, paths, sift, matcher, settings, {})
        finally:
            conn.close()

    def test_a_long_static_burst_is_abandoned(self, tmp_path):
        db = tmp_path / 'pano.db'
        paths = _seed_sweep(db, frames=12, degrees_per_frame=0.0)
        assert self._probe(db, paths) is True
        assert detect_panoramas(str(db))['sets'] == 0

    def test_a_moving_sweep_is_not_abandoned(self, tmp_path):
        db = tmp_path / 'pano.db'
        paths = _seed_sweep(db, frames=12, degrees_per_frame=5.0)
        assert self._probe(db, paths) is False

    def test_a_run_shorter_than_the_stride_is_never_abandoned(self, tmp_path):
        """Too short to probe means unknown, not static."""
        db = tmp_path / 'pano.db'
        paths = _seed_sweep(db, frames=4, degrees_per_frame=0.0)
        assert self._probe(db, paths) is False

    def test_a_probe_that_cannot_match_escalates(self, tmp_path):
        """A failed match means large motion, so the run must be analysed.

        Abandoning it would silently drop exactly the fast sweeps that have no
        overlap left at the probe stride.
        """
        from db.schema import init_database
        db = tmp_path / 'pano.db'
        init_database(str(db))
        rows = []
        for index in range(12):
            frame = _textured_frame(seed=100 + index)
            ok, buffer = cv2.imencode('.jpg', frame)
            assert ok
            rows.append((f'/x{index}.jpg', f'x{index}.jpg',
                         f'2025:04:15 12:00:{index:02d}', 'Canon EOS R6', 24.0,
                         8.0, '0.005', 100, buffer.tobytes()))
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO photos (path, filename, date_taken, camera_model, focal_length, "
                "f_stop, shutter_speed, iso, thumbnail) VALUES (?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()

        assert self._probe(db, [r[0] for r in rows]) is False


class TestConfigGuards:
    """A hand-edited config bypasses the API's bounds entirely."""

    def test_a_zero_probe_stride_does_not_hang(self):
        """`probe_stride: 0` is a plausible "disable it" edit.

        The sibling keys use 0 as a "pick for me" sentinel, and the loop it
        drives advances by the stride -- so 0 spun forever while the scan held
        the library lock, which no exception guard catches.
        """
        from utils.panorama import _is_static_run

        calls = []

        class _Sift:
            def detectAndCompute(self, *_):
                return None, None

        def fake_features(conn, path, sift, cache):
            calls.append(path)
            if len(calls) > 500:
                raise AssertionError("_is_static_run did not advance")
            return None

        import utils.panorama as module
        original = module._features
        module._features = fake_features
        try:
            assert _is_static_run(None, _paths(12), _Sift(), None,
                                  _settings(probe_stride=0), {}) is False
        finally:
            module._features = original

    def test_a_run_is_capped_so_the_feature_cache_stays_bounded(self, tmp_path):
        """Each cached frame holds ~220 KB per worker, and a run was unbounded."""
        from db.schema import init_database
        from utils.panorama import _load_candidates

        db = tmp_path / 'cap.db'
        init_database(str(db))
        rows = [(f'/f{i}.jpg', f'f{i}.jpg', f'2025:04:15 12:{i // 60:02d}:{i % 60:02d}',
                 'Canon EOS R6', 24.0) for i in range(60)]
        with sqlite3.connect(db) as conn:
            conn.executemany(
                "INSERT INTO photos (path, filename, date_taken, camera_model, focal_length) "
                "VALUES (?,?,?,?,?)", rows)
            conn.commit()
            conn.row_factory = sqlite3.Row
            runs = _load_candidates(conn, _settings(max_run_frames=10))

        assert runs, "expected the frames to form candidate runs"
        assert max(len(run) for run in runs) <= 10


class TestAdjacentSegments:
    """Where one sweep ends and the next begins.

    The two sets used to share their boundary frame: the segment was cut at
    `paths[index:end + 1]` and the next one resumed at step `end`, whose own
    first photo is `paths[end]`. Both then wrote it, and the later group won --
    so the earlier set served one frame fewer than it had reported.
    """

    def test_back_to_back_sweeps_share_no_frame(self):
        settings = _settings(min_frames=4)
        steps = [_step(dx=0.1) for _ in range(12)] + [_step(dx=-0.1) for _ in range(12)]
        segments = find_segments(_paths(25), steps, settings)

        assert len(segments) == 2
        claimed = [path for segment in segments for path in segment['paths']]
        assert len(claimed) == len(set(claimed)), f"a frame is in two sets: {claimed}"

    def test_a_rejected_stretch_still_yields_its_frames(self):
        """Only an *accepted* segment owns its last frame.

        A stretch that fails the drift floor claims nothing, so the scan must
        resume on it rather than past it -- otherwise the fix above would skip a
        frame that could still open a real sweep.
        """
        settings = _settings(min_frames=4, min_drift=0.5)
        steps = [_step(dx=0.02) for _ in range(4)] + [_step(dx=0.2) for _ in range(8)]
        segments = find_segments(_paths(13), steps, settings)

        assert len(segments) == 1
        assert segments[0]['paths'][-1] == '/p12.jpg'


def _run_photo(path, **columns):
    """A candidate-run photo dict carrying every field `run_signature` reads."""
    photo = {'path': path, 'date_taken': '2025:04:15 12:00:00', 'camera_model': 'Canon EOS R6',
             'focal_length': 24.0, 'f_stop': 8.0, 'shutter_speed': '0.005', 'iso': 100,
             'scanned_at': '2025-01-01T00:00:00', 'render_version': 1, 'thumbnail_bytes': 2500}
    photo.update(columns)
    return photo


class TestIncrementalPass:
    """Coverage stays whole-library; only the measuring is skipped."""

    def test_split_runs_measures_everything_without_a_cache(self):
        runs = [[_run_photo('/a.jpg')]]
        fresh, settled = split_runs(runs, None)
        assert (fresh, settled) == (runs, [])

    def test_split_runs_reuses_a_run_whose_signature_is_cached(self):
        settled_run = [_run_photo('/a.jpg'), _run_photo('/b.jpg')]
        fresh_run = [_run_photo('/c.jpg'), _run_photo('/d.jpg', scanned_at='2025-06-01T00:00:00')]
        cached = {run_signature(settled_run): []}
        fresh, settled = split_runs([settled_run, fresh_run], cached)
        assert fresh == [fresh_run]
        assert settled == [settled_run]

    def test_split_runs_reuses_a_run_of_unknown_age(self):
        """A NULL `scanned_at` is just another signature component.

        Treating it as "unknown, measure it" made 2,506 of 2,527 runs re-measure on
        every scan of a real library: the defect the content signature fixes.
        """
        unknown = [_run_photo('/a.jpg', scanned_at=None)]
        fresh, settled = split_runs([unknown], {run_signature(unknown): []})
        assert (fresh, settled) == ([], [unknown])

    def test_split_runs_measures_a_run_whose_signature_is_unknown(self):
        known = [_run_photo('/a.jpg')]
        unknown = [_run_photo('/b.jpg', scanned_at=None)]
        fresh, settled = split_runs([known, unknown], {run_signature(known): []})
        assert fresh == [unknown]
        assert settled == [known]

    def test_an_incremental_rerun_reproduces_the_full_pass(
            self, tmp_path, pass_overrides, measured_runs):
        db = tmp_path / 'pano.db'
        _seed_sweep(db)
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE photos SET scanned_at = '2025-04-15T12:30:00'")
            conn.commit()

        full = detect_panoramas(str(db))
        labels = _sequence_labels(db)
        assert len(measured_runs) == 1
        del measured_runs[:]

        assert detect_panoramas(str(db), incremental=True) == full
        assert _sequence_labels(db) == labels
        assert measured_runs == [], "the incremental pass must reuse the cached run"

    def test_a_library_whose_scanned_at_is_null_is_still_reused(
            self, tmp_path, pass_overrides, measured_runs):
        """The defect the signature fixes: a NULL `scanned_at` used to read as
        "unknown, measure it", so such runs were re-measured on every scan."""
        db = tmp_path / 'pano.db'
        _seed_sweep(db)
        with sqlite3.connect(db) as conn:
            assert conn.execute("SELECT count(*) FROM photos WHERE scanned_at IS NOT NULL"
                                ).fetchone()[0] == 0
        detect_panoramas(str(db), incremental=True)
        assert len(measured_runs) == 1, "the first pass has an empty cache"
        labels = _sequence_labels(db)
        del measured_runs[:]

        detect_panoramas(str(db), incremental=True)

        assert measured_runs == []
        assert _sequence_labels(db) == labels

    def test_editing_a_threshold_invalidates_the_run_cache(self, tmp_path, pass_overrides):
        """A cached answer belongs to the thresholds it was measured under."""
        db = tmp_path / 'pano.db'
        _seed_sweep(db)
        with sqlite3.connect(db) as conn:
            conn.execute("UPDATE photos SET scanned_at = '2025-04-15T12:30:00'")
            conn.commit()
        detect_panoramas(str(db))

        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            from utils.panorama import _load_candidates
            signatures = {run_signature(run) for run in _load_candidates(conn, _settings())}
            assert set(read_run_cache(conn, _settings())) == signatures
            assert len(signatures) == 1
            assert read_run_cache(conn, _settings(min_drift=0.9)) == {}


class TestIncrementalRunnerVsBracketPass:
    """The bracket pass runs first and rewrites `sequence_kind` by path, including
    rows currently labelled a panorama kind. The panorama pass must therefore never
    read its own labels back: a settled HDR panorama is re-derived from the cached
    raw detector result on every pass, so the rewrite cannot make it vanish.
    """

    def _seed_hdr_panorama(self, db_path):
        """9 frames, 3 positions of 3 brackets each (base-first: 0/-2/+2).

        No thumbnails needed: the run under test is "settled", so its raw detector
        result is cached rather than re-measured by the CV pipeline. Distinct
        `phash` per position keeps the bracket pass from stitching all three
        positions into one run, matching what three real camera positions would
        do -- and each position's own three frames share a `phash`, keeping them
        inside one bracket run.
        """
        from db.schema import init_database
        init_database(str(db_path))
        rows = []
        second = 0
        position_phashes = ['0' * 16, 'f' * 16, 'a' * 16]
        for position in range(3):
            phash = position_phashes[position]
            # Base-first order (0, -2, +2), the common Canon/Sony/Nikon default.
            for shutter in ('0.005', '0.02', '0.00125'):  # EV 0, -2, +2
                rows.append((
                    f'/pos{position}_f{second}.jpg', f'pos{position}_f{second}.jpg',
                    f'2025:04:15 12:00:{second:02d}', 'Canon EOS R6', 24.0,
                    8.0, shutter, 100, phash, '2025-01-01T00:00:00',
                ))
                second += 1
        with sqlite3.connect(db_path) as conn:
            conn.executemany(
                "INSERT INTO photos (path, filename, date_taken, camera_model, "
                "focal_length, f_stop, shutter_speed, iso, phash, scanned_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            conn.commit()
        return [row[0] for row in rows]

    def _mark_settled_hdr_panorama(self, db_path, paths):
        """Cache the HDR panorama as the raw detector result of its run.

        Takes the run `_load_candidates` returns for the seeded rows and stores the
        HDR segment under that run's real signature, so no thumbnails are needed
        and nothing is hand-written into `photos`.
        """
        from utils.panorama import _load_candidates

        settings = _pass_settings()
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            runs = _load_candidates(conn, settings)
            assert len(runs) == 1 and [p['path'] for p in runs[0]] == paths
            write_run_cache(conn, settings, {run_signature(runs[0]): [{
                'paths': paths, 'kind': HDR_PANORAMA, 'axis': 'x', 'drift': 0.87, 'ortho': 0.0}]})
            conn.commit()

    def test_an_incremental_rerun_does_not_lose_a_settled_hdr_panorama(
            self, tmp_path, pass_overrides):
        from facet import detect_all_sequences

        db = tmp_path / 'pano.db'
        paths = self._seed_hdr_panorama(db)
        self._mark_settled_hdr_panorama(db, paths)

        detect_all_sequences(str(db), None, incremental=True, contain_failure=False)

        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            kinds = {row['path']: row['sequence_kind'] for row in conn.execute(
                "SELECT path, sequence_kind FROM photos WHERE path IN ({})".format(
                    ','.join('?' * len(paths))), paths)}
        assert set(kinds.values()) == {'hdr_panorama'}, (
            f"the settled HDR panorama must survive the bracket pass, got {kinds}")

    def _kinds(self, db_path, paths):
        with sqlite3.connect(db_path) as conn:
            return {row[0]: row[1] for row in conn.execute(
                "SELECT path, sequence_kind FROM photos WHERE path IN ({})".format(
                    ','.join('?' * len(paths))), paths)}

    def test_a_panorama_pass_that_dies_after_the_bracket_pass_loses_nothing(
            self, tmp_path, pass_overrides, monkeypatch):
        """B2: the bracket pass commits and relabels the frames `bracket`, then the
        panorama pass dies. The next pass, with no explicit re-run, must restore the
        HDR panorama: its raw result outlives the labels because it lives in the
        cache, not in `photos`."""
        from facet import detect_all_sequences
        import utils.panorama as module

        db = tmp_path / 'pano.db'
        paths = self._seed_hdr_panorama(db)
        self._mark_settled_hdr_panorama(db, paths)
        detect_all_sequences(str(db), None, incremental=True, contain_failure=False)
        assert set(self._kinds(db, paths).values()) == {HDR_PANORAMA}

        def boom(*args, **kwargs):
            raise RuntimeError("killed after the bracket pass")

        with monkeypatch.context() as patched:
            patched.setattr(module, 'detect_panoramas', boom)
            detect_all_sequences(str(db), None, incremental=True, contain_failure=True)
        assert set(self._kinds(db, paths).values()) == {BRACKET}, (
            "precondition: the bracket pass alone relabels the HDR frames as brackets")

        detect_all_sequences(str(db), None, incremental=True, contain_failure=False)
        assert set(self._kinds(db, paths).values()) == {HDR_PANORAMA}

    def test_a_panorama_pass_killed_inside_the_label_transaction_keeps_its_measuring(
            self, tmp_path, pass_overrides, monkeypatch, measured_runs):
        """B2 again, one step later: the label transaction itself dies (rolled back).
        The raw results measured in the same pass were committed first, so the next
        pass restores the HDR panorama AND measures nothing."""
        from facet import detect_all_sequences
        import utils.panorama as module

        db = tmp_path / 'pano.db'
        paths = self._seed_hdr_panorama(db)
        self._mark_settled_hdr_panorama(db, paths)
        sweep = _add_sweep(db, 'sweep', minute=30, seed=21)

        original = module._apply_labels

        def apply_then_die(conn, found):
            original(conn, found)
            raise RuntimeError("killed inside the label transaction")

        with monkeypatch.context() as patched:
            patched.setattr(module, '_apply_labels', apply_then_die)
            detect_all_sequences(str(db), None, incremental=True, contain_failure=True)
        assert measured_runs == [sweep], "only the unseen sweep is measured"
        assert set(self._kinds(db, paths).values()) == {BRACKET}
        assert not any(self._kinds(db, sweep).values()), "the labels rolled back"
        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            assert len(read_run_cache(conn, _pass_settings())) == 2, (
                "the sweep's raw result was committed before the label transaction")

        del measured_runs[:]
        detect_all_sequences(str(db), None, incremental=True, contain_failure=False)
        assert measured_runs == [], "nothing is measured twice"
        assert set(self._kinds(db, paths).values()) == {HDR_PANORAMA}
        assert set(self._kinds(db, sweep).values()) == {PANORAMA}


class TestOverrideApplied:
    """`sequence_override` says a correction exists; `applied_at` says it landed."""

    def test_a_pass_stamps_the_corrections_it_applied(self, tmp_path):
        db = tmp_path / 'pano.db'
        paths = _seed_sweep(db)
        with sqlite3.connect(db) as conn:
            set_sequence_overrides(conn, paths[:3], None)
            conn.commit()
            assert conn.execute("SELECT count(*) FROM photo_sequence_overrides "
                                "WHERE applied_at IS NULL").fetchone()[0] == 3

        detect_panoramas(str(db))

        with sqlite3.connect(db) as conn:
            pending = conn.execute("SELECT count(*) FROM photo_sequence_overrides "
                                   "WHERE applied_at IS NULL").fetchone()[0]
            kept = conn.execute("SELECT count(*) FROM photo_sequence_overrides").fetchone()[0]
        assert pending == 0, "the run applied them, so none is still waiting on one"
        assert kept == 3, "the correction itself must survive -- it is what keeps applying"

    def test_correcting_again_marks_it_pending_once_more(self, tmp_path):
        db = tmp_path / 'pano.db'
        paths = _seed_sweep(db)
        with sqlite3.connect(db) as conn:
            set_sequence_overrides(conn, paths[:3], None)
            conn.commit()
        detect_panoramas(str(db))

        with sqlite3.connect(db) as conn:
            set_sequence_overrides(conn, paths[:3], PANORAMA)
            conn.commit()
            pending = conn.execute("SELECT count(*) FROM photo_sequence_overrides "
                                   "WHERE applied_at IS NULL").fetchone()[0]
        assert pending == 3

    def test_the_panorama_pass_leaves_a_pending_bracket_override_pending(self, tmp_path):
        """The panorama pass owns only `KINDS` (panorama/hdr_panorama). A pending
        `bracket` override belongs to the bracket pass, which has not run --
        `detect_panoramas` alone must not stamp it as applied."""
        db = tmp_path / 'pano.db'
        paths = _seed_sweep(db)
        with sqlite3.connect(db) as conn:
            set_sequence_overrides(conn, paths[:3], BRACKET)
            conn.commit()

        detect_panoramas(str(db))

        with sqlite3.connect(db) as conn:
            pending = conn.execute(
                "SELECT count(*) FROM photo_sequence_overrides "
                "WHERE applied_at IS NULL AND sequence_kind = ?", (BRACKET,)).fetchone()[0]
        assert pending == 3, "the bracket pass owns this override -- the panorama pass must not touch it"


# ---------------------------------------------------------------------------
# Step 5: the matcher is exact, so two passes agree
# ---------------------------------------------------------------------------

def _spotted_pair(seed=11, shift=37):
    """Two 480x640 frames of one scene, the second shifted `shift` px sideways.

    Blurred noise spotted with 120 circles: enough structure for 400 SIFT
    features, and exactly the case where a randomized kd-tree returned a different
    match set on every call for identical descriptors.
    """
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.integers(0, 255, (480, 640), dtype=np.uint8), (0, 0), 3)
    for _ in range(120):
        cv2.circle(image, (int(rng.integers(0, 640)), int(rng.integers(0, 480))),
                   int(rng.integers(4, 20)), int(rng.integers(0, 255)), -1)
    moved = cv2.warpAffine(image, np.float32([[1, 0, shift], [0, 1, 0]]), (640, 480))
    return image, moved


class TestMatcherIsDeterministic:
    def _distinct_results(self, calls=12):
        from utils.panorama import _sift_and_matcher, _translation
        settings = _settings()
        base, moved = _spotted_pair()
        results = set()
        for _ in range(calls):
            sift, matcher = _sift_and_matcher(settings)

            def features(image, sift=sift):
                keypoints, descriptors = sift.detectAndCompute(image, None)
                return keypoints, descriptors, image.shape

            step = _translation(features(base), features(moved), matcher, settings)
            results.add((step['inliers'], step['dx'], step['dy']))
        return results

    def test_twelve_fresh_matchers_agree_in_a_pool_worker(self):
        """`cv2.setNumThreads(1)` is what `_init_worker` does."""
        threads = cv2.getNumThreads()
        cv2.setNumThreads(1)
        try:
            results = self._distinct_results()
        finally:
            cv2.setNumThreads(threads)
        assert len(results) == 1, f"identical descriptors gave {len(results)} results: {results}"

    def test_twelve_fresh_matchers_agree_on_the_serial_path(self):
        """`workers <= 1` runs in the main process and never touches the thread count."""
        results = self._distinct_results()
        assert len(results) == 1, f"identical descriptors gave {len(results)} results: {results}"
        (inliers, dx, _), = results
        assert inliers >= DEFAULTS['min_inliers']
        assert dx == pytest.approx(37 / 640, abs=0.01)

    def test_the_fingerprint_names_the_matcher(self, monkeypatch):
        """A future change of matcher must read as a cache miss without a config edit."""
        from utils.panorama import _settings_fingerprint
        before = _settings_fingerprint(_settings())
        monkeypatch.setattr('utils.panorama._MATCHER_ID', 'something-else')
        assert _settings_fingerprint(_settings()) != before

    def test_the_fingerprint_names_the_detector_revision(self, monkeypatch):
        from utils.panorama import _DETECTOR_REV, _settings_fingerprint
        before = _settings_fingerprint(_settings())
        monkeypatch.setattr('utils.panorama._DETECTOR_REV', _DETECTOR_REV + 1)
        assert _settings_fingerprint(_settings()) != before

    def test_workers_and_enabled_do_not_change_the_fingerprint(self):
        from utils.panorama import _settings_fingerprint
        assert (_settings_fingerprint(_settings(workers=1, enabled=True))
                == _settings_fingerprint(_settings(workers=7, enabled=False)))


class TestTwoPassesAgree:
    def _two_sweeps(self, tmp_path):
        db = tmp_path / 'pano.db'
        _seed_sweep(db)
        _add_sweep(db, 'second', minute=5, seed=21)
        return db

    def test_workers_one_workers_one_and_workers_four_label_every_row_identically(
            self, tmp_path, pass_overrides):
        """S5: the serial path twice and the process pool, ids included."""
        db = self._two_sweeps(tmp_path)

        pass_overrides['workers'] = 1
        assert detect_panoramas(str(db))['sets'] == 2
        first = _sequence_labels(db)
        detect_panoramas(str(db))
        second = _sequence_labels(db)
        pass_overrides['workers'] = 4
        detect_panoramas(str(db))
        pooled = _sequence_labels(db)

        assert len(first) == 24
        assert first == second == pooled


# ---------------------------------------------------------------------------
# Step 6: the content-keyed cache of each run's raw detector result
# ---------------------------------------------------------------------------

def _add_sweep(db_path, prefix, minute, seed=7, frames=12, degrees_per_frame=5.0):
    """A synthetic sweep at 12:`minute`, apart from every other in time and path."""
    rows = []
    for index in range(frames):
        frame = _rotated(_textured_frame(seed=seed), degrees_per_frame * index)
        ok, buffer = cv2.imencode('.jpg', frame)
        assert ok
        rows.append((f'/{prefix}{index}.jpg', f'{prefix}{index}.jpg',
                     f'2025:04:15 12:{minute:02d}:{index:02d}', 'Canon EOS R6', 24.0,
                     8.0, '0.005', 100, '2025-01-01T00:00:00', buffer.tobytes()))
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT INTO photos (path, filename, date_taken, camera_model, focal_length, "
            "f_stop, shutter_speed, iso, scanned_at, thumbnail) VALUES (?,?,?,?,?,?,?,?,?,?)",
            rows)
        conn.commit()
    return [row[0] for row in rows]


def _connect(db):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    return conn


def _candidate_runs(db, **overrides):
    from utils.panorama import _load_candidates
    with _connect(db) as conn:
        return _load_candidates(conn, _pass_settings(**overrides))


def _cached_signatures(db):
    with _connect(db) as conn:
        return set(read_run_cache(conn, _pass_settings()))


@pytest.fixture(scope='module')
def warm_library(tmp_path_factory):
    """Three sweeps, A (`/pano*`), B (`/b*`) and C (`/c*`), after one full pass.

    Built once; every test copies it. All `scanned_at` are old and non-NULL, so a
    pass that still keyed its reuse on that column would treat every run as settled.
    """
    from config import ScoringConfig
    db = tmp_path_factory.mktemp('warm') / 'warm.db'
    _seed_sweep(db)
    _add_sweep(db, 'b', minute=10, seed=21)
    _add_sweep(db, 'c', minute=20, seed=33)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE photos SET scanned_at = '2025-01-01T00:00:00'")
        conn.commit()
    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(ScoringConfig, 'get_panorama_detection_settings',
                        lambda self: {'workers': 1})
        assert detect_panoramas(str(db))['sets'] == 3
    return db


class _Library:
    """A private copy of `warm_library` and the three runs' paths."""

    A = [f'/pano{i}.jpg' for i in range(12)]
    B = [f'/b{i}.jpg' for i in range(12)]
    C = [f'/c{i}.jpg' for i in range(12)]

    def __init__(self, db):
        self.db = db

    def labels(self):
        return _sequence_labels(self.db)

    def labelled_paths(self):
        return {row[0] for row in self.labels()}

    def execute(self, sql, params=()):
        with sqlite3.connect(self.db) as conn:
            conn.execute(sql, params)
            conn.commit()


@pytest.fixture()
def library(tmp_path, warm_library):
    target = tmp_path / 'copy.db'
    source = sqlite3.connect(warm_library)
    destination = sqlite3.connect(target)
    try:
        source.backup(destination)
    finally:
        destination.close()
        source.close()
    return _Library(target)


def _sorted(runs):
    return sorted(runs)


class TestRunSignature:
    def test_is_32_hex_characters_and_stable(self):
        run = [_run_photo('/a.jpg'), _run_photo('/b.jpg')]
        signature = run_signature(run)
        assert len(signature) == 32 and int(signature, 16) >= 0
        assert run_signature([dict(p) for p in run]) == signature

    @pytest.mark.parametrize('column,value', [
        ('path', '/z.jpg'), ('date_taken', '2025:04:15 12:00:01'), ('camera_model', 'Sony'),
        ('focal_length', 50.0), ('f_stop', 5.6), ('shutter_speed', '0.01'), ('iso', 200),
        ('scanned_at', '2026-01-01T00:00:00'), ('render_version', 2), ('thumbnail_bytes', 2501),
    ])
    def test_every_component_is_hashed(self, column, value):
        run = [_run_photo('/a.jpg'), _run_photo('/b.jpg')]
        second = _run_photo(value) if column == 'path' else _run_photo('/b.jpg', **{column: value})
        assert run_signature([run[0], second]) != run_signature(run)

    def test_member_order_is_part_of_the_signature(self):
        first, second = _run_photo('/a.jpg'), _run_photo('/b.jpg')
        assert run_signature([first, second]) != run_signature([second, first])


class TestRunCacheStorage:
    def test_round_trips_segments_and_an_empty_result(self, tmp_path):
        from db.schema import init_database
        db = tmp_path / 'c.db'
        init_database(str(db))
        segment = {'paths': ['/a.jpg', '/b.jpg'], 'kind': HDR_PANORAMA, 'axis': 'y',
                   'drift': 0.8721, 'ortho': 0.0123}
        with _connect(db) as conn:
            write_run_cache(conn, _pass_settings(), {'sig-hit': [segment], 'sig-empty': []})
            assert read_run_cache(conn, _pass_settings()) == {'sig-hit': [segment],
                                                              'sig-empty': []}

    def test_a_row_of_another_fingerprint_is_not_returned(self, tmp_path):
        from db.schema import init_database
        db = tmp_path / 'c.db'
        init_database(str(db))
        with _connect(db) as conn:
            write_run_cache(conn, _pass_settings(), {'sig': []})
            assert read_run_cache(conn, _pass_settings(min_drift=0.9)) == {}

    def test_prune_deletes_only_unreferenced_rows(self, tmp_path):
        from db.schema import init_database
        from utils.panorama import prune_run_cache
        db = tmp_path / 'c.db'
        init_database(str(db))
        with _connect(db) as conn:
            write_run_cache(conn, _pass_settings(), {'keep': [], 'drop': []})
            assert prune_run_cache(conn, ['keep', 'unrelated']) == 1
            assert set(read_run_cache(conn, _pass_settings())) == {'keep'}


class TestOverridesAreRederivedFromTheCache:
    """B1: labels come from the cached raw result plus the CURRENT overrides, never
    from what `photos` happens to hold, so removing a correction takes effect on the
    scan path with nothing re-measured."""

    def _incremental(self, library, measured_runs):
        del measured_runs[:]
        detect_panoramas(str(library.db), incremental=True)
        assert measured_runs == [], "every run is unchanged, so none may be measured"

    def _full_labels(self, library):
        detect_panoramas(str(library.db))
        return library.labels()

    def test_clearing_a_forced_override_unforces_its_set(
            self, library, pass_overrides, measured_runs):
        baseline = library.labels()
        forced = library.A[:3]
        with sqlite3.connect(library.db) as conn:
            set_sequence_overrides(conn, forced, PANORAMA)
            conn.commit()

        self._incremental(library, measured_runs)
        assert library.labelled_paths() == set(forced) | set(library.B) | set(library.C), (
            "the forced set replaces the detected one")

        with sqlite3.connect(library.db) as conn:
            clear_sequence_overrides(conn, forced)
            conn.commit()
        self._incremental(library, measured_runs)
        assert library.labels() == baseline, "the detector's set is back, ids included"
        assert self._full_labels(library) == baseline

    def test_clearing_a_suppression_brings_the_detected_set_back(
            self, library, pass_overrides, measured_runs):
        baseline = library.labels()
        suppressed = library.A[:3]
        with sqlite3.connect(library.db) as conn:
            set_sequence_overrides(conn, suppressed, None)
            conn.commit()

        self._incremental(library, measured_runs)
        assert library.labelled_paths() == set(library.B) | set(library.C)

        with sqlite3.connect(library.db) as conn:
            clear_sequence_overrides(conn, suppressed)
            conn.commit()
        self._incremental(library, measured_runs)
        assert library.labels() == baseline
        assert self._full_labels(library) == baseline

    def test_a_forced_set_spanning_two_runs_leaves_no_partial_set(
            self, library, pass_overrides, measured_runs):
        baseline = library.labels()
        forced = library.A[-3:] + library.B[:3]
        with sqlite3.connect(library.db) as conn:
            set_sequence_overrides(conn, forced, PANORAMA)
            conn.commit()

        self._incremental(library, measured_runs)
        assert library.labelled_paths() == set(forced) | set(library.C), (
            "both detected sets overlap the forced one and give way to it")

        with sqlite3.connect(library.db) as conn:
            clear_sequence_overrides(conn, forced)
            conn.commit()
        self._incremental(library, measured_runs)
        assert library.labels() == baseline, "no partial set remains"
        assert self._full_labels(library) == baseline

    def test_a_bracket_mark_on_a_cached_panoramas_members_removes_it(
            self, library, pass_overrides, measured_runs):
        with sqlite3.connect(library.db) as conn:
            set_sequence_overrides(conn, library.A[:3], BRACKET)
            conn.commit()
        del measured_runs[:]

        detect_panoramas(str(library.db), incremental=True)

        assert measured_runs == []
        assert library.labelled_paths() == set(library.B) | set(library.C)


class TestSignatureInvalidation:
    """A run is re-measured when any one component of its signature changes."""

    def _measure_after(self, library, pass_overrides, measured_runs):
        """Run the incremental pass; return (runs measured, runs that had no cache row)."""
        before = _cached_signatures(library.db)
        expected = [[p['path'] for p in run]
                    for run in _candidate_runs(library.db) if run_signature(run) not in before]
        del measured_runs[:]
        detect_panoramas(str(library.db), incremental=True)
        return _sorted(measured_runs), _sorted(expected)

    @pytest.mark.parametrize('name,sql,params', [
        ('f_stop', "UPDATE photos SET f_stop = 5.6 WHERE path = '/pano4.jpg'", ()),
        ('shutter_speed', "UPDATE photos SET shutter_speed = '0.01' WHERE path = '/pano4.jpg'", ()),
        ('iso', "UPDATE photos SET iso = 400 WHERE path = '/pano4.jpg'", ()),
        ('scanned_at',
         "UPDATE photos SET scanned_at = '2026-02-02T00:00:00' WHERE path = '/pano4.jpg'", ()),
        ('render_version', "UPDATE photos SET render_version = 2 WHERE path = '/pano4.jpg'", ()),
        ('date_taken_by_one_second',
         "UPDATE photos SET date_taken = '2025:04:15 12:00:12' WHERE path = '/pano11.jpg'", ()),
        ('path_renamed_in_place',
         "UPDATE photos SET path = '/pano4a.jpg', filename = 'pano4a.jpg' "
         "WHERE path = '/pano4.jpg'", ()),
    ])
    def test_a_component_that_keeps_the_run_intact_measures_exactly_that_run(
            self, library, pass_overrides, measured_runs, name, sql, params):
        library.execute(sql, params)

        measured, expected = self._measure_after(library, pass_overrides, measured_runs)

        assert len(expected) == 1 and all(path.startswith('/pano') for path in expected[0])
        assert len(expected[0]) == 12, "the run is intact"
        assert measured == expected, f"{name}: only run A changed"

    def test_a_thumbnail_of_another_length_measures_exactly_that_run(
            self, library, pass_overrides, measured_runs):
        with sqlite3.connect(library.db) as conn:
            blob = conn.execute("SELECT thumbnail FROM photos WHERE path = '/pano4.jpg'"
                                ).fetchone()[0]
            # Python-side: `thumbnail || x'00'` in SQL would turn the BLOB into TEXT.
            conn.execute("UPDATE photos SET thumbnail = ? WHERE path = '/pano4.jpg'",
                         (blob + b'\x00',))
            conn.commit()

        measured, expected = self._measure_after(library, pass_overrides, measured_runs)

        assert len(expected) == 1 and len(expected[0]) == 12
        assert measured == expected

    @pytest.mark.parametrize('name,sql', [
        ('camera_model',
         "UPDATE photos SET camera_model = 'Sony ILCE-7M4' WHERE path = '/pano0.jpg'"),
        ('focal_length', "UPDATE photos SET focal_length = 50.0 WHERE path = '/pano0.jpg'"),
        ('date_taken_beyond_the_gap',
         "UPDATE photos SET date_taken = '2025:04:15 11:40:00' WHERE path = '/pano0.jpg'"),
    ])
    def test_a_component_that_changes_membership_measures_only_the_runs_it_reshapes(
            self, library, pass_overrides, measured_runs, name, sql):
        library.execute(sql)

        measured, expected = self._measure_after(library, pass_overrides, measured_runs)

        assert expected, "the frames left behind form a run nobody has measured"
        assert all(path.startswith('/pano') for run in expected for path in run)
        assert measured == expected, f"{name}: B and C are untouched and must be reused"
        assert not any(path.startswith(('/b', '/c')) for run in measured for path in run)

    def test_a_thumbnail_swapped_for_another_image_of_the_same_length_is_reused(
            self, library, pass_overrides, measured_runs):
        """The documented limit (Q2), pinned so it cannot become a surprise.

        Same byte length, same `scanned_at`, same `render_version`: the signature
        cannot see it. A rescan rewrites `scanned_at`, and the explicit
        `--detect-panoramas` always re-measures everything."""
        with sqlite3.connect(library.db) as conn:
            blob = conn.execute("SELECT thumbnail FROM photos WHERE path = '/pano4.jpg'"
                                ).fetchone()[0]
            changed = bytearray(blob)
            changed[len(changed) // 2] ^= 0xFF
            conn.execute("UPDATE photos SET thumbnail = ? WHERE path = '/pano4.jpg'",
                         (bytes(changed),))
            conn.commit()
        del measured_runs[:]

        detect_panoramas(str(library.db), incremental=True)

        assert measured_runs == []

    def test_a_full_pass_measures_everything_regardless_of_the_cache(
            self, library, pass_overrides, measured_runs):
        del measured_runs[:]
        detect_panoramas(str(library.db))
        assert _sorted(measured_runs) == _sorted(
            [[p['path'] for p in run] for run in _candidate_runs(library.db)])
        assert len(measured_runs) == 3

    def test_a_threshold_edit_measures_every_run(self, library, pass_overrides, measured_runs):
        pass_overrides['min_drift'] = 0.9
        del measured_runs[:]
        detect_panoramas(str(library.db), incremental=True)
        assert len(measured_runs) == 3

    def test_a_workers_edit_measures_nothing(self, library, pass_overrides, measured_runs):
        pass_overrides['workers'] = 3
        del measured_runs[:]
        detect_panoramas(str(library.db), incremental=True)
        assert measured_runs == []

    @pytest.mark.parametrize('constant,value', [('_DETECTOR_REV', 2), ('_MATCHER_ID', 'flann')])
    def test_a_detector_or_matcher_revision_measures_every_run(
            self, library, pass_overrides, measured_runs, monkeypatch, constant, value):
        monkeypatch.setattr(f'utils.panorama.{constant}', value)
        del measured_runs[:]
        detect_panoramas(str(library.db), incremental=True)
        assert len(measured_runs) == 3

    def test_corrupt_cache_rows_are_measured_and_never_raise(
            self, library, pass_overrides, measured_runs):
        runs = _candidate_runs(library.db)
        by_prefix = {run[0]['path'][:2]: run_signature(run) for run in runs}
        assert len(by_prefix) == 3
        with sqlite3.connect(library.db) as conn:
            conn.execute("UPDATE panorama_run_cache SET segments = 'not json{' "
                         "WHERE signature = ?", (by_prefix['/p'],))
            conn.execute("UPDATE panorama_run_cache SET fingerprint = 'garbage' "
                         "WHERE signature = ?", (by_prefix['/b'],))
            conn.execute("UPDATE panorama_run_cache SET segments = '{\"paths\": 1}' "
                         "WHERE signature = ?", (by_prefix['/c'],))
            conn.commit()
        baseline = library.labels()
        del measured_runs[:]

        detect_panoramas(str(library.db), incremental=True)

        assert len(measured_runs) == 3, "a row that cannot be trusted is re-measured"
        assert library.labels() == baseline
        assert _cached_signatures(library.db) == set(by_prefix.values()), "and repaired"


class TestCacheHousekeeping:
    def test_a_pass_removes_the_retired_watermark_row(self, library, pass_overrides):
        with sqlite3.connect(library.db) as conn:
            conn.execute("INSERT OR REPLACE INTO stats_cache (key, value, updated_at) "
                         "VALUES ('panorama_detection:watermark', '{}', 0)")
            conn.commit()

        detect_panoramas(str(library.db), incremental=True)

        with sqlite3.connect(library.db) as conn:
            assert conn.execute("SELECT count(*) FROM stats_cache WHERE "
                                "key = 'panorama_detection:watermark'").fetchone()[0] == 0

    def test_a_deleted_photos_run_is_pruned(self, library, pass_overrides):
        before = _cached_signatures(library.db)
        assert len(before) == 3
        library.execute("DELETE FROM photos WHERE path LIKE '/c%'")
        expected = {run_signature(run) for run in _candidate_runs(library.db)}
        assert len(expected) == 2

        detect_panoramas(str(library.db), incremental=True)

        assert _cached_signatures(library.db) == expected
        assert expected < before


class TestCanonicalGroupIds:
    """G1: numbered by each set's smallest member path, one counter across both
    kinds, so a pass that mixed cached and measured runs agrees with a full one."""

    def _group_min_paths(self, labels):
        groups = {}
        for path, group_id, *_ in labels:
            groups.setdefault(group_id, []).append(path)
        return [min(groups[group_id]) for group_id in sorted(groups)]

    def test_a_pass_mixing_cached_and_measured_runs_numbers_like_a_full_pass(
            self, library, pass_overrides, measured_runs):
        library.execute("UPDATE photos SET scanned_at = '2026-03-03T00:00:00' "
                        "WHERE path = '/b5.jpg'")
        del measured_runs[:]
        detect_panoramas(str(library.db), incremental=True)
        assert len(measured_runs) == 1 and measured_runs[0][0] == '/b0.jpg'
        mixed = library.labels()

        detect_panoramas(str(library.db))
        assert library.labels() == mixed

        mins = self._group_min_paths(mixed)
        assert mins == sorted(mins) and mins == ['/b0.jpg', '/c0.jpg', '/pano0.jpg']
        assert sorted({row[1] for row in mixed}) == [1, 2, 3]

    def test_a_plain_set_before_an_hdr_set_in_run_order_keeps_path_order(self):
        from utils.panorama import _canonical_order
        hdr = {'paths': ['/z0.jpg', '/z1.jpg'], 'kind': HDR_PANORAMA}
        plain = {'paths': ['/a0.jpg', '/a1.jpg'], 'kind': PANORAMA}
        forced = {'paths': ['/m0.jpg', '/m1.jpg'], 'kind': PANORAMA}
        assert _canonical_order([hdr, forced, plain]) == [plain, forced, hdr]
        assert _canonical_order([plain, hdr, forced]) == [plain, forced, hdr]


@pytest.mark.skipif(
    sqlite3.sqlite_version_info < MIN_COVERING_SQLITE_EXPRESSION,
    reason=(f"SQLite {sqlite3.sqlite_version} is older than {MIN_COVERING_SQLITE_EXPRESSION}: "
            "a read naming length(thumbnail) is planned as SEARCH USING INDEX (a table lookup "
            "per row), not as a covering scan, below it"))
def test_the_candidate_read_is_answered_from_the_covering_index(tmp_path):
    from db.schema import init_database
    from utils.panorama import _LOAD_CANDIDATES_SQL

    db = str(tmp_path / 'plan.db')
    init_database(db)
    with sqlite3.connect(db) as conn:
        conn.executemany(
            "INSERT INTO photos (path, filename, date_taken, camera_model, focal_length, "
            "f_stop, shutter_speed, iso, scanned_at, render_version, thumbnail) "
            "VALUES (?,?,?,?,?,?,?,?,?,?, zeroblob(2500))",
            [(f'/p/{i:04d}.jpg', f'{i:04d}.jpg', f'2024:05:{1 + i % 28:02d} 10:00:00',
              'Canon EOS R6', 24.0 + i % 4, 8.0, '0.005', 100, '2026-01-01 00:00:00', 1)
             for i in range(300)])
        conn.execute("ANALYZE photos")
        conn.commit()

    def plan():
        # A fresh connection per plan: a statement prepared before the DROP would
        # keep reporting the dropped index.
        with sqlite3.connect(db) as probe:
            return [row[3] for row in probe.execute("EXPLAIN QUERY PLAN " + _LOAD_CANDIDATES_SQL)]

    covered = plan()
    assert any('COVERING INDEX idx_date_asc_cover' in line for line in covered), covered

    # Control: without the index the same SQL is not answered from an index alone.
    with sqlite3.connect(db) as conn:
        conn.execute("DROP INDEX idx_date_asc_cover")
    uncovered = plan()
    assert not any('COVERING INDEX' in line for line in uncovered), uncovered
