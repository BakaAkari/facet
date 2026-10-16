"""Run with: python -m unittest discover -s tests_capture -v.

Pure policy tests intentionally avoid importing the heavyweight ML/API package.
Integration tests must additionally exercise the production database and routes.
"""
import importlib.util
import unittest
from datetime import datetime, timedelta
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'capture_sequences', Path(__file__).resolve().parents[1] / 'utils/capture_sequences.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CapturePolicyTests(unittest.TestCase):
    def rows(self, count=16, interval=2):
        return [dict(path=f'/fixtures/{i:03}.raf', camera_model='GFX',
                     camera_serial='test-body', focal_length=50, phash='abcdef',
                     date_taken=(datetime(2026, 1, 1) + timedelta(seconds=i*interval)).isoformat())
                for i in range(count)]

    def test_fuji_grouped_tags(self):
        for count in (4, 16):
            self.assertEqual(capture.pixel_shift_evidence(
                {'FujiFilm:PixelShiftShots': count})['pixel_shift_shots'], count)

    def test_invalid_fuji_modes(self):
        for value in (None, '', 0, 1, 4.5, float('inf'), 'unknown'):
            self.assertIsNone(capture.pixel_shift_evidence(
                {'PixelShiftShots': value})['pixel_shift_shots'])

    def test_sony_source_and_non_source(self):
        self.assertEqual(capture.sony_pixel_shift('01020304 1 4 0x0'),
                         ('01020304:0x0', 1, 4))
        self.assertEqual(capture.sony_pixel_shift('Group 01020304, Shot 16/16 (0x1)'),
                         ('01020304:0x1', 16, 16))
        for value in ('n/a', '00000000 0 0 0x0', '01020304 0 4 0x0',
                      '01020304 17 16 0x0', 'yes', None):
            self.assertIsNone(capture.sony_pixel_shift(value))

    def test_cadence(self):
        for interval in (2, 5):
            self.assertEqual(len(capture.find_timelapses(self.rows(interval=interval))[0]), 16)

    def test_short_and_fast_bursts(self):
        self.assertEqual(capture.find_timelapses(self.rows(count=14)), [])
        self.assertEqual(capture.find_timelapses(self.rows(interval=0.1)), [])

    def test_missing_evidence(self):
        for key in ('camera_model', 'date_taken', 'focal_length', 'phash'):
            rows = self.rows()
            for row in rows:
                row[key] = None
            self.assertEqual(capture.find_timelapses(rows), [])

    def test_camera_separation(self):
        rows = self.rows()
        for row in rows[8:]:
            row['camera_serial'] = 'other-body'
        self.assertEqual(capture.find_timelapses(rows), [])

    def test_visual_jump(self):
        rows = self.rows()
        for row in rows[8:]:
            row['phash'] = '000000'
        self.assertEqual(capture.find_timelapses(rows), [])

    def test_gap_splits(self):
        rows = self.rows()
        for index, row in enumerate(rows[8:], 8):
            row['date_taken'] = (datetime(2026, 1, 1) + timedelta(seconds=index*2+100)).isoformat()
        self.assertEqual(capture.find_timelapses(rows), [])

    def test_group_identity(self):
        paths = ['/a/1', '/a/2']
        self.assertEqual(capture.stable_group_id('pixel_shift', paths),
                         capture.stable_group_id('pixel_shift', paths[::-1]))
        self.assertNotEqual(capture.stable_group_id('pixel_shift', paths),
                            capture.stable_group_id('timelapse', paths))


if __name__ == '__main__':
    unittest.main()
