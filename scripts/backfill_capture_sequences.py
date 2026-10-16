#!/usr/bin/env python3
"""Metadata-only capture protection backfill. Dry-run by default; never edits photos."""
import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db.schema import init_database  # noqa: E402
from exiftool.exiftool_batch import get_exif_batch  # noqa: E402
from utils.capture_sequences import detect_capture_sequences, persist_capture_metadata  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database', required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', required=True)
    parser.add_argument('--min-frames', type=int, default=15)
    parser.add_argument('--tolerance', type=float, default=0.15)
    args = parser.parse_args()
    source = Path(args.database).resolve(strict=True)
    # Dry run operates on a consistent SQLite backup, including schema migration.
    with tempfile.TemporaryDirectory(prefix='facet-capture-') as tmp:
        target = str(source) if args.apply else str(Path(tmp) / 'preview.db')
        if not args.apply:
            with sqlite3.connect(source.as_uri() + '?mode=ro', uri=True) as src:
                with sqlite3.connect(target) as dst:
                    src.backup(dst)
        init_database(target)
        failures = []
        with sqlite3.connect(target) as conn:
            paths = [r[0] for r in conn.execute('SELECT path FROM photos ORDER BY path')]
            for offset in range(0, len(paths), 50):
                batch = paths[offset:offset + 50]
                metadata = get_exif_batch(batch)
                for path in batch:
                    evidence = metadata.get(str(Path(path).resolve()))
                    if evidence is None:
                        failures.append(path)
                        continue
                    persist_capture_metadata(conn, {'path': path, **evidence})
                conn.commit()
        report = detect_capture_sequences(target, settings={
            'min_frames': args.min_frames, 'tolerance': args.tolerance})
        report.update(dry_run=not args.apply, metadata_failures=failures,
                      examined=len(paths), database=str(source))
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({'groups': len(report['groups']), 'examined': len(paths),
                          'metadata_failures': len(failures), 'dry_run': not args.apply}))
        return 2 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
