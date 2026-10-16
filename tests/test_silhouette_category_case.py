"""The hard-coded category CASE in ``config/percentile_normalizer.py`` honours ``is_silhouette``.

Two queries (the category-imbalance analysis and the stats health summary) bucket photos with
a SQL CASE that mirrors the categorizer. ``portrait`` and ``portrait_bw`` carry
``is_silhouette: false`` in the shipped config, so a silhouette with a large face is NOT a
portrait; the CASE sent it to portrait because its ``face_ratio > 0.05`` branches ran first.
A NULL flag means "not a silhouette" to the categorizer (``bool(pd.get('is_silhouette', 0))``),
so those rows stay portraits: the SQL must say ``IFNULL(is_silhouette, 0) = 0``, not ``= 0``
(``NULL = 0`` is not true).
"""

import logging
import sqlite3

import pytest

from config.percentile_normalizer import PercentileNormalizer, _calc_stats
from db.schema import init_database

GROUP = 51          # the analysis keeps a bucket only when it holds more than 50 photos

# (name, is_silhouette, is_monochrome, aggregate): every row has face_ratio 0.2, one face.
ROWS = (
    ('silhouette_flagged', 1, 0, 1.0),
    ('silhouette_bw_flagged', 1, 1, 1.0),
    ('portrait_null_flag', None, 0, 3.0),
    ('portrait_bw_null_flag', None, 1, 6.0),
)


@pytest.fixture()
def db_path(tmp_path):
    path = str(tmp_path / 'case.db')
    init_database(path)
    conn = sqlite3.connect(path)
    for name, flag, mono, aggregate in ROWS:
        for i in range(GROUP):
            conn.execute(
                "INSERT INTO photos (path, filename, face_ratio, face_count, is_silhouette, "
                "is_monochrome, aggregate) VALUES (?, ?, 0.2, 1, ?, ?, ?)",
                (f'/case/{name}/{i}.jpg', f'{i}.jpg', flag, mono, aggregate))
    for i in range(GROUP):
        conn.execute("INSERT INTO photos (path, filename, face_ratio, face_count, aggregate) "
                     "VALUES (?, ?, 0, 0, 9.0)", (f'/case/none/{i}.jpg', f'{i}.jpg'))
    conn.commit()
    conn.close()
    return path


def test_the_imbalance_analysis_buckets_a_strict_silhouette_as_silhouette(db_path):
    normalizer = PercentileNormalizer(db_path)
    conn = sqlite3.connect(db_path)
    try:
        issues = normalizer._analyze_scoring_issues(conn, None, _calc_stats, normalizer=normalizer)
    finally:
        conn.close()
    buckets = {}
    for issue in issues:
        if issue['issue_type'] == 'category_imbalance':
            name = issue['description'].split(' ', 1)[0]
            buckets[name] = issue['description']
    # 51 + 51 flagged rows (plain and B&W) -> silhouette; NULL-flag rows stay portraits
    assert 'silhouette' in buckets, buckets
    assert '(n=102' in buckets['silhouette']
    assert 'portrait' in buckets and '(n=51' in buckets['portrait']
    assert 'portrait_bw' in buckets and '(n=51' in buckets['portrait_bw']


def test_the_stats_health_summary_buckets_a_strict_silhouette_as_silhouette(db_path, caplog):
    normalizer = PercentileNormalizer(db_path)
    with caplog.at_level(logging.INFO, logger='facet.normalizer'):
        normalizer.print_database_statistics(config=None, verbose=True)
    counts = {r.args[0]: r.args[1] for r in caplog.records
              if r.args and len(r.args) == 6 and isinstance(r.args[0], str) and isinstance(r.args[1], int)}
    assert counts.get('Silhouette') == 102, counts
    assert counts.get('Portrait') == 51, counts           # the IFNULL rows
    assert counts.get('Portrait B&W') == 51, counts       # the IFNULL B&W rows
    assert counts.get('Others') == 51, counts
