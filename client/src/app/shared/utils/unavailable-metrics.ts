/**
 * Commercial-mode metric gating. The server reports `unavailable_metrics` as
 * photos-column names (`liqe_score`, `aesthetic_iaa`, `face_quality_iqa`,
 * `deqa_score`); the filter keys (`min_liqe`, `min_deqa`) and weight keys
 * (`liqe_percent`, `deqa_percent`) spell LIQE and DeQA differently, so every
 * spelling is normalised to the column name here.
 */
const METRIC_ALIASES: Record<string, string> = { liqe: 'liqe_score', deqa: 'deqa_score' };

/** Column-level metric name behind a filter key (`min_*`/`max_*`) or weight key (`*_percent`). */
export function metricColumn(key: string): string {
  const base = key.replace(/^(min|max)_/, '').replace(/_percent$/, '');
  return METRIC_ALIASES[base] ?? base;
}

/** Set of column names for O(1) lookups; tolerates a missing config field. */
export function unavailableSet(metrics: readonly string[] | null | undefined): ReadonlySet<string> {
  return new Set(metrics ?? []);
}

/** True when the filter/weight key belongs to a metric the licensing policy hides. */
export function isUnavailableKey(key: string, unavailable: ReadonlySet<string>): boolean {
  return unavailable.size > 0 && unavailable.has(metricColumn(key));
}
