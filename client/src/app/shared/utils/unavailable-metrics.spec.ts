import { isUnavailableKey, metricColumn, unavailableSet } from './unavailable-metrics';

describe('unavailable-metrics', () => {
  it('normalises filter, weight and column spellings to the column name', () => {
    expect(metricColumn('min_liqe')).toBe('liqe_score');
    expect(metricColumn('max_liqe')).toBe('liqe_score');
    expect(metricColumn('liqe_percent')).toBe('liqe_score');
    expect(metricColumn('liqe_score')).toBe('liqe_score');
    expect(metricColumn('min_aesthetic_iaa')).toBe('aesthetic_iaa');
    expect(metricColumn('face_quality_iqa_percent')).toBe('face_quality_iqa');
  });

  it('flags only keys whose metric is hidden', () => {
    const hidden = unavailableSet(['liqe_score']);
    expect(isUnavailableKey('min_liqe', hidden)).toBe(true);
    expect(isUnavailableKey('min_face_quality', hidden)).toBe(false);
  });

  it('tolerates a missing list', () => {
    expect(isUnavailableKey('min_liqe', unavailableSet(undefined))).toBe(false);
  });
});
