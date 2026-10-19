import type { Mock } from 'vitest';
import { TestBed } from '@angular/core/testing';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MAT_DIALOG_DATA } from '@angular/material/dialog';
import { I18nService } from '../../core/services/i18n.service';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import { MismatchReasonPipe, PhotoCritiqueDialogComponent } from './photo-critique-dialog.component';

describe('MismatchReasonPipe', () => {
  let pipe: MismatchReasonPipe;
  let mockI18n: { t: Mock };

  beforeEach(() => {
    mockI18n = { t: vi.fn((key: string) => key) };

    TestBed.configureTestingModule({
      providers: [
        { provide: I18nService, useValue: mockI18n },
      ],
    });

    pipe = TestBed.runInInjectionContext(() => new MismatchReasonPipe());
  });

  describe('required_tags', () => {
    it('formats required tags up to 3', () => {
      pipe.transform({ key: 'required_tags', required: ['landscape', 'mountain'], actual: [] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.required_tags', { tags: 'landscape, mountain' });
    });

    it('truncates with ellipsis when more than 3 tags', () => {
      pipe.transform({ key: 'required_tags', required: ['a', 'b', 'c', 'd'], actual: [] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.required_tags', { tags: 'a, b, c, …' });
    });

    it('handles empty required array', () => {
      pipe.transform({ key: 'required_tags', required: [], actual: [] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.required_tags', { tags: '' });
    });
  });

  describe('excluded_tags', () => {
    it('formats matched excluded tags', () => {
      pipe.transform({ key: 'excluded_tags', required: ['indoor'], actual: ['indoor'] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.excluded_tags', { tags: 'indoor' });
    });

    it('joins multiple excluded tags', () => {
      pipe.transform({ key: 'excluded_tags', required: ['indoor', 'text'], actual: ['indoor', 'text'] });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.excluded_tags', { tags: 'indoor, text' });
    });
  });

  describe('boolean filters', () => {
    it('uses base key when required is true', () => {
      pipe.transform({ key: 'has_face', required: true, actual: false });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.has_face');
    });

    it('uses _false suffix when required is false', () => {
      pipe.transform({ key: 'is_monochrome', required: false, actual: true });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.is_monochrome_false');
    });

    it('handles is_silhouette', () => {
      pipe.transform({ key: 'is_silhouette', required: true, actual: false });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.is_silhouette');
    });

    it('handles is_group_portrait', () => {
      pipe.transform({ key: 'is_group_portrait', required: false, actual: true });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.is_group_portrait_false');
    });
  });

  describe('numeric filters', () => {
    it('reports no_value when actual is null', () => {
      pipe.transform({ key: 'face_ratio_min', required: 0.05, actual: null });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.no_value');
    });

    it('reports no_value when actual is undefined', () => {
      pipe.transform({ key: 'iso_max', required: 6400, actual: undefined });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.no_value');
    });

    it('formats numeric mismatch with required and actual', () => {
      pipe.transform({ key: 'face_ratio_min', required: 0.05, actual: 0.02 });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.face_ratio_min', {
        required: '0.05',
        actual: '0.02',
      });
    });

    it('handles zero actual', () => {
      pipe.transform({ key: 'face_count_min', required: 1, actual: 0 });
      expect(mockI18n.t).toHaveBeenCalledWith('critique.reason.mismatch.face_count_min', {
        required: '1',
        actual: '0',
      });
    });
  });
});

describe('PhotoCritiqueDialogComponent overlay error handling', () => {
  it('reverts the overlay toggle and notifies when the overlay image fails to load', () => {
    const mockApi = { get: vi.fn() };
    const mockAuth = { hasFeature: vi.fn(() => true) };
    const mockI18n = { t: vi.fn((key: string) => key), locale: vi.fn(() => 'en') };
    const mockSnack = { open: vi.fn() };

    TestBed.resetTestingModule();
    TestBed.configureTestingModule({
      providers: [
        PhotoCritiqueDialogComponent,
        { provide: ApiService, useValue: mockApi },
        { provide: AuthService, useValue: mockAuth },
        { provide: I18nService, useValue: mockI18n },
        { provide: MatSnackBar, useValue: mockSnack },
        { provide: MAT_DIALOG_DATA, useValue: { photoPath: '/p/x.jpg', vlmAvailable: false } },
      ],
    });
    const component: any = TestBed.inject(PhotoCritiqueDialogComponent);
    component.overlayOn.set(true);

    component.onOverlayError();

    expect(component.overlayOn()).toBe(false);
    expect(mockSnack.open).toHaveBeenCalled();
  });
});

describe('PhotoCritiqueDialogComponent current calculation DOM', () => {
  async function render(overrides: Record<string, unknown> = {}) {
    const { of } = await import('rxjs');
    const response = {
      category: 'default', category_reason: { reason_key: 'default', category: 'default', details: [] },
      aggregate: 2, score_basis: 'current_formula', current_formula_score: 8.6,
      native_adjustment: 0, stored_score_matches_current_formula: false,
      breakdown: [{ metric: 'Noise quality', metric_key: 'noise_sigma', value: 8.6, weight: 1, contribution: 8.6 }],
      strengths: [], weaknesses: [], suggestions: [], penalties: {}, ...overrides,
    };
    TestBed.resetTestingModule();
    await TestBed.configureTestingModule({
      imports: [PhotoCritiqueDialogComponent],
      providers: [
        { provide: ApiService, useValue: { get: vi.fn(() => of(response)) } },
        { provide: AuthService, useValue: { hasFeature: vi.fn(() => false) } },
        { provide: I18nService, useValue: { t: (key: string) => key, locale: () => 'en', translations: () => ({}) } },
        { provide: MatSnackBar, useValue: { open: vi.fn() } },
        { provide: MAT_DIALOG_DATA, useValue: { photoPath: '/synthetic', vlmAvailable: false } },
      ],
    }).compileComponents();
    const fixture = TestBed.createComponent(PhotoCritiqueDialogComponent);
    fixture.detectChanges(); await fixture.whenStable(); fixture.detectChanges();
    return fixture;
  }
  it('shows stored score separately from the native total and adjustment', async () => {
    const fixture = await render();
    const context = fixture.nativeElement.querySelector('[data-testid="score-calculation-context"]');
    const total = fixture.nativeElement.querySelector('[data-testid="score-calculation-total"]');
    expect(context.textContent).toContain('2.00');
    expect(context.textContent).toContain('critique.different_score');
    expect(total.textContent).toContain('8.60');
    expect(total.textContent).toContain('critique.adjustments');
    expect(total.textContent).toContain('0.00');
  });
  it('renders zero stored/current scores and zero adjustments', async () => {
    const fixture = await render({ aggregate: 0, current_formula_score: 0, native_adjustment: 0, stored_score_matches_current_formula: true });
    const context = fixture.nativeElement.querySelector('[data-testid="score-calculation-context"]');
    const total = fixture.nativeElement.querySelector('[data-testid="score-calculation-total"]');
    expect(context.textContent).toContain('0.00');
    expect(total.textContent.match(/0\.00/g)).toHaveLength(2);
    expect(context.textContent).not.toContain('critique.different_score');
  });
  it('keeps stored score when context is unavailable without fabricating a total', async () => {
    const fixture = await render({ aggregate: 7, current_formula_score: null, native_adjustment: null, breakdown: [], category_reason: { reason_key: 'unavailable', category: 'portrait', details: [] }, explanation_unavailable_reason: 'normalization_snapshot_unavailable' });
    expect(fixture.nativeElement.textContent).toContain('7.00');
    expect(fixture.nativeElement.textContent).toContain('critique.calculation_unavailable');
    expect(fixture.nativeElement.textContent).not.toContain('critique.reason.default');
    expect(fixture.nativeElement.querySelector('[data-testid="score-calculation-total"]')).toBeNull();
  });
  it('marks cached commentary without claiming verified scoring context', async () => {
    const fixture = await render({ vlm_critique: 'Synthetic cached commentary', vlm_source: 'cached' });
    expect(fixture.nativeElement.textContent).toContain('critique.cached_commentary');
    expect(fixture.nativeElement.textContent).toContain('Synthetic cached commentary');
  });
  it('keeps older response rendering without claiming a current calculation', async () => {
    const fixture = await render({ score_basis: undefined, current_formula_score: undefined });
    expect(fixture.nativeElement.querySelector('[data-testid="score-calculation-context"]')).toBeNull();
    expect(fixture.nativeElement.querySelector('[data-testid="score-calculation-total"]')).toBeNull();
    expect(fixture.nativeElement.textContent).toContain('8.60');
  });
});
