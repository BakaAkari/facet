import { TestBed } from '@angular/core/testing';
import { ExtendedQualityComponent } from './extended-quality.component';
import { I18nService } from '../../../core/services/i18n.service';

describe('ExtendedQualityComponent score availability', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [ExtendedQualityComponent],
    providers: [{
      provide: I18nService,
      useValue: {
        t: (key: string) => key === 'tooltip.not_computed' ? 'Not computed' : key,
        locale: () => 'en',
        translations: () => ({}),
      },
    }],
  }));

  const ALL = ['aesthetic_v25', 'qrealign_score', 'deqa_score'];

  function render(photo: object, tooltip = false, populated: string[] = ALL) {
    const fixture = TestBed.createComponent(ExtendedQualityComponent);
    fixture.componentRef.setInput('photo', photo);
    fixture.componentRef.setInput('populated', populated);
    fixture.componentRef.setInput('tooltip', tooltip);
    fixture.detectChanges();
    return fixture;
  }

  for (const key of ['aesthetic_v25', 'qrealign_score', 'deqa_score'] as const) {
    it(`shows present NULL ${key} as not computed, never N/A or a numeric default`, () => {
      const fixture = render({ [key]: null });
      const row = fixture.nativeElement.querySelector(`[data-testid="${key}-row"]`);
      expect(row.textContent).toContain('tooltip.' + key);
      expect(row.textContent).toContain('Not computed');
      expect(row.textContent).not.toContain('N/A');
      expect(row.textContent).not.toContain('0.0');
    });

    for (const value of [0, 7.23, 10]) {
      it(`preserves the actual ${key} score ${value}`, () => {
        const fixture = render({ [key]: value });
        const score = fixture.nativeElement.querySelector(`[data-testid="${key}-score"]`);
        expect(score.textContent.trim()).toBe(value.toFixed(1));
        expect(score.textContent).not.toContain('Not computed');
      });
    }

    it(`hides present NULL ${key} when that metric is not populated in the library`, () => {
      const fixture = render({ [key]: null }, false, ALL.filter(k => k !== key));
      expect(fixture.nativeElement.querySelector(`[data-testid="${key}-row"]`)).toBeNull();
    });

    it(`still shows a non-null ${key} when that metric is not populated`, () => {
      const fixture = render({ [key]: 3 }, false, []);
      expect(fixture.nativeElement.querySelector(`[data-testid="${key}-score"]`).textContent.trim()).toBe('3.0');
    });

    it(`keeps omitted/undefined ${key} hidden for older payloads`, () => {
      for (const photo of [{}, { [key]: undefined }, Object.create({ [key]: 8 })]) {
        const fixture = render(photo);
        expect(fixture.nativeElement.querySelector(`[data-testid="${key}-row"]`)).toBeNull();
        fixture.destroy();
      }
    });
  }

  it('renders any finite number as its value, with no 0..10 clamp', () => {
    for (const [value, text] of [[-1, '-1.0'], [11, '11.0']] as const) {
      const fixture = render({ aesthetic_v25: value });
      expect(fixture.nativeElement.querySelector('[data-testid="aesthetic_v25-score"]').textContent.trim()).toBe(text);
      fixture.destroy();
    }
  });

  it('hides non-finite and non-number values instead of labelling them not computed', () => {
    for (const value of [NaN, Infinity, -Infinity, '0']) {
      const fixture = render({ aesthetic_v25: value });
      expect(fixture.nativeElement.querySelector('[data-testid="aesthetic_v25-row"]')).toBeNull();
      expect(fixture.nativeElement.textContent).not.toContain('Not computed');
      fixture.destroy();
    }
  });

  it('colours the label by tooltip mode', () => {
    const label = (tooltip: boolean) =>
      render({ deqa_score: 5 }, tooltip).nativeElement.querySelector('[data-testid="deqa_score-row"] span') as HTMLElement;
    expect(label(true).style.color).toBe('var(--facet-tooltip-text-secondary)');
    expect(label(false).style.color).toBe('var(--mat-sys-on-surface-variant)');
  });

  it('updates available rows and their state when a different photo is selected', () => {
    const fixture = render({ aesthetic_v25: 8, qrealign_score: null, deqa_score: null }, true);
    expect(fixture.nativeElement.querySelectorAll('[data-testid$="-row"]').length).toBe(3);
    expect(fixture.nativeElement.textContent.match(/Not computed/g)).toHaveLength(2);
    fixture.componentRef.setInput('photo', { deqa_score: 0 });
    fixture.detectChanges();
    expect(fixture.nativeElement.querySelectorAll('[data-testid$="-row"]').length).toBe(1);
    expect(fixture.nativeElement.textContent).toContain('0.0');
    expect(fixture.nativeElement.textContent).not.toContain('Not computed');
  });
});
