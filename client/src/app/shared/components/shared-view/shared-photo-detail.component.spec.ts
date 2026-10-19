import { TestBed } from '@angular/core/testing';
import { ActivatedRoute, Router } from '@angular/router';
import { of } from 'rxjs';
import { SharedPhotoDetailComponent } from './shared-photo-detail.component';
import { ApiService } from '../../../core/services/api.service';
import { I18nService } from '../../../core/services/i18n.service';
import { makePhoto } from '../../../../testing/photo.fixture';

describe('Shared photo detail extended metric availability', () => {
  beforeEach(() => TestBed.configureTestingModule({
    imports: [SharedPhotoDetailComponent],
    providers: [
      { provide: ApiService, useValue: { imageUrl: () => '/image', get: () => of({ options: [] }) } },
      { provide: I18nService, useValue: { t: (key: string) => key, locale: () => 'en', translations: () => ({}) } },
      { provide: Router, useValue: { navigate: vi.fn() } },
      { provide: ActivatedRoute, useValue: { snapshot: { queryParamMap: { get: () => null }, url: [] } } },
    ],
  }));

  it('preserves zero, hides present NULL (no library config in the shared view), and hides fields omitted by an older payload', () => {
    const view = TestBed.createComponent(SharedPhotoDetailComponent);
    vi.spyOn(view.componentInstance, 'ngOnInit').mockResolvedValue(undefined);
    view.componentInstance.photo.set(makePhoto({ aesthetic_v25: 0, qrealign_score: null, deqa_score: null }));
    view.detectChanges();
    expect(view.nativeElement.querySelector('[data-testid="aesthetic_v25-score"]').textContent.trim()).toBe('0.0');
    for (const key of ['qrealign_score', 'deqa_score']) {
      expect(view.nativeElement.querySelector(`[data-testid="${key}-row"]`)).toBeNull();
    }
    view.componentInstance.photo.set(makePhoto());
    view.detectChanges();
    expect(view.nativeElement.querySelectorAll('app-extended-quality [data-testid$="-row"]').length).toBe(0);
  });
});
