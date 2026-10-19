import { Component, computed, input } from '@angular/core';
import { Photo } from '../../models/photo.model';
import { FixedPipe } from '../../pipes/fixed.pipe';
import { TranslatePipe } from '../../pipes/translate.pipe';

type ExtendedQuality = Pick<Photo, 'aesthetic_v25' | 'qrealign_score' | 'deqa_score'>;

/** Omitted API fields stay hidden; present null values have no computed score. */
@Component({
  selector: 'app-extended-quality',
  imports: [FixedPipe, TranslatePipe],
  host: { class: 'block' },
  template: `
    @for (metric of metrics(); track metric.key) {
      <div class="flex justify-between items-baseline gap-2" [attr.data-testid]="metric.key + '-row'">
        <span [style.color]="tooltip() ? 'var(--facet-tooltip-text-secondary)' : 'var(--mat-sys-on-surface-variant)'">{{ metric.label | translate }}</span>
        <span class="text-[var(--mat-sys-primary)] font-medium" [attr.data-testid]="metric.key + '-score'">@if (metric.value !== null) {
          {{ metric.value | fixed:1 }}
        } @else {
          {{ 'tooltip.not_computed' | translate }}
        }</span>
      </div>
    }
  `,
})
export class ExtendedQualityComponent {
  photo = input.required<ExtendedQuality>();
  tooltip = input(false);
  metrics = computed(() => {
    const photo = this.photo();
    return (['aesthetic_v25', 'qrealign_score', 'deqa_score'] as const)
      .flatMap(key => {
        const value = photo[key];
        if (!Object.hasOwn(photo, key) || value === undefined) return []; // Older schemas may omit the field.
        const score = typeof value === 'number' && Number.isFinite(value)
          && value >= 0 && value <= 10 ? value : null;
        return [{ key, label: 'tooltip.' + key, value: score }];
      });
  });
}
