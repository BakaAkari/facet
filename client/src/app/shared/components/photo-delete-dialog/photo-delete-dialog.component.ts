import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { MAT_DIALOG_DATA, MatDialogRef, MatDialogModule } from '@angular/material/dialog';
import { MatButtonModule } from '@angular/material/button';
import { MatCheckboxModule } from '@angular/material/checkbox';
import { TranslatePipe } from '../../pipes/translate.pipe';
import { i18nKeys } from '../../../core/i18n/keys';

export interface PhotoDeleteDialogResult {
  includeCompanions: boolean;
  includeSequenceSiblings: boolean;
}

export interface PhotoDeleteDialogData {
  /** Which call site opened this: each has its own copy (decision: `photo_detail.delete.*`
   *  for the single-photo surface, `cull.delete_*` for the bulk one -- see i18n step 14). */
  surface: 'photo_detail' | 'bulk';
  paths: string[];
  /** How many photos the action targets -- `paths.length` for photo-detail. */
  count: number;
  /** Whether to offer the companion RAW/XMP checkbox at all. Computed by the
   *  caller (there is no cheap client-side signal of "this file actually has
   *  a companion" -- mirrors `cull-dialog`'s own unconditional checkbox). */
  hasCompanion: boolean;
  /** Whether to offer the sequence-sibling checkbox -- true when the target(s)
   *  belong to a bracket/panorama/hdr_panorama set. */
  hasSiblings: boolean;
  /** photo-detail only: true when the single target is itself a bracket-kind
   *  lead frame, which the endpoint refuses without `includeSequenceSiblings`
   *  (decision 7). Always false for a bulk selection -- there is no per-path
   *  lead signal client-side; the gallery instead reports
   *  `refused_bracket_lead` after the call returns. */
  hasBracketLead: boolean;
}

/**
 * Confirm dialog for `POST /api/photo/delete`. States plainly that the file(s)
 * go to the OS trash and stay recoverable (decision 1). Cannot reuse the
 * shared `ConfirmDialogComponent`, which renders a single `<p>` and has no
 * room for the companion/sibling checkboxes (G11) -- structured like
 * `cull-dialog.component.ts` instead.
 */
@Component({
  selector: 'app-photo-delete-dialog',
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [MatDialogModule, MatButtonModule, MatCheckboxModule, TranslatePipe],
  template: `
    <h2 mat-dialog-title>
      {{ (data.surface === 'photo_detail' ? I18N.photo_detail.delete.confirm_title : I18N.cull.delete_confirm_title) | translate }}
    </h2>
    <mat-dialog-content class="!pt-2 min-w-[20rem] max-w-[30rem]">
      @if (data.surface === 'photo_detail') {
        <p class="text-sm mb-3">{{ bodyKey() | translate }}</p>
      } @else {
        <p class="text-sm mb-3">{{ I18N.cull.delete_confirm_body | translate:{ count: data.count } }}</p>
      }

      @if (data.hasCompanion) {
        <mat-checkbox [checked]="includeCompanions()" (change)="includeCompanions.set($event.checked)" class="text-sm block">
          {{ (data.surface === 'photo_detail' ? I18N.photo_detail.delete.include_companions : I18N.cull.include_companions) | translate }}
        </mat-checkbox>
      }
      @if (data.hasSiblings) {
        <mat-checkbox [checked]="includeSequenceSiblings()" (change)="includeSequenceSiblings.set($event.checked)" class="text-sm block">
          {{ (data.surface === 'photo_detail' ? I18N.photo_detail.delete.include_siblings : I18N.cull.include_sequence_siblings) | translate }}
        </mat-checkbox>
      }
    </mat-dialog-content>
    <mat-dialog-actions align="end">
      <button mat-button mat-dialog-close>{{ I18N.cull.cancel | translate }}</button>
      <button mat-flat-button color="warn" (click)="confirm()">
        {{ (data.surface === 'photo_detail' ? I18N.photo_detail.delete.button_label : I18N.cull.delete_action) | translate }}
      </button>
    </mat-dialog-actions>
  `,
})
export class PhotoDeleteDialogComponent {
  protected readonly I18N = i18nKeys();
  private readonly dialogRef = inject(MatDialogRef<PhotoDeleteDialogComponent, PhotoDeleteDialogResult>);
  protected readonly data = inject<PhotoDeleteDialogData>(MAT_DIALOG_DATA);

  protected readonly includeCompanions = signal(false);
  protected readonly includeSequenceSiblings = signal(false);

  /** photo-detail only: which body variant to show, layered by what is ticked
   *  (and, independent of the tick, whether this single target is itself a
   *  refused bracket lead) -- decision 1's OS-trash/recoverable statement is
   *  the common suffix carried by every one of these keys. */
  protected readonly bodyKey = computed(() => {
    if (this.data.hasBracketLead && !this.includeSequenceSiblings()) {
      return this.I18N.photo_detail.delete.confirm_body_bracket_refused;
    }
    if (this.includeSequenceSiblings()) return this.I18N.photo_detail.delete.confirm_body_with_siblings;
    if (this.includeCompanions()) return this.I18N.photo_detail.delete.confirm_body_with_companion;
    return this.I18N.photo_detail.delete.confirm_body;
  });

  protected confirm(): void {
    this.dialogRef.close({
      includeCompanions: this.includeCompanions(),
      includeSequenceSiblings: this.includeSequenceSiblings(),
    });
  }
}
