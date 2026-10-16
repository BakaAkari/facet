const MAX_ERROR_DETAIL_LENGTH = 300;

/**
 * Pull a human-readable `detail`/`message` string out of an HTTP error body.
 * FastAPI validation errors put a list of objects in `detail`, so anything
 * non-string is dropped rather than rendered, and the result is capped so a
 * pathological server message cannot blow out a toast or dialog layout.
 */
export function extractErrorDetail(error: unknown): string | undefined {
  const body = (error as { error?: unknown } | null | undefined)?.error;
  if (!body || typeof body !== 'object') return undefined;
  const record = body as Record<string, unknown>;
  const raw = typeof record['detail'] === 'string' ? record['detail']
    : typeof record['message'] === 'string' ? record['message']
    : undefined;
  if (!raw) return undefined;
  return raw.length > MAX_ERROR_DETAIL_LENGTH ? `${raw.slice(0, MAX_ERROR_DETAIL_LENGTH)}…` : raw;
}

/**
 * The `detail` of the server's 503 for a request refused because another writer
 * holds the library's write lock -- most often the one-time gallery index build
 * after an upgrade (api/__init__.py). The request can be repeated in a moment.
 */
export const LIBRARY_BUSY_DETAIL = 'library_busy';

/** True for the server's 503 `library_busy`: the library is busy, not broken. */
export function isLibraryBusyError(error: unknown): boolean {
  const e = error as { status?: unknown; error?: unknown } | null | undefined;
  if (e?.status !== 503) return false;
  const body = e.error;
  return !!body && typeof body === 'object' && (body as Record<string, unknown>)['detail'] === LIBRARY_BUSY_DETAIL;
}
