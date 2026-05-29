// Shared client-side JSON fetch helper with request de-duplication, a short
// TTL cache, and retry on rate-limit/transient failures. Lets many components
// request the same endpoint on a ticker switch without fanning out duplicate
// network calls or hammering an upstream that's already throttling us.

export interface FetchJSONOptions {
  /** Serve a cached response younger than this (ms). 0 disables caching (default 0). */
  ttlMs?: number;
  /** Retry attempts after the first try on 429/5xx/network errors (default 1). */
  retries?: number;
  /** Overall per-attempt timeout (ms, default 12000). */
  timeoutMs?: number;
  /** Caller-supplied abort signal (disables coalescing for this call). */
  signal?: AbortSignal;
}

interface CacheEntry {
  expires: number;
  data: unknown;
}

const cache = new Map<string, CacheEntry>();
const inFlight = new Map<string, Promise<unknown>>();

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Fetch JSON with caching, de-duplication and retry.
 * Concurrent calls for the same URL share one network request.
 */
export async function fetchJSON<T = any>(url: string, options: FetchJSONOptions = {}): Promise<T> {
  const { ttlMs = 0, retries = 1, timeoutMs = 12000, signal } = options;

  const now = Date.now();
  if (ttlMs > 0) {
    const cached = cache.get(url);
    if (cached && cached.expires > now) return cached.data as T;
  }

  // Don't coalesce calls that carry a caller signal — they may be cancelled
  // independently, which would reject the shared promise for everyone.
  if (!signal && inFlight.has(url)) {
    return inFlight.get(url) as Promise<T>;
  }

  const run = (async (): Promise<T> => {
    let lastError: unknown;

    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const res = await fetch(url, { signal: signal ?? AbortSignal.timeout(timeoutMs) });

        if ((res.status === 429 || res.status >= 500) && attempt < retries) {
          const retryAfter = Number(res.headers.get('retry-after'));
          const delay = Number.isFinite(retryAfter) && retryAfter > 0
            ? retryAfter * 1000
            : 500 * 2 ** attempt + Math.random() * 150;
          await sleep(delay);
          continue;
        }

        if (!res.ok) throw new Error(`Request to ${url} failed with status ${res.status}`);

        const data = (await res.json()) as T;
        if (ttlMs > 0) cache.set(url, { expires: Date.now() + ttlMs, data });
        return data;
      } catch (err) {
        lastError = err;
        if (attempt < retries) {
          await sleep(500 * 2 ** attempt + Math.random() * 150);
          continue;
        }
        throw err;
      }
    }

    throw lastError ?? new Error('fetchJSON: exhausted retries');
  })();

  if (!signal) {
    inFlight.set(url, run);
    run.finally(() => {
      if (inFlight.get(url) === run) inFlight.delete(url);
    });
  }

  return run;
}

/** Invalidate one URL (or the whole cache) — useful after a forced refresh. */
export function invalidateCache(url?: string): void {
  if (url) cache.delete(url);
  else cache.clear();
}
