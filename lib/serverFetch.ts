// Shared server-side fetch helper with retry/backoff, timeout, and in-flight
// coalescing. Used by API routes that hit rate-limit-prone upstreams (Yahoo
// Finance, etc.) so a burst of requests degrades gracefully instead of failing.
//
// Workers-safe: relies only on the global fetch, AbortSignal.timeout and
// setTimeout — no Node-only APIs.

export interface RetryOptions {
  /** Number of retry attempts after the first try (default 3). */
  retries?: number;
  /** Base backoff in ms; doubles each attempt (default 400). */
  baseDelayMs?: number;
  /** Maximum backoff in ms (default 4000). */
  maxDelayMs?: number;
  /** Per-attempt timeout in ms (default 10000). Ignored if init.signal is set. */
  timeoutMs?: number;
  /** HTTP status codes that trigger a retry (default 429 + 5xx). */
  retryOn?: number[];
  /** Coalesce concurrent identical GETs into one upstream request (default true). */
  coalesce?: boolean;
}

const DEFAULT_RETRY_ON = [429, 500, 502, 503, 504];

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

// Concurrent identical GETs share a single in-flight request within this isolate.
const inFlight = new Map<string, Promise<Response>>();

/** Parse a Retry-After header (delta-seconds or HTTP-date) into milliseconds. */
function parseRetryAfter(value: string | null): number | null {
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, seconds * 1000);
  const date = Date.parse(value);
  if (!Number.isNaN(date)) return Math.max(0, date - Date.now());
  return null;
}

function backoffDelay(attempt: number, base: number, max: number): number {
  const exp = Math.min(max, base * 2 ** attempt);
  return exp + Math.random() * 150; // jitter to avoid thundering herd
}

/**
 * fetch() with automatic retry/backoff on rate-limit and transient errors.
 * Honours Retry-After when the upstream provides it.
 */
export async function fetchWithRetry(
  url: string,
  init: RequestInit = {},
  options: RetryOptions = {}
): Promise<Response> {
  const {
    retries = 3,
    baseDelayMs = 400,
    maxDelayMs = 4000,
    timeoutMs = 10000,
    retryOn = DEFAULT_RETRY_ON,
    coalesce = true,
  } = options;

  const method = (init.method || 'GET').toUpperCase();
  const canCoalesce = coalesce && method === 'GET';
  const key = `${method} ${url}`;

  if (canCoalesce) {
    const existing = inFlight.get(key);
    // Clone so each concurrent caller gets an independently-readable body.
    if (existing) return existing.then((res) => res.clone());
  }

  const run = (async (): Promise<Response> => {
    let lastError: unknown;

    for (let attempt = 0; attempt <= retries; attempt++) {
      try {
        const signal = init.signal ?? AbortSignal.timeout(timeoutMs);
        const res = await fetch(url, { ...init, signal });

        if (retryOn.includes(res.status) && attempt < retries) {
          const retryAfter = parseRetryAfter(res.headers.get('retry-after'));
          await sleep(retryAfter ?? backoffDelay(attempt, baseDelayMs, maxDelayMs));
          continue;
        }

        return res;
      } catch (err) {
        lastError = err;
        if (attempt < retries) {
          await sleep(backoffDelay(attempt, baseDelayMs, maxDelayMs));
          continue;
        }
        throw err;
      }
    }

    throw lastError ?? new Error('fetchWithRetry: exhausted retries');
  })();

  if (canCoalesce) {
    inFlight.set(key, run);
    run.finally(() => {
      if (inFlight.get(key) === run) inFlight.delete(key);
    });
    return run.then((res) => res.clone());
  }

  return run;
}

/** Convenience wrapper that returns parsed JSON, throwing on a non-OK response. */
export async function fetchJSONWithRetry<T = any>(
  url: string,
  init: RequestInit = {},
  options: RetryOptions = {}
): Promise<T> {
  const res = await fetchWithRetry(url, init, options);
  if (!res.ok) {
    throw new Error(`Request to ${url} failed with status ${res.status}`);
  }
  return res.json() as Promise<T>;
}
