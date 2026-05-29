// Persistent prediction log for accuracy tracking.
//
// This is distinct from predictionsCache.ts: that cache expires entries after
// 24h (so the UI can reload a recent run), which is far too short to ever score
// a 30-day-ahead forecast against what actually happened. The accuracy log keeps
// each ML run's forecast points — with the base price they were made from — long
// enough (up to a year) for their target dates to arrive, so the Prediction
// Scorecard can grade each model on real, out-of-sample outcomes.

export interface LoggedModelPoint {
  date: string;      // forecast target date (YYYY-MM-DD)
  predicted: number; // forecast close
}

export interface LoggedPrediction {
  id: string;
  symbol: string;
  createdAt: number;        // ms epoch when the forecast was made
  basePrice: number;        // last actual close at forecast time (the "from" price)
  horizon: number;          // forecast horizon in days
  models: Record<string, LoggedModelPoint[]>; // model name -> forecast points
}

const LOG_KEY = 'predictionAccuracyLog';
const MAX_ENTRIES = 150;
const MAX_AGE_MS = 365 * 24 * 60 * 60 * 1000; // keep up to a year
const DEDUPE_WINDOW_MS = 6 * 60 * 60 * 1000;  // don't double-log within 6h

function readLog(): LoggedPrediction[] {
  if (typeof window === 'undefined') return [];
  try {
    const raw = localStorage.getItem(LOG_KEY);
    if (!raw) return [];
    const entries: LoggedPrediction[] = JSON.parse(raw);
    const cutoff = Date.now() - MAX_AGE_MS;
    const fresh = entries.filter((e) => e.createdAt >= cutoff);
    if (fresh.length !== entries.length) {
      localStorage.setItem(LOG_KEY, JSON.stringify(fresh));
    }
    return fresh;
  } catch (e) {
    console.error('Error reading prediction log:', e);
    return [];
  }
}

/** All logged predictions (newest first), pruned of anything older than a year. */
export function getPredictionLog(): LoggedPrediction[] {
  return readLog();
}

/** Logged predictions for a single symbol. */
export function getLogForSymbol(symbol: string): LoggedPrediction[] {
  const upper = symbol.toUpperCase();
  return readLog().filter((e) => e.symbol.toUpperCase() === upper);
}

/**
 * Record a completed ML run for later scoring. Each model's forecast points are
 * reduced to {date, predicted}. No-ops when there's nothing worth logging, and
 * de-dupes runs for the same symbol/horizon made within a few hours.
 */
export function logPredictionRun(
  symbol: string,
  basePrice: number,
  horizon: number,
  // Loosely typed like predictionsCache: model series come from several
  // generators (MLPrediction / MLForecast / ensemble) with slightly different
  // shapes. We validate each point at runtime below.
  models: Record<string, any[] | undefined>
): void {
  if (typeof window === 'undefined') return;
  if (!symbol || !Number.isFinite(basePrice) || basePrice <= 0) return;

  // Reduce to {date, predicted}, dropping empty/invalid model series.
  const cleaned: Record<string, LoggedModelPoint[]> = {};
  for (const [name, points] of Object.entries(models)) {
    if (!Array.isArray(points) || points.length === 0) continue;
    const pts = points
      .filter((p) => p && typeof p.date === 'string' && Number.isFinite(p.predicted))
      .map((p) => ({ date: p.date.slice(0, 10), predicted: p.predicted }));
    if (pts.length > 0) cleaned[name] = pts;
  }
  if (Object.keys(cleaned).length === 0) return;

  const log = readLog();
  const upper = symbol.toUpperCase();
  const now = Date.now();

  // Skip if we already logged a near-identical run very recently.
  const dup = log.find(
    (e) =>
      e.symbol.toUpperCase() === upper &&
      e.horizon === horizon &&
      now - e.createdAt < DEDUPE_WINDOW_MS &&
      Math.abs(e.basePrice - basePrice) / basePrice < 0.005
  );
  if (dup) return;

  const entry: LoggedPrediction = {
    id: `${upper}-${now}-${Math.random().toString(36).slice(2, 8)}`,
    symbol: upper,
    createdAt: now,
    basePrice,
    horizon,
    models: cleaned,
  };

  const next = [entry, ...log].slice(0, MAX_ENTRIES);
  try {
    localStorage.setItem(LOG_KEY, JSON.stringify(next));
  } catch (e) {
    console.error('Error writing prediction log:', e);
  }
}

export function clearLogEntry(id: string): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.setItem(LOG_KEY, JSON.stringify(readLog().filter((e) => e.id !== id)));
  } catch (e) {
    console.error('Error clearing log entry:', e);
  }
}

export function clearPredictionLog(): void {
  if (typeof window === 'undefined') return;
  try {
    localStorage.removeItem(LOG_KEY);
  } catch (e) {
    console.error('Error clearing prediction log:', e);
  }
}
