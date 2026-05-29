// Scoring engine for the Prediction Scorecard.
//
// Two complementary measures, both dependency-light (no tfjs):
//   1. scoreLoggedPredictions — grades real, out-of-sample forecasts from the
//      prediction log against the actual closes that have since materialised.
//   2. holdoutBacktest — an instant sanity check: re-runs a model on a holdout
//      tail of history so the panel shows signal immediately, before any logged
//      forecast has had time to resolve.

import { StockData } from '@/types';
import { LoggedPrediction } from './predictionLog';

export type Grade = 'A' | 'B' | 'C' | 'D' | 'F';

export interface ModelScore {
  model: string;
  resolved: number;             // forecast points whose target date has passed and matched an actual close
  pending: number;              // forecast points not yet resolvable
  directionalAccuracy: number;  // % of points that called up/down (vs base price) correctly
  mape: number;                 // mean absolute percentage error over resolved points
  grade: Grade;
}

export interface BacktestScore {
  model: string;
  testedPoints: number;
  directionalAccuracy: number;
  mape: number;
  grade: Grade;
}

export interface ForecastPoint {
  date: string;
  predicted: number;
}

export type Generator = (data: StockData[], days: number) => ForecastPoint[];

const DAY_MS = 86400000;
const DEFAULT_TOLERANCE_DAYS = 4;

interface ActualIndex {
  byDay: Map<string, number>;
  days: { t: number; c: number }[];
  lastTime: number;
}

/** Index actual OHLCV by calendar day for fast forecast-date lookups. */
export function buildActualIndex(actual: StockData[]): ActualIndex {
  const byDay = new Map<string, number>();
  const days: { t: number; c: number }[] = [];
  for (const bar of actual) {
    if (!bar || !(bar.close > 0)) continue;
    const key = bar.date.slice(0, 10);
    const t = Date.parse(key);
    if (Number.isNaN(t)) continue;
    byDay.set(key, bar.close);
    days.push({ t, c: bar.close });
  }
  days.sort((a, b) => a.t - b.t);
  return { byDay, days, lastTime: days.length ? days[days.length - 1].t : 0 };
}

/** Actual close on a forecast date, falling back to the nearest trading day within tolerance. */
function matchActualClose(index: ActualIndex, dateStr: string, toleranceDays = DEFAULT_TOLERANCE_DAYS): number | null {
  const key = dateStr.slice(0, 10);
  const exact = index.byDay.get(key);
  if (exact !== undefined) return exact;

  const target = Date.parse(key);
  if (Number.isNaN(target)) return null;

  let best: number | null = null;
  let bestDiff = Infinity;
  for (const d of index.days) {
    const diff = Math.abs(d.t - target);
    if (diff <= toleranceDays * DAY_MS && diff < bestDiff) {
      bestDiff = diff;
      best = d.c;
    }
  }
  return best;
}

export function gradeFromDirectional(accuracy: number): Grade {
  if (accuracy >= 60) return 'A';
  if (accuracy >= 55) return 'B';
  if (accuracy >= 50) return 'C';
  if (accuracy >= 45) return 'D';
  return 'F';
}

export function gradeColor(grade: Grade): string {
  switch (grade) {
    case 'A': return 'var(--green-1)';
    case 'B': return 'var(--blue-1)';
    case 'C': return 'var(--yellow-1)';
    case 'D': return 'var(--red-1)';
    case 'F': return 'var(--red-2)';
  }
}

interface Tally {
  resolved: number;
  pending: number;
  hits: number;
  mapeSum: number;
}

function addPoint(tally: Tally, basePrice: number, predicted: number, actual: number) {
  tally.resolved += 1;
  tally.mapeSum += (Math.abs(actual - predicted) / actual) * 100;
  const predUp = predicted >= basePrice;
  const actUp = actual >= basePrice;
  if (predUp === actUp) tally.hits += 1;
}

/**
 * Grade every model in the prediction log against actual closes.
 * Aggregates across all logged runs for whatever symbol(s) the entries cover.
 */
export function scoreLoggedPredictions(
  entries: LoggedPrediction[],
  actual: StockData[],
  toleranceDays = DEFAULT_TOLERANCE_DAYS
): ModelScore[] {
  const index = buildActualIndex(actual);
  if (index.days.length === 0) return [];

  const tallies = new Map<string, Tally>();
  const get = (model: string): Tally => {
    let t = tallies.get(model);
    if (!t) {
      t = { resolved: 0, pending: 0, hits: 0, mapeSum: 0 };
      tallies.set(model, t);
    }
    return t;
  };

  for (const entry of entries) {
    if (!(entry.basePrice > 0)) continue;
    for (const [model, points] of Object.entries(entry.models)) {
      const tally = get(model);
      for (const point of points) {
        const targetTime = Date.parse(point.date.slice(0, 10));
        if (Number.isNaN(targetTime)) continue;
        // Target still in the future relative to the data we have -> pending.
        if (targetTime > index.lastTime + toleranceDays * DAY_MS) {
          tally.pending += 1;
          continue;
        }
        const actualClose = matchActualClose(index, point.date, toleranceDays);
        if (actualClose === null || !(actualClose > 0)) continue;
        addPoint(tally, entry.basePrice, point.predicted, actualClose);
      }
    }
  }

  const scores: ModelScore[] = [];
  for (const [model, t] of tallies) {
    if (t.resolved === 0 && t.pending === 0) continue;
    const directionalAccuracy = t.resolved > 0 ? (t.hits / t.resolved) * 100 : 0;
    const mape = t.resolved > 0 ? t.mapeSum / t.resolved : 0;
    scores.push({
      model,
      resolved: t.resolved,
      pending: t.pending,
      directionalAccuracy,
      mape,
      grade: gradeFromDirectional(directionalAccuracy),
    });
  }

  // Resolved models first (by accuracy), then purely-pending models.
  return scores.sort((a, b) => {
    if ((b.resolved > 0 ? 1 : 0) !== (a.resolved > 0 ? 1 : 0)) {
      return (b.resolved > 0 ? 1 : 0) - (a.resolved > 0 ? 1 : 0);
    }
    return b.directionalAccuracy - a.directionalAccuracy;
  });
}

/**
 * Instant holdout backtest: train each model on all-but-the-last `testDays`,
 * forecast that window, and grade the forecast against what actually happened.
 * Directional accuracy is measured relative to the base (last training) price,
 * matching how live tracking scores forecasts.
 */
export function holdoutBacktest(
  data: StockData[],
  generators: Record<string, Generator>,
  testDays = 10
): BacktestScore[] {
  const results: BacktestScore[] = [];
  if (data.length < testDays + 60) return results;

  const trainData = data.slice(0, -testDays);
  const actualSlice = data.slice(-testDays);
  const basePrice = trainData[trainData.length - 1].close;
  if (!(basePrice > 0)) return results;

  for (const [model, generate] of Object.entries(generators)) {
    let preds: ForecastPoint[];
    try {
      preds = generate(trainData, testDays);
    } catch (e) {
      console.error(`Holdout backtest failed for ${model}:`, e);
      continue;
    }
    if (!preds || preds.length === 0) continue;

    const n = Math.min(preds.length, actualSlice.length);
    const tally: Tally = { resolved: 0, pending: 0, hits: 0, mapeSum: 0 };
    for (let i = 0; i < n; i++) {
      const actual = actualSlice[i].close;
      if (!(actual > 0) || !Number.isFinite(preds[i].predicted)) continue;
      addPoint(tally, basePrice, preds[i].predicted, actual);
    }
    if (tally.resolved === 0) continue;

    const directionalAccuracy = (tally.hits / tally.resolved) * 100;
    results.push({
      model,
      testedPoints: tally.resolved,
      directionalAccuracy,
      mape: tally.mapeSum / tally.resolved,
      grade: gradeFromDirectional(directionalAccuracy),
    });
  }

  return results.sort((a, b) => b.directionalAccuracy - a.directionalAccuracy);
}

/** Human-friendly model labels for display. */
export const MODEL_LABELS: Record<string, string> = {
  linearRegression: 'Linear Regression',
  ema: 'EMA',
  arima: 'ARIMA',
  prophetLite: 'Prophet-Lite',
  lstm: 'LSTM',
  gru: 'GRU',
  cnnLstm: 'CNN-LSTM',
  ensemble: 'Ensemble',
};

export function modelLabel(model: string): string {
  return MODEL_LABELS[model] ?? model;
}
