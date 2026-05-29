import { describe, it, expect } from 'vitest';
import {
  generateLinearRegression,
  generateEMAForecast,
  generateARIMAForecast,
  generateProphetLiteForecast,
} from '@/lib/mlAlgorithms';
import { StockData } from '@/types';

// ~200 sessions of a noisy uptrend so every model has enough history.
function makeBars(): StockData[] {
  const bars: StockData[] = [];
  for (let i = 0; i < 200; i++) {
    const base = 100 + i * 0.4 + Math.sin(i / 6) * 4;
    bars.push({
      date: new Date(2023, 0, i + 1).toISOString(),
      open: base - 0.5,
      high: base + 1.5,
      low: base - 1.5,
      close: base,
      volume: 1_000_000 + i * 1000,
    });
  }
  return bars;
}

const FORECAST_DAYS = 5;
const models = [
  ['linear regression', generateLinearRegression],
  ['EMA', generateEMAForecast],
  ['ARIMA', generateARIMAForecast],
  ['prophet-lite', generateProphetLiteForecast],
] as const;

describe('ML forecast algorithms', () => {
  const bars = makeBars();

  for (const [name, fn] of models) {
    describe(name, () => {
      const predictions = fn(bars, FORECAST_DAYS);

      it('returns one prediction per forecast day', () => {
        expect(predictions).toHaveLength(FORECAST_DAYS);
      });

      it('produces finite, positive prices with a labelled algorithm', () => {
        for (const p of predictions) {
          expect(Number.isFinite(p.predicted)).toBe(true);
          expect(p.predicted).toBeGreaterThan(0);
          expect(typeof p.algorithm).toBe('string');
          expect(p.algorithm.length).toBeGreaterThan(0);
          expect(Number.isNaN(Date.parse(p.date))).toBe(false);
        }
      });

      it('forecasts dates after the last observed bar', () => {
        const lastObserved = new Date(bars[bars.length - 1].date).getTime();
        expect(new Date(predictions[0].date).getTime()).toBeGreaterThan(lastObserved);
      });
    });
  }
});
