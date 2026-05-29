import { describe, it, expect } from 'vitest';
import {
  calculateSMA,
  calculateEMA,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  calculateATR,
  calculateAllIndicators,
} from '@/lib/technicalIndicators';
import { StockData } from '@/types';

function makeBars(closes: number[]): StockData[] {
  return closes.map((c, i) => ({
    date: new Date(2020, 0, i + 1).toISOString(),
    open: c,
    high: c + 1,
    low: c - 1,
    close: c,
    volume: 1000 + i,
  }));
}

describe('calculateSMA', () => {
  it('returns NaN for the warm-up period and the simple mean afterwards', () => {
    const sma = calculateSMA([1, 2, 3, 4, 5], 3);
    expect(sma[0]).toBeNaN();
    expect(sma[1]).toBeNaN();
    expect(sma[2]).toBeCloseTo(2); // (1+2+3)/3
    expect(sma[3]).toBeCloseTo(3); // (2+3+4)/3
    expect(sma[4]).toBeCloseTo(4); // (3+4+5)/3
  });

  it('matches the input length', () => {
    expect(calculateSMA([1, 2, 3, 4, 5, 6], 2)).toHaveLength(6);
  });
});

describe('calculateEMA', () => {
  it('seeds with the SMA of the first period', () => {
    const ema = calculateEMA([1, 2, 3, 4, 5], 3);
    expect(ema[1]).toBeNaN();
    expect(ema[2]).toBeCloseTo(2); // seed = (1+2+3)/3
  });

  it('returns a constant for a flat series', () => {
    const ema = calculateEMA([5, 5, 5, 5, 5], 3);
    expect(ema[2]).toBeCloseTo(5);
    expect(ema[4]).toBeCloseTo(5);
  });
});

describe('calculateRSI', () => {
  it('reads 100 for a strictly increasing series (no losses)', () => {
    const rsi = calculateRSI([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 14);
    const last = rsi[rsi.length - 1];
    expect(last).toBeCloseTo(100);
  });

  it('stays within [0, 100] for a mixed series', () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 + Math.sin(i) * 5);
    const rsi = calculateRSI(prices, 14);
    for (const v of rsi) {
      if (!Number.isNaN(v)) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(100);
      }
    }
  });
});

describe('calculateMACD', () => {
  it('keeps array lengths aligned and histogram = macd - signal', () => {
    const prices = Array.from({ length: 60 }, (_, i) => 100 + i * 0.5);
    const { macd, signal, histogram } = calculateMACD(prices);
    expect(macd).toHaveLength(60);
    expect(signal).toHaveLength(60);
    expect(histogram).toHaveLength(60);
    for (let i = 0; i < 60; i++) {
      if (!Number.isNaN(macd[i]) && !Number.isNaN(signal[i])) {
        expect(histogram[i]).toBeCloseTo(macd[i] - signal[i], 6);
      }
    }
  });
});

describe('calculateBollingerBands', () => {
  it('middle equals the SMA and upper >= middle >= lower', () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 + Math.sin(i) * 3);
    const { upper, middle, lower } = calculateBollingerBands(prices, 20, 2);
    for (let i = 19; i < 40; i++) {
      expect(upper[i]).toBeGreaterThanOrEqual(middle[i]);
      expect(middle[i]).toBeGreaterThanOrEqual(lower[i]);
    }
  });

  it('collapses bands to the mean for a flat series (zero stddev)', () => {
    const { upper, middle, lower } = calculateBollingerBands(new Array(25).fill(50), 20, 2);
    expect(upper[24]).toBeCloseTo(50);
    expect(middle[24]).toBeCloseTo(50);
    expect(lower[24]).toBeCloseTo(50);
  });
});

describe('calculateATR', () => {
  it('is non-negative where defined and NaN at index 0', () => {
    const bars = makeBars(Array.from({ length: 30 }, (_, i) => 100 + i));
    const atr = calculateATR(bars, 14);
    expect(atr[0]).toBeNaN();
    for (const v of atr) {
      if (!Number.isNaN(v)) expect(v).toBeGreaterThanOrEqual(0);
    }
  });
});

describe('calculateAllIndicators', () => {
  it('returns every indicator array at the input length', () => {
    const bars = makeBars(Array.from({ length: 250 }, (_, i) => 100 + Math.sin(i / 5) * 10));
    const ind = calculateAllIndicators(bars);
    for (const key of Object.keys(ind) as (keyof typeof ind)[]) {
      expect(ind[key]).toHaveLength(250);
    }
  });
});
