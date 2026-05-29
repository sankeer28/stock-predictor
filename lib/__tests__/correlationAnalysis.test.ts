import { describe, it, expect } from 'vitest';
import {
  calculateCorrelation,
  calculateCorrelationMatrix,
  calculateReturns,
  categorizeSector,
} from '@/lib/correlationAnalysis';
import { StockData } from '@/types';

function makeBars(closes: number[]): StockData[] {
  return closes.map((c, i) => ({
    date: new Date(2022, 0, i + 1).toISOString(),
    open: c,
    high: c,
    low: c,
    close: c,
    volume: 1000,
  }));
}

describe('calculateCorrelation', () => {
  it('returns ~1 for a perfectly positive linear relationship', () => {
    const x = [1, 2, 3, 4, 5];
    const y = x.map((v) => 2 * v + 1);
    expect(calculateCorrelation(x, y)).toBeCloseTo(1, 5);
  });

  it('returns ~-1 for a perfectly negative relationship', () => {
    const x = [1, 2, 3, 4, 5];
    const y = x.map((v) => -3 * v);
    expect(calculateCorrelation(x, y)).toBeCloseTo(-1, 5);
  });

  it('returns 0 for a constant array (no variance)', () => {
    expect(calculateCorrelation([1, 2, 3], [5, 5, 5])).toBe(0);
  });

  it('returns 0 for empty input', () => {
    expect(calculateCorrelation([], [])).toBe(0);
  });
});

describe('calculateReturns', () => {
  it('computes percentage day-over-day returns', () => {
    const returns = calculateReturns(makeBars([100, 110, 99]));
    expect(returns).toHaveLength(2);
    expect(returns[0]).toBeCloseTo(10);
    expect(returns[1]).toBeCloseTo(-10);
  });
});

describe('calculateCorrelationMatrix', () => {
  it('has a unit diagonal and ~1 correlation for identical series', () => {
    const a = makeBars([10, 11, 12, 13, 14]);
    const b = makeBars([10, 11, 12, 13, 14]);
    const result = calculateCorrelationMatrix([
      { symbol: 'A', data: a },
      { symbol: 'B', data: b },
    ]);
    expect(result.symbols).toEqual(['A', 'B']);
    expect(result.matrix[0][0]).toBe(1);
    expect(result.matrix[1][1]).toBe(1);
    expect(result.matrix[0][1]).toBeCloseTo(1, 5);
  });
});

describe('categorizeSector', () => {
  it('maps keywords to broad sectors', () => {
    expect(categorizeSector('Computer Software')).toBe('Technology');
    expect(categorizeSector('National Commercial Bank')).toBe('Finance');
    expect(categorizeSector('Pharmaceutical Preparations')).toBe('Healthcare');
    expect(categorizeSector('Crude Petroleum & Natural Gas')).toBe('Energy');
  });

  it('falls back to Technology for empty input', () => {
    expect(categorizeSector('')).toBe('Technology');
  });
});
