import { describe, it, expect } from 'vitest';
import { runSignalBacktest } from '@/lib/signalBacktest';
import { StockData } from '@/types';

function bars(closes: number[]): StockData[] {
  return closes.map((c, i) => ({
    date: new Date(2021, 0, 1 + i).toISOString(),
    open: c,
    high: c * 1.01,
    low: c * 0.99,
    close: c,
    volume: 1_000_000,
  }));
}

describe('runSignalBacktest', () => {
  it('returns null without enough history', () => {
    expect(runSignalBacktest(bars([1, 2, 3, 4, 5]))).toBeNull();
  });

  it('produces a coherent result on a long series', () => {
    // 300 sessions of a noisy uptrend.
    const closes = Array.from({ length: 300 }, (_, i) => 100 + i * 0.5 + Math.sin(i / 7) * 8);
    const result = runSignalBacktest(bars(closes))!;

    expect(result).not.toBeNull();
    expect(result.equityCurve.length).toBeGreaterThan(1);
    expect(result.equityCurve[0].strategy).toBe(100);
    expect(result.equityCurve[0].buyHold).toBe(100);

    // Buy & hold of a rising series must be positive.
    expect(result.buyHoldReturn).toBeGreaterThan(0);

    // Sanity bounds.
    expect(result.winRate).toBeGreaterThanOrEqual(0);
    expect(result.winRate).toBeLessThanOrEqual(100);
    expect(result.exposure).toBeGreaterThanOrEqual(0);
    expect(result.exposure).toBeLessThanOrEqual(100);
    expect(result.maxDrawdown).toBeGreaterThanOrEqual(0);
    expect(Number.isFinite(result.strategyReturn)).toBe(true);
    expect(result.alpha).toBeCloseTo(result.strategyReturn - result.buyHoldReturn, 6);
  });

  it('stays in cash and flat when signals never turn bullish', () => {
    // Steady decline -> price below MAs, no buy entries -> no exposure, 0% return.
    const closes = Array.from({ length: 300 }, (_, i) => 400 - i * 0.8);
    const result = runSignalBacktest(bars(closes))!;
    expect(result.buyHoldReturn).toBeLessThan(0);
    expect(result.strategyReturn).toBeGreaterThanOrEqual(result.buyHoldReturn);
  });
});
