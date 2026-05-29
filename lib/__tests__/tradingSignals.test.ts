import { describe, it, expect } from 'vitest';
import { generateTradingSignal } from '@/lib/tradingSignals';
import { StockData, TechnicalIndicators } from '@/types';

const LEN = 12;
const LAST = LEN - 1;

/** Build an indicators object where every series is a constant `fill` value. */
function fill(value: number): number[] {
  return new Array(LEN).fill(value);
}

function makeIndicators(overrides: Partial<TechnicalIndicators>): TechnicalIndicators {
  return {
    ma20: fill(NaN),
    ma50: fill(NaN),
    ma200: fill(NaN),
    rsi: fill(NaN),
    macd: fill(NaN),
    macdSignal: fill(NaN),
    macdHistogram: fill(NaN),
    bbUpper: fill(NaN),
    bbMiddle: fill(NaN),
    bbLower: fill(NaN),
    volumeMA: fill(NaN),
    ema12: fill(NaN),
    ema26: fill(NaN),
    ...overrides,
  };
}

function makeBars(lastClose: number, fiveAgoClose: number): StockData[] {
  const bars: StockData[] = [];
  for (let i = 0; i < LEN; i++) {
    let close = 100;
    if (i === LAST) close = lastClose;
    else if (i === LAST - 5) close = fiveAgoClose;
    bars.push({
      date: new Date(2021, 0, i + 1).toISOString(),
      open: close,
      high: close + 1,
      low: close - 1,
      close,
      volume: 1000, // well below volumeMA so the volume rule stays neutral
    });
  }
  return bars;
}

describe('generateTradingSignal', () => {
  it('returns a buy-side signal when indicators line up bullishly', () => {
    const bars = makeBars(110, 100); // +10% 5-day momentum, price above MA20
    const indicators = makeIndicators({
      ma20: fill(100),
      ma50: fill(95),
      ma200: fill(90),
      rsi: fill(60),
      macd: fill(2),
      macdSignal: fill(1),
      bbUpper: fill(120),
      bbLower: fill(80),
      volumeMA: fill(1e9),
    });
    const signal = generateTradingSignal(bars, indicators);
    expect(['strong_buy', 'buy', 'weak_buy']).toContain(signal.type);
    expect(signal.confidence).toBeGreaterThan(0);
    expect(signal.reasons.length).toBeGreaterThan(0);
  });

  it('returns a sell-side signal when indicators line up bearishly', () => {
    const bars = makeBars(90, 100); // -10% 5-day momentum, price below MA20
    const indicators = makeIndicators({
      ma20: fill(100),
      ma50: fill(105),
      ma200: fill(110),
      rsi: fill(40),
      macd: fill(1),
      macdSignal: fill(2),
      bbUpper: fill(95),
      bbLower: fill(85),
      volumeMA: fill(1e9),
    });
    const signal = generateTradingSignal(bars, indicators);
    expect(['strong_sell', 'sell', 'weak_sell']).toContain(signal.type);
    expect(signal.confidence).toBeGreaterThan(0);
  });

  it('always returns a valid signal type from the union', () => {
    const bars = makeBars(100, 100);
    const signal = generateTradingSignal(bars, makeIndicators({ ma20: fill(100) }));
    expect([
      'strong_buy', 'buy', 'weak_buy', 'hold', 'weak_sell', 'sell', 'strong_sell',
    ]).toContain(signal.type);
  });
});
