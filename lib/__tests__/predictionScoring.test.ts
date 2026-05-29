import { describe, it, expect } from 'vitest';
import {
  scoreLoggedPredictions,
  holdoutBacktest,
  gradeFromDirectional,
  buildActualIndex,
} from '@/lib/predictionScoring';
import { LoggedPrediction } from '@/lib/predictionLog';
import { StockData } from '@/types';

function bars(closes: number[], startDay = 1): StockData[] {
  return closes.map((c, i) => ({
    date: new Date(Date.UTC(2024, 0, startDay + i)).toISOString(),
    open: c,
    high: c,
    low: c,
    close: c,
    volume: 1000,
  }));
}

describe('buildActualIndex', () => {
  it('indexes closes by calendar day', () => {
    const idx = buildActualIndex(bars([10, 20, 30]));
    expect(idx.byDay.get('2024-01-01')).toBe(10);
    expect(idx.byDay.get('2024-01-03')).toBe(30);
    expect(idx.days).toHaveLength(3);
  });
});

describe('scoreLoggedPredictions', () => {
  it('resolves past points and marks future points pending', () => {
    // 30 days; the bar on 2024-01-15 closes at 110, the rest at 100.
    const closes = Array.from({ length: 30 }, (_, i) => (i === 14 ? 110 : 100));
    const actual = bars(closes);

    const entry: LoggedPrediction = {
      id: 'x',
      symbol: 'TEST',
      createdAt: Date.UTC(2024, 0, 1),
      basePrice: 100,
      horizon: 30,
      models: {
        good: [{ date: '2024-01-15', predicted: 120 }], // up call, actual up -> hit
        future: [{ date: '2024-06-01', predicted: 130 }], // beyond data -> pending
      },
    };

    const scores = scoreLoggedPredictions([entry], actual);
    const good = scores.find((s) => s.model === 'good')!;
    const future = scores.find((s) => s.model === 'future')!;

    expect(good.resolved).toBe(1);
    expect(good.pending).toBe(0);
    expect(good.directionalAccuracy).toBe(100);

    expect(future.resolved).toBe(0);
    expect(future.pending).toBe(1);
  });

  it('counts a wrong-direction call as a miss', () => {
    const actual = bars(Array.from({ length: 20 }, (_, i) => (i === 9 ? 80 : 100)));
    const entry: LoggedPrediction = {
      id: 'y',
      symbol: 'TEST',
      createdAt: Date.UTC(2024, 0, 1),
      basePrice: 100,
      horizon: 20,
      models: { bad: [{ date: '2024-01-10', predicted: 130 }] }, // predicted up, actual down
    };
    const score = scoreLoggedPredictions([entry], actual)[0];
    expect(score.resolved).toBe(1);
    expect(score.directionalAccuracy).toBe(0);
  });
});

describe('holdoutBacktest', () => {
  it('grades a model that calls the trend correctly', () => {
    const data = bars(Array.from({ length: 80 }, (_, i) => 100 + i)); // steady uptrend
    const generators = {
      trendUp: (train: StockData[], days: number) =>
        Array.from({ length: days }, (_, k) => ({
          date: `2099-01-${String(k + 1).padStart(2, '0')}`,
          predicted: train[train.length - 1].close + (k + 1),
        })),
    };
    const [score] = holdoutBacktest(data, generators, 10);
    expect(score.model).toBe('trendUp');
    expect(score.testedPoints).toBe(10);
    expect(score.directionalAccuracy).toBe(100);
  });

  it('returns nothing without enough history', () => {
    expect(holdoutBacktest(bars([1, 2, 3]), {}, 10)).toEqual([]);
  });
});

describe('gradeFromDirectional', () => {
  it('maps accuracy buckets to letter grades', () => {
    expect(gradeFromDirectional(65)).toBe('A');
    expect(gradeFromDirectional(56)).toBe('B');
    expect(gradeFromDirectional(51)).toBe('C');
    expect(gradeFromDirectional(46)).toBe('D');
    expect(gradeFromDirectional(40)).toBe('F');
  });
});
