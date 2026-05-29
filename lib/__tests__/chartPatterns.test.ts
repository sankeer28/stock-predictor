import { describe, it, expect } from 'vitest';
import { detectChartPatterns } from '@/lib/chartPatterns';
import { ChartDataPoint } from '@/types';

function makePoints(closes: number[]): ChartDataPoint[] {
  return closes.map((c, i) => ({
    date: new Date(2023, 0, i + 1).toISOString(),
    open: c,
    high: c + Math.abs(c) * 0.01,
    low: c - Math.abs(c) * 0.01,
    close: c,
    volume: 1_000_000,
  }));
}

describe('detectChartPatterns', () => {
  it('returns an empty array for empty or too-short input', () => {
    expect(detectChartPatterns([])).toEqual([]);
    expect(detectChartPatterns(makePoints([1, 2, 3]))).toEqual([]);
  });

  it('never throws on a realistic series and returns an array', () => {
    const closes = Array.from({ length: 160 }, (_, i) => 100 + i * 0.3 + Math.sin(i / 8) * 6);
    const result = detectChartPatterns(makePoints(closes));
    expect(Array.isArray(result)).toBe(true);
  });

  it('returns structurally valid patterns', () => {
    const closes = Array.from({ length: 200 }, (_, i) => 100 + Math.sin(i / 10) * 15 + i * 0.1);
    const points = makePoints(closes);
    const patterns = detectChartPatterns(points);

    for (const p of patterns) {
      expect(typeof p.id).toBe('string');
      expect(typeof p.label).toBe('string');
      expect(['bullish', 'bearish', 'neutral']).toContain(p.direction);
      expect(typeof p.confidence).toBe('number');
      expect(Number.isFinite(p.confidence)).toBe(true);
      expect(p.startIndex).toBeLessThanOrEqual(p.endIndex);
      expect(p.startIndex).toBeGreaterThanOrEqual(0);
      expect(p.endIndex).toBeLessThan(points.length);
    }
  });
});
