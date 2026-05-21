import {
  ChartDataPoint,
  ChartPattern,
  ChartPatternMeta,
  ChartPatternType,
} from '@/types';
import { PatternSettings, DEFAULT_PATTERN_SETTINGS } from '@/types/patternSettings';

const PATTERN_LABELS: Record<ChartPatternType, string> = {
  trendline_support: 'Trendline Support',
  trendline_resistance: 'Trendline Resistance',
  horizontal_sr: 'Horizontal S/R',
  wedge_up: 'Rising Wedge',
  wedge_down: 'Falling Wedge',
  wedge: 'Symmetrical Wedge',
  triangle_ascending: 'Ascending Triangle',
  triangle_descending: 'Descending Triangle',
  triangle_symmetrical: 'Symmetrical Triangle',
  channel_up: 'Ascending Channel',
  channel: 'Sideways Channel',
  channel_down: 'Descending Channel',
  double_top: 'Double Top',
  double_bottom: 'Double Bottom',
  multiple_top: 'Multiple Top',
  multiple_bottom: 'Multiple Bottom',
  head_and_shoulders: 'Head & Shoulders',
  inverse_head_and_shoulders: 'Inv. Head & Shoulders',
};

const PATTERN_DIRECTION: Record<ChartPatternType, ChartPattern['direction']> = {
  trendline_support: 'bullish',
  trendline_resistance: 'bearish',
  horizontal_sr: 'neutral',
  wedge_up: 'bearish',
  wedge_down: 'bullish',
  wedge: 'neutral',
  triangle_ascending: 'bullish',
  triangle_descending: 'bearish',
  triangle_symmetrical: 'neutral',
  channel_up: 'bullish',
  channel: 'neutral',
  channel_down: 'bearish',
  double_top: 'bearish',
  double_bottom: 'bullish',
  multiple_top: 'bearish',
  multiple_bottom: 'bullish',
  head_and_shoulders: 'bearish',
  inverse_head_and_shoulders: 'bullish',
};

const VOLATILITY_FLOOR = 0.002;
const VOLATILITY_CAP = 0.2;
const MAX_PATTERN_OVERLAP = 0.7;

interface LineStats {
  slope: number;
  intercept: number;
  r2: number;
  avg: number;
  slopePct: number;
}

interface LineTouches {
  touches: number;
  ratio: number;
  avgDeviation: number;
}

interface LevelCluster {
  level: number;
  indexes: number[];
  std: number;
}

export function detectChartPatterns(
  data: ChartDataPoint[],
  settings: PatternSettings = DEFAULT_PATTERN_SETTINGS
): ChartPattern[] {
  const MIN_WINDOW = settings.minWindow;
  const MAX_PATTERNS = settings.maxPatterns;
  const MAX_PATTERNS_PER_TYPE = settings.maxPatternsPerType;
  const DETECTION_WINDOWS = settings.detectionWindows;
  const MIN_CONFIDENCE = settings.minConfidence;

  if (!data || data.length < MIN_WINDOW) {
    return [];
  }

  try {
    const fullDatasetWindow = data.length;
    const windows = Array.from(
      new Set(
        [...DETECTION_WINDOWS, fullDatasetWindow]
          .map(size => Math.min(size, data.length))
          .filter(size => size >= MIN_WINDOW)
          .sort((a, b) => a - b)
      )
    );

    const patterns: ChartPattern[] = [];
    const EARLY_EXIT_THRESHOLD = Math.ceil(MAX_PATTERNS * 1.5);

    for (const windowSize of windows) {
      if (patterns.length >= EARLY_EXIT_THRESHOLD) break;

      patterns.push(...detectTrendlines(data, windowSize, MIN_WINDOW));
      if (patterns.length >= EARLY_EXIT_THRESHOLD) break;

      patterns.push(...detectHorizontalSR(data, windowSize, MIN_WINDOW));
      if (patterns.length >= EARLY_EXIT_THRESHOLD) break;

      patterns.push(...detectWedges(data, windowSize, MIN_WINDOW));
      if (patterns.length >= EARLY_EXIT_THRESHOLD) break;

      patterns.push(...detectTriangles(data, windowSize, MIN_WINDOW));
      if (patterns.length >= EARLY_EXIT_THRESHOLD) break;

      patterns.push(...detectChannels(data, windowSize, MIN_WINDOW));
    }

    if (patterns.length < EARLY_EXIT_THRESHOLD) {
      patterns.push(...detectDoubleTops(data, MIN_WINDOW));
      patterns.push(...detectDoubleBottoms(data, MIN_WINDOW));
      patterns.push(...detectMultipleTops(data, MIN_WINDOW));
      patterns.push(...detectMultipleBottoms(data, MIN_WINDOW));
      patterns.push(...detectHeadAndShoulders(data, MIN_WINDOW));
      patterns.push(...detectInverseHeadAndShoulders(data, MIN_WINDOW));
    }

    const filteredPatterns = patterns.filter(p => p.confidence >= MIN_CONFIDENCE);
    return consolidatePatterns(filteredPatterns, MAX_PATTERNS, MAX_PATTERNS_PER_TYPE);
  } catch (error) {
    console.error('Pattern detection failed', error);
    return [];
  }
}

function detectTrendlines(data: ChartDataPoint[], windowSize: number, minWindow: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);

  if (slice.length < minWindow) return [];

  const lows = slice.map(point => getLow(point));
  const highs = slice.map(point => getHigh(point));
  const volatilityPct = getWindowVolatilityPct(data, startIndex, data.length - 1);
  const supportTolerance = getAdaptiveTolerance(0.006, volatilityPct, 1.2);
  const resistanceTolerance = getAdaptiveTolerance(0.006, volatilityPct, 1.2);

  // Adaptive pivot distance: ~1 pivot per 20 bars, min 2
  const pivotDist = Math.max(2, Math.floor(slice.length / 20));
  const troughIdxs = findLocalExtrema(lows, pivotDist, 'min');
  const peakIdxs = findLocalExtrema(highs, pivotDist, 'max');

  // Swing-point regression: fit trendline through local extrema only
  const lowStats = linearRegressionPoints(troughIdxs.map(i => ({ x: i, y: lows[i] })));
  const highStats = linearRegressionPoints(peakIdxs.map(i => ({ x: i, y: highs[i] })));

  const patterns: ChartPattern[] = [];
  let supportCandidate: ChartPattern | null = null;
  let resistanceCandidate: ChartPattern | null = null;

  if (lowStats) {
    const touches = countLineTouches(lows, lowStats.slope, lowStats.intercept, supportTolerance);
    const minTouches = Math.max(2, Math.round(slice.length * 0.12));
    if (
      lowStats.slopePct >= -0.0005 &&
      touches.touches >= minTouches &&
      lowStats.r2 >= 0.25 &&
      touches.avgDeviation <= supportTolerance * 1.2
    ) {
      const confidence = clamp(
        0.40 +
          Math.min(0.30, (touches.touches / slice.length) * 1.5) +
          Math.min(0.30, lowStats.r2 * 1.2),
        0,
        0.95
      );
      supportCandidate = buildPattern(data, 'trendline_support', startIndex, data.length - 1, confidence, {
        slopePct: lowStats.slopePct * 100,
        touches: touches.touches,
        r2: lowStats.r2,
        priceMin: Math.min(...lows),
        priceMax: Math.max(...lows),
      });
    }
  }

  if (highStats) {
    const touches = countLineTouches(
      highs,
      highStats.slope,
      highStats.intercept,
      resistanceTolerance
    );
    const minTouches = Math.max(2, Math.round(slice.length * 0.12));
    if (
      highStats.slopePct <= 0.0005 &&
      touches.touches >= minTouches &&
      highStats.r2 >= 0.25 &&
      touches.avgDeviation <= resistanceTolerance * 1.2
    ) {
      const confidence = clamp(
        0.40 +
          Math.min(0.30, (touches.touches / slice.length) * 1.5) +
          Math.min(0.30, highStats.r2 * 1.2),
        0,
        0.95
      );
      resistanceCandidate = buildPattern(data, 'trendline_resistance', startIndex, data.length - 1, confidence, {
        slopePct: highStats.slopePct * 100,
        touches: touches.touches,
        r2: highStats.r2,
        priceMin: Math.min(...highs),
        priceMax: Math.max(...highs),
      });
    }
  }

  // Emit both — consolidation handles de-duplication across windows
  if (supportCandidate) patterns.push(supportCandidate);
  if (resistanceCandidate) patterns.push(resistanceCandidate);

  return patterns;
}

function detectHorizontalSR(data: ChartDataPoint[], windowSize: number, minWindow: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);
  if (slice.length < minWindow) return [];

  const highs = slice.map(point => getHigh(point));
  const lows = slice.map(point => getLow(point));
  const volatilityPct = getWindowVolatilityPct(data, startIndex, startIndex + slice.length - 1);
  const clusterTolerance = getAdaptiveTolerance(0.008, volatilityPct, 1.5);

  const highPeaks = findLocalExtrema(highs, 2, 'max');
  const lowTroughs = findLocalExtrema(lows, 2, 'min');

  const highClusters = clusterLevels(highs, highPeaks, clusterTolerance);
  const lowClusters = clusterLevels(lows, lowTroughs, clusterTolerance);

  const topCluster = highClusters.find(cluster => cluster.indexes.length >= 2);
  const bottomCluster = lowClusters.find(cluster => cluster.indexes.length >= 2);

  if (!topCluster || !bottomCluster) {
    return [];
  }

  const bandWidth = topCluster.level - bottomCluster.level;
  const midpoint = (topCluster.level + bottomCluster.level) / 2;

  if (midpoint === 0 || bandWidth / midpoint > 0.08) {
    return [];
  }

  const combinedIndexes = [...topCluster.indexes, ...bottomCluster.indexes];
  const start = startIndex + Math.max(0, Math.min(...combinedIndexes) - 3);
  const end = startIndex + Math.min(slice.length - 1, Math.max(...combinedIndexes) + 3);

  const confidence = clamp(
    0.45 +
      Math.min(0.25, combinedIndexes.length * 0.06) +
      Math.min(0.25, (0.08 - bandWidth / midpoint) * 2.5),
    0,
    0.95
  );

  return [
    buildPattern(data, 'horizontal_sr', start, Math.min(end, data.length - 1), confidence, {
      resistance: topCluster.level,
      support: bottomCluster.level,
      touches: combinedIndexes.length,
      priceMin: bottomCluster.level,
      priceMax: topCluster.level,
    }),
  ];
}

function detectWedges(data: ChartDataPoint[], windowSize: number, minWindow: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);
  if (slice.length < minWindow) return [];

  const highs = slice.map(point => getHigh(point));
  const lows = slice.map(point => getLow(point));
  const volatilityPct = getWindowVolatilityPct(data, startIndex, data.length - 1);
  const touchTolerance = getAdaptiveTolerance(0.005, volatilityPct, 1);

  const highStats = linearRegression(highs);
  const lowStats = linearRegression(lows);
  if (!highStats || !lowStats) return [];

  const rangeStats = calcRangeCompression(slice.length, highStats, lowStats);
  if (!rangeStats) return [];

  const { rangeStart, rangeEnd, compressionRatio } = rangeStats;
  if (compressionRatio > 0.92) return [];

  const upperTouches = countLineTouches(highs, highStats.slope, highStats.intercept, touchTolerance);
  const lowerTouches = countLineTouches(lows, lowStats.slope, lowStats.intercept, touchTolerance);

  if (upperTouches.touches < 2 || lowerTouches.touches < 2) {
    return [];
  }

  const slopeHigh = highStats.slopePct;
  const slopeLow = lowStats.slopePct;
  const patterns: ChartPattern[] = [];
  const baseMeta: ChartPatternMeta = {
    slopeHigh: slopeHigh * 100,
    slopeLow: slopeLow * 100,
    rangeCompression: compressionRatio,
    touchesHigh: upperTouches.touches,
    touchesLow: lowerTouches.touches,
    priceMin: Math.min(...lows, rangeEnd, rangeStart),
    priceMax: Math.max(...highs, rangeEnd, rangeStart),
  };

  if (slopeHigh > 0 && slopeLow > 0 && slopeLow > slopeHigh * 1.2) {
    const confidence = clamp(
      0.50 +
        Math.min(0.25, (1 - compressionRatio) * 1.5) +
        Math.min(0.15, (slopeLow - slopeHigh) * 400) +
        Math.min(0.10, lowerTouches.touches * 0.06),
      0,
      0.95
    );
    patterns.push(
      buildPattern(data, 'wedge_up', startIndex, data.length - 1, confidence, baseMeta)
    );
  } else if (slopeHigh < 0 && slopeLow < 0 && Math.abs(slopeHigh) > Math.abs(slopeLow) * 1.2) {
    const confidence = clamp(
      0.50 +
        Math.min(0.25, (1 - compressionRatio) * 1.5) +
        Math.min(0.15, (Math.abs(slopeHigh) - Math.abs(slopeLow)) * 400) +
        Math.min(0.10, upperTouches.touches * 0.06),
      0,
      0.95
    );
    patterns.push(
      buildPattern(data, 'wedge_down', startIndex, data.length - 1, confidence, baseMeta)
    );
  } else if (
    Math.sign(slopeHigh) !== Math.sign(slopeLow) &&
    Math.abs(Math.abs(slopeHigh) - Math.abs(slopeLow)) < 0.00035 &&
    Math.abs(slopeHigh) > 0.0002 &&
    Math.abs(slopeLow) > 0.0002
  ) {
    const confidence = clamp(
      0.4 +
        Math.min(0.25, (1 - compressionRatio) * 1.3) +
        Math.min(0.2, (0.00035 - Math.abs(Math.abs(slopeHigh) - Math.abs(slopeLow))) * 600) +
        Math.min(0.1, (upperTouches.touches + lowerTouches.touches) * 0.03),
      0,
      0.88
    );
    patterns.push(
      buildPattern(data, 'wedge', startIndex, data.length - 1, confidence, baseMeta)
    );
  }

  return patterns;
}

function detectTriangles(data: ChartDataPoint[], windowSize: number, minWindow: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);
  if (slice.length < minWindow) return [];

  const highs = slice.map(point => getHigh(point));
  const lows = slice.map(point => getLow(point));
  const volatilityPct = getWindowVolatilityPct(data, startIndex, data.length - 1);
  const touchTolerance = getAdaptiveTolerance(0.0055, volatilityPct, 1.05);

  const highStats = linearRegression(highs);
  const lowStats = linearRegression(lows);
  if (!highStats || !lowStats) return [];

  const rangeStats = calcRangeCompression(slice.length, highStats, lowStats);
  if (!rangeStats || rangeStats.compressionRatio > 0.93) return [];

  const upperTouches = countLineTouches(
    highs,
    highStats.slope,
    highStats.intercept,
    touchTolerance
  );
  const lowerTouches = countLineTouches(
    lows,
    lowStats.slope,
    lowStats.intercept,
    touchTolerance
  );

  if (upperTouches.touches < 2 || lowerTouches.touches < 2) {
    return [];
  }

  const patterns: ChartPattern[] = [];
  const slopeHigh = highStats.slopePct;
  const slopeLow = lowStats.slopePct;
  const flatThreshold = 0.00015;
  const trendThreshold = 0.0002;

  const baseMeta: ChartPatternMeta = {
    slopeHigh: slopeHigh * 100,
    slopeLow: slopeLow * 100,
    rangeCompression: rangeStats.compressionRatio,
    touchesHigh: upperTouches.touches,
    touchesLow: lowerTouches.touches,
    priceMin: Math.min(...lows),
    priceMax: Math.max(...highs),
  };

  const start = startIndex;
  const end = data.length - 1;

  if (Math.abs(slopeHigh) <= flatThreshold && slopeLow >= trendThreshold) {
    const confidence = clamp(
      0.50 +
        Math.min(0.25, slopeLow * 1000) +
        Math.min(0.20, (1 - rangeStats.compressionRatio) * 1.8) +
        Math.min(0.10, lowerTouches.touches * 0.06),
      0,
      0.95
    );
    patterns.push(buildPattern(data, 'triangle_ascending', start, end, confidence, baseMeta));
  } else if (Math.abs(slopeLow) <= flatThreshold && slopeHigh <= -trendThreshold) {
    const confidence = clamp(
      0.50 +
        Math.min(0.25, Math.abs(slopeHigh) * 1000) +
        Math.min(0.20, (1 - rangeStats.compressionRatio) * 1.8) +
        Math.min(0.10, upperTouches.touches * 0.06),
      0,
      0.95
    );
    patterns.push(buildPattern(data, 'triangle_descending', start, end, confidence, baseMeta));
  } else if (
    Math.sign(slopeHigh) !== Math.sign(slopeLow) &&
    Math.abs(Math.abs(slopeHigh) - Math.abs(slopeLow)) < 0.00025
  ) {
    const confidence = clamp(
      0.4 +
        Math.min(0.25, (1 - rangeStats.compressionRatio) * 1.4) +
        Math.min(0.2, (0.00025 - Math.abs(Math.abs(slopeHigh) - Math.abs(slopeLow))) * 700),
      0,
      0.85
    );
    patterns.push(
      buildPattern(data, 'triangle_symmetrical', start, end, confidence, baseMeta)
    );
  }

  return patterns;
}

function detectChannels(data: ChartDataPoint[], windowSize: number, minWindow: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);
  if (slice.length < minWindow) return [];

  const highs = slice.map(point => getHigh(point));
  const lows = slice.map(point => getLow(point));
  const volatilityPct = getWindowVolatilityPct(data, startIndex, data.length - 1);
  const touchTolerance = getAdaptiveTolerance(0.006, volatilityPct, 1.1);

  const highStats = linearRegression(highs);
  const lowStats = linearRegression(lows);
  if (!highStats || !lowStats) return [];

  const rangeStats = calcRangeCompression(slice.length, highStats, lowStats);
  if (!rangeStats) return [];

  const slopeDiff = Math.abs(highStats.slopePct - lowStats.slopePct);
  const sameDirection = Math.sign(highStats.slopePct) === Math.sign(lowStats.slopePct);
  const rangeDelta =
    rangeStats.rangeStart === 0
      ? 0
      : Math.abs(rangeStats.rangeEnd - rangeStats.rangeStart) / rangeStats.rangeStart;

  const upperTouches = countLineTouches(highs, highStats.slope, highStats.intercept, touchTolerance);
  const lowerTouches = countLineTouches(lows, lowStats.slope, lowStats.intercept, touchTolerance);

  if (upperTouches.touches < 2 || lowerTouches.touches < 2) return [];

  const baseMeta: ChartPatternMeta = {
    slopeHigh: highStats.slopePct * 100,
    slopeLow: lowStats.slopePct * 100,
    rangeDelta,
    touchesHigh: upperTouches.touches,
    touchesLow: lowerTouches.touches,
    priceMin: Math.min(...lows),
    priceMax: Math.max(...highs),
  };

  const start = startIndex;
  const end = data.length - 1;
  const patterns: ChartPattern[] = [];

  if (sameDirection && slopeDiff < 0.00035 && rangeDelta < 0.3) {
    if (highStats.slopePct > 0.00015) {
      const confidence = clamp(
        0.50 +
          Math.min(0.25, highStats.slopePct * 1500) +
          Math.min(0.20, (0.3 - rangeDelta) * 1.0) +
          Math.min(0.10, (upperTouches.touches + lowerTouches.touches) * 0.04),
        0,
        0.95
      );
      patterns.push(buildPattern(data, 'channel_up', start, end, confidence, baseMeta));
    } else if (highStats.slopePct < -0.00015) {
      const confidence = clamp(
        0.50 +
          Math.min(0.25, Math.abs(highStats.slopePct) * 1500) +
          Math.min(0.20, (0.3 - rangeDelta) * 1.0) +
          Math.min(0.10, (upperTouches.touches + lowerTouches.touches) * 0.04),
        0,
        0.95
      );
      patterns.push(buildPattern(data, 'channel_down', start, end, confidence, baseMeta));
    } else {
      const confidence = clamp(
        0.45 +
          Math.min(0.25, (0.3 - rangeDelta) * 1.0) +
          Math.min(0.15, (upperTouches.touches + lowerTouches.touches) * 0.04),
        0,
        0.85
      );
      patterns.push(buildPattern(data, 'channel', start, end, confidence, baseMeta));
    }
  }

  return patterns;
}

function detectDoubleTops(data: ChartDataPoint[], minWindow: number): ChartPattern[] {
  const highs = data.map(point => getHigh(point));
  const closes = data.map(point => getClose(point));
  const volumes = data.map(point => getVolume(point));
  const pivotDist = Math.max(3, Math.floor(data.length / 25));
  const peaks = findLocalExtrema(highs, pivotDist, 'max');
  const volatilityPct = getWindowVolatilityPct(data, 0, data.length - 1);
  const tolerance = clamp(volatilityPct * 2, 0.015, 0.05);
  const minSeparation = Math.max(3, pivotDist * 2);
  const results: ChartPattern[] = [];

  for (let i = peaks.length - 1; i >= 1; i--) {
    const second = peaks[i];
    for (let j = i - 1; j >= 0; j--) {
      const first = peaks[j];
      const separation = second - first;
      if (separation < minSeparation || separation > 200) continue;

      const firstVal = highs[first];
      const secondVal = highs[second];
      if (!isFinite(firstVal) || !isFinite(secondVal)) continue;

      const diff = Math.abs(firstVal - secondVal) / ((firstVal + secondVal) / 2);
      if (diff > tolerance) continue;

      // Declining volume at second top confirms distribution
      const firstVol = volumes[first];
      const secondVol = volumes[second];
      const volScore =
        isFinite(firstVol) && isFinite(secondVol) && firstVol > 0 && secondVol < firstVol
          ? Math.min(0.15, ((firstVol - secondVol) / firstVol) * 0.5)
          : 0;

      const postIndex = Math.min(data.length - 1, second + 5);
      const postMove = (closes[postIndex] - secondVal) / (secondVal || 1);
      const dropScore = postMove < 0 ? Math.min(0.15, Math.abs(postMove) * 3) : 0;

      const confidence = clamp(
        0.55 + Math.min(0.20, (tolerance - diff) * 12) + dropScore + volScore,
        0,
        0.95
      );

      const start = Math.max(0, first - 5);
      const end = Math.min(data.length - 1, second + 5);
      results.push(
        buildPattern(data, 'double_top', start, end, confidence, {
          level: (firstVal + secondVal) / 2,
          touches: 2,
        })
      );
      if (results.length >= 3) return results;
    }
  }

  return results;
}

function detectDoubleBottoms(data: ChartDataPoint[], minWindow: number): ChartPattern[] {
  const lows = data.map(point => getLow(point));
  const closes = data.map(point => getClose(point));
  const volumes = data.map(point => getVolume(point));
  const pivotDist = Math.max(3, Math.floor(data.length / 25));
  const troughs = findLocalExtrema(lows, pivotDist, 'min');
  const volatilityPct = getWindowVolatilityPct(data, 0, data.length - 1);
  const tolerance = clamp(volatilityPct * 2, 0.015, 0.05);
  const minSeparation = Math.max(3, pivotDist * 2);
  const results: ChartPattern[] = [];

  for (let i = troughs.length - 1; i >= 1; i--) {
    const second = troughs[i];
    for (let j = i - 1; j >= 0; j--) {
      const first = troughs[j];
      const separation = second - first;
      if (separation < minSeparation || separation > 200) continue;

      const firstVal = lows[first];
      const secondVal = lows[second];
      if (!isFinite(firstVal) || !isFinite(secondVal)) continue;

      const diff = Math.abs(firstVal - secondVal) / ((firstVal + secondVal) / 2);
      if (diff > tolerance) continue;

      // Declining volume at second trough confirms capitulation easing
      const firstVol = volumes[first];
      const secondVol = volumes[second];
      const volScore =
        isFinite(firstVol) && isFinite(secondVol) && firstVol > 0 && secondVol < firstVol
          ? Math.min(0.15, ((firstVol - secondVol) / firstVol) * 0.5)
          : 0;

      const postIndex = Math.min(data.length - 1, second + 5);
      const postMove = (closes[postIndex] - secondVal) / (secondVal || 1);
      const bounceScore = postMove > 0 ? Math.min(0.15, postMove * 3) : 0;

      const confidence = clamp(
        0.55 + Math.min(0.20, (tolerance - diff) * 12) + bounceScore + volScore,
        0,
        0.95
      );

      const start = Math.max(0, first - 5);
      const end = Math.min(data.length - 1, second + 5);
      results.push(
        buildPattern(data, 'double_bottom', start, end, confidence, {
          level: (firstVal + secondVal) / 2,
          touches: 2,
        })
      );
      if (results.length >= 3) return results;
    }
  }

  return results;
}

function detectMultipleTops(data: ChartDataPoint[], minWindow: number): ChartPattern[] {
  const highs = data.map(point => getHigh(point));
  const pivotDist = Math.max(2, Math.floor(data.length / 25));
  const peaks = findLocalExtrema(highs, pivotDist, 'max').filter(index => index >= highs.length - 250);
  if (peaks.length < 3) return [];

  const volatilityPct = getWindowVolatilityPct(data, Math.max(0, highs.length - 250), highs.length - 1);
  const clusterTolerance = getAdaptiveTolerance(0.015, volatilityPct, 1.6);
  const clusters = clusterLevels(highs, peaks, clusterTolerance);
  const cluster = clusters.find(c => c.indexes.length >= 3);
  if (!cluster) return [];

  const start = Math.max(0, Math.min(...cluster.indexes) - 5);
  const end = Math.min(data.length - 1, Math.max(...cluster.indexes) + 5);

  const confidence = clamp(
    0.55 + Math.min(0.30, cluster.indexes.length * 0.10),
    0,
    0.95
  );

  const padding = cluster.level * 0.01;

  return [
    buildPattern(data, 'multiple_top', start, end, confidence, {
      level: cluster.level,
      touches: cluster.indexes.length,
      priceMin: cluster.level - padding,
      priceMax: cluster.level + padding,
    }),
  ];
}

function detectMultipleBottoms(data: ChartDataPoint[], minWindow: number): ChartPattern[] {
  const lows = data.map(point => getLow(point));
  const pivotDist = Math.max(2, Math.floor(data.length / 25));
  const troughs = findLocalExtrema(lows, pivotDist, 'min').filter(index => index >= lows.length - 250);
  if (troughs.length < 3) return [];

  const volatilityPct = getWindowVolatilityPct(data, Math.max(0, lows.length - 250), lows.length - 1);
  const clusterTolerance = getAdaptiveTolerance(0.015, volatilityPct, 1.6);
  const clusters = clusterLevels(lows, troughs, clusterTolerance);
  const cluster = clusters.find(c => c.indexes.length >= 3);
  if (!cluster) return [];

  const start = Math.max(0, Math.min(...cluster.indexes) - 5);
  const end = Math.min(data.length - 1, Math.max(...cluster.indexes) + 5);

  const confidence = clamp(
    0.55 + Math.min(0.30, cluster.indexes.length * 0.10),
    0,
    0.95
  );

  const padding = cluster.level * 0.01;

  return [
    buildPattern(data, 'multiple_bottom', start, end, confidence, {
      level: cluster.level,
      touches: cluster.indexes.length,
      priceMin: cluster.level - padding,
      priceMax: cluster.level + padding,
    }),
  ];
}

function detectHeadAndShoulders(data: ChartDataPoint[], minWindow: number): ChartPattern[] {
  const highs = data.map(point => getHigh(point));
  const lows = data.map(point => getLow(point));
  const pivotDist = Math.max(3, Math.floor(data.length / 20));
  const peaks = findLocalExtrema(highs, pivotDist, 'max');
  if (peaks.length < 3) return [];
  const volatilityPct = getWindowVolatilityPct(data, 0, data.length - 1);
  const patternTolerance = getAdaptiveTolerance(0.025, volatilityPct, 4.5);
  const results: ChartPattern[] = [];

  for (let i = peaks.length - 3; i >= 0; i--) {
    const left = peaks[i];
    const head = peaks[i + 1];
    const right = peaks[i + 2];

    if (head - left < pivotDist * 2 || right - head < pivotDist * 2) continue;
    if (right - left > 250) continue;

    const leftVal = highs[left];
    const headVal = highs[head];
    const rightVal = highs[right];

    if (!isFinite(leftVal) || !isFinite(headVal) || !isFinite(rightVal)) continue;

    const headPremium = headVal / ((leftVal + rightVal) / 2) - 1;
    if (headPremium < Math.max(0.02, patternTolerance)) continue;

    const shoulderDiff = Math.abs(leftVal - rightVal) / ((leftVal + rightVal) / 2);
    if (shoulderDiff > Math.max(0.04, patternTolerance * 1.4)) continue;

    const leftNeckline = findRangeMin(lows, left, head);
    const rightNeckline = findRangeMin(lows, head, right);
    const necklineDiff =
      Math.abs(leftNeckline.value - rightNeckline.value) /
      (((leftNeckline.value + rightNeckline.value) / 2) || 1);
    if (necklineDiff > Math.max(0.04, patternTolerance * 1.5)) continue;

    const confidence = clamp(
      0.60 +
        Math.min(0.20, headPremium * 4) +
        Math.min(0.15, (0.04 - necklineDiff) * 4),
      0,
      0.95
    );

    const start = Math.max(0, left - 5);
    const end = Math.min(data.length - 1, right + 5);

    results.push(
      buildPattern(data, 'head_and_shoulders', start, end, confidence, {
        leftShoulder: leftVal,
        head: headVal,
        rightShoulder: rightVal,
        neckline: (leftNeckline.value + rightNeckline.value) / 2,
      })
    );
    if (results.length >= 2) break;
  }

  return results;
}

function detectInverseHeadAndShoulders(data: ChartDataPoint[], minWindow: number): ChartPattern[] {
  const lows = data.map(point => getLow(point));
  const highs = data.map(point => getHigh(point));
  const pivotDist = Math.max(3, Math.floor(data.length / 20));
  const troughs = findLocalExtrema(lows, pivotDist, 'min');
  if (troughs.length < 3) return [];
  const volatilityPct = getWindowVolatilityPct(data, 0, data.length - 1);
  const patternTolerance = getAdaptiveTolerance(0.025, volatilityPct, 4.5);
  const results: ChartPattern[] = [];

  for (let i = troughs.length - 3; i >= 0; i--) {
    const left = troughs[i];
    const head = troughs[i + 1];
    const right = troughs[i + 2];

    if (head - left < pivotDist * 2 || right - head < pivotDist * 2) continue;
    if (right - left > 250) continue;

    const leftVal = lows[left];
    const headVal = lows[head];
    const rightVal = lows[right];

    if (!isFinite(leftVal) || !isFinite(headVal) || !isFinite(rightVal)) continue;

    // Head must be below both shoulders
    const headDiscount = ((leftVal + rightVal) / 2) / headVal - 1;
    if (headDiscount < Math.max(0.02, patternTolerance)) continue;

    const shoulderDiff = Math.abs(leftVal - rightVal) / ((leftVal + rightVal) / 2);
    if (shoulderDiff > Math.max(0.04, patternTolerance * 1.4)) continue;

    // Neckline connects the peaks between each pair of troughs
    const leftNeckline = findRangeMax(highs, left, head);
    const rightNeckline = findRangeMax(highs, head, right);
    const necklineDiff =
      Math.abs(leftNeckline.value - rightNeckline.value) /
      (((leftNeckline.value + rightNeckline.value) / 2) || 1);
    if (necklineDiff > Math.max(0.04, patternTolerance * 1.5)) continue;

    const confidence = clamp(
      0.60 +
        Math.min(0.20, headDiscount * 4) +
        Math.min(0.15, (0.04 - necklineDiff) * 4),
      0,
      0.95
    );

    const start = Math.max(0, left - 5);
    const end = Math.min(data.length - 1, right + 5);

    results.push(
      buildPattern(data, 'inverse_head_and_shoulders', start, end, confidence, {
        leftShoulder: leftVal,
        head: headVal,
        rightShoulder: rightVal,
        neckline: (leftNeckline.value + rightNeckline.value) / 2,
      })
    );
    if (results.length >= 2) break;
  }

  return results;
}

function consolidatePatterns(
  patterns: ChartPattern[],
  maxPatterns: number = 20,
  maxPatternsPerType: number = 3
): ChartPattern[] {
  if (!patterns.length) return [];

  const sorted = [...patterns]
    .filter(pattern => pattern.startDate && pattern.endDate)
    .sort((a, b) => b.confidence - a.confidence);

  const perTypeCount = new Map<ChartPatternType, number>();
  const result: ChartPattern[] = [];

  for (const pattern of sorted) {
    const count = perTypeCount.get(pattern.type) ?? 0;
    if (count >= maxPatternsPerType) continue;
    const overlaps = result.some(
      existing =>
        existing.type === pattern.type &&
        getRangeOverlap(existing, pattern) >= MAX_PATTERN_OVERLAP
    );
    if (overlaps) continue;

    perTypeCount.set(pattern.type, count + 1);
    result.push(pattern);
    if (result.length >= maxPatterns) break;
  }

  return result;
}

function buildPattern(
  data: ChartDataPoint[],
  type: ChartPatternType,
  startIndex: number,
  endIndex: number,
  confidence: number,
  meta: ChartPatternMeta = {}
): ChartPattern {
  if (!data.length) {
    throw new Error('Cannot build pattern without data');
  }

  const safeStart = clampIndex(startIndex, 0, data.length - 1);
  const safeEnd = clampIndex(endIndex, safeStart, data.length - 1);

  const { minPrice, maxPrice } = getPriceRange(data, safeStart, safeEnd);

  const mergedMeta: ChartPatternMeta = {
    ...meta,
    priceMin: typeof meta.priceMin === 'number' ? meta.priceMin : minPrice,
    priceMax: typeof meta.priceMax === 'number' ? meta.priceMax : maxPrice,
  };

  return {
    id: `${type}-${data[safeEnd]?.date ?? safeEnd}-${safeStart}`,
    type,
    label: PATTERN_LABELS[type],
    direction: PATTERN_DIRECTION[type],
    startIndex: safeStart,
    endIndex: safeEnd,
    startDate: data[safeStart]?.date ?? '',
    endDate: data[safeEnd]?.date ?? '',
    confidence: clamp(confidence, 0, 0.99),
    meta: mergedMeta,
  };
}

function calcRangeCompression(
  length: number,
  highStats: LineStats,
  lowStats: LineStats
) {
  const startHigh = highStats.intercept;
  const startLow = lowStats.intercept;
  const endHigh = highStats.slope * (length - 1) + highStats.intercept;
  const endLow = lowStats.slope * (length - 1) + lowStats.intercept;

  const rangeStart = startHigh - startLow;
  const rangeEnd = endHigh - endLow;

  if (!isFinite(rangeStart) || !isFinite(rangeEnd) || rangeStart <= 0 || rangeEnd <= 0) {
    return null;
  }

  return {
    rangeStart,
    rangeEnd,
    compressionRatio: rangeEnd / rangeStart,
  };
}

function findLocalExtrema(
  values: number[],
  distance: number,
  type: 'max' | 'min'
): number[] {
  const indexes: number[] = [];
  for (let i = distance; i < values.length - distance; i++) {
    const current = values[i];
    if (!isFinite(current)) continue;

    let isExtreme = true;
    for (let j = 1; j <= distance; j++) {
      const prev = values[i - j];
      const next = values[i + j];
      if (!isFinite(prev) || !isFinite(next)) {
        isExtreme = false;
        break;
      }

      if (type === 'max' && (current < prev || current <= next)) {
        isExtreme = false;
        break;
      }

      if (type === 'min' && (current > prev || current >= next)) {
        isExtreme = false;
        break;
      }
    }

    if (isExtreme) {
      indexes.push(i);
    }
  }
  return indexes;
}

function clusterLevels(values: number[], indexes: number[], tolerancePct: number): LevelCluster[] {
  const clusters: LevelCluster[] = [];
  indexes.forEach(index => {
    const price = values[index];
    if (!isFinite(price)) return;

    let cluster = clusters.find(
      c => Math.abs(price - c.level) / c.level <= tolerancePct
    );

    if (!cluster) {
      cluster = { level: price, indexes: [], std: 0 };
      clusters.push(cluster);
    }

    cluster.indexes.push(index);
    cluster.level =
      (cluster.level * (cluster.indexes.length - 1) + price) / cluster.indexes.length;
    cluster.std = calcStd(
      cluster.indexes.map(i => values[i])
    );
  });

  return clusters.sort((a, b) => b.indexes.length - a.indexes.length);
}

function countLineTouches(
  values: number[],
  slope: number,
  intercept: number,
  tolerancePct: number
): LineTouches {
  let touches = 0;
  let totalDeviation = 0;

  values.forEach((val, idx) => {
    const expected = slope * idx + intercept;
    const denom = Math.max(1, Math.abs(expected));
    const diff = Math.abs(val - expected) / denom;
    if (diff <= tolerancePct) {
      touches++;
      totalDeviation += diff;
    }
  });

  return {
    touches,
    ratio: values.length ? touches / values.length : 0,
    avgDeviation: touches ? totalDeviation / touches : tolerancePct,
  };
}

function linearRegression(values: number[]): LineStats | null {
  const n = values.length;
  if (n < 5) return null;

  const xMean = (n - 1) / 2;
  const yMean = average(values);

  let numerator = 0;
  let denominator = 0;
  let ssTot = 0;
  let ssRes = 0;

  for (let i = 0; i < n; i++) {
    const x = i;
    const y = values[i];
    if (!isFinite(y)) return null;

    numerator += (x - xMean) * (y - yMean);
    denominator += (x - xMean) ** 2;
  }

  if (denominator === 0) return null;

  const slope = numerator / denominator;
  const intercept = yMean - slope * xMean;

  for (let i = 0; i < n; i++) {
    const predicted = slope * i + intercept;
    ssTot += (values[i] - yMean) ** 2;
    ssRes += (values[i] - predicted) ** 2;
  }

  const r2 = ssTot === 0 ? 0 : 1 - ssRes / ssTot;
  const avg = yMean;
  const slopePct = avg === 0 ? 0 : slope / avg;

  return { slope, intercept, r2, avg, slopePct };
}

function linearRegressionPoints(points: Array<{ x: number; y: number }>): LineStats | null {
  const n = points.length;
  if (n < 2) return null;

  const xMean = points.reduce((s, p) => s + p.x, 0) / n;
  const yMean = points.reduce((s, p) => s + p.y, 0) / n;

  let num = 0;
  let den = 0;
  for (const p of points) {
    num += (p.x - xMean) * (p.y - yMean);
    den += (p.x - xMean) ** 2;
  }
  if (den === 0) return null;

  const slope = num / den;
  const intercept = yMean - slope * xMean;

  let ssTot = 0;
  let ssRes = 0;
  for (const p of points) {
    ssTot += (p.y - yMean) ** 2;
    ssRes += (p.y - (slope * p.x + intercept)) ** 2;
  }

  const r2 = ssTot === 0 ? 1 : 1 - ssRes / ssTot;
  const slopePct = yMean === 0 ? 0 : slope / yMean;

  return { slope, intercept, r2, avg: yMean, slopePct };
}

function average(values: number[]): number {
  if (!values.length) return 0;
  const sum = values.reduce((acc, val) => acc + (isFinite(val) ? val : 0), 0);
  return sum / values.length;
}

function calcStd(values: number[]): number {
  if (values.length < 2) return 0;
  const avg = average(values);
  const variance =
    values.reduce((sum, val) => sum + (val - avg) ** 2, 0) / (values.length - 1);
  return Math.sqrt(variance);
}

function getPriceRange(data: ChartDataPoint[], start: number, end: number) {
  let minPrice = Number.POSITIVE_INFINITY;
  let maxPrice = Number.NEGATIVE_INFINITY;

  for (let i = start; i <= end; i++) {
    const point = data[i];
    if (!point) continue;

    const low = getLow(point);
    const high = getHigh(point);
    const close = getClose(point);

    minPrice = Math.min(minPrice, low, close);
    maxPrice = Math.max(maxPrice, high, close);
  }

  if (!isFinite(minPrice)) {
    minPrice = getClose(data[start] ?? data[0]);
  }

  if (!isFinite(maxPrice)) {
    maxPrice = getClose(data[end] ?? data[data.length - 1]);
  }

  return { minPrice, maxPrice };
}

function getWindowVolatilityPct(data: ChartDataPoint[], start: number, end: number) {
  if (!data.length) return VOLATILITY_FLOOR;

  const s = clampIndex(start, 0, data.length - 1);
  const e = clampIndex(end, s, data.length - 1);

  let prevClose = getClose(data[s]);
  let sumTR = 0;
  let sumPrice = 0;
  let count = 0;

  for (let i = s; i <= e; i++) {
    const point = data[i];
    if (!point) continue;
    const high = getHigh(point);
    const low = getLow(point);
    const close = getClose(point);
    const tr = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    sumTR += tr;
    sumPrice += close;
    prevClose = close;
    count++;
  }

  if (!count || sumPrice === 0) {
    return VOLATILITY_FLOOR;
  }

  const atr = sumTR / count;
  const avgPrice = sumPrice / count;
  const pct = avgPrice === 0 ? VOLATILITY_FLOOR : atr / avgPrice;
  return clamp(pct || VOLATILITY_FLOOR, VOLATILITY_FLOOR, VOLATILITY_CAP);
}

function getAdaptiveTolerance(base: number, volatilityPct: number, multiplier = 1) {
  return Math.max(base, volatilityPct * multiplier);
}

function getRangeOverlap(a: ChartPattern, b: ChartPattern) {
  const start = Math.max(a.startIndex, b.startIndex);
  const end = Math.min(a.endIndex, b.endIndex);
  if (end <= start) return 0;
  const minLen = Math.min(a.endIndex - a.startIndex + 1, b.endIndex - b.startIndex + 1);
  if (minLen <= 0) return 0;
  return (end - start + 1) / minLen;
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function clampIndex(index: number, min: number, max: number): number {
  return Math.min(Math.max(index, min), max);
}

function ensureNumber(value: number | undefined | null, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

function getClose(point: ChartDataPoint): number {
  return ensureNumber(
    point.close,
    ensureNumber(point.open ?? point.high ?? point.low, 0)
  );
}

function getHigh(point: ChartDataPoint): number {
  const fallback = getClose(point);
  return ensureNumber(point.high, fallback);
}

function getLow(point: ChartDataPoint): number {
  const fallback = getClose(point);
  return ensureNumber(point.low, fallback);
}

function getVolume(point: ChartDataPoint): number {
  return ensureNumber(point.volume, 0);
}

function findRangeMin(values: number[], start: number, end: number) {
  const s = Math.min(start, end);
  const e = Math.max(start, end);
  let minIndex = s;
  let minValue = values[s];

  for (let i = s; i <= e; i++) {
    if (values[i] < minValue) {
      minValue = values[i];
      minIndex = i;
    }
  }

  return { index: minIndex, value: minValue };
}

function findRangeMax(values: number[], start: number, end: number) {
  const s = Math.min(start, end);
  const e = Math.max(start, end);
  let maxIndex = s;
  let maxValue = values[s];

  for (let i = s; i <= e; i++) {
    if (values[i] > maxValue) {
      maxValue = values[i];
      maxIndex = i;
    }
  }

  return { index: maxIndex, value: maxValue };
}
