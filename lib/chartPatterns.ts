import {
  ChartDataPoint,
  ChartPattern,
  ChartPatternMeta,
  ChartPatternType,
} from '@/types';

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
};

const DETECTION_WINDOWS = [60, 120, 200];
const MIN_WINDOW = 45;
const MAX_PATTERNS = 12;
const MAX_PATTERNS_PER_TYPE = 2;

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

export function detectChartPatterns(data: ChartDataPoint[]): ChartPattern[] {
  if (!data || data.length < MIN_WINDOW) {
    return [];
  }

  try {
    const windows = Array.from(
      new Set(
        DETECTION_WINDOWS.map(size => Math.min(size, data.length)).filter(
          size => size >= MIN_WINDOW
        )
      )
    );

    const patterns: ChartPattern[] = [];

    windows.forEach(windowSize => {
      patterns.push(...detectTrendlines(data, windowSize));
      patterns.push(...detectHorizontalSR(data, windowSize));
      patterns.push(...detectWedges(data, windowSize));
      patterns.push(...detectTriangles(data, windowSize));
      patterns.push(...detectChannels(data, windowSize));
    });

    patterns.push(...detectDoubleTops(data));
    patterns.push(...detectDoubleBottoms(data));
    patterns.push(...detectMultipleTops(data));
    patterns.push(...detectMultipleBottoms(data));
    patterns.push(...detectHeadAndShoulders(data));

    return consolidatePatterns(patterns);
  } catch (error) {
    console.error('Pattern detection failed', error);
    return [];
  }
}

function detectTrendlines(data: ChartDataPoint[], windowSize: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);

  if (slice.length < MIN_WINDOW) return [];

  const lows = slice.map(point => getLow(point));
  const highs = slice.map(point => getHigh(point));

  const lowStats = linearRegression(lows);
  const highStats = linearRegression(highs);

  const patterns: ChartPattern[] = [];

  if (lowStats) {
    const touches = countLineTouches(lows, lowStats.slope, lowStats.intercept, 0.007);
    const minTouches = Math.max(3, Math.round(slice.length * 0.18));
    if (lowStats.slopePct >= -0.0008 && touches.touches >= minTouches && lowStats.r2 >= 0.35) {
      const confidence = clamp(
        0.35 +
          Math.min(0.35, touches.touches / slice.length) +
          Math.min(0.3, lowStats.r2),
        0,
        0.95
      );
      patterns.push(
        buildPattern(data, 'trendline_support', startIndex, data.length - 1, confidence, {
          slopePct: lowStats.slopePct * 100,
          touches: touches.touches,
          r2: lowStats.r2,
          priceMin: Math.min(...lows),
          priceMax: Math.max(...lows),
        })
      );
    }
  }

  if (highStats) {
    const touches = countLineTouches(highs, highStats.slope, highStats.intercept, 0.007);
    const minTouches = Math.max(3, Math.round(slice.length * 0.18));
    if (highStats.slopePct <= 0.0008 && touches.touches >= minTouches && highStats.r2 >= 0.35) {
      const confidence = clamp(
        0.35 +
          Math.min(0.35, touches.touches / slice.length) +
          Math.min(0.3, highStats.r2),
        0,
        0.95
      );
      patterns.push(
        buildPattern(data, 'trendline_resistance', startIndex, data.length - 1, confidence, {
          slopePct: highStats.slopePct * 100,
          touches: touches.touches,
          r2: highStats.r2,
          priceMin: Math.min(...highs),
          priceMax: Math.max(...highs),
        })
      );
    }
  }

  return patterns;
}

function detectHorizontalSR(data: ChartDataPoint[], windowSize: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);
  if (slice.length < MIN_WINDOW) return [];

  const highs = slice.map(point => getHigh(point));
  const lows = slice.map(point => getLow(point));

  const highPeaks = findLocalExtrema(highs, 3, 'max');
  const lowTroughs = findLocalExtrema(lows, 3, 'min');

  const highClusters = clusterLevels(highs, highPeaks, 0.008);
  const lowClusters = clusterLevels(lows, lowTroughs, 0.008);

  const topCluster = highClusters.find(cluster => cluster.indexes.length >= 3);
  const bottomCluster = lowClusters.find(cluster => cluster.indexes.length >= 3);

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
    0.4 +
      Math.min(0.3, combinedIndexes.length * 0.05) +
      Math.min(0.25, (0.08 - bandWidth / midpoint) * 2),
    0,
    0.9
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

function detectWedges(data: ChartDataPoint[], windowSize: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);
  if (slice.length < MIN_WINDOW) return [];

  const highs = slice.map(point => getHigh(point));
  const lows = slice.map(point => getLow(point));

  const highStats = linearRegression(highs);
  const lowStats = linearRegression(lows);
  if (!highStats || !lowStats) return [];

  const rangeStats = calcRangeCompression(slice.length, highStats, lowStats);
  if (!rangeStats) return [];

  const { rangeStart, rangeEnd, compressionRatio } = rangeStats;
  if (compressionRatio > 0.92) return [];

  const upperTouches = countLineTouches(highs, highStats.slope, highStats.intercept, 0.008);
  const lowerTouches = countLineTouches(lows, lowStats.slope, lowStats.intercept, 0.008);

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
      0.45 +
        Math.min(0.25, (1 - compressionRatio) * 1.2) +
        Math.min(0.15, (slopeLow - slopeHigh) * 350) +
        Math.min(0.15, lowerTouches.touches * 0.05),
      0,
      0.92
    );
    patterns.push(
      buildPattern(data, 'wedge_up', startIndex, data.length - 1, confidence, baseMeta)
    );
  } else if (slopeHigh < 0 && slopeLow < 0 && Math.abs(slopeHigh) > Math.abs(slopeLow) * 1.2) {
    const confidence = clamp(
      0.45 +
        Math.min(0.25, (1 - compressionRatio) * 1.2) +
        Math.min(0.15, (Math.abs(slopeHigh) - Math.abs(slopeLow)) * 350) +
        Math.min(0.15, upperTouches.touches * 0.05),
      0,
      0.92
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

function detectTriangles(data: ChartDataPoint[], windowSize: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);
  if (slice.length < MIN_WINDOW) return [];

  const highs = slice.map(point => getHigh(point));
  const lows = slice.map(point => getLow(point));

  const highStats = linearRegression(highs);
  const lowStats = linearRegression(lows);
  if (!highStats || !lowStats) return [];

  const rangeStats = calcRangeCompression(slice.length, highStats, lowStats);
  if (!rangeStats || rangeStats.compressionRatio > 0.93) return [];

  const upperTouches = countLineTouches(highs, highStats.slope, highStats.intercept, 0.009);
  const lowerTouches = countLineTouches(lows, lowStats.slope, lowStats.intercept, 0.009);

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
      0.45 +
        Math.min(0.25, slopeLow * 800) +
        Math.min(0.2, (1 - rangeStats.compressionRatio) * 1.5) +
        Math.min(0.1, lowerTouches.touches * 0.05),
      0,
      0.9
    );
    patterns.push(buildPattern(data, 'triangle_ascending', start, end, confidence, baseMeta));
  } else if (Math.abs(slopeLow) <= flatThreshold && slopeHigh <= -trendThreshold) {
    const confidence = clamp(
      0.45 +
        Math.min(0.25, Math.abs(slopeHigh) * 800) +
        Math.min(0.2, (1 - rangeStats.compressionRatio) * 1.5) +
        Math.min(0.1, upperTouches.touches * 0.05),
      0,
      0.9
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

function detectChannels(data: ChartDataPoint[], windowSize: number): ChartPattern[] {
  const startIndex = Math.max(0, data.length - windowSize);
  const slice = data.slice(startIndex);
  if (slice.length < MIN_WINDOW) return [];

  const highs = slice.map(point => getHigh(point));
  const lows = slice.map(point => getLow(point));

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

  const upperTouches = countLineTouches(highs, highStats.slope, highStats.intercept, 0.009);
  const lowerTouches = countLineTouches(lows, lowStats.slope, lowStats.intercept, 0.009);

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
        0.4 +
          Math.min(0.25, highStats.slopePct * 1200) +
          Math.min(0.2, (0.3 - rangeDelta) * 0.8) +
          Math.min(0.15, (upperTouches.touches + lowerTouches.touches) * 0.03),
        0,
        0.9
      );
      patterns.push(buildPattern(data, 'channel_up', start, end, confidence, baseMeta));
    } else if (highStats.slopePct < -0.00015) {
      const confidence = clamp(
        0.4 +
          Math.min(0.25, Math.abs(highStats.slopePct) * 1200) +
          Math.min(0.2, (0.3 - rangeDelta) * 0.8) +
          Math.min(0.15, (upperTouches.touches + lowerTouches.touches) * 0.03),
        0,
        0.9
      );
      patterns.push(buildPattern(data, 'channel_down', start, end, confidence, baseMeta));
    } else {
      const confidence = clamp(
        0.35 +
          Math.min(0.2, (0.3 - rangeDelta) * 0.8) +
          Math.min(0.15, (upperTouches.touches + lowerTouches.touches) * 0.03),
        0,
        0.8
      );
      patterns.push(buildPattern(data, 'channel', start, end, confidence, baseMeta));
    }
  }

  return patterns;
}

function detectDoubleTops(data: ChartDataPoint[]): ChartPattern[] {
  const closes = data.map(point => getClose(point));
  const peaks = findLocalExtrema(closes, 3, 'max');
  const tolerance = 0.012;
  const minSeparation = 5;

  for (let i = peaks.length - 1; i >= 1; i--) {
    const second = peaks[i];
    for (let j = i - 1; j >= 0; j--) {
      const first = peaks[j];
      const separation = second - first;
      if (separation < minSeparation || separation > 160) continue;

      const firstVal = closes[first];
      const secondVal = closes[second];
      if (!isFinite(firstVal) || !isFinite(secondVal)) continue;

      const diff = Math.abs(firstVal - secondVal) / ((firstVal + secondVal) / 2);
      if (diff > tolerance) continue;

      const postIndex = Math.min(data.length - 1, second + 5);
      const postMove = (closes[postIndex] - secondVal) / (secondVal || 1);
      const dropScore = postMove < 0 ? Math.min(0.25, Math.abs(postMove) * 3) : 0;

      const confidence = clamp(
        0.45 + Math.min(0.3, (tolerance - diff) * 10) + dropScore,
        0,
        0.9
      );

      const start = Math.max(0, first - 5);
      const end = Math.min(data.length - 1, second + 5);
      return [
        buildPattern(data, 'double_top', start, end, confidence, {
          level: (firstVal + secondVal) / 2,
          touches: 2,
        }),
      ];
    }
  }

  return [];
}

function detectDoubleBottoms(data: ChartDataPoint[]): ChartPattern[] {
  const closes = data.map(point => getClose(point));
  const troughs = findLocalExtrema(closes, 3, 'min');
  const tolerance = 0.012;
  const minSeparation = 5;

  for (let i = troughs.length - 1; i >= 1; i--) {
    const second = troughs[i];
    for (let j = i - 1; j >= 0; j--) {
      const first = troughs[j];
      const separation = second - first;
      if (separation < minSeparation || separation > 160) continue;

      const firstVal = closes[first];
      const secondVal = closes[second];
      if (!isFinite(firstVal) || !isFinite(secondVal)) continue;

      const diff = Math.abs(firstVal - secondVal) / ((firstVal + secondVal) / 2);
      if (diff > tolerance) continue;

      const postIndex = Math.min(data.length - 1, second + 5);
      const postMove = (closes[postIndex] - secondVal) / (secondVal || 1);
      const bounceScore = postMove > 0 ? Math.min(0.25, postMove * 3) : 0;

      const confidence = clamp(
        0.45 + Math.min(0.3, (tolerance - diff) * 10) + bounceScore,
        0,
        0.9
      );

      const start = Math.max(0, first - 5);
      const end = Math.min(data.length - 1, second + 5);
      return [
        buildPattern(data, 'double_bottom', start, end, confidence, {
          level: (firstVal + secondVal) / 2,
          touches: 2,
        }),
      ];
    }
  }

  return [];
}

function detectMultipleTops(data: ChartDataPoint[]): ChartPattern[] {
  const closes = data.map(point => getClose(point));
  const peaks = findLocalExtrema(closes, 3, 'max').filter(index => index >= closes.length - 220);
  if (peaks.length < 3) return [];

  const clusters = clusterLevels(closes, peaks, 0.015);
  const cluster = clusters.find(c => c.indexes.length >= 3);
  if (!cluster) return [];

  const start = Math.max(0, Math.min(...cluster.indexes) - 5);
  const end = Math.min(data.length - 1, Math.max(...cluster.indexes) + 5);

  const confidence = clamp(
    0.45 + Math.min(0.35, cluster.indexes.length * 0.08),
    0,
    0.9
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

function detectMultipleBottoms(data: ChartDataPoint[]): ChartPattern[] {
  const closes = data.map(point => getClose(point));
  const troughs = findLocalExtrema(closes, 3, 'min').filter(index => index >= closes.length - 220);
  if (troughs.length < 3) return [];

  const clusters = clusterLevels(closes, troughs, 0.015);
  const cluster = clusters.find(c => c.indexes.length >= 3);
  if (!cluster) return [];

  const start = Math.max(0, Math.min(...cluster.indexes) - 5);
  const end = Math.min(data.length - 1, Math.max(...cluster.indexes) + 5);

  const confidence = clamp(
    0.45 + Math.min(0.35, cluster.indexes.length * 0.08),
    0,
    0.9
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

function detectHeadAndShoulders(data: ChartDataPoint[]): ChartPattern[] {
  const closes = data.map(point => getClose(point));
  const peaks = findLocalExtrema(closes, 4, 'max');
  if (peaks.length < 3) return [];

  for (let i = peaks.length - 3; i >= 0; i--) {
    const left = peaks[i];
    const head = peaks[i + 1];
    const right = peaks[i + 2];

    if (head - left < 5 || right - head < 5) continue;
    if (right - left > 200) continue;

    const leftVal = closes[left];
    const headVal = closes[head];
    const rightVal = closes[right];

    if (!isFinite(leftVal) || !isFinite(headVal) || !isFinite(rightVal)) continue;

    const headPremium = headVal / ((leftVal + rightVal) / 2) - 1;
    if (headPremium < 0.025) continue;

    const shoulderDiff = Math.abs(leftVal - rightVal) / ((leftVal + rightVal) / 2);
    if (shoulderDiff > 0.035) continue;

    const leftNeckline = findRangeMin(closes, left, head);
    const rightNeckline = findRangeMin(closes, head, right);
    const necklineDiff =
      Math.abs(leftNeckline.value - rightNeckline.value) /
      (((leftNeckline.value + rightNeckline.value) / 2) || 1);
    if (necklineDiff > 0.04) continue;

    const confidence = clamp(
      0.55 +
        Math.min(0.25, headPremium * 3) +
        Math.min(0.2, (0.04 - necklineDiff) * 3),
      0,
      0.95
    );

    const start = Math.max(0, left - 5);
    const end = Math.min(data.length - 1, right + 5);

    return [
      buildPattern(data, 'head_and_shoulders', start, end, confidence, {
        leftShoulder: leftVal,
        head: headVal,
        rightShoulder: rightVal,
        neckline: (leftNeckline.value + rightNeckline.value) / 2,
      }),
    ];
  }

  return [];
}

function consolidatePatterns(patterns: ChartPattern[]): ChartPattern[] {
  if (!patterns.length) return [];

  const sorted = [...patterns]
    .filter(pattern => pattern.startDate && pattern.endDate)
    .sort((a, b) => b.confidence - a.confidence);

  const perTypeCount = new Map<ChartPatternType, number>();
  const result: ChartPattern[] = [];

  for (const pattern of sorted) {
    const count = perTypeCount.get(pattern.type) ?? 0;
    if (count >= MAX_PATTERNS_PER_TYPE) continue;

    perTypeCount.set(pattern.type, count + 1);
    result.push(pattern);
    if (result.length >= MAX_PATTERNS) break;
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

