'use client';

import React, { useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Area,
  ComposedChart,
  ReferenceLine,
  ReferenceArea,
  Brush,
  Bar,
  BarChart,
  Cell,
  Customized,
} from 'recharts';
import { ChartDataPoint, ChartPattern } from '@/types';

const PATTERN_DIRECTION_STYLES: Record<
  ChartPattern['direction'],
  { stroke: string; fill: string }
> = {
  bullish: {
    stroke: 'oklch(72% 0.15 150)',
    fill: 'oklch(72% 0.15 150)',
  },
  bearish: {
    stroke: 'oklch(72% 0.16 25)',
    fill: 'oklch(72% 0.16 25)',
  },
  neutral: {
    stroke: 'oklch(72% 0.05 250)',
    fill: 'oklch(72% 0.05 250)',
  },
};

// Custom pattern renderer component with access to chart scales
const PatternRenderer = ({ xAxisMap, yAxisMap, patterns, data }: any) => {
  if (!patterns || !patterns.length || !xAxisMap || !yAxisMap) return null;
  
  const xScale = xAxisMap[0]?.scale;
  const yScale = yAxisMap['price']?.scale;
  
  if (!xScale || !yScale) return null;

  return (
    <g className="pattern-overlays">
      {patterns.map((pattern: ChartPattern) => {
        const style = PATTERN_DIRECTION_STYLES[pattern.direction];
        const patternData = data.slice(pattern.startIndex, Math.min(pattern.endIndex + 1, data.length));
        
        if (!patternData.length) return null;

        const labelValue = `${pattern.label} ${(pattern.confidence * 100).toFixed(0)}%`;

        // Get coordinates - simpler approach
        const x1 = xScale(pattern.startDate);
        const x2 = xScale(pattern.endDate);

        // Skip if coordinates are invalid
        if (typeof x1 !== 'number' || typeof x2 !== 'number' || isNaN(x1) || isNaN(x2)) {
          return null;
        }

        const midX = (x1 + x2) / 2;

        // Calculate price range for the pattern
        const prices = patternData.map((d: ChartDataPoint) => [d.low || d.close, d.high || d.close, d.close]).flat().filter(p => p);
        const minPrice = Math.min(...prices);
        const maxPrice = Math.max(...prices);

        switch (pattern.type) {
          case 'trendline_support':
          case 'trendline_resistance': {
            const isSupport = pattern.type === 'trendline_support';
            const trendPrices = patternData.map((d: ChartDataPoint) => isSupport ? (d.low || d.close) : (d.high || d.close));
            const y1 = yScale(trendPrices[0]);
            const y2 = yScale(trendPrices[trendPrices.length - 1]);
            
            return (
              <g key={pattern.id}>
                {/* Diagonal trendline */}
                <line
                  x1={x1}
                  y1={y1}
                  x2={x2}
                  y2={y2}
                  stroke={style.stroke}
                  strokeWidth={2.5}
                  strokeDasharray="8 4"
                  opacity={0.85}
                />
                {/* Label */}
                <text
                  x={midX}
                  y={Math.min(y1, y2) - 10}
                  fill={style.stroke}
                  fontSize={11}
                  fontWeight={600}
                  fontFamily="DM Mono, monospace"
                  textAnchor="middle"
                >
                  {labelValue}
                </text>
                {/* Touch point circles */}
                {patternData.map((d: ChartDataPoint, i: number) => {
                  const price = isSupport ? (d.low || d.close) : (d.high || d.close);
                  const px = x1 + (x2 - x1) * (i / (patternData.length - 1));
                  const py = yScale(price);
                  const lineY = y1 + (y2 - y1) * (i / (patternData.length - 1));
                  if (Math.abs(py - lineY) < 10) {
                    return (
                      <circle
                        key={i}
                        cx={px}
                        cy={py}
                        r={3}
                        fill={style.fill}
                        opacity={0.7}
                      />
                    );
                  }
                  return null;
                })}
              </g>
            );
          }

          case 'wedge_up':
          case 'wedge_down':
          case 'wedge': {
            const highs = patternData.map((d: ChartDataPoint) => d.high || d.close);
            const lows = patternData.map((d: ChartDataPoint) => d.low || d.close);
            const y1High = yScale(highs[0]);
            const y2High = yScale(highs[highs.length - 1]);
            const y1Low = yScale(lows[0]);
            const y2Low = yScale(lows[lows.length - 1]);
            
            return (
              <g key={pattern.id}>
                {/* Wedge fill */}
                <path
                  d={`M ${x1} ${y1High} L ${x2} ${y2High} L ${x2} ${y2Low} L ${x1} ${y1Low} Z`}
                  fill={style.fill}
                  fillOpacity={0.08}
                  stroke="none"
                />
                {/* Upper line */}
                <line
                  x1={x1}
                  y1={y1High}
                  x2={x2}
                  y2={y2High}
                  stroke={style.stroke}
                  strokeWidth={2}
                  strokeDasharray="6 3"
                  opacity={0.7}
                />
                {/* Lower line */}
                <line
                  x1={x1}
                  y1={y1Low}
                  x2={x2}
                  y2={y2Low}
                  stroke={style.stroke}
                  strokeWidth={2}
                  strokeDasharray="6 3"
                  opacity={0.7}
                />
                {/* Label */}
                <text
                  x={midX}
                  y={(y1High + y2High + y1Low + y2Low) / 4}
                  fill={style.stroke}
                  fontSize={11}
                  fontWeight={600}
                  fontFamily="DM Mono, monospace"
                  textAnchor="middle"
                >
                  {labelValue}
                </text>
                {/* Convergence arrows */}
                <path
                  d={`M ${x2 - 20} ${y2High + 10} L ${x2 - 10} ${(y2High + y2Low) / 2} L ${x2 - 20} ${y2Low - 10}`}
                  stroke={style.stroke}
                  strokeWidth={1.5}
                  fill="none"
                  opacity={0.6}
                />
              </g>
            );
          }

          case 'triangle_ascending':
          case 'triangle_descending':
          case 'triangle_symmetrical': {
            const highs = patternData.map((d: ChartDataPoint) => d.high || d.close);
            const lows = patternData.map((d: ChartDataPoint) => d.low || d.close);
            const maxHigh = Math.max(...highs);
            const minLow = Math.min(...lows);
            
            let y1Upper, y2Upper, y1Lower, y2Lower;
            
            if (pattern.type === 'triangle_ascending') {
              y1Upper = y2Upper = yScale(maxHigh);
              y1Lower = yScale(lows[0]);
              y2Lower = yScale(lows[lows.length - 1]);
            } else if (pattern.type === 'triangle_descending') {
              y1Upper = yScale(highs[0]);
              y2Upper = yScale(highs[highs.length - 1]);
              y1Lower = y2Lower = yScale(minLow);
            } else {
              y1Upper = yScale(highs[0]);
              y2Upper = yScale(highs[highs.length - 1]);
              y1Lower = yScale(lows[0]);
              y2Lower = yScale(lows[lows.length - 1]);
            }
            
            return (
              <g key={pattern.id}>
                {/* Triangle fill */}
                <path
                  d={`M ${x1} ${y1Upper} L ${x2} ${y2Upper} L ${x2} ${y2Lower} L ${x1} ${y1Lower} Z`}
                  fill={style.fill}
                  fillOpacity={0.1}
                  stroke={style.stroke}
                  strokeWidth={1.5}
                  strokeDasharray="5 3"
                  opacity={0.5}
                />
                {/* Upper line */}
                <line
                  x1={x1}
                  y1={y1Upper}
                  x2={x2}
                  y2={y2Upper}
                  stroke={style.stroke}
                  strokeWidth={2.5}
                  strokeDasharray="6 3"
                  opacity={0.8}
                />
                {/* Lower line */}
                <line
                  x1={x1}
                  y1={y1Lower}
                  x2={x2}
                  y2={y2Lower}
                  stroke={style.stroke}
                  strokeWidth={2.5}
                  strokeDasharray="6 3"
                  opacity={0.8}
                />
                {/* Label */}
                <text
                  x={midX}
                  y={(y1Upper + y2Upper + y1Lower + y2Lower) / 4}
                  fill={style.stroke}
                  fontSize={11}
                  fontWeight={600}
                  fontFamily="DM Mono, monospace"
                  textAnchor="middle"
                >
                  {labelValue}
                </text>
                {/* Breakout arrow */}
                {pattern.direction !== 'neutral' && (
                  <path
                    d={pattern.direction === 'bullish'
                      ? `M ${x2 + 5} ${(y2Upper + y2Lower) / 2} L ${x2 + 15} ${(y2Upper + y2Lower) / 2 - 10} M ${x2 + 5} ${(y2Upper + y2Lower) / 2} L ${x2 + 15} ${(y2Upper + y2Lower) / 2 + 10}`
                      : `M ${x2 + 5} ${(y2Upper + y2Lower) / 2} L ${x2 + 15} ${(y2Upper + y2Lower) / 2 + 10} M ${x2 + 5} ${(y2Upper + y2Lower) / 2} L ${x2 + 15} ${(y2Upper + y2Lower) / 2 - 10}`
                    }
                    stroke={style.stroke}
                    strokeWidth={2}
                    fill="none"
                    opacity={0.7}
                  />
                )}
              </g>
            );
          }

          case 'channel_up':
          case 'channel_down':
          case 'channel': {
            const highs = patternData.map((d: ChartDataPoint) => d.high || d.close);
            const lows = patternData.map((d: ChartDataPoint) => d.low || d.close);
            const y1High = yScale(highs[0]);
            const y2High = yScale(highs[highs.length - 1]);
            const y1Low = yScale(lows[0]);
            const y2Low = yScale(lows[lows.length - 1]);
            
            return (
              <g key={pattern.id}>
                {/* Channel fill */}
                <path
                  d={`M ${x1} ${y1High} L ${x2} ${y2High} L ${x2} ${y2Low} L ${x1} ${y1Low} Z`}
                  fill={style.fill}
                  fillOpacity={0.06}
                  stroke="none"
                />
                {/* Upper line */}
                <line
                  x1={x1}
                  y1={y1High}
                  x2={x2}
                  y2={y2High}
                  stroke={style.stroke}
                  strokeWidth={2}
                  strokeDasharray="8 4"
                  opacity={0.7}
                />
                {/* Lower line */}
                <line
                  x1={x1}
                  y1={y1Low}
                  x2={x2}
                  y2={y2Low}
                  stroke={style.stroke}
                  strokeWidth={2}
                  strokeDasharray="8 4"
                  opacity={0.7}
                />
                {/* Label */}
                <text
                  x={midX}
                  y={(y1High + y2High + y1Low + y2Low) / 4}
                  fill={style.stroke}
                  fontSize={11}
                  fontWeight={600}
                  fontFamily="DM Mono, monospace"
                  textAnchor="middle"
                >
                  {labelValue}
                </text>
                {/* Parallel indicators */}
                <line x1={midX - 15} y1={(y1High + y2High) / 2} x2={midX - 15} y2={(y1Low + y2Low) / 2} stroke={style.stroke} strokeWidth={1.5} opacity={0.5} />
                <line x1={midX + 15} y1={(y1High + y2High) / 2} x2={midX + 15} y2={(y1Low + y2Low) / 2} stroke={style.stroke} strokeWidth={1.5} opacity={0.5} />
              </g>
            );
          }

          case 'double_top':
          case 'double_bottom': {
            const level = pattern.meta?.level as number || (pattern.type === 'double_top' ? maxPrice : minPrice);
            const isTop = pattern.type === 'double_top';
            const yLevel = yScale(level);
            
            return (
              <g key={pattern.id}>
                {/* Level line */}
                <line
                  x1={x1}
                  y1={yLevel}
                  x2={x2}
                  y2={yLevel}
                  stroke={style.stroke}
                  strokeWidth={2.5}
                  strokeDasharray="6 3"
                  opacity={0.8}
                />
                {/* Label with background */}
                <rect
                  x={x1 + 5}
                  y={yLevel + (isTop ? -20 : 5)}
                  width={labelValue.length * 6.5}
                  height={14}
                  fill="oklch(23% 0 0)"
                  fillOpacity={0.85}
                  rx={2}
                />
                <text
                  x={x1 + 8}
                  y={yLevel + (isTop ? -9 : 16)}
                  fill={style.stroke}
                  fontSize={10}
                  fontWeight={600}
                  fontFamily="DM Mono, monospace"
                  textAnchor="start"
                >
                  {labelValue}
                </text>
              </g>
            );
          }

          case 'head_and_shoulders': {
            const head = pattern.meta?.head as number || maxPrice;
            const neckline = pattern.meta?.neckline as number || minPrice;
            const yHead = yScale(head);
            const yNeckline = yScale(neckline);
            
            return (
              <g key={pattern.id}>
                {/* Pattern area - subtle */}
                <rect
                  x={x1}
                  y={Math.min(yHead, yNeckline)}
                  width={Math.max(0, x2 - x1)}
                  height={Math.abs(yNeckline - yHead)}
                  fill={style.fill}
                  fillOpacity={0.05}
                  stroke="none"
                />
                {/* Neckline - main feature */}
                <line
                  x1={x1}
                  y1={yNeckline}
                  x2={x2}
                  y2={yNeckline}
                  stroke={style.stroke}
                  strokeWidth={2.5}
                  strokeDasharray="8 4"
                  opacity={0.85}
                />
                {/* Label with background */}
                <rect
                  x={x1 + 5}
                  y={yHead - 20}
                  width={labelValue.length * 6.5}
                  height={14}
                  fill="oklch(23% 0 0)"
                  fillOpacity={0.85}
                  rx={2}
                />
                <text
                  x={x1 + 8}
                  y={yHead - 9}
                  fill={style.stroke}
                  fontSize={10}
                  fontWeight={600}
                  fontFamily="DM Mono, monospace"
                  textAnchor="start"
                >
                  {labelValue}
                </text>
              </g>
            );
          }

          case 'horizontal_sr': {
            const resistance = pattern.meta?.resistance as number;
            const support = pattern.meta?.support as number;
            
            if (!resistance || !support) return null;
            
            const yResistance = yScale(resistance);
            const ySupport = yScale(support);
            
            return (
              <g key={pattern.id}>
                {/* Zone fill */}
                <rect
                  x={x1}
                  y={Math.min(yResistance, ySupport)}
                  width={Math.max(0, x2 - x1)}
                  height={Math.abs(ySupport - yResistance)}
                  fill={style.fill}
                  fillOpacity={0.08}
                  stroke="none"
                />
                {/* Resistance line */}
                <line
                  x1={x1}
                  y1={yResistance}
                  x2={x2}
                  y2={yResistance}
                  stroke={style.stroke}
                  strokeWidth={2}
                  strokeDasharray="6 2"
                  opacity={0.8}
                />
                {/* Support line */}
                <line
                  x1={x1}
                  y1={ySupport}
                  x2={x2}
                  y2={ySupport}
                  stroke={style.stroke}
                  strokeWidth={2}
                  strokeDasharray="6 2"
                  opacity={0.8}
                />
                {/* Labels with backgrounds */}
                <rect
                  x={x1 + 5}
                  y={yResistance - 18}
                  width={Math.max(labelValue.length * 6.5, 120)}
                  height={14}
                  fill="oklch(23% 0 0)"
                  fillOpacity={0.85}
                  rx={2}
                />
                <text
                  x={x1 + 8}
                  y={yResistance - 7}
                  fill={style.stroke}
                  fontSize={10}
                  fontWeight={600}
                  fontFamily="DM Mono, monospace"
                  textAnchor="start"
                >
                  {labelValue}
                </text>
              </g>
            );
          }

          case 'multiple_top':
          case 'multiple_bottom': {
            const level = pattern.meta?.level as number || (pattern.type === 'multiple_top' ? maxPrice : minPrice);
            const touches = pattern.meta?.touches as number || 3;
            const isTop = pattern.type === 'multiple_top';
            const yLevel = yScale(level);
            
            return (
              <g key={pattern.id}>
                {/* Level line */}
                <line
                  x1={x1}
                  y1={yLevel}
                  x2={x2}
                  y2={yLevel}
                  stroke={style.stroke}
                  strokeWidth={2.5}
                  strokeDasharray="5 3"
                  opacity={0.8}
                />
                {/* Label with background */}
                <rect
                  x={x1 + 5}
                  y={yLevel + (isTop ? -20 : 5)}
                  width={Math.max(labelValue.length * 6.5, 100)}
                  height={14}
                  fill="oklch(23% 0 0)"
                  fillOpacity={0.85}
                  rx={2}
                />
                <text
                  x={x1 + 8}
                  y={yLevel + (isTop ? -9 : 16)}
                  fill={style.stroke}
                  fontSize={10}
                  fontWeight={600}
                  fontFamily="DM Mono, monospace"
                  textAnchor="start"
                >
                  {labelValue} • {touches}×
                </text>
              </g>
            );
          }

          default:
            return null;
        }
      })}
    </g>
  );
};

interface StockChartProps {
  data: ChartDataPoint[];
  showMA20?: boolean;
  showMA50?: boolean;
  showBB?: boolean;
  showForecast?: boolean;
  forecastData?: Array<{ date: string; predicted: number; upper: number; lower: number }>;
  chartType?: 'line' | 'candlestick';
  showVolume?: boolean;
  patterns?: ChartPattern[];
  dataInterval?: string;
  enablePatterns?: boolean;
  onVisibleRangeChange?: (startDate: string, endDate: string) => void;
}


export default function StockChart({
  data,
  showMA20 = true,
  showMA50 = true,
  showBB = false,
  showForecast = true,
  forecastData = [],
  chartType = 'line',
  showVolume = true,
  patterns = [],
  dataInterval = '1d',
  enablePatterns = false,
  onVisibleRangeChange,
}: StockChartProps) {
  // Combine historical and forecast data - memoized to avoid recalculation
  const combinedData = React.useMemo(() => {
    const historicalData = data.map(d => ({ ...d, isForecast: false }));

    // Bridge the gap by adding the last historical price as the first forecast point
    const forecastWithBridge = forecastData.length > 0 && data.length > 0
      ? [
          {
            date: data[data.length - 1].date,
            close: data[data.length - 1].close,
            predicted: data[data.length - 1].close,
            upper: data[data.length - 1].close,
            lower: data[data.length - 1].close,
            isForecast: true,
          },
          ...forecastData.map(f => ({
            date: f.date,
            predicted: f.predicted,
            upper: f.upper,
            lower: f.lower,
            isForecast: true,
          })),
        ]
      : [];

    return [...historicalData, ...forecastWithBridge];
  }, [data, forecastData]);

  // Calculate the index where forecast starts
  const historicalDataLength = data.length;

  const priceExtents = React.useMemo(() => {
    if (!data.length) {
      return { min: 0, max: 0 };
    }

    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;

    data.forEach(point => {
      const low = typeof point.low === 'number' ? point.low : point.close;
      const high = typeof point.high === 'number' ? point.high : point.close;
      const close = point.close;

      min = Math.min(min, low, close);
      max = Math.max(max, high, close);
    });

    if (!isFinite(min)) min = 0;
    if (!isFinite(max)) max = 0;

    return { min, max };
  }, [data]);

  const [brushKey, setBrushKey] = useState(0);
  const [brushRange, setBrushRange] = useState<{ startIndex: number; endIndex: number }>({
    startIndex: Math.max(0, historicalDataLength - 30), // Initial placeholder; updated via effect below
    endIndex: combinedData.length - 1, // Include all forecast data
  });

  const [activeRange, setActiveRange] = useState<number | 'all' | 'default' | null>('default');
  const [zoomLevel, setZoomLevel] = useState(1);
  const chartRef = React.useRef<HTMLDivElement>(null);

  // Use patterns prop directly instead of state to avoid infinite loops
  const viewPatterns = React.useMemo(() => {
    return enablePatterns ? (patterns || []) : [];
  }, [patterns, enablePatterns]);

  const clampIndex = React.useCallback(
    (value: number, min: number, max: number) => {
      return Math.min(Math.max(value, min), max);
    },
    []
  );

  const getTrimmedPatternDates = React.useCallback(
    (pattern: ChartPattern) => {
      if (!data.length) {
        return {
          startDate: pattern.startDate,
          endDate: pattern.endDate,
        };
      }

      const span = pattern.endIndex - pattern.startIndex;
      if (span <= 2) {
        const start = clampIndex(pattern.startIndex, 0, data.length - 1);
        const end = clampIndex(pattern.endIndex, start, data.length - 1);
        return {
          startDate: data[start]?.date ?? pattern.startDate,
          endDate: data[end]?.date ?? pattern.endDate,
        };
      }

      const trim = Math.max(1, Math.floor(span * 0.2));
      const trimmedStartIndex = clampIndex(pattern.startIndex + trim, 0, data.length - 2);
      const trimmedEndIndex = clampIndex(
        pattern.endIndex - trim,
        trimmedStartIndex + 1,
        data.length - 1
      );

      return {
        startDate: data[trimmedStartIndex]?.date ?? pattern.startDate,
        endDate: data[trimmedEndIndex]?.date ?? pattern.endDate,
      };
    },
    [data, clampIndex]
  );

  const getTrimmedPatternPrices = React.useCallback((minPrice: number, maxPrice: number) => {
    if (!Number.isFinite(minPrice) || !Number.isFinite(maxPrice)) {
      return { y1: minPrice, y2: maxPrice };
    }

    const range = maxPrice - minPrice;
    if (range <= 0) {
      return { y1: minPrice, y2: maxPrice };
    }

      const pad = Math.max(range * 0.2, 0.015 * Math.max(Math.abs(maxPrice), Math.abs(minPrice)));
    const y1 = minPrice + pad;
    const y2 = maxPrice - pad;

    if (y2 <= y1) {
      return { y1: minPrice, y2: maxPrice };
    }

    return { y1, y2 };
  }, []);

  const normalizedInterval = (dataInterval || '1d').toLowerCase();
  const isIntradayInterval = React.useMemo(() => {
    return normalizedInterval.includes('m') || normalizedInterval.includes('h');
  }, [normalizedInterval]);

  const getStartIndexForDays = React.useCallback(
    (days: number) => {
      if (!data.length || !Number.isFinite(days) || days <= 0) {
        return 0;
      }

      const lastHistoricalIndex = historicalDataLength - 1;
      if (lastHistoricalIndex < 0) {
        return 0;
      }

      const lastPoint = data[lastHistoricalIndex];
      const lastTimestamp = Date.parse(lastPoint?.date ?? '');
      if (Number.isNaN(lastTimestamp)) {
        return Math.max(0, historicalDataLength - Math.ceil(days));
      }

      const cutoff = lastTimestamp - days * 24 * 60 * 60 * 1000;
      for (let i = 0; i < historicalDataLength; i++) {
        const pointTimestamp = Date.parse(data[i]?.date ?? '');
        if (Number.isNaN(pointTimestamp)) {
          continue;
        }
        if (pointTimestamp >= cutoff) {
          return Math.max(0, i - 1);
        }
      }

      return 0;
    },
    [data, historicalDataLength]
  );

  // Update brush range when data changes - show recent history + forecast
  useEffect(() => {
    const startIndex = getStartIndexForDays(30);
    const endIndex = combinedData.length - 1; // Include forecast

    // Only update if values actually changed
    setBrushRange(prev => {
      if (prev.startIndex === startIndex && prev.endIndex === endIndex) {
        return prev;
      }
      setBrushKey(k => k + 1); // Force brush remount only when range changes
      setActiveRange('default'); // Set default as active
      return { startIndex, endIndex };
    });
  }, [historicalDataLength, combinedData.length, getStartIndexForDays]);

  const visibleStartIndex = React.useMemo(() => {
    if (!data.length) return 0;
    return clampIndex(brushRange.startIndex ?? 0, 0, historicalDataLength - 1);
  }, [brushRange.startIndex, clampIndex, data.length, historicalDataLength]);

  const visibleEndIndex = React.useMemo(() => {
    if (!data.length) return 0;
    const historicalEnd = historicalDataLength - 1;
    const end = Math.min(brushRange.endIndex ?? historicalEnd, historicalEnd);
    return clampIndex(end, visibleStartIndex, historicalEnd);
  }, [brushRange.endIndex, clampIndex, data.length, historicalDataLength, visibleStartIndex]);

  const visibleData = React.useMemo(() => {
    if (!data.length) return [];
    return data.slice(visibleStartIndex, visibleEndIndex + 1);
  }, [data, visibleStartIndex, visibleEndIndex]);
  // Notify parent of visible range changes
  useEffect(() => {
    if (onVisibleRangeChange && data.length > 0) {
      const startDate = data[visibleStartIndex]?.date;
      const endDate = data[visibleEndIndex]?.date;
      if (startDate && endDate) {
        onVisibleRangeChange(startDate, endDate);
      }
    }
  }, [visibleStartIndex, visibleEndIndex, data, onVisibleRangeChange]);

  const visiblePatterns = React.useMemo(() => {
    if (!enablePatterns || !viewPatterns?.length) return [];
    const start = brushRange.startIndex ?? 0;
    const end = brushRange.endIndex ?? combinedData.length - 1;
    const cappedEnd = Math.min(end, data.length - 1);
    const filtered = viewPatterns.filter(
      pattern => pattern.endIndex >= start && pattern.startIndex <= cappedEnd
    );
    console.log(`🎯 Pattern filtering: Total=${viewPatterns.length}, Visible=${filtered.length}, Range=[${start}, ${cappedEnd}] (${cappedEnd - start + 1} points)`);
    return filtered;
  }, [enablePatterns, viewPatterns, brushRange, combinedData.length, data.length]);

  const patternBadges = React.useMemo(
    () => visiblePatterns.slice(0, 6),
    [visiblePatterns]
  );

  const handleTimeRange = (range: number | 'all' | 'default') => {
    setActiveRange(range);
    setZoomLevel(1); // Reset zoom level
    
    if (range === 'default') {
      // Show last 30 days + forecast
      const startIndex = getStartIndexForDays(30);
      const endIndex = combinedData.length - 1;
      setBrushRange({ startIndex, endIndex });
      setBrushKey(prev => prev + 1); // Force brush remount
    } else if (range === 'all') {
      // Show all historical data + forecast
      setBrushRange({ startIndex: 0, endIndex: combinedData.length - 1 });
      setBrushKey(prev => prev + 1); // Force brush remount
    } else {
      // Show last N days of historical data (not including forecast)
      const endIndex = historicalDataLength - 1;
      const startIndex = getStartIndexForDays(range);
      setBrushRange({
        startIndex,
        endIndex,
      });
      setBrushKey(prev => prev + 1); // Force brush remount
    }
  };

  // Zoom in/out functions
  const handleZoomIn = () => {
    const { startIndex, endIndex } = brushRange;
    const currentRange = endIndex - startIndex;
    const newRange = Math.max(5, Math.floor(currentRange * 0.7)); // Zoom in by 30%
    const center = Math.floor((startIndex + endIndex) / 2);
    const newStartIndex = Math.max(0, center - Math.floor(newRange / 2));
    const newEndIndex = Math.min(combinedData.length - 1, newStartIndex + newRange);
    
    setBrushRange({ startIndex: newStartIndex, endIndex: newEndIndex });
    setBrushKey(prev => prev + 1);
    setActiveRange(null); // Clear preset
    setZoomLevel(prev => Math.min(prev + 0.3, 3));
  };

  const handleZoomOut = () => {
    const { startIndex, endIndex } = brushRange;
    const currentRange = endIndex - startIndex;
    const newRange = Math.min(combinedData.length, Math.floor(currentRange * 1.4)); // Zoom out by 40%
    const center = Math.floor((startIndex + endIndex) / 2);
    const newStartIndex = Math.max(0, center - Math.floor(newRange / 2));
    const newEndIndex = Math.min(combinedData.length - 1, newStartIndex + newRange);
    
    setBrushRange({ startIndex: newStartIndex, endIndex: newEndIndex });
    setBrushKey(prev => prev + 1);
    setActiveRange(null);
    setZoomLevel(prev => Math.max(prev - 0.3, 0.5));
  };

  const handleResetZoom = () => {
    handleTimeRange('default');
  };

  // Mouse wheel zoom
  useEffect(() => {
    const chartElement = chartRef.current;
    if (!chartElement) return;

    const handleWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        if (e.deltaY < 0) {
          handleZoomIn();
        } else {
          handleZoomOut();
        }
      }
    };

    chartElement.addEventListener('wheel', handleWheel, { passive: false });
    return () => chartElement.removeEventListener('wheel', handleWheel);
  }, [brushRange, combinedData.length]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      // Only trigger if not in an input field
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key === '+' || e.key === '=') {
        handleZoomIn();
      } else if (e.key === '-' || e.key === '_') {
        handleZoomOut();
      } else if (e.key === '0' || e.key === 'r' || e.key === 'R') {
        handleResetZoom();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [brushRange, combinedData.length]);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    const { startIndex, endIndex } = brushRange;
    const visibleRange = endIndex - startIndex;

    if (Number.isNaN(date.getTime())) {
      return dateStr;
    }

    if (isIntradayInterval) {
      const showDate = visibleRange > 400;
      return date.toLocaleString('en-US', {
        month: showDate ? 'short' : undefined,
        day: showDate ? 'numeric' : undefined,
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      });
    }

    // Show year if viewing more than 180 days (6 months)
    if (visibleRange > 180) {
      // For long ranges, show abbreviated format with year
      return date.toLocaleDateString('en-US', { 
        month: 'short', 
        day: 'numeric',
        year: '2-digit' // Use 2-digit year (e.g., '23 instead of 2023)
      });
    }

    // For short ranges, just month and day
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  const formatTooltipLabel = (dateStr: string) => {
    const date = new Date(dateStr);
    if (Number.isNaN(date.getTime())) {
      return dateStr;
    }

    if (isIntradayInterval) {
      return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false,
      });
    }

    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      year: 'numeric' 
    });
  };

  const formatPrice = (value: number | undefined | null) => {
    if (value === undefined || value === null || isNaN(value)) return '$0.00';
    return `$${value.toFixed(2)}`;
  };

  // Prepare data with low as base and range for candlestick height
  const candlestickData = React.useMemo(() => {
    return combinedData.map(d => ({
      ...d,
      lowValue: 'low' in d ? d.low : undefined,
      candleHeight: 'high' in d && 'low' in d && d.high && d.low ? d.high - d.low : 0,
    }));
  }, [combinedData]);

  return (
    <div className="w-full">
      {/* Time Range Buttons */}
      <div className="flex gap-2 mb-4 flex-wrap items-center">
        {/* Quick Time Range */}
        <div className="flex gap-2 flex-wrap">
          <button
            onClick={() => handleTimeRange('default')}
            className="px-3 py-1 text-xs font-medium border transition-all"
            style={{
              background: activeRange === 'default' ? 'var(--accent)' : 'var(--bg-3)',
              borderColor: activeRange === 'default' ? 'var(--accent)' : 'var(--bg-1)',
              color: activeRange === 'default' ? 'var(--text-0)' : 'var(--text-3)',
            }}
          >
            DEFAULT
          </button>
          <button
            onClick={() => handleTimeRange(5)}
            className="px-3 py-1 text-xs font-medium border transition-all"
            style={{
              background: activeRange === 5 ? 'var(--accent)' : 'var(--bg-3)',
              borderColor: activeRange === 5 ? 'var(--accent)' : 'var(--bg-1)',
              color: activeRange === 5 ? 'var(--text-0)' : 'var(--text-3)',
            }}
            title="5 Days View"
          >
            5D
          </button>
          <button
            onClick={() => handleTimeRange(30)}
            className="px-3 py-1 text-xs font-medium border transition-all"
            style={{
              background: activeRange === 30 ? 'var(--accent)' : 'var(--bg-3)',
              borderColor: activeRange === 30 ? 'var(--accent)' : 'var(--bg-1)',
              color: activeRange === 30 ? 'var(--text-0)' : 'var(--text-3)',
            }}
          >
            1M
          </button>
        <button
          onClick={() => handleTimeRange(90)}
          className="px-3 py-1 text-xs font-medium border transition-all"
          style={{
            background: activeRange === 90 ? 'var(--accent)' : 'var(--bg-3)',
            borderColor: activeRange === 90 ? 'var(--accent)' : 'var(--bg-1)',
            color: activeRange === 90 ? 'var(--text-0)' : 'var(--text-3)',
          }}
        >
          3M
        </button>
        <button
          onClick={() => handleTimeRange(180)}
          className="px-3 py-1 text-xs font-medium border transition-all"
          style={{
            background: activeRange === 180 ? 'var(--accent)' : 'var(--bg-3)',
            borderColor: activeRange === 180 ? 'var(--accent)' : 'var(--bg-1)',
            color: activeRange === 180 ? 'var(--text-0)' : 'var(--text-3)',
          }}
        >
          6M
        </button>
        <button
          onClick={() => handleTimeRange(365)}
          className="px-3 py-1 text-xs font-medium border transition-all"
          style={{
            background: activeRange === 365 ? 'var(--accent)' : 'var(--bg-3)',
            borderColor: activeRange === 365 ? 'var(--accent)' : 'var(--bg-1)',
            color: activeRange === 365 ? 'var(--text-0)' : 'var(--text-3)',
          }}
        >
          1Y
        </button>
        <button
          onClick={() => handleTimeRange(730)}
          className="px-3 py-1 text-xs font-medium border transition-all"
          style={{
            background: activeRange === 730 ? 'var(--accent)' : 'var(--bg-3)',
            borderColor: activeRange === 730 ? 'var(--accent)' : 'var(--bg-1)',
            color: activeRange === 730 ? 'var(--text-0)' : 'var(--text-3)',
          }}
        >
          2Y
        </button>
        <button
          onClick={() => handleTimeRange(1095)}
          className="px-3 py-1 text-xs font-medium border transition-all"
          style={{
            background: activeRange === 1095 ? 'var(--accent)' : 'var(--bg-3)',
            borderColor: activeRange === 1095 ? 'var(--accent)' : 'var(--bg-1)',
            color: activeRange === 1095 ? 'var(--text-0)' : 'var(--text-3)',
          }}
        >
          3Y
        </button>
        <button
          onClick={() => handleTimeRange('all')}
          className="px-3 py-1 text-xs font-medium border transition-all"
          style={{
            background: activeRange === 'all' ? 'var(--accent)' : 'var(--bg-3)',
            borderColor: activeRange === 'all' ? 'var(--accent)' : 'var(--bg-1)',
            color: activeRange === 'all' ? 'var(--text-0)' : 'var(--text-3)',
          }}
          title="All available history (5 years)"
        >
          5Y (ALL)
        </button>
        </div>

        {/* Divider */}
        <div style={{ 
          width: '1px', 
          height: '24px', 
          background: 'var(--bg-1)',
          margin: '0 4px'
        }} />

        {/* Zoom Controls */}
        <div className="flex gap-2">
          <button
            onClick={handleZoomIn}
            className="px-3 py-1 text-xs font-medium border transition-all"
            style={{
              background: 'var(--bg-3)',
              borderColor: 'var(--bg-1)',
              color: 'var(--text-3)',
            }}
            title="Zoom In (or press +)"
          >
            🔍+
          </button>
          <button
            onClick={handleZoomOut}
            className="px-3 py-1 text-xs font-medium border transition-all"
            style={{
              background: 'var(--bg-3)',
              borderColor: 'var(--bg-1)',
              color: 'var(--text-3)',
            }}
            title="Zoom Out (or press -)"
          >
            🔍-
          </button>
          <button
            onClick={handleResetZoom}
            className="px-3 py-1 text-xs font-medium border transition-all"
            style={{
              background: 'var(--bg-3)',
              borderColor: 'var(--bg-1)',
              color: 'var(--text-3)',
            }}
            title="Reset Zoom (or press R)"
          >
            ↺ Reset
          </button>
        </div>

        {/* Zoom Level Indicator */}
        <div className="text-xs" style={{ color: 'var(--text-4)', marginLeft: '8px' }}>
          Zoom: {(zoomLevel * 100).toFixed(0)}%
        </div>
      </div>

      {enablePatterns && data.length > 0 && (
        visiblePatterns.length > 0 ? (
          <div
            className="mb-4 p-3 border"
            style={{
              background: 'var(--bg-3)',
              borderColor: 'var(--bg-1)',
            }}
          >
            <div
              className="text-xs font-semibold uppercase tracking-wide"
              style={{ color: 'var(--text-4)' }}
            >
              patterns WIP ({visiblePatterns.length})
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              {patternBadges.map(pattern => {
                const colors = PATTERN_DIRECTION_STYLES[pattern.direction];
                return (
                  <span
                    key={pattern.id}
                    className="px-2 py-1 text-[11px] font-semibold border uppercase tracking-wide"
                    style={{
                      borderColor: colors.stroke,
                      color: colors.stroke,
                      background: 'var(--bg-4)',
                    }}
                    title={`${pattern.label} — ${(pattern.confidence * 100).toFixed(
                      0
                    )}% confidence`}
                  >
                    {pattern.label}{' '}
                    <span style={{ color: 'var(--text-5)', fontWeight: 400 }}>
                      {(pattern.confidence * 100).toFixed(0)}%
                    </span>
                  </span>
                );
              })}
            </div>
          </div>
        ) : (
          <div
            className="mb-4 text-xs italic"
            style={{ color: 'var(--text-5)' }}
          >
            No patterns detected on this timeframe.
          </div>
        )
      )}
      {!enablePatterns && (
        <div
          className="mb-4 text-xs italic"
          style={{ color: 'var(--text-5)' }}
        >
          Chart patterns overlay disabled (toggle in controls above).
        </div>
      )}

      <div ref={chartRef} className="w-full h-[300px] sm:h-[400px] md:h-[500px] lg:h-[550px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartType === 'candlestick' ? candlestickData : combinedData}
            margin={{ top: 5, right: 10, left: 0, bottom: 30 }}
            barCategoryGap="10%"
          >
          <CartesianGrid strokeDasharray="3 3" stroke="oklch(31% 0 0)" opacity={0.3} />
          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            stroke="oklch(75% 0 0)"
            style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace' }}
          />
          <YAxis
            yAxisId="price"
            tickFormatter={formatPrice}
            stroke="oklch(75% 0 0)"
            style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace' }}
            domain={['auto', 'auto']}
          />
          {showVolume && (
            <YAxis
              yAxisId="volume"
              orientation="right"
              stroke="oklch(75% 0 0)"
              style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace' }}
              tickFormatter={(value) => {
                if (value >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
                if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
                return value.toString();
              }}
              domain={[0, (dataMax: number) => dataMax * 4]}
            />
          )}
          <Tooltip
            content={(props: any) => {
              const { active, payload, label } = props;
              if (!active || !payload || !payload.length) return null;

              // Filter out the duplicate "lower" entry (from Area component)
              const filteredPayload = payload.filter((entry: any) => {
                // Keep all entries except the one with dataKey "lower" and no name
                if (entry.dataKey === 'lower' && (!entry.name || entry.name === 'lower' || entry.name === '')) {
                  return false;
                }
                return true;
              });

              return (
                <div style={{
                  backgroundColor: 'oklch(23% 0 0)',
                  border: '2px solid oklch(70% 0.12 170)',
                  borderRadius: '0',
                  color: 'oklch(85% 0 0)',
                  fontFamily: 'DM Mono, monospace',
                  fontSize: '12px',
                  padding: '8px'
                }}>
                  <p style={{
                    margin: '0 0 4px 0',
                    color: 'oklch(70% 0.12 170)',
                    fontWeight: 'bold'
                  }}>
                    {formatTooltipLabel(label)}
                  </p>
                  {chartType === 'candlestick' && payload[0]?.payload && (
                    (() => {
                      const { open, high, low, close } = payload[0].payload;
                      if (open !== undefined && high !== undefined && low !== undefined && close !== undefined) {
                        return (
                          <div style={{ fontSize: '11px', marginBottom: '4px' }}>
                            <div>O: {formatPrice(open)}</div>
                            <div>H: {formatPrice(high)}</div>
                            <div>L: {formatPrice(low)}</div>
                            <div>C: {formatPrice(close)}</div>
                          </div>
                        );
                      }
                      return null;
                    })()
                  )}
                  {filteredPayload.map((entry: any, index: number) => (
                    <div key={`item-${index}`} style={{ margin: '2px 0' }}>
                      <span style={{ color: entry.color }}>{entry.name}: </span>
                      <span>{entry.value !== undefined && entry.value !== null ? formatPrice(Number(entry.value)) : 'N/A'}</span>
                    </div>
                  ))}
                </div>
              );
            }}
          />
          <Legend
            wrapperStyle={{
              paddingTop: '10px',
              fontFamily: 'DM Mono, monospace',
              fontSize: '11px'
            }}
            formatter={(value, entry: any) => {
              return <span style={{ color: 'var(--text-3)' }}>{value}</span>;
            }}
            payload={undefined}
            content={(props: any) => {
              const { payload } = props;
              if (!payload) return null;

              // Filter out the duplicate "lower" entry
              const filteredPayload = payload.filter((entry: any) =>
                entry.dataKey !== 'lower' || entry.value !== 'lower'
              );

              return (
                <ul style={{
                  listStyle: 'none',
                  padding: 0,
                  margin: 0,
                  display: 'flex',
                  flexWrap: 'wrap',
                  justifyContent: 'center'
                }}>
                  {filteredPayload.map((entry: any, index: number) => (
                    <li key={`item-${index}`} style={{
                      display: 'inline-block',
                      marginRight: '10px',
                      color: 'var(--text-3)',
                      fontFamily: 'DM Mono, monospace',
                      fontSize: '11px'
                    }}>
                      <svg width="14" height="14" viewBox="0 0 32 32" style={{
                        display: 'inline-block',
                        verticalAlign: 'middle',
                        marginRight: '4px'
                      }}>
                        <path
                          strokeWidth="4"
                          fill="none"
                          stroke={entry.color}
                          d="M0,16h10.666666666666666 A5.333333333333333,5.333333333333333,0,1,1,21.333333333333332,16 H32M21.333333333333332,16 A5.333333333333333,5.333333333333333,0,1,1,10.666666666666666,16"
                        />
                      </svg>
                      <span style={{ color: 'var(--text-3)' }}>{entry.value}</span>
                    </li>
                  ))}
                </ul>
              );
            }}
          />

          {/* All Pattern Overlays - Now using custom renderer with proper coordinates */}
          {enablePatterns && visiblePatterns.length > 0 && (
            <Customized component={(props: any) => <PatternRenderer {...props} patterns={visiblePatterns} data={data} />} />
          )}

          {/* Brush for zooming and panning */}
          <Brush
            key={brushKey}
            dataKey="date"
            height={35}
            stroke="oklch(70% 0.12 170)"
            fill="oklch(23% 0 0)"
            tickFormatter={formatDate}
            startIndex={brushRange.startIndex}
            endIndex={brushRange.endIndex}
            onChange={(range) => {
              // Update brush range when manually dragging
              if (range && range.startIndex !== undefined && range.endIndex !== undefined) {
                setBrushRange({
                  startIndex: range.startIndex,
                  endIndex: range.endIndex,
                });
                setActiveRange(null); // Clear preset selection
              }
            }}
            travellerWidth={10}
            gap={1}
          />

          {/* Volume Bars - rendered first so they're behind */}
          {showVolume && (
            <Bar
              yAxisId="volume"
              dataKey="volume"
              fill="oklch(70% 0.12 170)"
              opacity={0.3}
              name="Volume"
            />
          )}

          {/* Forecast separator line */}
          {showForecast && forecastData.length > 0 && (
            <ReferenceLine
              yAxisId="price"
              x={data[data.length - 1]?.date}
              stroke="oklch(60% 0 0)"
              strokeDasharray="5 5"
              label={{
                value: 'Forecast →',
                position: 'top',
                fill: 'oklch(60% 0 0)',
                fontSize: 11,
                fontFamily: 'DM Mono, monospace'
              }}
            />
          )}

          {/* Bollinger Bands */}
          {showBB && (
            <>
              <Area
                yAxisId="price"
                type="monotone"
                dataKey="bbUpper"
                stroke="oklch(70% 0.12 310)"
                fill="oklch(70% 0.12 310)"
                fillOpacity={0.1}
                name="BB Upper"
                strokeDasharray="3 3"
              />
              <Area
                yAxisId="price"
                type="monotone"
                dataKey="bbLower"
                stroke="oklch(70% 0.12 310)"
                fill="transparent"
                fillOpacity={0.1}
                name="BB Lower"
                strokeDasharray="3 3"
              />
            </>
          )}

          {/* Moving Averages */}
          {showMA50 && (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="ma50"
              stroke="oklch(75% 0.12 90)"
              strokeWidth={2}
              dot={false}
              name="MA 50"
            />
          )}
          {showMA20 && (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="ma20"
              stroke="oklch(70% 0.13 0)"
              strokeWidth={2}
              dot={false}
              name="MA 20"
            />
          )}

          {/* Price Line or Candlesticks */}
          {chartType === 'line' ? (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="close"
              stroke="oklch(70% 0.11 215)"
              strokeWidth={3}
              dot={false}
              name="Close Price"
            />
          ) : (
            <>
              {/* Use scatter/line to establish the Y domain properly */}
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="high"
                stroke="transparent"
                strokeWidth={0}
                dot={false}
                isAnimationActive={false}
                legendType="none"
              />
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="low"
                stroke="transparent"
                strokeWidth={0}
                dot={false}
                isAnimationActive={false}
                legendType="none"
              />
              {/* Candlestick visualization using Line with custom dot */}
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="close"
                stroke="transparent"
                strokeWidth={0}
                dot={(dotProps: any) => {
                  const { cx, cy, payload, key, index, width, height, xAxis, yAxis } = dotProps;

                  if (!payload || payload.isForecast || !payload.open || !payload.close || !payload.high || !payload.low) {
                    return null;
                  }

                  const { open, close, high, low } = payload;
                  const isGreen = close >= open;
                  const color = isGreen ? 'oklch(70% 0.12 170)' : 'oklch(70% 0.13 0)';
                  const fillColor = isGreen ? 'oklch(70% 0.12 170)' : 'oklch(23% 0 0)';

                  // Calculate bar width based on available space
                  // Get the number of visible data points from the brush range
                  const visibleDataPoints = brushRange.endIndex - brushRange.startIndex + 1;
                  // Calculate width per data point (use xAxis if available, otherwise estimate)
                  const chartWidth = width || 800; // fallback width
                  const availableWidth = chartWidth * 0.8; // accounting for margins
                  const widthPerPoint = availableWidth / visibleDataPoints;
                  // Set bar width to 60% of available space to avoid overlap, with min/max bounds
                  const barWidth = Math.max(2, Math.min(widthPerPoint * 0.6, 12));

                  // Calculate Y positions using the yAxis scale if available
                  let yHigh, yLow, yOpen, yClose;

                  if (yAxis && yAxis.scale) {
                    // Use the actual chart scale for accurate positioning
                    yHigh = yAxis.scale(high);
                    yLow = yAxis.scale(low);
                    yOpen = yAxis.scale(open);
                    yClose = yAxis.scale(close);
                  } else {
                    // Fallback to estimation
                    const priceRange = high - low;
                    const pixelsPerDollar = priceRange > 0 ? 50 / priceRange : 1;
                    const highOffset = (close - high) * pixelsPerDollar;
                    const lowOffset = (close - low) * pixelsPerDollar;
                    const openOffset = (close - open) * pixelsPerDollar;
                    yHigh = cy + highOffset;
                    yLow = cy + lowOffset;
                    yOpen = cy + openOffset;
                    yClose = cy;
                  }

                  const bodyTop = Math.min(yOpen, yClose);
                  const bodyBottom = Math.max(yOpen, yClose);
                  const bodyHeight = Math.max(Math.abs(bodyBottom - bodyTop), 1);

                  return (
                    <g key={`candle-${index}`}>
                      {/* Wick */}
                      <line
                        x1={cx}
                        y1={yHigh}
                        x2={cx}
                        y2={yLow}
                        stroke={color}
                        strokeWidth={Math.max(1, barWidth * 0.15)}
                      />
                      {/* Body */}
                      <rect
                        x={cx - barWidth / 2}
                        y={bodyTop}
                        width={barWidth}
                        height={bodyHeight}
                        fill={fillColor}
                        stroke={color}
                        strokeWidth={Math.max(1, barWidth * 0.2)}
                      />
                    </g>
                  );
                }}
                isAnimationActive={false}
                name="Price"
              />
            </>
          )}

          {/* Forecast Line */}
          {showForecast && forecastData.length > 0 && (
            <>
              {/* Confidence Interval Band - filled area between upper and lower */}
              <Area
                yAxisId="price"
                type="monotone"
                dataKey="upper"
                stroke="none"
                fill="oklch(70% 0.13 0)"
                fillOpacity={0.15}
                name="Confidence Interval"
              />
              <Area
                yAxisId="price"
                type="monotone"
                dataKey="lower"
                stroke="none"
                fill="var(--bg-4)"
                fillOpacity={1}
              />

              {/* Upper and Lower Bounds - dotted lines */}
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="upper"
                stroke="oklch(60% 0.15 30)"
                strokeWidth={1}
                strokeDasharray="3 3"
                strokeOpacity={0.7}
                dot={false}
                name="Upper Bound"
              />
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="lower"
                stroke="oklch(60% 0.15 0)"
                strokeWidth={1}
                strokeDasharray="3 3"
                strokeOpacity={0.7}
                dot={false}
                name="Lower Bound"
              />

              {/* Forecast Line - render on top */}
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="predicted"
                stroke="oklch(70% 0.15 260)"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Forecast"
              />
            </>
          )}
        </ComposedChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}
