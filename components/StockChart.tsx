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

  const visiblePatterns = React.useMemo(() => {
    if (!patterns?.length) return [];
    const start = brushRange.startIndex ?? 0;
    const end = brushRange.endIndex ?? combinedData.length - 1;
    const cappedEnd = Math.min(end, data.length - 1);
    return patterns.filter(
      pattern => pattern.endIndex >= start && pattern.startIndex <= cappedEnd
    );
  }, [patterns, brushRange, combinedData.length, data.length]);

  const patternBadges = React.useMemo(() => patterns.slice(0, 6), [patterns]);

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
    setBrushRange({
      startIndex,
      endIndex,
    });
    setBrushKey(prev => prev + 1); // Force brush remount
    setActiveRange('default'); // Set default as active
  }, [historicalDataLength, combinedData.length, getStartIndexForDays]);

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

      {/* Interactive Controls Help */}
      <div className="text-xs mb-2 p-2 border" style={{ 
        background: 'var(--bg-2)', 
        borderColor: 'var(--bg-1)',
        color: 'var(--text-4)'
      }}>
        <strong style={{ color: 'var(--text-3)' }}>💡 Interactive Controls:</strong> Drag the brush at bottom to pan • Ctrl+Scroll to zoom • Keyboard: +/- zoom, R reset • Click buttons above
      </div>

      {data.length > 0 && (
        patterns.length > 0 ? (
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
              patterns ({patterns.length})
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
            No Finviz-style patterns detected on this timeframe.
          </div>
        )
      )}

      <div ref={chartRef} className="w-full h-[550px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={chartType === 'candlestick' ? candlestickData : combinedData}
            margin={{ top: 5, right: 30, left: 20, bottom: 50 }}
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
            formatter={(value: any, name: any, props: any) => {
              if (chartType === 'candlestick' && props?.payload) {
                const { open, high, low, close } = props.payload;
                // Check if OHLC data exists
                if (open !== undefined && high !== undefined && low !== undefined && close !== undefined) {
                  return [
                    <div key="ohlc" style={{ fontSize: '11px' }}>
                      <div>O: {formatPrice(open)}</div>
                      <div>H: {formatPrice(high)}</div>
                      <div>L: {formatPrice(low)}</div>
                      <div>C: {formatPrice(close)}</div>
                    </div>,
                    'OHLC'
                  ];
                }
              }
              // Safe formatting with null check
              if (value === undefined || value === null) return 'N/A';
              return formatPrice(Number(value));
            }}
            labelFormatter={formatTooltipLabel}
            contentStyle={{
              backgroundColor: 'oklch(23% 0 0)',
              border: '2px solid oklch(70% 0.12 170)',
              borderRadius: '0',
              color: 'oklch(85% 0 0)',
              fontFamily: 'DM Mono, monospace',
              fontSize: '12px',
            }}
            labelStyle={{
              color: 'oklch(70% 0.12 170)',
              fontWeight: 'bold',
            }}
          />
          <Legend
            wrapperStyle={{
              paddingTop: '10px',
              fontFamily: 'DM Mono, monospace',
              fontSize: '11px'
            }}
          />

          {visiblePatterns.map(pattern => {
            const style = PATTERN_DIRECTION_STYLES[pattern.direction];
            const minPrice =
              typeof pattern.meta?.priceMin === 'number'
                ? pattern.meta.priceMin
                : priceExtents.min;
            const maxPrice =
              typeof pattern.meta?.priceMax === 'number'
                ? pattern.meta.priceMax
                : priceExtents.max;
            const labelValue = `${pattern.label} ${(pattern.confidence * 100).toFixed(
              0
            )}%`;

            return (
              <ReferenceArea
                key={`${pattern.id}-${pattern.startDate}`}
                yAxisId="price"
                x1={pattern.startDate}
                x2={pattern.endDate}
                y1={minPrice}
                y2={maxPrice}
                stroke={style.stroke}
                fill={style.fill}
                fillOpacity={0.08}
                strokeOpacity={0.6}
                strokeDasharray="4 4"
                ifOverflow="extendDomain"
                label={{
                  value: labelValue,
                  position: 'top',
                  fill: style.stroke,
                  fontSize: 10,
                  fontWeight: 600,
                  fontFamily: 'DM Mono, monospace',
                }}
              />
            );
          })}

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
              // Reset active range when manually dragging
              if (range && range.startIndex !== undefined && range.endIndex !== undefined) {
                setActiveRange(null);
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
                name=""
              />

              {/* Upper and Lower Bounds - dotted lines */}
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="upper"
                stroke="oklch(70% 0.13 0)"
                strokeWidth={1}
                strokeDasharray="3 3"
                strokeOpacity={0.5}
                dot={false}
                name="Upper Bound"
              />
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="lower"
                stroke="oklch(70% 0.13 0)"
                strokeWidth={1}
                strokeDasharray="3 3"
                strokeOpacity={0.5}
                dot={false}
                name="Lower Bound"
              />

              {/* Forecast Line - render on top */}
              <Line
                yAxisId="price"
                type="monotone"
                dataKey="predicted"
                stroke="oklch(70% 0.13 0)"
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
