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
  Brush,
  Bar,
  BarChart,
  Cell,
} from 'recharts';
import { ChartDataPoint } from '@/types';

interface StockChartProps {
  data: ChartDataPoint[];
  showMA20?: boolean;
  showMA50?: boolean;
  showMA200?: boolean;
  showBB?: boolean;
  showForecast?: boolean;
  forecastData?: Array<{ date: string; predicted: number; upper: number; lower: number }>;
  chartType?: 'line' | 'candlestick';
  showVolume?: boolean;
}


export default function StockChart({
  data,
  showMA20 = true,
  showMA50 = true,
  showMA200 = false,
  showBB = false,
  showForecast = true,
  forecastData = [],
  chartType = 'line',
  showVolume = true,
}: StockChartProps) {
  // Combine historical and forecast data - memoized to avoid recalculation
  const combinedData = React.useMemo(() => [
    ...data.map(d => ({ ...d, isForecast: false })),
    ...forecastData.map(f => ({
      date: f.date,
      predicted: f.predicted,
      upper: f.upper,
      lower: f.lower,
      isForecast: true,
    })),
  ], [data, forecastData]);

  // Calculate the index where forecast starts
  const historicalDataLength = data.length;

  const [brushKey, setBrushKey] = useState(0);
  const [brushRange, setBrushRange] = useState<{ startIndex: number; endIndex: number }>({
    startIndex: Math.max(0, historicalDataLength - 30), // Show last 30 days of historical data
    endIndex: combinedData.length - 1, // Include all forecast data
  });

  const [activeRange, setActiveRange] = useState<number | 'all' | 'default' | null>('default');

  // Update brush range when data changes - show recent history + forecast
  useEffect(() => {
    const startIndex = Math.max(0, historicalDataLength - 30);
    const endIndex = combinedData.length - 1; // Include forecast
    setBrushRange({
      startIndex,
      endIndex,
    });
    setBrushKey(prev => prev + 1); // Force brush remount
    setActiveRange('default'); // Set default as active
  }, [historicalDataLength, combinedData.length]);

  const handleTimeRange = (range: number | 'all' | 'default') => {
    setActiveRange(range);
    if (range === 'default') {
      // Show last 30 days + forecast
      const startIndex = Math.max(0, historicalDataLength - 30);
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
      const startIndex = Math.max(0, historicalDataLength - range);
      setBrushRange({
        startIndex,
        endIndex,
      });
      setBrushKey(prev => prev + 1); // Force brush remount
    }
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
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
      <div className="flex gap-2 mb-4 flex-wrap">
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
          onClick={() => handleTimeRange('all')}
          className="px-3 py-1 text-xs font-medium border transition-all"
          style={{
            background: activeRange === 'all' ? 'var(--accent)' : 'var(--bg-3)',
            borderColor: activeRange === 'all' ? 'var(--accent)' : 'var(--bg-1)',
            color: activeRange === 'all' ? 'var(--text-0)' : 'var(--text-3)',
          }}
        >
          ALL
        </button>
      </div>

      <div className="w-full h-[550px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={chartType === 'candlestick' ? candlestickData : combinedData} margin={{ top: 5, right: 30, left: 20, bottom: 50 }}>
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
            labelFormatter={formatDate}
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
          {showMA200 && (
            <Line
              yAxisId="price"
              type="monotone"
              dataKey="ma200"
              stroke="oklch(70% 0.12 170)"
              strokeWidth={2}
              dot={false}
              name="MA 200"
            />
          )}
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
                  const { cx, cy, payload, key, index } = dotProps;

                  if (!payload || payload.isForecast || !payload.open || !payload.close || !payload.high || !payload.low) {
                    return null;
                  }

                  // We need to calculate Y positions based on the actual price scale
                  // Since we're in a dot, we only have cx (X position) and the payload data
                  // We'll need to estimate the scale

                  const { open, close, high, low } = payload;
                  const isGreen = close >= open;
                  const color = isGreen ? 'oklch(70% 0.12 170)' : 'oklch(70% 0.13 0)';
                  const fillColor = isGreen ? 'oklch(70% 0.12 170)' : 'oklch(23% 0 0)';

                  // cy is at the close price position
                  // We need to calculate relative positions for open, high, low
                  // This is a hack but should work: estimate pixel-per-dollar ratio
                  const priceRange = high - low;
                  const pixelsPerDollar = priceRange > 0 ? 20 / priceRange : 1; // rough estimate

                  const highOffset = (close - high) * pixelsPerDollar;
                  const lowOffset = (close - low) * pixelsPerDollar;
                  const openOffset = (close - open) * pixelsPerDollar;

                  const yHigh = cy + highOffset;
                  const yLow = cy + lowOffset;
                  const yOpen = cy + openOffset;
                  const yClose = cy;

                  const bodyTop = Math.min(yOpen, yClose);
                  const bodyBottom = Math.max(yOpen, yClose);
                  const bodyHeight = Math.max(Math.abs(bodyBottom - bodyTop), 1);

                  const barWidth = 8;

                  return (
                    <g key={`candle-${index}`}>
                      {/* Wick */}
                      <line
                        x1={cx}
                        y1={yHigh}
                        x2={cx}
                        y2={yLow}
                        stroke={color}
                        strokeWidth={1}
                      />
                      {/* Body */}
                      <rect
                        x={cx - barWidth / 2}
                        y={bodyTop}
                        width={barWidth}
                        height={bodyHeight}
                        fill={fillColor}
                        stroke={color}
                        strokeWidth={1.5}
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
              <Area
                yAxisId="price"
                type="monotone"
                dataKey="upper"
                stroke="oklch(70% 0.13 0)"
                fill="oklch(70% 0.13 0)"
                fillOpacity={0.1}
                strokeDasharray="2 2"
                name="Upper Bound"
              />
              <Area
                yAxisId="price"
                type="monotone"
                dataKey="lower"
                stroke="oklch(70% 0.13 0)"
                fill="transparent"
                fillOpacity={0.1}
                strokeDasharray="2 2"
                name="Lower Bound"
              />
            </>
          )}
        </ComposedChart>
      </ResponsiveContainer>
      </div>
    </div>
  );
}
