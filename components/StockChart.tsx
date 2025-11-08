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
}

export default function StockChart({
  data,
  showMA20 = true,
  showMA50 = true,
  showMA200 = false,
  showBB = false,
  showForecast = true,
  forecastData = [],
}: StockChartProps) {
  // Combine historical and forecast data
  const combinedData = [
    ...data.map(d => ({ ...d, isForecast: false })),
    ...forecastData.map(f => ({
      date: f.date,
      predicted: f.predicted,
      upper: f.upper,
      lower: f.lower,
      isForecast: true,
    })),
  ];

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

  const formatPrice = (value: number) => {
    return `$${value.toFixed(2)}`;
  };

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
          <ComposedChart data={combinedData} margin={{ top: 5, right: 30, left: 20, bottom: 50 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="oklch(31% 0 0)" opacity={0.3} />
          <XAxis
            dataKey="date"
            tickFormatter={formatDate}
            stroke="oklch(75% 0 0)"
            style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace' }}
          />
          <YAxis
            tickFormatter={formatPrice}
            stroke="oklch(75% 0 0)"
            style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace' }}
            domain={['auto', 'auto']}
          />
          <Tooltip
            formatter={(value: any) => formatPrice(Number(value))}
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

          {/* Forecast separator line */}
          {showForecast && forecastData.length > 0 && (
            <ReferenceLine
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
                type="monotone"
                dataKey="bbUpper"
                stroke="oklch(70% 0.12 310)"
                fill="oklch(70% 0.12 310)"
                fillOpacity={0.1}
                name="BB Upper"
                strokeDasharray="3 3"
              />
              <Area
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
              type="monotone"
              dataKey="ma20"
              stroke="oklch(70% 0.13 0)"
              strokeWidth={2}
              dot={false}
              name="MA 20"
            />
          )}

          {/* Price Line */}
          <Line
            type="monotone"
            dataKey="close"
            stroke="oklch(70% 0.11 215)"
            strokeWidth={3}
            dot={false}
            name="Close Price"
          />

          {/* Forecast Line */}
          {showForecast && forecastData.length > 0 && (
            <>
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="oklch(70% 0.13 0)"
                strokeWidth={2}
                strokeDasharray="5 5"
                dot={false}
                name="Forecast"
              />
              <Area
                type="monotone"
                dataKey="upper"
                stroke="oklch(70% 0.13 0)"
                fill="oklch(70% 0.13 0)"
                fillOpacity={0.1}
                strokeDasharray="2 2"
                name="Upper Bound"
              />
              <Area
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
