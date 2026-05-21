'use client';

import React, { useEffect, useRef, useState } from 'react';
import { ChartDataPoint, ChartPattern } from '@/types';

type ForecastPoint = { date: string; predicted: number; upper: number; lower: number };

interface Props {
  data: ChartDataPoint[];
  chartType: 'line' | 'candlestick';
  showVolume?: boolean;
  dataInterval?: string;
  patterns?: ChartPattern[];
  enablePatterns?: boolean;
  forecastData?: ForecastPoint[];
  showForecast?: boolean;
}

const isIntraday = (interval: string) =>
  interval.endsWith('m') || interval === '60m' || interval === '1h' || interval === '90m';

const RANGES: { label: string; days: number | 'all' }[] = [
  { label: '5D',  days: 5 },
  { label: '1M',  days: 30 },
  { label: '3M',  days: 90 },
  { label: '6M',  days: 180 },
  { label: '1Y',  days: 365 },
  { label: '2Y',  days: 730 },
  { label: '3Y',  days: 1095 },
  { label: 'ALL', days: 'all' },
];

const PAT_COLORS = {
  bullish: { stroke: '#26a69a', fill: '#26a69a' },
  bearish: { stroke: '#ef5350', fill: '#ef5350' },
  neutral: { stroke: '#9b9b9b', fill: '#9b9b9b' },
};

const CHART_HEIGHT = 460;
const FONT = 'DM Mono, monospace';

// Build SVG elements for all patterns using chart coordinate APIs
function buildPatternSvg(
  chart: any,
  series: any,
  patterns: ChartPattern[],
  data: ChartDataPoint[],
): React.ReactNode[] {
  const toTime = (d: string) => Math.floor(new Date(d).getTime() / 1000) as any;
  const toX = (d: string): number | null => chart.timeScale().timeToCoordinate(toTime(d));
  const toY = (price: number): number | null => series.priceToCoordinate(price);

  const elems: React.ReactNode[] = [];

  patterns.forEach(pattern => {
    const x1 = toX(pattern.startDate);
    const x2 = toX(pattern.endDate);
    // Allow one end to be off-screen but not both
    if (x1 === null && x2 === null) return;
    const px1 = x1 ?? (x2! - 1);
    const px2 = x2 ?? (x1! + 1);

    const col = PAT_COLORS[pattern.direction];
    const label = `${pattern.label} ${(pattern.confidence * 100).toFixed(0)}%`;
    const midX = (px1 + px2) / 2;

    const slice = data.slice(pattern.startIndex, Math.min(pattern.endIndex + 1, data.length));
    if (!slice.length) return;

    const prices = slice.flatMap(d => [d.low || d.close, d.high || d.close]).filter(Boolean);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    switch (pattern.type) {
      case 'trendline_support':
      case 'trendline_resistance': {
        const isSup = pattern.type === 'trendline_support';
        const pts = slice.map(d => isSup ? (d.low || d.close) : (d.high || d.close));
        const y1 = toY(pts[0]);
        const y2 = toY(pts[pts.length - 1]);
        if (y1 === null || y2 === null) return;

        elems.push(
          <g key={pattern.id}>
            <line x1={px1} y1={y1} x2={px2} y2={y2}
              stroke={col.stroke} strokeWidth={2.5} strokeDasharray="8 4" opacity={0.85} />
            <text x={midX} y={Math.min(y1, y2) - 10} fill={col.stroke}
              fontSize={11} fontWeight={600} fontFamily={FONT} textAnchor="middle">
              {label}
            </text>
            {slice.map((d, i) => {
              const price = isSup ? (d.low || d.close) : (d.high || d.close);
              const cx = px1 + (px2 - px1) * (i / Math.max(slice.length - 1, 1));
              const cy = toY(price);
              const lineY = y1 + (y2 - y1) * (i / Math.max(slice.length - 1, 1));
              if (cy === null) return null;
              if (Math.abs(cy - lineY) < 10) {
                return <circle key={i} cx={cx} cy={cy} r={3} fill={col.fill} opacity={0.7} />;
              }
              return null;
            })}
          </g>
        );
        break;
      }

      case 'wedge_up':
      case 'wedge_down':
      case 'wedge': {
        const highs = slice.map(d => d.high || d.close);
        const lows  = slice.map(d => d.low  || d.close);
        const y1H = toY(highs[0]);
        const y2H = toY(highs[highs.length - 1]);
        const y1L = toY(lows[0]);
        const y2L = toY(lows[lows.length - 1]);
        if (y1H === null || y2H === null || y1L === null || y2L === null) return;

        elems.push(
          <g key={pattern.id}>
            <path d={`M ${px1} ${y1H} L ${px2} ${y2H} L ${px2} ${y2L} L ${px1} ${y1L} Z`}
              fill={col.fill} fillOpacity={0.08} stroke="none" />
            <line x1={px1} y1={y1H} x2={px2} y2={y2H} stroke={col.stroke} strokeWidth={2} strokeDasharray="6 3" opacity={0.7} />
            <line x1={px1} y1={y1L} x2={px2} y2={y2L} stroke={col.stroke} strokeWidth={2} strokeDasharray="6 3" opacity={0.7} />
            <text x={midX} y={(y1H + y2H + y1L + y2L) / 4}
              fill={col.stroke} fontSize={11} fontWeight={600} fontFamily={FONT} textAnchor="middle">
              {label}
            </text>
            <path d={`M ${px2 - 20} ${y2H + 10} L ${px2 - 10} ${(y2H + y2L) / 2} L ${px2 - 20} ${y2L - 10}`}
              stroke={col.stroke} strokeWidth={1.5} fill="none" opacity={0.6} />
          </g>
        );
        break;
      }

      case 'triangle_ascending':
      case 'triangle_descending':
      case 'triangle_symmetrical': {
        const highs = slice.map(d => d.high || d.close);
        const lows  = slice.map(d => d.low  || d.close);
        const maxH  = Math.max(...highs);
        const minL  = Math.min(...lows);

        let y1U: number | null, y2U: number | null, y1L: number | null, y2L: number | null;
        if (pattern.type === 'triangle_ascending') {
          y1U = y2U = toY(maxH);
          y1L = toY(lows[0]); y2L = toY(lows[lows.length - 1]);
        } else if (pattern.type === 'triangle_descending') {
          y1U = toY(highs[0]); y2U = toY(highs[highs.length - 1]);
          y1L = y2L = toY(minL);
        } else {
          y1U = toY(highs[0]); y2U = toY(highs[highs.length - 1]);
          y1L = toY(lows[0]);  y2L = toY(lows[lows.length - 1]);
        }
        if (y1U === null || y2U === null || y1L === null || y2L === null) return;

        elems.push(
          <g key={pattern.id}>
            <path d={`M ${px1} ${y1U} L ${px2} ${y2U} L ${px2} ${y2L} L ${px1} ${y1L} Z`}
              fill={col.fill} fillOpacity={0.1} stroke={col.stroke} strokeWidth={1.5} strokeDasharray="5 3" opacity={0.5} />
            <line x1={px1} y1={y1U} x2={px2} y2={y2U} stroke={col.stroke} strokeWidth={2.5} strokeDasharray="6 3" opacity={0.8} />
            <line x1={px1} y1={y1L} x2={px2} y2={y2L} stroke={col.stroke} strokeWidth={2.5} strokeDasharray="6 3" opacity={0.8} />
            <text x={midX} y={(y1U + y2U + y1L + y2L) / 4}
              fill={col.stroke} fontSize={11} fontWeight={600} fontFamily={FONT} textAnchor="middle">
              {label}
            </text>
            {pattern.direction !== 'neutral' && (
              <path d={pattern.direction === 'bullish'
                ? `M ${px2 + 5} ${(y2U + y2L) / 2} L ${px2 + 15} ${(y2U + y2L) / 2 - 10} M ${px2 + 5} ${(y2U + y2L) / 2} L ${px2 + 15} ${(y2U + y2L) / 2 + 10}`
                : `M ${px2 + 5} ${(y2U + y2L) / 2} L ${px2 + 15} ${(y2U + y2L) / 2 + 10} M ${px2 + 5} ${(y2U + y2L) / 2} L ${px2 + 15} ${(y2U + y2L) / 2 - 10}`}
                stroke={col.stroke} strokeWidth={2} fill="none" opacity={0.7} />
            )}
          </g>
        );
        break;
      }

      case 'channel_up':
      case 'channel_down':
      case 'channel': {
        const highs = slice.map(d => d.high || d.close);
        const lows  = slice.map(d => d.low  || d.close);
        const y1H = toY(highs[0]);
        const y2H = toY(highs[highs.length - 1]);
        const y1L = toY(lows[0]);
        const y2L = toY(lows[lows.length - 1]);
        if (y1H === null || y2H === null || y1L === null || y2L === null) return;

        elems.push(
          <g key={pattern.id}>
            <path d={`M ${px1} ${y1H} L ${px2} ${y2H} L ${px2} ${y2L} L ${px1} ${y1L} Z`}
              fill={col.fill} fillOpacity={0.06} stroke="none" />
            <line x1={px1} y1={y1H} x2={px2} y2={y2H} stroke={col.stroke} strokeWidth={2} strokeDasharray="8 4" opacity={0.7} />
            <line x1={px1} y1={y1L} x2={px2} y2={y2L} stroke={col.stroke} strokeWidth={2} strokeDasharray="8 4" opacity={0.7} />
            <text x={midX} y={(y1H + y2H + y1L + y2L) / 4}
              fill={col.stroke} fontSize={11} fontWeight={600} fontFamily={FONT} textAnchor="middle">
              {label}
            </text>
            <line x1={midX - 15} y1={(y1H + y2H) / 2} x2={midX - 15} y2={(y1L + y2L) / 2} stroke={col.stroke} strokeWidth={1.5} opacity={0.5} />
            <line x1={midX + 15} y1={(y1H + y2H) / 2} x2={midX + 15} y2={(y1L + y2L) / 2} stroke={col.stroke} strokeWidth={1.5} opacity={0.5} />
          </g>
        );
        break;
      }

      case 'double_top':
      case 'double_bottom': {
        const isTop = pattern.type === 'double_top';
        const level = (pattern.meta?.level as number) || (isTop ? maxPrice : minPrice);
        const yLevel = toY(level);
        if (yLevel === null) return;

        elems.push(
          <g key={pattern.id}>
            <line x1={px1} y1={yLevel} x2={px2} y2={yLevel}
              stroke={col.stroke} strokeWidth={2.5} strokeDasharray="6 3" opacity={0.8} />
            <rect x={px1 + 5} y={yLevel + (isTop ? -20 : 5)} width={label.length * 6.5} height={14}
              fill="oklch(23% 0 0)" fillOpacity={0.85} rx={2} />
            <text x={px1 + 8} y={yLevel + (isTop ? -9 : 16)}
              fill={col.stroke} fontSize={10} fontWeight={600} fontFamily={FONT} textAnchor="start">
              {label}
            </text>
          </g>
        );
        break;
      }

      case 'head_and_shoulders': {
        const head     = (pattern.meta?.head as number) || maxPrice;
        const neckline = (pattern.meta?.neckline as number) || minPrice;
        const yHead = toY(head);
        const yNeck = toY(neckline);
        if (yHead === null || yNeck === null) return;

        elems.push(
          <g key={pattern.id}>
            <rect x={px1} y={Math.min(yHead, yNeck)} width={Math.max(0, px2 - px1)} height={Math.abs(yNeck - yHead)}
              fill={col.fill} fillOpacity={0.05} stroke="none" />
            <line x1={px1} y1={yNeck} x2={px2} y2={yNeck}
              stroke={col.stroke} strokeWidth={2.5} strokeDasharray="8 4" opacity={0.85} />
            <rect x={px1 + 5} y={yHead - 20} width={label.length * 6.5} height={14}
              fill="oklch(23% 0 0)" fillOpacity={0.85} rx={2} />
            <text x={px1 + 8} y={yHead - 9}
              fill={col.stroke} fontSize={10} fontWeight={600} fontFamily={FONT} textAnchor="start">
              {label}
            </text>
          </g>
        );
        break;
      }

      case 'horizontal_sr': {
        const resistance = pattern.meta?.resistance as number;
        const support    = pattern.meta?.support    as number;
        if (!resistance || !support) return;
        const yR = toY(resistance);
        const yS = toY(support);
        if (yR === null || yS === null) return;

        elems.push(
          <g key={pattern.id}>
            <rect x={px1} y={Math.min(yR, yS)} width={Math.max(0, px2 - px1)} height={Math.abs(yS - yR)}
              fill={col.fill} fillOpacity={0.08} stroke="none" />
            <line x1={px1} y1={yR} x2={px2} y2={yR} stroke={col.stroke} strokeWidth={2} strokeDasharray="6 2" opacity={0.8} />
            <line x1={px1} y1={yS} x2={px2} y2={yS} stroke={col.stroke} strokeWidth={2} strokeDasharray="6 2" opacity={0.8} />
            <rect x={px1 + 5} y={yR - 18} width={Math.max(label.length * 6.5, 120)} height={14}
              fill="oklch(23% 0 0)" fillOpacity={0.85} rx={2} />
            <text x={px1 + 8} y={yR - 7}
              fill={col.stroke} fontSize={10} fontWeight={600} fontFamily={FONT} textAnchor="start">
              {label}
            </text>
          </g>
        );
        break;
      }

      case 'multiple_top':
      case 'multiple_bottom': {
        const isTop  = pattern.type === 'multiple_top';
        const level  = (pattern.meta?.level as number) || (isTop ? maxPrice : minPrice);
        const touches = (pattern.meta?.touches as number) || 3;
        const yLevel  = toY(level);
        if (yLevel === null) return;
        const fullLabel = `${label} • ${touches}×`;

        elems.push(
          <g key={pattern.id}>
            <line x1={px1} y1={yLevel} x2={px2} y2={yLevel}
              stroke={col.stroke} strokeWidth={2.5} strokeDasharray="5 3" opacity={0.8} />
            <rect x={px1 + 5} y={yLevel + (isTop ? -20 : 5)} width={Math.max(fullLabel.length * 6.5, 100)} height={14}
              fill="oklch(23% 0 0)" fillOpacity={0.85} rx={2} />
            <text x={px1 + 8} y={yLevel + (isTop ? -9 : 16)}
              fill={col.stroke} fontSize={10} fontWeight={600} fontFamily={FONT} textAnchor="start">
              {fullLabel}
            </text>
          </g>
        );
        break;
      }
    }
  });

  return elems;
}

export default function LightweightChartWrapper({
  data,
  chartType,
  showVolume = true,
  dataInterval = '1d',
  patterns = [],
  enablePatterns = false,
  forecastData = [],
  showForecast = true,
}: Props) {
  const chartDivRef  = useRef<HTMLDivElement>(null);
  const chartObjRef  = useRef<any>(null);
  const seriesRef    = useRef<any>(null);
  const timeScaleRef = useRef<any>(null);
  // Stable ref so subscriptions always call the latest render
  const renderRef    = useRef<() => void>(() => {});

  const [activeRange,  setActiveRange]  = useState<number | 'all'>('all');
  const [patternElems, setPatternElems] = useState<React.ReactNode[]>([]);

  // ── Pattern re-render (called on zoom/pan AND when patterns/data changes) ──
  useEffect(() => {
    renderRef.current = () => {
      const chart  = chartObjRef.current;
      const series = seriesRef.current;
      if (!chart || !series || !enablePatterns || !patterns.length) {
        setPatternElems([]);
        return;
      }
      setPatternElems(buildPatternSvg(chart, series, patterns, data));
    };

    // Re-render immediately whenever patterns or data changes
    renderRef.current();
  }, [patterns, enablePatterns, data]);

  // ── Chart setup ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!chartDivRef.current || !data.length) return;

    let chart: any;
    let mounted = true;

    import('lightweight-charts').then(({
      createChart,
      CandlestickSeries,
      LineSeries,
      HistogramSeries,
      ColorType,
    }) => {
      if (!mounted || !chartDivRef.current) return;

      const intraday = isIntraday(dataInterval);

      chart = createChart(chartDivRef.current, {
        autoSize: true,
        layout: {
          background: { type: ColorType.Solid, color: 'transparent' },
          textColor: 'rgba(180,180,180,0.9)',
        },
        grid: {
          vertLines: { color: 'rgba(255,255,255,0.04)' },
          horzLines: { color: 'rgba(255,255,255,0.04)' },
        },
        crosshair: { mode: 1 },
        timeScale: {
          borderColor: 'rgba(255,255,255,0.08)',
          timeVisible: intraday,
          secondsVisible: false,
        },
        rightPriceScale: { borderColor: 'rgba(255,255,255,0.08)' },
      });

      chartObjRef.current  = chart;
      timeScaleRef.current = chart.timeScale();

      const sorted  = [...data].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
      const toTime  = (d: string) => Math.floor(new Date(d).getTime() / 1000) as any;

      let priceSeries: any;
      if (chartType === 'candlestick') {
        priceSeries = chart.addSeries(CandlestickSeries, {
          upColor: '#26a69a', downColor: '#ef5350',
          borderVisible: false,
          wickUpColor: '#26a69a', wickDownColor: '#ef5350',
        });
        priceSeries.setData(sorted.map((d: ChartDataPoint) => ({
          time: toTime(d.date), open: d.open, high: d.high, low: d.low, close: d.close,
        })));
      } else {
        priceSeries = chart.addSeries(LineSeries, { color: '#5b9cf6', lineWidth: 2 });
        priceSeries.setData(sorted.map((d: ChartDataPoint) => ({
          time: toTime(d.date), value: d.close,
        })));
      }

      seriesRef.current = priceSeries;

      if (showVolume) {
        const vol = chart.addSeries(HistogramSeries, {
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
        });
        chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
        vol.setData(sorted.map((d: ChartDataPoint) => ({
          time: toTime(d.date),
          value: d.volume,
          color: d.close >= d.open ? 'rgba(38,166,154,0.35)' : 'rgba(239,83,80,0.35)',
        })));
      }

      // ── Forecast overlay ─────────────────────────────────────────────────
      if (showForecast && forecastData.length > 0) {
        const sortedFc = [...forecastData].sort(
          (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
        );

        // Bridge: start each forecast series from the last historical close
        const lastBar = sorted[sorted.length - 1];
        const bridge = { date: lastBar.date, predicted: lastBar.close, upper: lastBar.close, lower: lastBar.close };
        const fcWithBridge = [bridge, ...sortedFc];

        // Predicted line
        const predSeries = chart.addSeries(LineSeries, {
          color: 'rgba(251,191,36,0.9)',
          lineWidth: 2,
          lineStyle: 0,
          lastValueVisible: false,
          priceLineVisible: false,
        });
        predSeries.setData(fcWithBridge.map(f => ({ time: toTime(f.date), value: f.predicted })));

        // Upper bound (dashed)
        const upperSeries = chart.addSeries(LineSeries, {
          color: 'rgba(251,191,36,0.35)',
          lineWidth: 1,
          lineStyle: 1, // dashed
          lastValueVisible: false,
          priceLineVisible: false,
        });
        upperSeries.setData(fcWithBridge.map(f => ({ time: toTime(f.date), value: f.upper })));

        // Lower bound (dashed)
        const lowerSeries = chart.addSeries(LineSeries, {
          color: 'rgba(251,191,36,0.35)',
          lineWidth: 1,
          lineStyle: 1,
          lastValueVisible: false,
          priceLineVisible: false,
        });
        lowerSeries.setData(fcWithBridge.map(f => ({ time: toTime(f.date), value: f.lower })));
      }

      chart.timeScale().fitContent();
      setActiveRange('all');

      // Re-render pattern overlay whenever the user zooms or pans
      const onUpdate = () => renderRef.current();
      chart.timeScale().subscribeVisibleTimeRangeChange(onUpdate);
      chart.timeScale().subscribeSizeChange(onUpdate);

      // Initial pattern render (coordinates now valid)
      renderRef.current();
    });

    return () => {
      mounted = false;
      chartObjRef.current  = null;
      seriesRef.current    = null;
      timeScaleRef.current = null;
      chart?.remove();
    };
  }, [data, chartType, showVolume, dataInterval, forecastData, showForecast]);

  const handleRange = (days: number | 'all') => {
    setActiveRange(days);
    const ts = timeScaleRef.current;
    if (!ts) return;
    if (days === 'all') { ts.fitContent(); return; }
    const now  = Math.floor(Date.now() / 1000) as any;
    const from = (now - (days as number) * 86400) as any;
    ts.setVisibleRange({ from, to: now });
  };

  return (
    <div className="w-full">
      {/* Range buttons */}
      <div className="flex gap-2 mb-4 flex-wrap items-center">
        <div className="flex gap-2 flex-wrap">
          {RANGES.map(({ label, days }) => (
            <button
              key={label}
              onClick={() => handleRange(days)}
              className="px-3 py-1 text-xs font-medium border transition-all"
              style={{
                background:   activeRange === days ? 'var(--accent)' : 'var(--bg-3)',
                borderColor:  activeRange === days ? 'var(--accent)' : 'var(--bg-1)',
                color:        activeRange === days ? 'var(--text-0)' : 'var(--text-3)',
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Chart + SVG pattern overlay */}
      <div className="relative" style={{ height: CHART_HEIGHT }}>
        <div ref={chartDivRef} style={{ height: '100%', width: '100%' }} />

        {enablePatterns && patternElems.length > 0 && (
          <svg
            style={{
              position: 'absolute', top: 0, left: 0,
              width: '100%', height: '100%',
              pointerEvents: 'none', overflow: 'visible',
            }}
          >
            {patternElems}
          </svg>
        )}
      </div>
    </div>
  );
}
