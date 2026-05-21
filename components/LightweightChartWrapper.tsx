'use client';

import { useEffect, useRef, useState } from 'react';
import { ChartDataPoint } from '@/types';

interface Props {
  data: ChartDataPoint[];
  chartType: 'line' | 'candlestick';
  showVolume?: boolean;
  dataInterval?: string;
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

export default function LightweightChartWrapper({
  data,
  chartType,
  showVolume = true,
  dataInterval = '1d',
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const timeScaleRef = useRef<any>(null);
  const [activeRange, setActiveRange] = useState<number | 'all'>('all');

  // Build and tear down the chart whenever data/type/interval changes
  useEffect(() => {
    if (!containerRef.current || !data.length) return;

    let chart: any;
    let mounted = true;

    import('lightweight-charts').then(({ createChart, CandlestickSeries, LineSeries, HistogramSeries, ColorType }) => {
      if (!mounted || !containerRef.current) return;

      const intraday = isIntraday(dataInterval);

      chart = createChart(containerRef.current, {
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
        rightPriceScale: {
          borderColor: 'rgba(255,255,255,0.08)',
        },
      });

      timeScaleRef.current = chart.timeScale();

      const sorted = [...data].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
      const toTime = (d: string) => Math.floor(new Date(d).getTime() / 1000) as any;

      if (chartType === 'candlestick') {
        const series = chart.addSeries(CandlestickSeries, {
          upColor: '#26a69a',
          downColor: '#ef5350',
          borderVisible: false,
          wickUpColor: '#26a69a',
          wickDownColor: '#ef5350',
        });
        series.setData(sorted.map((d: ChartDataPoint) => ({
          time: toTime(d.date),
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        })));
      } else {
        const series = chart.addSeries(LineSeries, {
          color: '#5b9cf6',
          lineWidth: 2,
        });
        series.setData(sorted.map((d: ChartDataPoint) => ({
          time: toTime(d.date),
          value: d.close,
        })));
      }

      if (showVolume) {
        const volSeries = chart.addSeries(HistogramSeries, {
          priceFormat: { type: 'volume' },
          priceScaleId: 'volume',
        });
        chart.priceScale('volume').applyOptions({
          scaleMargins: { top: 0.82, bottom: 0 },
        });
        volSeries.setData(sorted.map((d: ChartDataPoint) => ({
          time: toTime(d.date),
          value: d.volume,
          color: d.close >= d.open ? 'rgba(38,166,154,0.35)' : 'rgba(239,83,80,0.35)',
        })));
      }

      chart.timeScale().fitContent();
      setActiveRange('all');
    });

    return () => {
      mounted = false;
      timeScaleRef.current = null;
      chart?.remove();
    };
  }, [data, chartType, showVolume, dataInterval]);

  const handleRange = (days: number | 'all') => {
    setActiveRange(days);
    const ts = timeScaleRef.current;
    if (!ts) return;

    if (days === 'all') {
      ts.fitContent();
      return;
    }

    const nowSec = Math.floor(Date.now() / 1000) as any;
    const fromSec = (nowSec - days * 86400) as any;
    ts.setVisibleRange({ from: fromSec, to: nowSec });
  };

  return (
    <div className="w-full">
      {/* Range buttons — same style as StockChart */}
      <div className="flex gap-2 mb-4 flex-wrap items-center">
        <div className="flex gap-2 flex-wrap">
          {RANGES.map(({ label, days }) => (
            <button
              key={label}
              onClick={() => handleRange(days)}
              className="px-3 py-1 text-xs font-medium border transition-all"
              style={{
                background: activeRange === days ? 'var(--accent)' : 'var(--bg-3)',
                borderColor: activeRange === days ? 'var(--accent)' : 'var(--bg-1)',
                color: activeRange === days ? 'var(--text-0)' : 'var(--text-3)',
              }}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div ref={containerRef} style={{ height: '460px', width: '100%' }} />
    </div>
  );
}
