'use client';

import { useEffect, useRef } from 'react';
import { ChartDataPoint } from '@/types';

interface Props {
  data: ChartDataPoint[];
  chartType: 'line' | 'candlestick';
  showVolume?: boolean;
  dataInterval?: string; // '5m', '15m', '60m', '1d', '1wk', '1mo'
}

const isIntraday = (interval: string) =>
  interval.endsWith('m') || interval === '60m' || interval === '1h' || interval === '90m';

export default function LightweightChartWrapper({
  data,
  chartType,
  showVolume = true,
  dataInterval = '1d',
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);

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
          tickMarkFormatter: intraday
            ? (time: number) => {
                const d = new Date(time * 1000);
                const h = d.getUTCHours().toString().padStart(2, '0');
                const m = d.getUTCMinutes().toString().padStart(2, '0');
                return `${h}:${m}`;
              }
            : undefined,
        },
        rightPriceScale: {
          borderColor: 'rgba(255,255,255,0.08)',
        },
      });

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
    });

    return () => {
      mounted = false;
      chart?.remove();
    };
  }, [data, chartType, showVolume, dataInterval]);

  return <div ref={containerRef} style={{ height: '500px', width: '100%' }} />;
}
