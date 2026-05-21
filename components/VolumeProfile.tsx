'use client';

import React, { useMemo, useState } from 'react';
import { ChartDataPoint } from '@/types';

interface Props {
  chartData:     ChartDataPoint[];
  currentPrice?: number;
  buckets?:      number;
  inlineMobile?: boolean;
}

interface Bucket {
  priceMid: number;
  totalVol: number;
  bullVol:  number;
  bearVol:  number;
}

function fmt(v: number): string {
  if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B';
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
  return String(Math.round(v));
}

const BAR_H = 120;

interface Insight { text: string; color?: string }

function generateInsights(
  profile:      Bucket[],
  pocIdx:       number,
  hvnThreshold: number,
  currentPrice?: number,
): Insight[] {
  if (!profile.length) return [];
  const out: Insight[] = [];

  const poc      = profile[pocIdx];
  const totalVol = profile.reduce((s, b) => s + b.totalVol, 0);
  const bullPct  = totalVol > 0
    ? Math.round(profile.reduce((s, b) => s + b.bullVol, 0) / totalVol * 100) : 50;

  const hvns = profile
    .map((b, i) => ({ ...b, idx: i }))
    .filter(b => b.idx !== pocIdx && b.totalVol >= hvnThreshold);

  // 1. Price vs POC
  if (currentPrice && poc) {
    const pct    = ((currentPrice - poc.priceMid) / poc.priceMid) * 100;
    const above  = currentPrice > poc.priceMid;
    if (Math.abs(pct) < 1.5) {
      out.push({ text: `Price is right at the POC ($${poc.priceMid.toFixed(2)}), the highest-volume price in this range. Expect consolidation and indecision — this is the market's "fair value" level.`, color: '#f59e0b' });
    } else if (above) {
      out.push({ text: `Price is ${pct.toFixed(1)}% above the POC ($${poc.priceMid.toFixed(2)}). The POC is a natural pullback target — buyers likely step back in there on dips.`, color: '#22c55e' });
    } else {
      out.push({ text: `Price is ${Math.abs(pct).toFixed(1)}% below the POC ($${poc.priceMid.toFixed(2)}), which now acts as overhead resistance. Reclaiming the POC would be a bullish shift.`, color: '#ef4444' });
    }
  } else if (poc) {
    out.push({ text: `POC at $${poc.priceMid.toFixed(2)} — the most-traded price in this period and the market's consensus fair value.`, color: '#f59e0b' });
  }

  // 2. Nearest HVN support and resistance
  if (currentPrice && hvns.length) {
    const below = hvns.filter(b => b.priceMid < currentPrice).sort((a, b) => b.priceMid - a.priceMid);
    const above = hvns.filter(b => b.priceMid > currentPrice).sort((a, b) => a.priceMid - b.priceMid);
    if (below.length)
      out.push({ text: `Nearest HVN support: $${below[0].priceMid.toFixed(2)} — heavy historical volume here tends to attract buyers on pullbacks.`, color: '#26a69a' });
    if (above.length)
      out.push({ text: `Nearest HVN resistance: $${above[0].priceMid.toFixed(2)} — sellers were active at this level; price may stall or reject here on the way up.`, color: '#ef5350' });
  }

  // 3. LVN gap between price and POC (fast-move zone)
  if (currentPrice && poc) {
    const abovePOC = currentPrice > poc.priceMid;
    const lo       = Math.min(currentPrice, poc.priceMid);
    const hi       = Math.max(currentPrice, poc.priceMid);
    const span     = profile.filter(b => b.priceMid >= lo && b.priceMid <= hi);
    const avg      = span.length ? span.reduce((s, b) => s + b.totalVol, 0) / span.length : 0;
    const lvnBuckets = span.filter(b => b.totalVol < avg * 0.35);
    if (lvnBuckets.length && Math.abs(currentPrice - poc.priceMid) / poc.priceMid > 0.03) {
      const lvnLo = Math.min(...lvnBuckets.map(b => b.priceMid)).toFixed(2);
      const lvnHi = Math.max(...lvnBuckets.map(b => b.priceMid)).toFixed(2);
      const dir   = abovePOC ? 'drops below' : 'breaks above';
      const target = abovePOC
        ? `the $${lvnLo}–$${lvnHi} gap has thin support — a sell-off could accelerate quickly toward the POC at $${poc.priceMid.toFixed(2)}`
        : `the $${lvnLo}–$${lvnHi} gap has little resistance — a rally could push rapidly to the POC at $${poc.priceMid.toFixed(2)}`;
      out.push({ text: `Thin-volume zone at $${lvnLo}–$${lvnHi}: if price ${dir} $${abovePOC ? lvnHi : lvnLo}, ${target}.`, color: '#a78bfa' });
    }
  }

  // 4. Volume bias
  if (bullPct >= 62)
    out.push({ text: `Volume skews bullish (${bullPct}% of bars closed up in this range). Most participants were buyers — suggests underlying accumulation.`, color: '#22c55e' });
  else if (bullPct <= 38)
    out.push({ text: `Volume skews bearish (${100 - bullPct}% of bars closed down). Sellers dominated — suggests distribution at these levels.`, color: '#ef4444' });
  else
    out.push({ text: `Volume is evenly split (${bullPct}% bull / ${100 - bullPct}% bear) — no strong directional bias in who's been in control.`, color: 'var(--text-4)' });

  return out;
}

const PERIODS: { label: string; days: number | null }[] = [
  { label: '3M',  days: 90  },
  { label: '6M',  days: 180 },
  { label: '1Y',  days: 365 },
  { label: '2Y',  days: 730 },
  { label: 'All', days: null },
];

export default function VolumeProfile({ chartData, currentPrice, buckets = 50, inlineMobile }: Props) {
  const [period, setPeriod] = useState<number | null>(365); // default 1Y
  const [tooltip, setTooltip] = useState<{
    price: string; total: string; bull: string; bear: string; x: number; y: number;
  } | null>(null);

  // Filter to selected period
  const filteredData = useMemo(() => {
    if (!period) return chartData;
    const cutoff = Date.now() - period * 24 * 60 * 60 * 1000;
    return chartData.filter(d => new Date(d.date).getTime() >= cutoff);
  }, [chartData, period]);

  // Build buckets low→high (left→right on the horizontal chart)
  const profile = useMemo<Bucket[]>(() => {
    if (!filteredData.length) return [];
    const allPrices = filteredData
      .flatMap(d => [d.high, d.low])
      .filter((p): p is number => p != null && isFinite(p));
    const minP  = Math.min(...allPrices);
    const maxP  = Math.max(...allPrices);
    const range = maxP - minP;
    if (range <= 0) return [];

    const size   = range / buckets;
    const result: Bucket[] = Array.from({ length: buckets }, (_, i) => ({
      priceMid: minP + (i + 0.5) * size,
      totalVol: 0, bullVol: 0, bearVol: 0,
    }));

    filteredData.forEach(d => {
      if (!d.volume || d.high == null || d.low == null) return;
      const candleRange = d.high - d.low;
      const isBull = d.close >= d.open;
      if (candleRange <= 0) {
        const idx = Math.max(0, Math.min(buckets - 1, Math.floor((d.close - minP) / size)));
        result[idx].totalVol += d.volume;
        if (isBull) result[idx].bullVol += d.volume;
        else        result[idx].bearVol += d.volume;
        return;
      }
      result.forEach((b, i) => {
        const lo = minP + i * size;
        const hi = lo + size;
        const overlap = Math.min(hi, d.high) - Math.max(lo, d.low);
        if (overlap <= 0) return;
        const vol = d.volume * (overlap / candleRange);
        b.totalVol += vol;
        if (isBull) b.bullVol += vol;
        else        b.bearVol += vol;
      });
    });

    return result; // low→high (left→right)
  }, [filteredData, buckets]);

  const maxVol      = Math.max(...profile.map(b => b.totalVol), 1);
  const pocIdx      = profile.reduce((best, b, i) => b.totalVol > profile[best].totalVol ? i : best, 0);
  const totalVolAll = profile.reduce((s, b) => s + b.totalVol, 0);
  const bullPct     = totalVolAll > 0
    ? Math.round(profile.reduce((s, b) => s + b.bullVol, 0) / totalVolAll * 100)
    : 50;

  if (!profile.length) return null;

  // HVN threshold: any bar within 70% of the POC volume is a significant node
  const hvnThreshold = maxVol * 0.70;

  const pocPrice = profile[pocIdx]?.priceMid ?? 0;

  const insights = useMemo(
    () => generateInsights(profile, pocIdx, hvnThreshold, currentPrice),
    [profile, pocIdx, hvnThreshold, currentPrice]
  );

  // Actual date range of the filtered data
  const dateRange = useMemo(() => {
    if (!filteredData.length) return '';
    const sorted  = [...filteredData].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    const fmt2    = (d: string) => new Date(d).toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
    return `${fmt2(sorted[0].date)} – ${fmt2(sorted[sorted.length - 1].date)}`;
  }, [filteredData]);
  const loPrice  = profile[0]?.priceMid ?? 0;
  const hiPrice  = profile[profile.length - 1]?.priceMid ?? 0;

  // X-axis label indices: lo, 25%, POC, 75%, hi — deduplicated & sorted
  const labelIdxs = Array.from(new Set([
    0,
    Math.round(profile.length * 0.25),
    pocIdx,
    Math.round(profile.length * 0.75),
    profile.length - 1,
  ])).sort((a, b) => a - b);

  return (
    <div className={`card ${inlineMobile ? 'w-full' : ''}`}>
      <span className="card-label">Volume Profile (VPVR)</span>

      {/* Header: stats + period selector */}
      <div className="flex items-center gap-2 mb-2 flex-wrap" style={{ fontSize: 11 }}>
        <span style={{ color: 'var(--text-4)' }}>
          POC <strong style={{ color: '#f59e0b' }}>${pocPrice.toFixed(2)}</strong>
        </span>
        <span style={{ color: '#26a69a' }}>▮ {bullPct}%</span>
        <span style={{ color: '#ef5350' }}>▮ {100 - bullPct}%</span>
        <span style={{ color: 'var(--text-5)' }}>{fmt(totalVolAll)}</span>
        <div className="flex gap-1 ml-auto">
          {PERIODS.map(p => (
            <button
              key={p.label}
              onClick={() => setPeriod(p.days)}
              style={{
                fontSize: 9, padding: '1px 5px', border: '1px solid',
                borderRadius: 2, cursor: 'pointer',
                background: period === p.days ? 'var(--accent)' : 'var(--bg-3)',
                borderColor: period === p.days ? 'var(--accent)' : 'var(--bg-1)',
                color: period === p.days ? 'var(--text-0)' : 'var(--text-4)',
              }}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      {/* Horizontal histogram — bars grow upward, price on X-axis */}
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-end',
          height: BAR_H,
          gap: 1,
          position: 'relative',
        }}
        onMouseLeave={() => setTooltip(null)}
      >
        {profile.map((b, i) => {
          const barH     = (b.totalVol / maxVol) * BAR_H;
          const bullFrac = b.totalVol > 0 ? b.bullVol / b.totalVol : 0;
          const bearFrac = 1 - bullFrac;
          const isPOC    = i === pocIdx;
          const isHVN    = !isPOC && b.totalVol >= hvnThreshold;

          // Colour scheme: POC = solid amber, HVN = muted amber tint over green/red, normal = green/red
          const bullColor = isPOC ? '#f59e0b' : isHVN ? '#d97706' : '#26a69a';
          const bearColor = isPOC ? '#f59e0b' : isHVN ? '#b45309' : '#ef5350';
          const bullOpacity = isPOC ? 1 : isHVN ? 0.9 : 0.85;
          const bearOpacity = isPOC ? 0.95 : isHVN ? 0.8 : 0.72;

          return (
            <div
              key={i}
              style={{
                flex: 1,
                height: Math.max(barH, 1),
                display: 'flex',
                flexDirection: 'column',
                cursor: 'default',
                outline: isPOC
                  ? '1px solid rgba(245,158,11,0.9)'
                  : isHVN
                    ? '1px solid rgba(217,119,6,0.5)'
                    : 'none',
              }}
              onMouseEnter={(e) => {
                const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
                setTooltip({
                  price: `$${b.priceMid.toFixed(2)}`,
                  total: fmt(b.totalVol),
                  bull:  fmt(b.bullVol),
                  bear:  fmt(b.bearVol),
                  x: r.left + r.width / 2,
                  y: r.top - 70,
                });
              }}
            >
              {/* Selling pressure (top) */}
              <div style={{
                flex: bearFrac || 0,
                background: bearColor,
                opacity: bearOpacity,
                minHeight: bearFrac > 0 ? 1 : 0,
              }} />
              {/* Buying pressure (bottom) */}
              <div style={{
                flex: bullFrac || 0,
                background: bullColor,
                opacity: bullOpacity,
                minHeight: bullFrac > 0 ? 1 : 0,
              }} />
            </div>
          );
        })}
      </div>

      {/* Baseline */}
      <div style={{ height: 1, background: 'var(--bg-1)', marginBottom: 2 }} />

      {/* X-axis price labels */}
      <div style={{ position: 'relative', height: 14 }}>
        {labelIdxs.map(idx => {
          const leftPct = profile.length > 1 ? (idx / (profile.length - 1)) * 100 : 0;
          const price   = profile[idx]?.priceMid ?? 0;
          const isPOC   = idx === pocIdx;
          return (
            <div
              key={idx}
              style={{
                position: 'absolute',
                left: `${leftPct}%`,
                transform: 'translateX(-50%)',
                fontSize: 8.5,
                color: isPOC ? '#f59e0b' : 'var(--text-5)',
                fontWeight: isPOC ? 700 : 400,
                whiteSpace: 'nowrap',
                lineHeight: '14px',
              }}
            >
              ${price.toFixed(0)}
            </div>
          );
        })}
      </div>

      {/* Date range */}
      {dateRange && (
        <div style={{ textAlign: 'center', fontSize: 9, color: 'var(--text-5)', marginTop: 3 }}>
          {dateRange}
        </div>
      )}

      {/* How to read */}
      <div style={{ marginTop: 8, fontSize: 9.5, color: 'var(--text-5)', lineHeight: 1.5 }}>
        Bar height = volume at that price.{' '}
        <span style={{ color: '#f59e0b', fontWeight: 600 }}>POC</span> = most-traded price.{' '}
        <span style={{ color: '#d97706', fontWeight: 600 }}>HVNs</span> = other high-volume nodes (support/resistance).{' '}
        <span style={{ color: '#26a69a' }}>Green</span> = buying pressure · <span style={{ color: '#ef5350' }}>Red</span> = selling pressure.
      </div>

      {/* Smart insights */}
      {insights.length > 0 && (
        <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 3 }}>
          {insights.map((ins, i) => (
            <div key={i} style={{ display: 'flex', gap: 5, alignItems: 'baseline', fontSize: 10, color: 'var(--text-4)', lineHeight: 1.45 }}>
              <span style={{ color: ins.color ?? 'var(--text-5)', flexShrink: 0, fontSize: 8 }}>●</span>
              {ins.text}
            </div>
          ))}
        </div>
      )}

      {tooltip && (
        <div style={{
          position: 'fixed',
          left: tooltip.x,
          top: tooltip.y,
          transform: 'translateX(-50%)',
          background: 'var(--bg-2)', border: '1px solid var(--bg-1)',
          padding: '4px 8px', borderRadius: 3,
          fontSize: 11, color: 'var(--text-2)', zIndex: 9999,
          pointerEvents: 'none', whiteSpace: 'nowrap', lineHeight: 1.7,
        }}>
          <div style={{ fontWeight: 700 }}>{tooltip.price}</div>
          <div>Vol: <strong>{tooltip.total}</strong></div>
          <div style={{ color: '#26a69a' }}>Bull {tooltip.bull}</div>
          <div style={{ color: '#ef5350' }}>Bear {tooltip.bear}</div>
        </div>
      )}
    </div>
  );
}
