'use client';

import React, { useMemo, useState } from 'react';
import { ChartDataPoint } from '@/types';

interface Props {
  chartData: ChartDataPoint[];
  inlineMobile?: boolean;
}

const CELL = 15;
const GAP  = 2;

function getColor(pct: number): string {
  if (pct >  4)   return '#15803d';
  if (pct >  2)   return '#22c55e';
  if (pct >  0.5) return '#86efac';
  if (pct >  0)   return '#d1fae5';
  if (pct < -4)   return '#991b1b';
  if (pct < -2)   return '#ef4444';
  if (pct < -0.5) return '#fca5a5';
  if (pct <  0)   return '#fee2e2';
  return 'var(--bg-2)';
}

const MONTHS = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const MONTHS_SHORT = ['J','F','M','A','M','J','J','A','S','O','N','D'];

const QUARTERS = [
  { label: 'Q1', months: [0, 1, 2] },
  { label: 'Q2', months: [3, 4, 5] },
  { label: 'Q3', months: [6, 7, 8] },
  { label: 'Q4', months: [9, 10, 11] },
];

function weekMonday(date: Date): string {
  const d   = new Date(date);
  const day = d.getDay();
  d.setDate(d.getDate() - ((day + 6) % 7));
  return d.toISOString().slice(0, 10);
}

function fmtDate(dateStr: string) {
  return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

interface QuarterCellProps {
  q: {
    label: string;
    weeks: { monday: string; days: { date: string; pct: number; dow: number; month: number; year: number }[] }[];
    up: number; down: number; avg: number; total: number;
  };
  setTooltip: (t: { date: string; pct: number; x: number; y: number } | null) => void;
}

function QuarterCell({ q, setTooltip }: QuarterCellProps) {
  return (
    <div style={{ minWidth: 0 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 5, marginBottom: 4, fontSize: 9 }}>
        <span style={{ fontWeight: 700, color: 'var(--text-2)' }}>{q.label}</span>
        {q.total > 0 ? (
          <>
            <span style={{ color: '#22c55e' }}>{q.up}↑</span>
            <span style={{ color: '#ef4444' }}>{q.down}↓</span>
            <span style={{ color: q.avg >= 0 ? '#22c55e' : '#ef4444' }}>
              {q.avg >= 0 ? '+' : ''}{q.avg.toFixed(2)}%
            </span>
          </>
        ) : (
          <span style={{ color: 'var(--text-5)' }}>no data yet</span>
        )}
      </div>

      {/* Calendar */}
      {q.weeks.length > 0 && (
        <div style={{ display: 'flex', gap: GAP + 2, alignItems: 'flex-start' }}>
          {/* M/W/F axis */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: GAP, paddingTop: 14, flexShrink: 0 }}>
            {['M', '', 'W', '', 'F'].map((l, i) => (
              <div key={i} style={{
                height: CELL, width: 8, fontSize: 7.5, color: 'var(--text-5)',
                lineHeight: `${CELL}px`, textAlign: 'right',
              }}>{l}</div>
            ))}
          </div>

          {/* Weeks */}
          <div style={{ overflowX: 'auto' }}>
            <div style={{ display: 'flex', gap: GAP }}>
              {q.weeks.map(({ monday, days }) => {
                const monDate   = new Date(monday);
                const showMonth = monDate.getDate() <= 7;
                return (
                  <div key={monday} style={{ display: 'flex', flexDirection: 'column', gap: GAP, flexShrink: 0 }}>
                    <div style={{ height: 13, fontSize: 8, color: 'var(--text-5)', whiteSpace: 'nowrap', lineHeight: '13px' }}>
                      {showMonth ? MONTHS[monDate.getMonth()] : ''}
                    </div>
                    {[1, 2, 3, 4, 5].map(dow => {
                      const entry = days.find(d => new Date(d.date).getDay() === dow);
                      return (
                        <div
                          key={dow}
                          style={{
                            width: CELL, height: CELL, borderRadius: 2, flexShrink: 0,
                            background: entry ? getColor(entry.pct) : 'var(--bg-3)',
                            opacity: entry ? 1 : 0.18,
                          }}
                          onMouseEnter={entry ? (e) => {
                            const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
                            setTooltip({ date: entry.date, pct: entry.pct, x: r.right + 4, y: r.top });
                          } : undefined}
                          onMouseLeave={() => setTooltip(null)}
                        />
                      );
                    })}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function DailyReturnHeatmap({ chartData, inlineMobile }: Props) {
  const [selectedYear, setSelectedYear] = useState<number | null>(null);
  const [tooltip, setTooltip] = useState<{
    date: string; pct: number; x: number; y: number;
  } | null>(null);

  const returns = useMemo(() => {
    const sorted = [...chartData].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    return sorted.slice(1).map((d, i) => {
      const dt = new Date(d.date);
      return {
        date:  d.date,
        pct:   ((d.close - sorted[i].close) / sorted[i].close) * 100,
        dow:   dt.getDay(),
        month: dt.getMonth(),
        year:  dt.getFullYear(),
      };
    });
  }, [chartData]);

  const years = useMemo(() => {
    const s = new Set(returns.map(r => r.year));
    return Array.from(s).sort((a, b) => b - a);
  }, [returns]);

  const year        = selectedYear ?? years[0] ?? new Date().getFullYear();
  const yearReturns = useMemo(() => returns.filter(r => r.year === year), [returns, year]);

  const quarterData = useMemo(() => QUARTERS.map(q => {
    const qr = yearReturns.filter(r => q.months.includes(r.month));

    const weekMap = new Map<string, typeof qr>();
    qr.forEach(r => {
      const key = weekMonday(new Date(r.date));
      if (!weekMap.has(key)) weekMap.set(key, []);
      weekMap.get(key)!.push(r);
    });

    const weeks = Array.from(weekMap.entries())
      .sort((a, b) => a[0].localeCompare(b[0]))
      .map(([monday, days]) => ({ monday, days }));

    const up    = qr.filter(r => r.pct > 0).length;
    const down  = qr.filter(r => r.pct < 0).length;
    const avg   = qr.length ? qr.reduce((s, r) => s + r.pct, 0) / qr.length : 0;
    const best  = qr.length ? qr.reduce((b, r) => r.pct > b.pct ? r : b) : null;
    const worst = qr.length ? qr.reduce((b, r) => r.pct < b.pct ? r : b) : null;

    return { ...q, weeks, up, down, avg, best, worst, total: qr.length };
  }), [yearReturns]);

  // Stats for the info panel
  const stats = useMemo(() => {
    if (!yearReturns.length) return null;

    const up   = yearReturns.filter(r => r.pct > 0).length;
    const down = yearReturns.filter(r => r.pct < 0).length;
    const winRate = Math.round((up / yearReturns.length) * 100);

    const best  = yearReturns.reduce((b, r) => r.pct > b.pct ? r : b);
    const worst = yearReturns.reduce((b, r) => r.pct < b.pct ? r : b);

    // Longest win/loss streak
    let curWin = 0, maxWin = 0, curLoss = 0, maxLoss = 0;
    const sorted = [...yearReturns].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
    sorted.forEach(r => {
      if (r.pct > 0) { curWin++; maxWin = Math.max(maxWin, curWin); curLoss = 0; }
      else            { curLoss++; maxLoss = Math.max(maxLoss, curLoss); curWin = 0; }
    });

    // Monthly avg returns
    const monthlyAvgs = Array.from({ length: 12 }, (_, m) => {
      const mr = yearReturns.filter(r => r.month === m);
      return { month: m, avg: mr.length ? mr.reduce((s, r) => s + r.pct, 0) / mr.length : null, count: mr.length };
    });

    const activeQs = quarterData.filter(q => q.total > 0);
    const bestQ  = activeQs.length ? activeQs.reduce((b, q) => q.avg > b.avg ? q : b) : null;
    const worstQ = activeQs.length ? activeQs.reduce((b, q) => q.avg < b.avg ? q : b) : null;

    return { up, down, winRate, best, worst, maxWin, maxLoss, monthlyAvgs, bestQ, worstQ };
  }, [yearReturns, quarterData]);

  if (!returns.length || !stats) return null;

  const maxMonthAbs = Math.max(...stats.monthlyAvgs.map(m => Math.abs(m.avg ?? 0)), 0.01);

  return (
    <div className={`card ${inlineMobile ? 'w-full' : ''}`}>

      {/* Title + year selector */}
      <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
        <span className="card-label" style={{ margin: 0 }}>Daily Return Heatmap</span>
        <div className="flex gap-1">
          {years.slice(0, 6).map(y => (
            <button
              key={y}
              onClick={() => setSelectedYear(y)}
              style={{
                fontSize: 9, padding: '1px 5px', border: '1px solid',
                borderRadius: 2, cursor: 'pointer',
                background:  y === year ? 'var(--accent)' : 'var(--bg-3)',
                borderColor: y === year ? 'var(--accent)' : 'var(--bg-1)',
                color:       y === year ? 'var(--text-0)' : 'var(--text-4)',
              }}
            >
              {y}
            </button>
          ))}
        </div>
      </div>

      {/* 3-column grid: [Q1][Q2][info] / [Q3][Q4][info cont.] */}
      <div style={{ display: 'grid', gridTemplateColumns: 'auto auto auto', gap: '10px 14px', alignItems: 'start' }}>

        {/* Q1 */}
        <QuarterCell q={quarterData[0]} setTooltip={setTooltip} />
        {/* Q2 */}
        <QuarterCell q={quarterData[1]} setTooltip={setTooltip} />

        {/* Info panel — spans both rows */}
        <div style={{
          gridRow: '1 / 3',
          borderLeft: '1px solid var(--bg-1)',
          paddingLeft: 8,
          display: 'flex',
          flexDirection: 'column',
          gap: 10,
          minWidth: 0,
          width: 'fit-content',
        }}>

          {/* Win rate */}
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-5)', marginBottom: 2 }}>Win Rate</div>
            <div style={{ fontSize: 22, fontWeight: 700, lineHeight: 1, color: stats.winRate >= 50 ? '#22c55e' : '#ef4444' }}>
              {stats.winRate}%
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-5)', marginTop: 2 }}>
              <span style={{ color: '#22c55e' }}>{stats.up}↑</span>
              {' · '}
              <span style={{ color: '#ef4444' }}>{stats.down}↓</span>
            </div>
          </div>

          <div style={{ height: 1, background: 'var(--bg-1)' }} />

          {/* Monthly avg — vertical bars in one row */}
          <div>
            <div style={{ fontSize: 9, color: 'var(--text-5)', marginBottom: 4 }}>Monthly Avg</div>
            <div style={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
              {stats.monthlyAvgs.map(m => {
                const pos    = (m.avg ?? 0) >= 0;
                const barH   = m.avg !== null ? Math.max((Math.abs(m.avg) / maxMonthAbs) * 28, 2) : 0;
                const HALF   = 28;
                return (
                  <div key={m.month} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2, width: 8, flexShrink: 0 }}>
                    {/* Positive zone */}
                    <div style={{ height: HALF, display: 'flex', alignItems: 'flex-end' }}>
                      {pos && m.avg !== null && (
                        <div style={{ width: '100%', width: 8, height: barH, borderRadius: '2px 2px 0 0', background: '#22c55e', opacity: 0.85 }} />
                      )}
                    </div>
                    {/* Zero line */}
                    <div style={{ width: 8, height: 1, background: 'var(--bg-1)' }} />
                    {/* Negative zone */}
                    <div style={{ height: HALF, display: 'flex', alignItems: 'flex-start' }}>
                      {!pos && m.avg !== null && (
                        <div style={{ width: '100%', width: 8, height: barH, borderRadius: '0 0 2px 2px', background: '#ef4444', opacity: 0.85 }} />
                      )}
                    </div>
                    <span style={{ fontSize: 7, color: 'var(--text-5)' }}>{MONTHS_SHORT[m.month]}</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div style={{ height: 1, background: 'var(--bg-1)' }} />

          {/* Best / Worst day — side by side */}
          <div style={{ display: 'flex', gap: 10 }}>
            <div>
              <div style={{ fontSize: 9, color: 'var(--text-5)' }}>Best Day</div>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#22c55e' }}>+{stats.best.pct.toFixed(2)}%</div>
              <div style={{ fontSize: 9, color: 'var(--text-5)' }}>{fmtDate(stats.best.date)}</div>
            </div>
            <div>
              <div style={{ fontSize: 9, color: 'var(--text-5)' }}>Worst Day</div>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#ef4444' }}>{stats.worst.pct.toFixed(2)}%</div>
              <div style={{ fontSize: 9, color: 'var(--text-5)' }}>{fmtDate(stats.worst.date)}</div>
            </div>
          </div>

          <div style={{ height: 1, background: 'var(--bg-1)' }} />

          {/* Streaks + best/worst quarter */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <div style={{ fontSize: 9, color: 'var(--text-5)' }}>
              Win streak <strong style={{ color: 'var(--text-3)' }}>{stats.maxWin}d</strong>
            </div>
            <div style={{ fontSize: 9, color: 'var(--text-5)' }}>
              Loss streak <strong style={{ color: 'var(--text-3)' }}>{stats.maxLoss}d</strong>
            </div>
            {stats.bestQ && (
              <div style={{ fontSize: 9, color: 'var(--text-5)' }}>
                Best Q <strong style={{ color: '#22c55e' }}>{stats.bestQ.label} {stats.bestQ.avg >= 0 ? '+' : ''}{stats.bestQ.avg.toFixed(2)}%</strong>
              </div>
            )}
            {stats.worstQ && (
              <div style={{ fontSize: 9, color: 'var(--text-5)' }}>
                Worst Q <strong style={{ color: '#ef4444' }}>{stats.worstQ.label} {stats.worstQ.avg.toFixed(2)}%</strong>
              </div>
            )}
          </div>

        </div>

        {/* Q3 */}
        <QuarterCell q={quarterData[2]} setTooltip={setTooltip} />
        {/* Q4 */}
        <QuarterCell q={quarterData[3]} setTooltip={setTooltip} />

      </div>

      {/* Legend */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 3, marginTop: 10 }}>
        <span style={{ fontSize: 8, color: 'var(--text-5)' }}>−5%</span>
        {[-4, -2, -0.5, 0.5, 2, 4].map((v, i) => (
          <div key={i} style={{ width: 8, height: 8, background: getColor(v), borderRadius: 1 }} />
        ))}
        <span style={{ fontSize: 8, color: 'var(--text-5)' }}>+5%</span>
      </div>

      {tooltip && (
        <div style={{
          position: 'fixed', left: tooltip.x, top: tooltip.y,
          background: 'var(--bg-2)', border: '1px solid var(--bg-1)',
          padding: '3px 7px', borderRadius: 3,
          fontSize: 11, color: 'var(--text-2)', zIndex: 9999,
          pointerEvents: 'none', whiteSpace: 'nowrap',
        }}>
          <span style={{ fontWeight: 600 }}>
            {new Date(tooltip.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
          </span>
          {' '}
          <span style={{ color: tooltip.pct >= 0 ? '#22c55e' : '#ef4444', fontWeight: 700 }}>
            {tooltip.pct >= 0 ? '+' : ''}{tooltip.pct.toFixed(2)}%
          </span>
        </div>
      )}
    </div>
  );
}
