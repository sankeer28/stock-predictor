'use client';

import React, { useEffect, useState } from 'react';

interface EarningsEntry {
  actual:          number | null;
  estimate:        number | null;
  period:          string;
  quarter:         number;
  surprise:        number | null;
  surprisePercent: number | null;
  year:            number;
}

interface Props {
  symbol:        string;
  inlineMobile?: boolean;
}

export default function EarningsHistory({ symbol, inlineMobile }: Props) {
  const [earnings, setEarnings] = useState<EarningsEntry[]>([]);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState('');

  useEffect(() => {
    if (!symbol) return;
    setLoading(true);
    setError('');
    fetch(`/api/earnings-calendar?symbol=${encodeURIComponent(symbol)}`)
      .then(r => r.json())
      .then(data => {
        if (data.success && Array.isArray(data.earnings)) {
          const filtered = (data.earnings as EarningsEntry[])
            .filter(e => e.actual != null && e.estimate != null)
            .sort((a, b) => new Date(b.period).getTime() - new Date(a.period).getTime())
            .slice(0, 8);
          setEarnings(filtered);
        } else {
          setError(data.error || 'No data');
        }
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false));
  }, [symbol]);

  if (!loading && (error || !earnings.length)) return null;

  const maxAbs = Math.max(
    ...earnings.map(e => Math.max(Math.abs(e.actual ?? 0), Math.abs(e.estimate ?? 0))),
    0.01
  );

  const beatCount = earnings.filter(e => (e.actual ?? 0) >= (e.estimate ?? 0)).length;

  return (
    <div className={`card ${inlineMobile ? 'w-full' : ''}`}>
      <span className="card-label">Earnings Surprise History</span>

      {loading ? (
        <div className="flex items-center justify-center py-6">
          <div className="w-5 h-5 border-2 rounded-full animate-spin" style={{ borderColor: 'var(--accent)', borderTopColor: 'transparent' }} />
        </div>
      ) : (
        <>
          {/* Beat rate summary */}
          <div className="flex items-center gap-3 mb-3" style={{ fontSize: 11, color: 'var(--text-4)' }}>
            <span>Beat rate: <strong style={{ color: beatCount / earnings.length >= 0.5 ? '#22c55e' : '#ef4444' }}>
              {beatCount}/{earnings.length}
            </strong></span>
            <span style={{ color: 'var(--text-5)' }}>Last {earnings.length}Q</span>
          </div>

          <div className="space-y-2.5">
            {earnings.map((e, i) => {
              const beat     = (e.actual ?? 0) >= (e.estimate ?? 0);
              const actualW  = (Math.abs(e.actual  ?? 0) / maxAbs) * 100;
              const estimW   = (Math.abs(e.estimate ?? 0) / maxAbs) * 100;
              const color    = beat ? '#22c55e' : '#ef4444';

              return (
                <div key={i}>
                  {/* Header row */}
                  <div className="flex items-center justify-between mb-1">
                    <span style={{ fontSize: 10, color: 'var(--text-4)', fontWeight: 600 }}>
                      Q{e.quarter} {e.year}
                    </span>
                    <div className="flex items-center gap-2">
                      <span style={{ fontSize: 10, color: 'var(--text-5)' }}>
                        Est ${(e.estimate ?? 0).toFixed(2)}
                      </span>
                      <span style={{ fontSize: 10, fontWeight: 700, color }}>
                        Act ${(e.actual ?? 0).toFixed(2)}
                      </span>
                      {e.surprisePercent != null && (
                        <span style={{
                          fontSize: 9, fontWeight: 700, color,
                          background: beat ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                          padding: '1px 4px', borderRadius: 2,
                        }}>
                          {beat ? '+' : ''}{e.surprisePercent.toFixed(1)}%
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Bar */}
                  <div style={{ position: 'relative', height: 6, background: 'var(--bg-3)', borderRadius: 2 }}>
                    {/* Estimate marker (background bar) */}
                    <div style={{
                      position: 'absolute', left: 0, top: 0, height: '100%',
                      width: `${estimW}%`, background: 'var(--bg-1)', borderRadius: 2,
                    }} />
                    {/* Actual bar */}
                    <div style={{
                      position: 'absolute', left: 0, top: 0, height: '100%',
                      width: `${actualW}%`, background: color, opacity: 0.85, borderRadius: 2,
                      transition: 'width 0.4s ease',
                    }} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Legend */}
          <div className="flex items-center gap-3 mt-3 pt-2" style={{ borderTop: '1px solid var(--bg-1)', fontSize: 9, color: 'var(--text-5)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
              <span style={{ display: 'inline-block', width: 8, height: 8, background: '#22c55e', borderRadius: 1 }} /> Beat
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
              <span style={{ display: 'inline-block', width: 8, height: 8, background: '#ef4444', borderRadius: 1 }} /> Miss
            </span>
            <span style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
              <span style={{ display: 'inline-block', width: 8, height: 8, background: 'var(--bg-1)', borderRadius: 1 }} /> Estimate
            </span>
          </div>
        </>
      )}
    </div>
  );
}
