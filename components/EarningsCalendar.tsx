'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, Loader2 } from 'lucide-react';

interface Earning {
  actual: number | null;
  estimate: number | null;
  period: string;
  quarter: number;
  surprise: number | null;
  surprisePercent: number | null;
  symbol: string;
  year: number;
}

interface EarningsCalendarProps {
  symbol: string;
  inlineMobile?: boolean;
}

export default function EarningsCalendar({ symbol, inlineMobile }: EarningsCalendarProps) {
  const [earnings, setEarnings] = useState<Earning[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchEarnings = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch(`/api/earnings-calendar?symbol=${symbol}`);
      const data = await response.json();

      if (data.success) {
        setEarnings(data.earnings.slice(0, 12)); // Show last 12 quarters
      } else {
        setError(data.error || 'Failed to fetch earnings calendar');
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (symbol) {
      fetchEarnings();
    }
  }, [symbol]);

  const beatEstimate = earnings.filter(e => e.actual !== null && e.estimate !== null && e.actual > e.estimate).length;
  const missedEstimate = earnings.filter(e => e.actual !== null && e.estimate !== null && e.actual < e.estimate).length;
  const avgSurprise = earnings.reduce((acc, e) => acc + (e.surprisePercent || 0), 0) / (earnings.length || 1);

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <div className="flex items-center mb-4">
        <span className="card-label">Earnings History</span>
      </div>

      {error && (
        <div className="mb-4 p-3 border-2" style={{
          background: 'var(--bg-2)',
          borderColor: 'var(--danger)',
          color: 'var(--danger)'
        }}>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-3 mb-4">
            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Beat</div>
              <div className="text-lg font-bold" style={{ color: 'var(--success)' }}>
                {beatEstimate}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>quarters</div>
            </div>

            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Missed</div>
              <div className="text-lg font-bold" style={{ color: 'var(--danger)' }}>
                {missedEstimate}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>quarters</div>
            </div>

            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Avg Surprise</div>
              <div className="text-lg font-bold" style={{ color: avgSurprise > 0 ? 'var(--success)' : avgSurprise < 0 ? 'var(--danger)' : 'var(--text-3)' }}>
                {avgSurprise > 0 ? '+' : ''}{avgSurprise.toFixed(1)}%
              </div>
            </div>
          </div>

          {/* Earnings List */}
          <div className="space-y-1 max-h-96 overflow-y-auto">
            {earnings.map((earning, index) => {
              const beat = earning.actual !== null && earning.estimate !== null && earning.actual > earning.estimate;
              const missed = earning.actual !== null && earning.estimate !== null && earning.actual < earning.estimate;
              const borderColor = beat ? 'var(--success)' : missed ? 'var(--danger)' : 'var(--text-4)';

              return (
                <div
                  key={index}
                  className="flex items-center gap-3 px-2 py-1.5 border-l-2"
                  style={{ background: 'var(--bg-2)', borderColor }}
                >
                  <span className="text-xs font-semibold w-14 flex-shrink-0" style={{ color: 'var(--text-2)' }}>
                    Q{earning.quarter} {String(earning.year).slice(2)}
                  </span>
                  <div className="flex items-center gap-3 flex-1 text-xs font-mono">
                    {earning.actual !== null && (
                      <span style={{ color: 'var(--text-2)' }}>
                        <span style={{ color: 'var(--text-4)' }}>Act </span>${earning.actual.toFixed(2)}
                      </span>
                    )}
                    {earning.estimate !== null && (
                      <span style={{ color: 'var(--text-3)' }}>
                        <span style={{ color: 'var(--text-4)' }}>Est </span>${earning.estimate.toFixed(2)}
                      </span>
                    )}
                  </div>
                  {earning.surprisePercent !== null && (
                    <span className="text-xs font-semibold flex-shrink-0" style={{ color: beat ? 'var(--success)' : missed ? 'var(--danger)' : 'var(--text-4)' }}>
                      {earning.surprisePercent > 0 ? '+' : ''}{earning.surprisePercent.toFixed(1)}%
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          {earnings.length === 0 && !loading && (
            <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
              <p className="text-sm">No earnings data found</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
