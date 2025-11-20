'use client';

import React, { useState, useEffect } from 'react';
import { Calendar, TrendingUp, Loader2, RefreshCw } from 'lucide-react';

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
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Calendar className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Earnings History</span>
        </div>

        <button
          onClick={() => fetchEarnings()}
          disabled={loading}
          className="p-2 transition-all border disabled:opacity-50"
          style={{
            background: 'var(--bg-3)',
            borderColor: 'var(--bg-1)',
            color: 'var(--text-3)',
          }}
          title="Refresh data"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
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
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {earnings.map((earning, index) => {
              const beat = earning.actual !== null && earning.estimate !== null && earning.actual > earning.estimate;
              const missed = earning.actual !== null && earning.estimate !== null && earning.actual < earning.estimate;

              return (
                <div
                  key={index}
                  className="p-3 border transition-all"
                  style={{
                    background: 'var(--bg-2)',
                    borderColor: 'var(--bg-1)',
                    borderLeftWidth: '3px',
                    borderLeftColor: beat ? 'var(--success)' : missed ? 'var(--danger)' : 'var(--text-4)',
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="font-semibold text-sm" style={{ color: 'var(--text-2)' }}>
                      Q{earning.quarter} {earning.year}
                    </div>
                    {earning.surprisePercent !== null && (
                      <div
                        className="px-2 py-1 text-xs font-semibold"
                        style={{
                          background: beat ? 'rgba(34, 197, 94, 0.1)' : missed ? 'rgba(239, 68, 68, 0.1)' : 'var(--bg-3)',
                          color: beat ? 'var(--success)' : missed ? 'var(--danger)' : 'var(--text-4)',
                        }}
                      >
                        {earning.surprisePercent > 0 ? '+' : ''}{earning.surprisePercent.toFixed(2)}%
                      </div>
                    )}
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    {earning.actual !== null && (
                      <div>
                        <div style={{ color: 'var(--text-4)' }}>Actual EPS</div>
                        <div className="font-mono font-bold text-sm" style={{ color: 'var(--text-2)' }}>
                          ${earning.actual.toFixed(2)}
                        </div>
                      </div>
                    )}
                    {earning.estimate !== null && (
                      <div>
                        <div style={{ color: 'var(--text-4)' }}>Estimate</div>
                        <div className="font-mono font-bold text-sm" style={{ color: 'var(--text-3)' }}>
                          ${earning.estimate.toFixed(2)}
                        </div>
                      </div>
                    )}
                  </div>
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
