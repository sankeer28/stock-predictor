'use client';

import React, { useState, useEffect } from 'react';
import { BarChart3, Loader2, RefreshCw, Target } from 'lucide-react';

interface Recommendation {
  buy: number;
  hold: number;
  period: string;
  sell: number;
  strongBuy: number;
  strongSell: number;
  symbol: string;
}

interface FinvizTarget {
  date: string;
  category: string;
  analyst: string;
  rating: string;
  target: string;
}

interface AnalystRecommendationsProps {
  symbol: string;
  inlineMobile?: boolean;
  finvizTargets?: FinvizTarget[] | null;
}

export default function AnalystRecommendations({ symbol, inlineMobile, finvizTargets }: AnalystRecommendationsProps) {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchRecommendations = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch(`/api/recommendations?symbol=${symbol}`);
      const data = await response.json();

      if (data.success) {
        setRecommendations(data.recommendations.slice(0, 6)); // Show last 6 months
      } else {
        setError(data.error || 'Failed to fetch recommendations');
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (symbol) {
      fetchRecommendations();
    }
  }, [symbol]);

  const latestRec = recommendations[0];
  const totalAnalysts = latestRec ? latestRec.strongBuy + latestRec.buy + latestRec.hold + latestRec.sell + latestRec.strongSell : 0;

  const getConsensus = (rec: Recommendation) => {
    const bullish = rec.strongBuy + rec.buy;
    const bearish = rec.sell + rec.strongSell;
    if (bullish > bearish + rec.hold) return 'Buy';
    if (bearish > bullish + rec.hold) return 'Sell';
    return 'Hold';
  };

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
  };

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Analyst Ratings</span>
        </div>

        <button
          onClick={() => fetchRecommendations()}
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
          {latestRec && (
            <>
              {/* Current Consensus — compact inline row */}
              <div className="flex items-center gap-2 mb-3 px-2 py-1.5 border-l-2" style={{ background: 'var(--bg-2)', borderColor: 'var(--accent)' }}>
                <span className="text-[10px] font-semibold" style={{ color: 'var(--text-4)' }}>Consensus</span>
                <span className="text-sm font-bold" style={{ color: 'var(--accent)' }}>{getConsensus(latestRec)}</span>
                <span className="text-[10px] font-mono ml-auto" style={{ color: 'var(--text-5)' }}>{formatDate(latestRec.period)} · {totalAnalysts} analysts</span>
              </div>

              {/* Compact 3-stat grid */}
              <div className="grid grid-cols-3 gap-2 mb-3">
                {([
                  { label: 'Buys', value: latestRec.strongBuy + latestRec.buy, sub: `${latestRec.strongBuy}SB+${latestRec.buy}B`, color: 'var(--success)' },
                  { label: 'Hold', value: latestRec.hold, sub: `${totalAnalysts} total`, color: 'var(--text-3)' },
                  { label: 'Sells', value: latestRec.sell + latestRec.strongSell, sub: `${latestRec.sell}S+${latestRec.strongSell}SS`, color: 'var(--danger)' },
                ] as const).map(({ label, value, sub, color }) => (
                  <div key={label} className="p-2 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
                    <div className="text-[10px] mb-0.5" style={{ color: 'var(--text-4)' }}>{label}</div>
                    <div className="text-base font-bold leading-none mb-0.5" style={{ color }}>{value}</div>
                    <div className="text-[9px] font-mono" style={{ color: 'var(--text-5)' }}>{sub}</div>
                  </div>
                ))}
              </div>
            </>
          )}

          {/* Historical Trend */}
          <div className="space-y-1 max-h-48 overflow-y-auto">
            <div className="text-[10px] font-semibold mb-1.5" style={{ color: 'var(--text-4)' }}>Historical Trend</div>
            {recommendations.map((rec, index) => {
              const consensus = getConsensus(rec);
              const consensusColor = consensus === 'Buy' ? 'var(--success)' : consensus === 'Sell' ? 'var(--danger)' : 'var(--text-3)';
              return (
                <div key={index} className="px-2 py-1.5 border-l-2" style={{ background: 'var(--bg-2)', borderColor: consensusColor }}>
                  <div className="flex items-center gap-2 mb-0.5">
                    <span className="text-xs font-semibold" style={{ color: 'var(--text-2)' }}>{formatDate(rec.period)}</span>
                    <span className="text-[10px] font-semibold ml-auto" style={{ color: consensusColor }}>{consensus}</span>
                  </div>
                  <div className="text-[10px] font-mono" style={{ color: 'var(--text-4)' }}>
                    <span style={{ color: 'var(--success)' }}>Buy {rec.strongBuy + rec.buy}</span>
                    <span className="mx-1" style={{ color: 'var(--text-5)' }}>·</span>
                    <span>Hold {rec.hold}</span>
                    <span className="mx-1" style={{ color: 'var(--text-5)' }}>·</span>
                    <span style={{ color: 'var(--danger)' }}>Sell {rec.sell + rec.strongSell}</span>
                  </div>
                </div>
              );
            })}
          </div>

          {recommendations.length === 0 && !loading && (
            <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
              <p className="text-sm">No analyst recommendations found</p>
            </div>
          )}
        </>
      )}

      {/* Finviz Price Targets */}
      {finvizTargets && finvizTargets.length > 0 && (
        <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--bg-1)' }}>
          <div className="flex items-center gap-2 mb-3">
            <Target className="w-4 h-4" style={{ color: 'var(--accent)' }} />
            <div className="text-xs font-semibold" style={{ color: 'var(--text-4)' }}>Price Targets</div>
          </div>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {finvizTargets.map((t, i) => (
              <div key={i} className="p-2 border text-xs" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold" style={{ color: 'var(--text-2)' }}>{t.analyst}</span>
                  {t.target && t.target !== '-' && (
                    <span className="font-mono font-bold" style={{ color: 'var(--success)' }}>{t.target}</span>
                  )}
                </div>
                <div className="flex items-center justify-between" style={{ color: 'var(--text-4)' }}>
                  <span>{t.category} · {t.rating}</span>
                  <span>{t.date}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
