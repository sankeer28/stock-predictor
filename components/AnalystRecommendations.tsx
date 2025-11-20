'use client';

import React, { useState, useEffect } from 'react';
import { BarChart3, Loader2, RefreshCw } from 'lucide-react';

interface Recommendation {
  buy: number;
  hold: number;
  period: string;
  sell: number;
  strongBuy: number;
  strongSell: number;
  symbol: string;
}

interface AnalystRecommendationsProps {
  symbol: string;
  inlineMobile?: boolean;
}

export default function AnalystRecommendations({ symbol, inlineMobile }: AnalystRecommendationsProps) {
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
              {/* Current Consensus */}
              <div className="mb-4 p-4 border-2" style={{
                background: 'var(--bg-2)',
                borderColor: 'var(--accent)',
                borderLeftWidth: '3px'
              }}>
                <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>
                  Current Consensus ({formatDate(latestRec.period)})
                </div>
                <div className="text-2xl font-bold mb-1" style={{ color: 'var(--accent)' }}>
                  {getConsensus(latestRec)}
                </div>
                <div className="text-sm" style={{ color: 'var(--text-4)' }}>
                  {totalAnalysts} analysts
                </div>
              </div>

              {/* Breakdown */}
              <div className="mb-4">
                <div className="text-xs font-semibold mb-2" style={{ color: 'var(--text-4)' }}>
                  Latest Breakdown
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: 'var(--text-3)' }}>Strong Buy</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2 bg-gray-200" style={{ background: 'var(--bg-3)' }}>
                        <div
                          className="h-full"
                          style={{
                            width: `${(latestRec.strongBuy / totalAnalysts) * 100}%`,
                            background: 'var(--success)'
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono w-8 text-right" style={{ color: 'var(--text-2)' }}>
                        {latestRec.strongBuy}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: 'var(--text-3)' }}>Buy</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2" style={{ background: 'var(--bg-3)' }}>
                        <div
                          className="h-full"
                          style={{
                            width: `${(latestRec.buy / totalAnalysts) * 100}%`,
                            background: 'rgba(34, 197, 94, 0.6)'
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono w-8 text-right" style={{ color: 'var(--text-2)' }}>
                        {latestRec.buy}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: 'var(--text-3)' }}>Hold</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2" style={{ background: 'var(--bg-3)' }}>
                        <div
                          className="h-full"
                          style={{
                            width: `${(latestRec.hold / totalAnalysts) * 100}%`,
                            background: 'var(--text-4)'
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono w-8 text-right" style={{ color: 'var(--text-2)' }}>
                        {latestRec.hold}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: 'var(--text-3)' }}>Sell</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2" style={{ background: 'var(--bg-3)' }}>
                        <div
                          className="h-full"
                          style={{
                            width: `${(latestRec.sell / totalAnalysts) * 100}%`,
                            background: 'rgba(239, 68, 68, 0.6)'
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono w-8 text-right" style={{ color: 'var(--text-2)' }}>
                        {latestRec.sell}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center justify-between">
                    <span className="text-xs" style={{ color: 'var(--text-3)' }}>Strong Sell</span>
                    <div className="flex items-center gap-2">
                      <div className="w-32 h-2" style={{ background: 'var(--bg-3)' }}>
                        <div
                          className="h-full"
                          style={{
                            width: `${(latestRec.strongSell / totalAnalysts) * 100}%`,
                            background: 'var(--danger)'
                          }}
                        />
                      </div>
                      <span className="text-xs font-mono w-8 text-right" style={{ color: 'var(--text-2)' }}>
                        {latestRec.strongSell}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* Historical Trend */}
          <div className="space-y-2 max-h-64 overflow-y-auto">
            <div className="text-xs font-semibold mb-2" style={{ color: 'var(--text-4)' }}>
              Historical Trend
            </div>
            {recommendations.map((rec, index) => (
              <div
                key={index}
                className="p-3 border"
                style={{
                  background: 'var(--bg-2)',
                  borderColor: 'var(--bg-1)',
                }}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="text-sm font-semibold" style={{ color: 'var(--text-2)' }}>
                    {formatDate(rec.period)}
                  </div>
                  <div className="text-xs font-semibold px-2 py-1" style={{
                    background: 'var(--bg-3)',
                    color: 'var(--text-3)'
                  }}>
                    {getConsensus(rec)}
                  </div>
                </div>
                <div className="text-xs" style={{ color: 'var(--text-4)' }}>
                  Buy: {rec.strongBuy + rec.buy} • Hold: {rec.hold} • Sell: {rec.sell + rec.strongSell}
                </div>
              </div>
            ))}
          </div>

          {recommendations.length === 0 && !loading && (
            <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
              <p className="text-sm">No analyst recommendations found</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
