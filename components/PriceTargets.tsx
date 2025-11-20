'use client';

import React, { useState, useEffect } from 'react';
import { Target, TrendingUp, TrendingDown, Loader2, RefreshCw } from 'lucide-react';

interface PriceTarget {
  targetHigh: number;
  targetLow: number;
  targetMean: number;
  targetMedian: number;
  lastUpdated: string;
}

interface PriceTargetsProps {
  symbol: string;
  currentPrice: number;
  inlineMobile?: boolean;
}

export default function PriceTargets({ symbol, currentPrice, inlineMobile }: PriceTargetsProps) {
  const [priceTarget, setPriceTarget] = useState<PriceTarget | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchPriceTarget = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch(`/api/price-target?symbol=${symbol}`);
      const data = await response.json();

      if (data.success && data.priceTarget) {
        setPriceTarget(data.priceTarget);
      } else {
        setError(data.error || 'Failed to fetch price targets');
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (symbol) {
      fetchPriceTarget();
    }
  }, [symbol]);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const upside = priceTarget ? ((priceTarget.targetMean - currentPrice) / currentPrice) * 100 : 0;
  const highUpside = priceTarget ? ((priceTarget.targetHigh - currentPrice) / currentPrice) * 100 : 0;
  const lowUpside = priceTarget ? ((priceTarget.targetLow - currentPrice) / currentPrice) * 100 : 0;

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Price Targets</span>
        </div>

        <button
          onClick={() => fetchPriceTarget()}
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
      ) : priceTarget ? (
        <>
          {/* Current Price vs Target */}
          <div className="mb-4 p-4 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: upside > 0 ? 'var(--success)' : 'var(--danger)',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>
              Analyst Consensus
            </div>
            <div className="flex items-baseline gap-2">
              <div className="text-2xl font-bold" style={{ color: 'var(--text-2)' }}>
                ${priceTarget.targetMean.toFixed(2)}
              </div>
              <div
                className="flex items-center gap-1 text-sm font-semibold"
                style={{ color: upside > 0 ? 'var(--success)' : 'var(--danger)' }}
              >
                {upside > 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                {upside > 0 ? '+' : ''}{upside.toFixed(1)}%
              </div>
            </div>
            <div className="text-xs mt-1" style={{ color: 'var(--text-5)' }}>
              vs Current ${currentPrice.toFixed(2)}
            </div>
          </div>

          {/* Price Range */}
          <div className="mb-4">
            <div className="text-xs font-semibold mb-3" style={{ color: 'var(--text-4)' }}>
              Analyst Price Range
            </div>

            {/* Visual Range */}
            <div className="relative h-8 mb-4" style={{ background: 'var(--bg-3)' }}>
              <div
                className="absolute h-full"
                style={{
                  left: `${Math.max(0, Math.min(100, ((priceTarget.targetLow - priceTarget.targetLow) / (priceTarget.targetHigh - priceTarget.targetLow)) * 100))}%`,
                  right: `${Math.max(0, 100 - Math.min(100, ((priceTarget.targetHigh - priceTarget.targetLow) / (priceTarget.targetHigh - priceTarget.targetLow)) * 100))}%`,
                  background: 'linear-gradient(to right, var(--danger), var(--success))',
                  opacity: 0.3
                }}
              />
              <div
                className="absolute w-1 h-full"
                style={{
                  left: `${((currentPrice - priceTarget.targetLow) / (priceTarget.targetHigh - priceTarget.targetLow)) * 100}%`,
                  background: 'var(--accent)',
                }}
              />
            </div>

            {/* Targets Grid */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 border" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
                <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Low Target</div>
                <div className="text-lg font-bold font-mono" style={{ color: 'var(--text-2)' }}>
                  ${priceTarget.targetLow.toFixed(2)}
                </div>
                <div className="text-xs" style={{ color: lowUpside > 0 ? 'var(--success)' : 'var(--danger)' }}>
                  {lowUpside > 0 ? '+' : ''}{lowUpside.toFixed(1)}%
                </div>
              </div>

              <div className="p-3 border" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
                <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>High Target</div>
                <div className="text-lg font-bold font-mono" style={{ color: 'var(--text-2)' }}>
                  ${priceTarget.targetHigh.toFixed(2)}
                </div>
                <div className="text-xs" style={{ color: 'var(--success)' }}>
                  +{highUpside.toFixed(1)}%
                </div>
              </div>

              <div className="p-3 border" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
                <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Mean Target</div>
                <div className="text-lg font-bold font-mono" style={{ color: 'var(--text-2)' }}>
                  ${priceTarget.targetMean.toFixed(2)}
                </div>
                <div className="text-xs" style={{ color: upside > 0 ? 'var(--success)' : 'var(--danger)' }}>
                  {upside > 0 ? '+' : ''}{upside.toFixed(1)}%
                </div>
              </div>

              <div className="p-3 border" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
                <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Median Target</div>
                <div className="text-lg font-bold font-mono" style={{ color: 'var(--text-2)' }}>
                  ${priceTarget.targetMedian.toFixed(2)}
                </div>
                <div className="text-xs" style={{
                  color: ((priceTarget.targetMedian - currentPrice) / currentPrice) * 100 > 0 ? 'var(--success)' : 'var(--danger)'
                }}>
                  {((priceTarget.targetMedian - currentPrice) / currentPrice) * 100 > 0 ? '+' : ''}
                  {(((priceTarget.targetMedian - currentPrice) / currentPrice) * 100).toFixed(1)}%
                </div>
              </div>
            </div>
          </div>

          {/* Last Updated */}
          {priceTarget.lastUpdated && (
            <div className="text-xs text-center" style={{ color: 'var(--text-5)' }}>
              Last updated: {formatDate(priceTarget.lastUpdated)}
            </div>
          )}
        </>
      ) : (
        <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
          <p className="text-sm">No price targets available</p>
        </div>
      )}
    </div>
  );
}
