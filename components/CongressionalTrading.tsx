'use client';

import React, { useState, useEffect } from 'react';
import { Landmark, TrendingUp, TrendingDown, Loader2, RefreshCw } from 'lucide-react';

interface CongressionalTrade {
  name: string;
  representative: string;
  amount: string;
  transactionDate: string;
  filingDate: string;
  type: string;
}

interface CongressionalTradingProps {
  symbol: string;
  inlineMobile?: boolean;
}

export default function CongressionalTrading({ symbol, inlineMobile }: CongressionalTradingProps) {
  const [trades, setTrades] = useState<CongressionalTrade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchTrades = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch(`/api/congressional-trading?symbol=${symbol}`);
      const data = await response.json();

      if (data.success) {
        setTrades(data.trades.slice(0, 20)); // Show latest 20
      } else {
        setError(data.error || 'Failed to fetch congressional trading');
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (symbol) {
      fetchTrades();
    }
  }, [symbol]);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const getTypeColor = (type: string) => {
    const typeLower = type.toLowerCase();
    if (typeLower.includes('purchase') || typeLower.includes('buy')) return 'var(--success)';
    if (typeLower.includes('sale') || typeLower.includes('sell')) return 'var(--danger)';
    return 'var(--text-4)';
  };

  const getTypeLabel = (type: string) => {
    const typeLower = type.toLowerCase();
    if (typeLower.includes('purchase') || typeLower.includes('buy')) return 'Buy';
    if (typeLower.includes('sale') || typeLower.includes('sell')) return 'Sell';
    return type;
  };

  const buyCount = trades.filter(t => getTypeLabel(t.type) === 'Buy').length;
  const sellCount = trades.filter(t => getTypeLabel(t.type) === 'Sell').length;

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Landmark className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Congressional Trading</span>
        </div>

        <button
          onClick={() => fetchTrades()}
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
          <div className="grid grid-cols-2 gap-3 mb-4">
            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Purchases</div>
              <div className="text-lg font-bold" style={{ color: 'var(--success)' }}>
                {buyCount}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>transactions</div>
            </div>

            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Sales</div>
              <div className="text-lg font-bold" style={{ color: 'var(--danger)' }}>
                {sellCount}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>transactions</div>
            </div>
          </div>

          {/* Trade List */}
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {trades.map((trade, index) => (
              <div
                key={index}
                className="p-3 border transition-all"
                style={{
                  background: 'var(--bg-2)',
                  borderColor: 'var(--bg-1)',
                  borderLeftWidth: '3px',
                  borderLeftColor: getTypeColor(trade.type),
                }}
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold text-sm truncate" style={{ color: 'var(--text-2)' }}>
                      {trade.representative}
                    </div>
                    <div className="text-xs" style={{ color: 'var(--text-5)' }}>
                      {formatDate(trade.transactionDate)}
                    </div>
                  </div>
                  <div
                    className="px-2 py-1 text-xs font-semibold flex items-center gap-1"
                    style={{
                      background: getTypeColor(trade.type) + '20',
                      color: getTypeColor(trade.type),
                    }}
                  >
                    {getTypeLabel(trade.type) === 'Buy' ? (
                      <TrendingUp className="w-3 h-3" />
                    ) : getTypeLabel(trade.type) === 'Sell' ? (
                      <TrendingDown className="w-3 h-3" />
                    ) : null}
                    {getTypeLabel(trade.type)}
                  </div>
                </div>

                <div className="text-xs">
                  <span style={{ color: 'var(--text-4)' }}>Amount: </span>
                  <span className="font-mono font-semibold" style={{ color: 'var(--text-2)' }}>
                    {trade.amount}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {trades.length === 0 && !loading && (
            <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
              <p className="text-sm">No congressional trades found</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
