'use client';

import React, { useState, useEffect } from 'react';
import { Users, TrendingUp, TrendingDown, Loader2, RefreshCw } from 'lucide-react';

interface InsiderTransaction {
  name: string;
  share: number;
  change: number;
  filingDate: string;
  transactionDate: string;
  transactionCode: string;
  transactionPrice: number;
}

interface InsiderTransactionsProps {
  symbol: string;
  inlineMobile?: boolean;
}

export default function InsiderTransactions({ symbol, inlineMobile }: InsiderTransactionsProps) {
  const [transactions, setTransactions] = useState<InsiderTransaction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchTransactions = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch(`/api/insider-transactions?symbol=${symbol}`);
      const data = await response.json();

      if (data.success) {
        setTransactions(data.transactions.slice(0, 20)); // Show latest 20
      } else {
        setError(data.error || 'Failed to fetch insider transactions');
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (symbol) {
      fetchTransactions();
    }
  }, [symbol]);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  };

  const getTransactionType = (code: string) => {
    const buyTypes = ['P', 'M'];
    const sellTypes = ['S'];
    if (buyTypes.includes(code)) return 'Buy';
    if (sellTypes.includes(code)) return 'Sell';
    return code;
  };

  const getTransactionColor = (code: string) => {
    const type = getTransactionType(code);
    if (type === 'Buy') return 'var(--success)';
    if (type === 'Sell') return 'var(--danger)';
    return 'var(--text-4)';
  };

  const formatNumber = (num: number) => {
    if (num >= 1000000) return (num / 1000000).toFixed(1) + 'M';
    if (num >= 1000) return (num / 1000).toFixed(1) + 'K';
    return num.toLocaleString();
  };

  const netBuying = transactions.reduce((acc, t) => {
    const type = getTransactionType(t.transactionCode);
    if (type === 'Buy') return acc + (t.change || 0);
    if (type === 'Sell') return acc - (t.change || 0);
    return acc;
  }, 0);

  const buyCount = transactions.filter(t => getTransactionType(t.transactionCode) === 'Buy').length;
  const sellCount = transactions.filter(t => getTransactionType(t.transactionCode) === 'Sell').length;

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Insider Trading</span>
        </div>

        <button
          onClick={() => fetchTransactions()}
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
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Buys</div>
              <div className="text-lg font-bold" style={{ color: 'var(--success)' }}>
                {buyCount}
              </div>
              <div className="text-[9px]" style={{ color: 'var(--text-5)' }}>transactions</div>
            </div>

            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Sells</div>
              <div className="text-lg font-bold" style={{ color: 'var(--danger)' }}>
                {sellCount}
              </div>
              <div className="text-[9px]" style={{ color: 'var(--text-5)' }}>transactions</div>
            </div>

            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Net</div>
              <div className="text-lg font-bold" style={{ color: netBuying > 0 ? 'var(--success)' : netBuying < 0 ? 'var(--danger)' : 'var(--text-3)' }}>
                {netBuying > 0 ? '+' : ''}{formatNumber(netBuying)}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>shares</div>
            </div>
          </div>

          {/* Transaction List */}
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {transactions.map((transaction, index) => (
              <div
                key={index}
                className="p-3 border transition-all"
                style={{
                  background: 'var(--bg-2)',
                  borderColor: 'var(--bg-1)',
                  borderLeftWidth: '3px',
                  borderLeftColor: getTransactionColor(transaction.transactionCode),
                }}
              >
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex-1 min-w-0">
                    <div className="font-semibold text-sm truncate" style={{ color: 'var(--text-2)' }}>
                      {transaction.name}
                    </div>
                    <div className="text-xs" style={{ color: 'var(--text-5)' }}>
                      {formatDate(transaction.transactionDate)}
                    </div>
                  </div>
                  <div
                    className="px-2 py-1 text-xs font-semibold flex items-center gap-1"
                    style={{
                      background: getTransactionColor(transaction.transactionCode) + '20',
                      color: getTransactionColor(transaction.transactionCode),
                    }}
                  >
                    {getTransactionType(transaction.transactionCode) === 'Buy' ? (
                      <TrendingUp className="w-3 h-3" />
                    ) : getTransactionType(transaction.transactionCode) === 'Sell' ? (
                      <TrendingDown className="w-3 h-3" />
                    ) : null}
                    {getTransactionType(transaction.transactionCode)}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span style={{ color: 'var(--text-4)' }}>Shares: </span>
                    <span className="font-mono font-semibold" style={{ color: 'var(--text-2)' }}>
                      {formatNumber(Math.abs(transaction.change))}
                    </span>
                  </div>
                  {transaction.transactionPrice > 0 && (
                    <div>
                      <span style={{ color: 'var(--text-4)' }}>Price: </span>
                      <span className="font-mono font-semibold" style={{ color: 'var(--text-2)' }}>
                        ${transaction.transactionPrice.toFixed(2)}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {transactions.length === 0 && !loading && (
            <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
              <p className="text-sm">No insider transactions found</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
