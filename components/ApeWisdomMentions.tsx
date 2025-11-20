'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, MessageSquare, ThumbsUp, Loader2, RefreshCw, ExternalLink, Minus } from 'lucide-react';

interface ApeWisdomStock {
  rank: number;
  ticker: string;
  name: string;
  mentions: number;
  upvotes: number;
  rank_24h_ago: number;
  mentions_24h_ago: number;
  rankChange: number;
  mentionsChange: number;
}

interface ApeWisdomMentionsProps {
  onTickerClick?: (ticker: string) => void;
  inlineMobile?: boolean;
}

export default function ApeWisdomMentions({ onTickerClick, inlineMobile }: ApeWisdomMentionsProps) {
  const [stocks, setStocks] = useState<ApeWisdomStock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [filter, setFilter] = useState<'all-stocks' | 'all-crypto'>('all-stocks');

  const fetchApeWisdomData = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch(`/api/apewisdom?filter=${filter}`);
      const data = await response.json();

      if (data.success) {
        setStocks(data.stocks);
        setLastUpdated(new Date());
      } else {
        setError(data.error || 'Failed to fetch ApeWisdom data');
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchApeWisdomData();
  }, [filter]);

  const getRankChangeIcon = (change: number) => {
    if (change > 0) return <TrendingUp className="w-3 h-3" style={{ color: 'var(--success)' }} />;
    if (change < 0) return <TrendingDown className="w-3 h-3" style={{ color: 'var(--danger)' }} />;
    return <Minus className="w-3 h-3" style={{ color: 'var(--text-5)' }} />;
  };

  const getRankChangeColor = (change: number) => {
    if (change > 0) return 'var(--success)';
    if (change < 0) return 'var(--danger)';
    return 'var(--text-5)';
  };

  const formatNumber = (num: number) => {
    if (num >= 1000) return (num / 1000).toFixed(1) + 'k';
    return num.toString();
  };

  const formatTimestamp = (date: Date | null) => {
    if (!date) return 'Never';
    const now = new Date();
    const diff = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diff < 60) return 'Just now';
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    return date.toLocaleTimeString();
  };

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="card-label">Reddit Stock Mentions</span>
          <a
            href="https://apewisdom.io"
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs flex items-center gap-1 transition-colors"
            style={{ color: 'var(--text-5)' }}
          >
            <ExternalLink className="w-3 h-3" />
          </a>
        </div>

        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-xs" style={{ color: 'var(--text-5)' }}>
              {formatTimestamp(lastUpdated)}
            </span>
          )}
          <button
            onClick={() => fetchApeWisdomData()}
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
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 mb-4">
        <button
          onClick={() => setFilter('all-stocks')}
          className="flex-1 px-3 py-2 text-xs font-semibold border transition-all"
          style={{
            background: filter === 'all-stocks' ? 'var(--accent)' : 'var(--bg-3)',
            borderColor: filter === 'all-stocks' ? 'var(--accent)' : 'var(--bg-1)',
            color: filter === 'all-stocks' ? 'var(--text-0)' : 'var(--text-3)',
          }}
        >
          Stocks
        </button>
        <button
          onClick={() => setFilter('all-crypto')}
          className="flex-1 px-3 py-2 text-xs font-semibold border transition-all"
          style={{
            background: filter === 'all-crypto' ? 'var(--accent)' : 'var(--bg-3)',
            borderColor: filter === 'all-crypto' ? 'var(--accent)' : 'var(--bg-1)',
            color: filter === 'all-crypto' ? 'var(--text-0)' : 'var(--text-3)',
          }}
        >
          Crypto
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

      {loading && stocks.length === 0 ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-3 mb-4">
            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Top Ticker</div>
              <div className="text-lg font-bold font-mono" style={{ color: 'var(--text-2)' }}>
                {stocks[0]?.ticker || '-'}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>
                Rank #{stocks[0]?.rank || '-'}
              </div>
            </div>

            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Total Mentions</div>
              <div className="text-lg font-bold" style={{ color: 'var(--purple-2)' }}>
                {stocks.length > 0 ? formatNumber(stocks.reduce((sum, s) => sum + s.mentions, 0)) : 0}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>across top 20</div>
            </div>

            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Hot Stock</div>
              <div className="text-lg font-bold" style={{ color: 'var(--success)' }}>
                {stocks.find(s => s.rankChange > 0)?.ticker || '-'}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>
                {stocks.find(s => s.rankChange > 0)?.rankChange ? `↑${stocks.find(s => s.rankChange > 0)?.rankChange}` : 'positions'}
              </div>
            </div>
          </div>

          {/* Stock List */}
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {stocks.map((stock) => (
              <button
                key={stock.ticker}
                onClick={() => onTickerClick?.(stock.ticker)}
                className="w-full text-left p-3 border transition-all hover:opacity-80"
                style={{
                  background: 'var(--bg-2)',
                  borderColor: 'var(--bg-1)',
                  borderLeftWidth: '3px',
                  borderLeftColor: stock.rankChange > 0 ? 'var(--success)' : stock.rankChange < 0 ? 'var(--danger)' : 'var(--text-5)',
                }}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-3 flex-1">
                    {/* Rank with change */}
                    <div className="flex items-center gap-1 w-12">
                      <span className="font-mono text-sm font-bold" style={{ color: 'var(--text-3)' }}>
                        #{stock.rank}
                      </span>
                      {getRankChangeIcon(stock.rankChange)}
                    </div>

                    {/* Ticker */}
                    <div className="flex flex-col flex-1">
                      <div className="font-mono font-bold text-sm" style={{ color: 'var(--text-2)' }}>
                        ${stock.ticker}
                      </div>
                      <div className="text-xs truncate" style={{ color: 'var(--text-5)' }}>
                        {stock.name}
                      </div>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="flex flex-col items-end gap-1">
                    <div className="flex items-center gap-1 text-xs" style={{ color: 'var(--text-3)' }}>
                      <MessageSquare className="w-3 h-3" />
                      <span className="font-mono">{formatNumber(stock.mentions)}</span>
                      {stock.mentionsChange !== 0 && (
                        <span className="text-xs" style={{ color: getRankChangeColor(stock.mentionsChange) }}>
                          ({stock.mentionsChange > 0 ? '+' : ''}{stock.mentionsChange})
                        </span>
                      )}
                    </div>
                    <div className="flex items-center gap-1 text-xs" style={{ color: 'var(--text-4)' }}>
                      <ThumbsUp className="w-3 h-3" />
                      <span className="font-mono">{formatNumber(stock.upvotes)}</span>
                    </div>
                  </div>
                </div>
              </button>
            ))}
          </div>

          {stocks.length === 0 && !loading && (
            <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
              <p className="text-sm">No data available</p>
            </div>
          )}

          {/* Footer Info */}
          <div className="mt-4 pt-3 border-t text-xs" style={{ borderColor: 'var(--bg-1)', color: 'var(--text-5)' }}>
            <p>Aggregated from multiple subreddits • Powered by ApeWisdom</p>
          </div>
        </>
      )}
    </div>
  );
}
