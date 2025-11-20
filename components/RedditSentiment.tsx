'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, MessageCircle, Loader2, RefreshCw, ExternalLink } from 'lucide-react';

interface RedditStock {
  no_of_comments: number;
  sentiment: 'Bullish' | 'Bearish' | 'Neutral';
  sentiment_score: number;
  ticker: string;
}

interface RedditSentimentProps {
  onTickerClick?: (ticker: string) => void;
  inlineMobile?: boolean;
}

export default function RedditSentiment({ onTickerClick, inlineMobile }: RedditSentimentProps) {
  const [stocks, setStocks] = useState<RedditStock[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const fetchRedditData = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch('/api/reddit');
      const data = await response.json();

      if (data.success) {
        setStocks(data.stocks);
        setLastUpdated(new Date());
      } else {
        setError(data.error || 'Failed to fetch Reddit data');
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRedditData();

    // Auto-refresh every 15 minutes if enabled
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchRedditData();
      }, 15 * 60 * 1000); // 15 minutes

      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'Bullish':
        return 'var(--success)';
      case 'Bearish':
        return 'var(--danger)';
      default:
        return 'var(--text-4)';
    }
  };

  const getSentimentBg = (sentiment: string) => {
    switch (sentiment) {
      case 'Bullish':
        return 'rgba(34, 197, 94, 0.1)';
      case 'Bearish':
        return 'rgba(239, 68, 68, 0.1)';
      default:
        return 'var(--bg-3)';
    }
  };

  const formatScore = (score: number) => {
    return (score * 100).toFixed(1) + '%';
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
          <span className="card-label">r/WallStreetBets Sentiment</span>
          <a
            href="https://www.reddit.com/r/wallstreetbets/"
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
              Updated {formatTimestamp(lastUpdated)}
            </span>
          )}
          <button
            onClick={() => fetchRedditData()}
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
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Most Discussed</div>
              <div className="text-lg font-bold font-mono" style={{ color: 'var(--text-2)' }}>
                {stocks[0]?.ticker || '-'}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>
                {stocks[0]?.no_of_comments || 0} comments
              </div>
            </div>

            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Bullish</div>
              <div className="text-lg font-bold" style={{ color: 'var(--success)' }}>
                {stocks.filter(s => s.sentiment === 'Bullish').length}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>stocks</div>
            </div>

            <div className="p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Bearish</div>
              <div className="text-lg font-bold" style={{ color: 'var(--danger)' }}>
                {stocks.filter(s => s.sentiment === 'Bearish').length}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-5)' }}>stocks</div>
            </div>
          </div>

          {/* Stock List */}
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {stocks.map((stock, index) => (
              <button
                key={stock.ticker}
                onClick={() => onTickerClick?.(stock.ticker)}
                className="w-full text-left p-3 border transition-all hover:opacity-80 overflow-hidden"
                style={{
                  background: 'var(--bg-2)',
                  borderColor: 'var(--bg-1)',
                  borderLeftWidth: '3px',
                  borderLeftColor: getSentimentColor(stock.sentiment),
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    {/* Rank */}
                    <div className="flex-shrink-0 w-6 text-center font-mono text-sm" style={{ color: 'var(--text-4)' }}>
                      {index + 1}
                    </div>

                    {/* Ticker */}
                    <div className="font-mono font-bold text-sm flex-shrink-0" style={{ color: 'var(--text-2)' }}>
                      ${stock.ticker}
                    </div>

                    {/* Sentiment Badge */}
                    <div
                      className="px-2 py-0.5 text-xs font-semibold whitespace-nowrap flex-shrink-0"
                      style={{
                        background: getSentimentBg(stock.sentiment),
                        color: getSentimentColor(stock.sentiment),
                      }}
                    >
                      {stock.sentiment}
                    </div>

                    {/* Score */}
                    <div className="text-xs font-mono flex-shrink-0" style={{ color: 'var(--text-3)' }}>
                      {formatScore(stock.sentiment_score)}
                    </div>
                  </div>

                  {/* Comments */}
                  <div className="flex items-center gap-1 text-xs whitespace-nowrap flex-shrink-0" style={{ color: 'var(--text-4)' }}>
                    <MessageCircle className="w-3 h-3 flex-shrink-0" />
                    <span>{stock.no_of_comments}</span>
                  </div>
                </div>
              </button>
            ))}
          </div>

          {stocks.length === 0 && !loading && (
            <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
              <p className="text-sm">No Reddit data available</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
