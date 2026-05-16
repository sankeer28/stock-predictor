'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { TrendingUp, TrendingDown, Zap, BarChart2, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';

interface Mover {
  symbol: string;
  name: string;
  price: number | null;
  change: number | null;
  changePercent: number | null;
  volume: number | null;
  marketCap: number | null;
}

interface MarketData {
  gainers: Mover[];
  losers: Mover[];
  trending: Mover[];
  indices: Mover[];
}

function fmt(n: number | null, decimals = 2) {
  if (n === null || n === undefined) return '—';
  return n.toFixed(decimals);
}

function fmtLarge(n: number | null) {
  if (!n) return '—';
  if (n >= 1e12) return `$${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9)  return `$${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6)  return `$${(n / 1e6).toFixed(1)}M`;
  return `$${n.toLocaleString()}`;
}

function fmtVol(n: number | null) {
  if (!n) return '—';
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}

function MoverRow({ mover, onClick }: { mover: Mover; onClick: (s: string) => void }) {
  const positive = (mover.changePercent ?? 0) >= 0;
  const color = positive ? 'var(--success)' : 'var(--danger)';
  return (
    <button
      onClick={() => onClick(mover.symbol)}
      className="w-full flex items-center justify-between gap-2 px-2 py-1.5 border-b transition-all hover:opacity-80 text-left"
      style={{ borderColor: 'var(--bg-1)' }}
    >
      <div className="min-w-0">
        <div className="text-xs font-bold font-mono truncate" style={{ color: 'var(--text-2)' }}>{mover.symbol}</div>
        <div className="text-[9px] truncate" style={{ color: 'var(--text-5)' }}>{mover.name}</div>
      </div>
      <div className="flex-shrink-0 text-right">
        <div className="text-xs font-mono font-semibold" style={{ color: 'var(--text-2)' }}>
          {mover.price !== null ? `$${fmt(mover.price)}` : '—'}
        </div>
        <div className="text-[10px] font-mono font-bold" style={{ color }}>
          {positive ? '+' : ''}{fmt(mover.changePercent)}%
        </div>
      </div>
    </button>
  );
}

function IndexRow({ mover }: { mover: Mover }) {
  const positive = (mover.changePercent ?? 0) >= 0;
  const color = positive ? 'var(--success)' : 'var(--danger)';
  const displayName = mover.symbol === '^VIX' ? 'VIX' : mover.symbol;
  return (
    <div className="flex items-center justify-between px-2 py-1.5 border-b" style={{ borderColor: 'var(--bg-1)' }}>
      <span className="text-xs font-bold font-mono" style={{ color: 'var(--text-3)' }}>{displayName}</span>
      <div className="flex items-center gap-3">
        <span className="text-xs font-mono" style={{ color: 'var(--text-2)' }}>
          {mover.price !== null ? (mover.symbol === '^VIX' ? fmt(mover.price) : `$${fmt(mover.price)}`) : '—'}
        </span>
        <span className="text-[10px] font-mono font-bold w-14 text-right" style={{ color }}>
          {positive ? '+' : ''}{fmt(mover.changePercent)}%
        </span>
      </div>
    </div>
  );
}

type Tab = 'gainers' | 'losers' | 'trending';

interface MarketMoversProps {
  onTickerClick?: (symbol: string) => void;
  inlineMobile?: boolean;
}

export default function MarketMovers({ onTickerClick, inlineMobile }: MarketMoversProps) {
  const [data, setData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<Tab>('gainers');
  const [collapsed, setCollapsed] = useState(false);

  const fetchData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/market-movers');
      if (!res.ok) throw new Error('Failed to fetch');
      setData(await res.json());
    } catch (e: any) {
      setError(e.message || 'Failed');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 60_000);
    return () => clearInterval(id);
  }, [fetchData]);

  const handleClick = (symbol: string) => {
    if (onTickerClick) onTickerClick(symbol);
  };

  const rows = data ? (tab === 'gainers' ? data.gainers : tab === 'losers' ? data.losers : data.trending) : [];

  const TABS: { id: Tab; label: string; icon: React.ReactNode }[] = [
    { id: 'gainers',  label: 'Gainers',  icon: <TrendingUp className="w-3 h-3" /> },
    { id: 'losers',   label: 'Losers',   icon: <TrendingDown className="w-3 h-3" /> },
    { id: 'trending', label: 'Trending', icon: <Zap className="w-3 h-3" /> },
  ];

  return (
    <div className="card">
      <span className="card-label">Market Movers</span>

      <div className="flex justify-end mb-2">
        <div className="flex items-center gap-1">
          <button
            onClick={fetchData}
            className="p-1 border transition-opacity hover:opacity-70"
            style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}
            title="Refresh"
          >
            <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={() => setCollapsed(c => !c)}
            className="p-1 border transition-opacity hover:opacity-70"
            style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}
          >
            {collapsed ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
          </button>
        </div>
      </div>

      {/* Major Indices */}
      {data?.indices && data.indices.length > 0 && !collapsed && (
        <div className="mb-2 border" style={{ borderColor: 'var(--bg-1)' }}>
          {data.indices.map(idx => <IndexRow key={idx.symbol} mover={idx} />)}
        </div>
      )}

      {!collapsed && (
        <>
          {/* Tabs */}
          <div className="flex mb-2 border" style={{ borderColor: 'var(--bg-1)' }}>
            {TABS.map(t => (
              <button
                key={t.id}
                onClick={() => setTab(t.id)}
                className="flex-1 flex items-center justify-center gap-1 py-1.5 text-[10px] font-semibold transition-all"
                style={{
                  background: tab === t.id ? 'var(--bg-3)' : 'transparent',
                  color: tab === t.id ? 'var(--accent)' : 'var(--text-4)',
                  borderRight: t.id !== 'trending' ? '1px solid var(--bg-1)' : 'none',
                }}
              >
                {t.icon}{t.label}
              </button>
            ))}
          </div>

          {error && (
            <p className="text-xs text-center py-3" style={{ color: 'var(--danger)' }}>{error}</p>
          )}

          {loading && !data && (
            <div className="flex justify-center py-4">
              <RefreshCw className="w-5 h-5 animate-spin" style={{ color: 'var(--accent)' }} />
            </div>
          )}

          {rows.map(m => (
            <MoverRow key={m.symbol} mover={m} onClick={handleClick} />
          ))}

          {!loading && rows.length === 0 && !error && (
            <p className="text-[10px] text-center py-3" style={{ color: 'var(--text-5)' }}>No data available</p>
          )}
        </>
      )}
    </div>
  );
}
