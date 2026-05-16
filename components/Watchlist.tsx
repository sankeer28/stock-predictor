'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Star, Plus, Trash2, RefreshCw, TrendingUp, TrendingDown, X } from 'lucide-react';

interface WatchlistItem {
  symbol: string;
  addedAt: number;
}

interface LivePrice {
  price: number | null;
  change: number | null;
  changePercent: number | null;
  name: string;
  marketState: string;
}

interface WatchlistProps {
  currentSymbol?: string;
  onSymbolClick?: (symbol: string) => void;
  inlineMobile?: boolean;
}

const STORAGE_KEY = 'stockWatchlist';
const DEFAULT_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'];

function fmtPrice(n: number | null) {
  if (n === null) return '—';
  return `$${n.toFixed(2)}`;
}

function fmtPct(n: number | null) {
  if (n === null) return '—';
  return `${n >= 0 ? '+' : ''}${n.toFixed(2)}%`;
}

export default function Watchlist({ currentSymbol, onSymbolClick, inlineMobile }: WatchlistProps) {
  const [items, setItems] = useState<WatchlistItem[]>([]);
  const [prices, setPrices] = useState<Record<string, LivePrice>>({});
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Load from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        setItems(JSON.parse(raw));
      } else {
        const defaults = DEFAULT_SYMBOLS.map(s => ({ symbol: s, addedAt: Date.now() }));
        setItems(defaults);
        localStorage.setItem(STORAGE_KEY, JSON.stringify(defaults));
      }
    } catch {
      setItems(DEFAULT_SYMBOLS.map(s => ({ symbol: s, addedAt: Date.now() })));
    }
  }, []);

  // Persist to localStorage
  useEffect(() => {
    if (items.length > 0) {
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(items)); } catch {}
    }
  }, [items]);

  const fetchPrices = useCallback(async (syms: string[]) => {
    if (!syms.length) return;
    setRefreshing(true);
    const results = await Promise.allSettled(
      syms.map(async (sym) => {
        const res = await fetch(`/api/price?symbol=${sym}`);
        if (!res.ok) throw new Error('Failed');
        const json = await res.json();
        return { sym, data: json };
      })
    );
    const newPrices: Record<string, LivePrice> = {};
    for (const r of results) {
      if (r.status === 'fulfilled') {
        const { sym, data } = r.value;
        newPrices[sym] = {
          price: data.price ?? null,
          change: data.change ?? null,
          changePercent: data.changePercent ?? null,
          name: data.companyName ?? sym,
          marketState: data.marketState ?? '',
        };
      }
    }
    setPrices(prev => ({ ...prev, ...newPrices }));
    setRefreshing(false);
  }, []);

  useEffect(() => {
    const syms = items.map(i => i.symbol);
    if (syms.length) {
      fetchPrices(syms);
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = setInterval(() => fetchPrices(syms), 30_000);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [items, fetchPrices]);

  const addSymbol = async () => {
    const sym = input.trim().toUpperCase();
    if (!sym || items.some(i => i.symbol === sym)) { setInput(''); return; }
    setLoading(true);
    try {
      const res = await fetch(`/api/price?symbol=${sym}`);
      if (!res.ok) throw new Error('Invalid symbol');
      setItems(prev => [...prev, { symbol: sym, addedAt: Date.now() }]);
      setInput('');
    } catch {
      // Invalid symbol — flash red briefly
    } finally {
      setLoading(false);
    }
  };

  const removeSymbol = (sym: string) => {
    setItems(prev => prev.filter(i => i.symbol !== sym));
    setPrices(prev => {
      const next = { ...prev };
      delete next[sym];
      return next;
    });
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') addSymbol();
    if (e.key === 'Escape') setInput('');
  };

  return (
    <div className="card">
      <span className="card-label">Watchlist</span>

      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5">
          <Star className="w-3.5 h-3.5" style={{ color: 'var(--accent)' }} />
          <span className="text-xs font-semibold" style={{ color: 'var(--text-3)' }}>{items.length} stocks</span>
        </div>
        <button
          onClick={() => fetchPrices(items.map(i => i.symbol))}
          className="p-1 border transition-opacity hover:opacity-70"
          style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}
          title="Refresh all"
        >
          <RefreshCw className={`w-3 h-3 ${refreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Add symbol input */}
      <div className="flex gap-1 mb-3">
        <input
          value={input}
          onChange={e => setInput(e.target.value.toUpperCase())}
          onKeyDown={handleKey}
          placeholder="Add ticker…"
          maxLength={10}
          className="flex-1 px-2 py-1.5 text-xs border"
          style={{
            background: 'var(--bg-3)',
            borderColor: 'var(--bg-1)',
            color: 'var(--text-2)',
            fontFamily: 'DM Mono, monospace',
          }}
        />
        <button
          onClick={addSymbol}
          disabled={loading || !input.trim()}
          className="px-2 py-1.5 border text-xs font-bold transition-opacity hover:opacity-80 disabled:opacity-40"
          style={{ background: 'var(--accent)', borderColor: 'var(--accent)', color: 'var(--text-0)' }}
        >
          {loading ? <RefreshCw className="w-3 h-3 animate-spin" /> : <Plus className="w-3 h-3" />}
        </button>
      </div>

      {/* Watchlist rows */}
      <div className="space-y-0 overflow-y-auto" style={{ maxHeight: '320px' }}>
        {items.length === 0 && (
          <p className="text-[10px] text-center py-4" style={{ color: 'var(--text-5)' }}>
            No stocks in watchlist. Add a ticker above.
          </p>
        )}
        {items.map(item => {
          const lp = prices[item.symbol];
          const isActive = item.symbol === currentSymbol;
          const positive = lp ? (lp.changePercent ?? 0) >= 0 : null;
          const color = positive === null ? 'var(--text-3)' : positive ? 'var(--success)' : 'var(--danger)';

          return (
            <div
              key={item.symbol}
              className="flex items-center justify-between gap-2 px-2 py-2 border-b group transition-all hover:opacity-90"
              style={{
                borderColor: 'var(--bg-1)',
                background: isActive ? 'var(--bg-3)' : 'transparent',
              }}
            >
              <button
                onClick={() => onSymbolClick?.(item.symbol)}
                className="flex items-center gap-2 flex-1 min-w-0 text-left"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-1">
                    <span className="text-xs font-bold font-mono" style={{ color: isActive ? 'var(--accent)' : 'var(--text-2)' }}>
                      {item.symbol}
                    </span>
                    {lp?.marketState === 'REGULAR' && (
                      <span className="w-1.5 h-1.5 rounded-full flex-shrink-0 animate-pulse" style={{ background: 'var(--success)' }} />
                    )}
                  </div>
                  {lp?.name && lp.name !== item.symbol && (
                    <div className="text-[9px] truncate" style={{ color: 'var(--text-5)' }}>{lp.name}</div>
                  )}
                </div>

                <div className="ml-auto flex-shrink-0 text-right">
                  <div className="text-xs font-mono" style={{ color: 'var(--text-2)' }}>
                    {lp ? fmtPrice(lp.price) : '…'}
                  </div>
                  <div className="flex items-center gap-0.5 justify-end">
                    {positive === true && <TrendingUp className="w-2.5 h-2.5" style={{ color }} />}
                    {positive === false && <TrendingDown className="w-2.5 h-2.5" style={{ color }} />}
                    <span className="text-[10px] font-mono font-bold" style={{ color }}>
                      {lp ? fmtPct(lp.changePercent) : '—'}
                    </span>
                  </div>
                </div>
              </button>

              <button
                onClick={() => removeSymbol(item.symbol)}
                className="opacity-0 group-hover:opacity-100 p-0.5 transition-all hover:opacity-70 flex-shrink-0"
                style={{ color: 'var(--danger)' }}
                title="Remove"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
          );
        })}
      </div>
    </div>
  );
}
