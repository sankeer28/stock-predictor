'use client';

import React, { useState, useCallback } from 'react';
import {
  Filter,
  Search,
  ChevronLeft,
  ChevronRight,
  Zap,
  RotateCcw,
  ExternalLink,
  TrendingDown,
  TrendingUp,
  Target,
  ArrowUpRight,
  GitMerge,
  BarChart2,
  Activity,
  Star,
  Wallet,
  Shield,
  Flame,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

// ────────────────────────────────────────────────────────────
// Presets
// ────────────────────────────────────────────────────────────

interface Preset {
  id: string;
  label: string;
  icon: LucideIcon;
  desc: string;
  scrId: string;
  direction: 'bullish' | 'bearish' | 'neutral';
}

const PRESETS: Preset[] = [
  {
    id: 'most_actives',
    label: 'Most Active',
    icon: Activity,
    desc: 'Highest trading volume today',
    scrId: 'most_actives',
    direction: 'neutral',
  },
  {
    id: 'day_gainers',
    label: 'Top Gainers',
    icon: TrendingUp,
    desc: 'Biggest price gainers today',
    scrId: 'day_gainers',
    direction: 'bullish',
  },
  {
    id: 'day_losers',
    label: 'Top Losers',
    icon: TrendingDown,
    desc: 'Biggest price decliners today',
    scrId: 'day_losers',
    direction: 'bearish',
  },
  {
    id: 'growth_technology_stocks',
    label: 'Growth Tech',
    icon: BarChart2,
    desc: 'High-growth technology stocks',
    scrId: 'growth_technology_stocks',
    direction: 'bullish',
  },
  {
    id: 'undervalued_growth_stocks',
    label: 'Undervalued Growth',
    icon: Target,
    desc: 'Growth stocks trading at a discount',
    scrId: 'undervalued_growth_stocks',
    direction: 'bullish',
  },
  {
    id: 'aggressive_small_caps',
    label: 'Aggressive Small Caps',
    icon: Flame,
    desc: 'Small cap stocks with aggressive growth',
    scrId: 'aggressive_small_caps',
    direction: 'bullish',
  },
  {
    id: 'small_cap_gainers',
    label: 'Small Cap Gainers',
    icon: ArrowUpRight,
    desc: 'Small caps up significantly today',
    scrId: 'small_cap_gainers',
    direction: 'bullish',
  },
  {
    id: 'undervalued_large_caps',
    label: 'Undervalued Large Caps',
    icon: Shield,
    desc: 'Large cap stocks trading below fair value',
    scrId: 'undervalued_large_caps',
    direction: 'bullish',
  },
  {
    id: 'most_shorted_stocks',
    label: 'Most Shorted',
    icon: GitMerge,
    desc: 'Stocks with highest short interest — squeeze candidates',
    scrId: 'most_shorted_stocks',
    direction: 'bearish',
  },
  {
    id: 'portfolio_anchors',
    label: 'Portfolio Anchors',
    icon: Wallet,
    desc: 'Stable, reliable large-cap holdings',
    scrId: 'portfolio_anchors',
    direction: 'neutral',
  },
  {
    id: 'strong_undervalued_stocks',
    label: 'Strong Undervalued',
    icon: Star,
    desc: 'Fundamentally strong stocks trading below fair value',
    scrId: 'strong_undervalued_stocks',
    direction: 'bullish',
  },
];

// ────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────

interface ScreenerResult {
  headers: string[];
  rows: Record<string, string>[];
  total: number;
  totalPages: number;
  page: number;
}

interface Props {
  onSelectTicker: (ticker: string) => void;
}

// ────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────

function changeColor(val: string) {
  if (!val || val === '-') return 'var(--text-4)';
  const n = parseFloat(val.replace('%', '').replace('+', ''));
  if (isNaN(n)) return 'var(--text-4)';
  return n > 0 ? 'var(--green-1)' : n < 0 ? 'var(--red-1)' : 'var(--text-4)';
}

const DISPLAY_COLS = ['Ticker', 'Company', 'Price', 'Change', 'Volume', 'Mkt Cap', 'Fwd P/E', '52W Chg'];

// ────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────

export default function StockScreener({ onSelectTicker }: Props) {
  const [activePreset, setActivePreset] = useState<Preset | null>(null);
  const [result, setResult] = useState<ScreenerResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [sortCol, setSortCol] = useState<string>('');
  const [sortAsc, setSortAsc] = useState(true);

  const runScreener = useCallback(async (preset: Preset | null, page = 1) => {
    if (!preset) return;
    setLoading(true);
    setError(null);
    setCurrentPage(page);

    try {
      const params = new URLSearchParams({ scrId: preset.scrId, page: String(page) });
      const res = await fetch(`/api/screener?${params}`);
      const data = await res.json();
      if (!data.success) throw new Error(data.error || 'Screener failed');
      setResult({
        headers: data.headers,
        rows: data.rows,
        total: data.total,
        totalPages: data.totalPages,
        page: data.page,
      });
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, []);

  const handlePreset = (preset: Preset) => {
    const next = activePreset?.id === preset.id ? null : preset;
    setActivePreset(next);
    if (next) runScreener(next, 1);
  };

  const handleReset = () => {
    setActivePreset(null);
    setResult(null);
    setError(null);
  };

  const handleSort = (col: string) => {
    if (sortCol === col) setSortAsc(a => !a);
    else { setSortCol(col); setSortAsc(true); }
  };

  const displayRows = result
    ? sortCol
      ? [...result.rows].sort((a, b) => {
          const av = parseFloat((a[sortCol] || '').replace(/[^0-9.-]/g, '')) || 0;
          const bv = parseFloat((b[sortCol] || '').replace(/[^0-9.-]/g, '')) || 0;
          return sortAsc ? av - bv : bv - av;
        })
      : result.rows
    : [];

  const TABLE_COLS = result ? DISPLAY_COLS.filter(col => result.headers.includes(col)) : [];

  return (
    <div className="card" style={{ position: 'relative' }}>
      <span className="card-label">Stock Screener</span>

      {/* ── Presets ─────────────────────────────────────── */}
      <div className="mb-4">
        <div className="text-[10px] uppercase tracking-widest mb-2" style={{ color: 'var(--text-5)' }}>
          Quick Presets
        </div>
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map(p => {
            const active = activePreset?.id === p.id;
            const dirColor =
              p.direction === 'bullish'
                ? 'var(--green-1)'
                : p.direction === 'bearish'
                ? 'var(--red-1)'
                : 'var(--text-3)';
            return (
              <button
                key={p.id}
                onClick={() => handlePreset(p)}
                title={p.desc}
                className="flex items-center gap-1.5 px-3 py-1.5 border transition-all text-xs"
                style={{
                  background: active ? 'var(--bg-2)' : 'var(--bg-4)',
                  borderColor: active ? dirColor : 'var(--bg-1)',
                  color: active ? dirColor : 'var(--text-4)',
                  borderLeftWidth: active ? '2px' : '1px',
                }}
              >
                <p.icon className="w-3.5 h-3.5" />
                <span className="font-medium">{p.label}</span>
                {active && <span className="opacity-60 text-[9px]">✕</span>}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Action bar ───────────────────────────────────── */}
      <div className="flex items-center gap-2 mb-3">
        {(activePreset || result) && (
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-1.5 border text-xs transition-all"
            style={{ background: 'var(--bg-4)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}
          >
            <RotateCcw className="w-3 h-3" />
            Reset
          </button>
        )}
        <button
          onClick={() => runScreener(activePreset, 1)}
          disabled={loading || !activePreset}
          className="flex items-center gap-2 px-4 py-1.5 border transition-all text-xs font-semibold disabled:opacity-50 ml-auto"
          style={{ background: 'var(--accent)', borderColor: 'var(--accent)', color: 'var(--text-0)' }}
        >
          {loading ? (
            <>
              <div
                className="w-3 h-3 border-2 rounded-full animate-spin"
                style={{ borderColor: 'var(--text-0)', borderTopColor: 'transparent' }}
              />
              Scanning...
            </>
          ) : (
            <>
              <Search className="w-3.5 h-3.5" />
              Run Screener
            </>
          )}
        </button>
      </div>

      {/* ── Error ────────────────────────────────────── */}
      {error && (
        <div
          className="mb-4 px-4 py-3 border text-sm"
          style={{ borderColor: 'var(--danger)', color: 'var(--danger)', background: 'rgba(200,50,50,0.08)' }}
        >
          {error}
        </div>
      )}

      {/* ── Results ──────────────────────────────────── */}
      {result && !loading && (
        <>
          {/* Summary bar */}
          <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold" style={{ color: 'var(--text-2)' }}>
                {result.total > 0
                  ? `${result.total.toLocaleString()} stocks found`
                  : result.rows.length > 0
                  ? `${result.rows.length} stocks found`
                  : 'No results'}
              </span>
              {result.total > 0 && result.totalPages > 1 && (
                <span className="text-xs" style={{ color: 'var(--text-5)' }}>
                  page {result.page} of {result.totalPages}
                </span>
              )}
            </div>

            {/* Pagination controls */}
            {result.totalPages > 1 && (
              <div className="flex items-center gap-1">
                <button
                  onClick={() => runScreener(activePreset, Math.max(1, currentPage - 1))}
                  disabled={currentPage <= 1 || loading}
                  className="p-1.5 border transition-all disabled:opacity-40"
                  style={{ borderColor: 'var(--bg-1)', color: 'var(--text-4)', background: 'var(--bg-4)' }}
                >
                  <ChevronLeft className="w-4 h-4" />
                </button>
                {Array.from({ length: Math.min(5, result.totalPages) }, (_, i) => {
                  const pg = Math.max(1, Math.min(result.totalPages - 4, currentPage - 2)) + i;
                  return (
                    <button
                      key={pg}
                      onClick={() => runScreener(activePreset, pg)}
                      disabled={loading}
                      className="min-w-[28px] px-1.5 py-1 border text-xs transition-all"
                      style={{
                        borderColor: pg === currentPage ? 'var(--accent)' : 'var(--bg-1)',
                        background: pg === currentPage ? 'var(--accent)' : 'var(--bg-4)',
                        color: pg === currentPage ? 'var(--text-0)' : 'var(--text-4)',
                      }}
                    >
                      {pg}
                    </button>
                  );
                })}
                <button
                  onClick={() => runScreener(activePreset, Math.min(result.totalPages, currentPage + 1))}
                  disabled={currentPage >= result.totalPages || loading}
                  className="p-1.5 border transition-all disabled:opacity-40"
                  style={{ borderColor: 'var(--bg-1)', color: 'var(--text-4)', background: 'var(--bg-4)' }}
                >
                  <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            )}
          </div>

          {/* Table */}
          {result.rows.length > 0 ? (
            <div className="overflow-x-auto" style={{ borderRadius: 0 }}>
              <table className="w-full text-xs border-collapse" style={{ minWidth: 700 }}>
                <thead>
                  <tr style={{ background: 'var(--bg-3)' }}>
                    {TABLE_COLS.map(col => (
                      <th
                        key={col}
                        className="px-3 py-2 text-left font-semibold border-b cursor-pointer select-none transition-colors hover:opacity-80"
                        style={{
                          borderColor: 'var(--bg-1)',
                          color: sortCol === col ? 'var(--accent)' : 'var(--text-5)',
                          letterSpacing: '0.06em',
                          textTransform: 'uppercase',
                          fontSize: 10,
                          whiteSpace: 'nowrap',
                        }}
                        onClick={() => handleSort(col)}
                      >
                        {col}
                        {sortCol === col && (
                          <span className="ml-1">{sortAsc ? '↑' : '↓'}</span>
                        )}
                      </th>
                    ))}
                    <th
                      className="px-3 py-2 border-b"
                      style={{ borderColor: 'var(--bg-1)', width: 32 }}
                    />
                  </tr>
                </thead>
                <tbody>
                  {displayRows.map((row, i) => {
                    const ticker = row['Ticker'] || '';
                    return (
                      <tr
                        key={ticker + i}
                        className="border-b transition-colors"
                        style={{
                          borderColor: 'var(--bg-1)',
                          background: i % 2 === 0 ? 'transparent' : 'rgba(0,0,0,0.1)',
                        }}
                        onMouseEnter={e => (e.currentTarget.style.background = 'var(--hover)')}
                        onMouseLeave={e =>
                          (e.currentTarget.style.background =
                            i % 2 === 0 ? 'transparent' : 'rgba(0,0,0,0.1)')
                        }
                      >
                        {TABLE_COLS.map(col => {
                          const val = row[col] || '-';
                          const isTicker = col === 'Ticker';
                          const isChange = col === 'Change' || col === '52W Chg';

                          if (isTicker) {
                            return (
                              <td key={col} className="px-3 py-2">
                                <button
                                  onClick={() => onSelectTicker(ticker)}
                                  className="font-bold font-mono transition-colors hover:opacity-80"
                                  style={{ color: 'var(--accent)' }}
                                >
                                  {val}
                                </button>
                              </td>
                            );
                          }

                          if (isChange) {
                            return (
                              <td
                                key={col}
                                className="px-3 py-2 font-mono"
                                style={{ color: changeColor(val), fontVariantNumeric: 'tabular-nums' }}
                              >
                                {val}
                              </td>
                            );
                          }

                          if (col === 'Company') {
                            return (
                              <td
                                key={col}
                                className="px-3 py-2 max-w-[180px]"
                                style={{ color: 'var(--text-3)' }}
                              >
                                <span className="truncate block" title={val}>{val}</span>
                              </td>
                            );
                          }

                          return (
                            <td
                              key={col}
                              className="px-3 py-2 font-mono"
                              style={{ color: 'var(--text-4)', fontVariantNumeric: 'tabular-nums' }}
                            >
                              {val}
                            </td>
                          );
                        })}
                        <td className="px-2 py-2">
                          <button
                            onClick={() => onSelectTicker(ticker)}
                            title="Load in chart"
                            className="p-1 border transition-all hover:opacity-80"
                            style={{ borderColor: 'var(--bg-1)', color: 'var(--text-5)', background: 'var(--bg-4)' }}
                          >
                            <ExternalLink className="w-3 h-3" />
                          </button>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div
              className="py-12 text-center border"
              style={{ borderColor: 'var(--bg-1)', color: 'var(--text-5)' }}
            >
              <Filter className="w-8 h-8 mx-auto mb-3 opacity-30" />
              <div className="text-sm">No stocks found.</div>
            </div>
          )}
        </>
      )}

      {/* Empty state before first run */}
      {!result && !loading && !error && (
        <div
          className="py-10 text-center border"
          style={{ borderColor: 'var(--bg-1)', color: 'var(--text-5)' }}
        >
          <Zap className="w-8 h-8 mx-auto mb-3 opacity-30" />
          <div className="text-sm mb-1">Select a preset and run the screener.</div>
          <div className="text-xs">Results are fetched live from Yahoo Finance.</div>
        </div>
      )}
    </div>
  );
}
