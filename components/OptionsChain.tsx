'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { BarChart2, RefreshCw, ChevronDown } from 'lucide-react';

interface Option {
  strike: number;
  lastPrice: number;
  bid: number;
  ask: number;
  volume: number;
  openInterest: number;
  impliedVolatility: number;
  inTheMoney: boolean;
  expiration: number;
}

interface OptionsData {
  symbol: string;
  underlyingPrice: number | null;
  expirationDates: number[];
  selectedExpiry: number | null;
  calls: Option[];
  puts: Option[];
}

interface OptionsChainProps {
  symbol: string;
}

function fmt(n: number | null | undefined, d = 2) {
  if (n == null) return '—';
  return n.toFixed(d);
}

function fmtVol(n: number | null | undefined) {
  if (!n) return '—';
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`;
  return String(n);
}

function fmtIV(n: number | null | undefined) {
  if (n == null) return '—';
  return `${(n * 100).toFixed(1)}%`;
}

function OptionRow({ opt, type, underlyingPrice }: { opt: Option; type: 'call' | 'put'; underlyingPrice: number | null }) {
  const itm = opt.inTheMoney;
  const itmColor = type === 'call'
    ? (itm ? 'rgba(34,197,94,0.08)' : 'transparent')
    : (itm ? 'rgba(239,68,68,0.08)' : 'transparent');
  const spread = opt.ask - opt.bid;

  return (
    <tr style={{ background: itmColor }}>
      <td className="px-2 py-1 text-[10px] font-mono font-bold text-right border-r"
        style={{ borderColor: 'var(--bg-1)', color: itm ? (type === 'call' ? 'var(--success)' : 'var(--danger)') : 'var(--text-3)' }}>
        ${opt.strike}
      </td>
      <td className="px-2 py-1 text-[10px] font-mono text-right border-r"
        style={{ borderColor: 'var(--bg-1)', color: 'var(--text-2)' }}>{fmt(opt.lastPrice)}</td>
      <td className="px-2 py-1 text-[10px] font-mono text-right border-r"
        style={{ borderColor: 'var(--bg-1)', color: 'var(--text-3)' }}>{fmt(opt.bid)}</td>
      <td className="px-2 py-1 text-[10px] font-mono text-right border-r"
        style={{ borderColor: 'var(--bg-1)', color: 'var(--text-3)' }}>{fmt(opt.ask)}</td>
      <td className="px-2 py-1 text-[10px] font-mono text-right border-r"
        style={{ borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}>{fmtVol(opt.volume)}</td>
      <td className="px-2 py-1 text-[10px] font-mono text-right border-r"
        style={{ borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}>{fmtVol(opt.openInterest)}</td>
      <td className="px-2 py-1 text-[10px] font-mono text-right"
        style={{ color: opt.impliedVolatility > 0.5 ? 'var(--warning)' : 'var(--text-4)' }}>
        {fmtIV(opt.impliedVolatility)}
      </td>
    </tr>
  );
}

export default function OptionsChain({ symbol }: OptionsChainProps) {
  const [data, setData] = useState<OptionsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [tab, setTab] = useState<'calls' | 'puts'>('calls');
  const [selectedExpiry, setSelectedExpiry] = useState<string>('');
  const [showChain, setShowChain] = useState(false);

  const fetchOptions = useCallback(async (expiry?: string) => {
    if (!symbol) return;
    setLoading(true);
    setError('');
    try {
      const url = expiry
        ? `/api/options?symbol=${symbol}&expiry=${expiry}`
        : `/api/options?symbol=${symbol}`;
      const res = await fetch(url);
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Failed');
      }
      const json: OptionsData = await res.json();
      setData(json);
      if (!expiry && json.selectedExpiry) {
        setSelectedExpiry(String(json.selectedExpiry));
      }
    } catch (e: any) {
      setError(e.message || 'Failed to load options');
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    if (showChain) fetchOptions();
  }, [symbol, showChain, fetchOptions]);

  const handleExpiryChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value;
    setSelectedExpiry(val);
    fetchOptions(val);
  };

  const options = tab === 'calls' ? (data?.calls ?? []) : (data?.puts ?? []);

  return (
    <div className="card">
      <span className="card-label">Options Chain</span>

      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-1.5">
          <BarChart2 className="w-3.5 h-3.5" style={{ color: 'var(--accent)' }} />
          <span className="text-xs font-semibold" style={{ color: 'var(--text-3)' }}>{symbol}</span>
          {data?.underlyingPrice && (
            <span className="text-xs font-mono" style={{ color: 'var(--text-4)' }}>@ ${data.underlyingPrice.toFixed(2)}</span>
          )}
        </div>
        <div className="flex items-center gap-1">
          {showChain && (
            <button
              onClick={() => fetchOptions(selectedExpiry || undefined)}
              className="p-1 border transition-opacity hover:opacity-70"
              style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}
            >
              <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
            </button>
          )}
          <button
            onClick={() => setShowChain(s => !s)}
            className="px-2 py-1 border text-[10px] font-semibold transition-all hover:opacity-80"
            style={{
              background: showChain ? 'var(--accent)' : 'var(--bg-3)',
              borderColor: showChain ? 'var(--accent)' : 'var(--bg-1)',
              color: showChain ? 'var(--text-0)' : 'var(--text-4)',
            }}
          >
            {showChain ? 'Hide' : 'Load Options'}
          </button>
        </div>
      </div>

      {!showChain && (
        <p className="text-[10px] text-center py-2" style={{ color: 'var(--text-5)' }}>
          Click "Load Options" to view the options chain for {symbol}
        </p>
      )}

      {showChain && loading && !data && (
        <div className="flex justify-center py-6">
          <RefreshCw className="w-5 h-5 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      )}

      {showChain && error && (
        <p className="text-xs text-center py-3" style={{ color: 'var(--danger)' }}>{error}</p>
      )}

      {showChain && data && (
        <div>
          {/* Expiry selector */}
          {data.expirationDates.length > 0 && (
            <div className="flex items-center gap-2 mb-3">
              <span className="text-[10px]" style={{ color: 'var(--text-4)' }}>Expiry</span>
              <div className="relative flex-1">
                <select
                  value={selectedExpiry}
                  onChange={handleExpiryChange}
                  className="w-full px-2 py-1 text-[10px] border appearance-none"
                  style={{
                    background: 'var(--bg-3)',
                    borderColor: 'var(--bg-1)',
                    color: 'var(--text-2)',
                    fontFamily: 'DM Mono, monospace',
                  }}
                >
                  {data.expirationDates.slice(0, 12).map(ts => {
                    const d = new Date(ts * 1000);
                    return (
                      <option key={ts} value={String(ts)}>
                        {d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                      </option>
                    );
                  })}
                </select>
                <ChevronDown className="w-3 h-3 absolute right-1.5 top-1/2 -translate-y-1/2 pointer-events-none" style={{ color: 'var(--text-4)' }} />
              </div>
            </div>
          )}

          {/* Calls / Puts toggle */}
          <div className="flex mb-2 border" style={{ borderColor: 'var(--bg-1)' }}>
            {(['calls', 'puts'] as const).map(t => (
              <button
                key={t}
                onClick={() => setTab(t)}
                className="flex-1 py-1.5 text-[10px] font-bold uppercase tracking-wider transition-all"
                style={{
                  background: tab === t ? (t === 'calls' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)') : 'transparent',
                  color: tab === t ? (t === 'calls' ? 'var(--success)' : 'var(--danger)') : 'var(--text-4)',
                  borderRight: t === 'calls' ? '1px solid var(--bg-1)' : 'none',
                }}
              >
                {t}
              </button>
            ))}
          </div>

          {/* ITM legend */}
          <div className="flex gap-3 mb-1.5 px-1">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-sm" style={{ background: tab === 'calls' ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)' }} />
              <span className="text-[8px]" style={{ color: 'var(--text-5)' }}>In the money</span>
            </div>
          </div>

          {/* Table */}
          <div className="overflow-x-auto border" style={{ borderColor: 'var(--bg-1)' }}>
            <table className="w-full border-collapse" style={{ minWidth: '480px' }}>
              <thead>
                <tr style={{ background: 'var(--bg-3)' }}>
                  {['Strike', 'Last', 'Bid', 'Ask', 'Volume', 'OI', 'IV'].map((h, i) => (
                    <th
                      key={h}
                      className="px-2 py-1 text-[9px] uppercase tracking-wider font-semibold text-right border-r"
                      style={{ color: 'var(--text-4)', borderColor: 'var(--bg-1)' }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {options.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="text-center py-4 text-[10px]" style={{ color: 'var(--text-5)' }}>
                      No options data for this expiry
                    </td>
                  </tr>
                ) : (
                  options.map((opt, i) => (
                    <OptionRow
                      key={i}
                      opt={opt}
                      type={tab === 'calls' ? 'call' : 'put'}
                      underlyingPrice={data.underlyingPrice}
                    />
                  ))
                )}
              </tbody>
            </table>
          </div>

          <p className="text-[8px] mt-2 text-center" style={{ color: 'var(--text-5)' }}>
            Options data from Yahoo Finance · 15-min delay · Not financial advice
          </p>
        </div>
      )}
    </div>
  );
}
