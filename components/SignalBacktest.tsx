'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { Activity, RefreshCw, Loader2 } from 'lucide-react';
import { StockData } from '@/types';
import { fetchJSON, invalidateCache } from '@/lib/clientFetch';
import { runSignalBacktest, SignalBacktestResult } from '@/lib/signalBacktest';

interface Props {
  symbol: string;
}

function fmtPct(v: number, withSign = true): string {
  const sign = withSign && v > 0 ? '+' : '';
  return `${sign}${v.toFixed(1)}%`;
}

function signColor(v: number): string {
  if (v > 0) return 'var(--green-1)';
  if (v < 0) return 'var(--red-1)';
  return 'var(--text-3)';
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ background: 'var(--bg-3)', padding: '8px 10px', border: '1px solid var(--bg-1)' }}>
      <div className="text-[10px] uppercase tracking-wide" style={{ color: 'var(--text-5)' }}>{label}</div>
      <div className="text-sm font-mono font-semibold mt-0.5" style={{ color: color ?? 'var(--text-2)' }}>{value}</div>
    </div>
  );
}

export default function SignalBacktest({ symbol }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SignalBacktestResult | null>(null);

  const run = useCallback(async (sym: string, forceFresh = false) => {
    if (!sym) return;
    setLoading(true);
    setError(null);
    const url = `/api/stock?symbol=${encodeURIComponent(sym)}&days=730&interval=1d`;
    if (forceFresh) invalidateCache(url);
    try {
      const res = await fetchJSON<{ data?: StockData[]; error?: string }>(url, {
        ttlMs: 5 * 60 * 1000,
        retries: 2,
      });
      if (res.error) throw new Error(res.error);
      const data = res.data ?? [];
      const backtest = runSignalBacktest(data);
      if (!backtest) throw new Error('Not enough history (need ~1 year+) to backtest this ticker.');
      setResult(backtest);
    } catch (e: any) {
      setError(e?.message || 'Failed to run backtest');
      setResult(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    run(symbol);
  }, [symbol, run]);

  const downsampled = result
    ? result.equityCurve.filter((_, i) => i % Math.ceil(result.equityCurve.length / 260) === 0 || i === result.equityCurve.length - 1)
    : [];

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Signal Backtester</span>
        </div>
        <button
          onClick={() => run(symbol, true)}
          disabled={loading}
          className="flex items-center gap-1.5 px-2 py-1 text-xs border transition-all disabled:opacity-50"
          style={{ background: 'var(--bg-4)', borderColor: 'var(--bg-1)', color: 'var(--text-3)' }}
          title="Re-run backtest"
        >
          {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
          Re-run
        </button>
      </div>
      <p className="text-xs mb-3" style={{ color: 'var(--text-5)' }}>
        Following Trading Signals (long/flat) vs. buy &amp; hold · ~2 years
      </p>

      {error && (
        <div className="text-sm py-3" style={{ color: 'var(--danger)' }}>{error}</div>
      )}

      {!error && loading && !result && (
        <div className="flex items-center gap-2 text-xs py-6" style={{ color: 'var(--text-4)' }}>
          <Loader2 className="w-4 h-4 animate-spin" /> Running backtest…
        </div>
      )}

      {!error && result && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mb-4">
            <Stat label="Strategy" value={fmtPct(result.strategyReturn)} color={signColor(result.strategyReturn)} />
            <Stat label="Buy & Hold" value={fmtPct(result.buyHoldReturn)} color={signColor(result.buyHoldReturn)} />
            <Stat label="Alpha" value={fmtPct(result.alpha)} color={signColor(result.alpha)} />
            <Stat label="Win rate" value={`${result.winRate.toFixed(0)}%`} />
            <Stat label="Trades" value={String(result.trades)} />
            <Stat label="Max DD" value={fmtPct(-result.maxDrawdown, false)} color="var(--red-1)" />
            <Stat label="Exposure" value={`${result.exposure.toFixed(0)}%`} />
            <Stat
              label="Verdict"
              value={result.alpha >= 0 ? 'Beats hold' : 'Lags hold'}
              color={result.alpha >= 0 ? 'var(--green-1)' : 'var(--red-1)'}
            />
          </div>

          <div style={{ width: '100%', height: 240 }}>
            <ResponsiveContainer>
              <LineChart data={downsampled} margin={{ top: 5, right: 8, bottom: 5, left: -10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-1)" />
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: 'var(--text-5)' }}
                  minTickGap={48}
                  tickFormatter={(d: string) => {
                    const dt = new Date(d);
                    return `${dt.getFullYear().toString().slice(2)}/${dt.getMonth() + 1}`;
                  }}
                />
                <YAxis
                  tick={{ fontSize: 10, fill: 'var(--text-5)' }}
                  tickFormatter={(v: number) => v.toFixed(0)}
                  domain={['auto', 'auto']}
                  width={42}
                />
                <Tooltip
                  contentStyle={{
                    background: 'var(--bg-4)',
                    border: '1px solid var(--bg-1)',
                    fontSize: '11px',
                  }}
                  labelStyle={{ color: 'var(--text-4)' }}
                  labelFormatter={(d: string) => new Date(d).toLocaleDateString()}
                  formatter={(v: number, name: string) => [v.toFixed(1), name]}
                />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line
                  type="monotone"
                  dataKey="strategy"
                  name="Signals"
                  stroke="var(--green-1)"
                  dot={false}
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="buyHold"
                  name="Buy & Hold"
                  stroke="var(--blue-1)"
                  dot={false}
                  strokeWidth={1.5}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div
            className="mt-3 pt-2 text-[10px] leading-relaxed"
            style={{ borderTop: '1px solid var(--bg-1)', color: 'var(--text-5)' }}
          >
            Indexed to 100 at start. Strategy holds while signals are buy-side, moves to cash on
            sell-side, and keeps its position on hold. Excludes fees, slippage and taxes — educational
            only, not financial advice.
          </div>
        </>
      )}
    </div>
  );
}
