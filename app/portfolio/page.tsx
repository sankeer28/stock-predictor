'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import Link from 'next/link';
import { ArrowLeft, Plus, Trash2, RefreshCw, Pencil, Check, X, Download, Upload } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

// ─── Types ───────────────────────────────────────────────────────────────────

interface Holding {
  id: string;
  symbol: string;
  quantity: number;
  avgPrice: number;
}

interface Quote {
  symbol: string;
  name: string;
  price: number | null;
  dividendYield: number | null;
  dividendRate: number | null;
  priorDividendRate?: number | null;
  dividendGrowth?: number | null;
  dividendFrequencyMonths?: number | null;
  nextDividendDate?: string | null;
  dividendEvents?: { date: string; amount: number }[];
}

interface Projection {
  cagr: number | null;
  dataYears?: number | null;
  loading: boolean;
  error?: string | null;
}

type BaseCurrency = 'USD' | 'CAD';
type SortKey = 'symbol' | 'name' | 'quantity' | 'avgPrice' | 'price' | 'mktValue' | 'gl' | 'glPct' | 'weight';
type SortDir = 'asc' | 'desc';

interface PortfolioSettings {
  baseCurrency: BaseCurrency;
  horizons: number[];
  posOverrides: Record<string, { ratePct: string; contrib: string; includeDivs: boolean }>;
  projExcluded: string[];
}

interface HistoricalPoint {
  date: string;
  close: number;
  adjClose?: number;
}

interface RiskMetrics {
  beta: number | null;
  volatility: number | null;
  maxDrawdown: number | null;
  sharpe: number | null;
  points: number;
}

// ─── Benchmark comparison constants ──────────────────────────────────────────

const BENCH_OPTIONS = [
  { symbol: 'SPY',    label: 'S&P 500' },
  { symbol: 'QQQ',    label: 'NASDAQ 100' },
  { symbol: 'DIA',    label: 'Dow Jones' },
  { symbol: 'XIU.TO', label: 'TSX 60' },
];
const BENCH_COLORS: Record<string, string> = {
  'SPY':    '#60a5fa',
  'QQQ':    '#a78bfa',
  'DIA':    '#fb923c',
  'XIU.TO': '#facc15',
};
// ─── Helpers ─────────────────────────────────────────────────────────────────

const LS_KEY = 'portfolio_holdings_v1';
const SETTINGS_KEY = 'portfolio_settings_v1';

function loadHoldings(): Holding[] {
  try { return JSON.parse(localStorage.getItem(LS_KEY) ?? '[]'); } catch { return []; }
}
function saveHoldings(h: Holding[]) { localStorage.setItem(LS_KEY, JSON.stringify(h)); }

function usd(v: number) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(v);
}
function money(v: number, currency: BaseCurrency) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency, minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(v);
}
function pct(v: number) { return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`; }
function projectFV(value: number, rate: number, years: number, contrib = 0) {
  if (Math.abs(rate) < 1e-10) return value + contrib * years;
  return value * Math.pow(1 + rate, years) + contrib * (Math.pow(1 + rate, years) - 1) / rate;
}
function nativeCurrency(symbol: string): BaseCurrency {
  return symbol.endsWith('.TO') ? 'CAD' : 'USD';
}
function loadSettings(): Partial<PortfolioSettings> {
  try { return JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? '{}'); } catch { return {}; }
}
function saveSettings(settings: PortfolioSettings) {
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
}
function csvEscape(value: string | number) {
  const s = String(value);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function mean(values: number[]) {
  return values.length ? values.reduce((s, v) => s + v, 0) / values.length : 0;
}
function stdev(values: number[]) {
  if (values.length < 2) return null;
  const avg = mean(values);
  const variance = values.reduce((s, v) => s + Math.pow(v - avg, 2), 0) / (values.length - 1);
  return Math.sqrt(variance);
}
function covariance(a: number[], b: number[]) {
  if (a.length < 2 || a.length !== b.length) return null;
  const ma = mean(a);
  const mb = mean(b);
  return a.reduce((s, v, i) => s + (v - ma) * (b[i] - mb), 0) / (a.length - 1);
}
function maxDrawdownFromReturns(returns: number[]) {
  let value = 1;
  let peak = 1;
  let maxDrawdown = 0;
  returns.forEach(r => {
    value *= 1 + r;
    peak = Math.max(peak, value);
    maxDrawdown = Math.min(maxDrawdown, value / peak - 1);
  });
  return maxDrawdown;
}
function returnsByDate(points: HistoricalPoint[]) {
  const out: Record<string, number> = {};
  const sorted = points
    .filter(p => (p.adjClose ?? p.close) > 0)
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  for (let i = 1; i < sorted.length; i++) {
    const prev = sorted[i - 1].adjClose ?? sorted[i - 1].close;
    const curr = sorted[i].adjClose ?? sorted[i].close;
    if (prev > 0 && curr > 0) out[sorted[i].date.slice(0, 10)] = curr / prev - 1;
  }
  return out;
}
function fmtDate(value: string) {
  return new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', year: 'numeric' }).format(new Date(value));
}

// ─── Ticker autocomplete input ────────────────────────────────────────────────

interface SearchResult { symbol: string; shortname?: string; longname?: string; typeDisp?: string; }

function TickerInput({ value, onChange, onSelect, inputWidth }: {
  value: string;
  onChange: (v: string) => void;
  onSelect: (symbol: string, name: string) => void;
  inputWidth?: number | string;
}) {
  const [results,  setResults]  = useState<SearchResult[]>([]);
  const [open,     setOpen]     = useState(false);
  const [focused,  setFocused]  = useState(-1);
  const [dropPos,  setDropPos]  = useState({ top: 0, left: 0, width: 0 });
  const timerRef  = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wrapRef   = useRef<HTMLDivElement>(null);
  const inputRef  = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const q = value.trim();
    if (!q) { setResults([]); setOpen(false); return; }
    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(async () => {
      try {
        const res  = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
        const data = await res.json();
        const quotes: SearchResult[] = (data.quotes ?? [])
          .filter((r: SearchResult) => {
            if (!r.symbol) return false;
            const t = (r.typeDisp ?? '').toLowerCase();
            return t === 'equity' || t === 'etf' || t === '' || !r.typeDisp;
          })
          .slice(0, 7);
        setResults(quotes);
        setFocused(-1);
        if (quotes.length > 0 && inputRef.current) {
          const rect = inputRef.current.getBoundingClientRect();
          setDropPos({ top: rect.bottom + window.scrollY + 2, left: rect.left + window.scrollX, width: Math.max(rect.width, 280) });
          setOpen(true);
        } else {
          setOpen(false);
        }
      } catch { setResults([]); setOpen(false); }
    }, 220);
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, [value]);

  useEffect(() => {
    function handler(e: MouseEvent) {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  function pick(r: SearchResult) {
    onSelect(r.symbol, r.longname || r.shortname || r.symbol);
    setOpen(false);
    setResults([]);
  }

  function handleKey(e: React.KeyboardEvent) {
    if (!open) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); setFocused(f => Math.min(f + 1, results.length - 1)); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setFocused(f => Math.max(f - 1, 0)); }
    else if (e.key === 'Enter' && focused >= 0) { e.preventDefault(); pick(results[focused]); }
    else if (e.key === 'Escape') setOpen(false);
  }

  return (
    <div ref={wrapRef} style={{ width: inputWidth ?? 180 }}>
      <input
        ref={inputRef}
        type="text"
        placeholder="AAPL or Apple…"
        value={value}
        onChange={e => onChange(e.target.value.toUpperCase())}
        onFocus={() => results.length > 0 && setOpen(true)}
        onKeyDown={handleKey}
        style={{ width: '100%' }}
        autoComplete="off"
      />
      {open && (
        <div style={{
          position: 'fixed',
          top: dropPos.top,
          left: dropPos.left,
          width: dropPos.width,
          zIndex: 99999,
          background: 'var(--bg-3)',
          border: '2px solid var(--accent)',
          boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
        }}>
          {results.map((r, i) => (
            <div
              key={r.symbol}
              onMouseDown={() => pick(r)}
              style={{
                padding: '7px 12px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 8,
                background: i === focused ? 'var(--bg-1)' : 'transparent',
                borderBottom: i < results.length - 1 ? '1px solid var(--bg-1)' : 'none',
              }}
            >
              <span style={{ fontWeight: 700, color: 'var(--accent)', minWidth: 56, fontSize: 12 }}>{r.symbol}</span>
              <span style={{ fontSize: 11, color: 'var(--text-4)', flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {r.longname || r.shortname || ''}
              </span>
              {r.typeDisp && (
                <span style={{ fontSize: 9, color: 'var(--text-5)', flexShrink: 0, textTransform: 'uppercase', letterSpacing: '0.05em' }}>{r.typeDisp}</span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function PortfolioPage() {
  const [holdings,     setHoldings]     = useState<Holding[]>([]);
  const [quotes,       setQuotes]       = useState<Record<string, Quote>>({});
  const [projections,  setProjections]  = useState<Record<string, Projection>>({});
  const [quoteLoading, setQuoteLoading] = useState(false);
  const [projLoading,  setProjLoading]  = useState(false);
  const [riskLoading,  setRiskLoading]  = useState(false);
  const [quoteErrors,  setQuoteErrors]  = useState<Record<string, string>>({});
  const [historicalData, setHistoricalData] = useState<Record<string, HistoricalPoint[]>>({});
  const [horizons,     setHorizons]     = useState<number[]>([1, 5, 10]);
  const [posOverrides, setPosOverrides] = useState<Record<string, { ratePct: string; contrib: string; includeDivs: boolean }>>({});
  const [mktConverted,  setMktConverted]  = useState<Record<string, boolean>>({});
  const [projExcluded,  setProjExcluded]  = useState<Set<string>>(new Set());
  const [fxRate,        setFxRate]        = useState(1.36);
  const [baseCurrency,  setBaseCurrency]  = useState<BaseCurrency>('USD');
  const [isShared,      setIsShared]      = useState(false);
  const [shareMsg,     setShareMsg]     = useState('');
  const [isMobile,     setIsMobile]     = useState(false);
  const [sort,           setSort]           = useState<{ key: SortKey; dir: SortDir }>({ key: 'mktValue', dir: 'desc' });
  const [activeBench,    setActiveBench]    = useState<string[]>(['SPY']);
  const [benchChartData, setBenchChartData] = useState<Record<string, number | string>[]>([]);
  const [benchLoading,   setBenchLoading]   = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [form,      setForm]      = useState({ symbol: '', quantity: '', avgPrice: '' });
  const [formErr,   setFormErr]   = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editForm,  setEditForm]  = useState({ symbol: '', quantity: '', avgPrice: '' });

  useEffect(() => {
    const check = () => setIsMobile(window.innerWidth < 640);
    check();
    window.addEventListener('resize', check);
    return () => window.removeEventListener('resize', check);
  }, []);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const shareParam = params.get('share');
    if (shareParam) {
      try {
        const decoded = JSON.parse(atob(shareParam));
        if (Array.isArray(decoded)) { setHoldings(decoded); setIsShared(true); return; }
      } catch {}
    }
    setHoldings(loadHoldings());
    const settings = loadSettings();
    if (settings.baseCurrency === 'USD' || settings.baseCurrency === 'CAD') setBaseCurrency(settings.baseCurrency);
    if (Array.isArray(settings.horizons) && settings.horizons.length) setHorizons(settings.horizons);
    if (settings.posOverrides && typeof settings.posOverrides === 'object') setPosOverrides(settings.posOverrides);
    if (Array.isArray(settings.projExcluded)) setProjExcluded(new Set(settings.projExcluded));
  }, []);

  useEffect(() => {
    if (isShared) return;
    saveSettings({
      baseCurrency,
      horizons,
      posOverrides,
      projExcluded: Array.from(projExcluded),
    });
  }, [baseCurrency, horizons, posOverrides, projExcluded, isShared]);

  useEffect(() => {
    fetch('/api/portfolio-quote?symbols=USDCAD%3DX')
      .then(r => r.json())
      .then(d => {
        const q = (d.quotes ?? []).find((q: Quote) => q.symbol === 'USDCAD=X');
        if (q?.price && q.price > 0) setFxRate(q.price);
      })
      .catch(() => {});
  }, []);

  const fetchQuotes = useCallback(async (syms: string[]) => {
    if (!syms.length) return;
    setQuoteLoading(true);
    try {
      const res  = await fetch(`/api/portfolio-quote?symbols=${syms.join(',')}`);
      if (!res.ok) throw new Error(`Quote request failed (${res.status})`);
      const data = await res.json();
      const map: Record<string, Quote> = {};
      const errors: Record<string, string> = {};
      (data.quotes ?? []).forEach((q: Quote) => { map[q.symbol] = q; });
      syms.forEach(sym => {
        if (!map[sym] || map[sym].price == null) errors[sym] = 'Price unavailable';
      });
      setQuotes(prev => ({ ...prev, ...map }));
      setQuoteErrors(prev => {
        const next = { ...prev };
        syms.forEach(sym => { delete next[sym]; });
        return { ...next, ...errors };
      });
    } catch {
      const errors: Record<string, string> = {};
      syms.forEach(sym => { errors[sym] = 'Quote request failed'; });
      setQuoteErrors(prev => ({ ...prev, ...errors }));
    } finally { setQuoteLoading(false); }
  }, []);

  const loadProjections = useCallback(async (syms: string[]) => {
    if (!syms.length) return;
    setProjLoading(true);
    const init: Record<string, Projection> = {};
    syms.forEach(s => { init[s] = { cagr: null, loading: true, error: null }; });
    setProjections(init);
    await Promise.allSettled(syms.map(async (sym) => {
      try {
        const res  = await fetch(`/api/stock?symbol=${sym}&days=1825&interval=1mo`);
        const data = await res.json();
        const pts  = (data.data ?? []) as { close: number }[];
        if (pts.length >= 6) {
          const years = (pts.length - 1) / 12;
          const cagr  = Math.pow(pts[pts.length - 1].close / pts[0].close, 1 / years) - 1;
          setProjections(prev => ({ ...prev, [sym]: { cagr, loading: false, error: null } }));
        } else {
          setProjections(prev => ({ ...prev, [sym]: { cagr: null, loading: false, error: 'Not enough history' } }));
        }
      } catch {
        setProjections(prev => ({ ...prev, [sym]: { cagr: null, loading: false, error: 'Projection request failed' } }));
      }
    }));
    setProjLoading(false);
  }, []);

  const loadHistoricalData = useCallback(async (syms: string[]) => {
    const targets = Array.from(new Set([...syms, 'SPY']));
    if (!targets.length) return;
    setRiskLoading(true);
    await Promise.allSettled(targets.map(async (sym) => {
      try {
        const res = await fetch(`/api/stock?symbol=${sym}&days=756&interval=1d`);
        if (!res.ok) throw new Error('history failed');
        const data = await res.json();
        setHistoricalData(prev => ({ ...prev, [sym]: (data.data ?? []) as HistoricalPoint[] }));
      } catch {
        setHistoricalData(prev => ({ ...prev, [sym]: [] }));
      }
    }));
    setRiskLoading(false);
  }, []);

  useEffect(() => {
    const syms = Array.from(new Set(holdings.map(h => h.symbol)));
    if (syms.length) {
      fetchQuotes(syms);
      loadProjections(syms);
      loadHistoricalData(syms);
    }
  }, [holdings, fetchQuotes, loadProjections, loadHistoricalData]);

  const loadBenchChart = useCallback(async (
    holdingSymbols: string[],
    weights: Record<string, number>,
    benchmarks: string[],
  ) => {
    if (!holdingSymbols.length) return;
    setBenchLoading(true);
    const days = 365;
    const allSymbols = Array.from(new Set([...holdingSymbols, ...benchmarks]));

    const results = await Promise.allSettled(
      allSymbols.map(sym =>
        fetch(`/api/stock?symbol=${sym}&days=${days}&interval=1mo`)
          .then(r => r.json())
          .then(d => ({ symbol: sym, data: (d.data ?? []) as { date: string; close: number }[] }))
      )
    );

    const seriesMap: Record<string, { date: string; close: number }[]> = {};
    results.forEach(r => {
      if (r.status === 'fulfilled' && r.value.data.length > 1)
        seriesMap[r.value.symbol] = r.value.data.sort((a, b) => a.date.localeCompare(b.date));
    });

    const holdingSeries = holdingSymbols.map(s => seriesMap[s]).filter(Boolean);
    if (!holdingSeries.length) { setBenchLoading(false); return; }
    const minLen = Math.min(...holdingSeries.map(s => s.length));
    if (minLen < 2) { setBenchLoading(false); return; }

    const normalized: Record<string, number[]> = {};
    for (const sym of allSymbols) {
      const s = seriesMap[sym];
      if (!s || s.length < minLen) continue;
      const trimmed = s.slice(-minLen);
      const base = trimmed[0].close;
      if (base <= 0) continue;
      normalized[sym] = trimmed.map(p => (p.close / base) * 100);
    }

    const refDates = holdingSeries[0].slice(-minLen).map(p => p.date.slice(0, 10));
    const portfolioSeries = Array.from({ length: minLen }, (_, i) =>
      holdingSymbols.reduce((sum, sym) => {
        const w = weights[sym] ?? 0;
        const n = normalized[sym];
        return n ? sum + w * n[i] : sum;
      }, 0)
    );

    setBenchChartData(refDates.map((date, i) => {
      const pt: Record<string, number | string> = { date };
      if (portfolioSeries[i] > 0) pt['Portfolio'] = parseFloat(portfolioSeries[i].toFixed(2));
      benchmarks.forEach(b => {
        if (normalized[b]?.[i] != null) pt[b] = parseFloat(normalized[b][i].toFixed(2));
      });
      return pt;
    }));
    setBenchLoading(false);
  }, []);

  useEffect(() => {
    const holdingSymbols = Array.from(new Set(holdings.map(h => h.symbol)));
    if (!holdingSymbols.length || !Object.keys(quotes).length) return;
    const totalVal = holdings.reduce((s, h) => {
      const price = quotes[h.symbol]?.price ?? h.avgPrice;
      return s + h.quantity * price;
    }, 0);
    const weights: Record<string, number> = {};
    holdings.forEach(h => {
      const price = quotes[h.symbol]?.price ?? h.avgPrice;
      const val = h.quantity * price;
      weights[h.symbol] = (weights[h.symbol] ?? 0) + (totalVal > 0 ? val / totalVal : 1 / holdings.length);
    });
    loadBenchChart(holdingSymbols, weights, activeBench);
  }, [holdings, quotes, activeBench, loadBenchChart]);

  function addHolding() {
    const sym   = form.symbol.trim().toUpperCase();
    const qty   = parseFloat(form.quantity);
    const price = parseFloat(form.avgPrice);
    if (!sym)         { setFormErr('Enter a ticker symbol'); return; }
    if (!(qty > 0))   { setFormErr('Enter a valid share count'); return; }
    if (!(price > 0)) { setFormErr('Enter a valid avg price'); return; }
    const next = [...holdings, { id: `${sym}_${Date.now()}`, symbol: sym, quantity: qty, avgPrice: price }];
    setHoldings(next);
    saveHoldings(next);
    setForm({ symbol: '', quantity: '', avgPrice: '' });
    setFormErr('');
  }

  function removeHolding(id: string) {
    const next = holdings.filter(h => h.id !== id);
    setHoldings(next);
    saveHoldings(next);
  }

  function startEdit(h: Holding) {
    setEditingId(h.id);
    setEditForm({ symbol: h.symbol, quantity: String(h.quantity), avgPrice: String(h.avgPrice) });
  }

  function saveEdit() {
    if (!editingId) return;
    const sym   = editForm.symbol.trim().toUpperCase();
    const qty   = parseFloat(editForm.quantity);
    const price = parseFloat(editForm.avgPrice);
    if (!sym || !(qty > 0) || !(price > 0)) return;
    const next = holdings.map(h =>
      h.id === editingId ? { ...h, symbol: sym, quantity: qty, avgPrice: price } : h
    );
    setHoldings(next);
    saveHoldings(next);
    setEditingId(null);
    const syms = Array.from(new Set(next.map(h => h.symbol)));
    fetchQuotes(syms);
    loadProjections(syms);
    loadHistoricalData(syms);
  }

  function convertToBase(h: Holding, raw: number) {
    const native = nativeCurrency(h.symbol);
    if (native === baseCurrency) return raw;
    return native === 'USD' ? raw * fxRate : raw / fxRate;
  }

  function exportCsv() {
    const header = ['symbol', 'quantity', 'avgPrice'];
    const body = holdings.map(h => [h.symbol, h.quantity, h.avgPrice].map(csvEscape).join(','));
    const blob = new Blob([[header.join(','), ...body].join('\n')], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'portfolio-holdings.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  function importCsv(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? '');
      const lines = text.split(/\r?\n/).map(l => l.trim()).filter(Boolean);
      if (!lines.length) return;
      const start = lines[0].toLowerCase().includes('symbol') ? 1 : 0;
      const imported: Holding[] = lines.slice(start).map((line, idx) => {
        const parts = line.match(/("([^"]|"")*"|[^,]+)/g)?.map(p => p.replace(/^"|"$/g, '').replace(/""/g, '"').trim()) ?? [];
        const symbol = (parts[0] ?? '').toUpperCase();
        const quantity = parseFloat(parts[1] ?? '');
        const avgPrice = parseFloat(parts[2] ?? '');
        if (!symbol || !(quantity > 0) || !(avgPrice > 0)) return null;
        return { id: `${symbol}_${Date.now()}_${idx}`, symbol, quantity, avgPrice };
      }).filter(Boolean) as Holding[];

      if (!imported.length) {
        setFormErr('No valid rows found. Use symbol, quantity, avgPrice.');
        return;
      }

      const next = [...holdings, ...imported];
      setHoldings(next);
      saveHoldings(next);
      setFormErr(`Imported ${imported.length} position${imported.length === 1 ? '' : 's'}`);
    };
    reader.readAsText(file);
  }

  const rows = holdings.map(h => {
    const q         = quotes[h.symbol];
    const price     = q?.price ?? null;
    const costBasis = h.quantity * h.avgPrice;
    const mktValue  = price !== null ? h.quantity * price : null;
    const baseCost  = convertToBase(h, costBasis);
    const baseValue = convertToBase(h, mktValue ?? costBasis);
    const gl        = mktValue !== null ? mktValue - costBasis : null;
    const baseGL    = mktValue !== null ? baseValue - baseCost : null;
    const glPct     = baseGL !== null && baseCost > 0 ? (baseGL / baseCost) * 100 : null;
    const divRate   = q?.dividendRate ?? (q?.dividendYield != null && price != null ? q.dividendYield * price : null);
    const annDiv    = divRate != null ? convertToBase(h, h.quantity * divRate) : null;
    const quoteError = quoteErrors[h.symbol] ?? null;
    return { h, q, price, costBasis, mktValue, baseCost, baseValue, gl, baseGL, glPct, annDiv, quoteError };
  });

  function convertedMktVal(h: Holding, raw: number): number {
    if (!mktConverted[h.id]) return raw;
    return h.symbol.endsWith('.TO') ? raw / fxRate : raw * fxRate;
  }

  const totalInvested = rows.reduce((s, r) => s + r.baseCost, 0);
  const totalValue    = rows.reduce((s, r) => s + r.baseValue, 0);
  const totalGL       = totalValue - totalInvested;
  const totalGLPct    = totalInvested > 0 ? (totalGL / totalInvested) * 100 : 0;
  const totalAnnDiv   = rows.reduce((s, r) => s + (r.annDiv ?? 0), 0);
  const riskMetrics: RiskMetrics = (() => {
    const weights = rows
      .filter(row => row.baseValue > 0)
      .map(row => ({ symbol: row.h.symbol, weight: totalValue > 0 ? row.baseValue / totalValue : 0 }));
    const spyReturns = returnsByDate(historicalData.SPY ?? []);
    const symbolReturns = Object.fromEntries(weights.map(w => [w.symbol, returnsByDate(historicalData[w.symbol] ?? [])]));
    const dates = Object.keys(spyReturns).sort();
    const portfolioReturns: number[] = [];
    const benchmarkReturns: number[] = [];

    dates.forEach(date => {
      let weightedReturn = 0;
      let includedWeight = 0;
      weights.forEach(({ symbol, weight }) => {
        const r = symbolReturns[symbol][date];
        if (typeof r === 'number') {
          weightedReturn += weight * r;
          includedWeight += weight;
        }
      });
      if (includedWeight > 0.5) {
        portfolioReturns.push(weightedReturn / includedWeight);
        benchmarkReturns.push(spyReturns[date]);
      }
    });

    const dailyVol = stdev(portfolioReturns);
    const benchVol = stdev(benchmarkReturns);
    const cov = covariance(portfolioReturns, benchmarkReturns);
    const beta = cov != null && benchVol != null && benchVol > 0 ? cov / Math.pow(benchVol, 2) : null;
    const volatility = dailyVol != null ? dailyVol * Math.sqrt(252) : null;
    const annualReturn = mean(portfolioReturns) * 252;
    const sharpe = volatility != null && volatility > 0 ? (annualReturn - 0.02) / volatility : null;

    return {
      beta,
      volatility,
      maxDrawdown: portfolioReturns.length ? maxDrawdownFromReturns(portfolioReturns) : null,
      sharpe,
      points: portfolioReturns.length,
    };
  })();
  const dividendCalendar = rows
    .filter(row => row.annDiv != null && row.annDiv > 0 && row.q?.nextDividendDate)
    .map(row => {
      const paymentsPerYear = Math.max(1, Math.round(12 / (row.q?.dividendFrequencyMonths ?? 3)));
      return {
        symbol: row.h.symbol,
        date: row.q!.nextDividendDate!,
        amount: row.q!.dividendRate != null ? convertToBase(row.h, row.h.quantity * row.q!.dividendRate / paymentsPerYear) : null,
        growth: row.q!.dividendGrowth ?? null,
        frequency: row.q!.dividendFrequencyMonths ?? null,
      };
    })
    .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  const dividendGrowthRows = rows
    .filter(row => row.annDiv != null && row.annDiv > 0)
    .map(row => ({
      symbol: row.h.symbol,
      yield: row.q?.dividendYield ?? null,
      annual: row.annDiv,
      growth: row.q?.dividendGrowth ?? null,
      priorAnnual: row.q?.priorDividendRate != null ? convertToBase(row.h, row.h.quantity * row.q.priorDividendRate) : null,
    }))
    .sort((a, b) => (b.annual ?? 0) - (a.annual ?? 0));
  const visibleRows = [...rows].sort((a, b) => {
    const dir = sort.dir === 'asc' ? 1 : -1;
    const get = (row: typeof rows[number]) => {
      switch (sort.key) {
        case 'symbol': return row.h.symbol;
        case 'name': return row.q?.name ?? '';
        case 'quantity': return row.h.quantity;
        case 'avgPrice': return row.h.avgPrice;
        case 'price': return row.price ?? -Infinity;
        case 'mktValue': return row.baseValue;
        case 'gl': return row.baseGL ?? -Infinity;
        case 'glPct': return row.glPct ?? -Infinity;
        case 'weight': return totalValue > 0 ? row.baseValue / totalValue : 0;
        default: return row.baseValue;
      }
    };
    const av = get(a);
    const bv = get(b);
    if (typeof av === 'string' || typeof bv === 'string') return String(av).localeCompare(String(bv)) * dir;
    return ((av as number) - (bv as number)) * dir;
  }).map(row => ({ ...row, weight: totalValue > 0 ? (row.baseValue / totalValue) * 100 : 0 }));

  const glColor = (v: number | null) => v == null ? 'var(--text-3)' : v >= 0 ? 'var(--green-2)' : 'var(--red-2)';

  function toggleProjExclude(sym: string) {
    setProjExcluded(prev => {
      const s = new Set(prev);
      s.has(sym) ? s.delete(sym) : s.add(sym);
      return s;
    });
  }

  // ── Shared currency toggle cell renderer ─────────────────────────────────
  function renderMktVal(h: Holding, mktValue: number | null, costBasis: number) {
    const native  = h.symbol.endsWith('.TO') ? 'CAD' : 'USD';
    const other   = native === 'USD' ? 'CAD' : 'USD';
    const raw     = mktValue ?? costBasis;
    const conv    = mktConverted[h.id];
    const display = conv ? (native === 'USD' ? raw * fxRate : raw / fxRate) : raw;
    const cur     = conv ? other : native;
    const formatted = new Intl.NumberFormat('en-US', {
      style: 'currency', currency: cur, minimumFractionDigits: 2, maximumFractionDigits: 2,
    }).format(display);
    return (
      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
        <span style={{ fontWeight: 600 }}>{formatted}</span>
        <button
          onClick={() => setMktConverted(prev => ({ ...prev, [h.id]: !prev[h.id] }))}
          title={`Show in ${conv ? native : other}`}
          style={{
            fontSize: 8, padding: '1px 4px', border: '1px solid var(--bg-1)',
            background: conv ? 'var(--accent)' : 'transparent',
            color: conv ? 'var(--text-0)' : 'var(--text-5)',
            cursor: 'pointer', fontFamily: 'inherit', lineHeight: 1.4, flexShrink: 0,
          }}
        >{conv ? other : `→${other}`}</button>
      </div>
    );
  }

  function renderGL(h: Holding, gl: number | null) {
    if (gl == null) return <span style={{ color: 'var(--text-5)' }}>—</span>;
    const conv   = mktConverted[h.id];
    const native = h.symbol.endsWith('.TO') ? 'CAD' : 'USD';
    const other  = native === 'USD' ? 'CAD' : 'USD';
    const cur    = conv ? other : native;
    const display = convertedMktVal(h, gl);
    return (
      <span style={{ color: glColor(gl), fontWeight: 600 }}>
        {(gl >= 0 ? '+' : '') + new Intl.NumberFormat('en-US', {
          style: 'currency', currency: cur, minimumFractionDigits: 2, maximumFractionDigits: 2,
        }).format(display)}
      </span>
    );
  }

  return (
    <div className="portfolio-page" style={{ minHeight: '100vh', background: 'var(--bg-4)', color: 'var(--text-2)', fontFamily: 'inherit' }}>
      <div style={{ padding: isMobile ? '12px 14px' : '16px 24px' }}>

        {/* ── Nav ── */}
        <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: isMobile ? 8 : 16, marginBottom: 20 }}>
          <Link href="/" style={{
            display: 'flex', alignItems: 'center', gap: 6, fontSize: 12,
            color: 'var(--text-4)', textDecoration: 'none',
            border: '2px solid var(--bg-1)', padding: '6px 12px',
            background: 'rgba(0,0,0,0.2)',
          }}>
            <ArrowLeft size={13} /> Back
          </Link>
          <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Portfolio Tracker
          </span>
          <div style={{ marginLeft: isMobile ? 0 : 'auto', width: isMobile ? '100%' : 'auto', display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            {isShared && (
              <span style={{ fontSize: 11, color: 'var(--yellow-2)', border: '1px solid var(--yellow-2)', padding: '4px 10px', letterSpacing: '0.06em' }}>
                READ-ONLY VIEW
              </span>
            )}
            {!isShared && holdings.length > 0 && (
              <button
                className="btn-secondary"
                style={{ padding: '6px 14px', fontSize: 12 }}
                onClick={async () => {
                  const encoded = btoa(JSON.stringify(holdings));
                  const url = `${window.location.origin}${window.location.pathname}?share=${encoded}`;
                  try { await navigator.clipboard.writeText(url); } catch { prompt('Copy this link:', url); }
                  setShareMsg('Link copied!');
                  setTimeout(() => setShareMsg(''), 2500);
                }}
              >
                {shareMsg || 'Share Portfolio'}
              </button>
            )}
            {!isShared && (
              <>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,text/csv"
                  style={{ display: 'none' }}
                  onChange={e => {
                    const file = e.target.files?.[0];
                    if (file) importCsv(file);
                    e.currentTarget.value = '';
                  }}
                />
                <button
                  className="btn-secondary"
                  style={{ padding: '6px 10px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}
                  onClick={() => fileInputRef.current?.click()}
                  title="Import CSV"
                >
                  <Upload size={12} /> Import
                </button>
                {holdings.length > 0 && (
                  <button
                    className="btn-secondary"
                    style={{ padding: '6px 10px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}
                    onClick={exportCsv}
                    title="Export CSV"
                  >
                    <Download size={12} /> Export
                  </button>
                )}
              </>
            )}
            <div style={{ display: 'flex', border: '1px solid var(--bg-1)' }}>
              {(['USD', 'CAD'] as BaseCurrency[]).map(cur => (
                <button
                  key={cur}
                  onClick={() => setBaseCurrency(cur)}
                  style={{
                    padding: '5px 9px',
                    fontSize: 11,
                    border: 'none',
                    cursor: 'pointer',
                    background: baseCurrency === cur ? 'var(--accent)' : 'transparent',
                    color: baseCurrency === cur ? 'var(--text-0)' : 'var(--text-4)',
                    fontFamily: 'inherit',
                  }}
                  title={`Show portfolio totals in ${cur}`}
                >
                  {cur}
                </button>
              ))}
            </div>
            <button
              className="btn-secondary"
              style={{ padding: '6px 14px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}
              onClick={() => fetchQuotes(Array.from(new Set(holdings.map(h => h.symbol))))}
            >
              <RefreshCw size={12} style={{ animation: quoteLoading ? 'spin 1s linear infinite' : 'none' }} />
              Refresh Prices
            </button>
          </div>
        </div>

        {/* ── Summary cards ── */}
        <div style={{ display: 'grid', gridTemplateColumns: isMobile ? 'repeat(2, 1fr)' : 'repeat(4, 1fr)', gap: isMobile ? 8 : 12, marginBottom: 16 }}>
          {[
            { label: `Invested (${baseCurrency})`,  value: money(totalInvested, baseCurrency), color: 'var(--text-1)' },
            { label: `Value (${baseCurrency})`,   value: money(totalValue, baseCurrency),    color: 'var(--text-1)' },
            { label: 'Total Gain/Loss', value: (totalGL >= 0 ? '+' : '') + money(totalGL, baseCurrency), color: glColor(totalGL) },
            { label: 'Return',          value: pct(totalGLPct), color: glColor(totalGLPct) },
          ].map(c => (
            <div key={c.label} className="card" style={{ padding: isMobile ? '14px 12px' : '18px 20px' }}>
              <span className="card-label">{c.label}</span>
              <div style={{ fontSize: isMobile ? 16 : 22, fontWeight: 700, color: c.color, lineHeight: 1.2, marginTop: 6 }}>
                {c.value}
              </div>
            </div>
          ))}
        </div>

        {holdings.length > 0 && (
          <div className="card" style={{ marginBottom: 16, padding: isMobile ? '14px 12px' : '18px 20px' }}>
            <span className="card-label">Risk Metrics</span>
            <div style={{ display: 'grid', gridTemplateColumns: isMobile ? 'repeat(2, 1fr)' : 'repeat(5, 1fr)', gap: 12, marginTop: 4 }}>
              {[
                { label: 'Beta vs SPY', value: riskMetrics.beta != null ? riskMetrics.beta.toFixed(2) : '—', color: riskMetrics.beta != null && riskMetrics.beta > 1.25 ? 'var(--yellow-2)' : 'var(--text-1)' },
                { label: 'Volatility', value: riskMetrics.volatility != null ? `${(riskMetrics.volatility * 100).toFixed(1)}%` : '—', color: riskMetrics.volatility != null && riskMetrics.volatility > 0.30 ? 'var(--yellow-2)' : 'var(--text-1)' },
                { label: 'Max Drawdown', value: riskMetrics.maxDrawdown != null ? `${(riskMetrics.maxDrawdown * 100).toFixed(1)}%` : '—', color: riskMetrics.maxDrawdown != null && riskMetrics.maxDrawdown < -0.25 ? 'var(--red-2)' : 'var(--text-1)' },
                { label: 'Sharpe', value: riskMetrics.sharpe != null ? riskMetrics.sharpe.toFixed(2) : '—', color: riskMetrics.sharpe != null && riskMetrics.sharpe < 0.5 ? 'var(--yellow-2)' : 'var(--text-1)' },
                { label: 'History', value: riskLoading ? 'Loading…' : `${riskMetrics.points} days`, color: 'var(--text-3)' },
              ].map(item => (
                <div key={item.label} style={{ background: 'var(--bg-3)', padding: '10px 12px' }}>
                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>{item.label}</div>
                  <div style={{ fontSize: isMobile ? 16 : 19, fontWeight: 700, color: item.color }}>{item.value}</div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 10, fontSize: 10, color: 'var(--text-5)', lineHeight: 1.5 }}>
              Metrics use current position weights, daily returns, SPY as benchmark, and a 2% annual risk-free assumption.
            </div>
          </div>
        )}

        {/* ── Holdings card ── */}
        <div className="card" style={{ marginBottom: 16 }}>
          <span className="card-label">Holdings</span>

          {/* Add form */}
          {!isShared && (
            <div style={{ display: 'flex', flexDirection: isMobile ? 'column' : 'row', gap: 8, alignItems: isMobile ? 'stretch' : 'flex-end', marginBottom: 20 }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                <span style={{ fontSize: 10, color: 'var(--text-5)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Ticker</span>
                <TickerInput
                  value={form.symbol}
                  onChange={v => setForm(f => ({ ...f, symbol: v }))}
                  onSelect={(sym) => setForm(f => ({ ...f, symbol: sym }))}
                  inputWidth={isMobile ? '100%' : 180}
                />
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: isMobile ? 1 : 'none' }}>
                  <span style={{ fontSize: 10, color: 'var(--text-5)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Shares</span>
                  <input
                    type="number"
                    placeholder="10"
                    min="0"
                    value={form.quantity}
                    onChange={e => setForm(f => ({ ...f, quantity: e.target.value }))}
                    onKeyDown={e => e.key === 'Enter' && addHolding()}
                    style={{ width: isMobile ? '100%' : 90 }}
                  />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4, flex: isMobile ? 1 : 'none' }}>
                  <span style={{ fontSize: 10, color: 'var(--text-5)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Avg Cost ($)</span>
                  <input
                    type="number"
                    placeholder="180.00"
                    min="0"
                    step="0.01"
                    value={form.avgPrice}
                    onChange={e => setForm(f => ({ ...f, avgPrice: e.target.value }))}
                    onKeyDown={e => e.key === 'Enter' && addHolding()}
                    style={{ width: isMobile ? '100%' : 110 }}
                  />
                </div>
              </div>
              <button
                onClick={addHolding}
                style={{
                  padding: '12px 18px', fontFamily: 'inherit', fontWeight: 500,
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
                  background: 'var(--accent)', color: 'var(--text-0)',
                  border: '2px solid var(--accent)', cursor: 'pointer',
                  width: isMobile ? '100%' : 'auto',
                }}
              >
                <Plus size={13} /> Add Position
              </button>
              {formErr && <span style={{ fontSize: 11, color: 'var(--red-2)' }}>{formErr}</span>}
            </div>
          )}


          {holdings.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-5)', fontSize: 13, lineHeight: 1.6 }}>
              <div style={{ color: 'var(--text-3)', fontWeight: 700, marginBottom: 4 }}>No positions yet</div>
              Add a ticker above or import a CSV with symbol, quantity, and avgPrice columns.
            </div>
          ) : quoteLoading && Object.keys(quotes).length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-5)', fontSize: 13, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
              <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
              Fetching prices…
            </div>
          ) : isMobile ? (

            /* ── MOBILE: card-per-holding ── */
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {visibleRows.map(({ h, q, price, costBasis, mktValue, gl, glPct, weight, quoteError }) => {
                const isEditing = editingId === h.id;

                if (isEditing) return (
                  <div key={h.id} style={{ background: 'var(--bg-3)', padding: 14, border: '1px solid var(--accent)' }}>
                    <div style={{ marginBottom: 8 }}>
                      <div style={{ fontSize: 10, color: 'var(--text-5)', textTransform: 'uppercase', marginBottom: 4 }}>Ticker</div>
                      <TickerInput
                        value={editForm.symbol}
                        onChange={v => setEditForm(f => ({ ...f, symbol: v }))}
                        onSelect={sym => setEditForm(f => ({ ...f, symbol: sym }))}
                        inputWidth="100%"
                      />
                    </div>
                    <div style={{ display: 'flex', gap: 8, marginBottom: 10 }}>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 10, color: 'var(--text-5)', textTransform: 'uppercase', marginBottom: 4 }}>Shares</div>
                        <input
                          type="number"
                          value={editForm.quantity}
                          onChange={e => setEditForm(f => ({ ...f, quantity: e.target.value }))}
                          onKeyDown={e => { if (e.key === 'Enter') saveEdit(); if (e.key === 'Escape') setEditingId(null); }}
                          style={{ width: '100%', padding: '6px 8px', fontSize: 13 }}
                        />
                      </div>
                      <div style={{ flex: 1 }}>
                        <div style={{ fontSize: 10, color: 'var(--text-5)', textTransform: 'uppercase', marginBottom: 4 }}>Avg Cost ($)</div>
                        <input
                          type="number"
                          value={editForm.avgPrice}
                          onChange={e => setEditForm(f => ({ ...f, avgPrice: e.target.value }))}
                          onKeyDown={e => { if (e.key === 'Enter') saveEdit(); if (e.key === 'Escape') setEditingId(null); }}
                          style={{ width: '100%', padding: '6px 8px', fontSize: 13 }}
                          step="0.01"
                        />
                      </div>
                    </div>
                    <div style={{ display: 'flex', gap: 8 }}>
                      <button onClick={saveEdit} style={{ flex: 1, padding: '8px', background: 'var(--green-2)', color: 'var(--text-0)', border: 'none', cursor: 'pointer', fontFamily: 'inherit', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, fontSize: 12 }}>
                        <Check size={13} /> Save
                      </button>
                      <button onClick={() => setEditingId(null)} style={{ flex: 1, padding: '8px', background: 'var(--bg-1)', color: 'var(--text-3)', border: 'none', cursor: 'pointer', fontFamily: 'inherit', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5, fontSize: 12 }}>
                        <X size={13} /> Cancel
                      </button>
                    </div>
                  </div>
                );

                return (
                  <div key={h.id} style={{ background: 'var(--bg-3)', padding: '12px 14px', borderLeft: '2px solid var(--accent)' }}>
                    {/* Header: symbol + name + actions */}
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10 }}>
                      <div>
                        <div style={{ fontWeight: 700, color: 'var(--accent)', fontSize: 15, letterSpacing: '0.04em' }}>{h.symbol}</div>
                        {q?.name && <div style={{ fontSize: 11, color: 'var(--text-5)', marginTop: 1 }}>{q.name}</div>}
                      </div>
                      {!isShared && (
                        <div style={{ display: 'flex', gap: 8 }}>
                          <button onClick={() => startEdit(h)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-4)', padding: 4 }}>
                            <Pencil size={14} />
                          </button>
                          <button onClick={() => removeHolding(h.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-4)', padding: 4 }}>
                            <Trash2 size={14} />
                          </button>
                        </div>
                      )}
                    </div>

                    {/* Stats grid */}
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px' }}>
                      <div>
                        <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>Shares</div>
                        <div style={{ fontSize: 13, color: 'var(--text-2)' }}>{h.quantity.toLocaleString()}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>Avg Cost</div>
                        <div style={{ fontSize: 13, color: 'var(--text-3)' }}>{usd(h.avgPrice)}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>Price</div>
                        <div style={{ fontSize: 13, color: 'var(--text-2)' }}>{price != null ? usd(price) : <span style={{ color: 'var(--text-5)' }}>—</span>}</div>
                        {quoteError && <div style={{ fontSize: 9, color: 'var(--yellow-2)', marginTop: 2 }}>{quoteError}</div>}
                      </div>
                      <div>
                        <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>Mkt Value</div>
                        <div style={{ fontSize: 13 }}>{renderMktVal(h, mktValue, costBasis)}</div>
                      </div>
                    </div>

                    {/* Gain/loss full-width */}
                    <div style={{ marginTop: 10, paddingTop: 10, borderTop: '1px solid var(--bg-1)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Gain / Loss</span>
                      <div style={{ display: 'flex', gap: 10, alignItems: 'baseline' }}>
                        <span style={{ fontSize: 11, color: 'var(--text-5)' }}>{weight.toFixed(1)}%</span>
                        <span style={{ fontSize: 14 }}>{renderGL(h, gl)}</span>
                        <span style={{ fontSize: 12, fontWeight: 700, color: glColor(glPct) }}>{glPct != null ? pct(glPct) : '—'}</span>
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Mobile totals */}
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '12px 14px', borderTop: '2px solid var(--bg-1)', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px 16px' }}>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>Invested</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-2)' }}>{money(totalInvested, baseCurrency)}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>Value</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-1)' }}>{money(totalValue, baseCurrency)}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>Gain / Loss</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: glColor(totalGL) }}>{(totalGL >= 0 ? '+' : '') + money(totalGL, baseCurrency)}</div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 2 }}>Return</div>
                  <div style={{ fontSize: 14, fontWeight: 700, color: glColor(totalGLPct) }}>{pct(totalGLPct)}</div>
                </div>
              </div>
            </div>

          ) : (

            /* ── DESKTOP: table ── */
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr>
                    {[
                      ['Ticker', 'symbol'],
                      ['Name', 'name'],
                      ['Shares', 'quantity'],
                      ['Avg Cost', 'avgPrice'],
                      ['Current Price', 'price'],
                      [`Mkt Value (${baseCurrency})`, 'mktValue'],
                      ['Weight', 'weight'],
                      ['Gain / Loss', 'gl'],
                      ['%', 'glPct'],
                    ].map(([h, key], i) => (
                      <th key={h} onClick={() => setSort(prev => prev.key === key ? { key: key as SortKey, dir: prev.dir === 'asc' ? 'desc' : 'asc' } : { key: key as SortKey, dir: 'desc' })} style={{
                        padding: '8px 12px', fontSize: 10, fontWeight: 600, letterSpacing: '0.08em',
                        textTransform: 'uppercase', color: 'var(--text-5)',
                        textAlign: i <= 1 ? 'left' : 'right',
                        borderBottom: '1px solid var(--bg-1)',
                        whiteSpace: 'nowrap',
                        cursor: 'pointer',
                      }}>
                        {h}{sort.key === key ? (sort.dir === 'asc' ? ' ↑' : ' ↓') : ''}
                      </th>
                    ))}
                    <th style={{ width: 32, borderBottom: '1px solid var(--bg-1)' }} />
                  </tr>
                </thead>
                <tbody>
                  {visibleRows.map(({ h, q, price, costBasis, mktValue, baseValue, baseGL, gl, glPct, weight, quoteError }) => {
                    const isEditing = editingId === h.id;
                    if (isEditing) return (
                      <tr key={h.id} style={{ borderBottom: '1px solid var(--bg-3)', background: 'rgba(0,0,0,0.2)' }}>
                        <td style={{ padding: '6px 8px' }}>
                          <TickerInput
                            value={editForm.symbol}
                            onChange={v => setEditForm(f => ({ ...f, symbol: v }))}
                            onSelect={sym => setEditForm(f => ({ ...f, symbol: sym }))}
                          />
                        </td>
                        <td style={{ padding: '6px 8px', fontSize: 11, color: 'var(--text-5)' }}>{q?.name ?? '—'}</td>
                        <td style={{ padding: '6px 8px', textAlign: 'right' }}>
                          <input type="number" value={editForm.quantity} onChange={e => setEditForm(f => ({ ...f, quantity: e.target.value }))} onKeyDown={e => { if (e.key === 'Enter') saveEdit(); if (e.key === 'Escape') setEditingId(null); }} style={{ width: 80, padding: '4px 8px', fontSize: 12, textAlign: 'right' }} />
                        </td>
                        <td style={{ padding: '6px 8px', textAlign: 'right' }}>
                          <input type="number" value={editForm.avgPrice} onChange={e => setEditForm(f => ({ ...f, avgPrice: e.target.value }))} onKeyDown={e => { if (e.key === 'Enter') saveEdit(); if (e.key === 'Escape') setEditingId(null); }} style={{ width: 90, padding: '4px 8px', fontSize: 12, textAlign: 'right' }} step="0.01" />
                        </td>
                        <td /><td /><td /><td /><td />
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                          <div style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
                            <button onClick={saveEdit} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--green-2)', padding: 2 }}><Check size={14} /></button>
                            <button onClick={() => setEditingId(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--red-2)', padding: 2 }}><X size={14} /></button>
                          </div>
                        </td>
                      </tr>
                    );
                    return (
                      <tr key={h.id} style={{ borderBottom: '1px solid var(--bg-3)' }}>
                        <td style={{ padding: '10px 12px', fontWeight: 700, color: 'var(--accent)', letterSpacing: '0.04em' }}>{h.symbol}</td>
                        <td style={{ padding: '10px 12px', color: 'var(--text-4)', fontSize: 12, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{q?.name ?? '—'}</td>
                        <td style={{ padding: '10px 12px', textAlign: 'right', color: 'var(--text-2)' }}>{h.quantity.toLocaleString()}</td>
                        <td style={{ padding: '10px 12px', textAlign: 'right', color: 'var(--text-3)' }}>{usd(h.avgPrice)}</td>
                        <td style={{ padding: '10px 12px', textAlign: 'right', color: 'var(--text-2)' }}>
                          {price != null ? usd(price) : <span style={{ color: 'var(--text-5)' }}>—</span>}
                          {quoteError && <div style={{ fontSize: 9, color: 'var(--yellow-2)', marginTop: 2 }}>{quoteError}</div>}
                        </td>
                        <td style={{ padding: '10px 12px', textAlign: 'right' }}>
                          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end' }}>
                            {money(baseValue, baseCurrency)}
                          </div>
                        </td>
                        <td style={{ padding: '10px 12px', textAlign: 'right', color: weight > 35 ? 'var(--yellow-2)' : 'var(--text-3)', fontWeight: 600 }}>{weight.toFixed(1)}%</td>
                        <td style={{ padding: '10px 12px', textAlign: 'right', color: glColor(baseGL), fontWeight: 600 }}>{baseGL != null ? `${baseGL >= 0 ? '+' : ''}${money(baseGL, baseCurrency)}` : renderGL(h, gl)}</td>
                        <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: glColor(glPct) }}>{glPct != null ? pct(glPct) : '—'}</td>
                        <td style={{ padding: '10px 8px', textAlign: 'center' }}>
                          {!isShared && <div style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
                            <button onClick={() => startEdit(h)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-5)', padding: 2 }}><Pencil size={12} /></button>
                            <button onClick={() => removeHolding(h.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-5)', padding: 2 }}><Trash2 size={12} /></button>
                          </div>}
                        </td>
                      </tr>
                    );
                  })}
                  <tr style={{ background: 'rgba(0,0,0,0.25)', borderTop: '2px solid var(--bg-1)' }}>
                    <td colSpan={2} style={{ padding: '10px 12px', fontWeight: 700, fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-4)' }}>Total</td>
                    <td style={{ padding: '10px 12px' }} />
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: 'var(--text-3)' }}>{money(totalInvested, baseCurrency)}</td>
                    <td style={{ padding: '10px 12px' }} />
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: 'var(--text-1)' }}>{money(totalValue, baseCurrency)}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: 'var(--text-3)' }}>100.0%</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: glColor(totalGL) }}>{(totalGL >= 0 ? '+' : '') + money(totalGL, baseCurrency)}</td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: glColor(totalGLPct) }}>{pct(totalGLPct)}</td>
                    <td />
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ── Benchmark Comparison ── */}
        {holdings.length > 0 && (
          <div className="card" style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
              <span className="card-label" style={{ margin: 0 }}>vs. Benchmarks</span>
              <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
                {BENCH_OPTIONS.map(b => (
                  <label key={b.symbol} style={{ display: 'flex', gap: 5, alignItems: 'center', fontSize: 11, cursor: 'pointer',
                    color: activeBench.includes(b.symbol) ? BENCH_COLORS[b.symbol] : 'var(--text-5)' }}>
                    <input
                      type="checkbox"
                      checked={activeBench.includes(b.symbol)}
                      onChange={e => setActiveBench(prev => e.target.checked ? [...prev, b.symbol] : prev.filter(s => s !== b.symbol))}
                      style={{ accentColor: BENCH_COLORS[b.symbol], cursor: 'pointer' }}
                    />
                    {b.label}
                  </label>
                ))}
                {benchLoading && <RefreshCw size={12} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />}
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 220px', gap: 16, alignItems: 'start' }}>
              {/* Chart */}
              <div>
                {benchChartData.length > 1 ? (
                  <ResponsiveContainer width="100%" height={240}>
                    <LineChart data={benchChartData} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis
                        dataKey="date"
                        tick={{ fontSize: 10, fill: 'var(--text-5)' }}
                        tickFormatter={d => {
                          const dt = new Date(d + 'T00:00:00');
                          return `${dt.toLocaleString('en-US', { month: 'short' })} '${String(dt.getFullYear()).slice(2)}`;
                        }}
                        interval="preserveStartEnd"
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        tick={{ fontSize: 10, fill: 'var(--text-5)' }}
                        tickFormatter={v => `${v.toFixed(0)}`}
                        width={38}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{ background: 'var(--bg-3)', border: '1px solid var(--bg-1)', fontSize: 11, borderRadius: 0 }}
                        labelStyle={{ color: 'var(--text-3)', marginBottom: 4 }}
                        formatter={(value: number, name: string) => {
                          const label = name === 'Portfolio' ? 'Portfolio' : (BENCH_OPTIONS.find(b => b.symbol === name)?.label ?? name);
                          const delta = value - 100;
                          return [`${delta >= 0 ? '+' : ''}${delta.toFixed(1)}%`, label];
                        }}
                        labelFormatter={d => fmtDate(d + 'T00:00:00')}
                      />
                      <ReferenceLine y={100} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
                      <Line type="monotone" dataKey="Portfolio" stroke="var(--accent)" strokeWidth={2.5} dot={false} />
                      {activeBench.map(sym => (
                        <Line key={sym} type="monotone" dataKey={sym} stroke={BENCH_COLORS[sym]} strokeWidth={1.5} dot={false} strokeOpacity={0.85} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-5)', fontSize: 12 }}>
                    {benchLoading ? 'Loading…' : 'Not enough data'}
                  </div>
                )}
              </div>

              {/* Stats */}
              {benchChartData.length > 1 && (() => {
                const last = benchChartData[benchChartData.length - 1];
                const portfolioReturn = typeof last['Portfolio'] === 'number' ? (last['Portfolio'] as number) - 100 : null;
                if (portfolioReturn == null) return null;

                const comparisons = activeBench.map(sym => {
                  const benchReturn = typeof last[sym] === 'number' ? (last[sym] as number) - 100 : null;
                  if (benchReturn == null) return null;
                  return { sym, label: BENCH_OPTIONS.find(b => b.symbol === sym)?.label ?? sym, benchReturn, diff: portfolioReturn - benchReturn };
                }).filter(Boolean) as { sym: string; label: string; benchReturn: number; diff: number }[];

                const beating = comparisons.filter(c => c.diff > 0);
                const trailing = comparisons.filter(c => c.diff <= 0);

                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16, paddingTop: isMobile ? 0 : 4 }}>
                    {/* Portfolio return */}
                    <div>
                      <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Your Portfolio · 1Y</div>
                      <div style={{ fontSize: 32, fontWeight: 700, lineHeight: 1, color: portfolioReturn >= 0 ? 'var(--green-2)' : 'var(--red-2)' }}>
                        {portfolioReturn >= 0 ? '+' : ''}{portfolioReturn.toFixed(1)}%
                      </div>
                    </div>

                    {/* Beating summary */}
                    {beating.length > 0 && (
                      <div>
                        <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Beating</div>
                        {beating.map(c => (
                          <div key={c.sym} style={{ marginBottom: 8 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 2 }}>
                              <span style={{ fontSize: 12, color: BENCH_COLORS[c.sym], fontWeight: 600 }}>{c.label}</span>
                              <span style={{ fontSize: 11, color: 'var(--text-4)' }}>{c.benchReturn >= 0 ? '+' : ''}{c.benchReturn.toFixed(1)}%</span>
                            </div>
                            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--green-2)' }}>
                              You&apos;re ahead by +{c.diff.toFixed(1)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Trailing summary */}
                    {trailing.length > 0 && (
                      <div>
                        <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 6 }}>Trailing</div>
                        {trailing.map(c => (
                          <div key={c.sym} style={{ marginBottom: 8 }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 2 }}>
                              <span style={{ fontSize: 12, color: BENCH_COLORS[c.sym], fontWeight: 600 }}>{c.label}</span>
                              <span style={{ fontSize: 11, color: 'var(--text-4)' }}>{c.benchReturn >= 0 ? '+' : ''}{c.benchReturn.toFixed(1)}%</span>
                            </div>
                            <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--red-2)' }}>
                              You&apos;re behind by {c.diff.toFixed(1)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          </div>
        )}

        {/* ── Dividends + Projections ── */}
        {holdings.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 2fr', gap: 12 }}>

            {/* Dividends */}
            <div className="card">
              <span className="card-label">Estimated Dividends</span>

              {/* Totals summary */}
              <div style={{ display: 'flex', gap: 20, marginTop: 8, marginBottom: 14, paddingBottom: 12, borderBottom: '1px solid var(--bg-1)' }}>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 3 }}>Monthly</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: totalAnnDiv > 0 ? 'var(--green-2)' : 'var(--text-5)' }}>
                    {totalAnnDiv > 0 ? money(totalAnnDiv / 12, baseCurrency) : '—'}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 3 }}>Yearly</div>
                  <div style={{ fontSize: 18, fontWeight: 700, color: totalAnnDiv > 0 ? 'var(--green-2)' : 'var(--text-5)' }}>
                    {totalAnnDiv > 0 ? money(totalAnnDiv, baseCurrency) : '—'}
                  </div>
                </div>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: isMobile ? '1fr' : '1fr 1fr', gap: 10, marginBottom: 14 }}>
                <div style={{ background: 'var(--bg-3)', padding: '10px 12px' }}>
                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>Next Payments</div>
                  {dividendCalendar.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {dividendCalendar.slice(0, 5).map(item => (
                        <div key={`${item.symbol}_${item.date}`} style={{ display: 'grid', gridTemplateColumns: '52px 1fr 72px', gap: 8, alignItems: 'center' }}>
                          <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)' }}>{item.symbol}</span>
                          <span style={{ fontSize: 10, color: 'var(--text-4)' }}>{fmtDate(item.date)}</span>
                          <span style={{ fontSize: 11, color: 'var(--green-2)', textAlign: 'right', fontWeight: 600 }}>{item.amount != null ? money(item.amount, baseCurrency) : '—'}</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: 11, color: 'var(--text-5)' }}>No dividend calendar estimates available.</div>
                  )}
                </div>
                <div style={{ background: 'var(--bg-3)', padding: '10px 12px' }}>
                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 8 }}>Dividend Growth</div>
                  {dividendGrowthRows.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {dividendGrowthRows.slice(0, 5).map(item => (
                        <div key={item.symbol} style={{ display: 'grid', gridTemplateColumns: '52px 1fr 64px', gap: 8, alignItems: 'center' }}>
                          <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--accent)' }}>{item.symbol}</span>
                          <span style={{ fontSize: 10, color: 'var(--text-4)' }}>
                            {item.priorAnnual != null ? `${money(item.priorAnnual, baseCurrency)} → ${money(item.annual ?? 0, baseCurrency)}` : `${money(item.annual ?? 0, baseCurrency)} annual`}
                          </span>
                          <span style={{ fontSize: 11, color: item.growth == null ? 'var(--text-5)' : item.growth >= 0 ? 'var(--green-2)' : 'var(--red-2)', textAlign: 'right', fontWeight: 600 }}>
                            {item.growth != null ? pct(item.growth * 100) : '—'}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div style={{ fontSize: 11, color: 'var(--text-5)' }}>No dividend growth history available.</div>
                  )}
                </div>
              </div>

              {isMobile ? (
                /* ── MOBILE: per-position rows ── */
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {rows.map(({ h, q, annDiv }) => (
                    <div key={h.id} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '10px 0', borderBottom: '1px solid var(--bg-3)' }}>
                      <div>
                        <div style={{ fontWeight: 700, color: 'var(--accent)', fontSize: 13 }}>{h.symbol}</div>
                        {q?.dividendYield != null && q.dividendYield > 0 && (
                          <div style={{ fontSize: 10, color: 'var(--text-5)', marginTop: 2 }}>{(q.dividendYield * 100).toFixed(2)}% yield</div>
                        )}
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: 13, fontWeight: 600, color: annDiv && annDiv > 0 ? 'var(--green-2)' : 'var(--text-5)' }}>
                          {annDiv && annDiv > 0 ? `${money(annDiv / 12, baseCurrency)}/mo` : '—'}
                        </div>
                        {annDiv && annDiv > 0 && (
                          <div style={{ fontSize: 11, color: 'var(--text-4)', marginTop: 1 }}>{money(annDiv, baseCurrency)}/yr</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                /* ── DESKTOP: column layout ── */
                <div style={{ overflowX: 'auto' }}>
                  <div style={{ display: 'flex', gap: 8, padding: '0 4px', marginBottom: 4, minWidth: 'max-content' }}>
                    {['Ticker', 'Yield', 'Monthly', 'Yearly'].map((label, i) => (
                      <div key={label} style={{
                        flex: i === 0 ? '0 0 52px' : '0 0 80px',
                        fontSize: 9, color: 'var(--text-5)',
                        textTransform: 'uppercase', letterSpacing: '0.08em',
                        textAlign: i === 0 ? 'left' : 'right',
                      }}>{label}</div>
                    ))}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                    {rows.map(({ h, q, annDiv }) => (
                      <div key={h.id} style={{ display: 'flex', gap: 8, padding: '5px 4px', background: 'var(--bg-3)', alignItems: 'center', minWidth: 'max-content' }}>
                        <div style={{ flex: '0 0 52px', fontWeight: 700, fontSize: 12, color: 'var(--accent)', letterSpacing: '0.04em' }}>{h.symbol}</div>
                        <div style={{ flex: '0 0 80px', textAlign: 'right', fontSize: 11, color: 'var(--text-4)' }}>
                          {q?.dividendYield != null && q.dividendYield > 0 ? `${(q.dividendYield * 100).toFixed(2)}%` : <span style={{ color: 'var(--text-5)' }}>—</span>}
                        </div>
                        <div style={{ flex: '0 0 80px', textAlign: 'right', fontSize: 11, fontWeight: 600, color: annDiv && annDiv > 0 ? 'var(--green-2)' : 'var(--text-5)' }}>
                          {annDiv && annDiv > 0 ? money(annDiv / 12, baseCurrency) : '—'}
                        </div>
                        <div style={{ flex: '0 0 80px', textAlign: 'right', fontSize: 11, fontWeight: 600, color: annDiv && annDiv > 0 ? 'var(--green-2)' : 'var(--text-5)' }}>
                          {annDiv && annDiv > 0 ? money(annDiv, baseCurrency) : '—'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Projections */}
            <div className="card">
              <span className="card-label">Projections</span>

              {/* Horizon selector */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginTop: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                <span style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Show</span>
                {[1, 3, 5, 10, 20, 30].map(yr => {
                  const active = horizons.includes(yr);
                  return (
                    <label key={yr} style={{ display: 'flex', gap: 4, alignItems: 'center', fontSize: 11, color: active ? 'var(--accent)' : 'var(--text-4)', cursor: 'pointer' }}>
                      <input
                        type="checkbox"
                        checked={active}
                        onChange={e => setHorizons(prev =>
                          e.target.checked ? [...prev, yr].sort((a, b) => a - b) : prev.filter(h => h !== yr)
                        )}
                        style={{ accentColor: 'var(--accent)', cursor: 'pointer' }}
                      />
                      {yr}Y
                    </label>
                  );
                })}
                {projLoading && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, color: 'var(--text-5)', marginLeft: 'auto' }}>
                    <RefreshCw size={10} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
                    Fetching CAGR…
                  </div>
                )}
              </div>

              {isMobile ? (
                /* ── MOBILE: per-position projection cards ── */
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {rows.map(({ h, price, baseValue, annDiv }) => {
                    const proj      = projections[h.symbol];
                    const val       = baseValue;
                    const histCagr  = proj?.cagr ?? null;
                    const isLoading = proj?.loading ?? true;
                    const ov        = posOverrides[h.symbol] ?? { ratePct: '', contrib: '', includeDivs: false };
                    const isCustom  = ov.ratePct !== '';
                    const rate      = parseFloat(ov.ratePct) / 100;
                    const manualContrib = parseFloat(ov.contrib) || 0;
                    const divContrib    = ov.includeDivs ? (annDiv ?? 0) : 0;
                    const contrib       = manualContrib + divContrib;
                    const effectiveRate = isCustom ? (isNaN(rate) ? null : rate) : histCagr;
                    const rateColor = effectiveRate != null ? (effectiveRate >= 0 ? 'var(--green-2)' : 'var(--red-2)') : 'var(--text-5)';
                    const unrealistic = !isCustom && histCagr != null && histCagr > 0.30;

                    const setOv = (patch: Partial<{ ratePct: string; contrib: string; includeDivs: boolean }>) =>
                      setPosOverrides(prev => ({ ...prev, [h.symbol]: { ...ov, ...patch } }));

                    const toggleStyle = (active: boolean): React.CSSProperties => ({
                      padding: '3px 8px', fontSize: 10, border: 'none', cursor: 'pointer',
                      background: active ? 'var(--accent)' : 'var(--bg-1)',
                      color: active ? 'var(--text-0)' : 'var(--text-5)',
                      fontFamily: 'inherit',
                    });

                    const excluded = projExcluded.has(h.symbol);
                    return (
                      <div key={h.id} style={{ background: 'var(--bg-3)', padding: '12px 14px', borderLeft: `2px solid ${excluded ? 'var(--bg-1)' : 'var(--bg-1)'}`, opacity: excluded ? 0.45 : 1 }}>
                        {/* Symbol + price */}
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                            <span style={{ fontWeight: 700, color: excluded ? 'var(--text-4)' : 'var(--accent)', fontSize: 14 }}>{h.symbol}</span>
                            <button
                              onClick={() => toggleProjExclude(h.symbol)}
                              style={{ fontSize: 9, padding: '2px 6px', border: `1px solid ${excluded ? 'var(--red-2)' : 'var(--bg-1)'}`, background: 'transparent', color: excluded ? 'var(--red-2)' : 'var(--text-5)', cursor: 'pointer', fontFamily: 'inherit', lineHeight: 1.4 }}
                            >{excluded ? 'excluded' : 'exclude'}</button>
                          </div>
                          <span style={{ fontSize: 13, color: 'var(--text-2)', fontWeight: 600 }}>{price != null ? usd(price) : '—'}</span>
                        </div>

                        {/* Rate row */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 10, flexWrap: 'wrap' }}>
                          <div style={{ display: 'flex', border: '1px solid var(--bg-1)' }}>
                            <button style={toggleStyle(!isCustom)} onClick={() => setOv({ ratePct: '' })}>CAGR</button>
                            <button style={toggleStyle(isCustom)} onClick={() => { if (!isCustom) setOv({ ratePct: histCagr != null ? (histCagr * 100).toFixed(2) : '' }); }}>Custom</button>
                          </div>
                          {isCustom ? (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                              <input
                                type="number"
                                value={ov.ratePct}
                                onChange={e => setOv({ ratePct: e.target.value })}
                                style={{ width: 70, padding: '4px 8px', fontSize: 13, color: rateColor }}
                                step="0.5"
                              />
                              <span style={{ fontSize: 11, color: 'var(--text-5)' }}>%</span>
                            </div>
                          ) : (
                            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                              <span style={{ fontSize: 13, color: rateColor, fontWeight: 600 }}>
                                {isLoading ? '…' : histCagr != null ? pct(histCagr * 100) : '—'}
                              </span>
                              {unrealistic && <span style={{ fontSize: 9, color: 'var(--yellow-2)' }}>⚠ high CAGR</span>}
                            </div>
                          )}
                        </div>

                        {/* Contrib + divs row */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <span style={{ fontSize: 10, color: 'var(--text-5)' }}>Contrib/yr $</span>
                            <input
                              type="number"
                              value={ov.contrib}
                              onChange={e => setOv({ contrib: e.target.value })}
                              placeholder="0"
                              style={{ width: 80, padding: '4px 8px', fontSize: 12 }}
                              min="0"
                              step="100"
                            />
                          </div>
                          <label style={{ display: 'flex', alignItems: 'center', gap: 5, cursor: annDiv ? 'pointer' : 'default', opacity: annDiv ? 1 : 0.4 }}>
                            <input
                              type="checkbox"
                              checked={ov.includeDivs}
                              onChange={e => annDiv && setOv({ includeDivs: e.target.checked })}
                              disabled={!annDiv}
                              style={{ accentColor: 'var(--accent)' }}
                            />
                            <span style={{ fontSize: 10, color: ov.includeDivs && annDiv ? 'var(--green-2)' : 'var(--text-5)' }}>
                              {annDiv && annDiv > 0 ? `+divs (${money(annDiv / 12, baseCurrency)}/mo)` : 'no div data'}
                            </span>
                          </label>
                        </div>

                        {/* Horizon results grid */}
                        {horizons.length > 0 && (
                          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(horizons.length, 3)}, 1fr)`, gap: 6 }}>
                            {horizons.map(yr => {
                              const fv   = effectiveRate != null ? projectFV(val, effectiveRate, yr, contrib) : null;
                              const gain = fv != null ? (fv / val - 1) * 100 : null;
                              return (
                                <div key={yr} style={{ background: 'rgba(0,0,0,0.25)', padding: '8px 10px', textAlign: 'center' }}>
                                  <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>{yr}Y</div>
                                  <div style={{ fontSize: 12, fontWeight: 600, color: unrealistic ? 'var(--yellow-2)' : 'var(--text-1)', whiteSpace: 'nowrap' }}>
                                    {fv != null ? money(fv, baseCurrency) : isLoading ? '…' : '—'}
                                  </div>
                                  {gain != null && (
                                    <div style={{ fontSize: 10, marginTop: 2, color: unrealistic ? 'var(--yellow-2)' : gain >= 0 ? 'var(--green-2)' : 'var(--red-2)' }}>{pct(gain)}</div>
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                /* ── DESKTOP: horizontal flex rows ── */
                <div style={{ overflowX: 'auto' }}>
                  {horizons.length > 0 && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '0 14px', marginBottom: 4, minWidth: 'max-content' }}>
                      <div style={{ width: 20 }} />
                      <div style={{ minWidth: 70 }} />
                      <div style={{ width: 85, fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Price</div>
                      <div style={{ width: 130, fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Rate % / yr</div>
                      <div style={{ width: 100, fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Contrib $ / yr</div>
                      {horizons.map(yr => (
                        <div key={yr} style={{ width: 110, flexShrink: 0, fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>{yr}Y</div>
                      ))}
                    </div>
                  )}
                  {(() => {
                    const allLoaded = rows.every(r => projections[r.h.symbol] && !projections[r.h.symbol].loading);
                    return (
                      <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                        {rows.map(({ h, price, baseValue, annDiv }) => {
                          const proj      = projections[h.symbol];
                          const val       = baseValue;
                          const histCagr  = proj?.cagr ?? null;
                          const isLoading = proj?.loading ?? true;
                          const ov        = posOverrides[h.symbol] ?? { ratePct: '', contrib: '', includeDivs: false };
                          const isCustom  = ov.ratePct !== '';
                          const rate      = parseFloat(ov.ratePct) / 100;
                          const manualContrib = parseFloat(ov.contrib) || 0;
                          const divContrib    = ov.includeDivs ? (annDiv ?? 0) : 0;
                          const contrib       = manualContrib + divContrib;
                          const effectiveRate = isCustom ? (isNaN(rate) ? null : rate) : histCagr;
                          const rateColor = effectiveRate != null ? (effectiveRate >= 0 ? 'var(--green-2)' : 'var(--red-2)') : 'var(--text-5)';

                          const setOv = (patch: Partial<{ ratePct: string; contrib: string; includeDivs: boolean }>) =>
                            setPosOverrides(prev => ({ ...prev, [h.symbol]: { ...ov, ...patch } }));

                          const toggleStyle = (active: boolean): React.CSSProperties => ({
                            padding: '2px 7px', fontSize: 9, border: 'none', cursor: 'pointer',
                            background: active ? 'var(--accent)' : 'var(--bg-1)',
                            color: active ? 'var(--text-0)' : 'var(--text-5)',
                            fontFamily: 'inherit',
                          });

                          const excluded = projExcluded.has(h.symbol);
                          return (
                            <div key={h.id} style={{ background: 'var(--bg-3)', padding: '8px 14px', display: 'flex', alignItems: 'center', gap: 16, minWidth: 'max-content', opacity: excluded ? 0.45 : 1 }}>
                              <div style={{ width: 20, display: 'flex', alignItems: 'center' }}>
                                <input
                                  type="checkbox"
                                  checked={!excluded}
                                  onChange={() => toggleProjExclude(h.symbol)}
                                  title={excluded ? `Include ${h.symbol} in total` : `Exclude ${h.symbol} from total`}
                                  style={{ accentColor: 'var(--accent)', cursor: 'pointer', width: 13, height: 13 }}
                                />
                              </div>
                              <div style={{ minWidth: 70 }}>
                                <span style={{ fontWeight: 700, color: excluded ? 'var(--text-4)' : 'var(--accent)', fontSize: 12, letterSpacing: '0.04em' }}>{h.symbol}</span>
                              </div>
                              <div style={{ width: 85, fontSize: 12, color: 'var(--text-2)', fontWeight: 600 }}>
                                {price != null ? usd(price) : <span style={{ color: 'var(--text-5)' }}>—</span>}
                              </div>
                              <div style={{ width: 130, display: 'flex', flexDirection: 'column', gap: 4 }}>
                                <div style={{ display: 'flex', border: '1px solid var(--bg-1)', width: 'fit-content' }}>
                                  <button style={toggleStyle(!isCustom)} onClick={() => setOv({ ratePct: '' })}>CAGR</button>
                                  <button style={toggleStyle(isCustom)} onClick={() => { if (!isCustom) setOv({ ratePct: histCagr != null ? (histCagr * 100).toFixed(2) : '' }); }}>Custom</button>
                                </div>
                                {isCustom ? (
                                  <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                                    <input type="number" value={ov.ratePct} onChange={e => setOv({ ratePct: e.target.value })} style={{ width: 60, padding: '3px 6px', fontSize: 11, color: rateColor }} step="0.5" />
                                    <span style={{ fontSize: 10, color: 'var(--text-5)' }}>%</span>
                                  </div>
                                ) : (
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                    <span style={{ fontSize: 11, color: rateColor, fontWeight: 600 }}>
                                      {isLoading ? '…' : histCagr != null ? pct(histCagr * 100) : '—'}
                                    </span>
                                    {!isLoading && histCagr != null && histCagr > 0.30 && (
                                      <span style={{ fontSize: 9, color: 'var(--yellow-2)', lineHeight: 1.3 }}>⚠ Based on recent history. Use Custom for realistic projections.</span>
                                    )}
                                  </div>
                                )}
                              </div>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: 3, width: 100 }}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                                  <span style={{ fontSize: 10, color: 'var(--text-5)' }}>$</span>
                                  <input type="number" value={ov.contrib} onChange={e => setOv({ contrib: e.target.value })} placeholder="0" style={{ width: 78, padding: '3px 6px', fontSize: 11 }} min="0" step="100" />
                                </div>
                                <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: annDiv ? 'pointer' : 'default', opacity: annDiv ? 1 : 0.4 }}>
                                  <input type="checkbox" checked={ov.includeDivs} onChange={e => annDiv && setOv({ includeDivs: e.target.checked })} disabled={!annDiv} style={{ accentColor: 'var(--accent)', cursor: annDiv ? 'pointer' : 'default' }} />
                                  <span style={{ fontSize: 9, color: ov.includeDivs && annDiv ? 'var(--green-2)' : 'var(--text-5)', letterSpacing: '0.04em' }}>
                                    {annDiv && annDiv > 0 ? `+${money(annDiv / 12, baseCurrency)}/mo div` : 'no div data'}
                                  </span>
                                </label>
                              </div>
                              {(() => {
                                const unrealistic = !isCustom && histCagr != null && histCagr > 0.30;
                                return horizons.map(yr => {
                                  const fv   = effectiveRate != null ? projectFV(val, effectiveRate, yr, contrib) : null;
                                  const gain = fv != null ? (fv / val - 1) * 100 : null;
                                  return (
                                    <div key={yr} style={{ width: 110, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 0 }}>
                                      <span style={{ fontSize: 12, fontWeight: 600, color: unrealistic ? 'var(--yellow-2)' : 'var(--text-1)', whiteSpace: 'nowrap' }}>
                                        {fv != null ? money(fv, baseCurrency) : isLoading ? '…' : '—'}
                                      </span>
                                      {gain != null && (
                                        <span style={{ fontSize: 10, color: unrealistic ? 'var(--yellow-2)' : gain >= 0 ? 'var(--green-2)' : 'var(--red-2)' }}>{pct(gain)}</span>
                                      )}
                                    </div>
                                  );
                                });
                              })()}
                            </div>
                          );
                        })}
                        {allLoaded && horizons.length > 0 && (
                          <div style={{ background: 'rgba(0,0,0,0.3)', borderTop: '2px solid var(--bg-1)', padding: '8px 14px', display: 'flex', alignItems: 'center', gap: 16, marginTop: 2, minWidth: 'max-content' }}>
                            <div style={{ width: 20 }} />
                            <div style={{ minWidth: 70 }}>
                              <span style={{ fontWeight: 700, fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-4)' }}>
                                Total{projExcluded.size > 0 && <span style={{ color: 'var(--text-5)', fontWeight: 400 }}> (excl. {Array.from(projExcluded).join(', ')})</span>}
                              </span>
                            </div>
                            <div style={{ width: 85 }} />
                            <div style={{ width: 130 }} />
                            <div style={{ width: 100 }} />
                            {horizons.map(yr => {
                              const total = rows.filter(({ h }) => !projExcluded.has(h.symbol)).reduce((s, { h, baseValue, annDiv }) => {
                                const val  = baseValue;
                                const ov   = posOverrides[h.symbol] ?? { ratePct: '', contrib: '', includeDivs: false };
                                const r    = parseFloat(ov.ratePct) / 100;
                                const rate = ov.ratePct !== '' ? (isNaN(r) ? null : r) : (projections[h.symbol]?.cagr ?? null);
                                const contrib = (parseFloat(ov.contrib) || 0) + (ov.includeDivs ? (annDiv ?? 0) : 0);
                                return s + (rate != null ? projectFV(val, rate, yr, contrib) : val);
                              }, 0);
                              const includedValue = rows.filter(({ h }) => !projExcluded.has(h.symbol)).reduce((s, r) => s + r.baseValue, 0);
                              const gain = includedValue > 0 ? (total / includedValue - 1) * 100 : 0;
                              return (
                                <div key={yr} style={{ width: 110, flexShrink: 0, display: 'flex', flexDirection: 'column', gap: 0 }}>
                                  <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-1)', whiteSpace: 'nowrap' }}>{money(total, baseCurrency)}</span>
                                  <span style={{ fontSize: 10, color: gain >= 0 ? 'var(--green-2)' : 'var(--red-2)' }}>{pct(gain)}</span>
                                </div>
                              );
                            })}
                          </div>
                        )}
                      </div>
                    );
                  })()}
                </div>
              )}

              <div style={{ marginTop: 10, fontSize: 10, color: 'var(--text-5)', lineHeight: 1.5 }}>
                Estimates only. Past performance does not guarantee future results. Not financial advice.
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
