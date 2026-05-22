'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import Link from 'next/link';
import { ArrowLeft, Plus, Trash2, RefreshCw, Pencil, Check, X } from 'lucide-react';

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
}

interface Projection {
  cagr: number | null;
  dataYears?: number | null;
  loading: boolean;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

const LS_KEY = 'portfolio_holdings_v1';

function loadHoldings(): Holding[] {
  try { return JSON.parse(localStorage.getItem(LS_KEY) ?? '[]'); } catch { return []; }
}
function saveHoldings(h: Holding[]) { localStorage.setItem(LS_KEY, JSON.stringify(h)); }

function usd(v: number) {
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(v);
}
function pct(v: number) { return `${v >= 0 ? '+' : ''}${v.toFixed(2)}%`; }
function projectFV(value: number, rate: number, years: number, contrib = 0) {
  if (Math.abs(rate) < 1e-10) return value + contrib * years;
  return value * Math.pow(1 + rate, years) + contrib * (Math.pow(1 + rate, years) - 1) / rate;
}

// ─── Ticker autocomplete input ────────────────────────────────────────────────

interface SearchResult { symbol: string; shortname?: string; longname?: string; typeDisp?: string; }

function TickerInput({ value, onChange, onSelect }: {
  value: string;
  onChange: (v: string) => void;
  onSelect: (symbol: string, name: string) => void;
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

  // Close on outside click
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
    <div ref={wrapRef}>
      <input
        ref={inputRef}
        type="text"
        placeholder="AAPL or Apple…"
        value={value}
        onChange={e => onChange(e.target.value.toUpperCase())}
        onFocus={() => results.length > 0 && setOpen(true)}
        onKeyDown={handleKey}
        style={{ width: 180 }}
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
  const [horizons,     setHorizons]     = useState<number[]>([1, 5, 10]);
  const [posOverrides, setPosOverrides] = useState<Record<string, { ratePct: string; contrib: string; includeDivs: boolean }>>({});
  const [mktConverted, setMktConverted] = useState<Record<string, boolean>>({});
  const [fxRate,       setFxRate]       = useState(1.36); // 1 USD = fxRate CAD

  const [form,      setForm]      = useState({ symbol: '', quantity: '', avgPrice: '' });
  const [formErr,   setFormErr]   = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editForm,  setEditForm]  = useState({ symbol: '', quantity: '', avgPrice: '' });

  useEffect(() => { setHoldings(loadHoldings()); }, []);

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
      const data = await res.json();
      const map: Record<string, Quote> = {};
      (data.quotes ?? []).forEach((q: Quote) => { map[q.symbol] = q; });
      setQuotes(prev => ({ ...prev, ...map }));
    } finally { setQuoteLoading(false); }
  }, []);

  const loadProjections = useCallback(async (syms: string[]) => {
    if (!syms.length) return;
    setProjLoading(true);
    const init: Record<string, Projection> = {};
    syms.forEach(s => { init[s] = { cagr: null, loading: true }; });
    setProjections(init);
    await Promise.allSettled(syms.map(async (sym) => {
      try {
        const res  = await fetch(`/api/stock?symbol=${sym}&days=1825&interval=1mo`);
        const data = await res.json();
        const pts  = (data.data ?? []) as { close: number }[];
        if (pts.length >= 6) {
          const years = (pts.length - 1) / 12;
          const cagr  = Math.pow(pts[pts.length - 1].close / pts[0].close, 1 / years) - 1;
          setProjections(prev => ({ ...prev, [sym]: { cagr, loading: false } }));
        } else {
          setProjections(prev => ({ ...prev, [sym]: { cagr: null, loading: false } }));
        }
      } catch {
        setProjections(prev => ({ ...prev, [sym]: { cagr: null, loading: false } }));
      }
    }));
    setProjLoading(false);
  }, []);

  useEffect(() => {
    const syms = Array.from(new Set(holdings.map(h => h.symbol)));
    if (syms.length) {
      fetchQuotes(syms);
      loadProjections(syms);
    }
  }, [holdings, fetchQuotes, loadProjections]);

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
  }

  // Derived rows
  const rows = holdings.map(h => {
    const q         = quotes[h.symbol];
    const price     = q?.price ?? null;
    const costBasis = h.quantity * h.avgPrice;
    const mktValue  = price !== null ? h.quantity * price : null;
    const gl        = mktValue !== null ? mktValue - costBasis : null;
    const glPct     = gl !== null && costBasis > 0 ? (gl / costBasis) * 100 : null;
    const divRate   = q?.dividendRate ?? (q?.dividendYield != null && price != null ? q.dividendYield * price : null);
    const annDiv    = divRate != null ? h.quantity * divRate : null;
    return { h, q, price, costBasis, mktValue, gl, glPct, annDiv };
  });

  function convertedMktVal(h: Holding, raw: number): number {
    if (!mktConverted[h.id]) return raw;
    return h.symbol.endsWith('.TO') ? raw / fxRate : raw * fxRate;
  }

  const totalInvested = rows.reduce((s, r) => s + r.costBasis, 0);
  const totalValue    = rows.reduce((s, r) => s + convertedMktVal(r.h, r.mktValue ?? r.costBasis), 0);
  const totalGL       = totalValue - totalInvested;
  const totalGLPct    = totalInvested > 0 ? (totalGL / totalInvested) * 100 : 0;
  const totalAnnDiv   = rows.reduce((s, r) => s + (r.annDiv ?? 0), 0);

  const glColor = (v: number | null) => v == null ? 'var(--text-3)' : v >= 0 ? 'var(--green-2)' : 'var(--red-2)';

  return (
    <div className="portfolio-page" style={{ minHeight: '100vh', background: 'var(--bg-4)', color: 'var(--text-2)', fontFamily: 'inherit' }}>
      <div style={{ padding: '16px 24px' }}>

        {/* ── Nav ── */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 16, marginBottom: 20 }}>
          <Link href="/" style={{
            display: 'flex', alignItems: 'center', gap: 6, fontSize: 12,
            color: 'var(--text-4)', textDecoration: 'none',
            border: '2px solid var(--bg-1)', padding: '6px 12px',
            background: 'rgba(0,0,0,0.2)', transition: 'border-color 0.2s',
          }}>
            <ArrowLeft size={13} /> Back
          </Link>
          <span style={{ fontSize: 14, fontWeight: 700, color: 'var(--accent)', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Portfolio Tracker
          </span>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
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
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
          {[
            { label: 'Total Invested',  value: usd(totalInvested), color: 'var(--text-1)' },
            { label: 'Current Value',   value: usd(totalValue),    color: 'var(--text-1)' },
            { label: 'Total Gain/Loss', value: (totalGL >= 0 ? '+' : '') + usd(totalGL), color: glColor(totalGL) },
            { label: 'Return',          value: pct(totalGLPct), color: glColor(totalGLPct) },
          ].map(c => (
            <div key={c.label} className="card" style={{ padding: '18px 20px' }}>
              <span className="card-label">{c.label}</span>
              <div style={{ fontSize: 22, fontWeight: 700, color: c.color, lineHeight: 1.2, marginTop: 6 }}>
                {c.value}
              </div>
            </div>
          ))}
        </div>

        {/* ── Holdings card ── */}
        <div className="card" style={{ marginBottom: 16 }}>
          <span className="card-label">Holdings</span>

          {/* Add row form */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'flex-end', marginBottom: 20, flexWrap: 'wrap' }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 10, color: 'var(--text-5)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Ticker</span>
              <TickerInput
                value={form.symbol}
                onChange={v => setForm(f => ({ ...f, symbol: v }))}
                onSelect={(sym) => setForm(f => ({ ...f, symbol: sym }))}
              />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 10, color: 'var(--text-5)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Shares</span>
              <input
                type="number"
                placeholder="10"
                min="0"
                value={form.quantity}
                onChange={e => setForm(f => ({ ...f, quantity: e.target.value }))}
                onKeyDown={e => e.key === 'Enter' && addHolding()}
                style={{ width: 90 }}
              />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 10, color: 'var(--text-5)', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Avg Cost ($)</span>
              <input
                type="number"
                placeholder="180.00"
                min="0"
                step="0.01"
                value={form.avgPrice}
                onChange={e => setForm(f => ({ ...f, avgPrice: e.target.value }))}
                onKeyDown={e => e.key === 'Enter' && addHolding()}
                style={{ width: 110 }}
              />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: 10, color: 'transparent', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Add</span>
              <button
                onClick={addHolding}
                style={{
                  padding: '12px 18px',
                  fontSize: 'inherit',
                  fontFamily: 'inherit',
                  lineHeight: 'inherit',
                  fontWeight: 500,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  background: 'var(--accent)',
                  color: 'var(--text-0)',
                  border: '2px solid var(--accent)',
                  boxShadow: 'none',
                  cursor: 'pointer',
                  whiteSpace: 'nowrap',
                }}
              >
                <Plus size={13} /> Add Position
              </button>
            </div>
            {formErr && <span style={{ fontSize: 11, color: 'var(--red-2)', alignSelf: 'center' }}>{formErr}</span>}
          </div>

          {holdings.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-5)', fontSize: 13 }}>
              No positions yet. Add a ticker above to get started.
            </div>
          ) : quoteLoading && Object.keys(quotes).length === 0 ? (
            <div style={{ textAlign: 'center', padding: '40px 20px', color: 'var(--text-5)', fontSize: 13, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 10 }}>
              <RefreshCw size={14} style={{ animation: 'spin 1s linear infinite', color: 'var(--accent)' }} />
              Fetching prices…
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                <thead>
                  <tr>
                    {['Ticker', 'Name', 'Shares', 'Avg Cost', 'Current Price', 'Mkt Value', 'Gain / Loss', '%'].map((h, i) => (
                      <th key={h} style={{
                        padding: '8px 12px', fontSize: 10, fontWeight: 600, letterSpacing: '0.08em',
                        textTransform: 'uppercase', color: 'var(--text-5)',
                        textAlign: i <= 1 ? 'left' : 'right',
                        borderBottom: '1px solid var(--bg-1)',
                        whiteSpace: 'nowrap',
                      }}>
                        {h}
                      </th>
                    ))}
                    <th style={{ width: 32, borderBottom: '1px solid var(--bg-1)' }} />
                  </tr>
                </thead>
                <tbody>
                  {rows.map(({ h, q, price, costBasis, mktValue, gl, glPct }) => {
                    const isEditing = editingId === h.id;
                    if (isEditing) return (
                      <tr key={h.id} style={{ borderBottom: '1px solid var(--bg-3)', background: 'rgba(0,0,0,0.2)' }}>
                        {/* Editable ticker with autocomplete */}
                        <td style={{ padding: '6px 8px' }}>
                          <TickerInput
                            value={editForm.symbol}
                            onChange={v => setEditForm(f => ({ ...f, symbol: v }))}
                            onSelect={sym => setEditForm(f => ({ ...f, symbol: sym }))}
                          />
                        </td>
                        {/* Name (read-only in edit mode) */}
                        <td style={{ padding: '6px 8px', fontSize: 11, color: 'var(--text-5)' }}>
                          {q?.name ?? '—'}
                        </td>
                        {/* Editable shares */}
                        <td style={{ padding: '6px 8px', textAlign: 'right' }}>
                          <input
                            type="number"
                            value={editForm.quantity}
                            onChange={e => setEditForm(f => ({ ...f, quantity: e.target.value }))}
                            onKeyDown={e => { if (e.key === 'Enter') saveEdit(); if (e.key === 'Escape') setEditingId(null); }}
                            style={{ width: 80, padding: '4px 8px', fontSize: 12, textAlign: 'right' }}
                          />
                        </td>
                        {/* Editable avg cost */}
                        <td style={{ padding: '6px 8px', textAlign: 'right' }}>
                          <input
                            type="number"
                            value={editForm.avgPrice}
                            onChange={e => setEditForm(f => ({ ...f, avgPrice: e.target.value }))}
                            onKeyDown={e => { if (e.key === 'Enter') saveEdit(); if (e.key === 'Escape') setEditingId(null); }}
                            style={{ width: 90, padding: '4px 8px', fontSize: 12, textAlign: 'right' }}
                            step="0.01"
                          />
                        </td>
                        {/* Remaining cols empty while editing */}
                        <td /><td /><td /><td />
                        {/* Save / Cancel */}
                        <td style={{ padding: '6px 8px', textAlign: 'center' }}>
                          <div style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
                            <button onClick={saveEdit} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--green-2)', padding: 2 }}>
                              <Check size={14} />
                            </button>
                            <button onClick={() => setEditingId(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--red-2)', padding: 2 }}>
                              <X size={14} />
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                    return (
                    <tr key={h.id} style={{ borderBottom: '1px solid var(--bg-3)' }}>
                      <td style={{ padding: '10px 12px', fontWeight: 700, color: 'var(--accent)', letterSpacing: '0.04em' }}>
                        {h.symbol}
                      </td>
                      <td style={{ padding: '10px 12px', color: 'var(--text-4)', fontSize: 12, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {q?.name ?? '—'}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', color: 'var(--text-2)' }}>
                        {h.quantity.toLocaleString()}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', color: 'var(--text-3)' }}>
                        {usd(h.avgPrice)}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', color: 'var(--text-2)' }}>
                        {price != null ? usd(price) : <span style={{ color: 'var(--text-5)' }}>—</span>}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: 'var(--text-1)' }}>
                        {(() => {
                          const native  = h.symbol.endsWith('.TO') ? 'CAD' : 'USD';
                          const other   = native === 'USD' ? 'CAD' : 'USD';
                          const raw     = mktValue ?? costBasis;
                          const conv    = mktConverted[h.id];
                          const display = conv
                            ? (native === 'USD' ? raw * fxRate : raw / fxRate)
                            : raw;
                          const cur     = conv ? other : native;
                          const formatted = new Intl.NumberFormat('en-US', {
                            style: 'currency', currency: cur,
                            minimumFractionDigits: 2, maximumFractionDigits: 2,
                          }).format(display);
                          return (
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 5 }}>
                              <span>{formatted}</span>
                              <button
                                onClick={() => setMktConverted(prev => ({ ...prev, [h.id]: !prev[h.id] }))}
                                title={`Show in ${conv ? native : other}`}
                                style={{
                                  fontSize: 8, padding: '1px 4px', border: '1px solid var(--bg-1)',
                                  background: conv ? 'var(--accent)' : 'transparent',
                                  color: conv ? 'var(--text-0)' : 'var(--text-5)',
                                  cursor: 'pointer', fontFamily: 'inherit', lineHeight: 1.4,
                                }}
                              >{conv ? other : `→${other}`}</button>
                            </div>
                          );
                        })()}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: glColor(gl) }}>
                        {gl != null ? (() => {
                          const conv    = mktConverted[h.id];
                          const native  = h.symbol.endsWith('.TO') ? 'CAD' : 'USD';
                          const other   = native === 'USD' ? 'CAD' : 'USD';
                          const display = convertedMktVal(h, gl);
                          const cur     = conv ? other : native;
                          const sign    = gl >= 0 ? '+' : '';
                          return sign + new Intl.NumberFormat('en-US', {
                            style: 'currency', currency: cur,
                            minimumFractionDigits: 2, maximumFractionDigits: 2,
                          }).format(display);
                        })() : '—'}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: glColor(glPct) }}>
                        {glPct != null ? pct(glPct) : '—'}
                      </td>
                      <td style={{ padding: '10px 8px', textAlign: 'center' }}>
                        <div style={{ display: 'flex', gap: 4, justifyContent: 'center' }}>
                          <button
                            onClick={() => startEdit(h)}
                            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-5)', padding: 2, lineHeight: 1 }}
                          >
                            <Pencil size={12} />
                          </button>
                          <button
                            onClick={() => removeHolding(h.id)}
                            style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-5)', padding: 2, lineHeight: 1 }}
                          >
                            <Trash2 size={12} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  )})}

                  {/* Totals row */}
                  <tr style={{ background: 'rgba(0,0,0,0.25)', borderTop: '2px solid var(--bg-1)' }}>
                    <td colSpan={2} style={{ padding: '10px 12px', fontWeight: 700, fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-4)' }}>
                      Total
                    </td>
                    <td style={{ padding: '10px 12px' }} />
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: 'var(--text-3)' }}>
                      {usd(totalInvested)}
                    </td>
                    <td style={{ padding: '10px 12px' }} />
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: 'var(--text-1)' }}>
                      {usd(totalValue)}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: glColor(totalGL) }}>
                      {(totalGL >= 0 ? '+' : '') + usd(totalGL)}
                    </td>
                    <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 700, color: glColor(totalGLPct) }}>
                      {pct(totalGLPct)}
                    </td>
                    <td />
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ── Dividends + Projections (only if holdings exist) ── */}
        {holdings.length > 0 && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 12 }}>

            {/* Dividends — always shown */}
            <div className="card">
                <span className="card-label">Estimated Dividends</span>

                {/* Totals */}
                <div style={{ display: 'flex', gap: 20, marginTop: 8, marginBottom: 14, paddingBottom: 12, borderBottom: '1px solid var(--bg-1)' }}>
                  <div>
                    <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 3 }}>Monthly</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: totalAnnDiv > 0 ? 'var(--green-2)' : 'var(--text-5)' }}>
                      {totalAnnDiv > 0 ? usd(totalAnnDiv / 12) : '—'}
                    </div>
                  </div>
                  <div>
                    <div style={{ fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 3 }}>Yearly</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: totalAnnDiv > 0 ? 'var(--green-2)' : 'var(--text-5)' }}>
                      {totalAnnDiv > 0 ? usd(totalAnnDiv) : '—'}
                    </div>
                  </div>
                </div>

                {/* Column header */}
                <div style={{ display: 'flex', gap: 8, padding: '0 4px', marginBottom: 4 }}>
                  {['Ticker', 'Yield', 'Monthly', 'Yearly'].map((label, i) => (
                    <div key={label} style={{
                      flex: i === 0 ? '0 0 52px' : 1,
                      fontSize: 9, color: 'var(--text-5)',
                      textTransform: 'uppercase', letterSpacing: '0.08em',
                      textAlign: i === 0 ? 'left' : 'right',
                    }}>{label}</div>
                  ))}
                </div>

                {/* Per-position rows — all holdings */}
                <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {rows.map(({ h, q, annDiv }) => (
                    <div key={h.id} style={{ display: 'flex', gap: 8, padding: '5px 4px', background: 'var(--bg-3)', alignItems: 'center' }}>
                      <div style={{ flex: '0 0 52px', fontWeight: 700, fontSize: 12, color: 'var(--accent)', letterSpacing: '0.04em' }}>{h.symbol}</div>
                      <div style={{ flex: 1, textAlign: 'right', fontSize: 11, color: 'var(--text-4)' }}>
                        {q?.dividendYield != null && q.dividendYield > 0 ? `${(q.dividendYield * 100).toFixed(2)}%` : <span style={{ color: 'var(--text-5)' }}>—</span>}
                      </div>
                      <div style={{ flex: 1, textAlign: 'right', fontSize: 11, fontWeight: 600, color: annDiv && annDiv > 0 ? 'var(--green-2)' : 'var(--text-5)' }}>
                        {annDiv && annDiv > 0 ? usd(annDiv / 12) : '—'}
                      </div>
                      <div style={{ flex: 1, textAlign: 'right', fontSize: 11, fontWeight: 600, color: annDiv && annDiv > 0 ? 'var(--green-2)' : 'var(--text-5)' }}>
                        {annDiv && annDiv > 0 ? usd(annDiv) : '—'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

            {/* Projections */}
            <div className="card">
              <span className="card-label">Projections</span>

              {/* ── Horizons + loading ── */}
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

              {/* ── Column header ── */}
              {horizons.length > 0 && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, padding: '0 14px', marginBottom: 4 }}>
                  <div style={{ minWidth: 70 }} />
                  <div style={{ width: 85, fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Price</div>
                  <div style={{ width: 130, fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Rate % / yr</div>
                  <div style={{ width: 100, fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em' }}>Contrib $ / yr</div>
                  {horizons.map(yr => (
                    <div key={yr} style={{ flex: 1, fontSize: 9, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.08em', minWidth: 0 }}>{yr}Y</div>
                  ))}
                </div>
              )}

              {/* ── Per-position rows ── */}
              {(() => {
                const allLoaded = rows.every(r => projections[r.h.symbol] && !projections[r.h.symbol].loading);
                return (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                    {rows.map(({ h, price, mktValue, costBasis, annDiv }) => {
                      const proj      = projections[h.symbol];
                      const val       = convertedMktVal(h, mktValue ?? costBasis);
                      const histCagr  = proj?.cagr ?? null;
                      const isLoading = proj?.loading ?? true;
                      const ov        = posOverrides[h.symbol] ?? { ratePct: '', contrib: '', includeDivs: false };
                      const isCustom  = ov.ratePct !== '';
                      const rate      = parseFloat(ov.ratePct) / 100;
                      const manualContrib = parseFloat(ov.contrib) || 0;
                      const divContrib    = ov.includeDivs ? (annDiv ?? 0) : 0;
                      const contrib       = manualContrib + divContrib;
                      const effectiveRate = isCustom ? (isNaN(rate) ? null : rate) : histCagr;
                      const rateColor = effectiveRate != null
                        ? (effectiveRate >= 0 ? 'var(--green-2)' : 'var(--red-2)')
                        : 'var(--text-5)';

                      const setOv = (patch: Partial<{ ratePct: string; contrib: string; includeDivs: boolean }>) =>
                        setPosOverrides(prev => ({ ...prev, [h.symbol]: { ...ov, ...patch } }));

                      const toggleStyle = (active: boolean): React.CSSProperties => ({
                        padding: '2px 7px', fontSize: 9, border: 'none', cursor: 'pointer',
                        background: active ? 'var(--accent)' : 'var(--bg-1)',
                        color: active ? 'var(--text-0)' : 'var(--text-5)',
                        fontFamily: 'inherit',
                      });

                      return (
                        <div key={h.id} style={{ background: 'var(--bg-3)', padding: '8px 14px', display: 'flex', alignItems: 'center', gap: 10 }}>
                          {/* Symbol */}
                          <div style={{ minWidth: 70 }}>
                            <span style={{ fontWeight: 700, color: 'var(--accent)', fontSize: 12, letterSpacing: '0.04em' }}>{h.symbol}</span>
                          </div>

                          {/* Current price */}
                          <div style={{ width: 85, fontSize: 12, color: 'var(--text-2)', fontWeight: 600 }}>
                            {price != null ? usd(price) : <span style={{ color: 'var(--text-5)' }}>—</span>}
                          </div>

                          {/* Rate — toggle + value/input */}
                          <div style={{ width: 130, display: 'flex', flexDirection: 'column', gap: 4 }}>
                            <div style={{ display: 'flex', border: '1px solid var(--bg-1)', width: 'fit-content' }}>
                              <button style={toggleStyle(!isCustom)} onClick={() => setOv({ ratePct: '' })}>CAGR</button>
                              <button style={toggleStyle(isCustom)} onClick={() => {
                                if (!isCustom) setOv({ ratePct: histCagr != null ? (histCagr * 100).toFixed(2) : '' });
                              }}>Custom</button>
                            </div>
                            {isCustom ? (
                              <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                                <input
                                  type="number"
                                  value={ov.ratePct}
                                  onChange={e => setOv({ ratePct: e.target.value })}
                                  style={{ width: 60, padding: '3px 6px', fontSize: 11, color: rateColor }}
                                  step="0.5"
                                />
                                <span style={{ fontSize: 10, color: 'var(--text-5)' }}>%</span>
                              </div>
                            ) : (
                              <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                <span style={{ fontSize: 11, color: rateColor, fontWeight: 600 }}>
                                  {isLoading ? '…' : histCagr != null ? pct(histCagr * 100) : '—'}
                                </span>
                                {!isLoading && histCagr != null && histCagr > 0.30 && (
                                  <span style={{ fontSize: 9, color: 'var(--yellow-2)', lineHeight: 1.3 }}>
                                    ⚠ Based on recent history. Use Custom for realistic projections.
                                  </span>
                                )}
                              </div>
                            )}
                          </div>

                          {/* Contribution input */}
                          <div style={{ display: 'flex', flexDirection: 'column', gap: 3, width: 100 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 3 }}>
                              <span style={{ fontSize: 10, color: 'var(--text-5)' }}>$</span>
                              <input
                                type="number"
                                value={ov.contrib}
                                onChange={e => setOv({ contrib: e.target.value })}
                                placeholder="0"
                                style={{ width: 78, padding: '3px 6px', fontSize: 11 }}
                                min="0"
                                step="100"
                              />
                            </div>
                            <label style={{ display: 'flex', alignItems: 'center', gap: 4, cursor: annDiv ? 'pointer' : 'default', opacity: annDiv ? 1 : 0.4 }}>
                              <input
                                type="checkbox"
                                checked={ov.includeDivs}
                                onChange={e => annDiv && setOv({ includeDivs: e.target.checked })}
                                disabled={!annDiv}
                                style={{ accentColor: 'var(--accent)', cursor: annDiv ? 'pointer' : 'default' }}
                              />
                              <span style={{ fontSize: 9, color: ov.includeDivs && annDiv ? 'var(--green-2)' : 'var(--text-5)', letterSpacing: '0.04em' }}>
                                {annDiv && annDiv > 0 ? `+${usd(annDiv / 12)}/mo div` : 'no div data'}
                              </span>
                            </label>
                          </div>

                          {/* Horizon columns */}
                          {(() => {
                            const unrealistic = !isCustom && histCagr != null && histCagr > 0.30;
                            return horizons.map(yr => {
                              const fv   = effectiveRate != null ? projectFV(val, effectiveRate, yr, contrib) : null;
                              const gain = fv != null ? (fv / val - 1) * 100 : null;
                              return (
                                <div key={yr} style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 0, minWidth: 0 }}>
                                  <span style={{ fontSize: 12, fontWeight: 600, color: unrealistic ? 'var(--yellow-2)' : 'var(--text-1)', whiteSpace: 'nowrap' }}>
                                    {fv != null ? usd(fv) : isLoading ? '…' : '—'}
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

                    {/* Portfolio total */}
                    {allLoaded && horizons.length > 0 && (
                      <div style={{ background: 'rgba(0,0,0,0.3)', borderTop: '2px solid var(--bg-1)', padding: '8px 14px', display: 'flex', alignItems: 'center', gap: 10, marginTop: 2 }}>
                        <div style={{ minWidth: 70 }}>
                          <span style={{ fontWeight: 700, fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-4)' }}>Total</span>
                        </div>
                        <div style={{ width: 85 }} />
                        <div style={{ width: 130 }} />
                        <div style={{ width: 100 }} />
                        {horizons.map(yr => {
                          const total = rows.reduce((s, { h, mktValue, costBasis, annDiv }) => {
                            const val  = convertedMktVal(h, mktValue ?? costBasis);
                            const ov   = posOverrides[h.symbol] ?? { ratePct: '', contrib: '', includeDivs: false };
                            const r    = parseFloat(ov.ratePct) / 100;
                            const rate = ov.ratePct !== '' ? (isNaN(r) ? null : r) : (projections[h.symbol]?.cagr ?? null);
                            const contrib = (parseFloat(ov.contrib) || 0) + (ov.includeDivs ? (annDiv ?? 0) : 0);
                            return s + (rate != null ? projectFV(val, rate, yr, contrib) : val);
                          }, 0);
                          const gain = totalValue > 0 ? (total / totalValue - 1) * 100 : 0;
                          return (
                            <div key={yr} style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 0, minWidth: 0 }}>
                              <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-1)', whiteSpace: 'nowrap' }}>{usd(total)}</span>
                              <span style={{ fontSize: 10, color: gain >= 0 ? 'var(--green-2)' : 'var(--red-2)' }}>{pct(gain)}</span>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })()}

              <div style={{ marginTop: 10, fontSize: 10, color: 'var(--text-5)', lineHeight: 1.5 }}>
                ⚠ Estimates only. Past performance does not guarantee future results. Not financial advice.
              </div>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
