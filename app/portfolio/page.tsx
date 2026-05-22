'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import Link from 'next/link';
import { ArrowLeft, Plus, Trash2, RefreshCw, TrendingUp, Pencil, Check, X } from 'lucide-react';

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
function project(value: number, cagr: number, years: number) { return value * Math.pow(1 + cagr, years); }

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
  const [showProj,     setShowProj]     = useState(false);

  const [form,      setForm]      = useState({ symbol: '', quantity: '', avgPrice: '' });
  const [formErr,   setFormErr]   = useState('');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editForm,  setEditForm]  = useState({ symbol: '', quantity: '', avgPrice: '' });

  useEffect(() => { setHoldings(loadHoldings()); }, []);

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

  useEffect(() => {
    const syms = Array.from(new Set(holdings.map(h => h.symbol)));
    if (syms.length) fetchQuotes(syms);
  }, [holdings, fetchQuotes]);

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
    fetchQuotes(Array.from(new Set(next.map(h => h.symbol))));
  }

  async function loadProjections() {
    const syms = Array.from(new Set(holdings.map(h => h.symbol)));
    if (!syms.length) return;
    setProjLoading(true);
    setShowProj(true);
    const init: Record<string, Projection> = {};
    syms.forEach(s => { init[s] = { cagr: null, loading: true }; });
    setProjections(init);
    await Promise.allSettled(syms.map(async (sym) => {
      try {
        const res  = await fetch(`/api/stock?symbol=${sym}&days=1825&interval=1mo`);
        const data = await res.json();
        const pts  = (data.data ?? []) as { close: number }[];
        if (pts.length >= 6) {
          const years = pts.length / 12;
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

  const totalInvested = rows.reduce((s, r) => s + r.costBasis, 0);
  const totalValue    = rows.reduce((s, r) => s + (r.mktValue ?? r.costBasis), 0);
  const totalGL       = totalValue - totalInvested;
  const totalGLPct    = totalInvested > 0 ? (totalGL / totalInvested) * 100 : 0;
  const totalAnnDiv   = rows.reduce((s, r) => s + (r.annDiv ?? 0), 0);

  const glColor = (v: number | null) => v == null ? 'var(--text-3)' : v >= 0 ? 'var(--green-2)' : 'var(--red-2)';

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-4)', color: 'var(--text-2)', fontFamily: 'inherit' }}>
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
            <button className="btn-primary" style={{ padding: '10px 18px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }} onClick={addHolding}>
              <Plus size={13} /> Add Position
            </button>
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
                        {mktValue != null ? usd(mktValue) : usd(costBasis)}
                      </td>
                      <td style={{ padding: '10px 12px', textAlign: 'right', fontWeight: 600, color: glColor(gl) }}>
                        {gl != null ? (gl >= 0 ? '+' : '') + usd(gl) : '—'}
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
          <div style={{ display: 'grid', gridTemplateColumns: totalAnnDiv > 0 ? '1fr 2fr' : '1fr', gap: 12 }}>

            {/* Dividends */}
            {totalAnnDiv > 0 && (
              <div className="card">
                <span className="card-label">Estimated Dividends</span>
                <div style={{ display: 'flex', gap: 24, marginTop: 6 }}>
                  <div>
                    <div style={{ fontSize: 10, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Monthly</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--green-2)' }}>{usd(totalAnnDiv / 12)}</div>
                  </div>
                  <div>
                    <div style={{ fontSize: 10, color: 'var(--text-5)', textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4 }}>Yearly</div>
                    <div style={{ fontSize: 20, fontWeight: 700, color: 'var(--green-2)' }}>{usd(totalAnnDiv)}</div>
                  </div>
                </div>
                <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {rows.filter(r => r.annDiv && r.annDiv > 0).map(({ h, annDiv }) => (
                    <div key={h.id} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'var(--text-4)' }}>
                      <span style={{ color: 'var(--accent)' }}>{h.symbol}</span>
                      <span style={{ color: 'var(--green-2)' }}>{usd(annDiv!)} / yr</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Projections */}
            <div className="card">
              <span className="card-label">Projections (Experimental)</span>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14, marginTop: 6 }}>
                <span style={{ fontSize: 11, color: 'var(--text-5)' }}>Based on 5yr historical CAGR per position</span>
                <button
                  className="btn-secondary"
                  style={{ padding: '5px 12px', fontSize: 11, display: 'flex', alignItems: 'center', gap: 5 }}
                  onClick={loadProjections}
                  disabled={projLoading}
                >
                  <TrendingUp size={12} /> {projLoading ? 'Loading…' : showProj ? 'Recalculate' : 'Calculate'}
                </button>
              </div>

              {showProj && (
                <>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                    <thead>
                      <tr>
                        {['Position', 'CAGR (5yr)', '1 Year', '5 Years', '10 Years'].map((h, i) => (
                          <th key={h} style={{
                            padding: '6px 10px', fontSize: 10, fontWeight: 600,
                            letterSpacing: '0.08em', textTransform: 'uppercase',
                            color: 'var(--text-5)', textAlign: i === 0 ? 'left' : 'right',
                            borderBottom: '1px solid var(--bg-1)', whiteSpace: 'nowrap',
                          }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map(({ h, mktValue, costBasis }) => {
                        const proj  = projections[h.symbol];
                        const val   = mktValue ?? costBasis;
                        const cagr  = proj?.cagr ?? null;
                        const isLoading = proj?.loading ?? true;
                        return (
                          <tr key={h.id} style={{ borderBottom: '1px solid var(--bg-3)' }}>
                            <td style={{ padding: '8px 10px', fontWeight: 700, color: 'var(--accent)' }}>{h.symbol}</td>
                            <td style={{ padding: '8px 10px', textAlign: 'right', color: cagr != null ? (cagr >= 0 ? 'var(--green-2)' : 'var(--red-2)') : 'var(--text-5)', fontWeight: 600 }}>
                              {isLoading ? '…' : cagr != null ? pct(cagr * 100) : 'N/A'}
                            </td>
                            {[1, 5, 10].map(yr => (
                              <td key={yr} style={{ padding: '8px 10px', textAlign: 'right', color: 'var(--text-2)' }}>
                                {isLoading ? '…' : cagr != null ? usd(project(val, cagr, yr)) : '—'}
                              </td>
                            ))}
                          </tr>
                        );
                      })}

                      {/* Total */}
                      {rows.every(r => projections[r.h.symbol] && !projections[r.h.symbol].loading) && (() => {
                        const totals = [1, 5, 10].map(yr =>
                          rows.reduce((s, { h, mktValue, costBasis }) => {
                            const cagr = projections[h.symbol]?.cagr;
                            const val  = mktValue ?? costBasis;
                            return s + (cagr != null ? project(val, cagr, yr) : val);
                          }, 0)
                        );
                        return (
                          <tr style={{ background: 'rgba(0,0,0,0.25)', borderTop: '2px solid var(--bg-1)' }}>
                            <td style={{ padding: '8px 10px', fontWeight: 700, fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--text-4)' }}>Total</td>
                            <td />
                            {totals.map((t, i) => (
                              <td key={i} style={{ padding: '8px 10px', textAlign: 'right', fontWeight: 700, color: 'var(--text-1)' }}>{usd(t)}</td>
                            ))}
                          </tr>
                        );
                      })()}
                    </tbody>
                  </table>
                  <div style={{ marginTop: 10, fontSize: 10, color: 'var(--text-5)', lineHeight: 1.5 }}>
                    ⚠ Estimates only. Past performance does not guarantee future results. Not financial advice.
                  </div>
                </>
              )}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
