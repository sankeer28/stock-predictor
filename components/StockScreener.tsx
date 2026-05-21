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
  ChevronsUp,
  BarChart2,
  Activity,
  Leaf,
  DollarSign,
  Star,
  Wallet,
  Shield,
  Flame,
} from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

// ────────────────────────────────────────────────────────────
// Filter definitions
// ────────────────────────────────────────────────────────────

export const CAP_OPTIONS = [
  { value: '', label: 'Any Size' },
  { value: 'cap_nano', label: 'Nano', desc: '< $50M' },
  { value: 'cap_micro', label: 'Micro', desc: '$50M–$300M' },
  { value: 'cap_small', label: 'Small', desc: '$300M–$2B' },
  { value: 'cap_mid', label: 'Mid', desc: '$2B–$10B' },
  { value: 'cap_large', label: 'Large', desc: '$10B–$200B' },
  { value: 'cap_mega', label: 'Mega', desc: '> $200B' },
  { value: 'cap_smallover', label: 'Small+', desc: '> $300M' },
  { value: 'cap_midover', label: 'Mid+', desc: '> $2B' },
  { value: 'cap_largeover', label: 'Large+', desc: '> $10B' },
];

const PE_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'fa_pe_u5', label: '< 5' },
  { value: 'fa_pe_u10', label: '< 10' },
  { value: 'fa_pe_u15', label: '< 15' },
  { value: 'fa_pe_u20', label: '< 20' },
  { value: 'fa_pe_u25', label: '< 25' },
  { value: 'fa_pe_u30', label: '< 30' },
  { value: 'fa_pe_u40', label: '< 40' },
  { value: 'fa_pe_u50', label: '< 50' },
];

const FORWARD_PE_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'fa_fpe_u5', label: '< 5' },
  { value: 'fa_fpe_u10', label: '< 10' },
  { value: 'fa_fpe_u15', label: '< 15' },
  { value: 'fa_fpe_u20', label: '< 20' },
  { value: 'fa_fpe_u25', label: '< 25' },
];

const RSI_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'ta_rsi_os30', label: 'Oversold < 30' },
  { value: 'ta_rsi_os40', label: 'Weak < 40' },
  { value: 'ta_rsi_os50', label: 'Below Mid < 50' },
  { value: 'ta_rsi_ob60', label: 'Above Mid > 60' },
  { value: 'ta_rsi_ob70', label: 'Strong > 70' },
  { value: 'ta_rsi_ob80', label: 'Overbought > 80' },
];

const DIV_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'fa_div_o1', label: '> 1%' },
  { value: 'fa_div_o2', label: '> 2%' },
  { value: 'fa_div_o3', label: '> 3%' },
  { value: 'fa_div_o4', label: '> 4%' },
  { value: 'fa_div_o5', label: '> 5%' },
  { value: 'fa_div_high', label: 'High' },
];

const PB_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'fa_pb_u1', label: '< 1' },
  { value: 'fa_pb_u2', label: '< 2' },
  { value: 'fa_pb_u3', label: '< 3' },
  { value: 'fa_pb_u5', label: '< 5' },
];

const SECTOR_OPTIONS = [
  { value: '', label: 'All Sectors' },
  { value: 'sec_technology', label: 'Technology' },
  { value: 'sec_healthcare', label: 'Healthcare' },
  { value: 'sec_financials', label: 'Financials' },
  { value: 'sec_energy', label: 'Energy' },
  { value: 'sec_utilities', label: 'Utilities' },
  { value: 'sec_consumerdiscretionary', label: 'Cons. Discretionary' },
  { value: 'sec_consumerstaples', label: 'Cons. Staples' },
  { value: 'sec_industrials', label: 'Industrials' },
  { value: 'sec_realestate', label: 'Real Estate' },
  { value: 'sec_materials', label: 'Materials' },
  { value: 'sec_communicationservices', label: 'Communication Svcs' },
];

const COUNTRY_OPTIONS = [
  { value: '', label: 'All Countries' },
  { value: 'geo_usa', label: 'USA' },
  { value: 'geo_notusa', label: 'International' },
  { value: 'geo_canada', label: 'Canada' },
  { value: 'geo_uk', label: 'United Kingdom' },
  { value: 'geo_china', label: 'China' },
  { value: 'geo_japan', label: 'Japan' },
];

const SMA_OPTIONS = [
  { value: 'ta_sma20_pa', label: 'Above SMA 20' },
  { value: 'ta_sma20_pb', label: 'Below SMA 20' },
  { value: 'ta_sma50_pa', label: 'Above SMA 50' },
  { value: 'ta_sma50_pb', label: 'Below SMA 50' },
  { value: 'ta_sma200_pa', label: 'Above SMA 200' },
  { value: 'ta_sma200_pb', label: 'Below SMA 200' },
];

const EPS_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'fa_epsqoq_pos', label: 'Positive QoQ EPS Growth' },
  { value: 'fa_epsyoy_pos', label: 'Positive YoY EPS Growth' },
  { value: 'fa_epsqoq_neg', label: 'Negative QoQ EPS Growth' },
];

const SALES_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'fa_salesqoq_pos', label: 'Positive QoQ Revenue Growth' },
  { value: 'fa_salesqoq_neg', label: 'Negative QoQ Revenue Growth' },
];

const CHANGE_OPTIONS = [
  { value: '', label: 'Any' },
  { value: 'ta_change_u5', label: 'Up > 5%' },
  { value: 'ta_change_u2', label: 'Up > 2%' },
  { value: 'ta_change_u', label: 'Up (any)' },
  { value: 'ta_change_d', label: 'Down (any)' },
  { value: 'ta_change_d2', label: 'Down > 2%' },
  { value: 'ta_change_d5', label: 'Down > 5%' },
];

// ────────────────────────────────────────────────────────────
// Presets
// ────────────────────────────────────────────────────────────

interface Preset {
  id: string;
  label: string;
  icon: LucideIcon;
  desc: string;
  filters: string[];
  direction: 'bullish' | 'bearish' | 'neutral';
}

const PRESETS: Preset[] = [
  {
    id: 'oversold_reversal',
    label: 'Oversold Reversal',
    icon: TrendingDown,
    desc: 'RSI < 30, price up today, high relative volume — potential bounce play',
    filters: ['sh_price_o5', 'sh_relvol_o2', 'ta_change_u', 'ta_rsi_os30'],
    direction: 'bullish',
  },
  {
    id: 'bounce_ma',
    label: 'Bounce at MA',
    icon: Target,
    desc: 'Price above SMA20 but below SMA50 with high volume — mean reversion setup',
    filters: ['sh_avgvol_o400', 'sh_curvol_o2000', 'sh_relvol_o1', 'ta_sma20_pa', 'ta_sma50_pb'],
    direction: 'bullish',
  },
  {
    id: 'breaking_out',
    label: 'Breaking Out',
    icon: ArrowUpRight,
    desc: 'New 50d high, above all SMAs, low debt, ROE > 20%',
    filters: ['fa_debteq_u1', 'fa_roe_o20', 'sh_avgvol_o100', 'ta_highlow50d_nh', 'ta_sma20_pa', 'ta_sma200_pa', 'ta_sma50_pa'],
    direction: 'bullish',
  },
  {
    id: 'sma_crossover',
    label: 'SMA Crossover',
    icon: GitMerge,
    desc: 'SMA50 just crossed above SMA20, profitable, low short interest',
    filters: ['fa_pe_profitable', 'sh_avgvol_o400', 'sh_relvol_o1', 'sh_short_low', 'ta_beta_o1', 'ta_sma50_cross20b'],
    direction: 'bullish',
  },
  {
    id: 'new_highs',
    label: 'New Highs',
    icon: ChevronsUp,
    desc: '52w/50d/20d new highs simultaneously, analyst buy rating, up today',
    filters: ['an_recom_buy', 'sh_price_u7', 'ta_change_u', 'ta_highlow20d_nh', 'ta_highlow50d_nh', 'ta_highlow52w_nh', 'ta_perf_dup'],
    direction: 'bullish',
  },
  {
    id: 'uptrend',
    label: 'Potential Uptrend',
    icon: BarChart2,
    desc: 'Channel-up pattern, dipped last week — buying the dip in an uptrend',
    filters: ['sh_avgvol_o400', 'ta_pattern_channelup', 'ta_perf_1wdown'],
    direction: 'bullish',
  },
  {
    id: 'canslim',
    label: 'CANSLIM',
    icon: TrendingUp,
    desc: "O'Neil method: 20%+ EPS & revenue growth both QoQ and over 5 years",
    filters: ['fa_eps5years_o20', 'fa_epsqoq_o20', 'fa_epsyoy_o20', 'fa_sales5years_o20', 'fa_salesqoq_o20', 'sh_curvol_o200'],
    direction: 'bullish',
  },
  {
    id: 'high_earnings',
    label: 'High EPS Growth',
    icon: Activity,
    desc: '25%+ EPS growth QoQ/YoY/forward estimates, above SMA200',
    filters: ['fa_epsqoq_o25', 'fa_epsyoy_o25', 'fa_epsyoy1_o25', 'fa_salesqoq_o25', 'sh_avgvol_o400', 'ta_rsi_nos50', 'ta_sma200_pa'],
    direction: 'bullish',
  },
  {
    id: 'consistent_growth',
    label: 'Consistent Growth',
    icon: Leaf,
    desc: 'Multi-year EPS growth, ROE >15%, near 52w high, institutional backing',
    filters: ['fa_eps5years_pos', 'fa_epsqoq_o20', 'fa_epsyoy_o25', 'fa_epsyoy1_o15', 'fa_estltgrowth_pos', 'fa_roe_o15', 'sh_instown_o10', 'sh_price_o15', 'ta_highlow52w_a90h', 'ta_rsi_nos50'],
    direction: 'bullish',
  },
  {
    id: 'high_sales',
    label: 'High Sales Growth',
    icon: DollarSign,
    desc: '20%+ revenue growth 5yr + QoQ, low debt, 60%+ institutional ownership',
    filters: ['fa_debteq_u0.5', 'fa_roe_o15', 'fa_sales5years_o20', 'fa_salesqoq_o20', 'sh_avgvol_o200', 'sh_instown_o60', 'sh_price_o5', 'sh_short_u5'],
    direction: 'bullish',
  },
  {
    id: 'undervalued_dividend',
    label: 'Undervalued Div',
    icon: Star,
    desc: 'Large cap, P/E < 20, low PEG, pays dividend, payout ratio < 50%',
    filters: ['cap_largeover', 'fa_div_pos', 'fa_epsyoy1_o5', 'fa_estltgrowth_o5', 'fa_payoutratio_u50', 'fa_pe_u20', 'fa_peg_low'],
    direction: 'bullish',
  },
  {
    id: 'high_dividend',
    label: 'High Dividend',
    icon: Wallet,
    desc: 'Dividend yield > 3%, Mid+ cap',
    filters: ['fa_div_o3', 'cap_midover'],
    direction: 'bullish',
  },
  {
    id: 'bluechip',
    label: 'Blue Chip',
    icon: Shield,
    desc: 'Mega cap (> $200B) market leaders',
    filters: ['cap_mega'],
    direction: 'neutral',
  },
  {
    id: 'short_squeeze',
    label: 'Short Squeeze',
    icon: Flame,
    desc: 'Short float > 15%, low institutional ownership — squeeze candidates',
    filters: ['sh_avgvol_o100', 'sh_instown_u50', 'sh_price_o2', 'sh_short_o15'],
    direction: 'bearish',
  },
];

// ────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────

interface CustomFilters {
  marketCap: string;
  peMax: string;
  forwardPEMax: string;
  pbMax: string;
  rsi: string;
  dividendMin: string;
  sector: string;
  country: string;
  epsGrowth: string;
  salesGrowth: string;
  priceChange: string;
  sma: string[];
}

const DEFAULT_FILTERS: CustomFilters = {
  marketCap: '',
  peMax: '',
  forwardPEMax: '',
  pbMax: '',
  rsi: '',
  dividendMin: '',
  sector: '',
  country: '',
  epsGrowth: '',
  salesGrowth: '',
  priceChange: '',
  sma: [],
};

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

function buildFilterString(preset: Preset | null, custom: CustomFilters): string {
  if (preset) return preset.filters.join(',');

  const codes: string[] = [];
  if (custom.marketCap) codes.push(custom.marketCap);
  if (custom.peMax) codes.push(custom.peMax);
  if (custom.forwardPEMax) codes.push(custom.forwardPEMax);
  if (custom.pbMax) codes.push(custom.pbMax);
  if (custom.rsi) codes.push(custom.rsi);
  if (custom.dividendMin) codes.push(custom.dividendMin);
  if (custom.sector) codes.push(custom.sector);
  if (custom.country) codes.push(custom.country);
  if (custom.epsGrowth) codes.push(custom.epsGrowth);
  if (custom.salesGrowth) codes.push(custom.salesGrowth);
  if (custom.priceChange) codes.push(custom.priceChange);
  codes.push(...custom.sma);
  return codes.join(',');
}

function formatMarketCap(raw: string) {
  if (!raw || raw === '-') return '-';
  return raw;
}

function changeColor(val: string) {
  if (!val || val === '-') return 'var(--text-4)';
  const n = parseFloat(val.replace('%', '').replace('+', ''));
  if (isNaN(n)) return 'var(--text-4)';
  return n > 0 ? 'var(--green-1)' : n < 0 ? 'var(--red-1)' : 'var(--text-4)';
}

// ────────────────────────────────────────────────────────────
// Sub-components
// ────────────────────────────────────────────────────────────

function SelectField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (v: string) => void;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label style={{ color: 'var(--text-5)', fontSize: 10, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
        {label}
      </label>
      <select
        value={value}
        onChange={e => onChange(e.target.value)}
        className="border px-2 py-1.5 text-xs font-mono"
        style={{
          background: 'var(--bg-4)',
          borderColor: value ? 'var(--accent)' : 'var(--bg-1)',
          color: value ? 'var(--text-2)' : 'var(--text-4)',
          outline: 'none',
          minWidth: 130,
        }}
      >
        {options.map(o => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function SmaToggle({
  value,
  selected,
  onToggle,
}: {
  value: string;
  selected: boolean;
  onToggle: () => void;
}) {
  const label = SMA_OPTIONS.find(o => o.value === value)?.label ?? value;
  return (
    <button
      onClick={onToggle}
      className="px-2 py-1 text-[10px] border transition-all"
      style={{
        background: selected ? 'rgba(var(--accent-rgb, 100,200,100), 0.15)' : 'var(--bg-4)',
        borderColor: selected ? 'var(--accent)' : 'var(--bg-1)',
        color: selected ? 'var(--accent)' : 'var(--text-4)',
      }}
    >
      {label}
    </button>
  );
}

// ────────────────────────────────────────────────────────────
// Main Component
// ────────────────────────────────────────────────────────────

export default function StockScreener({ onSelectTicker }: Props) {
  const [activePreset, setActivePreset] = useState<Preset | null>(null);
  const [customFilters, setCustomFilters] = useState<CustomFilters>(DEFAULT_FILTERS);
  const [result, setResult] = useState<ScreenerResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [sortCol, setSortCol] = useState<string>('');
  const [sortAsc, setSortAsc] = useState(true);
  const [showCustom, setShowCustom] = useState(false);

  const runScreener = useCallback(
    async (preset: Preset | null, custom: CustomFilters, page = 1) => {
      setLoading(true);
      setError(null);
      setCurrentPage(page);

      const filters = buildFilterString(preset, custom);
      try {
        const params = new URLSearchParams({ filters, page: String(page) });
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
    },
    []
  );

  const handlePreset = (preset: Preset) => {
    const next = activePreset?.id === preset.id ? null : preset;
    setActivePreset(next);
    setShowCustom(false);
    if (next) runScreener(next, customFilters, 1);
  };

  const handleRun = () => {
    runScreener(activePreset, customFilters, 1);
  };

  const handleReset = () => {
    setActivePreset(null);
    setCustomFilters(DEFAULT_FILTERS);
    setResult(null);
    setError(null);
  };

  const handleSort = (col: string) => {
    if (sortCol === col) setSortAsc(a => !a);
    else { setSortCol(col); setSortAsc(true); }
  };

  const setCustom = (key: keyof CustomFilters, val: string | string[]) => {
    setActivePreset(null);
    setCustomFilters(prev => ({ ...prev, [key]: val }));
  };

  const toggleSma = (val: string) => {
    setActivePreset(null);
    setCustomFilters(prev => ({
      ...prev,
      sma: prev.sma.includes(val) ? prev.sma.filter(s => s !== val) : [...prev.sma, val],
    }));
  };

  const activeFilterCount =
    (activePreset ? activePreset.filters.length : 0) +
    (!activePreset
      ? Object.entries(customFilters)
          .filter(([k, v]) => k !== 'sma' && v !== '')
          .length + customFilters.sma.length
      : 0);

  // Sort rows client-side
  const displayRows = result
    ? sortCol
      ? [...result.rows].sort((a, b) => {
          const av = parseFloat((a[sortCol] || '').replace(/[^0-9.-]/g, '')) || 0;
          const bv = parseFloat((b[sortCol] || '').replace(/[^0-9.-]/g, '')) || 0;
          return sortAsc ? av - bv : bv - av;
        })
      : result.rows
    : [];

  const DISPLAY_COLS = ['Ticker', 'Company', 'Sector', 'Market Cap', 'P/E', 'Price', 'Change', 'Volume'];
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

      {/* ── Custom Filters Toggle ─────────────────────── */}
      <div className="flex items-center gap-3 mb-3">
        <button
          onClick={() => setShowCustom(v => !v)}
          className="flex items-center gap-2 px-3 py-1.5 border transition-all text-xs"
          style={{
            background: showCustom ? 'var(--bg-2)' : 'var(--bg-4)',
            borderColor: showCustom ? 'var(--accent)' : 'var(--bg-1)',
            color: showCustom ? 'var(--accent)' : 'var(--text-4)',
          }}
        >
          <Filter className="w-3.5 h-3.5" />
          Custom Filters
          {!activePreset && activeFilterCount > 0 && (
            <span
              className="px-1.5 py-0.5 text-[9px] font-bold rounded-full"
              style={{ background: 'var(--accent)', color: 'var(--text-0)' }}
            >
              {activeFilterCount}
            </span>
          )}
        </button>

        <div className="flex items-center gap-2 ml-auto">
          {activeFilterCount > 0 && (
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 px-3 py-1.5 border text-xs transition-all"
              style={{
                background: 'var(--bg-4)',
                borderColor: 'var(--bg-1)',
                color: 'var(--text-4)',
              }}
            >
              <RotateCcw className="w-3 h-3" />
              Reset
            </button>
          )}
          <button
            onClick={handleRun}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-1.5 border transition-all text-xs font-semibold disabled:opacity-50"
            style={{
              background: 'var(--accent)',
              borderColor: 'var(--accent)',
              color: 'var(--text-0)',
            }}
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
      </div>

      {/* ── Custom Filters Panel ─────────────────────── */}
      {showCustom && (
        <div
          className="mb-4 p-4 border"
          style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}
        >
          {activePreset && (
            <div
              className="mb-3 px-3 py-2 border text-xs flex items-center gap-2"
              style={{
                borderColor: 'var(--warning)',
                background: 'rgba(200,150,0,0.08)',
                color: 'var(--warning)',
              }}
            >
              <Zap className="w-3 h-3" />
              Preset "{activePreset.label}" is active. Selecting a custom filter will clear it.
            </div>
          )}

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
            {/* Market Cap */}
            <SelectField
              label="Market Cap"
              value={activePreset ? '' : customFilters.marketCap}
              options={CAP_OPTIONS}
              onChange={v => setCustom('marketCap', v)}
            />

            {/* P/E Max */}
            <SelectField
              label="Max P/E"
              value={activePreset ? '' : customFilters.peMax}
              options={PE_OPTIONS}
              onChange={v => setCustom('peMax', v)}
            />

            {/* Forward P/E */}
            <SelectField
              label="Max Fwd P/E"
              value={activePreset ? '' : customFilters.forwardPEMax}
              options={FORWARD_PE_OPTIONS}
              onChange={v => setCustom('forwardPEMax', v)}
            />

            {/* P/B */}
            <SelectField
              label="Max P/Book"
              value={activePreset ? '' : customFilters.pbMax}
              options={PB_OPTIONS}
              onChange={v => setCustom('pbMax', v)}
            />

            {/* RSI */}
            <SelectField
              label="RSI (14)"
              value={activePreset ? '' : customFilters.rsi}
              options={RSI_OPTIONS}
              onChange={v => setCustom('rsi', v)}
            />

            {/* Dividend */}
            <SelectField
              label="Min Div Yield"
              value={activePreset ? '' : customFilters.dividendMin}
              options={DIV_OPTIONS}
              onChange={v => setCustom('dividendMin', v)}
            />

            {/* Sector */}
            <SelectField
              label="Sector"
              value={activePreset ? '' : customFilters.sector}
              options={SECTOR_OPTIONS}
              onChange={v => setCustom('sector', v)}
            />

            {/* Country */}
            <SelectField
              label="Country"
              value={activePreset ? '' : customFilters.country}
              options={COUNTRY_OPTIONS}
              onChange={v => setCustom('country', v)}
            />

            {/* EPS Growth */}
            <SelectField
              label="EPS Growth"
              value={activePreset ? '' : customFilters.epsGrowth}
              options={EPS_OPTIONS}
              onChange={v => setCustom('epsGrowth', v)}
            />

            {/* Sales Growth */}
            <SelectField
              label="Revenue Growth"
              value={activePreset ? '' : customFilters.salesGrowth}
              options={SALES_OPTIONS}
              onChange={v => setCustom('salesGrowth', v)}
            />

            {/* Price Change */}
            <SelectField
              label="Daily Change"
              value={activePreset ? '' : customFilters.priceChange}
              options={CHANGE_OPTIONS}
              onChange={v => setCustom('priceChange', v)}
            />
          </div>

          {/* SMA toggles */}
          <div className="mt-4">
            <div
              className="text-[10px] uppercase tracking-widest mb-2"
              style={{ color: 'var(--text-5)' }}
            >
              Price vs Moving Average
            </div>
            <div className="flex flex-wrap gap-1.5">
              {SMA_OPTIONS.map(o => (
                <SmaToggle
                  key={o.value}
                  value={o.value}
                  selected={!activePreset && customFilters.sma.includes(o.value)}
                  onToggle={() => toggleSma(o.value)}
                />
              ))}
            </div>
          </div>

          {/* Active filter string preview */}
          {!activePreset && buildFilterString(null, customFilters) && (
            <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--bg-1)' }}>
              <span className="text-[10px] uppercase tracking-widest" style={{ color: 'var(--text-5)' }}>
                FinViz filter string:{' '}
              </span>
              <code className="text-[10px]" style={{ color: 'var(--text-4)' }}>
                {buildFilterString(null, customFilters)}
              </code>
            </div>
          )}
        </div>
      )}

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
                  onClick={() => runScreener(activePreset, customFilters, Math.max(1, currentPage - 1))}
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
                      onClick={() => runScreener(activePreset, customFilters, pg)}
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
                  onClick={() =>
                    runScreener(activePreset, customFilters, Math.min(result.totalPages, currentPage + 1))
                  }
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
                        onMouseEnter={e =>
                          (e.currentTarget.style.background = 'var(--hover)')
                        }
                        onMouseLeave={e =>
                          (e.currentTarget.style.background =
                            i % 2 === 0 ? 'transparent' : 'rgba(0,0,0,0.1)')
                        }
                      >
                        {TABLE_COLS.map(col => {
                          const val = row[col] || '-';
                          const isTicker = col === 'Ticker';
                          const isChange =
                            col === 'Change' || col === 'Chg' || col.includes('Perf');

                          if (isTicker) {
                            return (
                              <td key={col} className="px-3 py-2">
                                <button
                                  onClick={() => onSelectTicker(ticker)}
                                  className="font-bold font-mono transition-colors hover:opacity-80"
                                  style={{ color: 'var(--accent)', textDecoration: 'none' }}
                                >
                                  {val}
                                </button>
                              </td>
                            );
                          }

                          if (isChange) {
                            return (
                              <td key={col} className="px-3 py-2 font-mono" style={{ color: changeColor(val), fontVariantNumeric: 'tabular-nums' }}>
                                {val}
                              </td>
                            );
                          }

                          if (col === 'Company' || col === 'Name') {
                            return (
                              <td key={col} className="px-3 py-2 max-w-[180px]" style={{ color: 'var(--text-3)' }}>
                                <span className="truncate block" title={val}>{val}</span>
                              </td>
                            );
                          }

                          if (col === 'Market Cap') {
                            return (
                              <td key={col} className="px-3 py-2 font-mono" style={{ color: 'var(--text-3)', fontVariantNumeric: 'tabular-nums' }}>
                                {formatMarketCap(val)}
                              </td>
                            );
                          }

                          return (
                            <td key={col} className="px-3 py-2 font-mono" style={{ color: 'var(--text-4)', fontVariantNumeric: 'tabular-nums' }}>
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
              <div className="text-sm">No stocks match the current filters.</div>
              <div className="text-xs mt-1">Try loosening your criteria.</div>
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
          <div className="text-sm mb-1">Select a preset or build custom filters, then run the screener.</div>
          <div className="text-xs">Results are fetched live from FinViz.</div>
        </div>
      )}
    </div>
  );
}
