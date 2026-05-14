'use client';

import React from 'react';
import { Building2, ExternalLink, ArrowUp, ArrowDown } from 'lucide-react';

interface FundamentalsData {
  fundamentals: {
    peRatio: number | null;
    eps: number | null;
    profitMargin: number | null;
    operatingMargin: number | null;
    roe: number | null;
    roa: number | null;
    debtToEquity: number | null;
    currentRatio: number | null;
    revenueGrowth: number | null;
    earningsGrowth: number | null;
    dividendYield: number | null;
    pegRatio: number | null;
    priceToBook: number | null;
  };
}

interface CompanyInfoProps {
  symbol: string;
  companyName: string;
  currentPrice: number;
  currentChange?: number | null;
  currentChangePercent?: number | null;
  fundamentalsData?: FundamentalsData | null;
  fundamentalsLoading?: boolean;
  finvizStock?: Record<string, string | null> | null;
  companyInfo: {
    sector?: string;
    industry?: string;
    website?: string;
    description?: string;
    marketCap?: number;
    trailingPE?: number;
    forwardPE?: number;
    priceToBook?: number;
    revenueGrowth?: number;
    grossMargins?: number;
    profitMargins?: number;
    dividendRate?: number;
    dividendYield?: number;
    beta?: number;
    fiftyTwoWeekHigh?: number;
    fiftyTwoWeekLow?: number;
    fiftyTwoWeekChange?: number;
    averageVolume?: number;
    sharesOutstanding?: number;
    change?: number | null;
    changePercent?: number | null;
    phone?: string;
    address?: {
      address1?: string;
      city?: string;
      state?: string;
      postal_code?: string;
    };
    totalEmployees?: number;
    primaryExchange?: string;
    listDate?: string;
    sicCode?: string;
    sicDescription?: string;
    logoUrl?: string;
    iconUrl?: string;
    cik?: string;
    locale?: string;
    market?: string;
    type?: string;
  };
}

// Finviz keys already covered by companyInfo / fundamentalsData — skip to avoid duplicates
const FV_SKIP = new Set([
  'Ticker', 'Company', 'Website', 'Price', 'Change', 'Volume', 'Avg Volume',
  'Market Cap', 'P/E', 'Forward P/E', 'PEG', 'EPS (ttm)', 'P/B',
  'ROE', 'ROA', 'Profit Margin', 'Oper. Margin', 'Gross Margin',
  'Debt/Eq', 'Current Ratio', 'Beta',
  '52W High', '52W Low', 'Dividend %', 'Dividend',
]);

const FV_VALUATION    = ['P/S', 'P/C', 'P/FCF'];
const FV_PROFITABILITY = ['ROI'];
const FV_HEALTH       = ['LT Debt/Eq', 'Quick Ratio', 'EPS next Y', 'EPS next Q', 'EPS this Y', 'EPS past 5Y', 'EPS next 5Y', 'Sales past 5Y', 'Sales Q/Q', 'EPS Q/Q'];
const FV_MARKET       = ['Volume', 'Prev Close', 'Target Price', 'Recom', 'Payout', '52W Range', 'ATR', 'Avg Volume'];
const FV_OWNERSHIP    = ['Insider Own', 'Insider Trans', 'Inst Own', 'Inst Trans', 'Short Float', 'Short Ratio', 'Shs Float', 'Shs Outstand', 'Optionable', 'Shortable'];
const FV_TECHNICAL    = ['RSI (14)', 'SMA20', 'SMA50', 'SMA200', 'Volatility', 'Rel Volume', 'Earnings'];
const FV_PERFORMANCE  = ['Perf Week', 'Perf Month', 'Perf Quarter', 'Perf Half Y', 'Perf Year', 'Perf YTD'];

export default function CompanyInfo({
  symbol, companyName, currentPrice, currentChange, currentChangePercent,
  companyInfo, fundamentalsData, fundamentalsLoading, finvizStock,
}: CompanyInfoProps) {
  const [imageError, setImageError] = React.useState(false);

  React.useEffect(() => { setImageError(false); }, [symbol, companyInfo.iconUrl]);

  const change = typeof currentChange === 'number' ? currentChange : companyInfo?.change ?? null;
  const changePercent = typeof currentChangePercent === 'number' ? currentChangePercent : companyInfo?.changePercent ?? null;

  const fmt = (num: number | null | undefined, decimals = 2): string => {
    if (num == null) return 'N/A';
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const fmtCap = (num: number | null | undefined): string => {
    if (num == null) return 'N/A';
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9)  return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6)  return `$${(num / 1e6).toFixed(2)}M`;
    return `$${num.toLocaleString()}`;
  };

  const fmtPct = (num: number | null | undefined): string => {
    if (num == null) return 'N/A';
    return `${(num * 100).toFixed(2)}%`;
  };

  const fmtVol = (num: number | null | undefined): string => {
    if (num == null) return 'N/A';
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
    return num.toLocaleString();
  };

  // Render a single row from the Finviz stock record
  const fv = (key: string) => {
    if (!finvizStock) return null;
    const val = finvizStock[key];
    if (!val || val === '-' || val === '') return null;
    return (
      <div key={key} className="flex justify-between">
        <span style={{ color: 'var(--text-4)' }}>{key}:</span>
        <span style={{ color: 'var(--text-2)' }}>{val}</span>
      </div>
    );
  };

  // Render a list of Finviz keys, returning null if all are empty
  const fvSection = (keys: string[]) => {
    const rows = keys.map(fv).filter(Boolean);
    return rows.length > 0 ? rows : null;
  };

  const hasData = !!(fundamentalsData || finvizStock);

  return (
    <div className="card">
      <span className="card-label">Company Overview</span>

      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-3">
            {companyInfo.iconUrl && !imageError ? (
              <img
                src={companyInfo.iconUrl}
                alt={`${companyName} icon`}
                className="w-12 h-12 rounded-lg object-contain"
                style={{ background: 'var(--bg-3)', padding: '4px' }}
                onError={() => setImageError(true)}
              />
            ) : (
              <Building2 className="w-6 h-6" style={{ color: 'var(--accent)' }} />
            )}
            <div>
              <h2 className="text-2xl font-bold" style={{ color: 'var(--text-1)' }}>{companyName}</h2>
              <p className="text-sm font-mono" style={{ color: 'var(--text-4)' }}>
                {symbol}
                {companyInfo.primaryExchange && (
                  <span className="ml-2 text-xs" style={{ color: 'var(--text-5)' }}>· {companyInfo.primaryExchange}</span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 md:gap-4">
            <div className="flex items-center gap-2 md:gap-3">
              {typeof change === 'number' && typeof changePercent === 'number' && (
                <div className="flex items-center text-xs md:text-sm font-mono" style={{ color: change > 0 ? 'var(--success)' : 'var(--danger)' }}>
                  {change > 0 ? <ArrowUp className="w-3 h-3 md:w-4 md:h-4" /> : <ArrowDown className="w-3 h-3 md:w-4 md:h-4" />}
                  <span className="ml-1">{change > 0 ? '+' : ''}{change.toFixed(2)} ({changePercent.toFixed(2)}%)</span>
                </div>
              )}
              <div className="text-2xl md:text-4xl font-bold font-mono" style={{ color: 'var(--accent)' }}>
                ${currentPrice.toFixed(2)}
              </div>
            </div>
            {companyInfo.website && (
              <a
                href={companyInfo.website}
                target="_blank"
                rel="noopener noreferrer"
                className="hidden md:flex items-center gap-2 px-3 py-2 border transition-colors"
                style={{ color: 'var(--accent)', borderColor: 'var(--accent)', background: 'var(--bg-3)' }}
              >
                <ExternalLink className="w-4 h-4" />
                <span className="text-sm">Website</span>
              </a>
            )}
          </div>
        </div>

        {/* Metadata row */}
        <div className="flex flex-wrap gap-4 mt-3 text-xs" style={{ color: 'var(--text-4)' }}>
          {companyInfo.sector && (
            <div className="flex items-center gap-1">
              <span>Sector:</span>
              <span style={{ color: 'var(--text-3)' }}>{companyInfo.sector}</span>
            </div>
          )}
          {companyInfo.industry && (
            <div className="flex items-center gap-1">
              <span>Industry:</span>
              <span style={{ color: 'var(--text-3)' }}>{companyInfo.industry}</span>
            </div>
          )}
          {finvizStock?.['Index'] && finvizStock['Index'] !== '-' && (
            <div className="flex items-center gap-1">
              <span>Index:</span>
              <span style={{ color: 'var(--text-3)' }}>{finvizStock['Index']}</span>
            </div>
          )}
          {companyInfo.totalEmployees && (
            <div className="flex items-center gap-1">
              <span>Employees:</span>
              <span style={{ color: 'var(--text-3)' }}>{companyInfo.totalEmployees.toLocaleString()}</span>
            </div>
          )}
          {companyInfo.listDate && (
            <div className="flex items-center gap-1">
              <span>Listed:</span>
              <span style={{ color: 'var(--text-3)' }}>{new Date(companyInfo.listDate).getFullYear()}</span>
            </div>
          )}
          {companyInfo.phone && (
            <div className="flex items-center gap-1">
              <span>Phone:</span>
              <span style={{ color: 'var(--text-3)' }}>{companyInfo.phone}</span>
            </div>
          )}
        </div>

        {companyInfo.address && (
          <div className="mt-2 text-xs" style={{ color: 'var(--text-4)' }}>
            <span>Address: </span>
            <span style={{ color: 'var(--text-3)' }}>
              {companyInfo.address.address1}
              {companyInfo.address.city && `, ${companyInfo.address.city}`}
              {companyInfo.address.state && `, ${companyInfo.address.state}`}
              {companyInfo.address.postal_code && ` ${companyInfo.address.postal_code}`}
            </span>
          </div>
        )}

        {companyInfo.description && (
          <p className="text-sm mt-3 leading-relaxed" style={{ color: 'var(--text-3)' }}>
            {companyInfo.description.substring(0, 1000)}
          </p>
        )}

        {companyInfo.trailingPE && (
          <div className="mt-3 p-3 border-2" style={{ background: 'var(--bg-2)', borderColor: '#8b5cf6', borderLeftWidth: '3px' }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>P/E Ratio</div>
            <div className="text-sm font-semibold font-mono" style={{ color: '#8b5cf6' }}>{fmt(companyInfo.trailingPE)}</div>
          </div>
        )}
      </div>

      {/* Metrics Sections */}
      {hasData && (
        <div className="mt-6 pt-6 border-t" style={{ borderColor: 'var(--bg-1)' }}>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4 text-sm">

            {/* Valuation */}
            <div>
              <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Valuation</div>
              <div className="space-y-1 text-xs">
                {companyInfo.forwardPE != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Fwd P/E:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fmt(companyInfo.forwardPE)}</span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.pegRatio && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>PEG:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fundamentalsData.fundamentals.pegRatio.toFixed(2)}</span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.eps != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>EPS:</span>
                    <span style={{ color: 'var(--text-2)' }}>${fmt(fundamentalsData.fundamentals.eps)}</span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.priceToBook && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>P/B:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fundamentalsData.fundamentals.priceToBook.toFixed(2)}</span>
                  </div>
                )}
                {fvSection(FV_VALUATION)}
                {companyInfo.dividendRate != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Div Rate:</span>
                    <span style={{ color: 'var(--text-2)' }}>${fmt(companyInfo.dividendRate)}</span>
                  </div>
                )}
                {companyInfo.dividendYield != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Div Yield:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fmtPct(companyInfo.dividendYield)}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Profitability */}
            <div>
              <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Profitability</div>
              <div className="space-y-1 text-xs">
                {companyInfo.grossMargins != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Gross Mgn:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fmtPct(companyInfo.grossMargins)}</span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.profitMargin && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Net Mgn:</span>
                    <span style={{ color: 'var(--text-2)' }}>{(fundamentalsData.fundamentals.profitMargin * 100).toFixed(2)}%</span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.operatingMargin && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Op Mgn:</span>
                    <span style={{ color: 'var(--text-2)' }}>{(fundamentalsData.fundamentals.operatingMargin * 100).toFixed(2)}%</span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.roe && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>ROE:</span>
                    <span style={{ color: 'var(--text-2)' }}>{(fundamentalsData.fundamentals.roe * 100).toFixed(2)}%</span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.roa && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>ROA:</span>
                    <span style={{ color: 'var(--text-2)' }}>{(fundamentalsData.fundamentals.roa * 100).toFixed(2)}%</span>
                  </div>
                )}
                {fvSection(FV_PROFITABILITY)}
              </div>
            </div>

            {/* Health & Growth */}
            <div>
              <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Health & Growth</div>
              <div className="space-y-1 text-xs">
                {fundamentalsData?.fundamentals.debtToEquity && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>D/E:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fundamentalsData.fundamentals.debtToEquity.toFixed(2)}</span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.currentRatio && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Curr Ratio:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fundamentalsData.fundamentals.currentRatio.toFixed(2)}</span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.revenueGrowth && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Rev Growth:</span>
                    <span style={{ color: fundamentalsData.fundamentals.revenueGrowth > 0 ? 'var(--success)' : 'var(--danger)' }}>
                      {(fundamentalsData.fundamentals.revenueGrowth * 100).toFixed(2)}%
                    </span>
                  </div>
                )}
                {fundamentalsData?.fundamentals.earningsGrowth && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>EPS Growth:</span>
                    <span style={{ color: fundamentalsData.fundamentals.earningsGrowth > 0 ? 'var(--success)' : 'var(--danger)' }}>
                      {(fundamentalsData.fundamentals.earningsGrowth * 100).toFixed(2)}%
                    </span>
                  </div>
                )}
                {fvSection(FV_HEALTH)}
              </div>
            </div>

            {/* Market */}
            <div>
              <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Market</div>
              <div className="space-y-1 text-xs">
                {companyInfo.marketCap != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Mkt Cap:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fmtCap(companyInfo.marketCap)}</span>
                  </div>
                )}
                {companyInfo.sharesOutstanding != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Shares:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fmtVol(companyInfo.sharesOutstanding)}</span>
                  </div>
                )}
                {companyInfo.averageVolume != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Avg Vol:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fmtVol(companyInfo.averageVolume)}</span>
                  </div>
                )}
                {companyInfo.beta != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>Beta:</span>
                    <span style={{ color: 'var(--text-2)' }}>{fmt(companyInfo.beta)}</span>
                  </div>
                )}
                {companyInfo.fiftyTwoWeekChange != null && (
                  <div className="flex justify-between gap-2">
                    <span style={{ color: 'var(--text-4)' }}>52W Chg:</span>
                    <span style={{ color: companyInfo.fiftyTwoWeekChange > 0 ? 'var(--success)' : 'var(--danger)' }}>
                      {fmtPct(companyInfo.fiftyTwoWeekChange)}
                    </span>
                  </div>
                )}
                {fvSection(FV_MARKET)}
              </div>
            </div>

            {/* Ownership (Finviz) */}
            {finvizStock && fvSection(FV_OWNERSHIP) && (
              <div>
                <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Ownership</div>
                <div className="space-y-1 text-xs">{fvSection(FV_OWNERSHIP)}</div>
              </div>
            )}

            {/* Technical */}
            {(companyInfo.fiftyTwoWeekHigh != null || companyInfo.fiftyTwoWeekLow != null || (finvizStock && fvSection(FV_TECHNICAL))) && (
              <div>
                <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Technical</div>
                <div className="space-y-1 text-xs">
                  {companyInfo.fiftyTwoWeekHigh != null && (
                    <div className="flex justify-between gap-2">
                      <span style={{ color: 'var(--text-4)' }}>52W High:</span>
                      <span style={{ color: 'var(--text-2)' }}>${fmt(companyInfo.fiftyTwoWeekHigh)}</span>
                    </div>
                  )}
                  {companyInfo.fiftyTwoWeekLow != null && (
                    <div className="flex justify-between gap-2">
                      <span style={{ color: 'var(--text-4)' }}>52W Low:</span>
                      <span style={{ color: 'var(--text-2)' }}>${fmt(companyInfo.fiftyTwoWeekLow)}</span>
                    </div>
                  )}
                  {finvizStock && fvSection(FV_TECHNICAL)}
                </div>
              </div>
            )}

            {/* Performance (Finviz) */}
            {finvizStock && fvSection(FV_PERFORMANCE) && (
              <div>
                <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Performance</div>
                <div className="space-y-1 text-xs">{fvSection(FV_PERFORMANCE)}</div>
              </div>
            )}
          </div>
        </div>
      )}

      {fundamentalsLoading && !fundamentalsData && (
        <div className="mt-6 pt-6 border-t" style={{ borderColor: 'var(--bg-1)' }}>
          <div className="flex items-center justify-center py-4">
            <div className="animate-spin rounded-full h-6 w-6 border-b-2" style={{ borderColor: 'var(--accent)' }} />
            <span className="ml-2 text-sm" style={{ color: 'var(--text-4)' }}>Loading fundamentals...</span>
          </div>
        </div>
      )}
    </div>
  );
}
