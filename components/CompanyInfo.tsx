'use client';

import React from 'react';
import { Building2, TrendingUp, DollarSign, BarChart3, Percent, Award, ExternalLink, ArrowUp, ArrowDown } from 'lucide-react';

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
    // optional fields added by API: today's change and percent
    change?: number | null;
    changePercent?: number | null;

    // Massive API fields
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

export default function CompanyInfo({ symbol, companyName, currentPrice, currentChange, currentChangePercent, companyInfo, fundamentalsData, fundamentalsLoading }: CompanyInfoProps) {
  const [imageError, setImageError] = React.useState(false);
  const [showFundamentals, setShowFundamentals] = React.useState(true);

  // Reset image error when symbol changes
  React.useEffect(() => {
    setImageError(false);
  }, [symbol, companyInfo.iconUrl]);

  // Prefer explicit props if provided, otherwise read from companyInfo
  const change = typeof currentChange === 'number' ? currentChange : companyInfo?.change ?? null;
  const changePercent = typeof currentChangePercent === 'number' ? currentChangePercent : companyInfo?.changePercent ?? null;
  const formatNumber = (num: number | null | undefined, decimals: number = 2): string => {
    if (num === null || num === undefined) return 'N/A';
    return num.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  };

  const formatMarketCap = (num: number | null | undefined): string => {
    if (num === null || num === undefined) return 'N/A';
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    return `$${num.toLocaleString()}`;
  };

  const formatPercent = (num: number | null | undefined): string => {
    if (num === null || num === undefined) return 'N/A';
    return `${(num * 100).toFixed(2)}%`;
  };

  const formatVolume = (num: number | null | undefined): string => {
    if (num === null || num === undefined) return 'N/A';
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
    return num.toLocaleString();
  };

  // Determine P/E to display: use trailingP/E from company info (trailingPE) only
  const fundamentalsPe = fundamentalsData?.fundamentals?.peRatio ?? null;

  const trailingPe = companyInfo?.trailingPE ?? null;
  const displayedPe = fundamentalsPe ?? trailingPe;
  const peSource = trailingPe != null ? 'Trailing' : null;

  return (
    <div className="card">
      <span className="card-label">Company Overview</span>

      {/* Header Section */}
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
              <h2 className="text-2xl font-bold" style={{ color: 'var(--text-1)' }}>
                {companyName}
              </h2>
              <p className="text-sm font-mono" style={{ color: 'var(--text-4)' }}>
                {symbol}
                {companyInfo.primaryExchange && (
                  <span className="ml-2 text-xs" style={{ color: 'var(--text-5)' }}>
                    · {companyInfo.primaryExchange}
                  </span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 md:gap-4">
            <div className="flex items-center gap-2 md:gap-3">
              {/* Today's change with arrow to the left of the price */}
              {typeof change === 'number' && typeof changePercent === 'number' ? (
                <div className="flex items-center text-xs md:text-sm font-mono" style={{ color: change > 0 ? 'var(--success)' : 'var(--danger)' }}>
                  {change > 0 ? <ArrowUp className="w-3 h-3 md:w-4 md:h-4" /> : <ArrowDown className="w-3 h-3 md:w-4 md:h-4" />}
                  <span className="ml-1">
                    {change > 0 ? '+' : ''}{change.toFixed(2)} ({changePercent.toFixed(2)}%)
                  </span>
                </div>
              ) : null}

              <div className="text-right">
                <div className="text-2xl md:text-4xl font-bold font-mono" style={{ color: 'var(--accent)' }}>
                  ${currentPrice.toFixed(2)}
                </div>
              </div>
            </div>
            {companyInfo.website && (
              <a
                href={companyInfo.website}
                target="_blank"
                rel="noopener noreferrer"
                className="hidden md:flex items-center gap-2 px-3 py-2 border transition-colors"
                style={{
                  color: 'var(--accent)',
                  borderColor: 'var(--accent)',
                  background: 'var(--bg-3)'
                }}
              >
                <ExternalLink className="w-4 h-4" />
                <span className="text-sm">Website</span>
              </a>
            )}
          </div>
        </div>

        {/* Company Metadata */}
        <div className="flex flex-wrap gap-4 mt-3 text-xs" style={{ color: 'var(--text-4)' }}>
          {companyInfo.sector && (
            <div className="flex items-center gap-1">
              <span>Sector:</span>
              <span style={{ color: 'var(--text-3)' }}>{companyInfo.sector}</span>
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

        {/* Address */}
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

        {/* Description */}
        {companyInfo.description && (
          <p className="text-sm mt-3 leading-relaxed" style={{ color: 'var(--text-3)' }}>
            {companyInfo.description.length > 400
              ? `${companyInfo.description.substring(0, 1000)}`
              : companyInfo.description}
          </p>
        )}

        {/* P/E Ratio (large card, prefer trailingPE) */}
        {companyInfo.trailingPE && (
          <div className="mt-3 p-3 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: '#8b5cf6',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>P/E Ratio</div>
            <div className="text-sm font-semibold font-mono" style={{ color: '#8b5cf6' }}>
              {formatNumber(companyInfo.trailingPE)}
            </div>
          </div>
        )}
      </div>

      {/* Fundamentals Section */}
      {fundamentalsData && (
        <div className="mt-6 pt-6 border-t" style={{ borderColor: 'var(--bg-1)' }}>
          {showFundamentals && (
            <>
              {/* Key Metrics Grid (compact) */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mb-4 text-xs" style={{ color: 'var(--text-4)' }}>

              </div>

              {/* Detailed Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                {/* Valuation */}
                <div>
                  <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Valuation</div>
                  <div className="space-y-1 text-xs">
                    {/* P/E displayed as a large card above; omit here to avoid duplication */}
                    {fundamentalsData.fundamentals.pegRatio && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>PEG Ratio:</span>
                        <span style={{ color: 'var(--text-2)' }}>{fundamentalsData.fundamentals.pegRatio.toFixed(2)}</span>
                      </div>
                    )}
                    {fundamentalsData.fundamentals.eps != null && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>EPS:</span>
                        <span style={{ color: 'var(--text-2)' }}>${formatNumber(fundamentalsData.fundamentals.eps)}</span>
                      </div>
                    )}
                    {fundamentalsData.fundamentals.priceToBook && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Price/Book:</span>
                        <span style={{ color: 'var(--text-2)' }}>{fundamentalsData.fundamentals.priceToBook.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Profitability */}
                <div>
                  <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Profitability</div>
                  <div className="space-y-1 text-xs">
                    {fundamentalsData.fundamentals.profitMargin && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Profit Margin:</span>
                        <span style={{ color: 'var(--text-2)' }}>{(fundamentalsData.fundamentals.profitMargin * 100).toFixed(2)}%</span>
                      </div>
                    )}
                    {fundamentalsData.fundamentals.operatingMargin && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Operating Margin:</span>
                        <span style={{ color: 'var(--text-2)' }}>{(fundamentalsData.fundamentals.operatingMargin * 100).toFixed(2)}%</span>
                      </div>
                    )}
                    {fundamentalsData.fundamentals.roe && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>ROE:</span>
                        <span style={{ color: 'var(--text-2)' }}>{(fundamentalsData.fundamentals.roe * 100).toFixed(2)}%</span>
                      </div>
                    )}
                    {fundamentalsData.fundamentals.roa && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>ROA:</span>
                        <span style={{ color: 'var(--text-2)' }}>{(fundamentalsData.fundamentals.roa * 100).toFixed(2)}%</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Financial Health & Growth */}
                <div>
                  <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Health & Growth</div>
                  <div className="space-y-1 text-xs">
                    {fundamentalsData.fundamentals.debtToEquity && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Debt/Equity:</span>
                        <span style={{ color: 'var(--text-2)' }}>{fundamentalsData.fundamentals.debtToEquity.toFixed(2)}</span>
                      </div>
                    )}
                    {fundamentalsData.fundamentals.currentRatio && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Current Ratio:</span>
                        <span style={{ color: 'var(--text-2)' }}>{fundamentalsData.fundamentals.currentRatio.toFixed(2)}</span>
                      </div>
                    )}
                    {fundamentalsData.fundamentals.revenueGrowth && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Revenue Growth:</span>
                        <span style={{ color: fundamentalsData.fundamentals.revenueGrowth > 0 ? 'var(--success)' : 'var(--danger)' }}>
                          {(fundamentalsData.fundamentals.revenueGrowth * 100).toFixed(2)}%
                        </span>
                      </div>
                    )}
                    {fundamentalsData.fundamentals.earningsGrowth && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Earnings Growth:</span>
                        <span style={{ color: fundamentalsData.fundamentals.earningsGrowth > 0 ? 'var(--success)' : 'var(--danger)' }}>
                          {(fundamentalsData.fundamentals.earningsGrowth * 100).toFixed(2)}%
                        </span>
                      </div>
                    )}
                  </div>
                </div>
                {/* Market */}
                <div>
                  <div className="font-semibold mb-2" style={{ color: 'var(--text-3)' }}>Market</div>
                  <div className="space-y-1 text-xs">
                    {companyInfo.marketCap != null && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Market Cap:</span>
                        <span style={{ color: 'var(--text-2)' }}>{formatMarketCap(companyInfo.marketCap)}</span>
                      </div>
                    )}

                    {companyInfo.sharesOutstanding != null && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Shares Out.:</span>
                        <span style={{ color: 'var(--text-2)' }}>{formatVolume(companyInfo.sharesOutstanding)}</span>
                      </div>
                    )}

                    {companyInfo.averageVolume != null && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Avg Volume:</span>
                        <span style={{ color: 'var(--text-2)' }}>{formatVolume(companyInfo.averageVolume)}</span>
                      </div>
                    )}

                    {companyInfo.fiftyTwoWeekHigh != null && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>52-Week High:</span>
                        <span style={{ color: 'var(--text-2)' }}>${formatNumber(companyInfo.fiftyTwoWeekHigh)}</span>
                      </div>
                    )}

                    {companyInfo.fiftyTwoWeekLow != null && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>52-Week Low:</span>
                        <span style={{ color: 'var(--text-2)' }}>${formatNumber(companyInfo.fiftyTwoWeekLow)}</span>
                      </div>
                    )}

                    {companyInfo.beta != null && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Beta:</span>
                        <span style={{ color: 'var(--text-2)' }}>{formatNumber(companyInfo.beta)}</span>
                      </div>
                    )}

                    {companyInfo.dividendYield != null && (
                      <div className="flex justify-between">
                        <span style={{ color: 'var(--text-4)' }}>Div Yield:</span>
                        <span style={{ color: 'var(--text-2)' }}>{formatPercent(companyInfo.dividendYield)}</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {fundamentalsLoading && (
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
