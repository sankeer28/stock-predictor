'use client';

import React from 'react';
import { Building2, TrendingUp, DollarSign, BarChart3, Percent, Award, ExternalLink, ArrowUp, ArrowDown } from 'lucide-react';

interface CompanyInfoProps {
  symbol: string;
  companyName: string;
  currentPrice: number;
  currentChange?: number | null;
  currentChangePercent?: number | null;
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

export default function CompanyInfo({ symbol, companyName, currentPrice, currentChange, currentChangePercent, companyInfo }: CompanyInfoProps) {
  const [imageError, setImageError] = React.useState(false);

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
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              {/* Today's change with arrow to the left of the price */}
              {typeof change === 'number' && typeof changePercent === 'number' ? (
                <div className="flex items-center text-sm font-mono" style={{ color: change > 0 ? 'var(--success)' : 'var(--danger)' }}>
                  {change > 0 ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />}
                  <span className="ml-1">
                    {change > 0 ? '+' : ''}{change.toFixed(2)} ({changePercent.toFixed(2)}%)
                  </span>
                </div>
              ) : null}

              <div className="text-right">
                <div className="text-4xl font-bold font-mono" style={{ color: 'var(--accent)' }}>
                  ${currentPrice.toFixed(2)}
                </div>
              </div>
            </div>
            {companyInfo.website && (
              <a
                href={companyInfo.website}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-3 py-2 border transition-colors"
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
              ? `${companyInfo.description.substring(0, 400)}...`
              : companyInfo.description}
          </p>
        )}
      </div>

      {/* Market Data - Only show available fields */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {/* Market Cap */}
        {companyInfo.marketCap && (
          <div className="p-3 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: 'var(--accent)',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Market Cap</div>
            <div className="text-sm font-semibold font-mono" style={{ color: 'var(--accent)' }}>
              {formatMarketCap(companyInfo.marketCap)}
            </div>
          </div>
        )}

        {/* Shares Outstanding */}
        {companyInfo.sharesOutstanding && (
          <div className="p-3 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: 'var(--success)',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Shares Out.</div>
            <div className="text-sm font-semibold font-mono" style={{ color: 'var(--success)' }}>
              {formatVolume(companyInfo.sharesOutstanding)}
            </div>
          </div>
        )}

        {/* 52 Week High */}
        {companyInfo.fiftyTwoWeekHigh && (
          <div className="p-3 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: 'var(--info)',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>52-Week High</div>
            <div className="text-sm font-semibold font-mono" style={{ color: 'var(--info)' }}>
              ${formatNumber(companyInfo.fiftyTwoWeekHigh)}
            </div>
          </div>
        )}

        {/* 52 Week Low */}
        {companyInfo.fiftyTwoWeekLow && (
          <div className="p-3 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: 'var(--info)',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>52-Week Low</div>
            <div className="text-sm font-semibold font-mono" style={{ color: 'var(--info)' }}>
              ${formatNumber(companyInfo.fiftyTwoWeekLow)}
            </div>
          </div>
        )}

        {/* Average Volume */}
        {companyInfo.averageVolume && (
          <div className="p-3 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: 'var(--warning)',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Avg Volume</div>
            <div className="text-sm font-semibold font-mono" style={{ color: 'var(--warning)' }}>
              {formatVolume(companyInfo.averageVolume)}
            </div>
          </div>
        )}

        {/* P/E Ratio */}
        {companyInfo.trailingPE && (
          <div className="p-3 border-2" style={{
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

        {/* Beta */}
        {companyInfo.beta && (
          <div className="p-3 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: '#ec4899',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Beta</div>
            <div className="text-sm font-semibold font-mono" style={{ color: '#ec4899' }}>
              {formatNumber(companyInfo.beta)}
            </div>
          </div>
        )}

        {/* Dividend Yield */}
        {companyInfo.dividendYield && (
          <div className="p-3 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: '#10b981',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Div Yield</div>
            <div className="text-sm font-semibold font-mono" style={{ color: '#10b981' }}>
              {formatPercent(companyInfo.dividendYield)}
            </div>
          </div>
        )}
      </div>

      {/* Footer with CIK and identifiers */}
      {(companyInfo.cik || companyInfo.sicCode) && (
        <div className="mt-4 pt-4 border-t" style={{ borderColor: 'var(--bg-1)' }}>
          <div className="flex flex-wrap gap-4 text-xs" style={{ color: 'var(--text-5)' }}>
            {companyInfo.cik && (
              <div>
                <span>CIK: </span>
                <span className="font-mono">{companyInfo.cik}</span>
              </div>
            )}
            {companyInfo.sicCode && companyInfo.sicDescription && (
              <div>
                <span>SIC: </span>
                <span className="font-mono">{companyInfo.sicCode}</span>
                <span className="ml-1">({companyInfo.sicDescription})</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
