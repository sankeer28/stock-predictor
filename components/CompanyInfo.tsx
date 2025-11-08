'use client';

import React from 'react';
import { Building2, TrendingUp, DollarSign, BarChart3, Percent, Award, ExternalLink } from 'lucide-react';

interface CompanyInfoProps {
  symbol: string;
  companyName: string;
  currentPrice: number;
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
  };
}

export default function CompanyInfo({ symbol, companyName, currentPrice, companyInfo }: CompanyInfoProps) {
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
            <Building2 className="w-6 h-6" style={{ color: 'var(--accent)' }} />
            <div>
              <h2 className="text-2xl font-bold" style={{ color: 'var(--text-1)' }}>
                {companyName}
              </h2>
              <p className="text-sm font-mono" style={{ color: 'var(--text-4)' }}>
                {symbol}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Current Price</div>
              <div className="text-3xl font-bold font-mono" style={{ color: 'var(--accent)' }}>
                ${currentPrice.toFixed(2)}
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

        {/* Description */}
        {companyInfo.description && (
          <p className="text-sm mt-3 leading-relaxed" style={{ color: 'var(--text-3)' }}>
            {companyInfo.description.length > 300
              ? `${companyInfo.description.substring(0, 300)}...`
              : companyInfo.description}
          </p>
        )}
      </div>

      {/* Market Data - Only show available fields */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        {/* 52 Week Range */}
        <div className="p-4 border-2" style={{
          background: 'var(--bg-2)',
          borderColor: 'var(--info)',
          borderLeftWidth: '3px'
        }}>
          <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>52-Week High</div>
          <div className="text-sm font-semibold font-mono" style={{ color: 'var(--info)' }}>
            ${formatNumber(companyInfo.fiftyTwoWeekHigh)}
          </div>
        </div>

        <div className="p-4 border-2" style={{
          background: 'var(--bg-2)',
          borderColor: 'var(--info)',
          borderLeftWidth: '3px'
        }}>
          <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>52-Week Low</div>
          <div className="text-sm font-semibold font-mono" style={{ color: 'var(--info)' }}>
            ${formatNumber(companyInfo.fiftyTwoWeekLow)}
          </div>
        </div>

        {companyInfo.averageVolume && (
          <div className="p-4 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: 'var(--warning)',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Average Volume</div>
            <div className="text-sm font-semibold font-mono" style={{ color: 'var(--warning)' }}>
              {formatVolume(companyInfo.averageVolume)}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
