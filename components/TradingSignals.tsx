'use client';

import React from 'react';
import { TradingSignal } from '@/types';
import {
  TrendingUp,
  TrendingDown,
  MinusCircle,
  AlertCircle,
  Target,
  Activity,
} from 'lucide-react';

interface TradingSignalsProps {
  signal: TradingSignal;
  currentPrice: number;
}

export default function TradingSignals({ signal, currentPrice }: TradingSignalsProps) {
  const getSignalConfig = () => {
    switch (signal.type) {
      case 'strong_buy':
        return {
          icon: <TrendingUp className="w-8 h-8" />,
          label: 'Strong Buy',
          color: 'bg-green-600 text-white',
          borderColor: 'border-green-600',
          textColor: 'text-green-600',
        };
      case 'buy':
        return {
          icon: <TrendingUp className="w-8 h-8" />,
          label: 'Buy',
          color: 'bg-green-500 text-white',
          borderColor: 'border-green-500',
          textColor: 'text-green-600',
        };
      case 'weak_buy':
        return {
          icon: <TrendingUp className="w-8 h-8" />,
          label: 'Weak Buy',
          color: 'bg-green-400 text-white',
          borderColor: 'border-green-400',
          textColor: 'text-green-600',
        };
      case 'hold':
        return {
          icon: <MinusCircle className="w-8 h-8" />,
          label: 'Hold',
          color: 'bg-gray-500 text-white',
          borderColor: 'border-gray-500',
          textColor: 'text-gray-600',
        };
      case 'weak_sell':
        return {
          icon: <TrendingDown className="w-8 h-8" />,
          label: 'Weak Sell',
          color: 'bg-red-400 text-white',
          borderColor: 'border-red-400',
          textColor: 'text-red-600',
        };
      case 'sell':
        return {
          icon: <TrendingDown className="w-8 h-8" />,
          label: 'Sell',
          color: 'bg-red-500 text-white',
          borderColor: 'border-red-500',
          textColor: 'text-red-600',
        };
      case 'strong_sell':
        return {
          icon: <TrendingDown className="w-8 h-8" />,
          label: 'Strong Sell',
          color: 'bg-red-600 text-white',
          borderColor: 'border-red-600',
          textColor: 'text-red-600',
        };
    }
  };

  const config = getSignalConfig();

  return (
    <div className="card">
      <span className="card-label">Trading Signal Analysis</span>

      {/* Main Signal */}
      <div
        className={`${config.color} p-6 mb-6 flex items-center justify-between border-2`}
        style={{ borderColor: 'var(--accent)' }}
      >
        <div className="flex items-center gap-4">
          {config.icon}
          <div>
            <div className="text-2xl font-bold">{config.label}</div>
            <div className="text-sm opacity-90">Confidence: {signal.confidence.toFixed(1)}%</div>
          </div>
        </div>
        {signal.priceTarget && (
          <div className="text-right">
            <div className="text-sm opacity-90">Target Price</div>
            <div className="text-2xl font-bold">${signal.priceTarget.toFixed(2)}</div>
            {signal.priceChange && (
              <div className="text-sm opacity-90">
                {signal.priceChange > 0 ? '+' : ''}
                {signal.priceChange.toFixed(2)}%
              </div>
            )}
          </div>
        )}
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 border-2" style={{
          background: 'var(--bg-2)',
          borderColor: 'var(--info)',
          borderLeftWidth: '3px'
        }}>
          <div className="flex items-center gap-2 mb-2">
            <Activity className="w-5 h-5" style={{ color: 'var(--info)' }} />
            <span className="text-sm font-medium" style={{ color: 'var(--text-3)' }}>Current Price</span>
          </div>
          <div className="text-2xl font-bold" style={{ color: 'var(--info)' }}>
            ${currentPrice.toFixed(2)}
          </div>
        </div>

        {signal.priceTarget && (
          <div className="p-4 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: 'var(--accent)',
            borderLeftWidth: '3px'
          }}>
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-5 h-5" style={{ color: 'var(--accent)' }} />
              <span className="text-sm font-medium" style={{ color: 'var(--text-3)' }}>Price Target</span>
            </div>
            <div className="text-2xl font-bold" style={{ color: 'var(--accent)' }}>
              ${signal.priceTarget.toFixed(2)}
            </div>
            <div className="text-sm" style={{ color: 'var(--accent)' }}>
              {signal.priceChange && signal.priceChange > 0 ? '+' : ''}
              {signal.priceChange?.toFixed(2)}%
            </div>
          </div>
        )}
      </div>

      {/* Signal Reasons */}
      <div className="mb-4">
        <div className="flex items-center gap-2 mb-3">
          <AlertCircle className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <h4 className="font-semibold" style={{ color: 'var(--text-2)' }}>Key Indicators</h4>
        </div>
        <div className="space-y-2">
          {signal.reasons.map((reason, index) => {
            const isBullish = reason.toLowerCase().includes('bullish') ||
              reason.toLowerCase().includes('above') ||
              reason.toLowerCase().includes('buy') ||
              reason.toLowerCase().includes('oversold') ||
              reason.toLowerCase().includes('golden') ||
              reason.toLowerCase().includes('strong') && reason.toLowerCase().includes('momentum');

            const isBearish = reason.toLowerCase().includes('bearish') ||
              reason.toLowerCase().includes('below') ||
              reason.toLowerCase().includes('sell') ||
              reason.toLowerCase().includes('overbought') ||
              reason.toLowerCase().includes('death') ||
              reason.toLowerCase().includes('weak') && reason.toLowerCase().includes('momentum');

            return (
              <div
                key={index}
                className="p-3 border-2"
                style={{
                  background: 'var(--bg-2)',
                  borderColor: isBullish ? 'var(--success)' : isBearish ? 'var(--danger)' : 'var(--bg-1)',
                  borderLeftWidth: '3px',
                  color: isBullish ? 'var(--success)' : isBearish ? 'var(--danger)' : 'var(--text-3)'
                }}
              >
                <div className="flex items-start gap-2">
                  <span className="text-lg mt-[-2px]">
                    {isBullish ? '📈' : isBearish ? '📉' : '➡️'}
                  </span>
                  <span className="text-sm flex-1">{reason}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mt-6 p-4 border-2" style={{
        background: 'var(--bg-2)',
        borderColor: 'var(--warning)',
        borderLeftWidth: '3px'
      }}>
        <p className="text-xs" style={{ color: 'var(--warning)' }}>
          <strong>Disclaimer:</strong> This signal is based on technical analysis and should not be
          considered financial advice. Always conduct your own research and consult with a financial
          advisor before making investment decisions.
        </p>
      </div>
    </div>
  );
}
