'use client';

import React from 'react';
import { TradingSignal } from '@/types';
import {
  TrendingUp,
  TrendingDown,
  MinusCircle,
  AlertCircle,
  Target,
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

  const getIndicatorExplanation = (reason: string): string => {
    const lowerReason = reason.toLowerCase();
    
    if (lowerReason.includes('rsi')) {
      if (lowerReason.includes('oversold')) {
        return 'RSI below 30 suggests the stock may be undervalued and due for a bounce';
      } else if (lowerReason.includes('overbought')) {
        return 'RSI above 70 indicates the stock may be overvalued and due for a correction';
      }
      return 'RSI measures momentum on a 0-100 scale to identify overbought/oversold conditions';
    }
    
    if (lowerReason.includes('macd')) {
      if (lowerReason.includes('bullish') || lowerReason.includes('crossover')) {
        return 'MACD line crossing above signal line suggests increasing upward momentum';
      } else if (lowerReason.includes('bearish')) {
        return 'MACD line crossing below signal line suggests increasing downward momentum';
      }
      return 'MACD tracks the relationship between two moving averages to show trend changes';
    }
    
    if (lowerReason.includes('moving average') || lowerReason.includes('ma ') || lowerReason.includes('sma')) {
      if (lowerReason.includes('golden cross')) {
        return 'Golden Cross: 50-day MA crossing above 200-day MA is a strong bullish signal';
      } else if (lowerReason.includes('death cross')) {
        return 'Death Cross: 50-day MA crossing below 200-day MA is a strong bearish signal';
      } else if (lowerReason.includes('above')) {
        return 'Price above moving average suggests an uptrend with support from the average';
      } else if (lowerReason.includes('below')) {
        return 'Price below moving average suggests a downtrend with resistance from the average';
      }
      return 'Moving averages smooth price data to identify trends and support/resistance levels';
    }
    
    if (lowerReason.includes('bollinger')) {
      if (lowerReason.includes('lower')) {
        return 'Price near lower band suggests the stock is potentially oversold';
      } else if (lowerReason.includes('upper')) {
        return 'Price near upper band suggests the stock is potentially overbought';
      }
      return 'Bollinger Bands measure volatility and help identify overbought/oversold conditions';
    }
    
    if (lowerReason.includes('volume')) {
      if (lowerReason.includes('increasing') || lowerReason.includes('high')) {
        return 'Rising volume confirms the strength and conviction behind the current price move';
      } else if (lowerReason.includes('decreasing') || lowerReason.includes('low')) {
        return 'Declining volume suggests weakening conviction in the current price trend';
      }
      return 'Volume indicates the strength of price movements and trader conviction';
    }
    
    if (lowerReason.includes('momentum')) {
      if (lowerReason.includes('strong')) {
        return 'Strong momentum indicates the current trend is likely to continue';
      } else if (lowerReason.includes('weak')) {
        return 'Weak momentum suggests the current trend may be losing strength';
      }
      return 'Momentum measures the rate of price change to predict trend continuation';
    }
    
    if (lowerReason.includes('support')) {
      return 'Support levels are price points where buying pressure tends to prevent further decline';
    }
    
    if (lowerReason.includes('resistance')) {
      return 'Resistance levels are price points where selling pressure tends to prevent further rise';
    }
    
    if (lowerReason.includes('trend')) {
      if (lowerReason.includes('uptrend') || lowerReason.includes('up trend')) {
        return 'Uptrend: Series of higher highs and higher lows indicates bullish momentum';
      } else if (lowerReason.includes('downtrend') || lowerReason.includes('down trend')) {
        return 'Downtrend: Series of lower highs and lower lows indicates bearish momentum';
      }
      return 'Trend analysis identifies the general direction of price movement over time';
    }
    
    return 'Technical indicator providing insight into potential price movement';
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
        <div className="text-right">
          <div className="text-sm opacity-90">Current Price</div>
          <div className="text-2xl font-bold">${currentPrice.toFixed(2)}</div>
        </div>
      </div>

      {/* Price Target */}
      {signal.priceTarget && (
        <div className="p-4 border-2 mb-6" style={{
          background: 'var(--bg-2)',
          borderColor: 'var(--accent)',
          borderLeftWidth: '3px'
        }}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5" style={{ color: 'var(--accent)' }} />
              <span className="text-sm font-medium" style={{ color: 'var(--text-3)' }}>Price Target</span>
            </div>
            <div className="text-right">
              <div className="text-2xl font-bold" style={{ color: 'var(--accent)' }}>
                ${signal.priceTarget.toFixed(2)}
              </div>
              <div className="text-sm" style={{ color: 'var(--accent)' }}>
                {signal.priceChange && signal.priceChange > 0 ? '+' : ''}
                {signal.priceChange?.toFixed(2)}%
              </div>
            </div>
          </div>
        </div>
      )}

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

            const explanation = getIndicatorExplanation(reason);
            
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
                  <div className="flex-1">
                    <div className="text-sm font-medium mb-1">{reason}</div>
                    <div className="text-xs opacity-70" style={{ color: 'var(--text-4)' }}>
                      {explanation}
                    </div>
                  </div>
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
