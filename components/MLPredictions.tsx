'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { Brain, TrendingUp, Loader2, X, Menu } from 'lucide-react';
import { MLPrediction } from '@/lib/mlAlgorithms';
import { MLForecast } from '@/lib/mlForecasting';

interface MLPredictionsProps {
  currentPrice: number;
  predictions: {
    lstm?: MLForecast[];
    arima?: MLPrediction[];
    prophetLite?: MLPrediction[];
    gru?: MLPrediction[];
    ensemble?: MLPrediction[];
    cnnLstm?: MLPrediction[];
    linearRegression?: MLPrediction[];
    ema?: MLPrediction[];
  };
  isTraining: boolean;
  fromCache?: boolean;
  onRecalculate?: () => void;
  // Render inline on mobile (stacked below news) when true
  inlineMobile?: boolean;
}

const MLPredictions = React.memo(function MLPredictions({ currentPrice, predictions, isTraining, fromCache, onRecalculate, inlineMobile }: MLPredictionsProps) {
  const [isOpen, setIsOpen] = useState<boolean>(!!inlineMobile);
  const [selectedDays, setSelectedDays] = useState<number>(7);

  // Memoize the calculateStats function
  const calculateStats = useCallback((preds: MLPrediction[] | MLForecast[] | undefined) => {
    if (!preds || preds.length === 0) return null;

    const targetIndex = Math.min(selectedDays - 1, preds.length - 1);
    const targetPrice = preds[targetIndex].predicted;
    const change = ((targetPrice - currentPrice) / currentPrice) * 100;

    return {
      price: targetPrice,
      change,
      direction: change > 0 ? 'up' : 'down',
    };
  }, [selectedDays, currentPrice]);

  // Memoize the algorithms array to avoid recreation on every render
  const algorithms = useMemo(() => [
    { name: '🏆 Ensemble', key: 'ensemble', data: predictions.ensemble, color: '#f59e0b', description: 'Best: Combines all models' },
    { name: 'LSTM', key: 'lstm', data: predictions.lstm, color: 'var(--accent)', description: 'Deep learning neural network' },
    { name: 'Prophet-Lite', key: 'prophetLite', data: predictions.prophetLite, color: '#10b981', description: 'Trend + seasonality analysis' },
    { name: 'GRU', key: 'gru', data: predictions.gru, color: '#3b82f6', description: 'Simplified LSTM, faster' },
    { name: 'CNN-LSTM', key: 'cnnLstm', data: predictions.cnnLstm, color: '#ec4899', description: 'Pattern recognition + time series' },
    { name: 'ARIMA', key: 'arima', data: predictions.arima, color: 'var(--info)', description: 'Statistical time series' },
    { name: 'Exponential MA', key: 'ema', data: predictions.ema, color: '#9333ea', description: 'Weighted moving average' },
    { name: 'Linear Regression', key: 'linearRegression', data: predictions.linearRegression, color: 'var(--success)', description: 'Simple trend analysis' },
  ], [predictions]);

  return (
    <>
      {/* Mobile Toggle Button (hide when rendering inline) */}
      {!inlineMobile && (
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-3 border-2 xl:hidden"
          aria-label="Toggle ML Predictions"
          style={{
            position: 'fixed',
            right: '1rem',
            top: '1rem',
            zIndex: 50,
            background: 'var(--bg-4)',
            borderColor: 'var(--accent)',
            color: 'var(--accent)'
          }}
        >
          <Brain className="w-5 h-5" />
        </button>
      )}

      {/* ML Predictions Sidebar */}
      <div
        className={`card ${inlineMobile ? 'w-full' : 'w-80'} transition-transform xl:translate-x-0 ${
          isOpen ? 'translate-x-0' : (inlineMobile ? '' : 'translate-x-[calc(100%+1rem)]')
        } xl:block`}
        style={{
          position: inlineMobile ? 'relative' : (isOpen ? 'fixed' : 'relative'),
          right: inlineMobile ? 'auto' : (isOpen ? 0 : 'auto'),
          top: inlineMobile ? 'auto' : (isOpen ? 0 : 'auto'),
          zIndex: inlineMobile ? 'auto' : (isOpen ? 40 : 'auto'),
          maxHeight: inlineMobile ? 'none' : (isOpen ? '100vh' : 'none'),
          overflowY: inlineMobile ? 'visible' : (isOpen ? 'auto' : 'visible'),
        }}
      >
        <span className="card-label">ML Predictions</span>

        {/* Cache Status and Retrain Button */}
        <div className="mb-3 flex items-center justify-between">
          {fromCache && (
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 rounded-full" style={{ background: 'var(--accent)' }} />
              <span className="text-xs" style={{ color: 'var(--text-4)' }}>From cache</span>
            </div>
          )}
          {onRecalculate && !isTraining && (
            <button
              onClick={onRecalculate}
              className="px-2 py-1 text-xs border transition-all ml-auto"
              style={{
                background: 'var(--bg-3)',
                borderColor: 'var(--bg-1)',
                color: 'var(--text-3)',
              }}
            >
              Retrain
            </button>
          )}
        </div>

        {isTraining && (
          <div className="flex items-center justify-end mb-4">
            <div className="flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin" style={{ color: 'var(--accent)' }} />
              <span className="text-xs" style={{ color: 'var(--accent)' }}>Training...</span>
            </div>
          </div>
        )}

        {/* Time Horizon Selector */}
        <div className="mb-4">
          <label className="text-xs mb-2 block" style={{ color: 'var(--text-4)' }}>
            Forecast Horizon
          </label>
          <div className="flex gap-2">
            {[7, 14, 30].map((days) => (
              <button
                key={days}
                onClick={() => setSelectedDays(days)}
                className="flex-1 px-3 py-2 text-xs font-semibold border-2 transition-all"
                style={{
                  background: selectedDays === days ? 'var(--bg-2)' : 'var(--bg-3)',
                  borderColor: selectedDays === days ? 'var(--accent)' : 'var(--bg-1)',
                  color: selectedDays === days ? 'var(--accent)' : 'var(--text-3)',
                }}
              >
                {days} Days
              </button>
            ))}
          </div>
        </div>

        {/* Algorithms List */}
        <div className="space-y-2">
          {algorithms.map((algo) => {
            const stats = calculateStats(algo.data);
            const isReady = algo.data && algo.data.length > 0;

            return (
              <div
                key={algo.key}
                className="p-2 border-2"
                style={{
                  background: 'var(--bg-2)',
                  borderColor: algo.color,
                  borderLeftWidth: '3px',
                  opacity: isReady ? 1 : 0.5,
                }}
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-1">
                    <span className="text-xs font-semibold" style={{ color: 'var(--text-2)' }}>
                      {algo.name}
                    </span>
                    {algo.key === 'ensemble' && isReady && (
                      <span className="text-xs px-1" style={{ 
                        background: 'linear-gradient(135deg, #f59e0b 0%, #ec4899 100%)',
                        color: 'white',
                        borderRadius: '2px',
                        fontWeight: 'bold'
                      }}>
                        BEST
                      </span>
                    )}
                  </div>
                  {!isReady && isTraining && (
                    <Loader2 className="w-3 h-3 animate-spin" style={{ color: algo.color }} />
                  )}
                </div>

                {isReady && stats ? (
                  <>
                    <div className="flex items-baseline gap-2 mb-1">
                      <span className="text-sm font-bold font-mono" style={{ color: algo.color }}>
                        ${stats.price.toFixed(2)}
                      </span>
                      <span
                        className="text-xs font-semibold"
                        style={{
                          color: stats.direction === 'up' ? 'var(--success)' : 'var(--danger)',
                        }}
                      >
                        {stats.direction === 'up' ? '↑' : '↓'} {Math.abs(stats.change).toFixed(2)}%
                      </span>
                    </div>
                    <div className="text-xs" style={{ color: 'var(--text-4)' }}>
                      {algo.description}
                    </div>
                  </>
                ) : (
                  <div className="text-xs" style={{ color: 'var(--text-4)' }}>
                    {isTraining ? (algo.key === 'lstm' ? 'Training...' : 'Computing...') : 'Waiting...'}
                  </div>
                )}
              </div>
            );
          })}
        </div>



        {/* Algorithm Info */}
        <div className="mt-6 space-y-3">
          <div className="p-3 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: 'var(--info)',
            borderLeftWidth: '3px'
          }}>
            <div className="text-xs" style={{ color: 'var(--text-4)' }}>
              <strong style={{ color: 'var(--info)' }}>Note:</strong> ML predictions are based on historical patterns.
              Actual prices may vary due to market events, news, and other factors not captured by algorithms.
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Overlay (skip when inline) */}
      {!inlineMobile && isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 xl:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
});

export default MLPredictions;
