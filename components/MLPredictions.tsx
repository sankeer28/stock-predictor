'use client';

import React, { useState } from 'react';
import { Brain, TrendingUp, Loader2, X, Menu } from 'lucide-react';
import { MLPrediction } from '@/lib/mlAlgorithms';
import { MLForecast } from '@/lib/mlForecasting';

interface MLPredictionsProps {
  currentPrice: number;
  predictions: {
    lstm?: MLForecast[];
    linearRegression?: MLPrediction[];
    polynomialRegression?: MLPrediction[];
    movingAverage?: MLPrediction[];
    ema?: MLPrediction[];
  };
  isTraining: boolean;
}

export default function MLPredictions({ currentPrice, predictions, isTraining }: MLPredictionsProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedDays, setSelectedDays] = useState<number>(7);

  // Calculate summary statistics for each algorithm
  const calculateStats = (preds: MLPrediction[] | MLForecast[] | undefined) => {
    if (!preds || preds.length === 0) return null;

    const targetIndex = Math.min(selectedDays - 1, preds.length - 1);
    const targetPrice = preds[targetIndex].predicted;
    const change = ((targetPrice - currentPrice) / currentPrice) * 100;

    return {
      price: targetPrice,
      change,
      direction: change > 0 ? 'up' : 'down',
    };
  };

  const algorithms = [
    { name: 'LSTM Neural Network', key: 'lstm', data: predictions.lstm, color: 'var(--accent)' },
    { name: 'Linear Regression', key: 'linearRegression', data: predictions.linearRegression, color: 'var(--info)' },
    { name: 'Polynomial Regression', key: 'polynomialRegression', data: predictions.polynomialRegression, color: 'var(--success)' },
    { name: 'Moving Average', key: 'movingAverage', data: predictions.movingAverage, color: 'var(--warning)' },
    { name: 'Exponential MA', key: 'ema', data: predictions.ema, color: 'var(--error)' },
  ];

  return (
    <>
      {/* Mobile Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-3 border-2 xl:hidden"
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

      {/* ML Predictions Sidebar */}
      <div
        className={`card w-80 transition-transform xl:translate-x-0 ${
          isOpen ? 'translate-x-0' : 'translate-x-[calc(100%+1rem)]'
        } xl:block`}
        style={{
          position: isOpen ? 'fixed' : 'relative',
          right: isOpen ? 0 : 'auto',
          top: isOpen ? 0 : 'auto',
          zIndex: isOpen ? 40 : 'auto',
          maxHeight: isOpen ? '100vh' : 'none',
          overflowY: isOpen ? 'auto' : 'visible',
        }}
      >
        <span className="card-label">ML Predictions</span>

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

        {/* Current Price Reference */}
        <div className="mb-4 p-3 border-2" style={{
          background: 'var(--bg-2)',
          borderColor: 'var(--bg-1)',
        }}>
          <div className="text-xs mb-1" style={{ color: 'var(--text-4)' }}>Current Price</div>
          <div className="text-xl font-bold font-mono" style={{ color: 'var(--text-1)' }}>
            ${currentPrice.toFixed(2)}
          </div>
        </div>

        {/* Algorithms List */}
        <div className="space-y-3">
          {algorithms.map((algo) => {
            const stats = calculateStats(algo.data);
            const isReady = algo.data && algo.data.length > 0;

            return (
              <div
                key={algo.key}
                className="p-4 border-2"
                style={{
                  background: 'var(--bg-2)',
                  borderColor: algo.color,
                  borderLeftWidth: '4px',
                  opacity: isReady ? 1 : 0.5,
                }}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-semibold" style={{ color: 'var(--text-2)' }}>
                    {algo.name}
                  </span>
                  {!isReady && algo.key === 'lstm' && isTraining && (
                    <Loader2 className="w-3 h-3 animate-spin" style={{ color: algo.color }} />
                  )}
                </div>

                {isReady && stats ? (
                  <>
                    <div className="flex items-baseline gap-2 mb-1">
                      <span className="text-lg font-bold font-mono" style={{ color: algo.color }}>
                        ${stats.price.toFixed(2)}
                      </span>
                      <span
                        className="text-xs font-semibold"
                        style={{
                          color: stats.direction === 'up' ? 'var(--success)' : 'var(--error)',
                        }}
                      >
                        {stats.direction === 'up' ? '↑' : '↓'} {Math.abs(stats.change).toFixed(2)}%
                      </span>
                    </div>
                    <div className="text-xs" style={{ color: 'var(--text-4)' }}>
                      {selectedDays} day forecast
                    </div>
                  </>
                ) : (
                  <div className="text-xs" style={{ color: 'var(--text-4)' }}>
                    {algo.key === 'lstm' ? 'Training...' : 'Calculating...'}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Detailed Table */}
        {algorithms.some(a => a.data && a.data.length > 0) && (
          <div className="mt-6">
            <h3 className="text-sm font-semibold mb-3" style={{ color: 'var(--text-2)' }}>
              Detailed Predictions ({selectedDays} days)
            </h3>
            <div className="border-2" style={{ borderColor: 'var(--bg-1)' }}>
              <table className="w-full text-xs">
                <thead style={{ background: 'var(--bg-2)' }}>
                  <tr>
                    <th className="p-2 text-left" style={{ color: 'var(--text-4)' }}>Day</th>
                    <th className="p-2 text-right" style={{ color: 'var(--text-4)' }}>Avg Price</th>
                    <th className="p-2 text-right" style={{ color: 'var(--text-4)' }}>Change</th>
                  </tr>
                </thead>
                <tbody>
                  {Array.from({ length: Math.min(selectedDays, 10) }).map((_, dayIndex) => {
                    // Calculate average prediction across all algorithms
                    const validPredictions = algorithms
                      .map(algo => algo.data?.[dayIndex]?.predicted)
                      .filter((p): p is number => p !== undefined);

                    if (validPredictions.length === 0) return null;

                    const avgPrice = validPredictions.reduce((sum, p) => sum + p, 0) / validPredictions.length;
                    const change = ((avgPrice - currentPrice) / currentPrice) * 100;

                    return (
                      <tr
                        key={dayIndex}
                        className="border-t"
                        style={{ borderColor: 'var(--bg-1)', background: dayIndex % 2 === 0 ? 'var(--bg-3)' : 'var(--bg-2)' }}
                      >
                        <td className="p-2" style={{ color: 'var(--text-3)' }}>Day {dayIndex + 1}</td>
                        <td className="p-2 text-right font-mono" style={{ color: 'var(--text-2)' }}>
                          ${avgPrice.toFixed(2)}
                        </td>
                        <td
                          className="p-2 text-right font-semibold"
                          style={{
                            color: change > 0 ? 'var(--success)' : 'var(--error)',
                          }}
                        >
                          {change > 0 ? '+' : ''}{change.toFixed(2)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Algorithm Info */}
        <div className="mt-6 p-3 border-2" style={{
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

      {/* Mobile Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-30 xl:hidden"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}
