'use client';

import { ChartPattern } from '@/types';
import { analyzePatterns, PatternAnalysisResult } from '@/lib/patternAnalysis';
import { useState, useEffect, useMemo } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';

interface PatternAnalysisProps {
  patterns: ChartPattern[];
  startDate?: string;
  endDate?: string;
  inlineMobile?: boolean;
  onRefreshPatterns?: () => void;
  isDetecting?: boolean;
}

export default function PatternAnalysis({
  patterns,
  startDate,
  endDate,
  inlineMobile,
  onRefreshPatterns,
  isDetecting = false
}: PatternAnalysisProps) {
  const [expanded, setExpanded] = useState(true);
  const [isCalculating, setIsCalculating] = useState(false);
  const [debouncedDateRange, setDebouncedDateRange] = useState({ startDate, endDate });

  // Debounce date range changes to avoid recalculating on every zoom/pan
  useEffect(() => {
    // If dates are initially undefined, don't debounce
    if (!startDate || !endDate) {
      setDebouncedDateRange({ startDate, endDate });
      setIsCalculating(false);
      return;
    }

    setIsCalculating(true);
    const timer = setTimeout(() => {
      setDebouncedDateRange({ startDate, endDate });
      setIsCalculating(false);
    }, 300); // Wait 300ms after last change

    return () => clearTimeout(timer);
  }, [startDate, endDate]);

  // Analyze patterns - use actual dates if debounced dates are undefined
  const analysis = useMemo(() => {
    const effectiveStartDate = debouncedDateRange.startDate || startDate;
    const effectiveEndDate = debouncedDateRange.endDate || endDate;
    return analyzePatterns(patterns, effectiveStartDate, effectiveEndDate);
  }, [patterns, debouncedDateRange.startDate, debouncedDateRange.endDate, startDate, endDate]);

  // Get signal color
  const getSignalColor = (signal: PatternAnalysisResult['signal']) => {
    switch (signal) {
      case 'strong_bullish':
        return '#10b981'; // green
      case 'bullish':
        return '#34d399'; // light green
      case 'neutral':
        return '#94a3b8'; // gray
      case 'bearish':
        return '#f87171'; // light red
      case 'strong_bearish':
        return '#ef4444'; // red
    }
  };

  // Get signal icon
  const getSignalIcon = (signal: PatternAnalysisResult['signal']) => {
    switch (signal) {
      case 'strong_bullish':
        return '🚀';
      case 'bullish':
        return '📈';
      case 'neutral':
        return '⚖️';
      case 'bearish':
        return '📉';
      case 'strong_bearish':
        return '🔻';
    }
  };

  // Get signal label
  const getSignalLabel = (signal: PatternAnalysisResult['signal']) => {
    switch (signal) {
      case 'strong_bullish':
        return 'STRONG BULLISH';
      case 'bullish':
        return 'BULLISH';
      case 'neutral':
        return 'NEUTRAL';
      case 'bearish':
        return 'BEARISH';
      case 'strong_bearish':
        return 'STRONG BEARISH';
    }
  };

  const signalColor = getSignalColor(analysis.signal);
  const signalIcon = getSignalIcon(analysis.signal);
  const signalLabel = getSignalLabel(analysis.signal);
  const confidencePercent = (analysis.confidence * 100).toFixed(0);
  const scorePercent = analysis.score.toFixed(0);

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div>
            <div className="flex items-center gap-2">
              <span className="card-label">Live Pattern Analysis</span>
              {(isCalculating || isDetecting) && (
                <Loader2
                  className="w-3 h-3 animate-spin"
                  style={{ color: 'var(--accent)' }}
                />
              )}
            </div>
            <p className="text-xs" style={{ color: 'var(--text-4)' }}>
              Analysis of {patterns.length} detected pattern{patterns.length !== 1 ? 's' : ''}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Refresh Button */}
          {onRefreshPatterns && (
            <button
              onClick={onRefreshPatterns}
              disabled={isDetecting}
              className="p-1.5 border transition-all disabled:opacity-50 disabled:cursor-not-allowed hover:bg-opacity-10"
              style={{
                background: 'var(--bg-3)',
                borderColor: 'var(--accent)',
                color: 'var(--accent)',
              }}
              title="Re-detect patterns"
            >
              <RefreshCw
                className={`w-4 h-4 ${isDetecting ? 'animate-spin' : ''}`}
              />
            </button>
          )}
          {/* Collapse Button */}
          <button
            onClick={() => setExpanded(!expanded)}
            className="transition-colors"
            style={{ color: 'var(--text-3)' }}
          >
            {expanded ? (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {expanded && (
        <div style={{ opacity: (isCalculating || isDetecting) ? 0.6 : 1, transition: 'opacity 0.2s' }}>
          {/* Refresh Info Banner */}
          {onRefreshPatterns && (
            <div
              className="border-2 p-2 mb-4 flex items-center gap-2"
              style={{
                background: 'var(--bg-3)',
                borderColor: 'var(--accent)',
                borderLeftWidth: '3px',
              }}
            >
              <RefreshCw className="w-4 h-4" style={{ color: 'var(--accent)' }} />
              <div className="flex-1">
                <div className="text-xs font-semibold mb-1" style={{ color: 'var(--text-2)' }}>
                  {isDetecting ? (
                    <>🔄 Re-detecting patterns on chart...</>
                  ) : (
                    <>Click ↻ to re-run pattern detection</>
                  )}
                </div>
                <div className="text-[10px]" style={{ color: 'var(--text-4)' }}>
                  {isDetecting ? (
                    <>Scanning chart with new settings...</>
                  ) : (
                    <>Re-scans entire chart & draws new patterns</>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Signal Badge */}
          <div 
            className="border-2 p-4 mb-4"
            style={{
              background: 'var(--bg-2)',
              borderColor: signalColor,
              borderLeftWidth: '3px',
            }}
          >
            <div className="flex items-center justify-between">
              <div>
                <div 
                  className="text-2xl font-bold mb-1"
                  style={{ color: signalColor }}
                >
                  {signalLabel}
                </div>
                <div className="text-sm" style={{ color: 'var(--text-2)' }}>{analysis.summary}</div>
              </div>
              <div className="text-right">
                <div 
                  className="text-3xl font-bold"
                  style={{ color: signalColor }}
                >
                  {analysis.score > 0 ? '+' : ''}{scorePercent}
                </div>
                <div className="text-xs" style={{ color: 'var(--text-4)' }}>Signal Score</div>
              </div>
            </div>

            {/* Confidence Bar */}
            <div className="mt-3">
              <div className="flex items-center justify-between text-xs mb-1" style={{ color: 'var(--text-4)' }}>
                <span>Confidence Level</span>
                <span className="font-semibold">{confidencePercent}%</span>
              </div>
              <div 
                className="w-full rounded-full h-2"
                style={{ background: 'var(--bg-1)' }}
              >
                <div
                  className="h-2 rounded-full transition-all duration-500"
                  style={{ 
                    width: `${confidencePercent}%`,
                    background: analysis.confidence >= 0.7
                      ? 'var(--success)'
                      : analysis.confidence >= 0.5
                      ? 'var(--warning)'
                      : 'var(--danger)'
                  }}
                />
              </div>
            </div>
          </div>

          {/* Pattern Breakdown */}
          <div className="grid grid-cols-3 gap-3 mb-4">
            <div 
              className="border-2 p-3 text-center"
              style={{
                background: 'var(--bg-2)',
                borderColor: 'var(--success)',
              }}
            >
              <div 
                className="text-2xl font-bold"
                style={{ color: 'var(--success)' }}
              >
                {analysis.patternBreakdown.bullish}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-4)' }}>Bullish</div>
            </div>
            <div 
              className="border-2 p-3 text-center"
              style={{
                background: 'var(--bg-2)',
                borderColor: 'var(--info)',
              }}
            >
              <div 
                className="text-2xl font-bold"
                style={{ color: 'var(--info)' }}
              >
                {analysis.patternBreakdown.neutral}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-4)' }}>Neutral</div>
            </div>
            <div 
              className="border-2 p-3 text-center"
              style={{
                background: 'var(--bg-2)',
                borderColor: 'var(--danger)',
              }}
            >
              <div 
                className="text-2xl font-bold"
                style={{ color: 'var(--danger)' }}
              >
                {analysis.patternBreakdown.bearish}
              </div>
              <div className="text-xs" style={{ color: 'var(--text-4)' }}>Bearish</div>
            </div>
          </div>

          {/* Key Reasoning */}
          <div className="mb-4">
            <h3 className="text-sm font-semibold mb-2 flex items-center gap-2" style={{ color: 'var(--text-1)' }}>
              <span>🎯</span> Key Insights
            </h3>
            <div className="space-y-2">
              {analysis.reasoning.map((reason, index) => (
                <div
                  key={index}
                  className="border p-2 text-sm"
                  style={{
                    background: 'var(--bg-3)',
                    borderColor: 'var(--bg-1)',
                    color: 'var(--text-2)',
                  }}
                >
                  • {reason}
                </div>
              ))}
            </div>
          </div>

          {/* Key Patterns */}
          {analysis.keyPatterns.length > 0 && (
            <div className="mb-4">
              <h3 className="text-sm font-semibold mb-2 flex items-center gap-2" style={{ color: 'var(--text-1)' }}>
                <span>📊</span> Key Patterns
              </h3>
              <div className="space-y-2">
                {analysis.keyPatterns.map((kp, index) => {
                  const impactColor =
                    kp.impact === 'high'
                      ? 'var(--danger)'
                      : kp.impact === 'medium'
                      ? 'var(--warning)'
                      : 'var(--info)';
                  const directionColor =
                    kp.pattern.direction === 'bullish'
                      ? 'var(--success)'
                      : kp.pattern.direction === 'bearish'
                      ? 'var(--danger)'
                      : 'var(--info)';

                  return (
                    <div
                      key={index}
                      className="border p-2 flex items-center justify-between"
                      style={{
                        background: 'var(--bg-3)',
                        borderColor: 'var(--bg-1)',
                      }}
                    >
                      <div className="flex-1">
                        <div className="text-sm font-medium" style={{ color: 'var(--text-1)' }}>
                          {kp.pattern.label}
                        </div>
                        <div className="text-xs" style={{ color: 'var(--text-4)' }}>
                          {new Date(kp.pattern.startDate).toLocaleDateString()} -{' '}
                          {new Date(kp.pattern.endDate).toLocaleDateString()}
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="text-right">
                          <div 
                            className="text-xs font-semibold"
                            style={{ color: directionColor }}
                          >
                            {kp.pattern.direction.toUpperCase()}
                          </div>
                          <div className="text-xs" style={{ color: 'var(--text-4)' }}>
                            {(kp.pattern.confidence * 100).toFixed(0)}% conf.
                          </div>
                        </div>
                        <div
                          className="px-2 py-1 text-xs font-semibold"
                          style={{
                            background: 'var(--bg-2)',
                            color: impactColor,
                          }}
                        >
                          {kp.impact.toUpperCase()}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Recommendations */}
          <div>
            <h3 className="text-sm font-semibold mb-2 flex items-center gap-2" style={{ color: 'var(--text-1)' }}>
              <span>💡</span> Recommendations
            </h3>
            <div className="space-y-2">
              {analysis.recommendations.map((rec, index) => (
                <div
                  key={index}
                  className="border-2 p-2 text-sm"
                  style={{
                    background: 'var(--bg-2)',
                    borderColor: 'var(--accent)',
                    color: 'var(--text-2)',
                  }}
                >
                  {rec}
                </div>
              ))}
            </div>
          </div>

          {/* Date Range Indicator */}
          {startDate && endDate && (
            <div className="mt-4 pt-3 border-t" style={{ borderColor: 'var(--bg-1)' }}>
              <div className="text-xs text-center" style={{ color: 'var(--text-4)' }}>
                Analysis for period: {new Date(startDate).toLocaleDateString()} -{' '}
                {new Date(endDate).toLocaleDateString()}
              </div>
              <div className="text-xs text-center mt-1" style={{ color: 'var(--text-5)' }}>
                ✨ This analysis updates dynamically as you zoom in/out
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

