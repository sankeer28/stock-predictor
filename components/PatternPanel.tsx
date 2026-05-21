'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  BarChart2, RefreshCw, ChevronUp, ChevronDown,
  Target, Lightbulb, SlidersHorizontal, Loader2,
} from 'lucide-react';
import { ChartPattern } from '@/types';
import { analyzePatterns } from '@/lib/patternAnalysis';
import { PatternSettings, PatternPreset, getPatternPresetSettings } from '@/types/patternSettings';

interface PatternPanelProps {
  patterns: ChartPattern[];
  startDate?: string;
  endDate?: string;
  onRefreshPatterns?: () => void;
  isDetecting?: boolean;
  settings: PatternSettings;
  onSettingsChange: (settings: PatternSettings) => void;
  onPresetChange: (preset: PatternPreset) => void;
  currentPreset: PatternPreset;
  inlineMobile?: boolean;
}

const SIGNAL_COLORS: Record<string, string> = {
  strong_bullish: 'var(--green-1)',
  bullish:        'var(--green-2)',
  neutral:        'var(--text-4)',
  bearish:        'var(--red-2)',
  strong_bearish: 'var(--red-1)',
};
const SIGNAL_LABELS: Record<string, string> = {
  strong_bullish: 'STRONG BULL',
  bullish:        'BULLISH',
  neutral:        'NEUTRAL',
  bearish:        'BEARISH',
  strong_bearish: 'STRONG BEAR',
};

export default function PatternPanel({
  patterns,
  startDate,
  endDate,
  onRefreshPatterns,
  isDetecting = false,
  settings,
  onSettingsChange,
  onPresetChange,
  currentPreset,
  inlineMobile,
}: PatternPanelProps) {
  const [expanded,     setExpanded]     = useState(false);
  const [activeTab,    setActiveTab]    = useState<'analysis' | 'settings'>('analysis');
  const [showAdv,      setShowAdv]      = useState(false);
  const [debouncedRange, setDebouncedRange] = useState({ startDate, endDate });
  const [isCalc,       setIsCalc]       = useState(false);
  const [windowsText,  setWindowsText]  = useState(settings.detectionWindows.join(', '));

  // Sync windows text when settings change externally
  useEffect(() => {
    setWindowsText(settings.detectionWindows.join(', '));
  }, [settings.detectionWindows]);

  // Debounce date-range so we don't re-analyse on every zoom tick
  useEffect(() => {
    if (!startDate || !endDate) {
      setDebouncedRange({ startDate, endDate });
      setIsCalc(false);
      return;
    }
    setIsCalc(true);
    const t = setTimeout(() => {
      setDebouncedRange({ startDate, endDate });
      setIsCalc(false);
    }, 300);
    return () => clearTimeout(t);
  }, [startDate, endDate]);

  const analysis = useMemo(() => {
    const s = debouncedRange.startDate || startDate;
    const e = debouncedRange.endDate   || endDate;
    return analyzePatterns(patterns, s, e);
  }, [patterns, debouncedRange.startDate, debouncedRange.endDate, startDate, endDate]);

  const signalColor = SIGNAL_COLORS[analysis.signal] ?? 'var(--text-4)';
  const signalLabel = SIGNAL_LABELS[analysis.signal] ?? analysis.signal;
  const confPct     = (analysis.confidence * 100).toFixed(0);
  const loading     = isDetecting || isCalc;

  // ── Settings handlers ───────────────────────────────────────
  const handlePreset = (preset: PatternPreset) => {
    onPresetChange(preset);
    onSettingsChange(getPatternPresetSettings(preset));
  };

  const handleSetting = (key: keyof PatternSettings, value: any) => {
    onSettingsChange({ ...settings, [key]: value });
    if (currentPreset !== 'custom') onPresetChange('custom');
  };

  const commitWindows = (raw: string) => {
    const windows = raw.split(',').map(w => parseInt(w.trim())).filter(w => !isNaN(w) && w > 0);
    if (windows.length > 0) handleSetting('detectionWindows', windows);
  };

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <span className="card-label">Pattern Analysis</span>

      {/* ── Header ── */}
      <div className="flex items-center gap-2 mb-3">
        <BarChart2 className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--accent)' }} />

        <span className="text-xs font-semibold flex-shrink-0" style={{ color: 'var(--text-3)' }}>
          {patterns.length} pattern{patterns.length !== 1 ? 's' : ''}
        </span>

        <span
          className="px-1.5 py-0.5 text-[10px] font-bold"
          style={{ color: signalColor, background: 'var(--bg-3)', border: `1px solid ${signalColor}` }}
        >
          {signalLabel}
        </span>

        {loading && <Loader2 className="w-3 h-3 animate-spin flex-shrink-0" style={{ color: 'var(--accent)' }} />}

        <div className="flex items-center gap-1 ml-auto">
          {onRefreshPatterns && (
            <button
              onClick={onRefreshPatterns}
              disabled={isDetecting}
              title="Re-detect patterns"
              className="p-1 border transition-all disabled:opacity-50"
              style={{ borderColor: 'var(--bg-1)', color: 'var(--text-4)', background: 'var(--bg-3)' }}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isDetecting ? 'animate-spin' : ''}`} />
            </button>
          )}
          <button
            onClick={() => setExpanded(v => !v)}
            className="p-1 transition-colors"
            style={{ color: 'var(--text-4)' }}
          >
            {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {expanded && (
        <>
          {/* ── Tabs ── */}
          <div className="flex mb-3 border-b" style={{ borderColor: 'var(--bg-1)' }}>
            {(['analysis', 'settings'] as const).map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className="px-3 py-1.5 text-xs font-semibold capitalize border-b-2 -mb-px transition-colors"
                style={{
                  borderColor: activeTab === tab ? 'var(--accent)' : 'transparent',
                  color: activeTab === tab ? 'var(--accent)' : 'var(--text-4)',
                }}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* ══ Analysis Tab ══ */}
          {activeTab === 'analysis' && (
            <div style={{ opacity: loading ? 0.7 : 1, transition: 'opacity 0.2s' }}>

              {/* Signal + score */}
              <div className="flex items-center gap-3 mb-3 px-2 py-2 border-l-2" style={{
                borderColor: signalColor, background: 'var(--bg-3)',
              }}>
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-bold" style={{ color: signalColor }}>{signalLabel}</div>
                  <div className="text-[10px] truncate" style={{ color: 'var(--text-4)' }}>{analysis.summary}</div>
                </div>
                <div className="text-right flex-shrink-0">
                  <div className="text-lg font-bold leading-none" style={{ color: signalColor }}>
                    {analysis.score > 0 ? '+' : ''}{analysis.score.toFixed(0)}
                  </div>
                  <div className="text-[10px]" style={{ color: 'var(--text-5)' }}>score</div>
                </div>
              </div>

              {/* Confidence bar */}
              <div className="mb-3">
                <div className="flex justify-between text-[10px] mb-1" style={{ color: 'var(--text-5)' }}>
                  <span>Confidence</span>
                  <span className="font-semibold" style={{ color: 'var(--text-3)' }}>{confPct}%</span>
                </div>
                <div className="h-1.5 w-full" style={{ background: 'var(--bg-1)' }}>
                  <div
                    className="h-1.5 transition-all duration-500"
                    style={{
                      width: `${confPct}%`,
                      background: analysis.confidence >= 0.7 ? 'var(--success)'
                        : analysis.confidence >= 0.5 ? 'var(--warning)' : 'var(--danger)',
                    }}
                  />
                </div>
              </div>

              {/* Breakdown */}
              <div className="grid grid-cols-3 gap-1.5 mb-3">
                {[
                  { label: 'Bullish', count: analysis.patternBreakdown.bullish, color: 'var(--success)' },
                  { label: 'Neutral', count: analysis.patternBreakdown.neutral, color: 'var(--info)' },
                  { label: 'Bearish', count: analysis.patternBreakdown.bearish, color: 'var(--danger)' },
                ].map(({ label, count, color }) => (
                  <div key={label} className="text-center py-1.5 border" style={{ background: 'var(--bg-3)', borderColor: color }}>
                    <div className="text-base font-bold leading-none" style={{ color }}>{count}</div>
                    <div className="text-[10px]" style={{ color: 'var(--text-5)' }}>{label}</div>
                  </div>
                ))}
              </div>

              {/* Insights */}
              {analysis.reasoning.length > 0 && (
                <div className="mb-3">
                  <div className="text-[10px] font-semibold uppercase tracking-wider mb-1 flex items-center gap-1" style={{ color: 'var(--text-5)' }}>
                    <Target className="w-3 h-3" /> Insights
                  </div>
                  <div className="space-y-0.5">
                    {analysis.reasoning.map((r, i) => (
                      <div key={i} className="text-[11px] px-2 py-1" style={{ color: 'var(--text-3)', background: 'var(--bg-3)' }}>
                        • {r}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Key patterns */}
              {analysis.keyPatterns.length > 0 && (
                <div className="mb-3">
                  <div className="text-[10px] font-semibold uppercase tracking-wider mb-1 flex items-center gap-1" style={{ color: 'var(--text-5)' }}>
                    <BarChart2 className="w-3 h-3" /> Key Patterns
                  </div>
                  <div className="space-y-0.5">
                    {analysis.keyPatterns.map((kp, i) => {
                      const dirColor = kp.pattern.direction === 'bullish' ? 'var(--success)' : kp.pattern.direction === 'bearish' ? 'var(--danger)' : 'var(--info)';
                      const impColor = kp.impact === 'high' ? 'var(--danger)' : kp.impact === 'medium' ? 'var(--warning)' : 'var(--info)';
                      return (
                        <div key={i} className="flex items-center gap-2 px-2 py-1" style={{ background: 'var(--bg-3)' }}>
                          <span className="text-[11px] font-medium flex-1 truncate" style={{ color: 'var(--text-2)' }}>{kp.pattern.label}</span>
                          <span className="text-[10px] font-semibold flex-shrink-0" style={{ color: dirColor }}>{kp.pattern.direction.slice(0,4).toUpperCase()}</span>
                          <span className="text-[10px] flex-shrink-0" style={{ color: 'var(--text-5)' }}>{(kp.pattern.confidence * 100).toFixed(0)}%</span>
                          <span className="text-[10px] font-bold flex-shrink-0" style={{ color: impColor }}>{kp.impact.toUpperCase()}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Recommendations */}
              {analysis.recommendations.length > 0 && (
                <div>
                  <div className="text-[10px] font-semibold uppercase tracking-wider mb-1 flex items-center gap-1" style={{ color: 'var(--text-5)' }}>
                    <Lightbulb className="w-3 h-3" /> Recommendations
                  </div>
                  <div className="space-y-0.5">
                    {analysis.recommendations.map((rec, i) => (
                      <div key={i} className="text-[11px] px-2 py-1 border-l" style={{ color: 'var(--text-3)', background: 'var(--bg-3)', borderColor: 'var(--accent)' }}>
                        {rec}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {startDate && endDate && (
                <div className="mt-3 text-[10px] text-center" style={{ color: 'var(--text-5)' }}>
                  {new Date(startDate).toLocaleDateString()} – {new Date(endDate).toLocaleDateString()} · updates on zoom
                </div>
              )}
            </div>
          )}

          {/* ══ Settings Tab ══ */}
          {activeTab === 'settings' && (
            <div>
              {/* Quick status */}
              <div className="flex items-center justify-between mb-3 text-[11px]">
                <div className="flex gap-3" style={{ color: 'var(--text-5)' }}>
                  <span>Detected: <span className="font-semibold" style={{ color: 'var(--accent)' }}>{patterns.length}</span></span>
                  <span>Conf ≥ <span className="font-mono" style={{ color: 'var(--text-3)' }}>{(settings.minConfidence * 100).toFixed(0)}%</span></span>
                  <span>Max: <span className="font-mono" style={{ color: 'var(--text-3)' }}>{settings.maxPatterns}</span></span>
                </div>
                {isDetecting && <span className="text-[10px] animate-pulse" style={{ color: 'var(--accent)' }}>Detecting…</span>}
              </div>

              {/* Presets */}
              <div className="mb-3">
                <div className="text-[10px] font-semibold uppercase tracking-wider mb-1.5" style={{ color: 'var(--text-5)' }}>Preset</div>
                <div className="grid grid-cols-4 gap-1">
                  {(['conservative', 'balanced', 'aggressive', 'custom'] as PatternPreset[]).map(p => (
                    <button
                      key={p}
                      onClick={() => handlePreset(p)}
                      className="py-1.5 text-[10px] font-semibold border transition-all"
                      style={{
                        background: currentPreset === p ? 'var(--accent)' : 'var(--bg-3)',
                        borderColor: currentPreset === p ? 'var(--accent)' : 'var(--bg-1)',
                        color: currentPreset === p ? 'var(--text-0)' : 'var(--text-4)',
                      }}
                    >
                      {p === 'conservative' ? 'Cons' : p === 'aggressive' ? 'Aggr' : p.charAt(0).toUpperCase() + p.slice(1, 4)}
                    </button>
                  ))}
                </div>
                <div className="text-[10px] mt-1" style={{ color: 'var(--text-5)' }}>
                  {currentPreset === 'conservative' && 'Fewer, high-confidence patterns only'}
                  {currentPreset === 'balanced'     && 'Optimal balance of quantity and quality'}
                  {currentPreset === 'aggressive'   && 'More patterns at lower confidence'}
                  {currentPreset === 'custom'       && 'Custom configuration active'}
                </div>
              </div>

              {/* Advanced toggle */}
              <button
                onClick={() => setShowAdv(v => !v)}
                className="flex items-center gap-1.5 text-[10px] font-semibold w-full py-1.5 px-2 border transition-all mb-2"
                style={{
                  color: showAdv ? 'var(--accent)' : 'var(--text-4)',
                  borderColor: showAdv ? 'var(--accent)' : 'var(--bg-1)',
                  background: 'var(--bg-3)',
                }}
              >
                <SlidersHorizontal className="w-3 h-3" />
                Advanced Settings
                {showAdv ? <ChevronUp className="w-3 h-3 ml-auto" /> : <ChevronDown className="w-3 h-3 ml-auto" />}
              </button>

              {showAdv && (
                <div className="space-y-3">
                  {[
                    { label: 'Min Confidence', value: (settings.minConfidence * 100).toFixed(0) + '%', min: 0.3, max: 0.9, step: 0.05, key: 'minConfidence' as const, parse: parseFloat },
                    { label: 'Max Patterns',   value: String(settings.maxPatterns),             min: 5,   max: 50,  step: 5,    key: 'maxPatterns' as const,   parse: parseInt },
                    { label: 'Max Per Type',   value: String(settings.maxPatternsPerType),       min: 1,   max: 10,  step: 1,    key: 'maxPatternsPerType' as const, parse: parseInt },
                    { label: 'Min Data Pts',   value: String(settings.minWindow),               min: 10,  max: 60,  step: 5,    key: 'minWindow' as const,     parse: parseInt },
                  ].map(({ label, value, min, max, step, key, parse }) => (
                    <div key={key}>
                      <div className="flex justify-between text-[10px] mb-1" style={{ color: 'var(--text-4)' }}>
                        <span>{label}</span>
                        <span className="font-mono" style={{ color: 'var(--accent)' }}>{value}</span>
                      </div>
                      <input
                        type="range"
                        min={min} max={max} step={step}
                        value={key === 'minConfidence' ? settings.minConfidence : (settings as any)[key]}
                        onChange={e => handleSetting(key, parse(e.target.value))}
                        className="w-full accent-green-400"
                        style={{ height: 4 }}
                      />
                    </div>
                  ))}

                  <div>
                    <div className="text-[10px] mb-1" style={{ color: 'var(--text-4)' }}>
                      Detection Windows (days, comma-separated)
                    </div>
                    <input
                      type="text"
                      value={windowsText}
                      onChange={e => setWindowsText(e.target.value)}
                      onBlur={() => commitWindows(windowsText)}
                      className="w-full px-2 py-1 border font-mono text-[10px]"
                      style={{ background: 'var(--bg-4)', borderColor: 'var(--bg-1)', color: 'var(--text-2)' }}
                      placeholder="45, 90, 180, 365"
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
