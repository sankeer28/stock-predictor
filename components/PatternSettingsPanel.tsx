'use client';

import React, { useState } from 'react';
import { Settings, ChevronDown, ChevronUp } from 'lucide-react';
import { PatternSettings, PatternPreset, getPatternPresetSettings } from '@/types/patternSettings';

interface PatternSettingsPanelProps {
  settings: PatternSettings;
  onSettingsChange: (settings: PatternSettings) => void;
  onPresetChange: (preset: PatternPreset) => void;
  currentPreset: PatternPreset;
  inlineMobile?: boolean;
  patternCount?: number;
  isDetecting?: boolean;
}

export default function PatternSettingsPanel({
  settings,
  onSettingsChange,
  onPresetChange,
  currentPreset,
  inlineMobile = false,
  patternCount = 0,
  isDetecting = false,
}: PatternSettingsPanelProps) {
  const [expanded, setExpanded] = useState(false);

  const handlePresetChange = (preset: PatternPreset) => {
    console.log(`🎯 Pattern preset changed to: ${preset}`);
    onPresetChange(preset);
    const presetSettings = getPatternPresetSettings(preset);
    console.log('📋 New settings:', presetSettings);
    onSettingsChange(presetSettings);
  };

  const handleSettingChange = (key: keyof PatternSettings, value: any) => {
    const newSettings = { ...settings, [key]: value };
    console.log(`⚙️ Pattern setting changed: ${key} =`, value);
    onSettingsChange(newSettings);
    // Switch to custom preset when manually adjusting
    if (currentPreset !== 'custom') {
      onPresetChange('custom');
    }
  };

  const handleWindowsChange = (value: string) => {
    try {
      const windows = value.split(',').map(w => parseInt(w.trim())).filter(w => !isNaN(w) && w > 0);
      if (windows.length > 0) {
        handleSettingChange('detectionWindows', windows);
      }
    } catch (error) {
      console.error('Invalid windows format:', error);
    }
  };

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'} mb-4`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Settings className="w-4 h-4" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Pattern Detection Settings</span>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="transition-colors"
          style={{ color: 'var(--text-3)' }}
        >
          {expanded ? <ChevronUp className="w-5 h-5" /> : <ChevronDown className="w-5 h-5" />}
        </button>
      </div>

      {/* Active Settings Status */}
      <div
        className="border p-2 mb-4 text-xs"
        style={{
          background: 'var(--bg-3)',
          borderColor: 'var(--bg-1)',
          color: 'var(--text-3)',
        }}
      >
        <div className="flex items-center justify-between mb-1">
          <span className="font-semibold" style={{ color: 'var(--text-2)' }}>
            Active Settings
          </span>
          {isDetecting && (
            <span className="animate-pulse" style={{ color: 'var(--accent)' }}>
              Detecting...
            </span>
          )}
        </div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-[10px]">
          <div>
            <span style={{ color: 'var(--text-4)' }}>Windows:</span>{' '}
            <span className="font-mono">{settings.detectionWindows.slice(0, 3).join(', ')}...</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-4)' }}>Min Pts:</span>{' '}
            <span className="font-mono">{settings.minWindow}</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-4)' }}>Max Total:</span>{' '}
            <span className="font-mono">{settings.maxPatterns}</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-4)' }}>Max/Type:</span>{' '}
            <span className="font-mono">{settings.maxPatternsPerType}</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-4)' }}>Min Conf:</span>{' '}
            <span className="font-mono">{(settings.minConfidence * 100).toFixed(0)}%</span>
          </div>
          <div>
            <span style={{ color: 'var(--text-4)' }}>Detected:</span>{' '}
            <span className="font-mono font-semibold" style={{ color: 'var(--accent)' }}>
              {patternCount}
            </span>
          </div>
        </div>
      </div>

      {/* Preset Selector */}
      <div className="mb-4">
        <label className="text-xs font-semibold mb-2 block" style={{ color: 'var(--text-3)' }}>
          Preset
        </label>
        <div className="grid grid-cols-2 gap-2">
          {(['conservative', 'balanced', 'aggressive', 'custom'] as PatternPreset[]).map(preset => (
            <button
              key={preset}
              onClick={() => handlePresetChange(preset)}
              className="px-3 py-2 text-xs font-semibold border transition-all"
              style={{
                background: currentPreset === preset ? 'var(--accent)' : 'var(--bg-3)',
                borderColor: currentPreset === preset ? 'var(--accent)' : 'var(--bg-1)',
                color: currentPreset === preset ? 'var(--text-0)' : 'var(--text-3)',
              }}
            >
              {preset.charAt(0).toUpperCase() + preset.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Preset Descriptions */}
      <div className="mb-4 p-2 border text-xs" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}>
        {currentPreset === 'conservative' && '📊 Fewer, high-confidence patterns'}
        {currentPreset === 'balanced' && '⚖️ Optimal balance of patterns'}
        {currentPreset === 'aggressive' && '🔍 More patterns, lower confidence'}
        {currentPreset === 'custom' && '⚙️ Custom configuration'}
      </div>

      {expanded && (
        <div className="space-y-4">
          {/* Detection Windows */}
          <div>
            <label className="text-xs font-semibold mb-1 block" style={{ color: 'var(--text-3)' }}>
              Detection Windows (days)
            </label>
            <textarea
              rows={2}
              value={settings.detectionWindows.join(', ')}
              onChange={(e) => handleWindowsChange(e.target.value)}
              className="w-full px-2 py-1.5 border font-mono text-xs resize-none"
              style={{
                background: 'var(--bg-4)',
                borderColor: 'var(--bg-1)',
                color: 'var(--text-2)',
              }}
              placeholder="45, 90, 180, 365"
            />
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-5)' }}>
              Comma-separated list of lookback periods
            </p>
          </div>

          {/* Min Window */}
          <div>
            <label className="text-xs font-semibold mb-1 flex items-center justify-between" style={{ color: 'var(--text-3)' }}>
              <span>Min Data Points</span>
              <span style={{ color: 'var(--accent)' }}>{settings.minWindow}</span>
            </label>
            <input
              type="range"
              min="10"
              max="60"
              step="5"
              value={settings.minWindow}
              onChange={(e) => handleSettingChange('minWindow', parseInt(e.target.value))}
              className="w-full"
            />
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-5)' }}>
              Minimum data points required for pattern
            </p>
          </div>

          {/* Max Patterns */}
          <div>
            <label className="text-xs font-semibold mb-1 flex items-center justify-between" style={{ color: 'var(--text-3)' }}>
              <span>Max Total Patterns</span>
              <span style={{ color: 'var(--accent)' }}>{settings.maxPatterns}</span>
            </label>
            <input
              type="range"
              min="5"
              max="50"
              step="5"
              value={settings.maxPatterns}
              onChange={(e) => handleSettingChange('maxPatterns', parseInt(e.target.value))}
              className="w-full"
            />
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-5)' }}>
              Maximum patterns to display
            </p>
          </div>

          {/* Max Patterns Per Type */}
          <div>
            <label className="text-xs font-semibold mb-1 flex items-center justify-between" style={{ color: 'var(--text-3)' }}>
              <span>Max Per Type</span>
              <span style={{ color: 'var(--accent)' }}>{settings.maxPatternsPerType}</span>
            </label>
            <input
              type="range"
              min="1"
              max="10"
              step="1"
              value={settings.maxPatternsPerType}
              onChange={(e) => handleSettingChange('maxPatternsPerType', parseInt(e.target.value))}
              className="w-full"
            />
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-5)' }}>
              Maximum patterns per pattern type
            </p>
          </div>

          {/* Min Confidence */}
          <div>
            <label className="text-xs font-semibold mb-1 flex items-center justify-between" style={{ color: 'var(--text-3)' }}>
              <span>Min Confidence</span>
              <span style={{ color: 'var(--accent)' }}>{(settings.minConfidence * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min="0.3"
              max="0.9"
              step="0.05"
              value={settings.minConfidence}
              onChange={(e) => handleSettingChange('minConfidence', parseFloat(e.target.value))}
              className="w-full"
            />
            <p className="text-[10px] mt-1" style={{ color: 'var(--text-5)' }}>
              Minimum confidence threshold for patterns
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
