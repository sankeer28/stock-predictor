'use client';

import React, { useState, useCallback } from 'react';
import { Settings, ChevronDown, ChevronUp, Info } from 'lucide-react';
import {
  MLSettings,
  MLPreset,
  getPresetSettings,
  DEFAULT_ML_SETTINGS,
  CONSERVATIVE_SETTINGS,
  AGGRESSIVE_SETTINGS,
} from '@/types/mlSettings';

interface MLSettingsPanelProps {
  settings: MLSettings;
  onSettingsChange: (settings: MLSettings) => void;
  onPresetChange: (preset: MLPreset) => void;
  currentPreset: MLPreset;
  inlineMobile?: boolean;
}

const MLSettingsPanel = React.memo(function MLSettingsPanel({
  settings,
  onSettingsChange,
  onPresetChange,
  currentPreset,
  inlineMobile,
}: MLSettingsPanelProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handlePresetChange = useCallback((preset: MLPreset) => {
    onPresetChange(preset);
    onSettingsChange(getPresetSettings(preset));
  }, [onPresetChange, onSettingsChange]);

  const handleSettingChange = useCallback((key: keyof MLSettings, value: number) => {
    const newSettings = { ...settings, [key]: value };
    onSettingsChange(newSettings);
    // If user manually changes a setting, switch to custom preset
    if (currentPreset !== 'custom') {
      onPresetChange('custom');
    }
  }, [settings, onSettingsChange, currentPreset, onPresetChange]);

  return (
    <div className={`card mb-6 ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center gap-2">
          <Settings className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">ML Model Settings</span>
          <span className="text-xs px-2 py-0.5 rounded" style={{
            background: 'var(--bg-2)',
            color: 'var(--text-4)',
            border: '1px solid var(--bg-1)'
          }}>
            {currentPreset.charAt(0).toUpperCase() + currentPreset.slice(1)}
          </span>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5" style={{ color: 'var(--text-4)' }} />
        ) : (
          <ChevronDown className="w-5 h-5" style={{ color: 'var(--text-4)' }} />
        )}
      </div>

      {isExpanded && (
        <div className="mt-4 space-y-4">
          {/* Preset Selector */}
          <div>
            <div className="text-xs mb-2 font-semibold" style={{ color: 'var(--text-4)' }}>
              PRESET PROFILES
            </div>
            <div className="grid grid-cols-4 gap-1">
              {(['conservative', 'balanced', 'aggressive', 'custom'] as MLPreset[]).map((preset) => (
                <button
                  key={preset}
                  onClick={() => handlePresetChange(preset)}
                  className="px-1 py-1.5 text-[9px] font-medium border transition-all whitespace-nowrap overflow-hidden"
                  style={{
                    background: currentPreset === preset ? 'var(--accent)' : 'var(--bg-3)',
                    borderColor: currentPreset === preset ? 'var(--accent)' : 'var(--bg-1)',
                    color: currentPreset === preset ? 'var(--text-0)' : 'var(--text-3)',
                  }}
                >
                  {preset === 'conservative' ? 'CONSERV.' : preset === 'aggressive' ? 'AGGRESS.' : preset.toUpperCase()}
                </button>
              ))}
            </div>
            <div className="mt-2 text-xs p-2 rounded" style={{
              background: 'var(--bg-2)',
              color: 'var(--text-5)',
              border: '1px solid var(--bg-1)'
            }}>
              {currentPreset === 'conservative' && '⚠️ More stable, wider confidence intervals, less responsive to recent changes'}
              {currentPreset === 'balanced' && '⚖️ Default settings, balanced between accuracy and stability'}
              {currentPreset === 'aggressive' && '🚀 More responsive to trends, tighter predictions, higher learning rate'}
              {currentPreset === 'custom' && '🛠️ Custom settings - manually configured parameters'}
            </div>
          </div>

          {/* Basic Settings */}
          <div>
            <div className="text-xs mb-3 font-semibold" style={{ color: 'var(--text-4)' }}>
              TRAINING PARAMETERS
            </div>
            <div className="space-y-3">
              {/* Epochs */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs" style={{ color: 'var(--text-3)' }}>
                    Training Epochs
                  </label>
                  <span className="text-xs font-mono px-2 py-0.5 rounded" style={{
                    background: 'var(--bg-2)',
                    color: 'var(--accent)'
                  }}>
                    {settings.epochs}
                  </span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  step="5"
                  value={settings.epochs}
                  onChange={(e) => handleSettingChange('epochs', parseInt(e.target.value))}
                  className="w-full"
                  style={{ accentColor: 'var(--accent)' }}
                />
                <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-5)' }}>
                  <span>10 (Fast)</span>
                  <span>100 (Thorough)</span>
                </div>
              </div>

              {/* Lookback Window */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs" style={{ color: 'var(--text-3)' }}>
                    Lookback Window (Days)
                  </label>
                  <span className="text-xs font-mono px-2 py-0.5 rounded" style={{
                    background: 'var(--bg-2)',
                    color: 'var(--accent)'
                  }}>
                    {settings.lookbackWindow}
                  </span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="90"
                  step="5"
                  value={settings.lookbackWindow}
                  onChange={(e) => handleSettingChange('lookbackWindow', parseInt(e.target.value))}
                  className="w-full"
                  style={{ accentColor: 'var(--accent)' }}
                />
                <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-5)' }}>
                  <span>10 (Short-term)</span>
                  <span>90 (Long-term)</span>
                </div>
              </div>

              {/* Damping Factor */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs" style={{ color: 'var(--text-3)' }}>
                    Prediction Damping
                  </label>
                  <span className="text-xs font-mono px-2 py-0.5 rounded" style={{
                    background: 'var(--bg-2)',
                    color: 'var(--accent)'
                  }}>
                    {settings.dampingFactor.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={settings.dampingFactor}
                  onChange={(e) => handleSettingChange('dampingFactor', parseFloat(e.target.value))}
                  className="w-full"
                  style={{ accentColor: 'var(--accent)' }}
                />
                <div className="flex justify-between text-xs mt-1" style={{ color: 'var(--text-5)' }}>
                  <span>0.1 (Aggressive)</span>
                  <span>1.0 (Conservative)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Advanced Settings Toggle */}
          <div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center gap-2 text-xs font-medium transition-colors"
              style={{ color: 'var(--accent)' }}
            >
              {showAdvanced ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              Advanced Settings
            </button>
          </div>

          {/* Advanced Settings */}
          {showAdvanced && (
            <div className="space-y-3 p-3 rounded" style={{
              background: 'var(--bg-2)',
              border: '1px solid var(--bg-1)'
            }}>
              {/* Learning Rate */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs" style={{ color: 'var(--text-3)' }}>
                    Learning Rate
                  </label>
                  <span className="text-xs font-mono px-2 py-0.5 rounded" style={{
                    background: 'var(--bg-3)',
                    color: 'var(--accent)'
                  }}>
                    {settings.learningRate.toFixed(4)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0001"
                  max="0.01"
                  step="0.0001"
                  value={settings.learningRate}
                  onChange={(e) => handleSettingChange('learningRate', parseFloat(e.target.value))}
                  className="w-full"
                  style={{ accentColor: 'var(--accent)' }}
                />
              </div>

              {/* Dropout */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs" style={{ color: 'var(--text-3)' }}>
                    Dropout Rate
                  </label>
                  <span className="text-xs font-mono px-2 py-0.5 rounded" style={{
                    background: 'var(--bg-3)',
                    color: 'var(--accent)'
                  }}>
                    {settings.dropout.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0"
                  max="0.5"
                  step="0.05"
                  value={settings.dropout}
                  onChange={(e) => handleSettingChange('dropout', parseFloat(e.target.value))}
                  className="w-full"
                  style={{ accentColor: 'var(--accent)' }}
                />
              </div>

              {/* L2 Regularization */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs" style={{ color: 'var(--text-3)' }}>
                    L2 Regularization
                  </label>
                  <span className="text-xs font-mono px-2 py-0.5 rounded" style={{
                    background: 'var(--bg-3)',
                    color: 'var(--accent)'
                  }}>
                    {settings.l2Regularization.toFixed(4)}
                  </span>
                </div>
                <input
                  type="range"
                  min="0.0001"
                  max="0.1"
                  step="0.0001"
                  value={settings.l2Regularization}
                  onChange={(e) => handleSettingChange('l2Regularization', parseFloat(e.target.value))}
                  className="w-full"
                  style={{ accentColor: 'var(--accent)' }}
                />
              </div>

              {/* Batch Size */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs" style={{ color: 'var(--text-3)' }}>
                    Batch Size
                  </label>
                  <span className="text-xs font-mono px-2 py-0.5 rounded" style={{
                    background: 'var(--bg-3)',
                    color: 'var(--accent)'
                  }}>
                    {settings.batchSize}
                  </span>
                </div>
                <input
                  type="range"
                  min="8"
                  max="64"
                  step="8"
                  value={settings.batchSize}
                  onChange={(e) => handleSettingChange('batchSize', parseInt(e.target.value))}
                  className="w-full"
                  style={{ accentColor: 'var(--accent)' }}
                />
              </div>

              {/* Confidence Interval */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-xs" style={{ color: 'var(--text-3)' }}>
                    Confidence Level
                  </label>
                  <span className="text-xs font-mono px-2 py-0.5 rounded" style={{
                    background: 'var(--bg-3)',
                    color: 'var(--accent)'
                  }}>
                    {settings.confidenceInterval === 1.64 ? '90%' :
                     settings.confidenceInterval === 1.96 ? '95%' :
                     settings.confidenceInterval === 2.58 ? '99%' : 'Custom'}
                  </span>
                </div>
                <select
                  value={settings.confidenceInterval}
                  onChange={(e) => handleSettingChange('confidenceInterval', parseFloat(e.target.value))}
                  className="w-full px-3 py-2 text-xs border"
                  style={{
                    background: 'var(--bg-3)',
                    borderColor: 'var(--bg-1)',
                    color: 'var(--text-2)',
                  }}
                >
                  <option value="1.64">90% Confidence</option>
                  <option value="1.96">95% Confidence</option>
                  <option value="2.58">99% Confidence</option>
                </select>
              </div>
            </div>
          )}

          {/* Info Box */}
          <div className="flex gap-2 p-3 rounded text-xs" style={{
            background: 'var(--bg-2)',
            borderLeft: '3px solid var(--info)',
            color: 'var(--text-4)'
          }}>
            <Info className="w-4 h-4 flex-shrink-0" style={{ color: 'var(--info)' }} />
            <div>
              <strong style={{ color: 'var(--info)' }}>Tip:</strong> Changing these settings will retrain all neural network models.
              Higher epochs and lookback windows = more accurate but slower training.
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default MLSettingsPanel;
