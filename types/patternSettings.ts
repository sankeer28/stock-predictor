export type PatternPreset = 'conservative' | 'balanced' | 'aggressive' | 'custom';

export interface PatternSettings {
  detectionWindows: number[];
  minWindow: number;
  maxPatterns: number;
  maxPatternsPerType: number;
  minConfidence: number; // Global minimum confidence threshold
}

export const DEFAULT_PATTERN_SETTINGS: PatternSettings = {
  detectionWindows: [45, 90, 180, 365, 730, 1095],
  minWindow: 25,
  maxPatterns: 20,
  maxPatternsPerType: 3,
  minConfidence: 0.60,
};

export const PATTERN_PRESETS: Record<PatternPreset, PatternSettings> = {
  conservative: {
    detectionWindows: [90, 180, 365, 730],
    minWindow: 45,
    maxPatterns: 10,
    maxPatternsPerType: 2,
    minConfidence: 0.70,
  },
  balanced: {
    detectionWindows: [45, 90, 180, 365, 730, 1095],
    minWindow: 25,
    maxPatterns: 20,
    maxPatternsPerType: 3,
    minConfidence: 0.60,
  },
  aggressive: {
    detectionWindows: [30, 60, 90, 180, 365, 730, 1095, 1825],
    minWindow: 15,
    maxPatterns: 30,
    maxPatternsPerType: 5,
    minConfidence: 0.50,
  },
  custom: DEFAULT_PATTERN_SETTINGS,
};

export function getPatternPresetSettings(preset: PatternPreset): PatternSettings {
  return PATTERN_PRESETS[preset];
}
