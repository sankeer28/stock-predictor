export interface MLSettings {
  // Neural Network Settings
  epochs: number;
  learningRate: number;
  batchSize: number;
  lookbackWindow: number;

  // Regularization
  dropout: number;
  l2Regularization: number;

  // Prediction Settings
  dampingFactor: number;
  confidenceInterval: number; // Multiplier for uncertainty (1.96 for 95%, 2.58 for 99%)

  // Advanced
  earlyStoppingPatience: number;
  validationSplit: number;
}

export const DEFAULT_ML_SETTINGS: MLSettings = {
  epochs: 30,
  learningRate: 0.001,
  batchSize: 32,
  lookbackWindow: 10,
  dropout: 0.1,
  l2Regularization: 0.001,
  dampingFactor: 0.5,
  confidenceInterval: 1.96,
  earlyStoppingPatience: 5,
  validationSplit: 0.2,
};

export const CONSERVATIVE_SETTINGS: MLSettings = {
  epochs: 20,
  learningRate: 0.0005,
  batchSize: 32,
  lookbackWindow: 15,
  dropout: 0.2,
  l2Regularization: 0.01,
  dampingFactor: 0.7,
  confidenceInterval: 2.58, // 99% confidence
  earlyStoppingPatience: 3,
  validationSplit: 0.25,
};

export const AGGRESSIVE_SETTINGS: MLSettings = {
  epochs: 50,
  learningRate: 0.002,
  batchSize: 16,
  lookbackWindow: 7,
  dropout: 0.05,
  l2Regularization: 0.0001,
  dampingFactor: 0.3,
  confidenceInterval: 1.64, // 90% confidence
  earlyStoppingPatience: 8,
  validationSplit: 0.15,
};

export const BALANCED_SETTINGS: MLSettings = DEFAULT_ML_SETTINGS;

export type MLPreset = 'conservative' | 'balanced' | 'aggressive' | 'custom';

export function getPresetSettings(preset: MLPreset): MLSettings {
  switch (preset) {
    case 'conservative':
      return CONSERVATIVE_SETTINGS;
    case 'aggressive':
      return AGGRESSIVE_SETTINGS;
    case 'balanced':
      return BALANCED_SETTINGS;
    case 'custom':
      return DEFAULT_ML_SETTINGS;
    default:
      return DEFAULT_ML_SETTINGS;
  }
}
