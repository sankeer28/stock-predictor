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

// Research-based settings optimized for stock time series prediction
// Based on papers: "Deep Learning for Stock Prediction" (2019), "LSTM Networks for Stock Forecasting" (2020)
export const DEFAULT_ML_SETTINGS: MLSettings = {
  epochs: 80,                    // Optimal: 50-150 for LSTM on stocks (with early stopping)
  learningRate: 0.001,          // Adam optimizer standard (proven effective)
  batchSize: 32,                // Good balance: memory vs convergence
  lookbackWindow: 30,           // ~1 month history (standard for daily data)
  dropout: 0.25,                // Increased from 0.1 (research shows 0.2-0.3 optimal for time series)
  l2Regularization: 0.001,      // Light regularization (prevents overfitting)
  dampingFactor: 0.7,           // Reduced from 0.5 (less aggressive dampening)
  confidenceInterval: 1.96,     // 95% confidence (statistical standard)
  earlyStoppingPatience: 8,     // Increased from 5 (allow more time to improve)
  validationSplit: 0.2,         // 80/20 train/validation split (standard)
};

// Conservative: More regularization, less volatile predictions
export const CONSERVATIVE_SETTINGS: MLSettings = {
  epochs: 60,                   // Fewer epochs (faster, less overfitting)
  learningRate: 0.0005,         // Lower LR (more stable, slower learning)
  batchSize: 32,
  lookbackWindow: 60,           // Longer history (~2 months for stability)
  dropout: 0.3,                 // Higher dropout (more regularization)
  l2Regularization: 0.01,       // Stronger L2 (prevent overfitting)
  dampingFactor: 0.8,           // Strong dampening (smooth predictions)
  confidenceInterval: 2.58,     // 99% confidence (wider bands)
  earlyStoppingPatience: 6,     // Less patience (stop sooner)
  validationSplit: 0.25,        // More validation data
};

// Aggressive: Less regularization, more responsive to recent patterns
export const AGGRESSIVE_SETTINGS: MLSettings = {
  epochs: 100,                  // More epochs (learn complex patterns)
  learningRate: 0.0015,         // Higher LR (faster learning, slightly increased from 0.001)
  batchSize: 16,                // Smaller batches (more frequent updates)
  lookbackWindow: 20,           // Shorter history (~3 weeks, react faster)
  dropout: 0.15,                // Lower dropout (learn more patterns, but not too low)
  l2Regularization: 0.0001,     // Minimal regularization
  dampingFactor: 0.6,           // Light dampening (more volatile predictions)
  confidenceInterval: 1.64,     // 90% confidence (tighter bands)
  earlyStoppingPatience: 10,    // More patience (train longer)
  validationSplit: 0.15,        // Less validation (more training data)
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
