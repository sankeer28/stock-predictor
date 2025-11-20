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

// Balanced settings - Optimized for SPEED (most users will use this)
// Fast training while maintaining good accuracy
export const DEFAULT_ML_SETTINGS: MLSettings = {
  epochs: 15,                    // Reduced from 30 for MUCH faster browser training
  learningRate: 0.002,          // Higher for faster convergence
  batchSize: 256,               // Doubled from 128 (faster per epoch)
  lookbackWindow: 10,           // Reduced from 15 (faster processing)
  dropout: 0.1,                 // Reduced for speed
  l2Regularization: 0.0001,     // Lighter regularization for speed
  dampingFactor: 0.7,           // Reduced from 0.5 (less aggressive dampening)
  confidenceInterval: 1.96,     // 95% confidence (statistical standard)
  earlyStoppingPatience: 3,     // Reduced from 4 (stop even earlier)
  validationSplit: 0.08,        // Reduced from 0.1 (more training data, faster)
};

// Conservative: More accurate, slower training (for users who want best accuracy)
export const CONSERVATIVE_SETTINGS: MLSettings = {
  epochs: 80,                   // More epochs for better accuracy
  learningRate: 0.0005,         // Lower LR (more stable, slower learning)
  batchSize: 32,                // Smaller batches (more accurate updates)
  lookbackWindow: 40,           // Longer history for more context
  dropout: 0.3,                 // Higher dropout for better regularization
  l2Regularization: 0.01,       // Stronger L2 (prevent overfitting)
  dampingFactor: 0.8,           // Strong dampening (smooth predictions)
  confidenceInterval: 2.58,     // 99% confidence (wider bands)
  earlyStoppingPatience: 10,    // More patience for better convergence
  validationSplit: 0.25,        // More validation for better accuracy assessment
};

// Aggressive: Less regularization, more responsive to recent patterns
export const AGGRESSIVE_SETTINGS: MLSettings = {
  epochs: 60,                   // Reduced from 100 (early stopping prevents overtraining)
  learningRate: 0.0015,         // Higher LR (faster learning, slightly increased from 0.001)
  batchSize: 64,                // Increased from 16 for much faster training
  lookbackWindow: 15,           // Reduced from 20 for faster processing
  dropout: 0.15,                // Lower dropout (learn more patterns, but not too low)
  l2Regularization: 0.0001,     // Minimal regularization
  dampingFactor: 0.6,           // Light dampening (more volatile predictions)
  confidenceInterval: 1.64,     // 90% confidence (tighter bands)
  earlyStoppingPatience: 8,     // Reduced from 10 (faster stopping)
  validationSplit: 0.1,         // Reduced from 0.15 (more training data)
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
