import * as tf from '@tensorflow/tfjs';
import { StockData } from '@/types';

// Normalize data to 0-1 range for better training
function normalizeData(data: number[]): { normalized: number[], min: number, max: number } {
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min;

  if (range === 0) {
    return { normalized: data.map(() => 0.5), min, max };
  }

  const normalized = data.map(val => (val - min) / range);
  return { normalized, min, max };
}

// Denormalize predictions back to original scale
function denormalizeData(normalized: number[], min: number, max: number): number[] {
  const range = max - min;
  return normalized.map(val => val * range + min);
}

// Create sequences for LSTM training
function createSequences(data: number[], lookback: number): { X: number[][][], y: number[] } {
  const X: number[][][] = [];
  const y: number[] = [];

  for (let i = lookback; i < data.length; i++) {
    const sequence = data.slice(i - lookback, i).map(val => [val]);
    X.push(sequence);
    y.push(data[i]);
  }

  return { X, y };
}

// Build LSTM model (balanced accuracy and speed)
function buildLSTMModel(lookback: number): tf.Sequential {
  const model = tf.sequential();

  // Single LSTM layer with more units for better accuracy
  model.add(tf.layers.lstm({
    units: 48,  // Increased from 32 for better pattern recognition
    returnSequences: false,
    inputShape: [lookback, 1],
    kernelInitializer: 'glorotNormal',  // Avoids orthogonal warning
    recurrentInitializer: 'glorotNormal',
  }));
  model.add(tf.layers.dropout({ rate: 0.25 }));

  // Dense layer for better fitting
  model.add(tf.layers.dense({
    units: 16,
    activation: 'relu',
    kernelInitializer: 'glorotNormal'
  }));
  model.add(tf.layers.dropout({ rate: 0.15 }));

  // Output layer
  model.add(tf.layers.dense({ units: 1 }));

  // Compile model with adaptive optimizer for faster convergence
  model.compile({
    optimizer: tf.train.adam(0.002, 0.9, 0.999, 1e-7),  // Higher LR with beta tuning
    loss: 'meanSquaredError',
    metrics: ['mae'],
  });

  return model;
}

export interface MLForecast {
  date: string;
  predicted: number;
  upper: number;
  lower: number;
}

/**
 * Generate ML-based forecast using LSTM neural network
 */
export async function generateMLForecast(
  stockData: StockData[],
  forecastDays: number = 30
): Promise<MLForecast[]> {
  try {
    // Extract closing prices
    const closePrices = stockData.map(d => d.close);

    // Need at least 90 days of data for training (20 lookback + enough training data)
    if (closePrices.length < 90) {
      throw new Error('Insufficient data for ML forecasting (minimum 90 days required)');
    }

    // Normalize data
    const { normalized, min, max } = normalizeData(closePrices);

    // Create training sequences (20 days lookback - more context for better accuracy)
    const lookback = 20;
    const { X, y } = createSequences(normalized, lookback);

    // Convert to tensors
    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build and train model
    console.log('Training LSTM model with optimized configuration...');
    const model = buildLSTMModel(lookback);

    // Early stopping callback for efficiency
    let bestValLoss = Infinity;
    let patienceCounter = 0;
    const patience = 5; // Stop if no improvement for 5 epochs
    let shouldStop = false;

    await model.fit(xsTensor, ysTensor, {
      epochs: 40,  // Max epochs, but will stop early if converged
      batchSize: 24,  // Larger batch = faster training
      validationSplit: 0.1,  // More data for training
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const valLoss = logs?.val_loss || Infinity;

          // Early stopping logic
          if (valLoss < bestValLoss) {
            bestValLoss = valLoss;
            patienceCounter = 0;
          } else {
            patienceCounter++;
          }

          if (patienceCounter >= patience) {
            shouldStop = true;
            console.log(`Early stopping at epoch ${epoch}. Best val_loss: ${bestValLoss.toFixed(4)}`);
            model.stopTraining = true;
          }

          if (epoch % 5 === 0 || shouldStop) {
            console.log(`Epoch ${epoch}: loss = ${logs?.loss.toFixed(4)}, val_loss = ${valLoss.toFixed(4)}`);
          }
        }
      }
    });

    console.log('LSTM model training complete!');

    // Make predictions with optimized memory management
    const predictions: number[] = [];
    let currentSequence = normalized.slice(-lookback);

    // Use tidy() to automatically dispose intermediate tensors
    for (let i = 0; i < forecastDays; i++) {
      const predictedValue = await tf.tidy(() => {
        // Prepare input
        const input = tf.tensor3d([currentSequence.map(val => [val])]);

        // Predict next value
        const prediction = model.predict(input) as tf.Tensor;
        return prediction;
      }).data().then(data => data[0]);

      predictions.push(predictedValue);

      // Update sequence for next prediction (memory efficient)
      currentSequence.shift();
      currentSequence.push(predictedValue);
    }

    // Denormalize predictions
    const denormalizedPredictions = denormalizeData(predictions, min, max);

    // Calculate volatility for more accurate confidence intervals
    const returns: number[] = [];
    for (let i = 1; i < closePrices.length; i++) {
      returns.push((closePrices[i] - closePrices[i - 1]) / closePrices[i - 1]);
    }
    const volatility = Math.sqrt(
      returns.reduce((sum, r) => sum + r * r, 0) / returns.length
    );

    // Calculate confidence intervals based on volatility
    const lastDate = new Date(stockData[stockData.length - 1].date);
    const forecasts: MLForecast[] = denormalizedPredictions.map((predicted, i) => {
      const forecastDate = new Date(lastDate);
      forecastDate.setDate(forecastDate.getDate() + i + 1);

      // Calculate uncertainty bounds (increases with time and volatility)
      // Using ±2 standard deviations for ~95% confidence interval
      const timeDecay = Math.sqrt(i + 1); // Uncertainty grows with sqrt of time
      const baseUncertainty = predicted * volatility * timeDecay * 2.0;

      return {
        date: forecastDate.toISOString().split('T')[0],
        predicted,
        upper: predicted + baseUncertainty,
        lower: Math.max(0, predicted - baseUncertainty),
      };
    });

    // Clean up
    xsTensor.dispose();
    ysTensor.dispose();
    model.dispose();

    return forecasts;

  } catch (error) {
    console.error('ML Forecasting error:', error);
    throw error;
  }
}

/**
 * Get forecast insights from ML predictions
 */
export function getMLForecastInsights(
  currentPrice: number,
  forecasts: MLForecast[]
) {
  if (forecasts.length === 0) {
    return null;
  }

  // Short term: 7 days
  const shortTermIndex = Math.min(6, forecasts.length - 1);
  const shortTermPrice = forecasts[shortTermIndex].predicted;
  const shortTermChange = ((shortTermPrice - currentPrice) / currentPrice) * 100;

  // Medium term: 30 days
  const mediumTermIndex = Math.min(29, forecasts.length - 1);
  const mediumTermPrice = forecasts[mediumTermIndex].predicted;
  const mediumTermChange = ((mediumTermPrice - currentPrice) / currentPrice) * 100;

  // Calculate trend
  const firstPrediction = forecasts[0].predicted;
  const lastPrediction = forecasts[forecasts.length - 1].predicted;
  const overallTrend = lastPrediction > firstPrediction ? 'bullish' : 'bearish';
  const trendStrength = Math.abs(((lastPrediction - firstPrediction) / firstPrediction) * 100);

  // Calculate volatility from confidence intervals
  const avgConfidenceRange = forecasts.reduce((sum, f) =>
    sum + (f.upper - f.lower), 0) / forecasts.length;
  const volatilityScore = (avgConfidenceRange / currentPrice) * 100;

  return {
    shortTerm: {
      price: shortTermPrice,
      change: shortTermChange,
      days: shortTermIndex + 1,
    },
    mediumTerm: {
      price: mediumTermPrice,
      change: mediumTermChange,
      days: mediumTermIndex + 1,
    },
    trend: {
      direction: overallTrend,
      strength: trendStrength,
    },
    volatility: {
      score: volatilityScore,
      level: volatilityScore < 5 ? 'low' : volatilityScore < 10 ? 'medium' : 'high',
    },
    confidence: Math.max(50, Math.min(95, 100 - (volatilityScore * 3))), // Higher volatility = lower confidence
  };
}
