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

// Build LSTM model (optimized for speed)
function buildLSTMModel(lookback: number): tf.Sequential {
  const model = tf.sequential();

  // Single LSTM layer with dropout (faster than 2 layers)
  model.add(tf.layers.lstm({
    units: 32,
    returnSequences: false,
    inputShape: [lookback, 1],
  }));
  model.add(tf.layers.dropout({ rate: 0.2 }));

  // Dense output layer
  model.add(tf.layers.dense({ units: 1 }));

  // Compile model
  model.compile({
    optimizer: tf.train.adam(0.001),
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

    // Need at least 60 days of data for training
    if (closePrices.length < 60) {
      throw new Error('Insufficient data for ML forecasting (minimum 60 days required)');
    }

    // Normalize data
    const { normalized, min, max } = normalizeData(closePrices);

    // Create training sequences (use last 10 days to predict next day - faster)
    const lookback = 10;
    const { X, y } = createSequences(normalized, lookback);

    // Convert to tensors
    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build and train model
    console.log('Training LSTM model...');
    const model = buildLSTMModel(lookback);

    await model.fit(xsTensor, ysTensor, {
      epochs: 15,  // Reduced from 50 for faster training
      batchSize: 16,  // Reduced from 32 for faster training
      validationSplit: 0.1,
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if (epoch % 5 === 0) {
            console.log(`Epoch ${epoch}: loss = ${logs?.loss.toFixed(4)}, val_loss = ${logs?.val_loss?.toFixed(4)}`);
          }
        }
      }
    });

    console.log('LSTM model training complete!');

    // Make predictions
    const predictions: number[] = [];
    let currentSequence = normalized.slice(-lookback);

    for (let i = 0; i < forecastDays; i++) {
      // Prepare input
      const input = tf.tensor3d([currentSequence.map(val => [val])]);

      // Predict next value
      const prediction = model.predict(input) as tf.Tensor;
      const predictedValue = (await prediction.data())[0];

      predictions.push(predictedValue);

      // Update sequence for next prediction
      currentSequence = [...currentSequence.slice(1), predictedValue];

      // Clean up tensors
      input.dispose();
      prediction.dispose();
    }

    // Denormalize predictions
    const denormalizedPredictions = denormalizeData(predictions, min, max);

    // Calculate confidence intervals (±5% for simplicity, can be improved with ensemble methods)
    const lastDate = new Date(stockData[stockData.length - 1].date);
    const forecasts: MLForecast[] = denormalizedPredictions.map((predicted, i) => {
      const forecastDate = new Date(lastDate);
      forecastDate.setDate(forecastDate.getDate() + i + 1);

      // Calculate uncertainty bounds (increases with time)
      const uncertaintyFactor = 1 + (i * 0.01); // 1% increase per day
      const baseUncertainty = predicted * 0.05 * uncertaintyFactor;

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
