import * as tf from '@tensorflow/tfjs';
import { StockData } from '@/types';
import { MLSettings, DEFAULT_ML_SETTINGS } from '@/types/mlSettings';
import { getTensorFlowInfo } from './tfConfig';

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

// Create sequences for LSTM training - predict price changes instead of absolute prices
function createSequences(data: number[], lookback: number): { X: number[][][], y: number[] } {
  const X: number[][][] = [];
  const y: number[] = [];

  for (let i = lookback; i < data.length; i++) {
    // Use last 'lookback' days as features
    const sequence = data.slice(i - lookback, i).map(val => [val]);
    X.push(sequence);

    // Predict the CHANGE from last day to current day (prevents drift)
    const priceChange = data[i] - data[i - 1];
    y.push(priceChange);
  }

  return { X, y };
}

// Build LSTM model (optimized for serverless - faster training, good accuracy)
function buildLSTMModel(lookback: number, settings: MLSettings): tf.Sequential {
  const model = tf.sequential();

  // LSTM layer highly optimized for speed
  model.add(tf.layers.lstm({
    units: 8,  // Reduced from 16 for MUCH faster browser training
    returnSequences: false,
    inputShape: [lookback, 1],
    kernelInitializer: 'glorotUniform',  // More stable than glorotNormal
    recurrentInitializer: 'orthogonal',  // Faster initialization
    kernelRegularizer: tf.regularizers.l2({ l2: settings.l2Regularization }),
  }));
  model.add(tf.layers.dropout({ rate: settings.dropout }));

  // Direct to output layer (simpler, faster)
  model.add(tf.layers.dense({
    units: 1,
    activation: 'linear'
  }));

  // Compile with user-defined learning rate
  model.compile({
    optimizer: tf.train.adam(settings.learningRate),
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
  forecastDays: number = 30,
  settings?: MLSettings
): Promise<MLForecast[]> {
  // Use default settings if not provided
  const mlSettings = settings || DEFAULT_ML_SETTINGS;

  // Track performance
  const startTime = performance.now();
  const memoryBefore = tf.memory();

  try {
    // Log TensorFlow backend info
    const tfInfo = getTensorFlowInfo();
    console.log(`📊 LSTM Training started on ${tfInfo.backend.toUpperCase()} backend`);

    // Extract closing prices
    const closePrices = stockData.map(d => d.close);

    // Need at least 60 days of data for training
    if (closePrices.length < 60) {
      throw new Error('Insufficient data for ML forecasting (minimum 60 days required)');
    }

    // Normalize data
    const { normalized, min, max } = normalizeData(closePrices);

    // Create training sequences using user-defined lookback window
    const lookback = mlSettings.lookbackWindow;
    const { X, y } = createSequences(normalized, lookback);

    // Convert to tensors
    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build and train model
    console.log('Training LSTM model with user-defined configuration...');
    const model = buildLSTMModel(lookback, mlSettings);

    // Early stopping callback for efficiency
    let bestValLoss = Infinity;
    let patienceCounter = 0;
    const patience = mlSettings.earlyStoppingPatience;
    let shouldStop = false;

    await model.fit(xsTensor, ysTensor, {
      epochs: mlSettings.epochs,
      batchSize: mlSettings.batchSize,
      validationSplit: mlSettings.validationSplit,
      shuffle: true,  // Shuffle data for better generalization
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const valLoss = logs?.val_loss || Infinity;
          const trainLoss = logs?.loss || Infinity;

          // Early stopping logic with overfitting detection
          if (valLoss < bestValLoss) {
            bestValLoss = valLoss;
            patienceCounter = 0;
          } else {
            patienceCounter++;
          }

          // Stop if overfitting (validation loss increasing while training loss decreasing)
          if (valLoss > trainLoss * 1.5 && epoch > 10) {
            shouldStop = true;
            console.log(`Stopping due to overfitting at epoch ${epoch}`);
            model.stopTraining = true;
          }

          if (patienceCounter >= patience) {
            shouldStop = true;
            console.log(`Early stopping at epoch ${epoch}. Best val_loss: ${bestValLoss.toFixed(4)}`);
            model.stopTraining = true;
          }

          if (epoch % 5 === 0 || shouldStop) {
            console.log(`Epoch ${epoch}: loss = ${trainLoss.toFixed(4)}, val_loss = ${valLoss.toFixed(4)}`);
          }
        }
      }
    });

    console.log('LSTM model training complete!');

    // Make predictions with optimized memory management
    const predictedChanges: number[] = [];
    let currentSequence = normalized.slice(-lookback);
    let currentPrice = closePrices[closePrices.length - 1];

    // Use tidy() to automatically dispose intermediate tensors
    for (let i = 0; i < forecastDays; i++) {
      const predictedChange = await tf.tidy(() => {
        // Prepare input
        const input = tf.tensor3d([currentSequence.map(val => [val])]);

        // Predict next CHANGE
        const prediction = model.predict(input) as tf.Tensor;
        return prediction;
      }).data().then(data => data[0]);

      // Denormalize the change
      const range = max - min;
      const denormalizedChange = predictedChange * range;

      // Apply user-defined damping to control prediction volatility
      const dampedChange = denormalizedChange * mlSettings.dampingFactor;

      predictedChanges.push(dampedChange);

      // Update current price for next iteration
      currentPrice = currentPrice + dampedChange;

      // Normalize the new price and update sequence
      const normalizedNewPrice = (currentPrice - min) / range;
      currentSequence.shift();
      currentSequence.push(normalizedNewPrice);
    }

    // Build actual price predictions from changes
    let denormalizedPredictions: number[] = [];
    let price = closePrices[closePrices.length - 1];
    for (const change of predictedChanges) {
      price += change;
      denormalizedPredictions.push(price);
    }

    // Calculate volatility for confidence intervals
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
      // Using user-defined confidence interval (1.64 = 90%, 1.96 = 95%, 2.58 = 99%)
      const timeDecay = Math.sqrt(i + 1); // Uncertainty grows with sqrt of time
      const baseUncertainty = predicted * volatility * timeDecay * mlSettings.confidenceInterval;

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

    // Performance metrics
    const endTime = performance.now();
    const memoryAfter = tf.memory();
    const duration = ((endTime - startTime) / 1000).toFixed(2);
    const memoryUsed = ((memoryAfter.numBytes - memoryBefore.numBytes) / 1024 / 1024).toFixed(2);

    console.log(`✅ LSTM Training completed in ${duration}s`);
    console.log(`   Memory: ${memoryUsed} MB used, ${memoryAfter.numTensors} tensors`);
    console.log(`   Backend: ${getTensorFlowInfo().backend.toUpperCase()}`);

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
