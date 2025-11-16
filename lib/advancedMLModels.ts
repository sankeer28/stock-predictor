import * as tf from '@tensorflow/tfjs';
import { StockData } from '@/types';
import { MLSettings, DEFAULT_ML_SETTINGS } from '@/types/mlSettings';

export interface MLPrediction {
  date: string;
  predicted: number;
  algorithm: string;
}

// Normalize data helper
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

// Create sequences for training
function createSequences(data: number[], lookback: number): { X: number[][][], y: number[] } {
  const X: number[][][] = [];
  const y: number[] = [];

  for (let i = lookback; i < data.length; i++) {
    const sequence = data.slice(i - lookback, i).map(val => [val]);
    X.push(sequence);
    const priceChange = data[i] - data[i - 1];
    y.push(priceChange);
  }

  return { X, y };
}

/**
 * GRU (Gated Recurrent Unit) - Simpler than LSTM, often faster
 */
export async function generateGRUForecast(
  stockData: StockData[],
  forecastDays: number = 30,
  settings?: MLSettings
): Promise<MLPrediction[]> {
  const mlSettings = settings || DEFAULT_ML_SETTINGS;
  try {
    const closePrices = stockData.map(d => d.close);
    if (closePrices.length < 60) {
      throw new Error('Insufficient data for GRU forecasting');
    }

    const { normalized, min, max } = normalizeData(closePrices);
    const lookback = mlSettings.lookbackWindow;
    const { X, y } = createSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build GRU model (optimized for serverless - smaller and faster)
    const model = tf.sequential();
    model.add(tf.layers.gru({
      units: 16,  // Reduced from 24 for faster training
      returnSequences: false,
      inputShape: [lookback, 1],
      kernelInitializer: 'glorotUniform',
      recurrentInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: mlSettings.l2Regularization }),
    }));
    model.add(tf.layers.dropout({ rate: mlSettings.dropout }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
      optimizer: tf.train.adam(mlSettings.learningRate),
      loss: 'meanSquaredError',
      metrics: ['mae'],
    });

    // Train with early stopping
    let bestValLoss = Infinity;
    let patienceCounter = 0;

    await model.fit(xsTensor, ysTensor, {
      epochs: mlSettings.epochs,
      batchSize: mlSettings.batchSize,
      validationSplit: mlSettings.validationSplit,
      shuffle: true,
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const valLoss = logs?.val_loss || Infinity;
          if (valLoss < bestValLoss) {
            bestValLoss = valLoss;
            patienceCounter = 0;
          } else {
            patienceCounter++;
          }
          if (patienceCounter >= mlSettings.earlyStoppingPatience) {
            model.stopTraining = true;
          }
        }
      }
    });

    // Make predictions
    const predictions: number[] = [];
    let currentSequence = normalized.slice(-lookback);
    let currentPrice = closePrices[closePrices.length - 1];

    for (let i = 0; i < forecastDays; i++) {
      const predictedChange = await tf.tidy(() => {
        const input = tf.tensor3d([currentSequence.map(val => [val])]);
        const prediction = model.predict(input) as tf.Tensor;
        return prediction;
      }).data().then(data => data[0]);

      const range = max - min;
      const denormalizedChange = predictedChange * range * mlSettings.dampingFactor;
      currentPrice += denormalizedChange;

      const normalizedNewPrice = (currentPrice - min) / range;
      currentSequence.shift();
      currentSequence.push(normalizedNewPrice);

      predictions.push(currentPrice);
    }

    xsTensor.dispose();
    ysTensor.dispose();
    model.dispose();

    const lastDate = new Date(stockData[stockData.length - 1].date);
    return predictions.map((predicted, i) => ({
      date: new Date(lastDate.getTime() + (i + 1) * 86400000).toISOString().split('T')[0],
      predicted,
      algorithm: 'GRU',
    }));
  } catch (error) {
    console.error('GRU error:', error);
    throw error;
  }
}

/**
 * 1D CNN (Convolutional Neural Network) - Good for pattern recognition
 */
export async function generate1DCNNForecast(
  stockData: StockData[],
  forecastDays: number = 30,
  settings?: MLSettings
): Promise<MLPrediction[]> {
  const mlSettings = settings || DEFAULT_ML_SETTINGS;
  try {
    const closePrices = stockData.map(d => d.close);
    if (closePrices.length < 60) {
      throw new Error('Insufficient data for CNN forecasting');
    }

    const { normalized, min, max } = normalizeData(closePrices);
    const lookback = mlSettings.lookbackWindow;
    const { X, y } = createSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build 1D CNN model (optimized for serverless)
    const model = tf.sequential();

    // Single Conv1D layer for faster training
    model.add(tf.layers.conv1d({
      filters: 16,  // Reduced from 32
      kernelSize: 3,
      activation: 'relu',
      inputShape: [lookback, 1],
      padding: 'same',
      kernelInitializer: 'glorotUniform',
    }));
    model.add(tf.layers.dropout({ rate: mlSettings.dropout }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 4, activation: 'relu' }));  // Reduced from 8
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
      optimizer: tf.train.adam(mlSettings.learningRate),
      loss: 'meanSquaredError',
      metrics: ['mae'],
    });

    // Train
    await model.fit(xsTensor, ysTensor, {
      epochs: mlSettings.epochs,
      batchSize: mlSettings.batchSize,
      validationSplit: mlSettings.validationSplit,
      shuffle: true,
      verbose: 0,
    });

    // Make predictions
    const predictions: number[] = [];
    let currentSequence = normalized.slice(-lookback);
    let currentPrice = closePrices[closePrices.length - 1];

    for (let i = 0; i < forecastDays; i++) {
      const predictedChange = await tf.tidy(() => {
        const input = tf.tensor3d([currentSequence.map(val => [val])]);
        const prediction = model.predict(input) as tf.Tensor;
        return prediction;
      }).data().then(data => data[0]);

      const range = max - min;
      const denormalizedChange = predictedChange * range * mlSettings.dampingFactor;
      currentPrice += denormalizedChange;

      const normalizedNewPrice = (currentPrice - min) / range;
      currentSequence.shift();
      currentSequence.push(normalizedNewPrice);

      predictions.push(currentPrice);
    }

    xsTensor.dispose();
    ysTensor.dispose();
    model.dispose();

    const lastDate = new Date(stockData[stockData.length - 1].date);
    return predictions.map((predicted, i) => ({
      date: new Date(lastDate.getTime() + (i + 1) * 86400000).toISOString().split('T')[0],
      predicted,
      algorithm: '1D CNN',
    }));
  } catch (error) {
    console.error('1D CNN error:', error);
    throw error;
  }
}

/**
 * Hybrid CNN-LSTM - Combines pattern recognition (CNN) with temporal dependencies (LSTM)
 */
export async function generateCNNLSTMForecast(
  stockData: StockData[],
  forecastDays: number = 30,
  settings?: MLSettings
): Promise<MLPrediction[]> {
  const mlSettings = settings || DEFAULT_ML_SETTINGS;
  try {
    const closePrices = stockData.map(d => d.close);
    if (closePrices.length < 60) {
      throw new Error('Insufficient data for CNN-LSTM forecasting');
    }

    const { normalized, min, max } = normalizeData(closePrices);
    const lookback = mlSettings.lookbackWindow;
    const { X, y } = createSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build Hybrid CNN-LSTM model (optimized)
    const model = tf.sequential();

    // CNN for feature extraction
    model.add(tf.layers.conv1d({
      filters: 12,  // Reduced from 16
      kernelSize: 2,
      activation: 'relu',
      inputShape: [lookback, 1],
      padding: 'same',
    }));
    model.add(tf.layers.dropout({ rate: mlSettings.dropout }));

    // LSTM for temporal modeling
    model.add(tf.layers.lstm({
      units: 12,  // Reduced from 16
      returnSequences: false,
      kernelRegularizer: tf.regularizers.l2({ l2: mlSettings.l2Regularization }),
    }));
    model.add(tf.layers.dropout({ rate: mlSettings.dropout }));

    model.add(tf.layers.dense({ units: 4, activation: 'relu' }));  // Reduced from 8
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
      optimizer: tf.train.adam(mlSettings.learningRate),
      loss: 'meanSquaredError',
      metrics: ['mae'],
    });

    // Train
    await model.fit(xsTensor, ysTensor, {
      epochs: mlSettings.epochs,
      batchSize: mlSettings.batchSize,
      validationSplit: mlSettings.validationSplit,
      shuffle: true,
      verbose: 0,
    });

    // Make predictions
    const predictions: number[] = [];
    let currentSequence = normalized.slice(-lookback);
    let currentPrice = closePrices[closePrices.length - 1];

    for (let i = 0; i < forecastDays; i++) {
      const predictedChange = await tf.tidy(() => {
        const input = tf.tensor3d([currentSequence.map(val => [val])]);
        const prediction = model.predict(input) as tf.Tensor;
        return prediction;
      }).data().then(data => data[0]);

      const range = max - min;
      const denormalizedChange = predictedChange * range * mlSettings.dampingFactor;
      currentPrice += denormalizedChange;

      const normalizedNewPrice = (currentPrice - min) / range;
      currentSequence.shift();
      currentSequence.push(normalizedNewPrice);

      predictions.push(currentPrice);
    }

    xsTensor.dispose();
    ysTensor.dispose();
    model.dispose();

    const lastDate = new Date(stockData[stockData.length - 1].date);
    return predictions.map((predicted, i) => ({
      date: new Date(lastDate.getTime() + (i + 1) * 86400000).toISOString().split('T')[0],
      predicted,
      algorithm: 'CNN-LSTM',
    }));
  } catch (error) {
    console.error('CNN-LSTM error:', error);
    throw error;
  }
}

/**
 * Ensemble Model - Combines multiple predictions for better accuracy
 * Uses weighted average based on recent model performance
 */
export async function generateEnsembleForecast(
  stockData: StockData[],
  forecastDays: number = 30,
  settings?: MLSettings
): Promise<MLPrediction[]> {
  const mlSettings = settings || DEFAULT_ML_SETTINGS;
  try {
    // Generate predictions from multiple models
    const [gruPreds, cnnPreds, cnnLstmPreds] = await Promise.all([
      generateGRUForecast(stockData, forecastDays, mlSettings).catch(() => null),
      generate1DCNNForecast(stockData, forecastDays, mlSettings).catch(() => null),
      generateCNNLSTMForecast(stockData, forecastDays, mlSettings).catch(() => null),
    ]);

    // Filter out failed models
    const validModels = [gruPreds, cnnPreds, cnnLstmPreds].filter(p => p !== null) as MLPrediction[][];
    
    if (validModels.length === 0) {
      throw new Error('All ensemble models failed');
    }

    // Combine predictions using weighted average
    const ensemblePredictions: MLPrediction[] = [];
    const lastDate = new Date(stockData[stockData.length - 1].date);

    for (let i = 0; i < forecastDays; i++) {
      const predictions = validModels.map(model => model[i].predicted);
      
      // Use median instead of mean for robustness against outliers
      const sortedPredictions = predictions.sort((a, b) => a - b);
      const medianPrediction = sortedPredictions[Math.floor(sortedPredictions.length / 2)];
      
      // Also calculate weighted average (more weight to middle values)
      const weights = predictions.map(p => {
        const distance = Math.abs(p - medianPrediction) / medianPrediction;
        return Math.exp(-distance * 5); // Exponential weighting
      });
      const totalWeight = weights.reduce((sum, w) => sum + w, 0);
      const weightedAvg = predictions.reduce((sum, p, idx) => sum + p * weights[idx], 0) / totalWeight;

      // Blend median and weighted average
      const finalPrediction = medianPrediction * 0.6 + weightedAvg * 0.4;

      ensemblePredictions.push({
        date: new Date(lastDate.getTime() + (i + 1) * 86400000).toISOString().split('T')[0],
        predicted: finalPrediction,
        algorithm: 'Ensemble',
      });
    }

    return ensemblePredictions;
  } catch (error) {
    console.error('Ensemble error:', error);
    throw error;
  }
}
