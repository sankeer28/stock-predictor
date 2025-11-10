import * as tf from '@tensorflow/tfjs';
import { StockData } from '@/types';

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
  forecastDays: number = 30
): Promise<MLPrediction[]> {
  try {
    const closePrices = stockData.map(d => d.close);
    if (closePrices.length < 60) {
      throw new Error('Insufficient data for GRU forecasting');
    }

    const { normalized, min, max } = normalizeData(closePrices);
    const lookback = 10;
    const { X, y } = createSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build GRU model (simpler and faster than LSTM)
    const model = tf.sequential();
    model.add(tf.layers.gru({
      units: 24,
      returnSequences: false,
      inputShape: [lookback, 1],
      kernelInitializer: 'glorotUniform',
      recurrentInitializer: 'glorotUniform',
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mae'],
    });

    // Train with early stopping
    let bestValLoss = Infinity;
    let patienceCounter = 0;

    await model.fit(xsTensor, ysTensor, {
      epochs: 25,
      batchSize: 32,
      validationSplit: 0.2,
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
          if (patienceCounter >= 5) {
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
      const denormalizedChange = predictedChange * range * 0.5;  // Dampen
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
  forecastDays: number = 30
): Promise<MLPrediction[]> {
  try {
    const closePrices = stockData.map(d => d.close);
    if (closePrices.length < 60) {
      throw new Error('Insufficient data for CNN forecasting');
    }

    const { normalized, min, max } = normalizeData(closePrices);
    const lookback = 10;
    const { X, y } = createSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build 1D CNN model
    const model = tf.sequential();

    // Conv1D layers for pattern extraction
    model.add(tf.layers.conv1d({
      filters: 32,
      kernelSize: 3,
      activation: 'relu',
      inputShape: [lookback, 1],
      padding: 'same',
      kernelInitializer: 'glorotUniform',
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.conv1d({
      filters: 16,
      kernelSize: 3,
      activation: 'relu',
      padding: 'same',
    }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mae'],
    });

    // Train
    await model.fit(xsTensor, ysTensor, {
      epochs: 20,
      batchSize: 32,
      validationSplit: 0.2,
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
      const denormalizedChange = predictedChange * range * 0.5;
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
  forecastDays: number = 30
): Promise<MLPrediction[]> {
  try {
    const closePrices = stockData.map(d => d.close);
    if (closePrices.length < 60) {
      throw new Error('Insufficient data for CNN-LSTM forecasting');
    }

    const { normalized, min, max } = normalizeData(closePrices);
    const lookback = 10;
    const { X, y } = createSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build Hybrid CNN-LSTM model
    const model = tf.sequential();

    // CNN for feature extraction
    model.add(tf.layers.conv1d({
      filters: 16,
      kernelSize: 2,
      activation: 'relu',
      inputShape: [lookback, 1],
      padding: 'same',
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));

    // LSTM for temporal modeling
    model.add(tf.layers.lstm({
      units: 16,
      returnSequences: false,
      kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }),
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mae'],
    });

    // Train
    await model.fit(xsTensor, ysTensor, {
      epochs: 25,
      batchSize: 32,
      validationSplit: 0.2,
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
      const denormalizedChange = predictedChange * range * 0.5;
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
 * TFT (Temporal Fusion Transformer) - Simplified version
 * Full TFT is complex; this is a lightweight transformer-inspired model
 */
export async function generateTFTForecast(
  stockData: StockData[],
  forecastDays: number = 30
): Promise<MLPrediction[]> {
  try {
    const closePrices = stockData.map(d => d.close);
    if (closePrices.length < 60) {
      throw new Error('Insufficient data for TFT forecasting');
    }

    const { normalized, min, max } = normalizeData(closePrices);
    const lookback = 10;
    const { X, y } = createSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build simplified transformer-inspired model
    const model = tf.sequential();

    // Multi-head attention simulation with dense layers
    model.add(tf.layers.dense({
      units: 32,
      activation: 'relu',
      inputShape: [lookback, 1],
      kernelInitializer: 'glorotUniform',
    }));
    model.add(tf.layers.dropout({ rate: 0.1 }));

    model.add(tf.layers.dense({
      units: 16,
      activation: 'relu',
    }));

    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mae'],
    });

    // Train
    await model.fit(xsTensor, ysTensor, {
      epochs: 20,
      batchSize: 32,
      validationSplit: 0.2,
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
      const denormalizedChange = predictedChange * range * 0.5;
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
      algorithm: 'TFT',
    }));
  } catch (error) {
    console.error('TFT error:', error);
    throw error;
  }
}
