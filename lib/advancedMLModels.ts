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

// Normalize multiple features
function normalizeMultiFeatureData(features: number[][]): { 
  normalized: number[][], 
  mins: number[], 
  maxs: number[] 
} {
  const numFeatures = features.length;
  const normalized: number[][] = [];
  const mins: number[] = [];
  const maxs: number[] = [];

  for (let f = 0; f < numFeatures; f++) {
    const result = normalizeData(features[f]);
    normalized.push(result.normalized);
    mins.push(result.min);
    maxs.push(result.max);
  }

  return { normalized, mins, maxs };
}

// Create multi-feature sequences for training (OHLCV + indicators)
function createMultiFeatureSequences(
  features: number[][], 
  lookback: number
): { X: number[][][], y: number[] } {
  const X: number[][][] = [];
  const y: number[] = [];
  const numSamples = features[0].length;
  const numFeatures = features.length;

  for (let i = lookback; i < numSamples; i++) {
    const sequence: number[][] = [];
    
    // Create lookback window for all features
    for (let t = i - lookback; t < i; t++) {
      const timeStep: number[] = [];
      for (let f = 0; f < numFeatures; f++) {
        timeStep.push(features[f][t]);
      }
      sequence.push(timeStep);
    }
    
    X.push(sequence);
    // Predict price change (using close price - first feature)
    const priceChange = features[0][i] - features[0][i - 1];
    y.push(priceChange);
  }

  return { X, y };
}

// Create sequences for training (single feature - backward compatibility)
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

// Prepare multi-feature data from stock data (OHLCV + technical indicators)
function prepareMultiFeatureData(stockData: StockData[]): number[][] {
  const features: number[][] = [];
  
  // Price features
  features.push(stockData.map(d => d.close));      // 0: Close
  features.push(stockData.map(d => d.open));       // 1: Open
  features.push(stockData.map(d => d.high));       // 2: High
  features.push(stockData.map(d => d.low));        // 3: Low
  
  // Volume feature (normalized by itself)
  features.push(stockData.map(d => d.volume));     // 4: Volume
  
  // Price-based features
  const typicalPrices = stockData.map((d, i) => (d.high + d.low + d.close) / 3);
  features.push(typicalPrices);                    // 5: Typical Price
  
  // Price range (volatility indicator)
  const priceRanges = stockData.map(d => d.high - d.low);
  features.push(priceRanges);                      // 6: Price Range
  
  // Returns (momentum indicator)
  const returns = [0];  // First return is 0
  for (let i = 1; i < stockData.length; i++) {
    returns.push((stockData[i].close - stockData[i - 1].close) / stockData[i - 1].close);
  }
  features.push(returns);                          // 7: Returns
  
  return features;
}

/**
 * GRU (Gated Recurrent Unit) - Enhanced with multi-feature OHLCV input
 */
export async function generateGRUForecast(
  stockData: StockData[],
  forecastDays: number = 30,
  settings?: MLSettings
): Promise<MLPrediction[]> {
  const mlSettings = settings || DEFAULT_ML_SETTINGS;
  try {
    if (stockData.length < 60) {
      throw new Error('Insufficient data for GRU forecasting');
    }

    // Prepare multi-feature data
    const features = prepareMultiFeatureData(stockData);
    const { normalized, mins, maxs } = normalizeMultiFeatureData(features);
    const numFeatures = features.length;
    
    const lookback = mlSettings.lookbackWindow;
    const { X, y } = createMultiFeatureSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build GRU model with multi-feature input
    const model = tf.sequential();
    model.add(tf.layers.gru({
      units: 24,  // Increased for multi-feature processing
      returnSequences: false,
      inputShape: [lookback, numFeatures],  // Multiple features per timestep
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

    // Make predictions with multi-features
    const predictions: number[] = [];
    const closePrices = stockData.map(d => d.close);
    let currentPrice = closePrices[closePrices.length - 1];
    
    // Initialize sequence with last lookback window (all features)
    let currentSequence: number[][] = [];
    for (let t = normalized[0].length - lookback; t < normalized[0].length; t++) {
      const timeStep: number[] = [];
      for (let f = 0; f < numFeatures; f++) {
        timeStep.push(normalized[f][t]);
      }
      currentSequence.push(timeStep);
    }

    for (let i = 0; i < forecastDays; i++) {
      const predictedChange = await tf.tidy(() => {
        const input = tf.tensor3d([currentSequence]);
        const prediction = model.predict(input) as tf.Tensor;
        return prediction;
      }).data().then(data => data[0]);

      const range = maxs[0] - mins[0];  // Use close price range
      const denormalizedChange = predictedChange * range * mlSettings.dampingFactor;
      currentPrice += denormalizedChange;

      // Update sequence with new prediction
      const normalizedNewPrice = (currentPrice - mins[0]) / range;
      const newTimeStep: number[] = [];
      
      // For features we can't predict (open, high, low), use close as estimate
      newTimeStep.push(normalizedNewPrice);  // close
      newTimeStep.push(normalizedNewPrice);  // open (approximate)
      newTimeStep.push(normalizedNewPrice * 1.005);  // high (slightly higher)
      newTimeStep.push(normalizedNewPrice * 0.995);  // low (slightly lower)
      
      // Volume and other features - use recent average
      for (let f = 4; f < numFeatures; f++) {
        const recentAvg = currentSequence.slice(-3).reduce((sum, ts) => sum + ts[f], 0) / 3;
        newTimeStep.push(recentAvg);
      }
      
      currentSequence.shift();
      currentSequence.push(newTimeStep);

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
 * 1D CNN (Convolutional Neural Network) - Enhanced with multi-feature OHLCV input
 */
export async function generate1DCNNForecast(
  stockData: StockData[],
  forecastDays: number = 30,
  settings?: MLSettings
): Promise<MLPrediction[]> {
  const mlSettings = settings || DEFAULT_ML_SETTINGS;
  try {
    if (stockData.length < 60) {
      throw new Error('Insufficient data for CNN forecasting');
    }

    // Prepare multi-feature data
    const features = prepareMultiFeatureData(stockData);
    const { normalized, mins, maxs } = normalizeMultiFeatureData(features);
    const numFeatures = features.length;
    
    const lookback = mlSettings.lookbackWindow;
    const { X, y } = createMultiFeatureSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build 1D CNN model with multi-feature input
    const model = tf.sequential();

    // Conv1D layer for pattern recognition across features
    model.add(tf.layers.conv1d({
      filters: 24,  // Increased for multi-feature processing
      kernelSize: 3,
      activation: 'relu',
      inputShape: [lookback, numFeatures],  // Multiple features per timestep
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

    // Make predictions with multi-features
    const predictions: number[] = [];
    const closePrices = stockData.map(d => d.close);
    let currentPrice = closePrices[closePrices.length - 1];
    
    // Initialize sequence with last lookback window (all features)
    let currentSequence: number[][] = [];
    for (let t = normalized[0].length - lookback; t < normalized[0].length; t++) {
      const timeStep: number[] = [];
      for (let f = 0; f < numFeatures; f++) {
        timeStep.push(normalized[f][t]);
      }
      currentSequence.push(timeStep);
    }

    for (let i = 0; i < forecastDays; i++) {
      const predictedChange = await tf.tidy(() => {
        const input = tf.tensor3d([currentSequence]);
        const prediction = model.predict(input) as tf.Tensor;
        return prediction;
      }).data().then(data => data[0]);

      const range = maxs[0] - mins[0];
      const denormalizedChange = predictedChange * range * mlSettings.dampingFactor;
      currentPrice += denormalizedChange;

      // Update sequence with new prediction
      const normalizedNewPrice = (currentPrice - mins[0]) / range;
      const newTimeStep: number[] = [];
      
      newTimeStep.push(normalizedNewPrice);
      newTimeStep.push(normalizedNewPrice);
      newTimeStep.push(normalizedNewPrice * 1.005);
      newTimeStep.push(normalizedNewPrice * 0.995);
      
      for (let f = 4; f < numFeatures; f++) {
        const recentAvg = currentSequence.slice(-3).reduce((sum, ts) => sum + ts[f], 0) / 3;
        newTimeStep.push(recentAvg);
      }
      
      currentSequence.shift();
      currentSequence.push(newTimeStep);

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
 * Hybrid CNN-LSTM - Enhanced with multi-feature OHLCV input
 */
export async function generateCNNLSTMForecast(
  stockData: StockData[],
  forecastDays: number = 30,
  settings?: MLSettings
): Promise<MLPrediction[]> {
  const mlSettings = settings || DEFAULT_ML_SETTINGS;
  try {
    if (stockData.length < 60) {
      throw new Error('Insufficient data for CNN-LSTM forecasting');
    }

    // Prepare multi-feature data
    const features = prepareMultiFeatureData(stockData);
    const { normalized, mins, maxs } = normalizeMultiFeatureData(features);
    const numFeatures = features.length;
    
    const lookback = mlSettings.lookbackWindow;
    const { X, y } = createMultiFeatureSequences(normalized, lookback);

    const xsTensor = tf.tensor3d(X);
    const ysTensor = tf.tensor2d(y, [y.length, 1]);

    // Build Hybrid CNN-LSTM model with multi-feature input
    const model = tf.sequential();

    // CNN for feature extraction across multiple features
    model.add(tf.layers.conv1d({
      filters: 16,  // Increased for multi-feature processing
      kernelSize: 2,
      activation: 'relu',
      inputShape: [lookback, numFeatures],  // Multiple features per timestep
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

    // Make predictions with multi-features
    const predictions: number[] = [];
    const closePrices = stockData.map(d => d.close);
    let currentPrice = closePrices[closePrices.length - 1];
    
    // Initialize sequence with last lookback window (all features)
    let currentSequence: number[][] = [];
    for (let t = normalized[0].length - lookback; t < normalized[0].length; t++) {
      const timeStep: number[] = [];
      for (let f = 0; f < numFeatures; f++) {
        timeStep.push(normalized[f][t]);
      }
      currentSequence.push(timeStep);
    }

    for (let i = 0; i < forecastDays; i++) {
      const predictedChange = await tf.tidy(() => {
        const input = tf.tensor3d([currentSequence]);
        const prediction = model.predict(input) as tf.Tensor;
        return prediction;
      }).data().then(data => data[0]);

      const range = maxs[0] - mins[0];
      const denormalizedChange = predictedChange * range * mlSettings.dampingFactor;
      currentPrice += denormalizedChange;

      // Update sequence with new prediction
      const normalizedNewPrice = (currentPrice - mins[0]) / range;
      const newTimeStep: number[] = [];
      
      newTimeStep.push(normalizedNewPrice);
      newTimeStep.push(normalizedNewPrice);
      newTimeStep.push(normalizedNewPrice * 1.005);
      newTimeStep.push(normalizedNewPrice * 0.995);
      
      for (let f = 4; f < numFeatures; f++) {
        const recentAvg = currentSequence.slice(-3).reduce((sum, ts) => sum + ts[f], 0) / 3;
        newTimeStep.push(recentAvg);
      }
      
      currentSequence.shift();
      currentSequence.push(newTimeStep);

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
 * FAST VERSION: Uses pre-computed predictions instead of retraining
 * 
 * @param existingPredictions - Already trained model predictions
 * @param forecastDays - Number of days to forecast
 */
export function generateEnsembleFromPredictions(
  existingPredictions: {
    gru?: MLPrediction[] | null;
    cnn?: MLPrediction[] | null;
    cnnLstm?: MLPrediction[] | null;
    lstm?: { predicted: number; date: string }[] | null;
  },
  forecastDays: number = 30
): MLPrediction[] | null {
  try {
    // Collect all valid predictions
    const validModels: MLPrediction[][] = [];
    
    if (existingPredictions.gru && existingPredictions.gru.length > 0) {
      validModels.push(existingPredictions.gru);
    }
    if (existingPredictions.cnn && existingPredictions.cnn.length > 0) {
      validModels.push(existingPredictions.cnn);
    }
    if (existingPredictions.cnnLstm && existingPredictions.cnnLstm.length > 0) {
      validModels.push(existingPredictions.cnnLstm);
    }
    if (existingPredictions.lstm && existingPredictions.lstm.length > 0) {
      // Convert LSTM format to MLPrediction format
      validModels.push(existingPredictions.lstm.map(p => ({
        date: p.date,
        predicted: p.predicted,
        algorithm: 'LSTM'
      })));
    }
    
    if (validModels.length < 2) {
      console.warn('Not enough models for ensemble (need at least 2)');
      return null;
    }

    // Combine predictions using weighted median approach
    const ensemblePredictions: MLPrediction[] = [];

    for (let i = 0; i < forecastDays; i++) {
      // Get predictions from all models for this day
      const predictions = validModels
        .filter(model => i < model.length)
        .map(model => model[i].predicted);
      
      if (predictions.length === 0) break;

      // Use median instead of mean for robustness against outliers
      const sortedPredictions = [...predictions].sort((a, b) => a - b);
      const medianPrediction = sortedPredictions[Math.floor(sortedPredictions.length / 2)];
      
      // Calculate weighted average (more weight to values closer to median)
      const weights = predictions.map(p => {
        const distance = Math.abs(p - medianPrediction) / Math.max(medianPrediction, 1);
        return Math.exp(-distance * 5); // Exponential weighting
      });
      const totalWeight = weights.reduce((sum, w) => sum + w, 0);
      const weightedAvg = predictions.reduce((sum, p, idx) => sum + p * weights[idx], 0) / totalWeight;

      // Blend median and weighted average (60% median for stability)
      const finalPrediction = medianPrediction * 0.6 + weightedAvg * 0.4;

      // Use date from first valid model
      const date = validModels.find(m => i < m.length)?.[i].date || '';

      ensemblePredictions.push({
        date,
        predicted: finalPrediction,
        algorithm: 'Ensemble',
      });
    }

    console.log(`✅ Ensemble created from ${validModels.length} models (instant, no retraining!)`);
    return ensemblePredictions;
  } catch (error) {
    console.error('Ensemble error:', error);
    return null;
  }
}

/**
 * DEPRECATED: Old slow version that retrains models
 * Use generateEnsembleFromPredictions() instead
 */
export async function generateEnsembleForecast(
  stockData: StockData[],
  forecastDays: number = 30,
  settings?: MLSettings
): Promise<MLPrediction[]> {
  console.warn('⚠️ Using slow ensemble (retraining models). Use generateEnsembleFromPredictions() instead!');
  const mlSettings = settings || DEFAULT_ML_SETTINGS;
  
  try {
    // Generate predictions from multiple models
    const [gruPreds, cnnPreds, cnnLstmPreds] = await Promise.all([
      generateGRUForecast(stockData, forecastDays, mlSettings).catch(() => null),
      generate1DCNNForecast(stockData, forecastDays, mlSettings).catch(() => null),
      generateCNNLSTMForecast(stockData, forecastDays, mlSettings).catch(() => null),
    ]);

    return generateEnsembleFromPredictions(
      { gru: gruPreds, cnn: cnnPreds, cnnLstm: cnnLstmPreds },
      forecastDays
    ) || [];
  } catch (error) {
    console.error('Ensemble error:', error);
    throw error;
  }
}
