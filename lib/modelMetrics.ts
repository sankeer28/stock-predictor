import { StockData } from '@/types';
import { MLPrediction } from './mlAlgorithms';
import { MLForecast } from './mlForecasting';

export interface ModelAccuracy {
  algorithm: string;
  mape: number;  // Mean Absolute Percentage Error
  rmse: number;  // Root Mean Squared Error
  mae: number;   // Mean Absolute Error
  direction: number;  // Direction accuracy (% of correct up/down predictions)
  score: number;  // Overall score (0-100, higher is better)
  grade: 'A' | 'B' | 'C' | 'D' | 'F';
}

/**
 * Calculate model accuracy using backtesting
 * Tests model predictions against actual historical data
 */
export function calculateModelAccuracy(
  stockData: StockData[],
  predictions: MLPrediction[] | MLForecast[],
  testDays: number = 7
): ModelAccuracy | null {
  if (stockData.length < testDays + 30 || predictions.length === 0) {
    return null;
  }

  // Use the last testDays of actual data for comparison
  const actualPrices = stockData.slice(-testDays).map(d => d.close);
  const predictedPrices = predictions.slice(0, testDays).map(p => p.predicted);

  if (actualPrices.length !== predictedPrices.length) {
    return null;
  }

  // Calculate MAPE (Mean Absolute Percentage Error)
  let mapeSum = 0;
  let maeSum = 0;
  let squaredErrors = 0;
  let correctDirections = 0;

  for (let i = 0; i < actualPrices.length; i++) {
    const actual = actualPrices[i];
    const predicted = predictedPrices[i];

    // MAPE
    const percentageError = Math.abs((actual - predicted) / actual) * 100;
    mapeSum += percentageError;

    // MAE
    const absoluteError = Math.abs(actual - predicted);
    maeSum += absoluteError;

    // RMSE
    squaredErrors += Math.pow(actual - predicted, 2);

    // Direction accuracy
    if (i > 0) {
      const actualDirection = actualPrices[i] > actualPrices[i - 1];
      const predictedDirection = predictedPrices[i] > predictedPrices[i - 1];
      if (actualDirection === predictedDirection) {
        correctDirections++;
      }
    }
  }

  const mape = mapeSum / actualPrices.length;
  const mae = maeSum / actualPrices.length;
  const rmse = Math.sqrt(squaredErrors / actualPrices.length);
  const directionAccuracy = (correctDirections / (actualPrices.length - 1)) * 100;

  // Calculate overall score (0-100)
  // Lower MAPE is better, so invert it
  const mapeScore = Math.max(0, 100 - mape * 2);  // MAPE of 50% = 0 score
  const directionScore = directionAccuracy;

  // Weighted score: 60% price accuracy, 40% direction accuracy
  const overallScore = Math.min(100, Math.max(0, mapeScore * 0.6 + directionScore * 0.4));

  // Assign grade
  let grade: 'A' | 'B' | 'C' | 'D' | 'F';
  if (overallScore >= 80) grade = 'A';
  else if (overallScore >= 70) grade = 'B';
  else if (overallScore >= 60) grade = 'C';
  else if (overallScore >= 50) grade = 'D';
  else grade = 'F';

  // Get algorithm name (handle both MLPrediction and MLForecast)
  const algorithmName = 'algorithm' in predictions[0] ? predictions[0].algorithm : 'Unknown';

  return {
    algorithm: algorithmName,
    mape,
    rmse,
    mae,
    direction: directionAccuracy,
    score: overallScore,
    grade,
  };
}

/**
 * Backtest a model by training on historical data and testing on holdout set
 */
export function backtestModel(
  stockData: StockData[],
  generatePredictions: (data: StockData[], days: number) => MLPrediction[] | Promise<MLPrediction[]>,
  testDays: number = 7
): Promise<ModelAccuracy | null> {
  return new Promise(async (resolve) => {
    try {
      if (stockData.length < testDays + 60) {
        resolve(null);
        return;
      }

      // Split data: train on all but last testDays
      const trainData = stockData.slice(0, -testDays);
      const testData = stockData.slice(-testDays);

      // Generate predictions
      const predictions = await generatePredictions(trainData, testDays);

      if (!predictions || predictions.length === 0) {
        resolve(null);
        return;
      }

      // Calculate errors
      const actualPrices = testData.map(d => d.close);
      const predictedPrices = predictions.map(p => p.predicted);

      let mapeSum = 0;
      let maeSum = 0;
      let squaredErrors = 0;
      let correctDirections = 0;

      for (let i = 0; i < Math.min(actualPrices.length, predictedPrices.length); i++) {
        const actual = actualPrices[i];
        const predicted = predictedPrices[i];

        mapeSum += Math.abs((actual - predicted) / actual) * 100;
        maeSum += Math.abs(actual - predicted);
        squaredErrors += Math.pow(actual - predicted, 2);

        if (i > 0) {
          const actualDirection = actualPrices[i] > actualPrices[i - 1];
          const predictedDirection = predictedPrices[i] > predictedPrices[i - 1];
          if (actualDirection === predictedDirection) {
            correctDirections++;
          }
        }
      }

      const n = Math.min(actualPrices.length, predictedPrices.length);
      const mape = mapeSum / n;
      const mae = maeSum / n;
      const rmse = Math.sqrt(squaredErrors / n);
      const directionAccuracy = (correctDirections / (n - 1)) * 100;

      const mapeScore = Math.max(0, 100 - mape * 2);
      const overallScore = Math.min(100, Math.max(0, mapeScore * 0.6 + directionAccuracy * 0.4));

      let grade: 'A' | 'B' | 'C' | 'D' | 'F';
      if (overallScore >= 80) grade = 'A';
      else if (overallScore >= 70) grade = 'B';
      else if (overallScore >= 60) grade = 'C';
      else if (overallScore >= 50) grade = 'D';
      else grade = 'F';

      // Get algorithm name (handle both MLPrediction and MLForecast)
      const algorithmName = 'algorithm' in predictions[0] ? predictions[0].algorithm : 'Unknown';

      resolve({
        algorithm: algorithmName,
        mape,
        rmse,
        mae,
        direction: directionAccuracy,
        score: overallScore,
        grade,
      });
    } catch (error) {
      console.error('Backtest error:', error);
      resolve(null);
    }
  });
}

/**
 * Compare multiple models and rank them by performance
 */
export function compareModels(accuracies: (ModelAccuracy | null)[]): ModelAccuracy[] {
  const validAccuracies = accuracies.filter(a => a !== null) as ModelAccuracy[];

  // Sort by overall score (descending)
  return validAccuracies.sort((a, b) => b.score - a.score);
}

/**
 * Get performance badge color based on grade
 */
export function getGradeColor(grade: 'A' | 'B' | 'C' | 'D' | 'F'): string {
  switch (grade) {
    case 'A': return '#10b981';  // green
    case 'B': return '#3b82f6';  // blue
    case 'C': return '#f59e0b';  // orange
    case 'D': return '#ef4444';  // red
    case 'F': return '#dc2626';  // dark red
  }
}

/**
 * Format MAPE for display
 */
export function formatMAPE(mape: number): string {
  return `${mape.toFixed(2)}%`;
}

/**
 * Get recommendation text based on model performance
 */
export function getModelRecommendation(accuracy: ModelAccuracy): string {
  if (accuracy.grade === 'A') {
    return 'Excellent - Highly reliable predictions';
  } else if (accuracy.grade === 'B') {
    return 'Good - Reliable predictions with minor deviations';
  } else if (accuracy.grade === 'C') {
    return 'Fair - Use with caution, moderate accuracy';
  } else if (accuracy.grade === 'D') {
    return 'Poor - Low accuracy, unreliable predictions';
  } else {
    return 'Very Poor - Not recommended for decision making';
  }
}

