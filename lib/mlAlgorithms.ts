import { StockData } from '@/types';

export interface MLPrediction {
  date: string;
  predicted: number;
  algorithm: string;
}

/**
 * Linear Regression - Fast and simple trend-based prediction
 */
export function generateLinearRegression(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);
  const n = closePrices.length;

  // Calculate linear regression: y = mx + b
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;

  for (let i = 0; i < n; i++) {
    sumX += i;
    sumY += closePrices[i];
    sumXY += i * closePrices[i];
    sumX2 += i * i;
  }

  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  // Generate predictions
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    const x = n + i;
    const predicted = Math.max(0, slope * x + intercept);

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'Linear Regression',
    });
  }

  return predictions;
}

/**
 * Polynomial Regression (degree 2) - Captures curved trends
 */
export function generatePolynomialRegression(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);

  // Use only recent 60 days to avoid extrapolation issues
  const recentWindow = Math.min(60, closePrices.length);
  const recentPrices = closePrices.slice(-recentWindow);
  const n = recentPrices.length;

  // Normalize x values to 0-1 range to prevent overflow
  const xValues = Array.from({ length: n }, (_, i) => i / (n - 1));

  // Calculate polynomial regression: y = ax^2 + bx + c
  let sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
  let sumY = 0, sumXY = 0, sumX2Y = 0;

  for (let i = 0; i < n; i++) {
    const x = xValues[i];
    const y = recentPrices[i];
    sumX += x;
    sumX2 += x * x;
    sumX3 += x * x * x;
    sumX4 += x * x * x * x;
    sumY += y;
    sumXY += x * y;
    sumX2Y += x * x * y;
  }

  // Solve system of equations using Cramer's rule
  const denominator = n * (sumX2 * sumX4 - sumX3 * sumX3) - sumX * (sumX * sumX4 - sumX2 * sumX3) + sumX2 * (sumX * sumX3 - sumX2 * sumX2);

  if (Math.abs(denominator) < 1e-10) {
    // Fallback to linear regression if polynomial is unstable
    return generateLinearRegression(stockData, forecastDays);
  }

  const a = (n * (sumX2Y * sumX2 - sumXY * sumX3) - sumY * (sumX * sumX2 - sumX2 * sumX) + sumXY * (sumX * sumX3 - sumX2 * sumX2)) / denominator;
  const b = (sumY * (sumX2 * sumX4 - sumX3 * sumX3) - sumX * (sumXY * sumX4 - sumX2Y * sumX3) + sumX2 * (sumXY * sumX3 - sumX2Y * sumX2)) / denominator;
  const c = (sumY * sumX2 * sumX4 - sumY * sumX3 * sumX3 - sumX * sumXY * sumX4 + sumX * sumX2Y * sumX3 + sumX2 * sumXY * sumX3 - sumX2 * sumX2Y * sumX2) / denominator;

  // Generate predictions
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];
  const currentPrice = recentPrices[recentPrices.length - 1];
  const avgPrice = recentPrices.reduce((sum, p) => sum + p, 0) / recentPrices.length;

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    // Normalize x value for prediction (continuing from 1.0)
    const x = 1 + (i / (n - 1));
    let predicted = a * x * x + b * x + c;

    // Apply reasonable bounds to prevent crazy extrapolations
    // Don't allow prediction to deviate more than 50% from current price
    const maxDeviation = currentPrice * 0.5;
    predicted = Math.max(currentPrice - maxDeviation, Math.min(currentPrice + maxDeviation, predicted));

    // Ensure positive price
    predicted = Math.max(avgPrice * 0.1, predicted);

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'Polynomial Regression',
    });
  }

  return predictions;
}

/**
 * Moving Average Prediction - Simple but effective
 */
export function generateMovingAverageForecast(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);
  const windowSize = Math.min(20, Math.floor(closePrices.length / 2));

  // Calculate moving average
  const recentPrices = closePrices.slice(-windowSize);
  const avgPrice = recentPrices.reduce((sum, price) => sum + price, 0) / windowSize;

  // Calculate trend from recent prices
  const halfWindow = Math.floor(windowSize / 2);
  const firstHalf = recentPrices.slice(0, halfWindow);
  const secondHalf = recentPrices.slice(-halfWindow);

  const firstAvg = firstHalf.reduce((sum, p) => sum + p, 0) / halfWindow;
  const secondAvg = secondHalf.reduce((sum, p) => sum + p, 0) / halfWindow;
  const trend = (secondAvg - firstAvg) / halfWindow;

  // Generate predictions
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    const predicted = Math.max(0, avgPrice + trend * (i + 1));

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'Moving Average',
    });
  }

  return predictions;
}

/**
 * Exponential Moving Average - Weights recent data more heavily
 */
export function generateEMAForecast(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);
  const period = 20;
  const multiplier = 2 / (period + 1);

  // Calculate EMA
  let ema = closePrices.slice(0, period).reduce((sum, p) => sum + p, 0) / period;

  for (let i = period; i < closePrices.length; i++) {
    ema = (closePrices[i] - ema) * multiplier + ema;
  }

  // Calculate trend from last 10 EMAs
  const recentEMAs: number[] = [ema];
  let tempEMA = ema;

  for (let i = closePrices.length - 10; i < closePrices.length; i++) {
    tempEMA = (closePrices[i] - tempEMA) * multiplier + tempEMA;
    recentEMAs.push(tempEMA);
  }

  // Linear trend from recent EMAs
  const trend = (recentEMAs[recentEMAs.length - 1] - recentEMAs[0]) / recentEMAs.length;

  // Generate predictions
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    const predicted = Math.max(0, ema + trend * (i + 1));

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'Exponential MA',
    });
  }

  return predictions;
}
