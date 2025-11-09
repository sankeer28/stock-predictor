import { StockData } from '@/types';

export interface MLPrediction {
  date: string;
  predicted: number;
  algorithm: string;
}

/**
 * Linear Regression - Weighted regression favoring recent data
 */
export function generateLinearRegression(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);

  // Use recent 120 days for better trend capture
  const recentWindow = Math.min(120, closePrices.length);
  const recentPrices = closePrices.slice(-recentWindow);
  const n = recentPrices.length;

  // Weighted linear regression: more recent data has higher weight
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;
  let sumW = 0;

  for (let i = 0; i < n; i++) {
    // Exponential weight: recent data weighted more heavily
    const weight = Math.exp((i - n) / (n / 2)); // Exponential decay

    sumX += i * weight;
    sumY += recentPrices[i] * weight;
    sumXY += i * recentPrices[i] * weight;
    sumX2 += i * i * weight;
    sumW += weight;
  }

  // Weighted least squares
  const meanX = sumX / sumW;
  const meanY = sumY / sumW;
  const slope = (sumXY - sumW * meanX * meanY) / (sumX2 - sumW * meanX * meanX);
  const intercept = meanY - slope * meanX;

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
 * Polynomial Regression (degree 2) - Captures curved trends with improved fitting
 */
export function generatePolynomialRegression(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);

  // Use recent 90 days for better curve fitting
  const recentWindow = Math.min(90, closePrices.length);
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

  // Calculate volatility for adaptive bounds
  const returns: number[] = [];
  for (let i = 1; i < recentPrices.length; i++) {
    returns.push(Math.abs((recentPrices[i] - recentPrices[i - 1]) / recentPrices[i - 1]));
  }
  const avgVolatility = returns.reduce((sum, r) => sum + r, 0) / returns.length;

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    // Normalize x value for prediction (continuing from 1.0)
    const x = 1 + (i / (n - 1));
    let predicted = a * x * x + b * x + c;

    // Apply adaptive bounds based on volatility and time horizon
    // More volatile stocks get wider bounds
    const timeAdjustment = Math.sqrt(i + 1); // Uncertainty grows with time
    const maxDeviation = currentPrice * Math.max(0.15, avgVolatility * 10) * timeAdjustment;
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
 * Moving Average Prediction - Adaptive window based on volatility
 */
export function generateMovingAverageForecast(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);

  // Calculate volatility to determine optimal window size
  const returns: number[] = [];
  for (let i = 1; i < closePrices.length; i++) {
    returns.push(Math.abs((closePrices[i] - closePrices[i - 1]) / closePrices[i - 1]));
  }
  const avgVolatility = returns.reduce((sum, r) => sum + r, 0) / returns.length;

  // Adaptive window: lower volatility = longer window (smoother)
  // Higher volatility = shorter window (more responsive)
  const baseWindow = Math.min(30, Math.floor(closePrices.length / 2));
  const windowSize = Math.max(10, Math.floor(baseWindow * (1 - avgVolatility * 20)));

  // Calculate moving average
  const recentPrices = closePrices.slice(-windowSize);
  const avgPrice = recentPrices.reduce((sum, price) => sum + price, 0) / windowSize;

  // Calculate trend from recent prices with weighted approach
  const halfWindow = Math.floor(windowSize / 2);
  const firstHalf = recentPrices.slice(0, halfWindow);
  const secondHalf = recentPrices.slice(-halfWindow);

  const firstAvg = firstHalf.reduce((sum, p) => sum + p, 0) / halfWindow;
  const secondAvg = secondHalf.reduce((sum, p) => sum + p, 0) / halfWindow;

  // Enhanced trend calculation with momentum
  const trend = (secondAvg - firstAvg) / halfWindow;
  const momentum = (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices.length;

  // Generate predictions
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    // Combine trend and momentum with decay over time
    const decayFactor = Math.exp(-i / forecastDays); // Momentum decays over time
    const predicted = Math.max(0, avgPrice + trend * (i + 1) + momentum * (i + 1) * decayFactor);

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'Moving Average',
    });
  }

  return predictions;
}

/**
 * Exponential Moving Average - Enhanced with dual EMA and momentum
 */
export function generateEMAForecast(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);

  // Dual EMA approach: fast (12) and slow (26) for better trend detection
  const fastPeriod = 12;
  const slowPeriod = 26;
  const fastMultiplier = 2 / (fastPeriod + 1);
  const slowMultiplier = 2 / (slowPeriod + 1);

  // Calculate Fast EMA
  let fastEMA = closePrices.slice(0, fastPeriod).reduce((sum, p) => sum + p, 0) / fastPeriod;
  for (let i = fastPeriod; i < closePrices.length; i++) {
    fastEMA = (closePrices[i] - fastEMA) * fastMultiplier + fastEMA;
  }

  // Calculate Slow EMA
  let slowEMA = closePrices.slice(0, slowPeriod).reduce((sum, p) => sum + p, 0) / slowPeriod;
  for (let i = slowPeriod; i < closePrices.length; i++) {
    slowEMA = (closePrices[i] - slowEMA) * slowMultiplier + slowEMA;
  }

  // MACD-like signal: difference between fast and slow EMA indicates momentum
  const macdSignal = fastEMA - slowEMA;

  // Calculate trend from recent price action
  const recentWindow = Math.min(20, closePrices.length);
  const recentPrices = closePrices.slice(-recentWindow);
  const trend = (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentWindow;

  // Generate predictions
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    // Use fast EMA as baseline and incorporate trend + MACD momentum
    // MACD signal decays over time as short-term momentum is less predictive long-term
    const decayFactor = Math.exp(-i / (forecastDays / 2));
    const predicted = Math.max(0, fastEMA + trend * (i + 1) + macdSignal * decayFactor);

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'Exponential MA',
    });
  }

  return predictions;
}
