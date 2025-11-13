import { StockData } from '@/types';
import { MLSettings, DEFAULT_ML_SETTINGS } from '@/types/mlSettings';

export interface ProphetForecast {
  date: string;
  predicted: number;
  upper: number;
  lower: number;
}

/**
 * Prophet-like Time Series Forecasting
 * Implements a simplified version of Facebook Prophet's additive model:
 * y(t) = g(t) + s(t) + h(t) + ε
 * where:
 * - g(t) is the trend (piecewise linear or logistic growth)
 * - s(t) is the seasonality (Fourier series)
 * - h(t) is the holiday effects (not implemented here)
 * - ε is the error term
 */

/**
 * Calculate linear trend using least squares regression
 */
function calculateTrend(data: number[]): { slope: number; intercept: number } {
  const n = data.length;
  const x = Array.from({ length: n }, (_, i) => i);

  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = data.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * data[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  return { slope, intercept };
}

/**
 * Calculate seasonality using Fourier series approximation
 * Detects weekly patterns (7-day cycles)
 */
function calculateSeasonality(data: number[], period: number = 7): number[] {
  const n = data.length;
  const seasonality: number[] = new Array(n).fill(0);

  // Use simplified Fourier series with first 2 harmonics
  const numHarmonics = 2;

  for (let k = 1; k <= numHarmonics; k++) {
    let a = 0;
    let b = 0;

    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / period;
      a += data[i] * Math.cos(angle);
      b += data[i] * Math.sin(angle);
    }

    a /= n;
    b /= n;

    for (let i = 0; i < n; i++) {
      const angle = (2 * Math.PI * k * i) / period;
      seasonality[i] += a * Math.cos(angle) + b * Math.sin(angle);
    }
  }

  return seasonality;
}

/**
 * Decompose time series into trend, seasonal, and residual components
 */
function decomposeTimeSeries(prices: number[]): {
  trend: number[];
  seasonal: number[];
  residual: number[];
} {
  const n = prices.length;

  // Calculate trend using moving average
  const windowSize = 7;
  const trend: number[] = [];

  for (let i = 0; i < n; i++) {
    const start = Math.max(0, i - Math.floor(windowSize / 2));
    const end = Math.min(n, i + Math.ceil(windowSize / 2));
    const window = prices.slice(start, end);
    const avg = window.reduce((sum, val) => sum + val, 0) / window.length;
    trend.push(avg);
  }

  // Detrend the data
  const detrended = prices.map((price, i) => price - trend[i]);

  // Calculate seasonality from detrended data
  const seasonal = calculateSeasonality(detrended, 7);

  // Calculate residuals
  const residual = prices.map((price, i) => price - trend[i] - seasonal[i]);

  return { trend, seasonal, residual };
}

/**
 * Calculate prediction intervals based on historical residuals
 */
function calculatePredictionIntervals(
  residuals: number[],
  forecastLength: number,
  confidenceInterval: number = 1.96
): { upper: number[]; lower: number[] } {
  // Calculate residual standard deviation
  const mean = residuals.reduce((sum, val) => sum + val, 0) / residuals.length;
  const variance = residuals.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / residuals.length;
  const std = Math.sqrt(variance);

  // User-defined confidence interval (1.64 = 90%, 1.96 = 95%, 2.58 = 99%)
  // Uncertainty grows with forecast horizon
  const upper: number[] = [];
  const lower: number[] = [];

  for (let i = 0; i < forecastLength; i++) {
    const uncertaintyMultiplier = 1 + (i / forecastLength) * 0.5; // Grows from 1 to 1.5
    const interval = confidenceInterval * std * uncertaintyMultiplier;
    upper.push(interval);
    lower.push(-interval);
  }

  return { upper, lower };
}

/**
 * Generate Prophet-like forecast
 */
export function generateProphetForecast(
  stockData: StockData[],
  forecastDays: number = 30,
  settings?: MLSettings
): ProphetForecast[] {
  const mlSettings = settings || DEFAULT_ML_SETTINGS;

  // Extract closing prices
  const prices = stockData.map(d => d.close);

  if (prices.length < 30) {
    throw new Error('Insufficient data for Prophet forecasting (minimum 30 days required)');
  }

  // Decompose the time series
  const { trend, seasonal, residual } = decomposeTimeSeries(prices);

  // Calculate trend parameters for extrapolation
  const trendParams = calculateTrend(trend);

  // Get average seasonality pattern for the last cycle
  const seasonalityPeriod = 7;
  const recentSeasonality = seasonal.slice(-seasonalityPeriod);

  // Calculate prediction intervals with user-defined confidence level
  const intervals = calculatePredictionIntervals(residual, forecastDays, mlSettings.confidenceInterval);

  // Generate forecasts
  const forecasts: ProphetForecast[] = [];
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const lastIndex = prices.length - 1;

  for (let i = 0; i < forecastDays; i++) {
    // Project trend forward
    const trendValue = trendParams.slope * (lastIndex + i + 1) + trendParams.intercept;

    // Add seasonal component (repeating pattern)
    const seasonalValue = recentSeasonality[i % seasonalityPeriod];

    // Combine components
    const predicted = trendValue + seasonalValue;

    // Add prediction intervals
    const upper = predicted + intervals.upper[i];
    const lower = Math.max(0, predicted + intervals.lower[i]); // Prices can't be negative

    // Calculate date
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    forecasts.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      upper,
      lower,
    });
  }

  return forecasts;
}

/**
 * Enhanced Prophet forecast with change point detection
 * Detects significant trend changes and adjusts accordingly
 */
export function generateProphetWithChangepoints(
  stockData: StockData[],
  forecastDays: number = 30,
  numChangepoints: number = 5,
  settings?: MLSettings
): ProphetForecast[] {
  const mlSettings = settings || DEFAULT_ML_SETTINGS;
  const prices = stockData.map(d => d.close);

  if (prices.length < 60) {
    // Fall back to simple forecast if not enough data
    return generateProphetForecast(stockData, forecastDays, settings);
  }

  // Detect changepoints (significant trend changes)
  const changepointIndices: number[] = [];
  const windowSize = Math.floor(prices.length / (numChangepoints + 1));

  for (let i = 1; i <= numChangepoints; i++) {
    changepointIndices.push(i * windowSize);
  }

  // Calculate trend for the most recent segment (after last changepoint)
  const lastChangepointIndex = changepointIndices[changepointIndices.length - 1] || 0;
  const recentPrices = prices.slice(lastChangepointIndex);
  const recentTrend = calculateTrend(recentPrices);

  // Decompose with focus on recent data
  const { seasonal, residual } = decomposeTimeSeries(prices);

  // Calculate intervals with user-defined confidence level
  const intervals = calculatePredictionIntervals(residual, forecastDays, mlSettings.confidenceInterval);

  // Generate forecasts using recent trend
  const forecasts: ProphetForecast[] = [];
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const recentStartIndex = prices.length - recentPrices.length;

  // Get recent seasonality
  const seasonalityPeriod = 7;
  const recentSeasonality = seasonal.slice(-seasonalityPeriod);

  for (let i = 0; i < forecastDays; i++) {
    // Use recent trend for projection
    const trendValue = recentTrend.slope * (recentPrices.length + i) + recentTrend.intercept;

    // Add seasonal component
    const seasonalValue = recentSeasonality[i % seasonalityPeriod];

    // Combine with user-defined damping factor to prevent extreme extrapolation
    const dampingFactor = Math.exp(-i / (forecastDays * 2)) * mlSettings.dampingFactor;
    const predicted = prices[prices.length - 1] + (trendValue - recentPrices[recentPrices.length - 1]) * dampingFactor + seasonalValue * 0.3;

    // Add prediction intervals
    const upper = predicted + intervals.upper[i];
    const lower = Math.max(0, predicted + intervals.lower[i]);

    // Calculate date
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    forecasts.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      upper,
      lower,
    });
  }

  return forecasts;
}
