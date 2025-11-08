import { StockData, ForecastData } from '@/types';
import { addDays, format, isWeekend } from 'date-fns';

/**
 * Simple linear regression
 */
function linearRegression(y: number[]): { slope: number; intercept: number } {
  const n = y.length;
  const x = Array.from({ length: n }, (_, i) => i);

  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
  const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;

  return { slope, intercept };
}

/**
 * Calculate exponential smoothing forecast
 */
function exponentialSmoothing(data: number[], alpha: number = 0.3): number {
  let forecast = data[0];

  for (let i = 1; i < data.length; i++) {
    forecast = alpha * data[i] + (1 - alpha) * forecast;
  }

  return forecast;
}

/**
 * Get next business day
 */
function getNextBusinessDay(date: Date): Date {
  let nextDay = addDays(date, 1);
  while (isWeekend(nextDay)) {
    nextDay = addDays(nextDay, 1);
  }
  return nextDay;
}

/**
 * Calculate volatility (standard deviation of returns)
 */
function calculateVolatility(prices: number[]): number {
  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }

  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;

  return Math.sqrt(variance);
}

/**
 * Generate price forecast using hybrid approach
 * Combines exponential smoothing, linear regression, and momentum
 */
export function generateForecast(
  stockData: StockData[],
  days: number = 30
): ForecastData[] {
  if (stockData.length < 30) {
    throw new Error('Insufficient data for forecasting (minimum 30 days required)');
  }

  const closePrices = stockData.map(d => d.close);
  const recentPrices = closePrices.slice(-60); // Use last 60 days for forecast

  // Calculate trend using linear regression
  const { slope, intercept } = linearRegression(recentPrices);

  // Calculate exponential smoothing baseline
  const esBaseline = exponentialSmoothing(recentPrices, 0.3);

  // Calculate volatility for confidence intervals
  const volatility = calculateVolatility(recentPrices);

  // Calculate momentum factor
  const shortTermAvg = recentPrices.slice(-5).reduce((a, b) => a + b, 0) / 5;
  const longTermAvg = recentPrices.slice(-20).reduce((a, b) => a + b, 0) / 20;
  const momentum = (shortTermAvg - longTermAvg) / longTermAvg;

  // Generate forecast
  const forecast: ForecastData[] = [];
  let currentDate = new Date(stockData[stockData.length - 1].date);
  const lastPrice = closePrices[closePrices.length - 1];

  for (let i = 1; i <= days; i++) {
    currentDate = getNextBusinessDay(currentDate);

    // Hybrid forecast: combine trend, exponential smoothing, and momentum
    const trendComponent = intercept + slope * (recentPrices.length + i);
    const esComponent = esBaseline;
    const momentumAdjustment = lastPrice * momentum * (i / days) * 0.5; // Decay momentum over time

    // Weighted combination
    let predicted = (
      trendComponent * 0.4 +
      esComponent * 0.4 +
      lastPrice * 0.2 +
      momentumAdjustment
    );

    // Apply mean reversion for longer forecasts
    if (i > 15) {
      const meanPrice = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length;
      const reversionFactor = (i - 15) / days;
      predicted = predicted * (1 - reversionFactor * 0.3) + meanPrice * (reversionFactor * 0.3);
    }

    // Calculate confidence intervals (widen over time)
    const timeDecay = Math.sqrt(i); // Standard error increases with sqrt of time
    const confidenceInterval = lastPrice * volatility * timeDecay * 1.96; // 95% confidence

    const upper = predicted + confidenceInterval;
    const lower = Math.max(0, predicted - confidenceInterval); // Price can't be negative

    forecast.push({
      date: format(currentDate, 'yyyy-MM-dd'),
      predicted,
      upper,
      lower,
    });
  }

  return forecast;
}

/**
 * Generate forecast insights
 */
export function getForecastInsights(
  currentPrice: number,
  forecast: ForecastData[]
) {
  if (forecast.length === 0) return null;

  const shortTermForecast = forecast[6]; // 7-day forecast
  const mediumTermForecast = forecast[29]; // 30-day forecast

  const shortTermChange = ((shortTermForecast.predicted - currentPrice) / currentPrice) * 100;
  const mediumTermChange = ((mediumTermForecast.predicted - currentPrice) / currentPrice) * 100;

  // Calculate trend strength
  const priceChange = mediumTermForecast.predicted - currentPrice;
  const avgRange = (mediumTermForecast.upper - mediumTermForecast.lower) / 2;
  const trendStrength = Math.min(Math.abs(priceChange / avgRange) * 100, 100);

  return {
    shortTerm: {
      price: shortTermForecast.predicted,
      change: shortTermChange,
      upper: shortTermForecast.upper,
      lower: shortTermForecast.lower,
    },
    mediumTerm: {
      price: mediumTermForecast.predicted,
      change: mediumTermChange,
      upper: mediumTermForecast.upper,
      lower: mediumTermForecast.lower,
    },
    trend: {
      direction: mediumTermChange > 0 ? 'upward' : 'downward',
      strength: trendStrength,
    },
  };
}
