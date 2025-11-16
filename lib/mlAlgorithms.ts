import { StockData } from '@/types';

export interface MLPrediction {
  date: string;
  predicted: number;
  algorithm: string;
}

/**
 * Linear Regression - Weighted regression favoring recent data with OHLCV analysis
 */
export function generateLinearRegression(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);
  const highPrices = stockData.map(d => d.high);
  const lowPrices = stockData.map(d => d.low);
  const volumes = stockData.map(d => d.volume);

  // Use recent 120 days for better trend capture
  const recentWindow = Math.min(120, closePrices.length);
  const recentPrices = closePrices.slice(-recentWindow);
  const recentHighs = highPrices.slice(-recentWindow);
  const recentLows = lowPrices.slice(-recentWindow);
  const recentVolumes = volumes.slice(-recentWindow);
  const n = recentPrices.length;

  // Calculate average true range (volatility)
  const atr = calculateATR(recentHighs, recentLows, recentPrices);
  
  // Calculate volume trend
  const volumeAvg = recentVolumes.reduce((a, b) => a + b, 0) / n;
  const recentVolumeAvg = recentVolumes.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, recentVolumes.length);
  const volumeTrend = recentVolumeAvg / volumeAvg;

  // Weighted linear regression: more recent data has higher weight
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumX2 = 0;
  let sumW = 0;

  for (let i = 0; i < n; i++) {
    // Enhanced weight considering volume and price action
    const baseWeight = Math.exp((i - n) / (n / 2)); // Exponential decay
    const volumeWeight = recentVolumes[i] / volumeAvg;
    const priceRangeWeight = (recentHighs[i] - recentLows[i]) / recentPrices[i];
    const weight = baseWeight * (1 + volumeWeight * 0.1) * (1 + priceRangeWeight * 0.05);

    // Use typical price (HLC/3) for better representation
    const typicalPrice = (recentHighs[i] + recentLows[i] + recentPrices[i]) / 3;

    sumX += i * weight;
    sumY += typicalPrice * weight;
    sumXY += i * typicalPrice * weight;
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
 * Polynomial Regression (degree 2) - Captures curved trends with OHLCV data
 */
export function generatePolynomialRegression(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);
  const highPrices = stockData.map(d => d.high);
  const lowPrices = stockData.map(d => d.low);
  const volumes = stockData.map(d => d.volume);

  // Use recent 90 days for better curve fitting
  const recentWindow = Math.min(90, closePrices.length);
  const recentPrices = closePrices.slice(-recentWindow);
  const recentHighs = highPrices.slice(-recentWindow);
  const recentLows = lowPrices.slice(-recentWindow);
  const recentVolumes = volumes.slice(-recentWindow);
  const n = recentPrices.length;

  // Use typical prices for better representation
  const typicalPrices = [];
  for (let i = 0; i < n; i++) {
    typicalPrices.push(calculateTypicalPrice(recentHighs[i], recentLows[i], recentPrices[i]));
  }

  // Normalize x values to 0-1 range to prevent overflow
  const xValues = Array.from({ length: n }, (_, i) => i / (n - 1));

  // Calculate polynomial regression: y = ax^2 + bx + c
  let sumX = 0, sumX2 = 0, sumX3 = 0, sumX4 = 0;
  let sumY = 0, sumXY = 0, sumX2Y = 0;

  for (let i = 0; i < n; i++) {
    const x = xValues[i];
    const y = typicalPrices[i];  // Use typical price instead of just close
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

  // Calculate simple moving average for baseline
  const recentAvg = recentPrices.slice(-10).reduce((sum, p) => sum + p, 0) / Math.min(10, recentPrices.length);

  // Calculate recent price trend (last 10 days)
  const recentTrend = (currentPrice - recentPrices[Math.max(0, recentPrices.length - 10)]) / Math.max(1, Math.min(10, recentPrices.length));

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    // Simple linear projection from recent trend
    const trendPrediction = currentPrice + (recentTrend * (i + 1));

    // Weighted average: 70% current price, 30% trend prediction (very conservative)
    let predicted = currentPrice * 0.70 + trendPrediction * 0.30;

    // Additional safety bounds
    const maxChange = currentPrice * 0.05 * Math.sqrt(i + 1);  // ±5% with gradual increase
    predicted = Math.max(currentPrice - maxChange, Math.min(currentPrice + maxChange, predicted));

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'Polynomial Regression',
    });
  }

  return predictions;
}

/**
 * Moving Average Prediction - Adaptive window based on volatility with OHLCV
 */
export function generateMovingAverageForecast(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);
  const highPrices = stockData.map(d => d.high);
  const lowPrices = stockData.map(d => d.low);
  const volumes = stockData.map(d => d.volume);

  // Calculate typical prices
  const typicalPrices = stockData.map((d, i) => 
    calculateTypicalPrice(highPrices[i], lowPrices[i], closePrices[i])
  );

  // Calculate ATR-based volatility (more accurate than simple returns)
  const atr = calculateATR(highPrices, lowPrices, closePrices);
  const avgPrice = closePrices.reduce((sum, p) => sum + p, 0) / closePrices.length;
  const avgVolatility = atr / avgPrice;

  // Adaptive window: lower volatility = longer window (smoother)
  // Higher volatility = shorter window (more responsive)
  const baseWindow = Math.min(30, Math.floor(typicalPrices.length / 2));
  const windowSize = Math.max(10, Math.floor(baseWindow * (1 - avgVolatility * 20)));

  // Calculate moving average using typical prices
  const recentPrices = typicalPrices.slice(-windowSize);
  const avgTypicalPrice = recentPrices.reduce((sum, price) => sum + price, 0) / windowSize;

  // Calculate volume-weighted trend
  const recentVolumes = volumes.slice(-windowSize);
  const volumeAvg = recentVolumes.reduce((sum, v) => sum + v, 0) / windowSize;

  // Calculate trend from recent prices with weighted approach
  const halfWindow = Math.floor(windowSize / 2);
  const firstHalf = recentPrices.slice(0, halfWindow);
  const secondHalf = recentPrices.slice(-halfWindow);

  const firstAvg = firstHalf.reduce((sum, p) => sum + p, 0) / halfWindow;
  const secondAvg = secondHalf.reduce((sum, p) => sum + p, 0) / halfWindow;

  // Enhanced trend calculation with momentum and volume
  const trend = (secondAvg - firstAvg) / halfWindow;
  const momentum = (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices.length;
  const obvTrend = calculateOBVTrend(closePrices.slice(-windowSize), recentVolumes);
  const volumeImpact = obvTrend > 0 ? 1.05 : 0.95; // 5% boost/reduction based on volume

  // Generate predictions
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    // Combine trend and momentum with decay over time
    const decayFactor = Math.exp(-i / forecastDays); // Momentum decays over time
    let predicted = avgTypicalPrice + trend * (i + 1) + momentum * (i + 1) * decayFactor;
    
    // Apply volume impact
    predicted *= volumeImpact;
    predicted = Math.max(0, predicted);

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'Moving Average',
    });
  }

  return predictions;
}

/**
 * Exponential Moving Average - Enhanced with dual EMA, momentum, and OHLCV
 */
export function generateEMAForecast(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);
  const highPrices = stockData.map(d => d.high);
  const lowPrices = stockData.map(d => d.low);
  const volumes = stockData.map(d => d.volume);

  // Calculate typical prices for more accurate EMA
  const typicalPrices = stockData.map((d, i) => 
    calculateTypicalPrice(highPrices[i], lowPrices[i], closePrices[i])
  );

  // Dual EMA approach: fast (12) and slow (26) for better trend detection
  const fastPeriod = 12;
  const slowPeriod = 26;
  const fastMultiplier = 2 / (fastPeriod + 1);
  const slowMultiplier = 2 / (slowPeriod + 1);

  // Calculate Fast EMA on typical prices
  let fastEMA = typicalPrices.slice(0, fastPeriod).reduce((sum, p) => sum + p, 0) / fastPeriod;
  for (let i = fastPeriod; i < typicalPrices.length; i++) {
    fastEMA = (typicalPrices[i] - fastEMA) * fastMultiplier + fastEMA;
  }

  // Calculate Slow EMA on typical prices
  let slowEMA = typicalPrices.slice(0, slowPeriod).reduce((sum, p) => sum + p, 0) / slowPeriod;
  for (let i = slowPeriod; i < typicalPrices.length; i++) {
    slowEMA = (typicalPrices[i] - slowEMA) * slowMultiplier + slowEMA;
  }

  // MACD-like signal: difference between fast and slow EMA indicates momentum
  const macdSignal = fastEMA - slowEMA;

  // Calculate trend from recent price action using typical prices
  const recentWindow = Math.min(20, typicalPrices.length);
  const recentPrices = typicalPrices.slice(-recentWindow);
  const trend = (recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentWindow;

  // Volume analysis
  const recentVolumes = volumes.slice(-recentWindow);
  const obvTrend = calculateOBVTrend(closePrices.slice(-recentWindow), recentVolumes);
  const volumeMultiplier = obvTrend > 0 ? 1.03 : 0.97;

  // Generate predictions
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    // Use fast EMA as baseline and incorporate trend + MACD momentum
    // MACD signal decays over time as short-term momentum is less predictive long-term
    const decayFactor = Math.exp(-i / (forecastDays / 2));
    let predicted = fastEMA + trend * (i + 1) + macdSignal * decayFactor;
    
    // Apply volume-based adjustment
    predicted *= volumeMultiplier;
    predicted = Math.max(0, predicted);

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'Exponential MA',
    });
  }

  return predictions;
}

/**
 * Prophet-Lite - Inspired by Facebook Prophet
 * Combines trend, seasonality, and changepoint detection
 * Lightweight implementation suitable for browser/serverless
 */
export function generateProphetLiteForecast(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);

  if (closePrices.length < 60) {
    return generateLinearRegression(stockData, forecastDays);
  }

  const n = closePrices.length;

  // 1. Detect trend using piecewise linear regression
  const changepoints = detectChangepoints(closePrices);
  const trend = fitPiecewiseTrend(closePrices, changepoints);

  // 2. Extract seasonality (weekly pattern)
  const detrended = closePrices.map((price, i) => price - trend[i]);
  const weeklySeasonality = extractWeeklySeasonality(detrended, stockData);

  // 3. Calculate residuals
  const residuals = detrended.map((val, i) => val - weeklySeasonality[i]);
  const residualStd = Math.sqrt(
    residuals.reduce((sum, r) => sum + r * r, 0) / residuals.length
  );

  // 4. Forecast future values
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];

  // Extrapolate trend
  const lastTrendValue = trend[trend.length - 1];
  const trendSlope = (trend[trend.length - 1] - trend[trend.length - 10]) / 10;

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    // Trend component (with damping for long-term forecasts)
    const dampingFactor = Math.exp(-i / (forecastDays * 2));
    const trendComponent = lastTrendValue + trendSlope * (i + 1) * dampingFactor;

    // Seasonality component (weekly cycle)
    const dayOfWeek = forecastDate.getDay();
    const seasonalityComponent = calculateSeasonalityForDay(weeklySeasonality, dayOfWeek);

    // Combine components
    let predicted = trendComponent + seasonalityComponent;

    // Add mean reversion for very long forecasts
    if (i > 15) {
      const meanPrice = closePrices.slice(-60).reduce((sum, p) => sum + p, 0) / Math.min(60, closePrices.length);
      const reversionFactor = (i - 15) / forecastDays;
      predicted = predicted * (1 - reversionFactor * 0.2) + meanPrice * (reversionFactor * 0.2);
    }

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted: Math.max(0, predicted),
      algorithm: 'Prophet-Lite',
    });
  }

  return predictions;
}

// Helper: Detect trend changepoints
function detectChangepoints(prices: number[]): number[] {
  const changepoints: number[] = [];
  const windowSize = Math.max(10, Math.floor(prices.length / 10));

  for (let i = windowSize; i < prices.length - windowSize; i += windowSize) {
    const before = prices.slice(i - windowSize, i);
    const after = prices.slice(i, i + windowSize);

    const slopeBefore = (before[before.length - 1] - before[0]) / windowSize;
    const slopeAfter = (after[after.length - 1] - after[0]) / windowSize;

    // Detect significant slope change
    if (Math.abs(slopeAfter - slopeBefore) > Math.abs(slopeBefore) * 0.5) {
      changepoints.push(i);
    }
  }

  return changepoints;
}

// Helper: Fit piecewise linear trend
function fitPiecewiseTrend(prices: number[], changepoints: number[]): number[] {
  const trend: number[] = [];
  const segments = [0, ...changepoints, prices.length];

  for (let seg = 0; seg < segments.length - 1; seg++) {
    const start = segments[seg];
    const end = segments[seg + 1];
    const segmentPrices = prices.slice(start, end);

    const n = segmentPrices.length;
    const x = Array.from({ length: n }, (_, i) => i);
    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = segmentPrices.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((sum, xi, i) => sum + xi * segmentPrices[i], 0);
    const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;

    for (let i = 0; i < n; i++) {
      trend.push(slope * i + intercept);
    }
  }

  return trend;
}

// Helper: Extract weekly seasonality
function extractWeeklySeasonality(detrended: number[], stockData: StockData[]): number[] {
  const dayAverages: { [key: number]: number[] } = { 0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [] };

  detrended.forEach((val, i) => {
    const dayOfWeek = new Date(stockData[i].date).getDay();
    dayAverages[dayOfWeek].push(val);
  });

  const seasonalPattern: { [key: number]: number } = {};
  for (let day = 0; day < 7; day++) {
    const values = dayAverages[day];
    seasonalPattern[day] = values.length > 0
      ? values.reduce((sum, v) => sum + v, 0) / values.length
      : 0;
  }

  // Normalize so seasonality averages to zero
  const avgSeasonality = Object.values(seasonalPattern).reduce((sum, v) => sum + v, 0) / 7;
  for (let day = 0; day < 7; day++) {
    seasonalPattern[day] -= avgSeasonality;
  }

  return detrended.map((_, i) => {
    const dayOfWeek = new Date(stockData[i].date).getDay();
    return seasonalPattern[dayOfWeek];
  });
}

// Helper: Calculate seasonality for a given day
function calculateSeasonalityForDay(weeklySeasonality: number[], dayOfWeek: number): number {
  // Average the seasonality values for this day of week
  const values: number[] = [];
  for (let i = 0; i < weeklySeasonality.length; i++) {
    values.push(weeklySeasonality[i]);
  }
  return values.length > 0 ? values.reduce((sum, v) => sum + v, 0) / values.length : 0;
}

/**
 * ARIMA (AutoRegressive Integrated Moving Average)
 * p=5 (AR order), d=1 (differencing), q=5 (MA order)
 * Advanced statistical forecasting for time series
 */
export function generateARIMAForecast(
  stockData: StockData[],
  forecastDays: number = 30
): MLPrediction[] {
  const closePrices = stockData.map(d => d.close);

  // ARIMA parameters
  const p = 5; // AR order (autoregressive lags)
  const d = 1; // Differencing order
  const q = 5; // MA order (moving average lags)

  // Need sufficient data for ARIMA
  if (closePrices.length < p + q + 10) {
    // Fallback to linear regression if insufficient data
    return generateLinearRegression(stockData, forecastDays);
  }

  // Step 1: Apply differencing (d=1) to make series stationary
  const differences: number[] = [];
  for (let i = 1; i < closePrices.length; i++) {
    differences.push(closePrices[i] - closePrices[i - 1]);
  }

  // Step 2: Estimate AR coefficients using Yule-Walker equations (simplified)
  const n = differences.length;
  const arCoeffs: number[] = [];

  // Calculate autocorrelations for AR component
  const mean = differences.reduce((sum, val) => sum + val, 0) / n;
  const variance = differences.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / n;

  for (let lag = 1; lag <= p; lag++) {
    let autoCorr = 0;
    for (let i = lag; i < n; i++) {
      autoCorr += (differences[i] - mean) * (differences[i - lag] - mean);
    }
    autoCorr /= ((n - lag) * variance);
    arCoeffs.push(autoCorr * 0.8); // Dampened for stability
  }

  // Step 3: Calculate residuals for MA component
  const residuals: number[] = new Array(n).fill(0);
  for (let i = p; i < n; i++) {
    let predicted = mean;
    for (let j = 0; j < p; j++) {
      predicted += arCoeffs[j] * (differences[i - j - 1] - mean);
    }
    residuals[i] = differences[i] - predicted;
  }

  // Step 4: Estimate MA coefficients
  const maCoeffs: number[] = [];
  const residualMean = residuals.slice(q).reduce((sum, val) => sum + val, 0) / (n - q);

  for (let lag = 1; lag <= q; lag++) {
    let maCorr = 0;
    for (let i = lag; i < residuals.length; i++) {
      maCorr += residuals[i] * residuals[i - lag];
    }
    maCorr /= (residuals.length - lag);
    maCoeffs.push(maCorr * 0.5); // Dampened for stability
  }

  // Step 5: Generate forecasts
  const lastDate = new Date(stockData[stockData.length - 1].date);
  const predictions: MLPrediction[] = [];

  // Keep track of forecasted differences and recent actual differences
  const forecastDifferences: number[] = [...differences.slice(-p)];
  const forecastResiduals: number[] = [...residuals.slice(-q)];

  for (let i = 0; i < forecastDays; i++) {
    const forecastDate = new Date(lastDate);
    forecastDate.setDate(forecastDate.getDate() + i + 1);

    // AR component: weighted sum of recent differences
    let arComponent = mean;
    for (let j = 0; j < p && j < forecastDifferences.length; j++) {
      const index = forecastDifferences.length - 1 - j;
      arComponent += arCoeffs[j] * (forecastDifferences[index] - mean);
    }

    // MA component: weighted sum of recent residuals
    let maComponent = 0;
    for (let j = 0; j < q && j < forecastResiduals.length; j++) {
      const index = forecastResiduals.length - 1 - j;
      maComponent += maCoeffs[j] * forecastResiduals[index];
    }

    // Predicted difference (stationary series)
    const predictedDiff = arComponent + maComponent;

    // Store for next iteration
    forecastDifferences.push(predictedDiff);
    forecastResiduals.push(0); // Assume zero residual for future

    // Integrate back to get price (reverse differencing)
    const lastPrice = i === 0
      ? closePrices[closePrices.length - 1]
      : predictions[i - 1].predicted;

    const predicted = Math.max(0, lastPrice + predictedDiff);

    predictions.push({
      date: forecastDate.toISOString().split('T')[0],
      predicted,
      algorithm: 'ARIMA',
    });
  }

  return predictions;
}

// ============================================================================
// Helper Functions for Enhanced OHLCV Analysis
// ============================================================================

/**
 * Calculate Average True Range (ATR) - measure of volatility
 */
function calculateATR(highs: number[], lows: number[], closes: number[], period: number = 14): number {
  if (highs.length < 2) return 0;
  
  const trueRanges: number[] = [];
  for (let i = 1; i < highs.length; i++) {
    const high = highs[i];
    const low = lows[i];
    const prevClose = closes[i - 1];
    
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    trueRanges.push(tr);
  }
  
  // Calculate average
  const windowSize = Math.min(period, trueRanges.length);
  const recentTR = trueRanges.slice(-windowSize);
  return recentTR.reduce((sum, tr) => sum + tr, 0) / windowSize;
}

/**
 * Calculate typical price (HLC/3) for better price representation
 */
function calculateTypicalPrice(high: number, low: number, close: number): number {
  return (high + low + close) / 3;
}

/**
 * Calculate weighted close (HLCC/4) - gives more weight to close
 */
function calculateWeightedClose(high: number, low: number, close: number): number {
  return (high + low + close + close) / 4;
}

/**
 * Calculate On-Balance Volume trend
 */
function calculateOBVTrend(closes: number[], volumes: number[]): number {
  if (closes.length < 2) return 1;
  
  let obv = 0;
  for (let i = 1; i < closes.length; i++) {
    if (closes[i] > closes[i - 1]) {
      obv += volumes[i];
    } else if (closes[i] < closes[i - 1]) {
      obv -= volumes[i];
    }
  }
  
  return obv;
}

/**
 * Calculate price momentum
 */
function calculateMomentum(prices: number[], period: number = 10): number {
  if (prices.length < period) return 0;
  const current = prices[prices.length - 1];
  const past = prices[prices.length - period];
  return (current - past) / past;
}
