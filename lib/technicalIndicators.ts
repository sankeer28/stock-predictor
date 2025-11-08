import { StockData, TechnicalIndicators } from '@/types';

/**
 * Calculate Simple Moving Average
 */
export function calculateSMA(data: number[], period: number): number[] {
  const result: number[] = [];

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(NaN);
    } else {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push(sum / period);
    }
  }

  return result;
}

/**
 * Calculate Exponential Moving Average
 */
export function calculateEMA(data: number[], period: number): number[] {
  const result: number[] = [];
  const multiplier = 2 / (period + 1);

  // Start with SMA for first value
  let ema = data.slice(0, period).reduce((a, b) => a + b, 0) / period;

  for (let i = 0; i < data.length; i++) {
    if (i < period - 1) {
      result.push(NaN);
    } else if (i === period - 1) {
      result.push(ema);
    } else {
      ema = (data[i] - ema) * multiplier + ema;
      result.push(ema);
    }
  }

  return result;
}

/**
 * Calculate RSI (Relative Strength Index)
 */
export function calculateRSI(prices: number[], period: number = 14): number[] {
  const result: number[] = [];
  const changes: number[] = [];

  // Calculate price changes
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i - 1]);
  }

  for (let i = 0; i < prices.length; i++) {
    if (i < period) {
      result.push(NaN);
    } else {
      const recentChanges = changes.slice(i - period, i);
      const gains = recentChanges.filter(c => c > 0);
      const losses = recentChanges.filter(c => c < 0).map(c => Math.abs(c));

      const avgGain = gains.length > 0 ? gains.reduce((a, b) => a + b, 0) / period : 0;
      const avgLoss = losses.length > 0 ? losses.reduce((a, b) => a + b, 0) / period : 0;

      if (avgLoss === 0) {
        result.push(100);
      } else {
        const rs = avgGain / avgLoss;
        const rsi = 100 - (100 / (1 + rs));
        result.push(rsi);
      }
    }
  }

  return result;
}

/**
 * Calculate MACD (Moving Average Convergence Divergence)
 */
export function calculateMACD(prices: number[], fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
  const emaFast = calculateEMA(prices, fastPeriod);
  const emaSlow = calculateEMA(prices, slowPeriod);

  const macd: number[] = [];
  for (let i = 0; i < prices.length; i++) {
    if (isNaN(emaFast[i]) || isNaN(emaSlow[i])) {
      macd.push(NaN);
    } else {
      macd.push(emaFast[i] - emaSlow[i]);
    }
  }

  const signal = calculateEMA(macd.filter(v => !isNaN(v)), signalPeriod);
  const macdSignal: number[] = [];

  let signalIndex = 0;
  for (let i = 0; i < macd.length; i++) {
    if (isNaN(macd[i])) {
      macdSignal.push(NaN);
    } else {
      macdSignal.push(signal[signalIndex] || NaN);
      signalIndex++;
    }
  }

  const histogram: number[] = [];
  for (let i = 0; i < macd.length; i++) {
    if (isNaN(macd[i]) || isNaN(macdSignal[i])) {
      histogram.push(NaN);
    } else {
      histogram.push(macd[i] - macdSignal[i]);
    }
  }

  return { macd, signal: macdSignal, histogram };
}

/**
 * Calculate Bollinger Bands
 */
export function calculateBollingerBands(prices: number[], period = 20, stdDev = 2) {
  const sma = calculateSMA(prices, period);
  const upper: number[] = [];
  const lower: number[] = [];

  for (let i = 0; i < prices.length; i++) {
    if (i < period - 1) {
      upper.push(NaN);
      lower.push(NaN);
    } else {
      const slice = prices.slice(i - period + 1, i + 1);
      const mean = sma[i];
      const variance = slice.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
      const std = Math.sqrt(variance);

      upper.push(mean + (std * stdDev));
      lower.push(mean - (std * stdDev));
    }
  }

  return { upper, middle: sma, lower };
}

/**
 * Calculate all technical indicators
 */
export function calculateAllIndicators(stockData: StockData[]): TechnicalIndicators {
  const closePrices = stockData.map(d => d.close);
  const volumes = stockData.map(d => d.volume);

  const ma20 = calculateSMA(closePrices, 20);
  const ma50 = calculateSMA(closePrices, 50);
  const ma200 = calculateSMA(closePrices, 200);

  const rsi = calculateRSI(closePrices, 14);

  const { macd, signal: macdSignal, histogram: macdHistogram } = calculateMACD(closePrices);

  const { upper: bbUpper, middle: bbMiddle, lower: bbLower } = calculateBollingerBands(closePrices);

  const volumeMA = calculateSMA(volumes, 20);

  const ema12 = calculateEMA(closePrices, 12);
  const ema26 = calculateEMA(closePrices, 26);

  return {
    ma20,
    ma50,
    ma200,
    rsi,
    macd,
    macdSignal,
    macdHistogram,
    bbUpper,
    bbMiddle,
    bbLower,
    volumeMA,
    ema12,
    ema26,
  };
}
