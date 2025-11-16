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
 * Calculate ATR (Average True Range) - Volatility indicator
 */
export function calculateATR(stockData: StockData[], period: number = 14): number[] {
  const result: number[] = [];
  const trueRanges: number[] = [];

  for (let i = 0; i < stockData.length; i++) {
    if (i === 0) {
      result.push(NaN);
      continue;
    }

    const high = stockData[i].high;
    const low = stockData[i].low;
    const prevClose = stockData[i - 1].close;

    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    trueRanges.push(tr);

    if (i < period) {
      result.push(NaN);
    } else {
      const recentTR = trueRanges.slice(-period);
      const atr = recentTR.reduce((sum, tr) => sum + tr, 0) / period;
      result.push(atr);
    }
  }

  return result;
}

/**
 * Calculate Stochastic Oscillator - Momentum indicator
 */
export function calculateStochastic(
  stockData: StockData[],
  period: number = 14,
  smoothK: number = 3,
  smoothD: number = 3
): { k: number[], d: number[] } {
  const k: number[] = [];
  const d: number[] = [];

  for (let i = 0; i < stockData.length; i++) {
    if (i < period - 1) {
      k.push(NaN);
      continue;
    }

    const slice = stockData.slice(i - period + 1, i + 1);
    const high = Math.max(...slice.map(d => d.high));
    const low = Math.min(...slice.map(d => d.low));
    const close = stockData[i].close;

    const stoch = ((close - low) / (high - low || 1)) * 100;
    k.push(stoch);
  }

  // Smooth %K
  const smoothedK: number[] = calculateSMA(k.filter(v => !isNaN(v)), smoothK);
  
  // Calculate %D (SMA of %K)
  const smoothedD: number[] = calculateSMA(smoothedK, smoothD);

  // Pad with NaN to match original length
  const padK = new Array(stockData.length - smoothedK.length).fill(NaN);
  const padD = new Array(stockData.length - smoothedD.length).fill(NaN);

  return {
    k: [...padK, ...smoothedK],
    d: [...padD, ...smoothedD],
  };
}

/**
 * Calculate ADX (Average Directional Index) - Trend strength indicator
 */
export function calculateADX(stockData: StockData[], period: number = 14): {
  adx: number[];
  plusDI: number[];
  minusDI: number[];
} {
  const adx: number[] = [];
  const plusDI: number[] = [];
  const minusDI: number[] = [];
  const plusDM: number[] = [];
  const minusDM: number[] = [];
  const tr: number[] = [];

  for (let i = 1; i < stockData.length; i++) {
    const high = stockData[i].high;
    const low = stockData[i].low;
    const prevHigh = stockData[i - 1].high;
    const prevLow = stockData[i - 1].low;
    const prevClose = stockData[i - 1].close;

    // Calculate directional movement
    const upMove = high - prevHigh;
    const downMove = prevLow - low;

    plusDM.push(upMove > downMove && upMove > 0 ? upMove : 0);
    minusDM.push(downMove > upMove && downMove > 0 ? downMove : 0);

    // Calculate true range
    const trueRange = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    tr.push(trueRange);
  }

  // Calculate smoothed averages
  for (let i = 0; i < stockData.length; i++) {
    if (i < period) {
      plusDI.push(NaN);
      minusDI.push(NaN);
      adx.push(NaN);
      continue;
    }

    const smoothPlusDM = plusDM.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
    const smoothMinusDM = minusDM.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
    const smoothTR = tr.slice(i - period, i).reduce((a, b) => a + b, 0) / period;

    const plusDIVal = (smoothPlusDM / (smoothTR || 1)) * 100;
    const minusDIVal = (smoothMinusDM / (smoothTR || 1)) * 100;

    plusDI.push(plusDIVal);
    minusDI.push(minusDIVal);

    // Calculate DX and ADX
    const dx = (Math.abs(plusDIVal - minusDIVal) / (plusDIVal + minusDIVal || 1)) * 100;
    
    if (i < period * 2) {
      adx.push(NaN);
    } else {
      const recentDX = [];
      for (let j = i - period; j < i; j++) {
        if (!isNaN(plusDI[j]) && !isNaN(minusDI[j])) {
          const dxVal = (Math.abs(plusDI[j] - minusDI[j]) / (plusDI[j] + minusDI[j] || 1)) * 100;
          recentDX.push(dxVal);
        }
      }
      const avgDX = recentDX.reduce((a, b) => a + b, 0) / recentDX.length;
      adx.push(avgDX);
    }
  }

  return { adx, plusDI, minusDI };
}

/**
 * Calculate OBV (On-Balance Volume) - Volume-based momentum indicator
 */
export function calculateOBV(stockData: StockData[]): number[] {
  const obv: number[] = [];
  let runningOBV = 0;

  for (let i = 0; i < stockData.length; i++) {
    if (i === 0) {
      obv.push(stockData[i].volume);
      runningOBV = stockData[i].volume;
    } else {
      if (stockData[i].close > stockData[i - 1].close) {
        runningOBV += stockData[i].volume;
      } else if (stockData[i].close < stockData[i - 1].close) {
        runningOBV -= stockData[i].volume;
      }
      obv.push(runningOBV);
    }
  }

  return obv;
}

/**
 * Calculate Williams %R - Momentum oscillator
 */
export function calculateWilliamsR(stockData: StockData[], period: number = 14): number[] {
  const result: number[] = [];

  for (let i = 0; i < stockData.length; i++) {
    if (i < period - 1) {
      result.push(NaN);
      continue;
    }

    const slice = stockData.slice(i - period + 1, i + 1);
    const high = Math.max(...slice.map(d => d.high));
    const low = Math.min(...slice.map(d => d.low));
    const close = stockData[i].close;

    const williamsR = ((high - close) / (high - low || 1)) * -100;
    result.push(williamsR);
  }

  return result;
}

/**
 * Calculate CCI (Commodity Channel Index) - Momentum indicator
 */
export function calculateCCI(stockData: StockData[], period: number = 20): number[] {
  const result: number[] = [];
  const typicalPrices: number[] = stockData.map(d => (d.high + d.low + d.close) / 3);

  for (let i = 0; i < stockData.length; i++) {
    if (i < period - 1) {
      result.push(NaN);
      continue;
    }

    const slice = typicalPrices.slice(i - period + 1, i + 1);
    const sma = slice.reduce((sum, val) => sum + val, 0) / period;
    
    // Calculate mean deviation
    const meanDeviation = slice.reduce((sum, val) => sum + Math.abs(val - sma), 0) / period;
    
    const cci = (typicalPrices[i] - sma) / (0.015 * meanDeviation);
    result.push(cci);
  }

  return result;
}

/**
 * Calculate Ichimoku Cloud components
 */
export function calculateIchimoku(stockData: StockData[]): {
  tenkanSen: number[];
  kijunSen: number[];
  senkouSpanA: number[];
  senkouSpanB: number[];
  chikouSpan: number[];
} {
  const tenkanPeriod = 9;
  const kijunPeriod = 26;
  const senkouBPeriod = 52;
  const displacement = 26;

  const tenkanSen: number[] = [];
  const kijunSen: number[] = [];
  const senkouSpanA: number[] = [];
  const senkouSpanB: number[] = [];
  const chikouSpan: number[] = [];

  // Helper function to calculate midpoint
  const getMidpoint = (slice: StockData[]) => {
    const high = Math.max(...slice.map(d => d.high));
    const low = Math.min(...slice.map(d => d.low));
    return (high + low) / 2;
  };

  for (let i = 0; i < stockData.length; i++) {
    // Tenkan-sen (Conversion Line)
    if (i >= tenkanPeriod - 1) {
      const slice = stockData.slice(i - tenkanPeriod + 1, i + 1);
      tenkanSen.push(getMidpoint(slice));
    } else {
      tenkanSen.push(NaN);
    }

    // Kijun-sen (Base Line)
    if (i >= kijunPeriod - 1) {
      const slice = stockData.slice(i - kijunPeriod + 1, i + 1);
      kijunSen.push(getMidpoint(slice));
    } else {
      kijunSen.push(NaN);
    }

    // Senkou Span B (Leading Span B)
    if (i >= senkouBPeriod - 1) {
      const slice = stockData.slice(i - senkouBPeriod + 1, i + 1);
      senkouSpanB.push(getMidpoint(slice));
    } else {
      senkouSpanB.push(NaN);
    }

    // Chikou Span (Lagging Span) - close price shifted back
    chikouSpan.push(stockData[i].close);
  }

  // Senkou Span A (Leading Span A) - average of Tenkan and Kijun
  for (let i = 0; i < stockData.length; i++) {
    if (!isNaN(tenkanSen[i]) && !isNaN(kijunSen[i])) {
      senkouSpanA.push((tenkanSen[i] + kijunSen[i]) / 2);
    } else {
      senkouSpanA.push(NaN);
    }
  }

  return {
    tenkanSen,
    kijunSen,
    senkouSpanA,
    senkouSpanB,
    chikouSpan,
  };
}

/**
 * Calculate MFI (Money Flow Index) - Volume-weighted RSI
 */
export function calculateMFI(stockData: StockData[], period: number = 14): number[] {
  const result: number[] = [];
  const typicalPrices: number[] = stockData.map(d => (d.high + d.low + d.close) / 3);
  const moneyFlows: number[] = typicalPrices.map((tp, i) => tp * stockData[i].volume);

  for (let i = 0; i < stockData.length; i++) {
    if (i < period) {
      result.push(NaN);
      continue;
    }

    let positiveFlow = 0;
    let negativeFlow = 0;

    for (let j = i - period + 1; j <= i; j++) {
      if (typicalPrices[j] > typicalPrices[j - 1]) {
        positiveFlow += moneyFlows[j];
      } else if (typicalPrices[j] < typicalPrices[j - 1]) {
        negativeFlow += moneyFlows[j];
      }
    }

    const moneyRatio = positiveFlow / (negativeFlow || 1);
    const mfi = 100 - (100 / (1 + moneyRatio));
    result.push(mfi);
  }

  return result;
}

/**
 * Calculate all technical indicators (enhanced with new indicators)
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
