import { StockData, TechnicalIndicators, TradingSignal } from '@/types';

/**
 * Generate trading signal based on technical analysis
 */
export function generateTradingSignal(
  stockData: StockData[],
  indicators: TechnicalIndicators,
  forecastChange?: number
): TradingSignal {
  const reasons: string[] = [];
  let bullishSignals = 0;
  let bearishSignals = 0;
  let totalSignals = 0;

  const currentIndex = stockData.length - 1;
  const currentPrice = stockData[currentIndex].close;

  // 1. Moving Average Analysis
  const ma20 = indicators.ma20[currentIndex];
  const ma50 = indicators.ma50[currentIndex];
  const ma200 = indicators.ma200[currentIndex];

  if (!isNaN(ma20) && !isNaN(ma50)) {
    totalSignals++;
    if (ma20 > ma50) {
      bullishSignals++;
      reasons.push('20-day MA above 50-day MA (bullish)');
    } else {
      bearishSignals++;
      reasons.push('20-day MA below 50-day MA (bearish)');
    }
  }

  if (!isNaN(ma50) && !isNaN(ma200)) {
    totalSignals++;
    if (ma50 > ma200) {
      bullishSignals++;
      reasons.push('50-day MA above 200-day MA (golden cross territory)');
    } else {
      bearishSignals++;
      reasons.push('50-day MA below 200-day MA (death cross territory)');
    }
  }

  if (!isNaN(ma20)) {
    totalSignals++;
    if (currentPrice > ma20) {
      bullishSignals++;
      reasons.push('Price above 20-day MA');
    } else {
      bearishSignals++;
      reasons.push('Price below 20-day MA');
    }
  }

  // 2. RSI Analysis
  const rsi = indicators.rsi[currentIndex];
  if (!isNaN(rsi)) {
    totalSignals++;
    if (rsi < 30) {
      bullishSignals++;
      reasons.push(`RSI oversold (${rsi.toFixed(1)})`);
    } else if (rsi > 70) {
      bearishSignals++;
      reasons.push(`RSI overbought (${rsi.toFixed(1)})`);
    } else if (rsi > 50) {
      bullishSignals++;
      reasons.push(`RSI shows momentum (${rsi.toFixed(1)})`);
    } else {
      bearishSignals++;
      reasons.push(`RSI shows weakness (${rsi.toFixed(1)})`);
    }
  }

  // 3. MACD Analysis
  const macd = indicators.macd[currentIndex];
  const macdSignal = indicators.macdSignal[currentIndex];

  if (!isNaN(macd) && !isNaN(macdSignal)) {
    totalSignals++;
    if (macd > macdSignal) {
      bullishSignals++;
      reasons.push('MACD above signal line (bullish)');
    } else {
      bearishSignals++;
      reasons.push('MACD below signal line (bearish)');
    }

    // Check for MACD crossover in last 5 days
    if (currentIndex >= 5) {
      const prevMacd = indicators.macd[currentIndex - 5];
      const prevSignal = indicators.macdSignal[currentIndex - 5];

      if (!isNaN(prevMacd) && !isNaN(prevSignal)) {
        if (prevMacd < prevSignal && macd > macdSignal) {
          bullishSignals++;
          reasons.push('Recent MACD bullish crossover');
        } else if (prevMacd > prevSignal && macd < macdSignal) {
          bearishSignals++;
          reasons.push('Recent MACD bearish crossover');
        }
      }
    }
  }

  // 4. Bollinger Bands Analysis
  const bbUpper = indicators.bbUpper[currentIndex];
  const bbLower = indicators.bbLower[currentIndex];

  if (!isNaN(bbUpper) && !isNaN(bbLower)) {
    totalSignals++;
    const bbPosition = (currentPrice - bbLower) / (bbUpper - bbLower);

    if (bbPosition < 0.2) {
      bullishSignals++;
      reasons.push('Price near lower Bollinger Band (potential bounce)');
    } else if (bbPosition > 0.8) {
      bearishSignals++;
      reasons.push('Price near upper Bollinger Band (potential pullback)');
    }
  }

  // 5. Volume Analysis
  const currentVolume = stockData[currentIndex].volume;
  const volumeMA = indicators.volumeMA[currentIndex];

  if (!isNaN(volumeMA)) {
    totalSignals++;
    if (currentVolume > volumeMA * 1.5) {
      const priceChange = stockData[currentIndex].close - stockData[currentIndex - 1].close;
      if (priceChange > 0) {
        bullishSignals++;
        reasons.push('High volume on up day (strong buying)');
      } else {
        bearishSignals++;
        reasons.push('High volume on down day (strong selling)');
      }
    }
  }

  // 6. Price Momentum (last 5 days)
  if (currentIndex >= 5) {
    const priceChange5d = ((currentPrice - stockData[currentIndex - 5].close) / stockData[currentIndex - 5].close) * 100;
    totalSignals++;

    if (priceChange5d > 3) {
      bullishSignals++;
      reasons.push(`Strong 5-day momentum (+${priceChange5d.toFixed(1)}%)`);
    } else if (priceChange5d < -3) {
      bearishSignals++;
      reasons.push(`Weak 5-day momentum (${priceChange5d.toFixed(1)}%)`);
    }
  }

  // 7. Forecast (if provided)
  if (forecastChange !== undefined) {
    totalSignals++;
    if (forecastChange > 5) {
      bullishSignals++;
      reasons.push(`Forecast shows strong upside (+${forecastChange.toFixed(1)}%)`);
    } else if (forecastChange < -5) {
      bearishSignals++;
      reasons.push(`Forecast shows downside (${forecastChange.toFixed(1)}%)`);
    }
  }

  // Calculate signal strength
  const bullishPercentage = (bullishSignals / totalSignals) * 100;
  const bearishPercentage = (bearishSignals / totalSignals) * 100;
  const confidence = Math.abs(bullishPercentage - bearishPercentage);

  // Determine signal type
  let signalType: TradingSignal['type'];

  if (bullishPercentage >= 75) {
    signalType = 'strong_buy';
  } else if (bullishPercentage >= 60) {
    signalType = 'buy';
  } else if (bullishPercentage >= 52) {
    signalType = 'weak_buy';
  } else if (bearishPercentage >= 75) {
    signalType = 'strong_sell';
  } else if (bearishPercentage >= 60) {
    signalType = 'sell';
  } else if (bearishPercentage >= 52) {
    signalType = 'weak_sell';
  } else {
    signalType = 'hold';
  }

  // Calculate price target based on support/resistance levels
  let priceTarget: number | undefined;
  if (!isNaN(ma50)) {
    if (signalType.includes('buy')) {
      priceTarget = Math.max(ma50 * 1.05, currentPrice * 1.03);
    } else if (signalType.includes('sell')) {
      priceTarget = Math.min(ma50 * 0.95, currentPrice * 0.97);
    }
  }

  return {
    type: signalType,
    confidence,
    reasons,
    priceTarget,
    priceChange: priceTarget ? ((priceTarget - currentPrice) / currentPrice) * 100 : undefined,
  };
}
