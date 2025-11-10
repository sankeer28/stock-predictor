// ML Predictions Cache Utility
// Stores ML predictions per stock with timestamp to avoid recalculation

export interface CachedPrediction {
  symbol: string;
  timestamp: number;
  date: string; // Human readable date
  predictions: {
    lstm?: any[];
    arima?: any[];
    gru?: any[];
    tft?: any[];
    cnn?: any[];
    cnnLstm?: any[];
    linearRegression?: any[];
    polynomialRegression?: any[];
    movingAverage?: any[];
    ema?: any[];
  };
  forecastHorizon: number;
}

const CACHE_KEY = 'mlPredictionsCache';
const CACHE_EXPIRY_HOURS = 24; // Predictions valid for 24 hours

/**
 * Get all cached predictions from localStorage
 */
export function getAllCachedPredictions(): CachedPrediction[] {
  try {
    const cached = localStorage.getItem(CACHE_KEY);
    if (!cached) return [];

    const predictions: CachedPrediction[] = JSON.parse(cached);

    // Filter out expired predictions (older than 24 hours)
    const now = Date.now();
    const valid = predictions.filter(p => {
      const age = now - p.timestamp;
      const ageInHours = age / (1000 * 60 * 60);
      return ageInHours < CACHE_EXPIRY_HOURS;
    });

    // Save filtered list back if we removed any
    if (valid.length !== predictions.length) {
      localStorage.setItem(CACHE_KEY, JSON.stringify(valid));
    }

    return valid;
  } catch (e) {
    console.error('Error reading predictions cache:', e);
    return [];
  }
}

/**
 * Get cached predictions for a specific stock
 */
export function getCachedPredictions(symbol: string, forecastHorizon: number): CachedPrediction | null {
  const allPredictions = getAllCachedPredictions();

  // Find prediction for this symbol with matching forecast horizon
  const cached = allPredictions.find(p =>
    p.symbol.toUpperCase() === symbol.toUpperCase() &&
    p.forecastHorizon === forecastHorizon
  );

  if (!cached) return null;

  // Check if it's from today (fresh)
  const now = Date.now();
  const age = now - cached.timestamp;
  const ageInHours = age / (1000 * 60 * 60);

  // Return if less than 24 hours old
  if (ageInHours < CACHE_EXPIRY_HOURS) {
    return cached;
  }

  return null;
}

/**
 * Save predictions to cache (keeps historical predictions, doesn't overwrite)
 */
export function savePredictionsToCache(
  symbol: string,
  predictions: CachedPrediction['predictions'],
  forecastHorizon: number
): void {
  try {
    const allPredictions = getAllCachedPredictions();

    // Add new prediction (don't remove old ones - keep history!)
    const newPrediction: CachedPrediction = {
      symbol: symbol.toUpperCase(),
      timestamp: Date.now(),
      date: new Date().toLocaleString(),
      predictions,
      forecastHorizon,
    };

    allPredictions.unshift(newPrediction); // Add to front

    // Keep only last 50 predictions to avoid filling localStorage (increased from 20)
    const toSave = allPredictions.slice(0, 50);

    localStorage.setItem(CACHE_KEY, JSON.stringify(toSave));
    console.log(`Saved ML predictions for ${symbol} to cache (historical)`);
  } catch (e) {
    console.error('Error saving predictions to cache:', e);
  }
}

/**
 * Get a specific cached prediction by timestamp
 */
export function getCachedPredictionByTimestamp(timestamp: number): CachedPrediction | null {
  const allPredictions = getAllCachedPredictions();
  return allPredictions.find(p => p.timestamp === timestamp) || null;
}

/**
 * Clear a specific cached prediction by timestamp
 */
export function clearCachedPredictionByTimestamp(timestamp: number): void {
  try {
    const allPredictions = getAllCachedPredictions();
    const filtered = allPredictions.filter(p => p.timestamp !== timestamp);
    localStorage.setItem(CACHE_KEY, JSON.stringify(filtered));
    console.log(`Cleared cache entry`);
  } catch (e) {
    console.error('Error clearing cache:', e);
  }
}

/**
 * Clear all cached predictions
 */
export function clearAllCachedPredictions(): void {
  try {
    localStorage.removeItem(CACHE_KEY);
    console.log('Cleared all cached predictions');
  } catch (e) {
    console.error('Error clearing all cache:', e);
  }
}

/**
 * Get cache statistics
 */
export function getCacheStats() {
  const predictions = getAllCachedPredictions();
  return {
    total: predictions.length,
    symbols: Array.from(new Set(predictions.map(p => p.symbol))),
    oldestTimestamp: predictions.length > 0 ? Math.min(...predictions.map(p => p.timestamp)) : null,
    newestTimestamp: predictions.length > 0 ? Math.max(...predictions.map(p => p.timestamp)) : null,
  };
}
