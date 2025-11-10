'use client';

import React, { useState, useEffect } from 'react';
import { Database, Trash2, RefreshCw } from 'lucide-react';
import {
  getAllCachedPredictions,
  clearCachedPrediction,
  clearAllCachedPredictions,
  CachedPrediction,
} from '@/lib/predictionsCache';

interface PredictionsCacheProps {
  onLoadPrediction?: (symbol: string) => void;
}

export default function PredictionsCache({ onLoadPrediction }: PredictionsCacheProps) {
  const [cachedPredictions, setCachedPredictions] = useState<CachedPrediction[]>([]);
  const [isExpanded, setIsExpanded] = useState(false);

  const loadCache = () => {
    const predictions = getAllCachedPredictions();
    setCachedPredictions(predictions);
  };

  useEffect(() => {
    loadCache();

    // Refresh cache display every minute
    const interval = setInterval(loadCache, 60000);
    return () => clearInterval(interval);
  }, []);

  const handleClearOne = (symbol: string, forecastHorizon: number) => {
    clearCachedPrediction(symbol, forecastHorizon);
    loadCache();
  };

  const handleClearAll = () => {
    if (confirm('Are you sure you want to clear all cached predictions?')) {
      clearAllCachedPredictions();
      loadCache();
    }
  };

  const formatTimeAgo = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const minutes = Math.floor(diff / (1000 * 60));
    const hours = Math.floor(diff / (1000 * 60 * 60));

    if (minutes < 1) return 'Just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    return new Date(timestamp).toLocaleDateString();
  };

  const countModels = (predictions: CachedPrediction['predictions']) => {
    return Object.values(predictions).filter(p => p && p.length > 0).length;
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Database className="w-4 h-4" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Predictions Cache</span>
          <span className="text-xs px-2 py-0.5 rounded" style={{ background: 'var(--bg-1)', color: 'var(--text-4)' }}>
            {cachedPredictions.length}
          </span>
        </div>
        {cachedPredictions.length > 0 && (
          <button
            onClick={handleClearAll}
            className="text-xs p-1 hover:bg-opacity-80 transition-all"
            style={{ color: 'var(--error)' }}
            title="Clear all cache"
          >
            <Trash2 className="w-4 h-4" />
          </button>
        )}
      </div>

      {cachedPredictions.length === 0 ? (
        <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
          <Database className="w-12 h-12 mx-auto mb-2 opacity-30" />
          <p className="text-sm">No cached predictions yet</p>
          <p className="text-xs mt-1">ML predictions will be saved here</p>
        </div>
      ) : (
        <>
          {/* Table View */}
          <div className="overflow-x-auto">
            <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-1)' }}>
                  <th className="text-left py-2 px-2" style={{ color: 'var(--text-4)' }}>Symbol</th>
                  <th className="text-left py-2 px-2" style={{ color: 'var(--text-4)' }}>Models</th>
                  <th className="text-left py-2 px-2" style={{ color: 'var(--text-4)' }}>Days</th>
                  <th className="text-left py-2 px-2" style={{ color: 'var(--text-4)' }}>Cached</th>
                  <th className="text-right py-2 px-2" style={{ color: 'var(--text-4)' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {cachedPredictions.slice(0, isExpanded ? undefined : 5).map((pred, index) => (
                  <tr
                    key={`${pred.symbol}-${pred.timestamp}`}
                    style={{
                      borderBottom: index < cachedPredictions.length - 1 ? '1px solid var(--bg-1)' : 'none',
                    }}
                  >
                    <td className="py-2 px-2">
                      <button
                        onClick={() => onLoadPrediction?.(pred.symbol)}
                        className="font-mono font-semibold hover:underline transition-all"
                        style={{ color: 'var(--accent)' }}
                      >
                        {pred.symbol}
                      </button>
                    </td>
                    <td className="py-2 px-2" style={{ color: 'var(--text-3)' }}>
                      {countModels(pred.predictions)}
                    </td>
                    <td className="py-2 px-2" style={{ color: 'var(--text-3)' }}>
                      {pred.forecastHorizon}d
                    </td>
                    <td className="py-2 px-2" style={{ color: 'var(--text-4)' }}>
                      {formatTimeAgo(pred.timestamp)}
                    </td>
                    <td className="py-2 px-2 text-right">
                      <button
                        onClick={() => handleClearOne(pred.symbol, pred.forecastHorizon)}
                        className="p-1 hover:bg-opacity-80 transition-all"
                        style={{ color: 'var(--error)' }}
                        title="Remove from cache"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Show More/Less Button */}
          {cachedPredictions.length > 5 && (
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="w-full mt-3 py-2 text-xs border transition-all"
              style={{
                background: 'var(--bg-3)',
                borderColor: 'var(--bg-1)',
                color: 'var(--text-3)',
              }}
            >
              {isExpanded ? 'Show Less' : `Show All (${cachedPredictions.length})`}
            </button>
          )}
        </>
      )}
    </div>
  );
}
