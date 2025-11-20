'use client';

import React, { useState, useEffect } from 'react';
import { Database, Trash2, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';
import {
  getAllCachedPredictions,
  clearCachedPredictionByTimestamp,
  clearAllCachedPredictions,
  CachedPrediction,
} from '@/lib/predictionsCache';

interface PredictionsCacheProps {
  onLoadPrediction?: (prediction: CachedPrediction) => void;
}

export default function PredictionsCache({ onLoadPrediction }: PredictionsCacheProps) {
  const [cachedPredictions, setCachedPredictions] = useState<CachedPrediction[]>([]);
  const [isCollapsed, setIsCollapsed] = useState(true);
  const [showAll, setShowAll] = useState(false);

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

  const handleClearOne = (timestamp: number, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent row click
    clearCachedPredictionByTimestamp(timestamp);
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

  return (
    <div className="card">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setIsCollapsed(!isCollapsed)}
      >
        <div className="flex items-center gap-2">
          <Database className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">ML Cache</span>
          <span className="text-xs px-2 py-0.5 rounded" style={{
            background: 'var(--bg-2)',
            color: 'var(--text-4)',
            border: '1px solid var(--bg-1)'
          }}>
            {cachedPredictions.length}
          </span>
        </div>
        {isCollapsed ? (
          <ChevronDown className="w-5 h-5" style={{ color: 'var(--text-4)' }} />
        ) : (
          <ChevronUp className="w-5 h-5" style={{ color: 'var(--text-4)' }} />
        )}
      </div>

      {!isCollapsed && (
        <>
          {cachedPredictions.length > 0 && (
            <div className="flex justify-end mb-2 mt-3">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleClearAll();
                }}
                className="text-xs px-2 py-1 hover:bg-opacity-80 transition-all flex items-center gap-1"
                style={{ color: 'var(--error)' }}
                title="Clear all cache"
              >
                <Trash2 className="w-3 h-3" />
                <span>Clear All</span>
              </button>
            </div>
          )}

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
                  <th className="text-left py-2 px-2" style={{ color: 'var(--text-4)' }}>Cached</th>
                  <th className="text-right py-2 px-2" style={{ color: 'var(--text-4)' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {cachedPredictions.slice(0, showAll ? undefined : 5).map((pred, index) => (
                  <tr
                    key={`${pred.symbol}-${pred.timestamp}`}
                    onClick={() => onLoadPrediction?.(pred)}
                    className="cursor-pointer hover:bg-opacity-80 transition-all"
                    style={{
                      borderBottom: index < cachedPredictions.length - 1 ? '1px solid var(--bg-1)' : 'none',
                      background: 'transparent',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.background = 'var(--bg-1)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    <td className="py-2 px-2">
                      <span className="font-mono font-semibold" style={{ color: 'var(--accent)' }}>
                        {pred.symbol}
                      </span>
                    </td>
                    <td className="py-2 px-2" style={{ color: 'var(--text-4)' }}>
                      {formatTimeAgo(pred.timestamp)}
                    </td>
                    <td className="py-2 px-2 text-right">
                      <button
                        onClick={(e) => handleClearOne(pred.timestamp, e)}
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
              onClick={() => setShowAll(!showAll)}
              className="w-full mt-3 py-2 text-xs border transition-all"
              style={{
                background: 'var(--bg-3)',
                borderColor: 'var(--bg-1)',
                color: 'var(--text-3)',
              }}
            >
              {showAll ? 'Show Less' : `Show All (${cachedPredictions.length})`}
            </button>
          )}
        </>
      )}
        </>
      )}
    </div>
  );
}
