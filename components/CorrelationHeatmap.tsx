'use client';

import React, { useState, useEffect } from 'react';
import { Loader2, RefreshCw } from 'lucide-react';
import { calculateCorrelationMatrix } from '@/lib/correlationAnalysis';

interface CorrelationHeatmapProps {
  symbol: string;
  inlineMobile?: boolean;
}

const CorrelationHeatmap: React.FC<CorrelationHeatmapProps> = ({
  symbol,
  inlineMobile = false
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [correlationMatrix, setCorrelationMatrix] = useState<number[][] | null>(null);
  const [symbols, setSymbols] = useState<string[]>([]);
  const [selectedYear, setSelectedYear] = useState<number>(new Date().getFullYear() - 1);
  const [customSymbols, setCustomSymbols] = useState('');
  const [relatedStocks, setRelatedStocks] = useState<string[]>([]);

  // Generate year options (last 5 years)
  const currentYear = new Date().getFullYear();
  const yearOptions = Array.from({ length: 5 }, (_, i) => currentYear - i);

  // Fetch related stocks when symbol changes
  useEffect(() => {
    const fetchRelatedStocks = async () => {
      if (!symbol) return;

      try {
        const response = await fetch(`/api/related-stocks?symbol=${symbol}`);
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.relatedStocks) {
            setRelatedStocks(data.relatedStocks);
          }
        }
      } catch (err) {
        console.error('Error fetching related stocks:', err);
        // Use fallback if API fails
        setRelatedStocks(['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD']);
      }
    };

    fetchRelatedStocks();
  }, [symbol]);

  const fetchCorrelationData = async () => {
    setLoading(true);
    setError('');

    try {
      // Determine which symbols to use
      let symbolsToFetch: string[] = [];

      if (customSymbols.trim()) {
        // Use custom symbols if provided
        symbolsToFetch = customSymbols.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
      } else {
        // Use the current symbol plus related stocks
        symbolsToFetch = [symbol, ...relatedStocks.filter(s => s !== symbol)].slice(0, 8);
      }

      // Date range for selected year
      const startDate = `${selectedYear}-01-01`;
      const endDate = `${selectedYear}-12-31`;

      console.log(`Fetching correlation data for: ${symbolsToFetch.join(', ')} (${selectedYear})`);

      const response = await fetch(
        `/api/correlation?symbols=${symbolsToFetch.join(',')}&startDate=${startDate}&endDate=${endDate}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch correlation data');
      }

      const data = await response.json();

      if (!data.success || !data.stocks || data.stocks.length < 2) {
        throw new Error('Insufficient data for correlation analysis');
      }

      // Calculate correlation matrix
      const matrix = calculateCorrelationMatrix(data.stocks, startDate, endDate);

      setCorrelationMatrix(matrix.matrix);
      setSymbols(matrix.symbols);
    } catch (err: any) {
      console.error('Error fetching correlation data:', err);
      setError(err.message || 'Failed to load correlation data');
    } finally {
      setLoading(false);
    }
  };

  // Auto-fetch on mount and when year/relatedStocks changes
  useEffect(() => {
    if (relatedStocks.length > 0) {
      fetchCorrelationData();
    }
  }, [selectedYear, relatedStocks]);

  // Get color for correlation value
  const getCorrelationColor = (value: number): string => {
    // Normalize to 0-1 range (correlation is -1 to 1)
    const normalized = (value + 1) / 2;

    if (normalized >= 0.8) return '#10b981'; // Strong positive - green
    if (normalized >= 0.6) return '#34d399';
    if (normalized >= 0.4) return '#fbbf24'; // Neutral - yellow
    if (normalized >= 0.2) return '#f87171'; // Negative - red
    return '#ef4444'; // Strong negative - dark red
  };

  return (
    <div
      className={`card ${inlineMobile ? '' : 'mb-6'}`}
      style={{ minWidth: inlineMobile ? '100%' : '400px' }}
    >
      <div className="flex items-center justify-between mb-4">
        <span className="card-label">Correlation Heatmap</span>
        <button
          onClick={fetchCorrelationData}
          disabled={loading}
          className="p-1.5 border transition-all disabled:opacity-50"
          style={{
            background: 'var(--bg-3)',
            borderColor: 'var(--bg-1)',
            color: 'var(--text-3)'
          }}
          title="Refresh"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* How to Read Section */}
      <div className="mb-4 p-4 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', borderLeftWidth: '3px', borderLeftColor: 'var(--accent)' }}>
        <div className="flex items-start gap-2 mb-2">
          <div className="text-sm font-semibold" style={{ color: 'var(--text-2)' }}>
            📊 How to Read This Matrix
          </div>
        </div>
        <div className="text-xs space-y-1.5" style={{ color: 'var(--text-4)' }}>
          <p>
            <strong>Correlation measures how stocks move together.</strong> Values range from -1.0 to +1.0:
          </p>
          <ul className="ml-4 space-y-1">
            <li>
              <span className="font-semibold" style={{ color: '#10b981' }}>+1.0 (Green)</span> = Perfect positive correlation. Stocks move in the same direction.
            </li>
            <li>
              <span className="font-semibold" style={{ color: '#fbbf24' }}>0.0 (Yellow)</span> = No correlation. Stocks move independently.
            </li>
            <li>
              <span className="font-semibold" style={{ color: '#ef4444' }}>-1.0 (Red)</span> = Perfect negative correlation. Stocks move in opposite directions.
            </li>
          </ul>
          <p className="pt-1 italic">
            <strong>Tip:</strong> High correlation (0.7+) means stocks tend to move together, useful for portfolio diversification.
          </p>
        </div>
      </div>

      {/* Year Selector */}
      <div className="mb-4 p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
        <label htmlFor="correlation-year" className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-4)' }}>
          Select Year
        </label>
        <select
          id="correlation-year"
          value={selectedYear}
          onChange={(e) => setSelectedYear(parseInt(e.target.value))}
          className="w-full px-3 py-2 border font-mono text-sm"
          style={{
            background: 'var(--bg-4)',
            borderColor: 'var(--bg-1)',
            color: 'var(--text-2)',
            outline: 'none'
          }}
        >
          {yearOptions.map(year => (
            <option key={year} value={year}>
              {year}
            </option>
          ))}
        </select>
      </div>

      {/* Custom Symbols Input */}
      <div className="mb-4 p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
        <label htmlFor="custom-symbols" className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-4)' }}>
          Custom Symbols (Optional)
        </label>
        <input
          id="custom-symbols"
          type="text"
          value={customSymbols}
          onChange={(e) => setCustomSymbols(e.target.value)}
          placeholder="e.g., AAPL,MSFT,GOOGL"
          className="w-full px-3 py-2 border font-mono text-sm mb-2"
          style={{
            background: 'var(--bg-4)',
            borderColor: 'var(--bg-1)',
            color: 'var(--text-2)',
            outline: 'none'
          }}
        />
        <button
          onClick={fetchCorrelationData}
          disabled={loading}
          className="w-full px-3 py-2 text-xs font-semibold border transition-all disabled:opacity-50"
          style={{
            background: 'var(--accent)',
            borderColor: 'var(--accent)',
            color: 'var(--text-0)'
          }}
        >
          Analyze Custom Symbols
        </button>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-10">
          <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      )}

      {error && (
        <div className="p-4 border-2 mb-4" style={{ background: 'var(--bg-2)', borderColor: 'var(--danger)' }}>
          <p className="text-sm" style={{ color: 'var(--danger)' }}>{error}</p>
        </div>
      )}

      {!loading && !error && correlationMatrix && symbols.length > 0 && (
        <div className="overflow-x-auto">
          <div className="text-xs mb-3 text-center" style={{ color: 'var(--text-4)' }}>
            Correlation matrix for {selectedYear}
          </div>

          {/* Heatmap Grid */}
          <div className="inline-block min-w-full">
            <table className="w-full border-collapse" style={{ fontSize: '11px' }}>
              <thead>
                <tr>
                  <th className="border p-1" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}></th>
                  {symbols.map((sym, idx) => (
                    <th
                      key={idx}
                      className="border p-1 font-mono font-bold text-center"
                      style={{
                        background: 'var(--bg-2)',
                        borderColor: 'var(--bg-1)',
                        color: 'var(--text-2)',
                        minWidth: '50px'
                      }}
                    >
                      {sym}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlationMatrix.map((row, i) => (
                  <tr key={i}>
                    <td
                      className="border p-1 font-mono font-bold text-center"
                      style={{
                        background: 'var(--bg-2)',
                        borderColor: 'var(--bg-1)',
                        color: 'var(--text-2)'
                      }}
                    >
                      {symbols[i]}
                    </td>
                    {row.map((value, j) => (
                      <td
                        key={j}
                        className="border p-2 text-center font-mono font-semibold transition-all cursor-help"
                        style={{
                          background: getCorrelationColor(value),
                          borderColor: 'var(--bg-1)',
                          color: '#000',
                          minWidth: '50px'
                        }}
                        title={`${symbols[i]} vs ${symbols[j]}: ${value.toFixed(3)}`}
                      >
                        {value.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Legend */}
          <div className="mt-4 p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
            <div className="text-xs font-semibold mb-2" style={{ color: 'var(--text-4)' }}>
              Correlation Scale
            </div>
            <div className="flex items-center gap-2 text-xs">
              <div className="flex-1 flex items-center gap-1">
                <div className="w-4 h-4 border" style={{ background: '#ef4444', borderColor: 'var(--bg-1)' }}></div>
                <span style={{ color: 'var(--text-5)' }}>-1.0</span>
              </div>
              <div className="flex-1 flex items-center gap-1">
                <div className="w-4 h-4 border" style={{ background: '#fbbf24', borderColor: 'var(--bg-1)' }}></div>
                <span style={{ color: 'var(--text-5)' }}>0.0</span>
              </div>
              <div className="flex-1 flex items-center gap-1">
                <div className="w-4 h-4 border" style={{ background: '#10b981', borderColor: 'var(--bg-1)' }}></div>
                <span style={{ color: 'var(--text-5)' }}>+1.0</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CorrelationHeatmap;
