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
            <strong>Tip:</strong> High correlation (0.7+) means stocks tend to move together - important to consider when building a diversified investment strategy.
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

          {/* Conclusion */}
          {(() => {
            // Calculate statistics from the correlation matrix
            const correlations: number[] = [];
            for (let i = 0; i < correlationMatrix.length; i++) {
              for (let j = i + 1; j < correlationMatrix[i].length; j++) {
                correlations.push(correlationMatrix[i][j]);
              }
            }

            if (correlations.length === 0) return null;

            const avgCorrelation = correlations.reduce((a, b) => a + b, 0) / correlations.length;
            const maxCorrelation = Math.max(...correlations);
            const minCorrelation = Math.min(...correlations);
            const maxIdx = correlations.indexOf(maxCorrelation);
            const minIdx = correlations.indexOf(minCorrelation);

            // Calculate standard deviation
            const variance = correlations.reduce((sum, val) => sum + Math.pow(val - avgCorrelation, 2), 0) / correlations.length;
            const stdDev = Math.sqrt(variance);

            // Find the stock pairs for max and min correlation
            let maxPair: [string, string] = ['', ''];
            let minPair: [string, string] = ['', ''];
            let currentIdx = 0;

            for (let i = 0; i < correlationMatrix.length; i++) {
              for (let j = i + 1; j < correlationMatrix[i].length; j++) {
                if (currentIdx === maxIdx) {
                  maxPair = [symbols[i], symbols[j]];
                }
                if (currentIdx === minIdx) {
                  minPair = [symbols[i], symbols[j]];
                }
                currentIdx++;
              }
            }

            // Correlation categories
            const highCorrelationCount = correlations.filter(c => c >= 0.7).length;
            const moderateCorrelationCount = correlations.filter(c => c >= 0.4 && c < 0.7).length;
            const lowCorrelationCount = correlations.filter(c => c >= 0 && c < 0.4).length;
            const negativeCorrelationCount = correlations.filter(c => c < 0).length;

            const highCorrelationPct = (highCorrelationCount / correlations.length) * 100;
            const moderateCorrelationPct = (moderateCorrelationCount / correlations.length) * 100;
            const lowCorrelationPct = (lowCorrelationCount / correlations.length) * 100;
            const negativeCorrelationPct = (negativeCorrelationCount / correlations.length) * 100;

            // Individual stock analysis - calculate average correlation for each stock
            const stockAvgCorrelations = symbols.map((stock, i) => {
              const stockCorrelations = correlationMatrix[i].filter((_, j) => i !== j);
              const avg = stockCorrelations.reduce((a, b) => a + b, 0) / stockCorrelations.length;
              return { stock, avgCorrelation: avg };
            });

            stockAvgCorrelations.sort((a, b) => b.avgCorrelation - a.avgCorrelation);
            const mostCorrelatedStock = stockAvgCorrelations[0];
            const leastCorrelatedStock = stockAvgCorrelations[stockAvgCorrelations.length - 1];

            // Find highly correlated clusters (stocks with avg correlation > 0.7)
            const highlyCorrelatedStocks = stockAvgCorrelations.filter(s => s.avgCorrelation >= 0.7);

            return (
              <div className="mt-4 p-4 border-2" style={{
                background: 'var(--bg-2)',
                borderColor: 'var(--accent)',
                borderLeftWidth: '4px'
              }}>
                <div className="text-sm font-bold mb-3" style={{ color: 'var(--text-2)' }}>
                  Analysis
                </div>

                <div className="text-xs space-y-3" style={{ color: 'var(--text-3)' }}>
                  {/* Overall Statistics */}
                  <div className="p-2 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
                    <div className="font-semibold mb-1.5" style={{ color: 'var(--text-2)' }}>Stock Group Statistics</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <span className="text-[11px]" style={{ color: 'var(--text-5)' }}>Average Correlation:</span>
                        <div className="font-mono font-bold" style={{ color: avgCorrelation >= 0.7 ? 'var(--warning)' : avgCorrelation >= 0.4 ? 'var(--warning)' : 'var(--success)' }}>
                          {avgCorrelation.toFixed(3)}
                        </div>
                      </div>
                      <div>
                        <span className="text-[11px]" style={{ color: 'var(--text-5)' }}>Std. Deviation:</span>
                        <div className="font-mono font-bold" style={{ color: 'var(--text-2)' }}>
                          {stdDev.toFixed(3)}
                        </div>
                      </div>
                      <div>
                        <span className="text-[11px]" style={{ color: 'var(--text-5)' }}>Range:</span>
                        <div className="font-mono text-[11px]" style={{ color: 'var(--text-3)' }}>
                          {minCorrelation.toFixed(2)} to {maxCorrelation.toFixed(2)}
                        </div>
                      </div>
                      <div>
                        <span className="text-[11px]" style={{ color: 'var(--text-5)' }}>Total Pairs:</span>
                        <div className="font-mono font-bold" style={{ color: 'var(--text-2)' }}>
                          {correlations.length}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Distribution */}
                  <div className="p-2 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
                    <div className="font-semibold mb-1.5" style={{ color: 'var(--text-2)' }}>Correlation Distribution</div>
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-[11px]">🟢 High (≥0.7):</span>
                        <span className="font-mono font-bold" style={{ color: highCorrelationPct > 50 ? 'var(--danger)' : 'var(--text-3)' }}>
                          {highCorrelationCount} pairs ({highCorrelationPct.toFixed(0)}%)
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[11px]">🟡 Moderate (0.4-0.7):</span>
                        <span className="font-mono" style={{ color: 'var(--text-3)' }}>
                          {moderateCorrelationCount} pairs ({moderateCorrelationPct.toFixed(0)}%)
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-[11px]">⚪ Low (0-0.4):</span>
                        <span className="font-mono" style={{ color: 'var(--text-3)' }}>
                          {lowCorrelationCount} pairs ({lowCorrelationPct.toFixed(0)}%)
                        </span>
                      </div>
                      {negativeCorrelationCount > 0 && (
                        <div className="flex items-center justify-between">
                          <span className="text-[11px]">🔴 Negative (&lt;0):</span>
                          <span className="font-mono" style={{ color: 'var(--success)' }}>
                            {negativeCorrelationCount} pairs ({negativeCorrelationPct.toFixed(0)}%)
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Individual Stock Analysis */}
                  <div className="p-2 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
                    <div className="font-semibold mb-1.5" style={{ color: 'var(--text-2)' }}>Individual Stock Analysis</div>
                    <div className="space-y-2">
                      <div>
                        <div className="text-[11px] mb-1" style={{ color: 'var(--text-5)' }}>Most Correlated (moves with others):</div>
                        <div className="font-mono font-bold" style={{ color: 'var(--warning)' }}>
                          {mostCorrelatedStock.stock}: {mostCorrelatedStock.avgCorrelation.toFixed(3)}
                        </div>
                        <div className="text-[10px] mt-0.5 italic" style={{ color: 'var(--text-5)' }}>
                          Tends to follow the group closely
                        </div>
                      </div>
                      <div>
                        <div className="text-[11px] mb-1" style={{ color: 'var(--text-5)' }}>Least Correlated (independent):</div>
                        <div className="font-mono font-bold" style={{ color: 'var(--success)' }}>
                          {leastCorrelatedStock.stock}: {leastCorrelatedStock.avgCorrelation.toFixed(3)}
                        </div>
                        <div className="text-[10px] mt-0.5 italic" style={{ color: 'var(--text-5)' }}>
                          Most independent movement, best for diversification
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Key Relationships */}
                  <div className="p-2 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
                    <div className="font-semibold mb-1.5" style={{ color: 'var(--text-2)' }}>Key Relationships</div>
                    <div className="space-y-1.5">
                      <div>
                        <span className="text-[11px]" style={{ color: 'var(--text-5)' }}>Strongest Link:</span>
                        <div className="font-mono font-bold" style={{ color: 'var(--danger)' }}>
                          {maxPair[0]} ↔ {maxPair[1]}: {maxCorrelation.toFixed(3)}
                        </div>
                        <div className="text-[10px] italic" style={{ color: 'var(--text-5)' }}>
                          These stocks move almost identically - very low diversification benefit
                        </div>
                      </div>
                      <div>
                        <span className="text-[11px]" style={{ color: 'var(--text-5)' }}>Weakest Link:</span>
                        <div className="font-mono font-bold" style={{ color: minCorrelation < 0 ? 'var(--success)' : 'var(--text-3)' }}>
                          {minPair[0]} ↔ {minPair[1]}: {minCorrelation.toFixed(3)}
                        </div>
                        <div className="text-[10px] italic" style={{ color: 'var(--text-5)' }}>
                          {minCorrelation < 0
                            ? "Negative correlation - excellent hedge pair!"
                            : "Low correlation - good diversification potential"}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Recommendations */}
                  <div className="p-3 border-2" style={{
                    background: 'var(--bg-4)',
                    borderColor: highCorrelationPct > 50 ? 'var(--warning)' : 'var(--success)',
                    borderLeftWidth: '3px'
                  }}>
                    <div className="font-semibold mb-2" style={{ color: 'var(--text-2)' }}>
                      {highCorrelationPct > 50 ? '⚠️ Investment Insights' : '✓ Investment Insights'}
                    </div>
                    <div className="space-y-2">
                      {highCorrelationPct > 50 ? (
                        <>
                          <p className="text-[11px]" style={{ color: 'var(--warning)' }}>
                            <strong>High Correlation Risk:</strong> {highCorrelationPct.toFixed(0)}% of these stock pairs are highly correlated.
                            If you're investing in multiple stocks from this group, they may decline together during market downturns.
                          </p>
                          <p className="text-[11px]" style={{ color: 'var(--text-4)' }}>
                            <strong>Diversification Tips:</strong>
                          </p>
                          <ul className="text-[10px] ml-4 space-y-0.5" style={{ color: 'var(--text-4)' }}>
                            <li>• {mostCorrelatedStock.stock} moves most closely with the group (avg: {mostCorrelatedStock.avgCorrelation.toFixed(2)})</li>
                            <li>• For diversification, look for stocks with &lt;0.4 correlation to {symbol}</li>
                            <li>• {leastCorrelatedStock.stock} has the most independent movement in this group</li>
                            {negativeCorrelationCount > 0 && <li>• Some pairs have negative correlation - good for hedging</li>}
                          </ul>
                        </>
                      ) : (
                        <>
                          <p className="text-[11px]" style={{ color: 'var(--success)' }}>
                            <strong>Well-Diversified Group:</strong> These stocks show varied movement patterns. If investing in multiple from this group, you'd have natural diversification.
                          </p>
                          <p className="text-[11px]" style={{ color: 'var(--text-4)' }}>
                            <strong>Key Observations:</strong>
                          </p>
                          <ul className="text-[10px] ml-4 space-y-0.5" style={{ color: 'var(--text-4)' }}>
                            <li>• {leastCorrelatedStock.stock} has the most independent performance (avg: {leastCorrelatedStock.avgCorrelation.toFixed(2)})</li>
                            <li>• {maxPair[0]} and {maxPair[1]} move together most closely ({maxCorrelation.toFixed(2)})</li>
                            <li>• Average correlation of {avgCorrelation.toFixed(2)} suggests moderate independence</li>
                            {stdDev > 0.2 && <li>• High variation ({stdDev.toFixed(2)}) shows diverse relationships between stocks</li>}
                          </ul>
                        </>
                      )}
                    </div>
                  </div>

                  {/* Market Context */}
                  <div className="pt-2 text-[10px] italic border-t" style={{ borderColor: 'var(--bg-1)', color: 'var(--text-5)' }}>
                    <strong>Note:</strong> Correlations are calculated for {selectedYear} and may change over time.
                    Low correlation between stocks can help reduce overall volatility if you invest in multiple assets.
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
};

export default CorrelationHeatmap;
