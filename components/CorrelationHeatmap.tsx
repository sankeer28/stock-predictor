'use client';

import React, { useState, useEffect } from 'react';
import { Loader2, RefreshCw, AlertTriangle, CheckCircle } from 'lucide-react';
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
      <span className="card-label">Correlation Heatmap</span>

      {/* Controls row */}
      <div className="flex gap-2 mb-4 mt-2 flex-wrap">
        <select
          id="correlation-year"
          value={selectedYear}
          onChange={(e) => setSelectedYear(parseInt(e.target.value))}
          className="px-2 py-1.5 border font-mono text-xs"
          style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-2)', outline: 'none' }}
        >
          {yearOptions.map(year => (
            <option key={year} value={year}>{year}</option>
          ))}
        </select>
        <input
          id="custom-symbols"
          type="text"
          value={customSymbols}
          onChange={(e) => setCustomSymbols(e.target.value)}
          placeholder="Custom symbols: AAPL,MSFT,GOOGL"
          className="flex-1 min-w-0 px-2 py-1.5 border font-mono text-xs"
          style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-2)', outline: 'none' }}
        />
        <button
          onClick={fetchCorrelationData}
          disabled={loading}
          className="px-3 py-1.5 text-xs font-semibold border disabled:opacity-50 flex items-center gap-1.5"
          style={{ background: 'var(--accent)', borderColor: 'var(--accent)', color: 'var(--text-0)' }}
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          Analyze
        </button>
      </div>

      {/* Compact legend */}
      <div className="flex items-center gap-3 mb-4 text-[11px]" style={{ color: 'var(--text-5)' }}>
        <span style={{ color: 'var(--text-4)' }}>Scale:</span>
        {[
          { color: '#ef4444', label: '-1.0 Opposite' },
          { color: '#fbbf24', label: '0.0 Independent' },
          { color: '#10b981', label: '+1.0 Together' },
        ].map(({ color, label }) => (
          <span key={label} className="flex items-center gap-1">
            <span style={{ display: 'inline-block', width: 10, height: 10, background: color, flexShrink: 0 }} />
            {label}
          </span>
        ))}
      </div>

      {loading && (
        <div className="flex items-center justify-center py-10">
          <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      )}

      {error && (
        <div className="p-3 border mb-4" style={{ background: 'var(--bg-2)', borderColor: 'var(--danger)' }}>
          <p className="text-xs" style={{ color: 'var(--danger)' }}>{error}</p>
        </div>
      )}

      {!loading && !error && correlationMatrix && symbols.length > 0 && (() => {
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
        const variance = correlations.reduce((sum, val) => sum + Math.pow(val - avgCorrelation, 2), 0) / correlations.length;
        const stdDev = Math.sqrt(variance);

        let maxPair: [string, string] = ['', ''];
        let minPair: [string, string] = ['', ''];
        let currentIdx = 0;
        for (let i = 0; i < correlationMatrix.length; i++) {
          for (let j = i + 1; j < correlationMatrix[i].length; j++) {
            if (currentIdx === maxIdx) maxPair = [symbols[i], symbols[j]];
            if (currentIdx === minIdx) minPair = [symbols[i], symbols[j]];
            currentIdx++;
          }
        }

        const highCount = correlations.filter(c => c >= 0.7).length;
        const modCount = correlations.filter(c => c >= 0.4 && c < 0.7).length;
        const lowCount = correlations.filter(c => c >= 0 && c < 0.4).length;
        const negCount = correlations.filter(c => c < 0).length;
        const highPct = (highCount / correlations.length) * 100;

        const stockAvgs = symbols.map((stock, i) => {
          const vals = correlationMatrix[i].filter((_, j) => i !== j);
          return { stock, avg: vals.reduce((a, b) => a + b, 0) / vals.length };
        }).sort((a, b) => b.avg - a.avg);
        const mostCorr = stockAvgs[0];
        const leastCorr = stockAvgs[stockAvgs.length - 1];

        const panelStyle = { background: 'var(--bg-3)', borderColor: 'var(--bg-1)' };
        const labelStyle = { color: 'var(--text-4)' } as React.CSSProperties;
        const dimStyle = { color: 'var(--text-5)' } as React.CSSProperties;

        return (
          <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr 1fr 1fr 1fr', gridTemplateRows: 'auto auto', gap: '8px', alignItems: 'start' }}>
            {/* Col 1: heatmap — spans both rows */}
            <div style={{ gridRow: '1 / 3', overflowX: 'auto' }}>
              <table className="border-collapse" style={{ fontSize: '11px' }}>
                <thead>
                  <tr>
                    <th className="border p-1" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}></th>
                    {symbols.map((sym, idx) => (
                      <th key={idx} className="border p-1 font-mono font-bold text-center"
                        style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)', color: 'var(--text-2)', minWidth: '48px' }}>
                        {sym}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {correlationMatrix.map((row, i) => (
                    <tr key={i}>
                      <td className="border p-1 font-mono font-bold text-center"
                        style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)', color: 'var(--text-2)' }}>
                        {symbols[i]}
                      </td>
                      {row.map((value, j) => (
                        <td key={j} className="border p-1.5 text-center font-mono font-semibold cursor-help"
                          style={{ background: getCorrelationColor(value), borderColor: 'var(--bg-1)', color: '#000', minWidth: '48px' }}
                          title={`${symbols[i]} vs ${symbols[j]}: ${value.toFixed(3)}`}>
                          {value.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Col 2: Statistics */}
            <div className="p-2 border h-full" style={panelStyle}>
              <div className="text-[10px] font-semibold mb-1.5" style={labelStyle}>STATISTICS</div>
              <div className="space-y-1 text-[11px]">
                <div className="flex justify-between gap-1">
                  <span style={dimStyle}>Avg</span>
                  <span className="font-mono font-bold" style={{ color: avgCorrelation >= 0.7 ? 'var(--warning)' : 'var(--success)' }}>{avgCorrelation.toFixed(3)}</span>
                </div>
                <div className="flex justify-between gap-1">
                  <span style={dimStyle}>StdDev</span>
                  <span className="font-mono" style={{ color: 'var(--text-3)' }}>{stdDev.toFixed(3)}</span>
                </div>
                <div className="flex justify-between gap-1">
                  <span style={dimStyle}>Range</span>
                  <span className="font-mono" style={{ color: 'var(--text-3)' }}>{minCorrelation.toFixed(2)}–{maxCorrelation.toFixed(2)}</span>
                </div>
                <div className="flex justify-between gap-1">
                  <span style={dimStyle}>Pairs</span>
                  <span className="font-mono" style={{ color: 'var(--text-3)' }}>{correlations.length}</span>
                </div>
              </div>
            </div>

            {/* Col 3: Distribution */}
            <div className="p-2 border h-full" style={panelStyle}>
              <div className="text-[10px] font-semibold mb-1.5" style={labelStyle}>DISTRIBUTION</div>
              <div className="space-y-1 text-[11px]">
                <div className="flex justify-between gap-1">
                  <span className="flex items-center gap-1"><span style={{ display:'inline-block', width:7, height:7, background:'#10b981', flexShrink:0 }} />High</span>
                  <span className="font-mono" style={{ color: highPct > 50 ? 'var(--danger)' : 'var(--text-3)' }}>{highCount} ({highPct.toFixed(0)}%)</span>
                </div>
                <div className="flex justify-between gap-1">
                  <span className="flex items-center gap-1"><span style={{ display:'inline-block', width:7, height:7, background:'#fbbf24', flexShrink:0 }} />Mod</span>
                  <span className="font-mono" style={{ color: 'var(--text-3)' }}>{modCount} ({((modCount / correlations.length) * 100).toFixed(0)}%)</span>
                </div>
                <div className="flex justify-between gap-1">
                  <span className="flex items-center gap-1"><span style={{ display:'inline-block', width:7, height:7, background:'#94a3b8', flexShrink:0 }} />Low</span>
                  <span className="font-mono" style={{ color: 'var(--text-3)' }}>{lowCount} ({((lowCount / correlations.length) * 100).toFixed(0)}%)</span>
                </div>
                {negCount > 0 && (
                  <div className="flex justify-between gap-1">
                    <span className="flex items-center gap-1"><span style={{ display:'inline-block', width:7, height:7, background:'#ef4444', flexShrink:0 }} />Neg</span>
                    <span className="font-mono" style={{ color: 'var(--success)' }}>{negCount} ({((negCount / correlations.length) * 100).toFixed(0)}%)</span>
                  </div>
                )}
              </div>
            </div>

            {/* Col 4: Stocks */}
            <div className="p-2 border h-full" style={panelStyle}>
              <div className="text-[10px] font-semibold mb-1.5" style={labelStyle}>STOCKS</div>
              <div className="space-y-2 text-[11px]">
                <div>
                  <div style={dimStyle}>Most correlated</div>
                  <div className="font-mono font-bold" style={{ color: 'var(--warning)' }}>
                    {mostCorr.stock} <span style={{ color: 'var(--text-4)' }}>{mostCorr.avg.toFixed(3)}</span>
                  </div>
                </div>
                <div>
                  <div style={dimStyle}>Least correlated</div>
                  <div className="font-mono font-bold" style={{ color: 'var(--success)' }}>
                    {leastCorr.stock} <span style={{ color: 'var(--text-4)' }}>{leastCorr.avg.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Col 5: Key Pairs */}
            <div className="p-2 border h-full" style={panelStyle}>
              <div className="text-[10px] font-semibold mb-1.5" style={labelStyle}>KEY PAIRS</div>
              <div className="space-y-2 text-[11px]">
                <div>
                  <div style={dimStyle}>Strongest</div>
                  <div className="font-mono font-bold" style={{ color: 'var(--danger)' }}>
                    {maxPair[0]}↔{maxPair[1]}<br />
                    <span style={{ color: 'var(--text-4)' }}>{maxCorrelation.toFixed(3)}</span>
                  </div>
                </div>
                <div>
                  <div style={dimStyle}>Weakest</div>
                  <div className="font-mono font-bold" style={{ color: minCorrelation < 0 ? 'var(--success)' : 'var(--text-3)' }}>
                    {minPair[0]}↔{minPair[1]}<br />
                    <span style={{ color: 'var(--text-4)' }}>{minCorrelation.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Row 2, cols 2–5: Insight */}
            <div className="p-2 border-l-2 text-[11px]" style={{
              gridColumn: '2 / 6',
              background: 'var(--bg-3)',
              borderColor: highPct > 50 ? 'var(--warning)' : 'var(--success)',
              color: 'var(--text-4)'
            }}>
              {highPct > 50 ? (
                <>
                  <span style={{ color: 'var(--warning)' }}>High correlation risk</span> — {highPct.toFixed(0)}% of pairs tightly linked.
                  {' '}{mostCorr.stock} moves most with the group; {leastCorr.stock} offers the best diversification.
                  {negCount > 0 && <>{' '}Some pairs negatively correlated — useful for hedging.</>}
                </>
              ) : (
                <>
                  <span style={{ color: 'var(--success)' }}>Well-diversified group</span> — avg {avgCorrelation.toFixed(2)}.
                  {' '}{leastCorr.stock} most independent; {maxPair[0]} & {maxPair[1]} closest ({maxCorrelation.toFixed(2)}).
                  {stdDev > 0.2 && <>{' '}High spread ({stdDev.toFixed(2)}) shows diverse relationships.</>}
                </>
              )}
            </div>
          </div>
        );
      })()}
    </div>
  );
};

export default CorrelationHeatmap;
