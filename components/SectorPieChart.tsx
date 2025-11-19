'use client';

import React, { useState, useEffect } from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { Loader2, RefreshCw } from 'lucide-react';
import { SECTOR_COMPANIES, categorizeSector } from '@/lib/correlationAnalysis';

interface SectorPieChartProps {
  symbol: string;
  sector?: string;
  inlineMobile?: boolean;
}

interface SectorData {
  name: string;
  value: number;
  color: string;
}

const SECTOR_COLORS: Record<string, string> = {
  'Technology': '#3b82f6',
  'Finance': '#10b981',
  'Healthcare': '#ef4444',
  'Consumer': '#f59e0b',
  'Energy': '#8b5cf6',
  'Industrial': '#6366f1',
  'Telecom': '#ec4899',
  'Utilities': '#14b8a6',
  'Real Estate': '#f97316',
  'Materials': '#84cc16',
  'Other': '#6b7280'
};

const SectorPieChart: React.FC<SectorPieChartProps> = ({
  symbol,
  sector,
  inlineMobile = false
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [sectorData, setSectorData] = useState<SectorData[]>([]);
  const [customSymbols, setCustomSymbols] = useState('');

  const fetchSectorData = async () => {
    setLoading(true);
    setError('');

    try {
      // Determine which symbols to analyze
      let symbolsToAnalyze: string[] = [];

      if (customSymbols.trim()) {
        symbolsToAnalyze = customSymbols.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);
      } else if (sector) {
        // Categorize the sector if it's a SIC description
        const categorizedSector = categorizeSector(sector);
        if (SECTOR_COMPANIES[categorizedSector]) {
          symbolsToAnalyze = SECTOR_COMPANIES[categorizedSector];
        } else {
          // Use a diverse portfolio as default
          symbolsToAnalyze = [
            'AAPL', 'MSFT', 'GOOGL', // Tech
            'JPM', 'BAC', // Finance
            'JNJ', 'PFE', // Healthcare
            'AMZN', 'WMT', // Consumer
            'XOM', 'CVX', // Energy
          ];
        }
      } else {
        // Use a diverse portfolio as default
        symbolsToAnalyze = [
          'AAPL', 'MSFT', 'GOOGL', // Tech
          'JPM', 'BAC', // Finance
          'JNJ', 'PFE', // Healthcare
          'AMZN', 'WMT', // Consumer
          'XOM', 'CVX', // Energy
        ];
      }

      console.log(`Fetching sector allocation for: ${symbolsToAnalyze.join(', ')}`);

      const response = await fetch(`/api/sector-allocation?symbols=${symbolsToAnalyze.join(',')}`);

      if (!response.ok) {
        throw new Error('Failed to fetch sector data');
      }

      const data = await response.json();

      if (!data.success || !data.sectorAllocation) {
        throw new Error('Invalid sector data received');
      }

      // Transform data for pie chart
      const chartData: SectorData[] = Object.entries(data.sectorAllocation).map(([sectorName, count]) => ({
        name: sectorName,
        value: count as number,
        color: SECTOR_COLORS[sectorName] || SECTOR_COLORS['Other']
      }));

      setSectorData(chartData);
    } catch (err: any) {
      console.error('Error fetching sector data:', err);
      setError(err.message || 'Failed to load sector data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSectorData();
  }, [symbol, sector]);

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0];
      const total = sectorData.reduce((sum, item) => sum + item.value, 0);
      const percentage = ((data.value / total) * 100).toFixed(1);

      return (
        <div className="p-3 border-2" style={{
          background: 'var(--bg-2)',
          borderColor: 'var(--bg-1)',
          borderLeftColor: data.payload.color,
          borderLeftWidth: '3px'
        }}>
          <p className="text-sm font-semibold" style={{ color: 'var(--text-2)' }}>
            {data.name}
          </p>
          <p className="text-xs" style={{ color: 'var(--text-3)' }}>
            Companies: {data.value}
          </p>
          <p className="text-xs font-bold" style={{ color: data.payload.color }}>
            {percentage}%
          </p>
        </div>
      );
    }
    return null;
  };

  return (
    <div
      className={`card ${inlineMobile ? '' : 'mb-6'}`}
      style={{ minWidth: inlineMobile ? '100%' : '400px' }}
    >
      <div className="flex items-center justify-between mb-4">
        <span className="card-label">Sector Holdings</span>
        <button
          onClick={fetchSectorData}
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

      {/* Custom Symbols Input */}
      <div className="mb-4 p-3 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
        <label htmlFor="sector-symbols" className="text-xs font-medium mb-2 block" style={{ color: 'var(--text-4)' }}>
          Portfolio Symbols (Optional)
        </label>
        <input
          id="sector-symbols"
          type="text"
          value={customSymbols}
          onChange={(e) => setCustomSymbols(e.target.value)}
          placeholder="e.g., AAPL,MSFT,JPM,JNJ"
          className="w-full px-3 py-2 border font-mono text-sm mb-2"
          style={{
            background: 'var(--bg-4)',
            borderColor: 'var(--bg-1)',
            color: 'var(--text-2)',
            outline: 'none'
          }}
        />
        <button
          onClick={fetchSectorData}
          disabled={loading}
          className="w-full px-3 py-2 text-xs font-semibold border transition-all disabled:opacity-50"
          style={{
            background: 'var(--accent)',
            borderColor: 'var(--accent)',
            color: 'var(--text-0)'
          }}
        >
          Analyze Portfolio
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

      {!loading && !error && sectorData.length > 0 && (
        <>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={sectorData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {sectorData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>

          {/* Sector Breakdown Table */}
          <div className="mt-4 border" style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
            <div className="p-3 border-b" style={{ borderColor: 'var(--bg-1)' }}>
              <span className="text-xs font-semibold" style={{ color: 'var(--text-4)' }}>
                Sector Breakdown
              </span>
            </div>
            <div className="p-3 space-y-2">
              {sectorData.map((sector, idx) => {
                const total = sectorData.reduce((sum, item) => sum + item.value, 0);
                const percentage = ((sector.value / total) * 100).toFixed(1);

                return (
                  <div key={idx} className="flex items-center justify-between text-xs">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 border"
                        style={{ background: sector.color, borderColor: 'var(--bg-1)' }}
                      ></div>
                      <span style={{ color: 'var(--text-3)' }}>{sector.name}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="font-mono" style={{ color: 'var(--text-4)' }}>
                        {sector.value} {sector.value === 1 ? 'company' : 'companies'}
                      </span>
                      <span className="font-bold" style={{ color: sector.color }}>
                        {percentage}%
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default SectorPieChart;
