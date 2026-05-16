'use client';

import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, RefreshCw } from 'lucide-react';

interface FGEntry {
  value: string;
  value_classification: string;
  timestamp: string;
}

interface FGData {
  name: string;
  data: FGEntry[];
  metadata: { error: string | null };
}

const ZONES = [
  { label: 'Ext. Fear', range: [0, 25],   hex: '#ef4444' },
  { label: 'Fear',      range: [25, 45],  hex: '#f97316' },
  { label: 'Neutral',   range: [45, 55],  hex: '#facc15' },
  { label: 'Greed',     range: [55, 75],  hex: '#86efac' },
  { label: 'Ext. Greed',range: [75, 100], hex: '#22c55e' },
];

const CLASSIFICATION_CONFIG: Record<string, { hex: string; icon: React.ReactNode }> = {
  'Extreme Fear':  { hex: '#ef4444', icon: <TrendingDown className="w-3 h-3" /> },
  'Fear':          { hex: '#f97316', icon: <TrendingDown className="w-3 h-3" /> },
  'Neutral':       { hex: '#facc15', icon: <Minus className="w-3 h-3" /> },
  'Greed':         { hex: '#86efac', icon: <TrendingUp className="w-3 h-3" /> },
  'Extreme Greed': { hex: '#22c55e', icon: <TrendingUp className="w-3 h-3" /> },
};

function getConfig(classification: string) {
  return CLASSIFICATION_CONFIG[classification] ?? CLASSIFICATION_CONFIG['Neutral'];
}

function hexWithAlpha(hex: string, alpha: number) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

export default function FearGreedIndex() {
  const [data, setData] = useState<FGData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchData = async () => {
    setLoading(true);
    setError('');
    try {
      const res = await fetch('/api/fear-greed', { cache: 'no-store' });
      if (!res.ok) throw new Error('Failed to fetch');
      const json = await res.json();
      setData(json);
      setLastUpdated(new Date());
    } catch (e: any) {
      setError(e.message || 'Failed');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, []);

  const current = data?.data?.[0];
  const value   = current ? parseInt(current.value, 10) : 0;
  const classification = current?.value_classification ?? 'Neutral';
  const cfg = getConfig(classification);
  const history = (data?.data ?? []).slice(1, 7).reverse();

  return (
      <div className="card">
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <span className="card-label" style={{ marginBottom: 0 }}>Fear &amp; Greed</span>
      </div>

      {loading && !data && (
        <div className="flex items-center justify-center py-6">
          <RefreshCw className="w-5 h-5 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      )}

      {error && (
        <p className="text-xs py-3 text-center" style={{ color: 'var(--danger)' }}>{error}</p>
      )}

      {current && (
        <div className="flex flex-col gap-3">

          {/* Score row */}
          <div className="flex items-center justify-between">
            <div className="flex items-end gap-2">
              <span
                className="text-5xl font-black leading-none tracking-tighter"
                style={{ color: cfg.hex }}
              >
                {value}
              </span>
              <span className="text-xs font-semibold mb-1" style={{ color: 'var(--text-5)' }}>/100</span>
            </div>

            <div
              className="flex items-center gap-1.5 px-2.5 py-1 border text-[10px] font-bold"
              style={{
                color: cfg.hex,
                background: hexWithAlpha(cfg.hex, 0.1),
                borderColor: hexWithAlpha(cfg.hex, 0.35),
              }}
            >
              {cfg.icon}
              <span>{classification.toUpperCase()}</span>
            </div>
          </div>

          {/* Gradient track + marker */}
          <div className="relative">
            <div
              className="h-2 w-full rounded-full"
              style={{
                background: 'linear-gradient(to right, #ef4444 0%, #f97316 25%, #facc15 50%, #86efac 75%, #22c55e 100%)',
              }}
            />
            {/* Marker */}
            <div
              className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full border-2 border-white"
              style={{
                left: `${value}%`,
                background: cfg.hex,
              }}
            />
          </div>

          {/* Zone labels */}
          <div className="flex justify-between mt-1">
            {ZONES.map((z) => (
              <span key={z.label} className="text-[7px] font-medium" style={{ color: 'var(--text-5)' }}>
                {z.label}
              </span>
            ))}
          </div>

          {/* 7-day history */}
          {history.length > 0 && (
            <div className="pt-2 border-t" style={{ borderColor: 'var(--bg-1)' }}>
              <div className="flex items-end justify-between gap-1">
                {history.map((entry, i) => {
                  const v = parseInt(entry.value, 10);
                  const c = getConfig(entry.value_classification);
                  const date = new Date(parseInt(entry.timestamp) * 1000);
                  const day = date.toLocaleDateString('en-US', { weekday: 'short' });
                  const barH = Math.max(6, (v / 100) * 28);
                  return (
                    <div
                      key={i}
                      className="flex flex-col items-center gap-0.5 flex-1"
                      title={`${day}: ${entry.value_classification} (${v})`}
                    >
                      <span className="text-[8px] font-bold" style={{ color: c.hex }}>{v}</span>
                      <div
                        className="w-full rounded-sm"
                        style={{ height: `${barH}px`, background: c.hex, opacity: 0.7 }}
                      />
                      <span className="text-[7px]" style={{ color: 'var(--text-5)' }}>{day}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {lastUpdated && (
            <p className="text-[9px] -mt-1" style={{ color: 'var(--text-5)' }}>
              Updated {lastUpdated.toLocaleTimeString()}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
