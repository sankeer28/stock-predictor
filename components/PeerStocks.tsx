'use client';

import React, { useState, useEffect } from 'react';
import { Users, Loader2, RefreshCw } from 'lucide-react';

interface PeerStocksProps {
  symbol: string;
  onPeerClick?: (symbol: string) => void;
  inlineMobile?: boolean;
}

export default function PeerStocks({ symbol, onPeerClick, inlineMobile }: PeerStocksProps) {
  const [peers, setPeers] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchPeers = async () => {
    try {
      setLoading(true);
      setError('');
      const response = await fetch(`/api/peers?symbol=${symbol}`);
      const data = await response.json();

      if (data.success) {
        setPeers(data.peers.filter((p: string) => p !== symbol)); // Exclude current symbol
      } else {
        setError(data.error || 'Failed to fetch peer stocks');
      }
    } catch (err: any) {
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (symbol) {
      fetchPeers();
    }
  }, [symbol]);

  return (
    <div className={`card ${inlineMobile ? 'w-full' : 'w-80'}`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Users className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Similar Stocks</span>
        </div>

        <button
          onClick={() => fetchPeers()}
          disabled={loading}
          className="p-2 transition-all border disabled:opacity-50"
          style={{
            background: 'var(--bg-3)',
            borderColor: 'var(--bg-1)',
            color: 'var(--text-3)',
          }}
          title="Refresh data"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {error && (
        <div className="mb-4 p-3 border-2" style={{
          background: 'var(--bg-2)',
          borderColor: 'var(--danger)',
          color: 'var(--danger)'
        }}>
          <p className="text-sm">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : (
        <>
          <div className="mb-3">
            <p className="text-xs" style={{ color: 'var(--text-4)' }}>
              Stocks in the same sector as <span className="font-mono font-semibold" style={{ color: 'var(--text-2)' }}>{symbol}</span>
            </p>
          </div>

          <div className="flex flex-wrap gap-2">
            {peers.map((peer, index) => (
              <button
                key={peer}
                onClick={() => onPeerClick?.(peer)}
                className="px-3 py-2 border transition-all hover:opacity-80 font-mono font-semibold text-sm"
                style={{
                  background: 'var(--bg-2)',
                  borderColor: 'var(--bg-1)',
                  color: 'var(--text-2)',
                  borderLeftWidth: '3px',
                  borderLeftColor: 'var(--accent)',
                }}
              >
                {peer}
              </button>
            ))}
          </div>

          {peers.length === 0 && !loading && (
            <div className="text-center py-8" style={{ color: 'var(--text-4)' }}>
              <p className="text-sm">No peer stocks found</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
