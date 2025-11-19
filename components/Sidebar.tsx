'use client';

import React, { useState } from 'react';
import { History, TrendingUp, Menu, Trash2 } from 'lucide-react';
import PredictionsCache from './PredictionsCache';
import { CachedPrediction } from '@/lib/predictionsCache';

export interface SearchHistoryItem {
  symbol: string;
  timestamp: number;
  price?: number;
  companyName?: string;
}

interface SidebarProps {
  searchHistory: SearchHistoryItem[];
  onSelectSymbol: (symbol: string) => void;
  onClearHistory: () => void;
  currentSymbol?: string;
  onLoadCachedPrediction?: (prediction: CachedPrediction) => void;
  // When true the sidebar will render inline for mobile (no fixed toggle/overlay)
  inlineMobile?: boolean;
}

export default function Sidebar({
  searchHistory,
  onSelectSymbol,
  onClearHistory,
  currentSymbol,
  onLoadCachedPrediction
  , inlineMobile
}: SidebarProps) {
  const [isOpen, setIsOpen] = useState<boolean>(!!inlineMobile);

  return (
    <>
      {/* Mobile Toggle Button (only when not rendering inline) */}
      {!inlineMobile && (
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-3 border-2 xl:hidden"
          style={{
            position: 'fixed',
            left: '1rem',
            top: '1rem',
            zIndex: 50,
            background: 'var(--bg-4)',
            borderColor: 'var(--accent)',
            color: 'var(--accent)'
          }}
        >
          <Menu className="w-5 h-5" />
        </button>
      )}

      {/* Sidebar */}
      <div
        className={`card ${inlineMobile ? 'w-full' : 'w-64'} transition-transform xl:translate-x-0 ${
          isOpen ? 'translate-x-0' : (inlineMobile ? '' : '-translate-x-[calc(100%+1rem)]')
        } xl:block`}
        style={{
          position: inlineMobile ? 'relative' : (isOpen ? 'fixed' : 'relative'),
          left: isOpen && !inlineMobile ? 0 : 'auto',
          top: isOpen && !inlineMobile ? 0 : 'auto',
          zIndex: isOpen && !inlineMobile ? 40 : 'auto',
          maxHeight: isOpen && !inlineMobile ? '100vh' : 'none',
          overflowY: isOpen && !inlineMobile ? 'auto' : 'visible',
        }}
      >
        <span className="card-label">Search History</span>

        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <History className="w-5 h-5" style={{ color: 'var(--accent)' }} />
            <h2 className="font-semibold" style={{ color: 'var(--text-2)' }}>Recent</h2>
          </div>
          {searchHistory.length > 0 && (
            <button
              onClick={onClearHistory}
              className="hover:opacity-80 transition-all"
              title="Clear history"
              style={{ background: 'none', border: 'none', padding: 0 }}
            >
              <Trash2 className="w-4 h-4" style={{ color: 'var(--error)' }} />
            </button>
          )}
        </div>

      <div className="space-y-2">
        {searchHistory.length === 0 ? (
          <div className="text-center py-8">
            <TrendingUp className="w-8 h-8 mx-auto mb-2 opacity-30" style={{ color: 'var(--text-4)' }} />
            <p className="text-xs" style={{ color: 'var(--text-4)' }}>
              No search history yet
            </p>
          </div>
        ) : (
          searchHistory.map((item, index) => (
            <button
              key={`${item.symbol}-${item.timestamp}-${index}`}
              onClick={() => {
                onSelectSymbol(item.symbol);
                setIsOpen(false);
              }}
              className="w-full p-3 border-2 text-left transition-all hover:border-l-4"
              style={{
                background: item.symbol === currentSymbol ? 'var(--bg-2)' : 'var(--bg-3)',
                borderColor: item.symbol === currentSymbol ? 'var(--accent)' : 'var(--bg-1)',
                borderLeftColor: item.symbol === currentSymbol ? 'var(--accent)' : 'var(--bg-1)',
                borderLeftWidth: item.symbol === currentSymbol ? '3px' : '2px'
              }}
            >
              <div className="flex items-center justify-between">
                <span className="font-bold font-mono" style={{ color: 'var(--text-2)' }}>
                  {item.symbol}
                </span>
                {item.price && (
                  <span className="text-xs font-mono" style={{ color: 'var(--accent)' }}>
                    ${item.price.toFixed(2)}
                  </span>
                )}
              </div>
            </button>
          ))
        )}
      </div>
    </div>

    {/* Predictions Cache - Separate Container Below Search History */}
    <div
      className={`${inlineMobile ? 'w-full' : 'w-64'} transition-transform xl:translate-x-0 ${
        isOpen ? 'translate-x-0' : (inlineMobile ? '' : '-translate-x-[calc(100%+1rem)]')
      } xl:block mt-6`}
      style={{
        position: inlineMobile ? 'relative' : (isOpen ? 'fixed' : 'relative'),
        left: isOpen && !inlineMobile ? 0 : 'auto',
        top: isOpen && !inlineMobile ? '50vh' : 'auto',
        zIndex: isOpen && !inlineMobile ? 40 : 'auto',
        maxHeight: isOpen && !inlineMobile ? '50vh' : 'none',
        overflowY: isOpen && !inlineMobile ? 'auto' : 'visible',
      }}
    >
      <PredictionsCache onLoadPrediction={(pred) => {
        onLoadCachedPrediction?.(pred);
        setIsOpen(false); // Close sidebar on mobile after loading
      }} />
    </div>

    {/* Mobile Overlay */}
    {/* Mobile Overlay (only when not inline) */}
    {!inlineMobile && isOpen && (
      <div
        className="fixed inset-0 bg-black bg-opacity-50 z-30 xl:hidden"
        onClick={() => setIsOpen(false)}
      />
    )}
    </>
  );
}
