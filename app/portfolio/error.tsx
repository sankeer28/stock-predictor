'use client';

import { useEffect } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import Link from 'next/link';

export default function PortfolioError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('[Portfolio Error]', error);
  }, [error]);

  return (
    <main
      className="min-h-screen flex items-center justify-center p-6"
      style={{ background: 'var(--bg-4)', color: 'var(--text-3)' }}
    >
      <div
        className="max-w-md w-full p-8 flex flex-col items-center text-center gap-4"
        style={{ background: 'rgba(0,0,0,0.2)', border: '2px solid var(--danger)' }}
      >
        <AlertTriangle className="w-12 h-12" style={{ color: 'var(--danger)' }} />
        <h1 className="text-xl font-semibold" style={{ color: 'var(--text-1)' }}>
          Portfolio failed to load
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-4)' }}>
          Your saved holdings are stored locally and are safe. Try reloading the portfolio.
        </p>
        <div className="flex items-center gap-3 mt-2">
          <button onClick={reset} className="btn-primary flex items-center gap-2">
            <RefreshCw className="w-4 h-4" /> Try again
          </button>
          <Link href="/" className="btn-secondary flex items-center gap-2">
            <Home className="w-4 h-4" /> Dashboard
          </Link>
        </div>
      </div>
    </main>
  );
}
