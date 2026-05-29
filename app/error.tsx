'use client';

import { useEffect } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';
import Link from 'next/link';

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error('[App Error]', error);
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
          Something went wrong
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-4)' }}>
          An unexpected error occurred while loading this view. You can try again or head back to the
          dashboard.
        </p>
        {error?.digest && (
          <code className="text-xs px-2 py-1" style={{ background: 'var(--bg-3)', color: 'var(--text-5)' }}>
            ref: {error.digest}
          </code>
        )}
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
