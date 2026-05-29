'use client';

// Catches errors thrown in the root layout itself. Must render its own
// <html>/<body> because it replaces the root layout when triggered.
import { AlertTriangle, RefreshCw } from 'lucide-react';

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          background: '#0f0f0f',
          color: '#bfbfbf',
          fontFamily: "'Courier New', monospace",
          minHeight: '100vh',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          margin: 0,
          padding: '24px',
        }}
      >
        <div
          style={{
            maxWidth: '420px',
            width: '100%',
            padding: '32px',
            textAlign: 'center',
            border: '2px solid #d9534f',
            background: 'rgba(0,0,0,0.3)',
          }}
        >
          <AlertTriangle style={{ width: 48, height: 48, color: '#d9534f', margin: '0 auto 16px' }} />
          <h1 style={{ fontSize: '20px', color: '#f0f0f0', marginBottom: '8px' }}>
            Application error
          </h1>
          <p style={{ fontSize: '13px', color: '#8a8a8a', marginBottom: '20px' }}>
            A critical error occurred. Reloading usually resolves it.
          </p>
          <button
            onClick={reset}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '8px',
              padding: '8px 16px',
              border: '1px solid #2e8b6f',
              background: 'transparent',
              color: '#3cb38e',
              cursor: 'pointer',
              fontFamily: 'inherit',
            }}
          >
            <RefreshCw style={{ width: 16, height: 16 }} /> Reload
          </button>
        </div>
      </body>
    </html>
  );
}
