'use client';

import React from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: React.ReactNode;
  /** Short name of the panel, shown in the fallback message. */
  label?: string;
  /** Custom fallback UI; overrides the default panel-error card. */
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
}

/**
 * Wrap an individual panel so a crash inside it (or a downstream data error)
 * shows a small inline fallback instead of taking down the whole page.
 */
export class ErrorBoundary extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error(`[ErrorBoundary${this.props.label ? ` · ${this.props.label}` : ''}]`, error, info);
  }

  reset = () => this.setState({ hasError: false, error: undefined });

  render() {
    if (!this.state.hasError) return this.props.children;
    if (this.props.fallback) return this.props.fallback;

    return (
      <div
        className="p-4 flex flex-col items-center justify-center text-center gap-2"
        style={{
          background: 'var(--bg-3)',
          border: '1px solid var(--danger)',
          minHeight: '120px',
        }}
      >
        <AlertTriangle className="w-6 h-6" style={{ color: 'var(--danger)' }} />
        <p className="text-sm" style={{ color: 'var(--text-2)' }}>
          {this.props.label ? `${this.props.label} failed to load` : 'This panel failed to load'}
        </p>
        <p className="text-xs" style={{ color: 'var(--text-4)' }}>
          The rest of the page is unaffected.
        </p>
        <button
          onClick={this.reset}
          className="mt-1 flex items-center gap-1.5 px-3 py-1 text-xs border transition-all"
          style={{ background: 'var(--bg-4)', borderColor: 'var(--bg-1)', color: 'var(--text-3)' }}
        >
          <RefreshCw className="w-3 h-3" /> Retry
        </button>
      </div>
    );
  }
}

export default ErrorBoundary;
