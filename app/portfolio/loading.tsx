import { Loader2 } from 'lucide-react';

export default function PortfolioLoading() {
  return (
    <main
      className="min-h-screen flex flex-col items-center justify-center gap-3"
      style={{ background: 'var(--bg-4)', color: 'var(--text-3)' }}
    >
      <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
      <p className="text-sm" style={{ color: 'var(--text-4)' }}>
        Loading portfolio…
      </p>
    </main>
  );
}
