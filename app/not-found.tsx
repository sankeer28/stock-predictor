import Link from 'next/link';
import { Compass, Home } from 'lucide-react';

export default function NotFound() {
  return (
    <main
      className="min-h-screen flex items-center justify-center p-6"
      style={{ background: 'var(--bg-4)', color: 'var(--text-3)' }}
    >
      <div
        className="max-w-md w-full p-8 flex flex-col items-center text-center gap-4"
        style={{ background: 'rgba(0,0,0,0.2)', border: '2px solid var(--accent)' }}
      >
        <Compass className="w-12 h-12" style={{ color: 'var(--accent)' }} />
        <h1 className="text-3xl font-bold" style={{ color: 'var(--text-1)' }}>
          404
        </h1>
        <p className="text-sm" style={{ color: 'var(--text-4)' }}>
          That page doesn&apos;t exist. The ticker or route you&apos;re looking for may have moved.
        </p>
        <Link href="/" className="btn-primary flex items-center gap-2 mt-2">
          <Home className="w-4 h-4" /> Back to dashboard
        </Link>
      </div>
    </main>
  );
}
