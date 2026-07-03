'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Sparkles, Terminal } from 'lucide-react';

// Launch default: `npm run dev:pro` / `build:pro` set NEXT_PUBLIC_UI_MODE=pro
// (and / redirects to /pro — see next.config.js).
const DEFAULT_UI = process.env.NEXT_PUBLIC_UI_MODE === 'pro' ? 'pro' : 'classic';

/**
 * Floating switcher between the two interfaces. The pro UI is its own app
 * shell at /pro (not a restyle of the classic page), so switching modes is
 * navigation. This component also keeps <html data-ui> — which scopes the two
 * CSS design systems — in sync with the current route.
 */
export default function UIModeToggle() {
  const pathname = usePathname();

  useEffect(() => {
    const el = document.documentElement;
    if (pathname === '/pro') el.setAttribute('data-ui', 'pro');
    else if (pathname === '/') el.setAttribute('data-ui', 'classic');
    else el.setAttribute('data-ui', DEFAULT_UI);
  }, [pathname]);

  if (pathname !== '/' && pathname !== '/pro') return null;

  const toPro = pathname === '/';

  return (
    <Link
      // `?ui=classic` bypasses the pro-mode / → /pro redirect (next.config.js)
      href={toPro ? '/pro' : '/?ui=classic'}
      className="ui-mode-toggle"
      title={
        toPro
          ? 'Open the modern Pro workspace'
          : 'Back to the classic terminal interface'
      }
      aria-label={toPro ? 'Switch to Pro UI' : 'Switch to Classic UI'}
    >
      {toPro ? <Sparkles aria-hidden /> : <Terminal aria-hidden />}
      <span className="ui-mode-toggle-label">{toPro ? 'Pro UI' : 'Classic UI'}</span>
    </Link>
  );
}
