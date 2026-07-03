'use client';

import ProDashboard from '@/components/pro/ProDashboard';

export default function ProPage() {
  return (
    <>
      {/* Pin the pro design system before first paint, regardless of which
          mode the server rendered as the default (see UIModeToggle for the
          route-driven sync after hydration). */}
      <script
        dangerouslySetInnerHTML={{
          __html: "document.documentElement.setAttribute('data-ui','pro');",
        }}
      />
      <ProDashboard />
    </>
  );
}
