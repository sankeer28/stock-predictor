import type { Metadata } from 'next'

export const metadata: Metadata = {
  title: 'Stock Predictor Pro',
  description:
    'Professional stock analysis workspace — live charts, AI forecasting, technicals, market intelligence and news in one place.',
}

export default function ProLayout({ children }: { children: React.ReactNode }) {
  return children
}
