import type { Metadata } from 'next'
import { DM_Mono } from 'next/font/google'
import './globals.css'
import './ui-pro.css'
import UIModeToggle from '@/components/UIModeToggle'

const dmMono = DM_Mono({
  weight: ['300', '400', '500'],
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-dm-mono',
})

export const metadata: Metadata = {
  title: 'Stock Predictor - Advanced Technical Analysis',
  description: 'Real-time stock analysis with technical indicators, price forecasting, and news sentiment',
}

// Default design system for first paint: `npm run dev:pro` / `build:pro` set
// NEXT_PUBLIC_UI_MODE=pro (and redirect / to the /pro workspace). After
// hydration the attribute follows the route (see UIModeToggle).
const UI_MODE = process.env.NEXT_PUBLIC_UI_MODE === 'pro' ? 'pro' : 'classic'

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={dmMono.variable} data-ui={UI_MODE} suppressHydrationWarning>
      <body className={dmMono.className}>
        {children}
        <UIModeToggle />
      </body>
    </html>
  )
}
