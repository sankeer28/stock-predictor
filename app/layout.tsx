import type { Metadata } from 'next'
import { DM_Mono } from 'next/font/google'
import './globals.css'

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

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={dmMono.variable}>
      <body className={dmMono.className}>{children}</body>
    </html>
  )
}
