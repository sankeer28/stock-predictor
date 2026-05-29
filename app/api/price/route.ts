import { NextRequest, NextResponse } from 'next/server';
import { fetchWithRetry } from '@/lib/serverFetch';

const ENDPOINTS = (symbol: string) => [
  `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1m&range=1d`,
  `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1m&range=1d`,
];

export async function GET(request: NextRequest) {
  const symbol = request.nextUrl.searchParams.get('symbol');
  if (!symbol) return NextResponse.json({ error: 'Missing symbol' }, { status: 400 });

  for (const url of ENDPOINTS(symbol)) {
    try {
      const res = await fetchWithRetry(
        url,
        {
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
          },
          next: { revalidate: 0 },
        },
        { retries: 1, timeoutMs: 8000 } // one backoff retry; falls through to the next endpoint otherwise
      );

      if (!res.ok) continue;

      const json = await res.json();
      const result = json?.chart?.result?.[0];
      const meta = result?.meta;
      if (!meta) continue;

      const currentPrice: number = meta.regularMarketPrice ?? meta.chartPreviousClose;
      if (!currentPrice || currentPrice <= 0) continue;

      const prevClose: number = meta.regularMarketPreviousClose ?? meta.chartPreviousClose ?? currentPrice;
      const change = prevClose ? currentPrice - prevClose : 0;
      const changePercent = prevClose ? (change / prevClose) * 100 : 0;

      // Volume: second-to-last bar avoids partially-completed current bar noise
      const rawVols: number[] = (result?.indicators?.quote?.[0]?.volume ?? [])
        .filter((v: any) => typeof v === 'number' && v > 0);
      const currentVolume = rawVols.length >= 2
        ? rawVols[rawVols.length - 2]
        : rawVols.length === 1 ? rawVols[0] : null;

      // Use 8-bar average for more stable baseline (vs old 5-bar)
      const prevVols = rawVols.slice(-9, -1);
      const avgVolume = prevVols.length >= 3
        ? prevVols.reduce((a, b) => a + b, 0) / prevVols.length
        : null;

      // Data freshness — how many ms since the last recorded bar
      const timestamps: number[] = result?.timestamp ?? [];
      const lastBarTs = timestamps.length > 0 ? timestamps[timestamps.length - 1] * 1000 : null;
      const dataAgeMs = lastBarTs ? Date.now() - lastBarTs : null;

      return NextResponse.json({
        symbol: meta.symbol,
        price: currentPrice,
        change,
        changePercent,
        marketState: meta.marketState,
        volume: currentVolume,
        avgVolume5: avgVolume,   // kept as avgVolume5 for backwards-compat
        dataAgeMs,
        timestamp: Date.now(),
      }, {
        headers: { 'Cache-Control': 'no-store' },
      });
    } catch { continue; }
  }

  return NextResponse.json({ error: 'Failed to fetch price data' }, { status: 500 });
}
