import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const symbol = request.nextUrl.searchParams.get('symbol');
  if (!symbol) return NextResponse.json({ error: 'Missing symbol' }, { status: 400 });

  try {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1m&range=1d`;
    const res = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
      },
      next: { revalidate: 0 },
    });

    if (!res.ok) throw new Error(`Yahoo returned ${res.status}`);

    const json = await res.json();
    const meta = json?.chart?.result?.[0]?.meta;
    if (!meta) throw new Error('No meta in response');

    const currentPrice: number = meta.regularMarketPrice ?? meta.chartPreviousClose;
    const prevClose: number = meta.regularMarketPreviousClose ?? meta.chartPreviousClose;
    const change = prevClose ? currentPrice - prevClose : 0;
    const changePercent = prevClose ? (change / prevClose) * 100 : 0;

    return NextResponse.json({
      symbol: meta.symbol,
      price: currentPrice,
      change,
      changePercent,
      marketState: meta.marketState,
      timestamp: Date.now(),
    }, {
      headers: { 'Cache-Control': 'no-store' },
    });
  } catch (err: any) {
    return NextResponse.json({ error: err.message }, { status: 500 });
  }
}
