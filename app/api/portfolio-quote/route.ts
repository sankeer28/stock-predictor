import { NextRequest, NextResponse } from 'next/server';

const YF_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
  'Accept': 'application/json',
  'Accept-Language': 'en-US,en;q=0.9',
};

// GET /api/portfolio-quote?symbols=AAPL,MSFT,VFV.TO
export async function GET(request: NextRequest) {
  const raw     = request.nextUrl.searchParams.get('symbols') ?? '';
  const symbols = raw.split(',').map(s => s.trim().toUpperCase()).filter(Boolean).slice(0, 30);
  if (!symbols.length) return NextResponse.json({ error: 'symbols required' }, { status: 400 });

  const now      = Math.floor(Date.now() / 1000);
  const oneYrAgo = now - 86400 * 370; // slightly over 1yr to catch all trailing dividends

  const results = await Promise.allSettled(
    symbols.map(async (sym) => {
      // Fetch 1yr of daily data + dividend events in one request
      const url = `https://query1.finance.yahoo.com/v8/finance/chart/${sym}` +
        `?period1=${oneYrAgo}&period2=${now}&interval=1d&events=dividends`;

      const res  = await fetch(url, {
        headers: YF_HEADERS,
        cache: 'no-store',
        signal: AbortSignal.timeout(10000),
      });
      const data = await res.json();
      const result = data.chart?.result?.[0];
      const meta   = result?.meta ?? {};

      const price = meta.regularMarketPrice ?? null;
      const name  = meta.longName || meta.shortName || sym;

      // Sum dividends paid in the trailing 12 months
      const divEvents: Record<string, { amount: number; date: number }> =
        result?.events?.dividends ?? {};

      const cutoff = now - 86400 * 365;
      let dividendRate: number | null = null;

      const recentDivs = Object.values(divEvents).filter(d => d.date >= cutoff);
      if (recentDivs.length > 0) {
        dividendRate = recentDivs.reduce((s, d) => s + d.amount, 0);
      }

      const dividendYield = dividendRate != null && price && price > 0
        ? dividendRate / price
        : null;

      return { symbol: sym, name, price, dividendYield, dividendRate };
    })
  );

  const quotes = results.map((r, i) =>
    r.status === 'fulfilled'
      ? r.value
      : { symbol: symbols[i], name: symbols[i], price: null, dividendYield: null, dividendRate: null }
  );

  return NextResponse.json({ quotes });
}
