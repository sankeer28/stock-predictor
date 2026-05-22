import { NextRequest, NextResponse } from 'next/server';

const YF_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
  'Accept': 'application/json',
  'Accept-Language': 'en-US,en;q=0.9',
};

// GET /api/portfolio-quote?symbols=AAPL,MSFT,VOO
export async function GET(request: NextRequest) {
  const raw     = request.nextUrl.searchParams.get('symbols') ?? '';
  const symbols = raw.split(',').map(s => s.trim().toUpperCase()).filter(Boolean).slice(0, 30);
  if (!symbols.length) return NextResponse.json({ error: 'symbols required' }, { status: 400 });

  const FINNHUB_KEY = process.env.FINN_HUB;
  const now         = Math.floor(Date.now() / 1000);

  // Fetch price+name (Yahoo) and dividends (Finnhub) in parallel per symbol
  const results = await Promise.allSettled(
    symbols.map(async (sym) => {
      // Yahoo chart meta — price + name
      const yahooUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${sym}?period1=${now - 86400 * 7}&period2=${now}&interval=1d`;
      const yahooRes = await fetch(yahooUrl, {
        headers: YF_HEADERS,
        cache: 'no-store',
        signal: AbortSignal.timeout(10000),
      });
      const yahooData = await yahooRes.json();
      const meta = yahooData.chart?.result?.[0]?.meta ?? {};

      const price = meta.regularMarketPrice ?? null;
      const name  = meta.longName || meta.shortName || sym;

      // Finnhub metric — dividend yield + rate
      let dividendYield: number | null = null;
      let dividendRate:  number | null = null;

      if (FINNHUB_KEY) {
        try {
          const fhUrl = `https://finnhub.io/api/v1/stock/metric?symbol=${sym}&metric=all&token=${FINNHUB_KEY}`;
          const fhRes = await fetch(fhUrl, {
            headers: { 'Accept': 'application/json' },
            cache: 'no-store',
            signal: AbortSignal.timeout(8000),
          });
          if (fhRes.ok) {
            const fhData = await fhRes.json();
            const m = fhData.metric ?? {};
            // dividendYieldIndicatedAnnual is in percent (e.g. 0.53 = 0.53%)
            if (m.dividendYieldIndicatedAnnual != null) {
              dividendYield = m.dividendYieldIndicatedAnnual / 100;
            }
            // annualDividends: total dividends paid per share in last 12 months
            if (m.dividendPerShareAnnual != null) {
              dividendRate = m.dividendPerShareAnnual;
            } else if (dividendYield != null && price != null) {
              dividendRate = dividendYield * price;
            }
          }
        } catch { /* skip dividends on error */ }
      }

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
