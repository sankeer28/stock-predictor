import { NextResponse } from 'next/server';

const YF_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
  'Accept': 'application/json',
  'Accept-Language': 'en-US,en;q=0.9',
};

async function fetchScreener(scrId: string, count = 6): Promise<any[]> {
  const url = `https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=${scrId}&count=${count}&fields=shortName,regularMarketPrice,regularMarketChange,regularMarketChangePercent,regularMarketVolume,marketCap`;
  try {
    const res = await fetch(url, {
      headers: YF_HEADERS,
      signal: AbortSignal.timeout(8000),
      cache: 'no-store',
    });
    if (!res.ok) return [];
    const json = await res.json();
    return json?.finance?.result?.[0]?.quotes ?? [];
  } catch {
    return [];
  }
}

async function fetchTrending(): Promise<string[]> {
  try {
    const res = await fetch('https://query1.finance.yahoo.com/v1/finance/trending/US?count=8', {
      headers: YF_HEADERS,
      signal: AbortSignal.timeout(8000),
      cache: 'no-store',
    });
    if (!res.ok) return [];
    const json = await res.json();
    const quotes: any[] = json?.finance?.result?.[0]?.quotes ?? [];
    return quotes.map((q: any) => q.symbol).filter(Boolean);
  } catch {
    return [];
  }
}

async function fetchQuotes(symbols: string[]): Promise<any[]> {
  if (!symbols.length) return [];
  const syms = symbols.join(',');
  try {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${syms}&fields=shortName,regularMarketPrice,regularMarketChange,regularMarketChangePercent,regularMarketVolume,marketCap`,
      {
        headers: YF_HEADERS,
        signal: AbortSignal.timeout(8000),
        cache: 'no-store',
      }
    );
    if (!res.ok) return [];
    const json = await res.json();
    return json?.quoteResponse?.result ?? [];
  } catch {
    return [];
  }
}

function mapQuote(q: any) {
  return {
    symbol: q.symbol ?? '',
    name: q.shortName ?? q.longName ?? q.symbol ?? '',
    price: q.regularMarketPrice ?? null,
    change: q.regularMarketChange ?? null,
    changePercent: q.regularMarketChangePercent ?? null,
    volume: q.regularMarketVolume ?? null,
    marketCap: q.marketCap ?? null,
  };
}

export async function GET() {
  try {
    const [gainers, losers, trendingSymbols] = await Promise.all([
      fetchScreener('day_gainers', 6),
      fetchScreener('day_losers', 6),
      fetchTrending(),
    ]);

    const trendingQuotes = await fetchQuotes(trendingSymbols.slice(0, 6));

    const indices = ['SPY', 'QQQ', 'DIA', 'IWM', '^VIX'];
    const indexQuotes = await fetchQuotes(indices);

    return NextResponse.json({
      gainers: gainers.map(mapQuote),
      losers: losers.map(mapQuote),
      trending: trendingQuotes.map(mapQuote),
      indices: indexQuotes.map(mapQuote),
    }, {
      headers: { 'Cache-Control': 'public, s-maxage=60, stale-while-revalidate=30' },
    });
  } catch (error: any) {
    return NextResponse.json({ error: error.message || 'Failed to fetch market movers' }, { status: 500 });
  }
}
