import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const maxDuration = 30;

const YF_URL =
  'https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved';

function fmtVolume(v: number): string {
  if (v >= 1e9) return (v / 1e9).toFixed(1) + 'B';
  if (v >= 1e6) return (v / 1e6).toFixed(1) + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'K';
  return String(v);
}


function fmtPct(v: number): string {
  return (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
}

const HEADERS = [
  'Ticker', 'Price', 'Change', 'Volume', 'P/E', 'P/B', 'EPS', 'Div Yield', '52W High', '52W Low', '52W Chg',
];

export async function GET(request: NextRequest) {
  const sp    = request.nextUrl.searchParams;
  const scrId = (sp.get('scrId') || 'most_actives').trim();
  const count = 250;

  try {
    const url = `${YF_URL}?scrIds=${encodeURIComponent(scrId)}&count=${count}&offset=0&lang=en-US&region=US`;

    const res = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        Accept: 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
      },
      cache: 'no-store',
      signal: AbortSignal.timeout(9000),
    });

    if (!res.ok) throw new Error(`Yahoo Finance returned ${res.status}`);

    const data   = await res.json();
    const result = data?.finance?.result?.[0];
    if (!result) throw new Error('No screener data returned');

    const total  = result.total ?? 0;
    const quotes: any[] = result.quotes ?? [];

    const rows = quotes.map((q: any) => ({
      Ticker:      q.symbol || '',
      Price:       q.regularMarketPrice          != null ? `$${Number(q.regularMarketPrice).toFixed(2)}`          : '-',
      Change:      q.regularMarketChangePercent  != null ? fmtPct(q.regularMarketChangePercent)                   : '-',
      Volume:      q.regularMarketVolume         != null ? fmtVolume(q.regularMarketVolume)                       : '-',
      'P/E':       q.trailingPE                 != null ? Number(q.trailingPE).toFixed(1)                        : '-',
      'P/B':       q.priceToBook                != null ? Number(q.priceToBook).toFixed(2)                       : '-',
      'EPS':       q.epsTrailingTwelveMonths    != null ? `$${Number(q.epsTrailingTwelveMonths).toFixed(2)}`     : '-',
      'Div Yield': q.trailingAnnualDividendYield != null ? (Number(q.trailingAnnualDividendYield) * 100).toFixed(2) + '%' : '-',
      '52W High':  q.fiftyTwoWeekHigh           != null ? `$${Number(q.fiftyTwoWeekHigh).toFixed(2)}`           : '-',
      '52W Low':   q.fiftyTwoWeekLow            != null ? `$${Number(q.fiftyTwoWeekLow).toFixed(2)}`            : '-',
      '52W Chg':   q.fiftyTwoWeekChangePercent  != null ? fmtPct(q.fiftyTwoWeekChangePercent)                   : '-',
    }));

    return NextResponse.json(
      {
        success: true,
        scrId,
        total: rows.length,
        headers: HEADERS,
        rows,
      },
      { headers: { 'Cache-Control': 'no-store' } }
    );
  } catch (err: any) {
    console.error('[screener]', err);
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
