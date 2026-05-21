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

function fmtCap(v: number): string {
  if (v >= 1e12) return '$' + (v / 1e12).toFixed(2) + 'T';
  if (v >= 1e9)  return '$' + (v / 1e9).toFixed(1)  + 'B';
  if (v >= 1e6)  return '$' + (v / 1e6).toFixed(0)  + 'M';
  return '$' + v;
}

function fmtPct(v: number): string {
  return (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
}

const HEADERS = ['Ticker', 'Company', 'Price', 'Change', 'Volume', 'Mkt Cap', 'Fwd P/E', '52W Chg'];

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
      Ticker:    q.symbol     || '',
      Company:   q.shortName  || q.displayName || q.longName || '',
      Price:     q.regularMarketPrice          != null ? `$${Number(q.regularMarketPrice).toFixed(2)}`       : '-',
      Change:    q.regularMarketChangePercent  != null ? fmtPct(q.regularMarketChangePercent)                : '-',
      Volume:    q.regularMarketVolume         != null ? fmtVolume(q.regularMarketVolume)                    : '-',
      'Mkt Cap': q.marketCap                  != null ? fmtCap(q.marketCap)                                 : '-',
      'Fwd P/E': q.forwardPE                  != null ? Number(q.forwardPE).toFixed(1)                      : '-',
      '52W Chg': q.fiftyTwoWeekChangePercent  != null ? fmtPct(q.fiftyTwoWeekChangePercent)                 : '-',
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
