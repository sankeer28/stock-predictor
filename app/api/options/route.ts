import { NextRequest, NextResponse } from 'next/server';

// NASDAQ public options API — no auth required
const NASDAQ_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
  'Accept': 'application/json, text/plain, */*',
  'Accept-Language': 'en-US,en;q=0.9',
  'Origin': 'https://www.nasdaq.com',
  'Referer': 'https://www.nasdaq.com/',
};

function parseNum(v: string | null | undefined): number | null {
  if (!v || v === '--' || v === 'N/A') return null;
  const n = parseFloat(v.replace(/,/g, ''));
  return isNaN(n) ? null : n;
}

export async function GET(request: NextRequest) {
  const symbol = request.nextUrl.searchParams.get('symbol');
  const expiry = request.nextUrl.searchParams.get('expiry') ?? '';

  if (!symbol) {
    return NextResponse.json({ error: 'Symbol required' }, { status: 400 });
  }

  const sym = symbol.toUpperCase();

  try {
    // Fetch both calls and puts for the selected (or nearest) expiry
    const expiryParam = expiry ? `&expirynode=${encodeURIComponent(expiry)}` : '&expirynode=1m';

    const [callRes, putRes, expiryRes] = await Promise.all([
      fetch(
        `https://api.nasdaq.com/api/quote/${sym}/option-chain?assetclass=stocks${expiryParam}&callput=call&type=all&limit=20&offset=0`,
        { headers: NASDAQ_HEADERS, signal: AbortSignal.timeout(10000), cache: 'no-store' }
      ),
      fetch(
        `https://api.nasdaq.com/api/quote/${sym}/option-chain?assetclass=stocks${expiryParam}&callput=put&type=all&limit=20&offset=0`,
        { headers: NASDAQ_HEADERS, signal: AbortSignal.timeout(10000), cache: 'no-store' }
      ),
      fetch(
        `https://api.nasdaq.com/api/quote/${sym}/option-chain?assetclass=stocks&expirynode=1m&callput=call&type=all&limit=1&offset=0`,
        { headers: NASDAQ_HEADERS, signal: AbortSignal.timeout(8000), cache: 'no-store' }
      ),
    ]);

    if (!callRes.ok && !putRes.ok) {
      // Fallback: try ETF asset class
      const callRes2 = await fetch(
        `https://api.nasdaq.com/api/quote/${sym}/option-chain?assetclass=etf${expiryParam}&callput=call&type=all&limit=20&offset=0`,
        { headers: NASDAQ_HEADERS, signal: AbortSignal.timeout(10000), cache: 'no-store' }
      );
      if (!callRes2.ok) {
        throw new Error(`NASDAQ options API returned ${callRes.status}. Options may not be available for ${sym}.`);
      }
    }

    const callJson = callRes.ok ? await callRes.json() : null;
    const putJson  = putRes.ok  ? await putRes.json()  : null;
    const expiryJson = expiryRes.ok ? await expiryRes.json() : null;

    // Parse calls
    const callRows: any[] = callJson?.data?.table?.rows ?? [];
    const calls = callRows
      .filter(r => r.strike && r.c_Last !== undefined)
      .map(r => {
        const strike = parseNum(r.strike);
        if (strike === null) return null;
        return {
          strike,
          lastPrice:        parseNum(r.c_Last),
          bid:              parseNum(r.c_Bid),
          ask:              parseNum(r.c_Ask),
          volume:           parseNum(r.c_Volume),
          openInterest:     parseNum(r.c_Openinterest),
          impliedVolatility: parseNum(r.c_IV) ? (parseNum(r.c_IV)! / 100) : null,
          inTheMoney:       r.c_ITM === 'Y' || r.c_InTheMoney === 'Yes',
        };
      })
      .filter(Boolean);

    // Parse puts
    const putRows: any[] = putJson?.data?.table?.rows ?? [];
    const puts = putRows
      .filter(r => r.strike && r.p_Last !== undefined)
      .map(r => {
        const strike = parseNum(r.strike);
        if (strike === null) return null;
        return {
          strike,
          lastPrice:        parseNum(r.p_Last),
          bid:              parseNum(r.p_Bid),
          ask:              parseNum(r.p_Ask),
          volume:           parseNum(r.p_Volume),
          openInterest:     parseNum(r.p_Openinterest),
          impliedVolatility: parseNum(r.p_IV) ? (parseNum(r.p_IV)! / 100) : null,
          inTheMoney:       r.p_ITM === 'Y' || r.p_InTheMoney === 'Yes',
        };
      })
      .filter(Boolean);

    // Expiry dates from the first response
    const expiryList: string[] = expiryJson?.data?.expiryList?.map((e: any) => e.value ?? e) ?? [];
    const selectedExpiry: string = callJson?.data?.expiryList?.[0]?.value ?? expiry;

    // Underlying price from the data if available
    const underlyingPrice = parseNum(callJson?.data?.underlyingPrice ?? null);

    if (calls.length === 0 && puts.length === 0) {
      return NextResponse.json(
        { error: `No options data found for ${sym}. This symbol may not have listed options.` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      symbol: sym,
      underlyingPrice,
      expirationDates: expiryList,
      selectedExpiry,
      calls,
      puts,
      source: 'NASDAQ',
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch options data' },
      { status: 500 }
    );
  }
}
