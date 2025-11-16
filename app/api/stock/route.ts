import { NextRequest, NextResponse } from 'next/server';
import { StockData } from '@/types';

const ALLOWED_INTERVALS = new Set([
  '1m',
  '2m',
  '5m',
  '15m',
  '30m',
  '60m',
  '90m',
  '1h',
  '1d',
  '5d',
  '1wk',
  '1mo',
  '3mo',
]);

const INTERVAL_MAX_DAYS: Record<string, number> = {
  '1m': 7,
  '2m': 14,
  '5m': 60,
  '15m': 60,
  '30m': 60,
  '60m': 730,
  '90m': 730,
  '1h': 730,
  '1d': 3650,
  '5d': 3650,
  '1wk': 3650,
  '1mo': 3650,
  '3mo': 3650,
};

/**
 * Fetch stock data from Yahoo Finance API
 * GET /api/stock?symbol=AAPL&days=365
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol');
  const requestedDays = parseInt(searchParams.get('days') || '365', 10);
  const requestedInterval = (searchParams.get('interval') || '1d').toLowerCase();
  const interval = ALLOWED_INTERVALS.has(requestedInterval) ? requestedInterval : '1d';
  const maxDaysForInterval = INTERVAL_MAX_DAYS[interval] ?? 1825;
  const fallbackDays = Number.isFinite(requestedDays) && requestedDays > 0 ? requestedDays : 365;
  const effectiveDays = Math.min(fallbackDays, maxDaysForInterval);

  if (!symbol) {
    return NextResponse.json(
      { error: 'Symbol parameter is required' },
      { status: 400 }
    );
  }

  try {
    console.log(`[Stock API] Fetching data for ${symbol}, days: ${effectiveDays}, interval: ${interval}`);

    // Calculate date range - add a buffer only for daily+ data to avoid exceeding intraday limits
    const isIntradayInterval = interval.includes('m') || interval === '60m' || interval === '90m' || interval === '1h';
    const nowSeconds = Math.floor(Date.now() / 1000);
    const daySeconds = 24 * 60 * 60;
    const endDate = isIntradayInterval ? nowSeconds : nowSeconds + daySeconds;
    const lookbackDays = isIntradayInterval ? effectiveDays : effectiveDays + 1;
    const startDate = endDate - (lookbackDays * daySeconds);

    // Fetch from Yahoo Finance
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${startDate}&period2=${endDate}&interval=${interval}`;
    console.log(`[Stock API] Requesting: ${url}`);

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
      },
      cache: 'no-store', // Disable caching to get fresh data
      signal: AbortSignal.timeout(10000), // 10 second timeout
    });

    console.log(`[Stock API] Response status: ${response.status}`);

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`[Stock API] Yahoo Finance error response:`, errorText);
      throw new Error(`Yahoo Finance API error: ${response.status} - ${errorText.substring(0, 200)}`);
    }

    const data = await response.json();
    console.log(`[Stock API] Successfully parsed JSON response`);

    // Check if data is valid
    if (!data.chart?.result?.[0]) {
      console.error(`[Stock API] Invalid data structure:`, JSON.stringify(data, null, 2).substring(0, 500));
      return NextResponse.json(
        { error: 'Invalid symbol or no data available' },
        { status: 404 }
      );
    }

    const result = data.chart.result[0];

    // Validate required fields
    if (!result.timestamp || !result.indicators?.quote?.[0]) {
      console.error(`[Stock API] Missing required fields in response`);
      throw new Error('Invalid data structure from Yahoo Finance');
    }

    const timestamps = result.timestamp;
    const quotes = result.indicators.quote[0];
    const adjClose = result.indicators.adjclose?.[0]?.adjclose || [];

    console.log(`[Stock API] Processing ${timestamps.length} data points`);

    // Transform to our format
    const stockData: StockData[] = timestamps.map((timestamp: number, index: number) => {
      // Convert timestamp to date string in UTC to avoid timezone issues
      const date = new Date(timestamp * 1000);
      const dateStr = date.toISOString();

      return {
        date: dateStr,
        open: quotes.open[index] || 0,
        high: quotes.high[index] || 0,
        low: quotes.low[index] || 0,
        close: quotes.close[index] || 0,
        volume: quotes.volume[index] || 0,
        adjClose: adjClose[index] || quotes.close[index] || 0,
      };
    });

    // Filter out invalid data points and sort by date
    const validData = stockData
      .filter(d => d.close > 0 && d.open > 0 && d.high > 0 && d.low > 0)
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

    // Log the most recent date for debugging
    if (validData.length > 0) {
      console.log(`Latest data for ${symbol}: ${validData[validData.length - 1].date}`);
    }

    // Fetch additional company info from multiple endpoints
    let companyInfo: any = {
      fiftyTwoWeekHigh: null,
      fiftyTwoWeekLow: null,
      averageVolume: null,
    };

    try {
      // Extract only the data available from chart meta (Yahoo Finance blocks other endpoints)
      const meta = result.meta;

      // Use what we can get from chart API
      companyInfo.fiftyTwoWeekHigh = meta.fiftyTwoWeekHigh || null;
      companyInfo.fiftyTwoWeekLow = meta.fiftyTwoWeekLow || null;
      companyInfo.averageVolume = meta.regularMarketVolume || null;

      // Note: Yahoo Finance now requires authentication for detailed company info
      // The quote and quoteSummary endpoints return 401 Unauthorized
      // We can only use the limited data from the chart API
      console.log('Company info limited to chart API data due to Yahoo Finance restrictions');
    } catch (err) {
      console.error('Failed to fetch company info:', err);
    }

    // Determine current price and previous close
    const currentPrice = typeof result.meta.regularMarketPrice !== 'undefined' ? result.meta.regularMarketPrice : (validData.length > 0 ? validData[validData.length - 1].close : null);
    const prevCloseFromMeta = typeof result.meta.regularMarketPreviousClose !== 'undefined' ? result.meta.regularMarketPreviousClose : null;

    // Try to read change values from meta first
    let change = typeof result.meta.regularMarketChange !== 'undefined' ? result.meta.regularMarketChange : null;
    let changePercent = typeof result.meta.regularMarketChangePercent !== 'undefined' ? result.meta.regularMarketChangePercent : null;

    // Fallback: compute from the last two valid data points if meta fields are missing
    if ((change === null || changePercent === null) && currentPrice !== null) {
      const prevClose = prevCloseFromMeta ?? (validData.length > 1 ? validData[validData.length - 2].close : null);
      if (prevClose !== null && typeof prevClose === 'number') {
        change = currentPrice - prevClose;
        changePercent = prevClose !== 0 ? (change / prevClose) * 100 : null;
      }
    }

    return NextResponse.json({
      symbol: result.meta.symbol,
      currency: result.meta.currency,
      exchangeName: result.meta.exchangeName,
      currentPrice,
      previousClose: prevCloseFromMeta,
      change,
      changePercent,
      companyName: result.meta.longName || result.meta.shortName || result.meta.symbol,
      marketState: result.meta.marketState || 'UNKNOWN',
      interval,
      companyInfo,
      data: validData,
    });

  } catch (error: any) {
    console.error('Error fetching stock data:', error);
    // Return more specific error message for debugging
    const errorMessage = error instanceof Error ? error.message : 'Failed to fetch stock data';
    return NextResponse.json(
      {
        error: errorMessage,
        details: error instanceof Error ? error.stack : String(error)
      },
      { status: 500 }
    );
  }
}
