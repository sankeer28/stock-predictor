import { NextRequest, NextResponse } from 'next/server';
import { StockData } from '@/types';

/**
 * Fetch multiple stocks data for correlation analysis
 * GET /api/correlation?symbols=AAPL,MSFT,GOOGL&startDate=2023-01-01&endDate=2024-01-01
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbolsParam = searchParams.get('symbols');
  const startDate = searchParams.get('startDate');
  const endDate = searchParams.get('endDate');

  if (!symbolsParam) {
    return NextResponse.json(
      { error: 'symbols parameter is required (comma-separated)' },
      { status: 400 }
    );
  }

  const symbols = symbolsParam.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);

  if (symbols.length < 2) {
    return NextResponse.json(
      { error: 'At least 2 symbols are required for correlation analysis' },
      { status: 400 }
    );
  }

  if (symbols.length > 20) {
    return NextResponse.json(
      { error: 'Maximum 20 symbols allowed' },
      { status: 400 }
    );
  }

  try {
    console.log(`[Correlation API] Fetching data for symbols: ${symbols.join(', ')}`);

    // Calculate date range
    const end = endDate ? new Date(endDate) : new Date();
    const start = startDate ? new Date(startDate) : new Date(end.getTime() - 365 * 24 * 60 * 60 * 1000);

    const endSeconds = Math.floor(end.getTime() / 1000);
    const startSeconds = Math.floor(start.getTime() / 1000);

    // Fetch data for all symbols in parallel
    const fetchPromises = symbols.map(async (symbol) => {
      try {
        const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${startSeconds}&period2=${endSeconds}&interval=1d`;

        const response = await fetch(url, {
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
          },
          cache: 'no-store',
          signal: AbortSignal.timeout(15000),
        });

        if (!response.ok) {
          console.error(`[Correlation API] Failed to fetch ${symbol}: ${response.status}`);
          return { symbol, data: [], error: `Failed to fetch data: ${response.status}` };
        }

        const data = await response.json();

        if (!data.chart?.result?.[0]) {
          console.error(`[Correlation API] Invalid data for ${symbol}`);
          return { symbol, data: [], error: 'Invalid data structure' };
        }

        const result = data.chart.result[0];
        const timestamps = result.timestamp;
        const quotes = result.indicators.quote[0];

        const stockData: StockData[] = timestamps.map((timestamp: number, index: number) => {
          const date = new Date(timestamp * 1000);
          return {
            date: date.toISOString(),
            open: quotes.open[index] || 0,
            high: quotes.high[index] || 0,
            low: quotes.low[index] || 0,
            close: quotes.close[index] || 0,
            volume: quotes.volume[index] || 0,
            adjClose: quotes.close[index] || 0,
          };
        });

        const validData = stockData
          .filter(d => d.close > 0)
          .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());

        console.log(`[Correlation API] Fetched ${validData.length} data points for ${symbol}`);

        return {
          symbol,
          data: validData,
          companyName: result.meta.longName || result.meta.shortName || symbol,
        };
      } catch (error) {
        console.error(`[Correlation API] Error fetching ${symbol}:`, error);
        return {
          symbol,
          data: [],
          error: error instanceof Error ? error.message : 'Unknown error'
        };
      }
    });

    const results = await Promise.all(fetchPromises);

    // Filter out failed fetches
    const successfulResults = results.filter(r => r.data.length > 0);

    if (successfulResults.length < 2) {
      return NextResponse.json(
        { error: 'Could not fetch sufficient data for correlation analysis' },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      stocks: successfulResults,
      startDate: start.toISOString(),
      endDate: end.toISOString(),
      totalSymbols: successfulResults.length,
    });

  } catch (error: any) {
    console.error('[Correlation API] Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch correlation data' },
      { status: 500 }
    );
  }
}
