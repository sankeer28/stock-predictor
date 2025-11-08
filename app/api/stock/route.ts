import { NextRequest, NextResponse } from 'next/server';
import { StockData } from '@/types';

/**
 * Fetch stock data from Yahoo Finance API
 * GET /api/stock?symbol=AAPL&days=365
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol');
  const days = parseInt(searchParams.get('days') || '365');

  if (!symbol) {
    return NextResponse.json(
      { error: 'Symbol parameter is required' },
      { status: 400 }
    );
  }

  try {
    // Calculate date range
    const endDate = Math.floor(Date.now() / 1000);
    const startDate = endDate - (days * 24 * 60 * 60);

    // Fetch from Yahoo Finance
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?period1=${startDate}&period2=${endDate}&interval=1d`;

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0',
      },
    });

    if (!response.ok) {
      throw new Error(`Yahoo Finance API error: ${response.status}`);
    }

    const data = await response.json();

    // Check if data is valid
    if (!data.chart?.result?.[0]) {
      return NextResponse.json(
        { error: 'Invalid symbol or no data available' },
        { status: 404 }
      );
    }

    const result = data.chart.result[0];
    const timestamps = result.timestamp;
    const quotes = result.indicators.quote[0];
    const adjClose = result.indicators.adjclose?.[0]?.adjclose || [];

    // Transform to our format
    const stockData: StockData[] = timestamps.map((timestamp: number, index: number) => ({
      date: new Date(timestamp * 1000).toISOString().split('T')[0],
      open: quotes.open[index] || 0,
      high: quotes.high[index] || 0,
      low: quotes.low[index] || 0,
      close: quotes.close[index] || 0,
      volume: quotes.volume[index] || 0,
      adjClose: adjClose[index] || quotes.close[index] || 0,
    }));

    // Filter out invalid data points
    const validData = stockData.filter(d =>
      d.close > 0 && d.open > 0 && d.high > 0 && d.low > 0
    );

    return NextResponse.json({
      symbol: result.meta.symbol,
      currency: result.meta.currency,
      exchangeName: result.meta.exchangeName,
      currentPrice: result.meta.regularMarketPrice,
      data: validData,
    });

  } catch (error) {
    console.error('Error fetching stock data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch stock data' },
      { status: 500 }
    );
  }
}
