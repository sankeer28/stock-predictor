import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol');

  if (!symbol) {
    return NextResponse.json({ error: 'Symbol is required' }, { status: 400 });
  }

  const FINNHUB_API_KEY = process.env.FINN_HUB;

  if (!FINNHUB_API_KEY) {
    console.error('FinnHub API key not configured');
    return NextResponse.json({ error: 'API key not configured' }, { status: 500 });
  }

  try {
    const url = `https://finnhub.io/api/v1/stock/recommendation?symbol=${symbol.toUpperCase()}&token=${FINNHUB_API_KEY}`;

    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      console.error(`FinnHub API error: ${response.status} ${response.statusText}`);
      return NextResponse.json({ error: 'Failed to fetch recommendations' }, { status: response.status });
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      recommendations: data || [],
    });

  } catch (error: any) {
    console.error('Error fetching recommendations from FinnHub:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch recommendations' },
      { status: 500 }
    );
  }
}
