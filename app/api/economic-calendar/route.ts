import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const FINNHUB_API_KEY = process.env.FINN_HUB;

  if (!FINNHUB_API_KEY) {
    console.error('FinnHub API key not configured');
    return NextResponse.json({ error: 'API key not configured' }, { status: 500 });
  }

  try {
    // Get upcoming economic events
    const url = `https://finnhub.io/api/v1/calendar/economic?token=${FINNHUB_API_KEY}`;

    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      console.error(`FinnHub API error: ${response.status} ${response.statusText}`);
      return NextResponse.json({ error: 'Failed to fetch economic calendar' }, { status: response.status });
    }

    const data = await response.json();

    return NextResponse.json({
      success: true,
      events: data.economicCalendar || [],
    });

  } catch (error: any) {
    console.error('Error fetching economic calendar from FinnHub:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch economic calendar' },
      { status: 500 }
    );
  }
}
