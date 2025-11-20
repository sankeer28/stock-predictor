import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

interface RedditStock {
  no_of_comments: number;
  sentiment: 'Bullish' | 'Bearish' | 'Neutral';
  sentiment_score: number;
  ticker: string;
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const date = searchParams.get('date'); // Optional: MM-DD-YYYY format

    // Build API URL
    const apiUrl = date
      ? `https://api.tradestie.com/v1/apps/reddit?date=${date}`
      : 'https://api.tradestie.com/v1/apps/reddit';

    const response = await fetch(apiUrl, {
      next: { revalidate: 900 }, // Cache for 15 minutes (API updates every 15 mins)
    });

    if (!response.ok) {
      throw new Error(`Tradestie API error: ${response.status}`);
    }

    const data: RedditStock[] = await response.json();

    // Return top 20 stocks (instead of all 50)
    const topStocks = data.slice(0, 20);

    return NextResponse.json({
      success: true,
      stocks: topStocks,
      total: data.length,
      lastUpdated: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('Reddit API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch Reddit sentiment data',
        stocks: [],
      },
      { status: 500 }
    );
  }
}
