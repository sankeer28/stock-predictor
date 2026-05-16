import { NextResponse } from 'next/server';

// CNN Fear & Greed Index — stock market, not crypto
const CNN_URL = 'https://production.dataviz.cnn.io/index/fearandgreed/graphdata';

const CNN_HEADERS = {
  'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
  'Referer': 'https://www.cnn.com/markets/fear-and-greed',
  'Origin': 'https://www.cnn.com',
  'Accept': 'application/json, text/plain, */*',
};

export async function GET() {
  try {
    const response = await fetch(CNN_URL, {
      headers: CNN_HEADERS,
      cache: 'no-store',
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      throw new Error(`CNN API returned ${response.status}`);
    }

    const json = await response.json();

    // Current value
    const fg = json.fear_and_greed;
    const currentScore = Math.round(fg.score);
    const currentRating = fg.rating as string;
    const currentTs = Math.floor(new Date(fg.timestamp).getTime() / 1000);

    // Historical — CNN returns newest-last, take last 7 entries and reverse to newest-first
    const historical: Array<{ x: number; y: number; rating: string }> =
      json.fear_and_greed_historical?.data ?? [];
    const recent = historical.slice(-7).reverse(); // [0] = most recent

    // Build data array matching the shape our component already uses
    const data = [
      { value: String(currentScore), value_classification: currentRating, timestamp: String(currentTs) },
      ...recent.slice(1).map((d) => ({
        value: String(Math.round(d.y)),
        value_classification: d.rating,
        timestamp: String(Math.floor(d.x / 1000)),
      })),
    ];

    return NextResponse.json(
      { name: 'Fear and Greed Index', data, metadata: { error: null } },
      { headers: { 'Cache-Control': 'no-store' } }
    );
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to fetch Fear & Greed index' },
      { status: 500 }
    );
  }
}
