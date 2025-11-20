import { NextResponse } from 'next/server';

export const dynamic = 'force-dynamic';
export const revalidate = 0;

interface ApeWisdomStock {
  rank: string | number;
  ticker: string;
  name: string;
  mentions: string | number;
  upvotes: string | number;
  rank_24h_ago: string | number;
  mentions_24h_ago: string | number;
}

interface ApeWisdomResponse {
  count: number;
  pages: number;
  current_page: number;
  results: ApeWisdomStock[];
}

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const filter = searchParams.get('filter') || 'all-stocks';
    const page = searchParams.get('page') || '1';

    // Build API URL
    const apiUrl = `https://apewisdom.io/api/v1.0/filter/${filter}/page/${page}`;

    const response = await fetch(apiUrl, {
      next: { revalidate: 900 }, // Cache for 15 minutes
    });

    if (!response.ok) {
      throw new Error(`ApeWisdom API error: ${response.status}`);
    }

    const data: ApeWisdomResponse = await response.json();

    // Return top 20 stocks from first page
    const topStocks = data.results.slice(0, 20).map(stock => ({
      rank: Number(stock.rank),
      ticker: stock.ticker,
      name: stock.name,
      mentions: Number(stock.mentions),
      upvotes: Number(stock.upvotes),
      rank_24h_ago: Number(stock.rank_24h_ago),
      mentions_24h_ago: Number(stock.mentions_24h_ago),
      rankChange: Number(stock.rank_24h_ago) - Number(stock.rank),
      mentionsChange: Number(stock.mentions) - Number(stock.mentions_24h_ago),
    }));

    return NextResponse.json({
      success: true,
      stocks: topStocks,
      total: data.count,
      pages: data.pages,
      currentPage: data.current_page,
      filter,
      lastUpdated: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('ApeWisdom API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error.message || 'Failed to fetch ApeWisdom data',
        stocks: [],
      },
      { status: 500 }
    );
  }
}
