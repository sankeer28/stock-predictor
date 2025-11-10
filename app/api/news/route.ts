import { NextRequest, NextResponse } from 'next/server';
import { NewsArticle } from '@/types';

/**
 * Fetch news articles from NewsAPI (fast - no sentiment)
 * GET /api/news?symbol=AAPL
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol');

  if (!symbol) {
    return NextResponse.json(
      { error: 'Symbol parameter is required' },
      { status: 400 }
    );
  }

  const apiKey = process.env.NEWS_API_KEY;

  if (!apiKey) {
    console.warn('NEWS_API_KEY not configured, returning empty news array');
    return NextResponse.json({ articles: [] });
  }

  try {
    // Fetch from NewsAPI
    const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(symbol)}&language=en&sortBy=relevancy&pageSize=10&apiKey=${apiKey}`;

    const response = await fetch(url, {
      next: { revalidate: 3600 }, // Cache for 1 hour
    });

    if (!response.ok) {
      throw new Error(`NewsAPI error: ${response.status}`);
    }

    const data = await response.json();

    // Transform to our format WITHOUT sentiment (for fast loading)
    const articles: NewsArticle[] = (data.articles || []).map((article: any) => ({
      title: article.title,
      description: article.description || '',
      url: article.url,
      publishedAt: article.publishedAt,
      source: article.source?.name || 'Unknown',
      sentiment: { sentiment: 'neutral' as const, score: 0, confidence: 0 }, // Placeholder
    }));

    return NextResponse.json({ articles });

  } catch (error) {
    console.error('Error fetching news:', error);
    // Return empty array instead of error to not break the UI
    return NextResponse.json({ articles: [] });
  }
}
