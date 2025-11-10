import { NextRequest, NextResponse } from 'next/server';
import { analyzeSentiment } from '@/lib/sentiment';

/**
 * Analyze sentiment for news articles
 * POST /api/sentiment
 * Body: { articles: Array<{ title: string, description: string }> }
 */
export async function POST(request: NextRequest) {
  try {
    const { articles } = await request.json();

    if (!articles || !Array.isArray(articles)) {
      return NextResponse.json(
        { error: 'Articles array is required' },
        { status: 400 }
      );
    }

    // Analyze sentiment for each article with timeout protection
    const sentiments = await Promise.all(
      articles.map(async (article: any) => {
        try {
          const combinedText = `${article.title || ''} ${article.description || ''}`;

          // Add timeout for Vercel (10 seconds per article)
          const sentimentPromise = analyzeSentiment(combinedText);
          const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('Sentiment analysis timeout')), 10000)
          );

          return await Promise.race([sentimentPromise, timeoutPromise]) as any;
        } catch (error) {
          console.error('Error analyzing article sentiment:', error);
          // Return neutral on error
          return {
            sentiment: 'neutral' as const,
            score: 0,
            confidence: 0,
          };
        }
      })
    );

    return NextResponse.json({ sentiments });

  } catch (error) {
    console.error('Error analyzing sentiment:', error);

    // Return neutral sentiments as fallback
    const articles = request.body;
    return NextResponse.json({
      sentiments: Array.isArray(articles)
        ? articles.map(() => ({ sentiment: 'neutral' as const, score: 0, confidence: 0 }))
        : []
    });
  }
}

// Configure Vercel timeout (max 10 seconds for hobby plan)
export const maxDuration = 10;
