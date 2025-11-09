import { SentimentResult } from '@/types';
import { analyzeSentimentWithTransformers } from './sentimentPipeline';

/**
 * Analyze sentiment of text using transformers.js
 * This function uses a pre-trained BERT model for accurate sentiment analysis
 */
export async function analyzeSentiment(text: string): Promise<SentimentResult> {
  if (!text || text.trim().length === 0) {
    return {
      sentiment: 'neutral',
      score: 0,
      confidence: 0,
    };
  }

  try {
    // Use transformers.js for sentiment analysis (DistilBERT)
    const result = await analyzeSentimentWithTransformers(text);

    // Map model output to our format
    // Returns: "positive", "negative", or "neutral" (lowercase)
    let sentimentCategory: 'positive' | 'negative' | 'neutral';
    let normalizedScore: number;

    if (result.label === 'positive') {
      sentimentCategory = 'positive';
      normalizedScore = result.score; // 0-1 range
    } else if (result.label === 'negative') {
      sentimentCategory = 'negative';
      normalizedScore = -result.score; // Negative for our format
    } else {
      sentimentCategory = 'neutral';
      normalizedScore = 0;
    }

    // Calculate confidence (0-100)
    const confidence = Math.round(result.score * 100);

    return {
      sentiment: sentimentCategory,
      score: normalizedScore,
      confidence,
    };
  } catch (error) {
    console.error('Sentiment analysis error:', error);
    return {
      sentiment: 'neutral',
      score: 0,
      confidence: 0,
    };
  }
}

/**
 * Aggregate sentiment from multiple texts
 */
export async function aggregateSentiment(texts: string[]): Promise<SentimentResult> {
  if (texts.length === 0) {
    return {
      sentiment: 'neutral',
      score: 0,
      confidence: 0,
    };
  }

  const sentiments = await Promise.all(texts.map(text => analyzeSentiment(text)));

  // Calculate weighted average (more recent = higher weight)
  let totalScore = 0;
  let totalWeight = 0;

  sentiments.forEach((s, index) => {
    const weight = 1 / (index + 1); // Decay weight for older items
    totalScore += s.score * weight;
    totalWeight += weight;
  });

  const avgScore = totalScore / totalWeight;

  // Determine overall sentiment
  let overallSentiment: 'positive' | 'negative' | 'neutral';
  if (avgScore >= 0.15) {
    overallSentiment = 'positive';
  } else if (avgScore <= -0.15) {
    overallSentiment = 'negative';
  } else {
    overallSentiment = 'neutral';
  }

  const confidence = Math.min(100, Math.abs(avgScore) * 100);

  return {
    sentiment: overallSentiment,
    score: avgScore,
    confidence,
  };
}
