import Sentiment from 'sentiment';
import { SentimentResult } from '@/types';

const sentiment = new Sentiment();

// Financial keywords with custom weights
const financialTerms: { [key: string]: number } = {
  // Positive terms
  'bullish': 3,
  'buy': 2,
  'upgrade': 3,
  'outperform': 3,
  'beat': 2,
  'exceeded': 2,
  'growth': 2,
  'profit': 2,
  'gain': 2,
  'surge': 3,
  'rally': 3,
  'soar': 3,
  'climb': 2,
  'rose': 2,
  'jump': 2,
  'strong': 2,
  'recovery': 2,

  // Negative terms
  'bearish': -3,
  'sell': -2,
  'downgrade': -3,
  'underperform': -3,
  'miss': -2,
  'missed': -2,
  'loss': -2,
  'decline': -2,
  'fall': -2,
  'drop': -2,
  'plunge': -3,
  'crash': -3,
  'tumble': -3,
  'weak': -2,
  'concern': -2,
  'warning': -2,
  'risk': -2,
};

// Register custom financial terms
sentiment.registerLanguage('en', {
  labels: financialTerms,
});

/**
 * Analyze sentiment of text with financial context
 */
export function analyzeSentiment(text: string): SentimentResult {
  if (!text || text.trim().length === 0) {
    return {
      sentiment: 'neutral',
      score: 0,
      confidence: 0,
    };
  }

  // Clean text
  const cleanText = text.toLowerCase().trim();

  // Get base sentiment
  const result = sentiment.analyze(cleanText);

  // Look for percentage changes
  const percentagePattern = /(\d+(?:\.\d+)?)\s*%/g;
  const percentages = cleanText.match(percentagePattern);

  let percentageScore = 0;
  if (percentages) {
    percentages.forEach(match => {
      const value = parseFloat(match);
      if (cleanText.includes('up') || cleanText.includes('gain') || cleanText.includes('rose')) {
        percentageScore += value * 0.1;
      } else if (cleanText.includes('down') || cleanText.includes('loss') || cleanText.includes('fell')) {
        percentageScore -= value * 0.1;
      }
    });
  }

  // Calculate combined score
  const combinedScore = result.score + percentageScore;

  // Normalize score to -1 to 1 range
  const normalizedScore = Math.max(-1, Math.min(1, combinedScore / 10));

  // Determine sentiment category
  let sentimentCategory: 'positive' | 'negative' | 'neutral';
  if (normalizedScore >= 0.1) {
    sentimentCategory = 'positive';
  } else if (normalizedScore <= -0.1) {
    sentimentCategory = 'negative';
  } else {
    sentimentCategory = 'neutral';
  }

  // Calculate confidence (0-100)
  const confidence = Math.min(100, Math.abs(normalizedScore) * 100);

  return {
    sentiment: sentimentCategory,
    score: normalizedScore,
    confidence,
  };
}

/**
 * Aggregate sentiment from multiple texts
 */
export function aggregateSentiment(texts: string[]): SentimentResult {
  if (texts.length === 0) {
    return {
      sentiment: 'neutral',
      score: 0,
      confidence: 0,
    };
  }

  const sentiments = texts.map(text => analyzeSentiment(text));

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
