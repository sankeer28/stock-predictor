import Sentiment from 'sentiment';

/**
 * Pure NLP-based sentiment analysis using the AFINN-165 lexicon
 * Automatically handles negation, intensifiers, and 3,382+ pre-scored words
 */

// Initialize sentiment analyzer (uses AFINN-165 lexicon)
const sentiment = new Sentiment();

/**
 * Extract sentiment from numeric patterns (percentages, price movements)
 */
function extractNumericSentiment(text: string): number {
  let score = 0;

  // Positive percentage patterns: "up 50%", "gain 30%", "rose 20%"
  const positivePatterns = [
    /(?:up|gain|gained|rose|surge|surged|jump|jumped|climb|climbed|increase|increased)(?:d|s)?\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%/gi,
    /(\d+(?:\.\d+)?)\s*%\s+(?:gain|upside|rise|growth|increase|higher)/gi,
    /\+\s*(\d+(?:\.\d+)?)\s*%/g,
  ];

  for (const pattern of positivePatterns) {
    const matches = Array.from(text.matchAll(pattern));
    for (const match of matches) {
      const percentage = parseFloat(match[1]);
      // Scale sentiment based on magnitude
      if (percentage >= 50) score += 4;
      else if (percentage >= 30) score += 3;
      else if (percentage >= 20) score += 2.5;
      else if (percentage >= 10) score += 2;
      else if (percentage >= 5) score += 1.5;
      else score += 1;
    }
  }

  // Negative percentage patterns: "down 50%", "loss 30%", "fell 20%"
  const negativePatterns = [
    /(?:down|loss|lost|fell|drop|dropped|decline|declined|plunge|plunged|tumble|tumbled|decrease|decreased)(?:d|s)?\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*%/gi,
    /(\d+(?:\.\d+)?)\s*%\s+(?:loss|downside|decline|decrease|drop|lower)/gi,
    /-\s*(\d+(?:\.\d+)?)\s*%/g,
  ];

  for (const pattern of negativePatterns) {
    const matches = Array.from(text.matchAll(pattern));
    for (const match of matches) {
      const percentage = parseFloat(match[1]);
      // Scale sentiment based on magnitude
      if (percentage >= 50) score -= 4;
      else if (percentage >= 30) score -= 3;
      else if (percentage >= 20) score -= 2.5;
      else if (percentage >= 10) score -= 2;
      else if (percentage >= 5) score -= 1.5;
      else score -= 1;
    }
  }

  // Price targets: "$500 price target" is positive
  const priceTargetPattern = /\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:price\s+)?target/gi;
  const priceTargetMatches = Array.from(text.matchAll(priceTargetPattern));
  for (const match of priceTargetMatches) {
    score += 2; // Price targets are generally positive
  }

  // Analyst ratings
  if (/buy\s+rating|strong\s+buy|outperform/gi.test(text)) {
    score += 2;
  }
  if (/sell\s+rating|strong\s+sell|underperform/gi.test(text)) {
    score -= 2;
  }

  return score;
}

/**
 * Analyze sentiment using NLP library with financial enhancements
 */
export async function analyzeSentimentWithTransformers(text: string): Promise<{
  label: string;
  score: number;
}> {
  if (!text || text.trim().length === 0) {
    return { label: 'neutral', score: 0.5 };
  }

  try {
    // Get base sentiment from NLP library (AFINN + financial lexicon)
    const result = sentiment.analyze(text);

    // Extract numeric sentiment from percentages and price movements
    const numericScore = extractNumericSentiment(text);

    // Combine scores
    const combinedScore = result.score + numericScore;

    // Normalize to 0-1 range for confidence
    // Use sqrt normalization to handle extreme values
    const alpha = 15;
    const normalizedScore = combinedScore / Math.sqrt((combinedScore * combinedScore) + alpha);

    // Determine label based on normalized score
    let label: string;
    let confidence: number;

    if (normalizedScore >= 0.1) {
      label = 'positive';
      confidence = Math.min(0.95, 0.6 + Math.abs(normalizedScore) * 0.35);
    } else if (normalizedScore <= -0.1) {
      label = 'negative';
      confidence = Math.min(0.95, 0.6 + Math.abs(normalizedScore) * 0.35);
    } else {
      label = 'neutral';
      confidence = 0.5;
    }

    return {
      label,
      score: confidence,
    };
  } catch (error) {
    console.error('NLP sentiment analysis error:', error);
    return { label: 'neutral', score: 0.5 };
  }
}
