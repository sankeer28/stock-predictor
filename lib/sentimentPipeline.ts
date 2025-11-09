import { pipeline } from '@xenova/transformers';

// Singleton pattern for sentiment analysis pipeline
let sentimentPipeline: any = null;

/**
 * Get or initialize the sentiment analysis pipeline
 * Uses DistilBERT with keyword-enhanced logic for better financial sentiment
 */
export async function getSentimentPipeline(): Promise<any> {
  if (sentimentPipeline === null) {
    // Use DistilBERT fine-tuned on SST-2
    sentimentPipeline = await pipeline(
      'sentiment-analysis',
      'Xenova/distilbert-base-uncased-finetuned-sst-2-english'
    );
    console.log('Loaded sentiment model: DistilBERT SST-2 with keyword enhancement');
  }
  return sentimentPipeline;
}

/**
 * NLP-enhanced sentiment adjustment for financial news
 * Uses keyword detection, negation handling, and intensity modifiers
 * Returns adjustment factor: positive number = boost positive, negative = boost negative
 */
function getKeywordAdjustment(text: string): { adjustment: number; override?: 'positive' | 'negative' } {
  const lowerText = text.toLowerCase();

  // Negation words that flip sentiment
  const negations = ['not', 'no', 'never', 'nothing', 'nowhere', 'neither', 'nobody',
                     'none', "n't", 'hardly', 'scarcely', 'barely'];

  // Intensity modifiers
  const strongModifiers = ['very', 'extremely', 'highly', 'significantly', 'substantially',
                           'tremendously', 'incredibly', 'exceptionally', 'remarkably'];
  const weakModifiers = ['slightly', 'somewhat', 'fairly', 'relatively', 'moderately',
                         'rather', 'quite', 'pretty'];

  // Comparative language (positive)
  const comparativePositive = [
    'better than expected', 'exceeded expectations', 'outperformed',
    'above forecast', 'beat estimates', 'topped expectations',
    'stronger than anticipated', 'surpassed projections'
  ];

  // Comparative language (negative)
  const comparativeNegative = [
    'worse than expected', 'below expectations', 'underperformed',
    'below forecast', 'missed estimates', 'fell short',
    'weaker than anticipated', 'disappointed investors'
  ];

  // Strong positive indicators
  const strongPositiveKeywords = [
    'all-time high', 'record high', 'surge', 'soar', 'rally', 'boom',
    'strong gains', 'big gains', 'massive gains', 'huge gains',
    'significant growth', 'revenue growth', 'profit surge', 'earnings beat',
    'stock soar', 'shares surge', 'record profit', 'record earnings',
    'blowout earnings', 'explosive growth', 'skyrocket',
    'upside', 'massive upside', 'huge upside', 'could double', 'could triple',
    'market cap could double', 'price target', 'raised price target',
    'analyst upgrade', 'buy rating', 'outperform rating'
  ];

  // Moderate positive indicators
  const positiveKeywords = [
    'gains', 'profit', 'growth', 'beat', 'exceed', 'strong', 'bullish',
    'upturn', 'success', 'outperform', 'rise', 'up', 'boost', 'increase',
    'positive', 'higher', 'improving', 'recovered', 'rebound', 'climbing',
    'advancing', 'jumping', 'upgraded', 'momentum', 'accelerate',
    'expansion', 'optimistic', 'favorable', 'solid', 'robust'
  ];

  // Strong negative indicators
  const strongNegativeKeywords = [
    'crash', 'plunge', 'crisis', 'bankruptcy', 'collapse', 'disaster',
    'massive loss', 'significant decline', 'stock crash', 'shares plunge',
    'tumble', 'plummet', 'nosedive', 'free fall', 'devastating loss',
    'catastrophic', 'meltdown', 'wipeout'
  ];

  // Moderate negative indicators
  const negativeKeywords = [
    'loss', 'decline', 'fall', 'drop', 'weak', 'miss', 'concern',
    'worry', 'risk', 'problem', 'bearish', 'slump', 'down', 'tumbles',
    'disappointing', 'struggle', 'trouble', 'threat', 'challenges',
    'downgrade', 'lower', 'falling', 'slipping', 'sliding', 'sinking',
    'weaken', 'deteriorate', 'worsen', 'negative', 'caution', 'warning'
  ];

  let score = 0;
  let hasStrongPositive = false;
  let hasStrongNegative = false;

  // Helper function to check for negation near a keyword
  const hasNegationNear = (index: number, windowSize: number = 15): boolean => {
    const start = Math.max(0, index - windowSize);
    const contextBefore = lowerText.substring(start, index);
    return negations.some(neg => contextBefore.includes(neg));
  };

  // Helper function to check for intensity modifiers
  const getIntensityMultiplier = (index: number, windowSize: number = 15): number => {
    const start = Math.max(0, index - windowSize);
    const contextBefore = lowerText.substring(start, index);

    if (strongModifiers.some(mod => contextBefore.includes(mod))) return 1.5;
    if (weakModifiers.some(mod => contextBefore.includes(mod))) return 0.7;
    return 1.0;
  };

  // Check strong positive keywords with NLP enhancements
  strongPositiveKeywords.forEach(keyword => {
    const index = lowerText.indexOf(keyword);
    if (index !== -1) {
      const isNegated = hasNegationNear(index);
      const intensity = getIntensityMultiplier(index);
      const keywordScore = 2 * intensity;

      score += isNegated ? -keywordScore : keywordScore;
      if (!isNegated) hasStrongPositive = true;
      else hasStrongNegative = true; // Negated positive = negative
    }
  });

  // Check moderate positive keywords with NLP
  positiveKeywords.forEach(keyword => {
    const index = lowerText.indexOf(keyword);
    if (index !== -1) {
      const isNegated = hasNegationNear(index);
      const intensity = getIntensityMultiplier(index);
      const keywordScore = 0.5 * intensity;

      score += isNegated ? -keywordScore : keywordScore;
    }
  });

  // Check strong negative keywords with NLP
  strongNegativeKeywords.forEach(keyword => {
    const index = lowerText.indexOf(keyword);
    if (index !== -1) {
      const isNegated = hasNegationNear(index);
      const intensity = getIntensityMultiplier(index);
      const keywordScore = 2 * intensity;

      score += isNegated ? keywordScore : -keywordScore; // Negated negative = positive
      if (!isNegated) hasStrongNegative = true;
      else hasStrongPositive = true; // Negated negative = positive
    }
  });

  // Check moderate negative keywords with NLP
  negativeKeywords.forEach(keyword => {
    const index = lowerText.indexOf(keyword);
    if (index !== -1) {
      const isNegated = hasNegationNear(index);
      const intensity = getIntensityMultiplier(index);
      const keywordScore = 0.5 * intensity;

      score += isNegated ? keywordScore : -keywordScore;
    }
  });

  // Check comparative language (these are strong signals)
  comparativePositive.forEach(phrase => {
    if (lowerText.includes(phrase)) {
      score += 1.5;
      hasStrongPositive = true;
    }
  });

  comparativeNegative.forEach(phrase => {
    if (lowerText.includes(phrase)) {
      score -= 1.5;
      hasStrongNegative = true;
    }
  });

  // Pattern matching for percentage gains (e.g., "80% upside", "50% gain")
  const percentageGainPattern = /(\d+)%\s*(upside|gain|increase|rise|growth|higher)/gi;
  const percentageLossPattern = /(\d+)%\s*(downside|loss|decrease|decline|drop|lower)/gi;

  const gainMatches = lowerText.match(percentageGainPattern);
  if (gainMatches && gainMatches.length > 0) {
    score += 2;
    hasStrongPositive = true;
  }

  const lossMatches = lowerText.match(percentageLossPattern);
  if (lossMatches && lossMatches.length > 0) {
    score -= 2;
    hasStrongNegative = true;
  }

  // If we have strong keywords, provide an override suggestion
  if (hasStrongPositive && score > 1) {
    return { adjustment: score, override: 'positive' };
  }
  if (hasStrongNegative && score < -1) {
    return { adjustment: score, override: 'negative' };
  }

  return { adjustment: score };
}

/**
 * Analyze sentiment using transformers.js (DistilBERT) + NLP enhancements
 * Returns label (positive/negative/neutral) and score
 *
 * Features:
 * - ML Model: DistilBERT for base sentiment
 * - Negation Detection: "not good", "no growth" → flips sentiment
 * - Intensity Modifiers: "very", "extremely" → boosts strength
 * - Comparative Language: "better than expected" → strong signal
 * - Financial Keywords: Domain-specific term detection
 * - Context-Aware: Analyzes surrounding words (15-char window)
 */
export async function analyzeSentimentWithTransformers(text: string): Promise<{
  label: string;
  score: number;
}> {
  if (!text || text.trim().length === 0) {
    return { label: 'neutral', score: 0 };
  }

  try {
    const pipe = await getSentimentPipeline();
    const truncatedText = text.slice(0, 512); // Limit to 512 tokens

    // Get ML model prediction
    const result = await pipe(truncatedText);
    const sentiment = Array.isArray(result) ? result[0] : result;
    const mlLabel = sentiment.label.toLowerCase();
    const mlScore = sentiment.score;

    // Get keyword-based adjustment
    const { adjustment, override } = getKeywordAdjustment(text);

    // Determine final sentiment
    let finalLabel: string;
    let finalScore: number;

    // If keywords strongly suggest an override, use it
    if (override && Math.abs(adjustment) >= 2) {
      finalLabel = override;
      finalScore = override === 'positive' ? 0.85 : 0.85;
    } else {
      // Otherwise, use ML prediction with keyword-based confidence adjustment
      finalLabel = mlLabel;
      finalScore = mlScore;

      // Adjust confidence based on keyword match
      if ((mlLabel === 'positive' && adjustment > 0) || (mlLabel === 'negative' && adjustment < 0)) {
        // Keywords agree with ML - boost confidence
        finalScore = Math.min(0.99, mlScore + Math.abs(adjustment) * 0.05);
      } else if ((mlLabel === 'positive' && adjustment < -1) || (mlLabel === 'negative' && adjustment > 1)) {
        // Strong keyword disagreement - maybe flip or make neutral
        if (mlScore < 0.7) {
          finalLabel = 'neutral';
          finalScore = 0.5;
        }
      }

      // Map to our format (positive/negative/neutral)
      if (finalScore < 0.6) {
        finalLabel = 'neutral';
        finalScore = 0.5;
      }
    }

    return {
      label: finalLabel,
      score: finalScore,
    };
  } catch (error) {
    console.error('Sentiment analysis error:', error);
    return { label: 'neutral', score: 0 };
  }
}
