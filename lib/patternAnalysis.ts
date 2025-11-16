import { ChartPattern } from '@/types';

export interface PatternAnalysisResult {
  signal: 'strong_bullish' | 'bullish' | 'neutral' | 'bearish' | 'strong_bearish';
  confidence: number;
  score: number; // -100 to +100
  summary: string;
  reasoning: string[];
  patternBreakdown: {
    bullish: number;
    bearish: number;
    neutral: number;
  };
  keyPatterns: {
    pattern: ChartPattern;
    impact: 'high' | 'medium' | 'low';
    weight: number;
  }[];
  recommendations: string[];
}

/**
 * Analyze all patterns within a date range and provide a comprehensive conclusion
 */
export function analyzePatterns(
  patterns: ChartPattern[],
  startDate?: string,
  endDate?: string
): PatternAnalysisResult {
  // Filter patterns by date range if provided
  let relevantPatterns = patterns;
  if (startDate && endDate) {
    const rangeStart = new Date(startDate).getTime();
    const rangeEnd = new Date(endDate).getTime();
    
    relevantPatterns = patterns.filter(pattern => {
      // Handle invalid dates gracefully
      const patternStart = new Date(pattern.startDate).getTime();
      const patternEnd = new Date(pattern.endDate).getTime();
      
      if (isNaN(patternStart) || isNaN(patternEnd)) {
        return true; // Include patterns with invalid dates to be safe
      }

      // Add a small buffer (1 day) to make filtering more lenient
      const bufferMs = 24 * 60 * 60 * 1000; // 1 day in milliseconds
      const adjustedRangeStart = rangeStart - bufferMs;
      const adjustedRangeEnd = rangeEnd + bufferMs;

      // Check if pattern overlaps with the view range (with buffer)
      return (
        (patternStart >= adjustedRangeStart && patternStart <= adjustedRangeEnd) ||
        (patternEnd >= adjustedRangeStart && patternEnd <= adjustedRangeEnd) ||
        (patternStart <= adjustedRangeStart && patternEnd >= adjustedRangeEnd)
      );
    });
  }

  if (relevantPatterns.length === 0) {
    return {
      signal: 'neutral',
      confidence: 0,
      score: 0,
      summary: 'No significant patterns detected in the current view.',
      reasoning: ['Insufficient pattern data for analysis'],
      patternBreakdown: { bullish: 0, bearish: 0, neutral: 0 },
      keyPatterns: [],
      recommendations: ['Consider zooming out to see more historical patterns'],
    };
  }

  // Calculate pattern weights and scores
  const scoredPatterns = relevantPatterns.map(pattern => {
    const baseWeight = pattern.confidence;
    
    // Calculate recency weight (more recent = more important)
    const patternEndDate = new Date(pattern.endDate).getTime();
    const now = endDate ? new Date(endDate).getTime() : Date.now();
    const daysSinceEnd = (now - patternEndDate) / (1000 * 60 * 60 * 24);
    const recencyWeight = Math.exp(-daysSinceEnd / 30); // Decay over 30 days
    
    // Calculate size weight (longer patterns = more significant)
    const patternDuration = new Date(pattern.endDate).getTime() - new Date(pattern.startDate).getTime();
    const durationDays = patternDuration / (1000 * 60 * 60 * 24);
    const sizeWeight = Math.min(1, durationDays / 60); // Max weight at 60 days
    
    // Combined weight
    const totalWeight = baseWeight * (0.5 + recencyWeight * 0.3 + sizeWeight * 0.2);
    
    // Calculate directional score
    let directionalScore = 0;
    if (pattern.direction === 'bullish') {
      directionalScore = 1;
    } else if (pattern.direction === 'bearish') {
      directionalScore = -1;
    }
    
    // Pattern type multipliers (some patterns are stronger signals)
    const typeMultiplier = getPatternTypeMultiplier(pattern.type);
    
    const finalScore = directionalScore * totalWeight * typeMultiplier;
    
    return {
      pattern,
      weight: totalWeight,
      score: finalScore,
      impact: getImpactLevel(totalWeight),
    };
  });

  // Sort by weight (most important first)
  scoredPatterns.sort((a, b) => b.weight - a.weight);

  // Calculate overall score (-100 to +100)
  const totalScore = scoredPatterns.reduce((sum, sp) => sum + sp.score, 0);
  const normalizedScore = Math.max(-100, Math.min(100, totalScore * 100));

  // Count pattern directions
  const breakdown = {
    bullish: scoredPatterns.filter(sp => sp.pattern.direction === 'bullish').length,
    bearish: scoredPatterns.filter(sp => sp.pattern.direction === 'bearish').length,
    neutral: scoredPatterns.filter(sp => sp.pattern.direction === 'neutral').length,
  };

  // Determine signal
  const signal = getSignalFromScore(normalizedScore);
  const confidence = calculateConfidence(scoredPatterns);

  // Generate reasoning
  const reasoning = generateReasoning(scoredPatterns, normalizedScore);

  // Generate summary
  const summary = generateSummary(signal, confidence, breakdown, relevantPatterns.length);

  // Generate recommendations
  const recommendations = generateRecommendations(signal, confidence, scoredPatterns);

  // Get key patterns (top 5)
  const keyPatterns = scoredPatterns.slice(0, 5).map(sp => ({
    pattern: sp.pattern,
    impact: sp.impact,
    weight: sp.weight,
  }));

  return {
    signal,
    confidence,
    score: normalizedScore,
    summary,
    reasoning,
    patternBreakdown: breakdown,
    keyPatterns,
    recommendations,
  };
}

function getPatternTypeMultiplier(type: string): number {
  const multipliers: Record<string, number> = {
    head_and_shoulders: 1.5,
    double_top: 1.3,
    double_bottom: 1.3,
    wedge_up: 1.2,
    wedge_down: 1.2,
    triangle_ascending: 1.1,
    triangle_descending: 1.1,
    channel_up: 1.0,
    channel_down: 1.0,
    trendline_support: 0.9,
    trendline_resistance: 0.9,
    horizontal_sr: 0.8,
    multiple_top: 1.1,
    multiple_bottom: 1.1,
  };
  return multipliers[type] || 1.0;
}

function getImpactLevel(weight: number): 'high' | 'medium' | 'low' {
  if (weight >= 0.75) return 'high';
  if (weight >= 0.60) return 'medium';
  return 'low';
}

function getSignalFromScore(score: number): PatternAnalysisResult['signal'] {
  if (score >= 40) return 'strong_bullish';
  if (score >= 15) return 'bullish';
  if (score <= -40) return 'strong_bearish';
  if (score <= -15) return 'bearish';
  return 'neutral';
}

function calculateConfidence(
  scoredPatterns: Array<{ pattern: ChartPattern; weight: number; score: number }>
): number {
  if (scoredPatterns.length === 0) return 0;

  // Average confidence of top patterns
  const topPatterns = scoredPatterns.slice(0, 3);
  const avgConfidence = topPatterns.reduce((sum, sp) => sum + sp.pattern.confidence, 0) / topPatterns.length;

  // Adjust for agreement (when patterns agree, confidence increases)
  const bullishCount = scoredPatterns.filter(sp => sp.pattern.direction === 'bullish').length;
  const bearishCount = scoredPatterns.filter(sp => sp.pattern.direction === 'bearish').length;
  const total = scoredPatterns.length;
  const agreement = Math.max(bullishCount, bearishCount) / total;

  // Combine factors
  return Math.min(0.95, avgConfidence * 0.6 + agreement * 0.4);
}

function generateReasoning(
  scoredPatterns: Array<{ pattern: ChartPattern; weight: number; score: number; impact: 'high' | 'medium' | 'low' }>,
  score: number
): string[] {
  const reasoning: string[] = [];

  // Count strong patterns
  const strongBullish = scoredPatterns.filter(
    sp => sp.pattern.direction === 'bullish' && sp.impact === 'high'
  );
  const strongBearish = scoredPatterns.filter(
    sp => sp.pattern.direction === 'bearish' && sp.impact === 'high'
  );

  // Overall direction
  if (score > 0) {
    reasoning.push(
      `Bullish bias detected with ${strongBullish.length} high-impact bullish pattern${strongBullish.length !== 1 ? 's' : ''}`
    );
  } else if (score < 0) {
    reasoning.push(
      `Bearish bias detected with ${strongBearish.length} high-impact bearish pattern${strongBearish.length !== 1 ? 's' : ''}`
    );
  } else {
    reasoning.push('Balanced pattern distribution suggests neutral market conditions');
  }

  // Mention key patterns
  const topPattern = scoredPatterns[0];
  if (topPattern) {
    const direction = topPattern.pattern.direction === 'bullish' ? '↗️ bullish' : 
                     topPattern.pattern.direction === 'bearish' ? '↘️ bearish' : 'neutral';
    reasoning.push(
      `Dominant pattern: ${topPattern.pattern.label} (${(topPattern.pattern.confidence * 100).toFixed(0)}% confidence, ${direction})`
    );
  }

  // Pattern diversity
  const uniqueTypes = new Set(scoredPatterns.map(sp => sp.pattern.type)).size;
  if (uniqueTypes >= 3) {
    reasoning.push(`${uniqueTypes} different pattern types confirm the signal`);
  }

  // Recent vs historical
  const recentPatterns = scoredPatterns.filter(sp => {
    const endDate = new Date(sp.pattern.endDate).getTime();
    const daysSince = (Date.now() - endDate) / (1000 * 60 * 60 * 24);
    return daysSince <= 10;
  });
  
  if (recentPatterns.length >= 2) {
    const recentDirection = recentPatterns[0].pattern.direction;
    const allRecentSame = recentPatterns.every(sp => sp.pattern.direction === recentDirection);
    if (allRecentSame) {
      reasoning.push(`${recentPatterns.length} recent patterns align ${recentDirection === 'bullish' ? 'bullishly' : 'bearishly'}`);
    }
  }

  return reasoning;
}

function generateSummary(
  signal: PatternAnalysisResult['signal'],
  confidence: number,
  breakdown: { bullish: number; bearish: number; neutral: number },
  totalPatterns: number
): string {
  const confidencePercent = (confidence * 100).toFixed(0);
  
  const signalText = {
    strong_bullish: '🚀 STRONG BULLISH',
    bullish: '📈 BULLISH',
    neutral: '⚖️ NEUTRAL',
    bearish: '📉 BEARISH',
    strong_bearish: '🔻 STRONG BEARISH',
  }[signal];

  return `${signalText} signal detected across ${totalPatterns} pattern${totalPatterns !== 1 ? 's' : ''} with ${confidencePercent}% confidence (${breakdown.bullish} bullish, ${breakdown.bearish} bearish, ${breakdown.neutral} neutral)`;
}

function generateRecommendations(
  signal: PatternAnalysisResult['signal'],
  confidence: number,
  scoredPatterns: Array<{ pattern: ChartPattern; weight: number; score: number }>
): string[] {
  const recommendations: string[] = [];

  // Trading recommendations based on signal
  if (signal === 'strong_bullish' && confidence >= 0.7) {
    recommendations.push('Strong bullish momentum - consider long positions or holding');
    recommendations.push('Watch for breakout confirmation above resistance levels');
  } else if (signal === 'bullish' && confidence >= 0.6) {
    recommendations.push('Moderate bullish trend - favorable for long positions');
    recommendations.push('Monitor for continued pattern development');
  } else if (signal === 'strong_bearish' && confidence >= 0.7) {
    recommendations.push('Strong bearish pressure - consider defensive positions');
    recommendations.push('Watch for breakdown below support levels');
  } else if (signal === 'bearish' && confidence >= 0.6) {
    recommendations.push('Moderate bearish trend - exercise caution with long positions');
    recommendations.push('Consider setting stop losses');
  } else {
    recommendations.push('Mixed signals - wait for clearer pattern formation');
    recommendations.push('Range-bound conditions - consider range trading strategies');
  }

  // Pattern-specific recommendations
  const hasHeadAndShoulders = scoredPatterns.some(
    sp => sp.pattern.type === 'head_and_shoulders'
  );
  if (hasHeadAndShoulders) {
    recommendations.push('⚠️ Head & Shoulders pattern present - significant reversal signal');
  }

  const hasDoubleTopBottom = scoredPatterns.some(
    sp => sp.pattern.type === 'double_top' || sp.pattern.type === 'double_bottom'
  );
  if (hasDoubleTopBottom) {
    recommendations.push('Double top/bottom pattern suggests potential trend reversal');
  }

  // Confidence warning
  if (confidence < 0.6) {
    recommendations.push('⚠️ Low confidence - wait for more confirmation before taking action');
  }

  return recommendations;
}

