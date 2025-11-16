export interface StockData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adjClose?: number;
}

export interface TechnicalIndicators {
  ma20: number[];
  ma50: number[];
  ma200: number[];
  rsi: number[];
  macd: number[];
  macdSignal: number[];
  macdHistogram: number[];
  bbUpper: number[];
  bbMiddle: number[];
  bbLower: number[];
  volumeMA: number[];
  ema12: number[];
  ema26: number[];
}

export interface ForecastData {
  date: string;
  predicted: number;
  upper: number;
  lower: number;
}

export interface NewsArticle {
  title: string;
  description: string;
  url: string;
  publishedAt: string;
  source: string;
  sentiment?: SentimentResult; // AI-powered sentiment analysis
}

export interface SentimentResult {
  sentiment: 'positive' | 'negative' | 'neutral';
  score: number;
  confidence: number;
}

export interface TradingSignal {
  type: 'strong_buy' | 'buy' | 'weak_buy' | 'hold' | 'weak_sell' | 'sell' | 'strong_sell';
  confidence: number;
  reasons: string[];
  priceTarget?: number;
  priceChange?: number;
}

export interface ChartDataPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  ma20?: number;
  ma50?: number;
  ma200?: number;
  bbUpper?: number;
  bbLower?: number;
  volume: number;
  rsi?: number;
  macd?: number;
  macdSignal?: number;
}

export type ChartPatternDirection = 'bullish' | 'bearish' | 'neutral';

export type ChartPatternType =
  | 'trendline_support'
  | 'trendline_resistance'
  | 'horizontal_sr'
  | 'wedge_up'
  | 'wedge_down'
  | 'wedge'
  | 'triangle_ascending'
  | 'triangle_descending'
  | 'triangle_symmetrical'
  | 'channel_up'
  | 'channel'
  | 'channel_down'
  | 'double_top'
  | 'double_bottom'
  | 'multiple_top'
  | 'multiple_bottom'
  | 'head_and_shoulders';

export type ChartPatternMeta = Record<string, number | string | boolean | undefined>;

export interface ChartPattern {
  id: string;
  type: ChartPatternType;
  label: string;
  direction: ChartPatternDirection;
  startIndex: number;
  endIndex: number;
  startDate: string;
  endDate: string;
  confidence: number;
  meta?: ChartPatternMeta;
}