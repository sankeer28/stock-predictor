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
  close: number;
  ma20?: number;
  ma50?: number;
  ma200?: number;
  bbUpper?: number;
  bbLower?: number;
  volume?: number;
  rsi?: number;
  macd?: number;
  macdSignal?: number;
}
