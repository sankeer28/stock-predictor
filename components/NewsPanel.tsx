'use client';

import React from 'react';
import { NewsArticle, SentimentResult } from '@/types';
import { TrendingUp, TrendingDown, Minus, ExternalLink } from 'lucide-react';

interface NewsPanelProps {
  articles: NewsArticle[];
  sentiments: SentimentResult[];
  isAnalyzingSentiment?: boolean;
}

export default function NewsPanel({ articles, sentiments, isAnalyzingSentiment = false }: NewsPanelProps) {
  if (articles.length === 0) {
    return (
      <div className="card">
        <span className="card-label">Latest News & Sentiment</span>
        <div className="flex items-center justify-center py-8">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: 'var(--accent)' }} />
            <span style={{ color: 'var(--text-4)' }}>Loading news...</span>
          </div>
        </div>
      </div>
    );
  }

  // Calculate overall sentiment
  const positiveSentiments = sentiments.filter(s => s.sentiment === 'positive').length;
  const negativeSentiments = sentiments.filter(s => s.sentiment === 'negative').length;
  const neutralSentiments = sentiments.filter(s => s.sentiment === 'neutral').length;
  const total = sentiments.length;

  const avgScore = sentiments.reduce((sum, s) => sum + s.score, 0) / total;
  const overallSentiment =
    avgScore > 0.15 ? 'positive' : avgScore < -0.15 ? 'negative' : 'neutral';

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return <TrendingUp className="w-4 h-4 text-green-600" />;
      case 'negative':
        return <TrendingDown className="w-4 h-4 text-red-600" />;
      default:
        return <Minus className="w-4 h-4 text-gray-600" />;
    }
  };

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'positive':
        return 'text-green-600 bg-green-50 border-green-200';
      case 'negative':
        return 'text-red-600 bg-red-50 border-red-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <span className="card-label">Latest News & Sentiment</span>
        {isAnalyzingSentiment && (
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: 'var(--accent)' }} />
            <span className="text-xs" style={{ color: 'var(--text-4)' }}>Analyzing sentiment...</span>
          </div>
        )}
      </div>

      {/* Overall Sentiment Summary */}
      <div className="mb-6 p-4 border-2" style={{
        background: 'var(--bg-2)',
        borderColor: 'var(--info)',
        borderLeftWidth: '3px'
      }}>
        <h4 className="font-semibold mb-1" style={{ color: 'var(--text-1)' }}>News Sentiment Consensus</h4>
        <p className="text-xs mb-3" style={{ color: 'var(--text-4)', fontStyle: 'italic' }}>
          AI-powered sentiment analysis. Results may not always be accurate and should be used as one of many factors in decision-making.
        </p>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold" style={{ color: 'var(--success)' }}>
              {positiveSentiments}/{total}
            </div>
            <div className="text-sm" style={{ color: 'var(--text-3)' }}>Positive</div>
            <div className="text-xs" style={{ color: 'var(--text-4)' }}>
              {((positiveSentiments / total) * 100).toFixed(0)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold" style={{ color: 'var(--danger)' }}>
              {negativeSentiments}/{total}
            </div>
            <div className="text-sm" style={{ color: 'var(--text-3)' }}>Negative</div>
            <div className="text-xs" style={{ color: 'var(--text-4)' }}>
              {((negativeSentiments / total) * 100).toFixed(0)}%
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold" style={{ color: 'var(--text-3)' }}>
              {neutralSentiments}/{total}
            </div>
            <div className="text-sm" style={{ color: 'var(--text-3)' }}>Neutral</div>
            <div className="text-xs" style={{ color: 'var(--text-4)' }}>
              {((neutralSentiments / total) * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        <div
          className="p-3 border-2"
          style={{
            background: 'var(--bg-3)',
            borderColor: overallSentiment === 'positive' ? 'var(--success)' : overallSentiment === 'negative' ? 'var(--danger)' : 'var(--bg-1)',
            borderLeftWidth: '3px',
            color: overallSentiment === 'positive' ? 'var(--success)' : overallSentiment === 'negative' ? 'var(--danger)' : 'var(--text-3)'
          }}
        >
          <div className="flex items-center gap-2">
            {getSentimentIcon(overallSentiment)}
            <span className="font-semibold">
              {overallSentiment === 'positive'
                ? 'Bullish Sentiment'
                : overallSentiment === 'negative'
                ? 'Bearish Sentiment'
                : 'Neutral Sentiment'}
            </span>
          </div>
          <p className="text-sm mt-1">
            {overallSentiment === 'positive'
              ? `Market news suggests positive momentum with ${positiveSentiments} supportive articles.`
              : overallSentiment === 'negative'
              ? `Market news suggests negative pressure with ${negativeSentiments} concerning articles.`
              : 'Market news shows balanced or unclear direction.'}
          </p>
        </div>
      </div>

      {/* News Articles */}
      <div className="space-y-3 max-h-[600px] overflow-y-auto">
        {articles.map((article, index) => {
          const sentiment = sentiments[index];
          return (
            <div
              key={index}
              className="p-4 border-2"
              style={{
                background: 'var(--bg-2)',
                borderColor: sentiment.sentiment === 'positive' ? 'var(--success)' : sentiment.sentiment === 'negative' ? 'var(--danger)' : 'var(--bg-1)',
                borderLeftWidth: '3px'
              }}
            >
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-2">
                    {getSentimentIcon(sentiment.sentiment)}
                    <h4 className="font-semibold text-sm line-clamp-2" style={{ color: 'var(--text-2)' }}>
                      {article.title}
                    </h4>
                  </div>
                  <p className="text-sm mb-2 line-clamp-2" style={{ color: 'var(--text-3)' }}>
                    {article.description}
                  </p>
                  <div className="flex items-center justify-between text-xs" style={{ color: 'var(--text-4)' }}>
                    <span>{article.source}</span>
                    <span>{new Date(article.publishedAt).toLocaleDateString()}</span>
                  </div>
                  <div className="mt-2 flex items-center gap-2">
                    <span
                      className="text-xs px-2 py-1 border"
                      style={{
                        background: 'var(--bg-3)',
                        borderColor: sentiment.sentiment === 'positive' ? 'var(--success)' : sentiment.sentiment === 'negative' ? 'var(--danger)' : 'var(--bg-1)',
                        color: sentiment.sentiment === 'positive' ? 'var(--success)' : sentiment.sentiment === 'negative' ? 'var(--danger)' : 'var(--text-3)'
                      }}
                    >
                      {sentiment.confidence.toFixed(0)}% confidence
                    </span>
                  </div>
                </div>
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex-shrink-0"
                  style={{ color: 'var(--info)' }}
                >
                  <ExternalLink className="w-4 h-4" />
                </a>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
