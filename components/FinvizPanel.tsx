'use client';

import { useEffect } from 'react';

type FinvizTable = {
  headers: string[];
  rows: Record<string, string>[];
};

type FinvizData = {
  success: boolean;
  error?: string;
  symbol: string;
  delayedQuoteNotice?: string;
  stock: Record<string, string | null>;
  news: Array<{ timestamp: string; headline: string; url: string; source: string }>;
  insider: Record<string, string>[];
  analystTargets: Array<{ date: string; category: string; analyst: string; rating: string; target: string }>;
  screener: Record<string, FinvizTable>;
  charts: Record<string, string>;
  links: Record<string, string>;
  timestamp: string;
};

interface FinvizPanelProps {
  symbol: string;
  inlineMobile?: boolean;
  onStockData?: (stock: Record<string, string | null>) => void;
  onAnalystTargets?: (targets: FinvizData['analystTargets']) => void;
  onNewsData?: (news: FinvizData['news']) => void;
  onChartsData?: (charts: Record<string, string>, links: Record<string, string>) => void;
}

export default function FinvizPanel({ symbol, onStockData, onAnalystTargets, onNewsData, onChartsData }: FinvizPanelProps) {
  const fetchFinviz = async () => {
    if (!symbol) return;

    try {
      const response = await fetch(`/api/finviz?symbol=${encodeURIComponent(symbol)}`);
      const result = await response.json();

      if (!response.ok || !result.success) return;

      if (onStockData && result.stock) onStockData(result.stock);
      if (onAnalystTargets && result.analystTargets) onAnalystTargets(result.analystTargets);
      if (onNewsData && result.news) onNewsData(result.news);
      if (onChartsData && result.charts) onChartsData(result.charts, result.links || {});
    } catch {
      // silently fail — data is supplementary
    }
  };

  useEffect(() => {
    fetchFinviz();
  }, [symbol]);

  return null;
}
