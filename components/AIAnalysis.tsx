'use client';

import React, { useState, useRef, useCallback } from 'react';
import { Brain, Sparkles, RefreshCw, AlertCircle, TrendingUp, TrendingDown, Minus, ShieldAlert, Zap } from 'lucide-react';

interface AnalysisData {
  summary: string;
  valuation: {
    pe_note: string;
    range_note: string;
    target_note: string;
    verdict: 'undervalued' | 'fair' | 'overvalued' | 'unknown';
  };
  technicals: {
    rsi_note: string;
    macd_note: string;
    ma_note: string;
    bb_note: string;
    outlook: 'bullish' | 'bearish' | 'mixed';
    outlook_reason: string;
  };
  forecasts: {
    note: string;
    ml_consensus: 'bullish' | 'bearish' | 'mixed' | 'inconclusive';
  };
  risks: string[];
  opportunities: string[];
  verdict: {
    rating: 'STRONG BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG SELL';
    confidence: 'Low' | 'Medium' | 'High';
    target_low: number | null;
    target_high: number | null;
    stop_loss: number | null;
    reasoning: string;
    suitable_for: string;
  };
}

interface AIAnalysisProps {
  symbol: string;
  companyName: string;
  currentPrice: number;
  companyInfo: any;
  fundamentalsData: any;
  tradingSignal: any;
  forecastInsights: any;
  mlPredictions: any;
  newsArticles: any[];
  newsSentiments: any[];
  chartData: any[];
  chartPatterns: any[];
}

// ── helpers ──────────────────────────────────────────────────────────────────

function parseAnalysisJSON(raw: string): AnalysisData | null {
  // Strip code fences if model wrapped the JSON anyway
  const stripped = raw.replace(/^```(?:json)?\s*/i, '').replace(/\s*```\s*$/, '').trim();
  try {
    return JSON.parse(stripped) as AnalysisData;
  } catch {
    // Try extracting the first {...} block
    const match = stripped.match(/\{[\s\S]*\}/);
    if (match) {
      try { return JSON.parse(match[0]) as AnalysisData; } catch { /* fall through */ }
    }
    return null;
  }
}

const ratingConfig: Record<string, { label: string; fg: string; bg: string; border: string }> = {
  'STRONG BUY': { label: 'STRONG BUY', fg: 'var(--success)', bg: 'rgba(34,197,94,0.12)', border: 'var(--success)' },
  'BUY':        { label: 'BUY',         fg: '#4ade80',        bg: 'rgba(74,222,128,0.10)', border: '#4ade80' },
  'HOLD':       { label: 'HOLD',        fg: 'var(--accent)',  bg: 'rgba(234,179,8,0.10)',  border: 'var(--accent)' },
  'SELL':       { label: 'SELL',        fg: '#f87171',        bg: 'rgba(248,113,113,0.10)',border: '#f87171' },
  'STRONG SELL':{ label: 'STRONG SELL', fg: 'var(--danger)',  bg: 'rgba(239,68,68,0.12)',  border: 'var(--danger)' },
};

const outlookConfig = {
  bullish: { label: 'BULLISH', color: 'var(--success)', icon: TrendingUp },
  bearish: { label: 'BEARISH', color: 'var(--danger)',  icon: TrendingDown },
  mixed:   { label: 'MIXED',   color: 'var(--accent)',  icon: Minus },
};

const verdictConfig = {
  undervalued: { label: 'Undervalued', color: 'var(--success)' },
  fair:        { label: 'Fair Value',  color: 'var(--accent)' },
  overvalued:  { label: 'Overvalued',  color: 'var(--danger)' },
  unknown:     { label: 'Unknown',     color: 'var(--text-4)' },
};

const consensusConfig = {
  bullish:      { color: 'var(--success)' },
  bearish:      { color: 'var(--danger)' },
  mixed:        { color: 'var(--accent)' },
  inconclusive: { color: 'var(--text-4)' },
};

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-xs font-bold uppercase tracking-widest mb-2" style={{ color: 'var(--text-4)' }}>
      {children}
    </div>
  );
}

function InfoRow({ label, value, valueColor }: { label: string; value: React.ReactNode; valueColor?: string }) {
  return (
    <div className="flex items-start justify-between gap-3 py-1 border-b" style={{ borderColor: 'var(--bg-1)' }}>
      <span className="text-xs flex-shrink-0" style={{ color: 'var(--text-4)' }}>{label}</span>
      <span className="text-xs text-right font-medium" style={{ color: valueColor || 'var(--text-2)' }}>{value}</span>
    </div>
  );
}

function Badge({ children, color, bg }: { children: React.ReactNode; color: string; bg: string }) {
  return (
    <span
      className="inline-block px-2 py-0.5 text-xs font-bold rounded-sm"
      style={{ color, background: bg, border: `1px solid ${color}` }}
    >
      {children}
    </span>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export default function AIAnalysis({
  symbol,
  companyName,
  currentPrice,
  companyInfo,
  fundamentalsData,
  tradingSignal,
  forecastInsights,
  mlPredictions,
  newsArticles,
  newsSentiments,
  chartData,
  chartPatterns,
}: AIAnalysisProps) {
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [rawBuffer, setRawBuffer] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasGenerated, setHasGenerated] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const generateAnalysis = useCallback(async () => {
    if (abortRef.current) abortRef.current.abort();
    abortRef.current = new AbortController();

    setIsLoading(true);
    setError('');
    setAnalysis(null);
    setRawBuffer('');

    const payload = {
      symbol, companyName, currentPrice, companyInfo, fundamentalsData,
      tradingSignal, forecastInsights, mlPredictions,
      newsArticles: (newsArticles || []).slice(0, 6),
      newsSentiments: (newsSentiments || []).slice(0, 6),
      chartData: (chartData || []).slice(-30),
      chartPatterns: (chartPatterns || []).slice(0, 8),
    };

    try {
      const response = await fetch('/api/ai-analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortRef.current.signal,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(errData.error || `Request failed: ${response.status}`);
      }

      // Buffer the full streaming response then parse JSON
      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        accumulated += decoder.decode(value, { stream: true });
        setRawBuffer(accumulated); // live char count for UX
      }

      const parsed = parseAnalysisJSON(accumulated);
      if (!parsed) throw new Error('AI returned an invalid response. Please try again.');

      setAnalysis(parsed);
      setHasGenerated(true);
    } catch (err: any) {
      if (err.name === 'AbortError') return;
      setError(err.message || 'Failed to generate analysis');
    } finally {
      setIsLoading(false);
    }
  }, [symbol, companyName, currentPrice, companyInfo, fundamentalsData, tradingSignal, forecastInsights, mlPredictions, newsArticles, newsSentiments, chartData, chartPatterns]);

  // ── Loading skeleton ──────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <span className="card-label">AI Analysis</span>
          <div className="flex items-center gap-1 px-2 py-0.5 border text-xs font-mono ml-auto"
            style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}>
            <Sparkles className="w-3 h-3" style={{ color: 'var(--accent)' }} />
            <span>deepseek-v3.1</span>
          </div>
        </div>
        <div className="flex flex-col items-center py-10 gap-4">
          <Brain className="w-10 h-10 animate-pulse" style={{ color: 'var(--accent)' }} />
          <div className="text-sm font-semibold" style={{ color: 'var(--text-2)' }}>
            Analyzing {companyName}...
          </div>
          <div className="text-xs" style={{ color: 'var(--text-4)' }}>
            {rawBuffer.length > 0 ? `Receiving data — ${rawBuffer.length} chars` : 'Waiting for DeepSeek V3.1 (671B)...'}
          </div>
          <div className="w-64 h-1 overflow-hidden" style={{ background: 'var(--bg-1)' }}>
            <div className="h-full animate-pulse" style={{ background: 'var(--accent)', width: '70%' }} />
          </div>
        </div>
      </div>
    );
  }

  // ── Not yet generated ──────────────────────────────────────────────────────
  if (!hasGenerated && !analysis) {
    return (
      <div className="card">
        <div className="flex items-center justify-between mb-3">
          <span className="card-label">AI Analysis</span>
          <div className="flex items-center gap-1 px-2 py-0.5 border text-xs font-mono"
            style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}>
            <Sparkles className="w-3 h-3" style={{ color: 'var(--accent)' }} />
            <span>deepseek-v3.1 · 671B</span>
          </div>
        </div>
        <p className="text-xs mb-4" style={{ color: 'var(--text-4)' }}>
          Comprehensive AI analysis using all available data — fundamentals, technicals, ML predictions, news sentiment, and chart patterns.
        </p>
        {error && (
          <div className="p-3 border mb-3 flex items-start gap-2"
            style={{ background: 'var(--bg-2)', borderColor: 'var(--danger)', borderLeftWidth: '3px' }}>
            <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" style={{ color: 'var(--danger)' }} />
            <span className="text-xs" style={{ color: 'var(--text-3)' }}>{error}</span>
          </div>
        )}
        <button
          onClick={generateAnalysis}
          className="w-full py-3 font-semibold text-sm flex items-center justify-center gap-2 border transition-opacity hover:opacity-90"
          style={{ background: 'var(--accent)', borderColor: 'var(--accent)', color: 'var(--text-0)' }}
        >
          <Brain className="w-4 h-4" />
          Generate AI Analysis for {symbol}
        </button>
      </div>
    );
  }

  // ── Rendered analysis ─────────────────────────────────────────────────────
  const d = analysis!;
  const rc = ratingConfig[d.verdict.rating] ?? ratingConfig['HOLD'];
  const oc = outlookConfig[d.technicals.outlook] ?? outlookConfig.mixed;
  const OutlookIcon = oc.icon;

  return (
    <div className="card">
      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="card-label">AI Analysis</span>
          <span className="text-xs px-2 py-0.5 border font-mono"
            style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}>
            {symbol}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1 px-2 py-0.5 border text-xs font-mono"
            style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}>
            <Sparkles className="w-3 h-3" style={{ color: 'var(--accent)' }} />
            <span>deepseek-v3.1</span>
          </div>
          <button
            onClick={generateAnalysis}
            className="p-1.5 border transition-opacity hover:opacity-70"
            style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)', color: 'var(--text-4)' }}
            title="Regenerate"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ── Summary ── */}
      <div className="p-3 mb-4 border-l-2" style={{ background: 'var(--bg-2)', borderColor: 'var(--accent)' }}>
        <p className="text-sm leading-relaxed" style={{ color: 'var(--text-2)' }}>{d.summary}</p>
      </div>

      {/* ── Row 1: Valuation + Technicals ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">

        {/* Valuation */}
        <div className="p-3 border" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
          <div className="flex items-center justify-between mb-2">
            <SectionLabel>Valuation</SectionLabel>
            <Badge
              color={verdictConfig[d.valuation.verdict].color}
              bg={`${verdictConfig[d.valuation.verdict].color}18`}
            >
              {verdictConfig[d.valuation.verdict].label}
            </Badge>
          </div>
          <div className="space-y-0">
            <InfoRow label="P/E Ratio" value={d.valuation.pe_note} />
            <InfoRow label="52W Range" value={d.valuation.range_note} />
            <InfoRow label="Analyst Target" value={d.valuation.target_note} />
          </div>
        </div>

        {/* Technicals */}
        <div className="p-3 border" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
          <div className="flex items-center justify-between mb-2">
            <SectionLabel>Technicals</SectionLabel>
            <div className="flex items-center gap-1">
              <OutlookIcon className="w-3.5 h-3.5" style={{ color: oc.color }} />
              <span className="text-xs font-bold" style={{ color: oc.color }}>{oc.label}</span>
            </div>
          </div>
          <div className="space-y-0">
            <InfoRow label="RSI" value={d.technicals.rsi_note} />
            <InfoRow label="MACD" value={d.technicals.macd_note} />
            <InfoRow label="Moving Avgs" value={d.technicals.ma_note} />
            <InfoRow label="Bollinger" value={d.technicals.bb_note} />
          </div>
          <p className="text-xs mt-2 pt-2 border-t leading-relaxed" style={{ color: 'var(--text-3)', borderColor: 'var(--bg-1)' }}>
            {d.technicals.outlook_reason}
          </p>
        </div>
      </div>

      {/* ── Forecasts & ML ── */}
      <div className="p-3 border mb-4" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
        <div className="flex items-center justify-between mb-2">
          <SectionLabel>Forecasts & ML Predictions</SectionLabel>
          <Badge
            color={consensusConfig[d.forecasts.ml_consensus].color}
            bg={`${consensusConfig[d.forecasts.ml_consensus].color}18`}
          >
            {d.forecasts.ml_consensus.toUpperCase()}
          </Badge>
        </div>
        <p className="text-xs leading-relaxed" style={{ color: 'var(--text-3)' }}>{d.forecasts.note}</p>
      </div>

      {/* ── Row 2: Risks + Opportunities ── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">

        {/* Risks */}
        <div className="p-3 border" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
          <div className="flex items-center gap-1.5 mb-2">
            <ShieldAlert className="w-3.5 h-3.5" style={{ color: 'var(--danger)' }} />
            <SectionLabel>Key Risks</SectionLabel>
          </div>
          <ul className="space-y-1.5">
            {(d.risks || []).map((risk, i) => (
              <li key={i} className="flex gap-2 items-start">
                <span className="flex-shrink-0 mt-1.5 w-1.5 h-1.5 rounded-full" style={{ background: 'var(--danger)' }} />
                <span className="text-xs leading-relaxed" style={{ color: 'var(--text-3)' }}>{risk}</span>
              </li>
            ))}
          </ul>
        </div>

        {/* Opportunities */}
        <div className="p-3 border" style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)' }}>
          <div className="flex items-center gap-1.5 mb-2">
            <Zap className="w-3.5 h-3.5" style={{ color: 'var(--success)' }} />
            <SectionLabel>Key Opportunities</SectionLabel>
          </div>
          <ul className="space-y-1.5">
            {(d.opportunities || []).map((opp, i) => (
              <li key={i} className="flex gap-2 items-start">
                <span className="flex-shrink-0 mt-1.5 w-1.5 h-1.5 rounded-full" style={{ background: 'var(--success)' }} />
                <span className="text-xs leading-relaxed" style={{ color: 'var(--text-3)' }}>{opp}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* ── Verdict ── */}
      <div className="p-4 border-2" style={{ background: rc.bg, borderColor: rc.border }}>
        <div className="flex flex-wrap items-start justify-between gap-3 mb-3">
          <div>
            <div className="text-xl font-black tracking-tight mb-0.5" style={{ color: rc.fg }}>
              {rc.label}
            </div>
            <div className="text-xs" style={{ color: 'var(--text-4)' }}>
              Confidence: <span className="font-semibold" style={{ color: 'var(--text-2)' }}>{d.verdict.confidence}</span>
            </div>
          </div>

          <div className="flex gap-3 flex-wrap">
            <div className="text-center">
              <div className="text-xs mb-0.5" style={{ color: 'var(--text-4)' }}>Target Range</div>
              <div className="text-sm font-bold" style={{ color: rc.fg }}>
                {d.verdict.target_low != null && d.verdict.target_high != null
                  ? `$${d.verdict.target_low} – $${d.verdict.target_high}`
                  : 'N/A'}
              </div>
            </div>
            <div className="text-center">
              <div className="text-xs mb-0.5" style={{ color: 'var(--text-4)' }}>Stop Loss</div>
              <div className="text-sm font-bold" style={{ color: 'var(--danger)' }}>
                {d.verdict.stop_loss != null ? `$${d.verdict.stop_loss}` : 'N/A'}
              </div>
            </div>
          </div>
        </div>

        <p className="text-xs leading-relaxed mb-2" style={{ color: 'var(--text-2)' }}>
          {d.verdict.reasoning}
        </p>
        <p className="text-xs" style={{ color: 'var(--text-4)' }}>
          <span className="font-semibold">Suitable for: </span>{d.verdict.suitable_for}
        </p>
      </div>

      {/* ── Disclaimer ── */}
      <p className="text-xs mt-3 text-center" style={{ color: 'var(--text-5)' }}>
        Educational AI analysis only — not financial advice. Always consult a licensed advisor before investing.
      </p>
    </div>
  );
}
