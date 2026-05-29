'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Target, RefreshCw, Loader2, TrendingUp, Info } from 'lucide-react';
import { StockData } from '@/types';
import { fetchJSON, invalidateCache } from '@/lib/clientFetch';
import { getLogForSymbol } from '@/lib/predictionLog';
import {
  scoreLoggedPredictions,
  holdoutBacktest,
  gradeColor,
  modelLabel,
  ModelScore,
  BacktestScore,
  Generator,
} from '@/lib/predictionScoring';
import {
  generateLinearRegression,
  generateEMAForecast,
  generateARIMAForecast,
  generateProphetLiteForecast,
} from '@/lib/mlAlgorithms';

// Only the cheap, synchronous models run in the instant holdout backtest.
// Neural-net models (LSTM/GRU/CNN-LSTM/Ensemble) are too heavy to retrain here;
// they appear under Live tracking once their logged forecasts resolve.
const HOLDOUT_GENERATORS: Record<string, Generator> = {
  linearRegression: generateLinearRegression,
  ema: generateEMAForecast,
  arima: generateARIMAForecast,
  prophetLite: generateProphetLiteForecast,
};

const HOLDOUT_TEST_DAYS = 10;

interface Props {
  symbol: string;
  inlineMobile?: boolean;
}

function GradeBadge({ grade }: { grade: ModelScore['grade'] }) {
  return (
    <span
      className="inline-block text-center font-bold rounded"
      style={{
        minWidth: '22px',
        padding: '1px 6px',
        fontSize: '11px',
        color: 'var(--text-0)',
        background: gradeColor(grade),
      }}
    >
      {grade}
    </span>
  );
}

export default function PredictionScorecard({ symbol }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveScores, setLiveScores] = useState<ModelScore[]>([]);
  const [backtest, setBacktest] = useState<BacktestScore[]>([]);
  const [trackedRuns, setTrackedRuns] = useState(0);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);

  const compute = useCallback(async (sym: string, forceFresh = false) => {
    if (!sym) return;
    setLoading(true);
    setError(null);
    const url = `/api/stock?symbol=${encodeURIComponent(sym)}&days=400&interval=1d`;
    if (forceFresh) invalidateCache(url);
    try {
      const result = await fetchJSON<{ data?: StockData[]; error?: string }>(url, {
        ttlMs: 5 * 60 * 1000,
        retries: 2,
      });
      if (result.error) throw new Error(result.error);
      const data = result.data ?? [];
      if (data.length < 60) {
        throw new Error('Not enough history to score this ticker.');
      }

      setBacktest(holdoutBacktest(data, HOLDOUT_GENERATORS, HOLDOUT_TEST_DAYS));

      const log = getLogForSymbol(sym);
      setTrackedRuns(log.length);
      setLiveScores(scoreLoggedPredictions(log, data));
      setUpdatedAt(Date.now());
    } catch (e: any) {
      setError(e?.message || 'Failed to score predictions');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    compute(symbol);
  }, [symbol, compute]);

  const resolvedScores = liveScores.filter((s) => s.resolved > 0);
  const pendingPoints = liveScores.reduce((sum, s) => sum + s.pending, 0);
  const bestBacktest = backtest[0];

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Target className="w-5 h-5" style={{ color: 'var(--accent)' }} />
          <span className="card-label">Prediction Scorecard</span>
        </div>
        <button
          onClick={() => compute(symbol, true)}
          disabled={loading}
          className="flex items-center gap-1.5 px-2 py-1 text-xs border transition-all disabled:opacity-50"
          style={{ background: 'var(--bg-4)', borderColor: 'var(--bg-1)', color: 'var(--text-3)' }}
          title="Recompute scores"
        >
          {loading ? <Loader2 className="w-3 h-3 animate-spin" /> : <RefreshCw className="w-3 h-3" />}
          Refresh
        </button>
      </div>

      {error && (
        <div className="text-sm py-3" style={{ color: 'var(--danger)' }}>
          {error}
        </div>
      )}

      {!error && (
        <>
          {/* Live forward tracking */}
          <div className="mb-4">
            <div className="flex items-center gap-2 mb-2">
              <TrendingUp className="w-4 h-4" style={{ color: 'var(--text-3)' }} />
              <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-3)' }}>
                Live forward tracking
              </span>
            </div>

            {resolvedScores.length > 0 ? (
              <ScoreTable
                rows={resolvedScores.map((s) => ({
                  model: modelLabel(s.model),
                  primary: `${s.directionalAccuracy.toFixed(0)}%`,
                  secondary: `${s.mape.toFixed(1)}%`,
                  count: `${s.resolved}${s.pending ? ` (+${s.pending})` : ''}`,
                  grade: s.grade,
                }))}
                countHeader="Resolved"
              />
            ) : (
              <div
                className="text-xs px-3 py-3 flex items-start gap-2"
                style={{ background: 'var(--bg-3)', color: 'var(--text-4)' }}
              >
                <Info className="w-4 h-4 mt-0.5 flex-shrink-0" style={{ color: 'var(--text-4)' }} />
                <span>
                  {trackedRuns > 0
                    ? `Tracking ${trackedRuns} forecast run${trackedRuns > 1 ? 's' : ''} for ${symbol} · ${pendingPoints} point${pendingPoints === 1 ? '' : 's'} pending. Models are graded here as each forecast's target date arrives.`
                    : `No forecasts logged for ${symbol} yet. Run the ML Predictions panel — each run is recorded and graded against the real close once its target date passes.`}
                </span>
              </div>
            )}
          </div>

          {/* Instant holdout backtest */}
          <div>
            <div className="flex items-center gap-2 mb-2">
              <Target className="w-4 h-4" style={{ color: 'var(--text-3)' }} />
              <span className="text-xs font-semibold uppercase tracking-wide" style={{ color: 'var(--text-3)' }}>
                Holdout backtest · last {HOLDOUT_TEST_DAYS} sessions
              </span>
            </div>

            {loading && backtest.length === 0 ? (
              <div className="flex items-center gap-2 text-xs py-3" style={{ color: 'var(--text-4)' }}>
                <Loader2 className="w-4 h-4 animate-spin" /> Scoring models…
              </div>
            ) : backtest.length > 0 ? (
              <ScoreTable
                rows={backtest.map((s) => ({
                  model: modelLabel(s.model),
                  primary: `${s.directionalAccuracy.toFixed(0)}%`,
                  secondary: `${s.mape.toFixed(1)}%`,
                  count: String(s.testedPoints),
                  grade: s.grade,
                }))}
                countHeader="Points"
              />
            ) : (
              <div className="text-xs py-3" style={{ color: 'var(--text-4)' }}>
                Not enough history for a holdout backtest.
              </div>
            )}
          </div>

          {/* Footer */}
          <div
            className="mt-3 pt-2 text-[10px] leading-relaxed"
            style={{ borderTop: '1px solid var(--bg-1)', color: 'var(--text-5)' }}
          >
            <p>
              <span style={{ color: 'var(--text-4)' }}>Hit-rate</span> = % of forecasts that called
              direction (up/down vs. price at forecast time) correctly.{' '}
              <span style={{ color: 'var(--text-4)' }}>Error</span> = mean absolute % price error.
              {bestBacktest && resolvedScores.length === 0 && (
                <> Best backtested model: <strong style={{ color: 'var(--text-3)' }}>{modelLabel(bestBacktest.model)}</strong>.</>
              )}
            </p>
            {updatedAt && (
              <p className="mt-1">Updated {new Date(updatedAt).toLocaleTimeString()}</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}

interface Row {
  model: string;
  primary: string;
  secondary: string;
  count: string;
  grade: ModelScore['grade'];
}

function ScoreTable({ rows, countHeader }: { rows: Row[]; countHeader: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs" style={{ borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--bg-1)' }}>
            <th className="text-left py-1.5 px-2" style={{ color: 'var(--text-4)' }}>Model</th>
            <th className="text-right py-1.5 px-2" style={{ color: 'var(--text-4)' }}>Hit-rate</th>
            <th className="text-right py-1.5 px-2" style={{ color: 'var(--text-4)' }}>Error</th>
            <th className="text-right py-1.5 px-2" style={{ color: 'var(--text-4)' }}>{countHeader}</th>
            <th className="text-center py-1.5 px-2" style={{ color: 'var(--text-4)' }}>Grade</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={r.model} style={{ borderBottom: i < rows.length - 1 ? '1px solid var(--bg-1)' : 'none' }}>
              <td className="py-1.5 px-2 font-medium" style={{ color: 'var(--text-2)' }}>{r.model}</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: 'var(--text-2)' }}>{r.primary}</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: 'var(--text-4)' }}>{r.secondary}</td>
              <td className="py-1.5 px-2 text-right font-mono" style={{ color: 'var(--text-4)' }}>{r.count}</td>
              <td className="py-1.5 px-2 text-center"><GradeBadge grade={r.grade} /></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
