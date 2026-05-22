'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  ComposedChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, ReferenceArea,
} from 'recharts';
import {
  TrendingUp, TrendingDown, Loader2, Brain, Zap,
  RotateCcw, ArrowUp, ArrowDown, Minus,
} from 'lucide-react';

// ── Horizon config ─────────────────────────────────────────────────────────────

const HORIZONS = [
  // moveScale controls the target-price offset: confStrength * price * moveScale
  // Calibrated to typical absolute price move for that horizon
  { id: '1m',  label: '1m',  resolveMs:  1*60_000, pollMs:  5_000, minPts:  8, maxPts: 200, preInterval: '1m',  preDays:  3, preHorizon: 1, moveScale: 0.0015 },
  { id: '5m',  label: '5m',  resolveMs:  5*60_000, pollMs: 10_000, minPts: 10, maxPts: 300, preInterval: '1m',  preDays:  7, preHorizon: 5, moveScale: 0.003  },
  { id: '15m', label: '15m', resolveMs: 15*60_000, pollMs: 20_000, minPts:  8, maxPts: 250, preInterval: '5m',  preDays: 90, preHorizon: 3, moveScale: 0.005  },
  { id: '30m', label: '30m', resolveMs: 30*60_000, pollMs: 30_000, minPts:  6, maxPts: 200, preInterval: '15m', preDays: 90, preHorizon: 2, moveScale: 0.007  },
  { id: '1h',  label: '1h',  resolveMs: 60*60_000, pollMs: 60_000, minPts:  5, maxPts: 150, preInterval: '60m', preDays: 60, preHorizon: 1, moveScale: 0.010  },
] as const;

type HorizonId = typeof HORIZONS[number]['id'];
type Horizon   = typeof HORIZONS[number];

const DEFAULT_HORIZON: HorizonId = '5m';
const PRETRAIN_TTL_MS  = 12 * 3600_000;
const REPLAY_SIZE      = 250;    // max training examples kept per symbol/horizon
const BATCH_SIZE       = 32;     // mini-batch size for gradient update

// ── Types ──────────────────────────────────────────────────────────────────────

interface PricePoint { time: number; price: number; volume?: number; }
interface Signal     { name: string; dir: 'up' | 'down'; strength: number; }

interface Prediction {
  id: string;
  madeAt: number; resolveAt: number;
  basePrice: number; targetPrice: number;
  direction: 'up' | 'down';
  confidence: number; volBoost: number;
  resolved: boolean; weightUpdated: boolean;
  correct?: boolean; actualPrice?: number;
  priceSnapshot: number[];
  signals?: Signal[];
  winnerModel?: SubModelId;
}

// Replay buffer entry: stores pre-computed features + outcome for batch training
interface ReplayEntry {
  f: number[]; // feature vector (FEAT_KEYS order, length 11)
  base: number;
  actual: number;
}

// ── Model: 3 sub-models + ensemble + replay ───────────────────────────────────

// 12 features — 9 original + vol_regime, vwap_dev, macd
interface Features {
  m1: number; m3: number; m6: number; m10: number;
  rsi: number; bb: number; tc: number; acc: number;
  rng: number;
  vol_regime: number; // short-term vol relative to medium-term (expansion = +1)
  vwap_dev: number;   // deviation from session VWAP (above = +, below = -)
  macd: number;       // fast EMA minus slow EMA, price-normalized (momentum divergence)
}

interface Weights {
  m1: number; m3: number; m6: number; m10: number;
  rsi: number; bb: number; tc: number; acc: number;
  rng: number; vol_regime: number; vwap_dev: number; macd: number; bias: number;
}

// Ordered key list used to iterate over features/weights generically
const FEAT_KEYS = ['m1','m3','m6','m10','rsi','bb','tc','acc','rng','vol_regime','vwap_dev','macd'] as const;
const W_KEYS    = [...FEAT_KEYS, 'bias'] as const;
type FKey = typeof FEAT_KEYS[number];
type WKey = typeof W_KEYS[number];

// Sub-model init priors — each starts with a different "belief" about the market
const SUB_INIT = {
  trend: {
    m1:0.12, m3:0.18, m6:0.28, m10:0.22,
    rsi:-0.08, bb:-0.06, tc:0.30, acc:0.06,
    rng:-0.05, vol_regime:-0.06, vwap_dev:-0.06, macd:0.18, bias:0.0,
  },
  momentum: {
    m1:0.38, m3:0.28, m6:0.12, m10:0.06,
    rsi:-0.14, bb:-0.08, tc:0.12, acc:0.22,
    rng:-0.08, vol_regime:0.10, vwap_dev:-0.04, macd:0.24, bias:0.0,
  },
  reversal: {
    m1:0.10, m3:0.10, m6:0.08, m10:0.04,
    rsi:-0.22, bb:-0.24, tc:0.08, acc:0.06,
    rng:-0.20, vol_regime:-0.12, vwap_dev:-0.16, macd:-0.08, bias:0.0,
  },
  mean_rev: {
    m1:-0.06, m3:-0.10, m6:-0.06, m10:-0.02,
    rsi:-0.32, bb:-0.30, tc:-0.08, acc:-0.06,
    rng:-0.26, vol_regime:-0.10, vwap_dev:-0.22, macd:-0.14, bias:0.0,
  },
} satisfies Record<string, Weights>;

type SubModelId = keyof typeof SUB_INIT;
const SUB_IDS   = Object.keys(SUB_INIT) as SubModelId[];
const SUB_LABEL: Record<SubModelId, string> = { trend:'Trend', momentum:'Mom', reversal:'Rev', mean_rev:'MRev' };

interface SubModelState {
  weights: Weights;
  history: boolean[]; // last 30 resolved outcomes for adaptive LR
}

type EnsembleState = Record<SubModelId, SubModelState>;

function initEnsemble(): EnsembleState {
  return Object.fromEntries(
    SUB_IDS.map(id => [id, { weights: { ...SUB_INIT[id] }, history: [] }])
  ) as EnsembleState;
}

// ── Helpers ────────────────────────────────────────────────────────────────────

function clip(v: number, lo = -1, hi = 1) { return Math.max(lo, Math.min(hi, v)); }

// Convert Features object → number[] (deterministic order via FEAT_KEYS)
const featToArr = (f: Features): number[] => FEAT_KEYS.map(k => f[k]);

// Convert number[] → Features (backwards-compatible: missing indices default to 0)
const arrToFeat = (arr: number[]): Features =>
  Object.fromEntries(FEAT_KEYS.map((k, i) => [k, arr[i] ?? 0])) as unknown as Features;

// Approximate session VWAP from tracked PricePoints
function computeVWAP(pts: PricePoint[]): number | null {
  const valid = pts.filter(p => (p.volume ?? 0) > 0);
  if (valid.length < 3) return null;
  const sumPV = valid.reduce((s, p) => s + p.price * p.volume!, 0);
  const sumV  = valid.reduce((s, p) => s + p.volume!, 0);
  return sumV > 0 ? sumPV / sumV : null;
}

// EMA helper — used for MACD feature
function emaOf(arr: number[], period: number): number {
  const k = 2 / (period + 1);
  return arr.reduce((e, v) => v * k + e * (1 - k), arr[0]);
}

// ── Feature extraction ─────────────────────────────────────────────────────────

function extractFeatures(prices: number[], vwap?: number | null): Features | null {
  const n = prices.length;
  if (n < 6) return null;
  const last = prices[n - 1];

  // Momentum at 4 lookback scales
  const pct = (a: number, b: number) => clip((b - a) / a, -0.05, 0.05) * 20;
  const m1  = pct(prices[n - 2], last);
  const m3  = n >=  4 ? pct(prices[n - 4],  last) : m1;
  const m6  = n >=  7 ? pct(prices[n - 7],  last) : m3;
  const m10 = n >= 11 ? pct(prices[n - 11], last) : m6;

  // RSI-14 (Wilder avg-gain/avg-loss)
  const rsiP = Math.min(14, n - 1);
  let avgG = 0, avgL = 0;
  for (let i = n - rsiP; i < n; i++) {
    const d = prices[i] - prices[i - 1];
    if (d > 0) avgG += d; else avgL -= d;
  }
  avgG /= rsiP; avgL /= rsiP;
  const rsi  = 100 - 100 / (1 + (avgL === 0 ? 999 : avgG / avgL));
  const rsiN = clip((rsi - 50) / 50);

  // Bollinger Band z-score (20-bar)
  const sl  = prices.slice(-Math.min(20, n));
  const ma  = sl.reduce((a, b) => a + b, 0) / sl.length;
  const std = Math.sqrt(sl.reduce((s, x) => s + (x - ma) ** 2, 0) / sl.length);
  const bb  = std > 0 ? clip((last - ma) / (std * 2)) : 0;

  // Trend consistency
  const dir6 = m6 >= 0 ? 1 : -1;
  const lb   = Math.min(6, n - 1);
  let cons = 0;
  for (let i = n - lb; i < n; i++) if ((prices[i] - prices[i - 1]) * dir6 > 0) cons++;
  const tc = clip((cons / lb - 0.5) * 2);

  // Momentum acceleration
  const m1p = n >= 3 ? pct(prices[n - 3], prices[n - 2]) : 0;
  const acc  = clip((m1 - m1p) * 5);

  // Range position (-1 = near 20-bar low, +1 = near 20-bar high)
  const rl = prices.slice(-Math.min(20, n));
  const hi = Math.max(...rl), lo = Math.min(...rl);
  const rng = hi > lo ? clip((last - lo) / (hi - lo) * 2 - 1) : 0;

  // Volatility regime: short-term vol vs medium-term vol
  // Positive = vol expanding (momentum may be more reliable)
  // Negative = vol contracting (mean reversion may be better)
  const retSlice = prices.slice(-Math.min(20, n));
  const returns  = retSlice.slice(1).map((p, i) => (p - retSlice[i]) / retSlice[i]);
  const shortVar = returns.slice(-5).reduce((s, r) => s + r * r, 0) / Math.max(1, Math.min(5, returns.length));
  const longVar  = returns.reduce((s, r) => s + r * r, 0) / Math.max(1, returns.length);
  const vol_regime = longVar > 1e-10 ? clip((shortVar / longVar - 1) * 2) : 0;

  // VWAP deviation: above session VWAP = slightly extended (slightly bearish)
  const vwap_dev = (vwap != null && vwap > 0)
    ? clip((last - vwap) / last * 1000) // 0.1% above VWAP → +1.0
    : 0;

  // MACD-like: fast EMA vs slow EMA, price-normalized — catches momentum divergence/convergence
  const fastP  = Math.min(6,  Math.max(2, Math.ceil(n / 3)));
  const slowP  = Math.min(13, Math.max(3, Math.ceil(n / 2)));
  const macdRaw = emaOf(prices.slice(-Math.min(fastP * 2, n)), fastP)
                - emaOf(prices.slice(-Math.min(slowP * 2, n)), slowP);
  const macd = last > 0 ? clip(macdRaw / last * 200) : 0;

  return { m1, m3, m6, m10, rsi: rsiN, bb, tc, acc, rng, vol_regime, vwap_dev, macd };
}

// ── Sub-model scoring ──────────────────────────────────────────────────────────

function subScore(prices: number[], w: Weights, vwap?: number | null): number | null {
  const f = extractFeatures(prices, vwap);
  if (!f) return null;
  return FEAT_KEYS.reduce((sum, k) => sum + w[k] * f[k], w.bias);
}

// ── Ensemble prediction (accuracy-weighted vote) ───────────────────────────────

function ensemblePredict(prices: number[], ens: EnsembleState, volRatio: number | null, vwap?: number | null, moveScale = 0.003) {
  const last = prices[prices.length - 1];

  // Exponential decay accuracy weight — recent results count more, older ones fade
  const accW = (hist: boolean[]) => {
    if (hist.length < 3) return 0.5;
    const recent = hist.slice(-20);
    const decay = 0.88;
    let sumW = 0, sumCorr = 0, w = 1.0;
    for (let i = recent.length - 1; i >= 0; i--) {
      if (recent[i]) sumCorr += w;
      sumW += w;
      w *= decay;
    }
    return sumCorr / sumW;
  };

  const items = SUB_IDS.map(id => ({
    id,
    score: subScore(prices, ens[id].weights, vwap),
    w:     accW(ens[id].history),
  })).filter(x => x.score !== null) as { id: SubModelId; score: number; w: number }[];

  if (items.length === 0) return null;

  const totalW = items.reduce((s, x) => s + x.w, 0);
  const wScore = items.reduce((s, x) => s + (x.w / totalW) * x.score, 0);

  let confidence = Math.min(0.99, Math.max(0.01, 1 / (1 + Math.exp(-wScore * 10))));

  let volBoost = 0;
  if (volRatio !== null && volRatio > 0) {
    volBoost = clip((volRatio - 1) * 0.10, -0.10, 0.10);
    confidence = clip(confidence + volBoost, 0.01, 0.99);
  }

  // Adaptive gate: loosen when model is hot, tighten when cold
  const avgAcc = items.reduce((s, x) => s + x.w, 0) / items.length;
  const threshold = avgAcc > 0.62 ? 0.04 : avgAcc > 0.55 ? 0.05 : avgAcc < 0.43 ? 0.08 : 0.06;
  if (Math.abs(confidence - 0.5) < threshold) return null;

  const direction: 'up' | 'down' = confidence >= 0.5 ? 'up' : 'down';
  const confStrength = Math.abs(confidence - 0.5);
  const move = confStrength * last * moveScale;
  const targetPrice = direction === 'up' ? last + move : last - move;

  const winner = items.reduce((best, x) =>
    Math.abs(x.score) > Math.abs(best.score) ? x : best, items[0]);

  const f = extractFeatures(prices, vwap)!;
  const avgW: Weights = Object.fromEntries(
    W_KEYS.map(k => [k, items.reduce((s, x) => s + ens[x.id].weights[k] * x.w, 0) / totalW])
  ) as unknown as Weights;

  const contribs = [
    { name: 'Momentum', v: avgW.m1*f.m1 + avgW.m3*f.m3 + avgW.m6*f.m6 },
    { name: 'RSI',      v: avgW.rsi*f.rsi },
    { name: 'Bollinger',v: avgW.bb*f.bb },
    { name: 'Trend',    v: avgW.tc*f.tc },
    { name: 'Accel',    v: avgW.acc*f.acc },
    { name: 'Range',    v: avgW.rng*f.rng },
    { name: 'VolReg',   v: avgW.vol_regime*f.vol_regime },
    { name: 'VWAP',     v: avgW.vwap_dev*f.vwap_dev },
    ...(volBoost !== 0 ? [{ name: 'Volume', v: volBoost * 2 }] : []),
  ];
  const signals: Signal[] = contribs
    .filter(c => Math.abs(c.v) > 0.005)
    .sort((a, b) => Math.abs(b.v) - Math.abs(a.v))
    .slice(0, 3)
    .map(c => ({ name: c.name, dir: c.v > 0 ? 'up' : 'down', strength: Math.min(1, Math.abs(c.v) * 10) }));

  return { direction, confidence, volBoost, targetPrice, signals, winnerModel: winner.id };
}

// ── Batch gradient descent on replay buffer ────────────────────────────────────

function subModelLR(hist: boolean[]): number {
  const n = hist.length;
  if (n < 3) return 0.10;
  const r = hist.slice(-10).filter(x => x).length / Math.min(10, n) * 100;
  if (r < 42) return 0.10;
  if (r > 68) return 0.015;
  if (r > 55) return 0.025;
  return 0.05;
}

function batchLearnSubModel(
  w: Weights, initW: Weights,
  buffer: ReplayEntry[],
  history: boolean[],
): Weights {
  if (buffer.length === 0) return w;

  // Mini-batch: 70% from most recent, 30% from older entries for diversity
  const recentN = Math.min(Math.floor(BATCH_SIZE * 0.7), buffer.length);
  const olderN  = Math.min(Math.ceil(BATCH_SIZE * 0.3), Math.max(0, buffer.length - recentN));
  const recent  = buffer.slice(-recentN);
  const older   = buffer.slice(0, buffer.length - recentN).slice(-olderN * 3);
  // Sample older entries evenly
  const olderSampled = older.filter((_, i) => i % Math.max(1, Math.floor(older.length / olderN)) === 0).slice(-olderN);
  const batch   = [...olderSampled, ...recent];

  const lr    = subModelLR(history) * 0.4; // scale down: batch gradients are more stable
  const decay = 0.002;

  // Accumulate gradients over the batch
  const sumG = Object.fromEntries(W_KEYS.map(k => [k, 0])) as Record<WKey, number>;
  let n = 0;

  for (const { f: fArr, base, actual } of batch) {
    const f   = arrToFeat(fArr);
    const dir = actual > base ? 1 : -1;
    const mag = clip(Math.abs((actual - base) / base) * 100, 0, 1);
    const sig = dir * (0.35 + 0.65 * mag);

    for (const k of W_KEYS) {
      const fk = k === 'bias' ? null : k as FKey;
      sumG[k] += sig * (fk ? (f[fk] as number) : 0.1);
    }
    n++;
  }

  const newW = { ...w };
  for (const k of W_KEYS) {
    const g   = sumG[k] / n;
    const reg = decay * (w[k] - initW[k]);
    newW[k]   = w[k] + lr * g - reg;
  }
  return newW;
}

function learnEnsemble(
  ens: EnsembleState,
  buffer: ReplayEntry[],
  correct: boolean,
): EnsembleState {
  const newEns = { ...ens };
  for (const id of SUB_IDS) {
    const { weights, history } = ens[id];
    const newWeights = batchLearnSubModel(weights, SUB_INIT[id], buffer, history);
    const newHistory = [...history, correct].slice(-30);
    newEns[id] = { weights: newWeights, history: newHistory };
  }
  return newEns;
}

// ── Pre-training (3 epochs, builds replay buffer seed) ────────────────────────

async function preTrainEnsemble(
  symbol: string, h: Horizon, existingBuffer: ReplayEntry[]
): Promise<{ ens: EnsembleState; buffer: ReplayEntry[]; samples: number }> {
  try {
    const res = await fetch(
      `/api/stock?symbol=${encodeURIComponent(symbol)}&interval=${h.preInterval}&days=${h.preDays}`
    );
    if (!res.ok) return { ens: initEnsemble(), buffer: existingBuffer, samples: 0 };
    const data = await res.json();
    const candles: Array<{ close: number }> = (data.data || []).slice(-500);
    if (candles.length < 25 + h.preHorizon) return { ens: initEnsemble(), buffer: existingBuffer, samples: 0 };

    let ens    = initEnsemble();
    let buffer = [...existingBuffer]; // start from existing live session data
    const trainEnd = Math.floor(candles.length * 0.8);

    // Collect all training examples
    const examples: { snap: number[]; base: number; actual: number }[] = [];
    for (let i = 20; i < trainEnd - h.preHorizon; i++) {
      const snap   = candles.slice(Math.max(0, i - 20), i).map(c => c.close);
      const base   = candles[i - 1].close;
      const actual = candles[i + h.preHorizon - 1].close;
      if (Math.abs((actual - base) / base) < 0.0001) continue;
      examples.push({ snap, base, actual });
    }

    let count = 0;

    // 3 epochs with shuffle between each
    for (let epoch = 0; epoch < 3; epoch++) {
      const shuffled = [...examples];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }

      for (const { snap, base, actual } of shuffled) {
        const f = extractFeatures(snap);
        if (!f) continue;
        buffer = [...buffer, { f: featToArr(f), base, actual }].slice(-REPLAY_SIZE);

        // Every 16 new samples do a mini-batch update
        if (buffer.length >= 16 && count % 16 === 0) {
          for (const id of SUB_IDS) {
            const newW = batchLearnSubModel(ens[id].weights, SUB_INIT[id], buffer, ens[id].history);
            ens[id] = { ...ens[id], weights: newW };
          }
        }
        count++;
      }
    }

    return { ens, buffer: buffer.slice(-REPLAY_SIZE), samples: count };
  } catch {
    return { ens: initEnsemble(), buffer: existingBuffer, samples: 0 };
  }
}

// ── Storage ────────────────────────────────────────────────────────────────────

const lsKey     = (sym: string, hid: HorizonId, t: 'mod' | 'p' | 'pt' | 'rb') => `lp${t}_${sym}_${hid}`;

const saveEnsemble = (sym: string, hid: HorizonId, ens: EnsembleState) => {
  try { localStorage.setItem(lsKey(sym, hid, 'mod'), JSON.stringify(ens)); } catch {}
};

const loadEnsemble = (sym: string, hid: HorizonId): EnsembleState | null => {
  try {
    const s = localStorage.getItem(lsKey(sym, hid, 'mod'));
    if (!s) return null;
    const raw = JSON.parse(s) as Partial<EnsembleState>;
    const result = initEnsemble();
    for (const id of SUB_IDS) {
      if (raw[id]?.weights) result[id].weights = { ...SUB_INIT[id], ...raw[id]!.weights };
      if (raw[id]?.history) result[id].history = raw[id]!.history;
    }
    return result;
  } catch { return null; }
};

const loadReplay = (sym: string, hid: HorizonId): ReplayEntry[] => {
  try {
    const s = localStorage.getItem(lsKey(sym, hid, 'rb'));
    return s ? (JSON.parse(s) as ReplayEntry[]).filter(e => Array.isArray(e.f) && e.f.length > 0) : [];
  } catch { return []; }
};

const saveReplay = (sym: string, hid: HorizonId, buf: ReplayEntry[]) => {
  try { localStorage.setItem(lsKey(sym, hid, 'rb'), JSON.stringify(buf.slice(-REPLAY_SIZE))); } catch {}
};

// ── Component ─────────────────────────────────────────────────────────────────

interface Props { symbol: string; }

export default function LivePredictionChart({ symbol }: Props) {
  const [horizonId,   setHorizonId]   = useState<HorizonId>(DEFAULT_HORIZON);
  const [prices,      setPrices]      = useState<PricePoint[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [ensemble,    setEnsemble]    = useState<EnsembleState>(initEnsemble);
  const [replaySize,  setReplaySize]  = useState(0);
  const [marketState, setMarketState] = useState('');
  const [loading,     setLoading]     = useState(true);
  const [preTraining, setPreTraining] = useState(false);
  const [preSamples,  setPreSamples]  = useState(0);
  const [error,       setError]       = useState('');
  const [tickNow,     setTickNow]     = useState(Date.now());

  const pricesRef    = useRef<PricePoint[]>([]);
  const pricesCacheRef = useRef<Partial<Record<HorizonId, PricePoint[]>>>({});
  const predsRef     = useRef<Prediction[]>([]);
  const ensRef       = useRef<EnsembleState>(initEnsemble());
  const replayRef    = useRef<ReplayEntry[]>([]);
  const horizonRef   = useRef<Horizon>(HORIZONS[0]);
  const lastPredAt   = useRef(0);
  const pollTimerId  = useRef<ReturnType<typeof setInterval> | null>(null);

  const horizon = HORIZONS.find(h => h.id === horizonId)!;

  useEffect(() => { horizonRef.current = horizon; },     [horizon]);
  useEffect(() => { ensRef.current     = ensemble; },    [ensemble]);
  useEffect(() => { predsRef.current   = predictions; }, [predictions]);
  useEffect(() => {
    const id = setInterval(() => setTickNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  // Persist ensemble + replay buffer + predictions
  useEffect(() => { saveEnsemble(symbol, horizonId, ensemble); }, [ensemble, symbol, horizonId]);
  useEffect(() => {
    if (replaySize > 0) saveReplay(symbol, horizonId, replayRef.current);
  }, [replaySize, symbol, horizonId]);
  useEffect(() => {
    if (predictions.length === 0) return;
    try { localStorage.setItem(lsKey(symbol, horizonId, 'p'), JSON.stringify(predictions.slice(-60))); } catch {}
  }, [predictions, symbol, horizonId]);

  // ── Poll ───────────────────────────────────────────────────────────────────

  const poll = useCallback(async () => {
    const h = horizonRef.current;
    try {
      const res = await fetch(`/api/price?symbol=${encodeURIComponent(symbol)}`);
      if (!res.ok) return;
      const data = await res.json();
      if (typeof data.price !== 'number') return;

      setMarketState(data.marketState || 'CLOSED');
      setLoading(false); setError('');

      const now  = Date.now();
      const prevP = pricesRef.current.length > 0
        ? pricesRef.current[pricesRef.current.length - 1].price : null;

      // Reject obvious data errors (>15% jump)
      if (prevP !== null && Math.abs(data.price - prevP) / prevP > 0.15) return;

      // Only append a new point when the price actually changed — avoids a flat
      // line of duplicate ticks when the market is closed (same price every poll)
      let newPrices = pricesRef.current;
      if (data.price !== prevP) {
        const pt: PricePoint = { time: now, price: data.price, volume: data.volume ?? undefined };
        newPrices = [...pricesRef.current, pt].slice(-h.maxPts);
        pricesRef.current = newPrices;
        pricesCacheRef.current[h.id] = newPrices;
        setPrices([...newPrices]);
      }

      if (newPrices.length === 0) return;

      // Compute live VWAP from tracked PricePoints
      const vwap = computeVWAP(newPrices);

      const allResolved = predsRef.current.filter(p => p.resolved);
      let preds = predsRef.current.map(p => ({ ...p }));
      let ens   = { ...ensRef.current };
      let ensChanged = false;

      // Resolve pending predictions, add to replay, train
      for (const pred of preds) {
        if (!pred.resolved && now >= pred.resolveAt) {
          pred.resolved    = true;
          pred.actualPrice = data.price;
          pred.correct     = pred.direction === 'up'
            ? data.price > pred.basePrice
            : data.price < pred.basePrice;
        }
        if (pred.resolved && !pred.weightUpdated && pred.actualPrice != null) {
          // Add to replay buffer (extract features from stored snapshot)
          const f = extractFeatures(pred.priceSnapshot);
          if (f) {
            replayRef.current = [...replayRef.current, {
              f: featToArr(f), base: pred.basePrice, actual: pred.actualPrice,
            }].slice(-REPLAY_SIZE);
            setReplaySize(replayRef.current.length);
          }
          // Batch train all sub-models on the updated buffer
          ens = learnEnsemble(ens, replayRef.current, !!pred.correct);
          pred.weightUpdated = true;
          ensChanged = true;
        }
      }
      if (ensChanged) { setEnsemble(ens); ensRef.current = ens; }

      const volRatio = (data.volume && data.avgVolume5 && data.avgVolume5 > 0)
        ? data.volume / data.avgVolume5 : null;

      // Make new prediction if horizon window has elapsed
      if (newPrices.length >= h.minPts && (now - lastPredAt.current) >= h.resolveMs) {
        const pVals  = newPrices.map(p => p.price);
        const result = ensemblePredict(pVals, ens, volRatio, vwap, h.moveScale);

        if (result) {
          preds = [...preds, {
            id: String(now), madeAt: now, resolveAt: now + h.resolveMs,
            basePrice: data.price, targetPrice: result.targetPrice,
            direction: result.direction, confidence: result.confidence,
            volBoost: result.volBoost, signals: result.signals,
            winnerModel: result.winnerModel,
            resolved: false, weightUpdated: false,
            priceSnapshot: pVals.slice(-20),
          }].slice(-60);
          lastPredAt.current = now;
        } else {
          // Not confident enough — retry after next poll
          lastPredAt.current = now - h.resolveMs + h.pollMs;
        }
      }
      predsRef.current = preds;
      setPredictions([...preds]);
    } catch { setError('Poll failed'); }
  }, [symbol]);

  // Restart on symbol / horizon change
  useEffect(() => {
    if (pollTimerId.current) clearInterval(pollTimerId.current);

    // Restore cached price history for this horizon (keeps chart alive when switching tabs)
    const h0 = HORIZONS.find(x => x.id === horizonId)!;
    const cachedPrices = (pricesCacheRef.current[horizonId] || []).slice(-h0.maxPts);
    pricesRef.current = cachedPrices;
    setPrices([...cachedPrices]);
    setLoading(cachedPrices.length === 0); // skip spinner if we already have data
    setError('');

    // Restore cached predictions
    try {
      const cached = localStorage.getItem(lsKey(symbol, horizonId, 'p'));
      if (cached) {
        const preds: Prediction[] = (JSON.parse(cached) as Prediction[])
          .filter(p => p && typeof p.basePrice === 'number' && typeof p.madeAt === 'number');
        lastPredAt.current = preds.reduce((mx, p) => Math.max(mx, p.madeAt), 0);
        predsRef.current   = preds;
        setPredictions(preds);
      } else {
        lastPredAt.current = 0; predsRef.current = []; setPredictions([]);
      }
    } catch { lastPredAt.current = 0; predsRef.current = []; setPredictions([]); }

    // Restore replay buffer
    const savedReplay = loadReplay(symbol, horizonId);
    replayRef.current = savedReplay;
    setReplaySize(savedReplay.length);

    const lastPT  = (() => { try { return Number(localStorage.getItem(lsKey(symbol, horizonId, 'pt')) || '0'); } catch { return 0; } })();
    const needsPT = (Date.now() - lastPT) > PRETRAIN_TTL_MS;
    const cachedE = loadEnsemble(symbol, horizonId);
    const h       = h0;
    horizonRef.current = h;

    const start = async () => {
      if (needsPT) {
        setPreTraining(true);
        const { ens: ptEns, buffer: ptBuf, samples } = await preTrainEnsemble(symbol, h, replayRef.current);

        // Blend: 70% pre-trained + 30% cached live session (if available)
        const blended: EnsembleState = cachedE
          ? (() => {
              const b = initEnsemble();
              for (const id of SUB_IDS) {
                const bW = { ...ptEns[id].weights };
                for (const k of W_KEYS) bW[k] = 0.7 * ptEns[id].weights[k] + 0.3 * cachedE[id].weights[k];
                b[id] = { weights: bW, history: cachedE[id].history };
              }
              return b;
            })()
          : ptEns;

        setEnsemble(blended); ensRef.current = blended;
        // Seed replay buffer with pre-training data (keep live data too)
        const merged = [...replayRef.current, ...ptBuf].slice(-REPLAY_SIZE);
        replayRef.current = merged; setReplaySize(merged.length);
        setPreSamples(samples); setPreTraining(false);
        try {
          localStorage.setItem(lsKey(symbol, horizonId, 'pt'), String(Date.now()));
          saveEnsemble(symbol, horizonId, blended);
          saveReplay(symbol, horizonId, merged);
        } catch {}
      } else if (cachedE) {
        setEnsemble(cachedE); ensRef.current = cachedE;
      } else {
        const e = initEnsemble(); setEnsemble(e); ensRef.current = e;
      }
      poll();
      pollTimerId.current = setInterval(poll, h.pollMs);
    };

    start();
    return () => { if (pollTimerId.current) clearInterval(pollTimerId.current); };
  }, [symbol, horizonId, poll]);

  // ── Derived ───────────────────────────────────────────────────────────────

  const resolved  = predictions.filter(p => p.resolved);
  const numRight  = resolved.filter(p => p.correct).length;
  const accuracy  = resolved.length > 0 ? (numRight / resolved.length) * 100 : null;
  const last10    = resolved.slice(-10);
  const rolling10 = last10.length >= 3 ? (last10.filter(p => p.correct).length / last10.length) * 100 : null;
  const trend     = accuracy !== null && rolling10 !== null && resolved.length >= 10
    ? (rolling10 > accuracy + 3 ? 'up' : rolling10 < accuracy - 3 ? 'down' : 'flat') : null;
  const pending   = [...predictions].reverse().find(p => !p.resolved);
  const recent    = [...predictions].reverse().slice(0, 12);

  let streak = 0;
  for (const p of [...resolved].reverse()) { if (p.correct) streak++; else break; }

  const subHistAcc = (id: SubModelId) => {
    const hist = ensemble[id].history;
    if (hist.length < 3) return 50;
    return (hist.slice(-15).filter(x => x).length / Math.min(15, hist.length)) * 100;
  };

  const timeToResolve  = pending ? Math.max(0, pending.resolveAt - tickNow) : null;
  const timeToNextPred = Math.max(0, horizon.resolveMs - (tickNow - lastPredAt.current));

  const fmtMs = (ms: number) => {
    const s = Math.ceil(ms / 1000);
    const m = Math.floor(s / 60);
    const hh = Math.floor(m / 60);
    if (hh > 0) return `${hh}h ${String(m % 60).padStart(2,'0')}m`;
    return m > 0 ? `${m}m ${String(s % 60).padStart(2,'0')}s` : `${s}s`;
  };
  const fmtTime = (ts: number) =>
    new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', hour12: false });

  const pVals = prices.map(p => p.price);
  const minP  = pVals.length > 0 ? Math.min(...pVals) : 0;
  const maxP  = pVals.length > 0 ? Math.max(...pVals) : 0;
  const pad   = ((maxP - minP) * 0.3) || 1;

  const chartData = prices.map(p => ({
    time: p.time, label: fmtTime(p.time), price: p.price,
    target: (pending && p.time >= pending.madeAt) ? pending.targetPrice : undefined,
  }));

  const resolvedMarkers = predictions
    .filter(p => p.resolved)
    .map(p => ({ pred: p, pt: chartData.find(d => d.time >= p.resolveAt) }))
    .filter(x => x.pt != null);
  const pendingMarker = pending ? chartData.find(d => d.time >= pending.madeAt) : null;

  const dirColor = (d: 'up' | 'down') => d === 'up' ? 'var(--success)' : 'var(--danger)';

  const resetModel = () => {
    const e = initEnsemble(); setEnsemble(e); ensRef.current = e;
    replayRef.current = []; setReplaySize(0);
    setPredictions([]); predsRef.current = [];
    lastPredAt.current = 0; setPreSamples(0);
    try {
      ['mod','p','pt','rb'].forEach(t => localStorage.removeItem(lsKey(symbol, horizonId, t as 'mod')));
    } catch {}
  };

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <div className="card">
      {/* ── Header ── */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="card-label">Live Prediction</span>
          <span className="text-[9px] px-1.5 py-0.5 font-bold"
            style={{ background: 'var(--bg-3)', color: 'var(--accent)', border: '1px solid var(--accent)' }}>
            EXPERIMENTAL
          </span>
        </div>
        <div className="flex items-center gap-2">
          {accuracy !== null && (
            <div className="flex items-center gap-1">
              <span className="text-[10px] font-mono font-semibold"
                style={{ color: accuracy >= 55 ? 'var(--success)' : accuracy >= 45 ? 'var(--text-3)' : 'var(--danger)' }}>
                {accuracy.toFixed(0)}%
              </span>
              {trend === 'up'   && <ArrowUp   className="w-3 h-3" style={{ color: 'var(--success)' }} />}
              {trend === 'down' && <ArrowDown  className="w-3 h-3" style={{ color: 'var(--danger)' }} />}
              {trend === 'flat' && <Minus      className="w-3 h-3" style={{ color: 'var(--text-4)' }} />}
            </div>
          )}
          {streak >= 3 && (
            <span className="text-[9px] px-1 py-0.5 font-bold"
              style={{ background:'rgba(34,197,94,0.15)', color:'var(--success)', border:'1px solid var(--success)' }}>
              🔥{streak}
            </span>
          )}
          <span className="text-[9px] px-1.5 py-0.5" style={{
            background: marketState === 'REGULAR' ? 'rgba(34,197,94,0.1)' : 'var(--bg-3)',
            color:      marketState === 'REGULAR' ? 'var(--success)'      : 'var(--text-5)',
            border:    `1px solid ${marketState === 'REGULAR' ? 'var(--success)' : 'var(--bg-1)'}`,
          }}>
            {marketState === 'REGULAR' ? '● LIVE' : marketState || '—'}
          </span>
        </div>
      </div>

      {/* ── Horizon selector ── */}
      <div className="flex gap-1 mb-1">
        {HORIZONS.map(h => (
          <button key={h.id} onClick={() => setHorizonId(h.id)}
            className="flex-1 py-1 text-[10px] font-semibold border transition-all"
            style={{
              background: horizonId === h.id ? 'var(--accent)' : 'var(--bg-4)',
              borderColor: horizonId === h.id ? 'var(--accent)' : 'var(--bg-1)',
              color: horizonId === h.id ? 'var(--text-0)' : 'var(--text-4)',
            }}>
            {h.label}
          </button>
        ))}
      </div>

      {/* ── Sub-model accuracy strip ── */}
      <div className="flex gap-1 mb-1">
        {SUB_IDS.map(id => {
          const acc = subHistAcc(id);
          const col = acc >= 55 ? 'var(--success)' : acc <= 45 ? 'var(--danger)' : 'var(--text-4)';
          return (
            <div key={id} className="flex-1 px-1.5 py-1 border text-center"
              style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
              <div className="text-[9px]" style={{ color: 'var(--text-5)' }}>{SUB_LABEL[id]}</div>
              <div className="text-[10px] font-mono font-bold" style={{ color: col }}>
                {acc.toFixed(0)}%
              </div>
            </div>
          );
        })}
      </div>

      {/* ── How it works ── */}
      <div className="mb-3 px-2 py-2 border-l-2 text-[10px] leading-relaxed"
        style={{ background: 'var(--bg-2)', borderColor: 'var(--bg-1)', color: 'var(--text-5)' }}>
        <span className="font-semibold" style={{ color: 'var(--text-4)' }}>How it works: </span>
        4 models (Trend, Momentum, Reversal, Mean Reversion) vote on each prediction, weighted by their recent accuracy using exponential decay.
        Features: momentum (4 scales), RSI, Bollinger, MACD, range position, volatility regime, VWAP deviation + volume.
        Each resolved prediction adds to a <span style={{ color: 'var(--accent)' }}>replay buffer</span>{' '}
        ({replaySize}/{REPLAY_SIZE}) — training is done on {BATCH_SIZE}-sample mini-batches for stability.
        Predicts {horizonId === '1m' ? '1 min' : horizonId === '5m' ? '5 min' : horizonId === '15m' ? '15 min' : horizonId === '30m' ? '30 min' : '1 h'} ahead.
      </div>


      {preTraining ? (
        <div className="flex flex-col items-center justify-center py-10 gap-3">
          <Brain className="w-6 h-6 animate-pulse" style={{ color: 'var(--accent)' }} />
          <div className="text-xs text-center" style={{ color: 'var(--text-3)' }}>
            Pre-training 4 models · 3 epochs · {horizon.preInterval} bars…
          </div>
          <div className="text-[10px]" style={{ color: 'var(--text-5)' }}>
            {horizon.preDays}d of {horizon.preInterval} data · building replay buffer
          </div>
        </div>
      ) : loading ? (
        <div className="flex items-center justify-center py-10">
          <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'var(--accent)' }} />
        </div>
      ) : error ? (
        <div className="text-xs py-4 text-center" style={{ color: 'var(--danger)' }}>{error}</div>
      ) : (
        <>
          {/* ── Chart ── */}
          <div style={{ height: 190 }}>
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartData} margin={{ top: 6, right: 4, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="var(--bg-1)" opacity={0.5} />
                <XAxis dataKey="label" tick={{ fontSize: 8, fill: 'var(--text-5)' }}
                  interval="preserveStartEnd" tickLine={false} axisLine={false} />
                <YAxis domain={[minP - pad, maxP + pad]}
                  tick={{ fontSize: 8, fill: 'var(--text-5)' }}
                  tickLine={false} axisLine={false} width={52}
                  tickFormatter={v => `$${v.toFixed(2)}`} />
                <Tooltip
                  contentStyle={{ background: 'var(--bg-2)', border: '1px solid var(--bg-1)', fontSize: 11 }}
                  labelStyle={{ color: 'var(--text-4)' }}
                  formatter={(v: any, name: string) => [`$${Number(v).toFixed(2)}`, name === 'target' ? 'Target' : 'Price']}
                />
                {pending && pendingMarker && chartData.length > 0 && (
                  <ReferenceArea
                    x1={pendingMarker.label} x2={chartData[chartData.length - 1].label}
                    fill={pending.direction === 'up' ? 'rgba(34,197,94,0.05)' : 'rgba(239,68,68,0.05)'}
                    strokeOpacity={0}
                  />
                )}
                <Line type="monotone" dataKey="price" stroke="var(--accent)"
                  strokeWidth={1.5} dot={false} isAnimationActive={false} />
                <Line type="monotone" dataKey="target"
                  stroke={pending ? dirColor(pending.direction) : 'transparent'}
                  strokeWidth={1} strokeDasharray="5 4"
                  dot={false} connectNulls={false} isAnimationActive={false} />
                {resolvedMarkers.map(({ pred, pt }) => (
                  <ReferenceLine key={pred.id} x={pt!.label}
                    stroke={pred.correct ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)'}
                    strokeWidth={1.5}
                    label={{ value: pred.correct ? '✓' : '✗', position: 'top',
                      fill: pred.correct ? 'var(--success)' : 'var(--danger)', fontSize: 10 }}
                  />
                ))}
                {pendingMarker && (
                  <ReferenceLine x={pendingMarker.label}
                    stroke={dirColor(pending!.direction)} strokeDasharray="3 3" strokeWidth={1.5}
                    label={{ value: pending!.direction === 'up' ? '▲' : '▼', position: 'top',
                      fill: dirColor(pending!.direction), fontSize: 10 }}
                  />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* ── Active prediction ── */}
          <div className="mt-3 mb-2">
            {pending ? (
              <div className="px-2 py-2 border-l-2"
                style={{ background: 'var(--bg-2)', borderColor: dirColor(pending.direction) }}>
                <div className="flex items-center gap-2 mb-1">
                  {pending.direction === 'up'
                    ? <TrendingUp   className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--success)' }} />
                    : <TrendingDown className="w-3.5 h-3.5 flex-shrink-0" style={{ color: 'var(--danger)' }} />}
                  <span className="text-xs font-semibold" style={{ color: dirColor(pending.direction) }}>
                    {pending.direction === 'up' ? 'UP' : 'DOWN'} → ${(pending.targetPrice ?? 0).toFixed(2)}
                  </span>
                  {pending.winnerModel && (
                    <span className="text-[9px] px-1 py-0.5 font-bold"
                      style={{ background:'var(--bg-3)', color:'var(--text-4)', border:'1px solid var(--bg-1)' }}>
                      {SUB_LABEL[pending.winnerModel]}
                    </span>
                  )}
                  <span className="text-[10px] font-mono ml-auto" style={{ color: 'var(--text-5)' }}>
                    {(pending.confidence * 100).toFixed(0)}% conf
                    {pending.volBoost !== 0 && (
                      <span style={{ color: pending.volBoost > 0 ? 'var(--success)' : 'var(--danger)' }}>
                        {pending.volBoost > 0 ? ' ↑vol' : ' ↓vol'}
                      </span>
                    )}
                  </span>
                  <Zap className="w-3 h-3 animate-pulse" style={{ color: 'var(--accent)' }} />
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-1">
                    {(pending.signals || []).map(s => (
                      <span key={s.name} className="text-[9px] px-1 py-0.5" style={{
                        background: s.dir === 'up' ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
                        color: dirColor(s.dir),
                        border: `1px solid ${s.dir === 'up' ? 'rgba(34,197,94,0.3)' : 'rgba(239,68,68,0.3)'}`,
                      }}>
                        {s.dir === 'up' ? '↑' : '↓'} {s.name}
                      </span>
                    ))}
                  </div>
                  <span className="text-[9px] font-mono flex-shrink-0 ml-2" style={{ color: 'var(--text-5)' }}>
                    {fmtMs(timeToResolve ?? 0)}
                  </span>
                </div>
              </div>
            ) : pVals.length < horizon.minPts ? (
              <div className="px-2 py-1.5 text-center" style={{ background: 'var(--bg-2)' }}>
                <span className="text-[10px]" style={{ color: 'var(--text-5)' }}>
                  Collecting data… {pVals.length}/{horizon.minPts} points
                </span>
              </div>
            ) : (
              <div className="px-2 py-1.5 flex items-center justify-between" style={{ background: 'var(--bg-2)' }}>
                <span className="text-[10px]" style={{ color: 'var(--text-5)' }}>Next {horizon.label} prediction</span>
                <span className="text-[10px] font-mono font-semibold" style={{ color: 'var(--accent)' }}>
                  {fmtMs(timeToNextPred)}
                </span>
              </div>
            )}
          </div>

          {/* ── Stats grid ── */}
          {resolved.length > 0 && (
            <div className="grid grid-cols-4 gap-1.5 mb-3">
              {[
                { label: 'Correct',  val: numRight,                    color: 'var(--success)' },
                { label: 'Wrong',    val: resolved.length - numRight,  color: 'var(--danger)' },
                { label: 'All-time', val: `${accuracy!.toFixed(0)}%`,  color: accuracy! >= 55 ? 'var(--success)' : accuracy! >= 45 ? 'var(--text-3)' : 'var(--danger)' },
                { label: 'Last 10',  val: rolling10 != null ? `${rolling10.toFixed(0)}%` : '—',
                  color: rolling10 != null ? (rolling10 >= 55 ? 'var(--success)' : rolling10 >= 45 ? 'var(--text-3)' : 'var(--danger)') : 'var(--text-5)' },
              ].map(s => (
                <div key={s.label} className="p-1.5 border text-center"
                  style={{ background: 'var(--bg-3)', borderColor: 'var(--bg-1)' }}>
                  <div className="text-[9px]" style={{ color: 'var(--text-5)' }}>{s.label}</div>
                  <div className="text-xs font-bold font-mono" style={{ color: s.color }}>{s.val}</div>
                </div>
              ))}
            </div>
          )}

          {/* ── History ── */}
          {recent.length > 0 && (
            <div className="space-y-1 max-h-52 overflow-y-auto mb-3">
              <div className="text-[10px] font-semibold mb-1" style={{ color: 'var(--text-4)' }}>History</div>
              {recent.map(pred => {
                const color  = !pred.resolved ? 'var(--info)' : pred.correct ? 'var(--success)' : 'var(--danger)';
                const errPct = pred.resolved && pred.actualPrice != null && pred.basePrice
                  ? ((pred.actualPrice - pred.basePrice) / pred.basePrice * 100).toFixed(2) : null;
                return (
                  <div key={pred.id} className="px-2 py-1.5 border-l-2"
                    style={{ background: 'var(--bg-2)', borderColor: color }}>
                    <div className="flex items-center gap-1.5 mb-0.5">
                      <span className="text-[10px] font-mono flex-shrink-0" style={{ color: 'var(--text-5)' }}>
                        {fmtTime(pred.madeAt)}
                      </span>
                      <span className="text-[10px] font-semibold" style={{ color: dirColor(pred.direction) }}>
                        {pred.direction === 'up' ? '▲' : '▼'} {pred.direction.toUpperCase()}
                      </span>
                      {pred.winnerModel && (
                        <span className="text-[9px]" style={{ color: 'var(--text-5)' }}>
                          {SUB_LABEL[pred.winnerModel]}
                        </span>
                      )}
                      <span className="text-[9px] font-mono ml-auto" style={{ color }}>
                        {!pred.resolved
                          ? `⏳ ${fmtMs(Math.max(0, pred.resolveAt - tickNow))}`
                          : pred.correct ? '✓ correct' : '✗ wrong'}
                      </span>
                    </div>
                    <div className="flex items-center gap-2 text-[9px] font-mono" style={{ color: 'var(--text-5)' }}>
                      {pred.targetPrice != null && (
                        <span>tgt <span style={{ color: 'var(--text-3)' }}>${pred.targetPrice.toFixed(2)}</span></span>
                      )}
                      {pred.resolved && pred.actualPrice != null && (
                        <span>act <span style={{ color }}>${pred.actualPrice.toFixed(2)}</span></span>
                      )}
                      {errPct != null && (
                        <span style={{ color: parseFloat(errPct) > 0 ? 'var(--success)' : 'var(--danger)' }}>
                          {parseFloat(errPct) > 0 ? '+' : ''}{errPct}%
                        </span>
                      )}
                      <span className="ml-auto">{(pred.confidence * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* ── Footer ── */}
          <div className="flex items-center gap-2 pt-2 border-t" style={{ borderColor: 'var(--bg-1)' }}>
            <Brain className="w-3 h-3 flex-shrink-0" style={{ color: 'var(--text-5)' }} />
            <span className="text-[9px]" style={{ color: 'var(--text-5)' }}>
              {preSamples > 0 ? `${preSamples} pre-train · ` : ''}
              replay {replaySize}/{REPLAY_SIZE} · batch {BATCH_SIZE} · {resolved.length} live · {horizon.pollMs/1000}s
            </span>
            <button onClick={resetModel}
              className="ml-auto flex items-center gap-1 px-1.5 py-0.5 border text-[9px]"
              style={{ color: 'var(--text-5)', borderColor: 'var(--bg-1)', background: 'var(--bg-3)' }}>
              <RotateCcw className="w-2.5 h-2.5" /> Reset
            </button>
          </div>
        </>
      )}
    </div>
  );
}
