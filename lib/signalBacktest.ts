// Walk-forward backtest of the Trading Signals engine.
//
// For each session we compute the exact same signal the live app shows
// (generateTradingSignal over data up to that day) and simulate a simple
// long/flat strategy: hold the stock while the signal is on the buy side, sit
// in cash while it's on the sell side, and keep the current position on hold.
// The result is compared against buy-and-hold so the user can see whether
// following the signals would actually have beaten doing nothing.
//
// Indicators are computed once on the full series. Every indicator here is
// causal (value at i depends only on data[0..i]), so indexing the full arrays
// at day i is identical to recomputing on a truncated history — i.e. there is
// no look-ahead bias.

import { StockData } from '@/types';
import { calculateAllIndicators } from './technicalIndicators';
import { generateTradingSignal } from './tradingSignals';

export interface EquityPoint {
  date: string;
  strategy: number; // indexed to 100 at the start
  buyHold: number;
}

export interface SignalBacktestResult {
  startDate: string;
  endDate: string;
  bars: number;
  strategyReturn: number; // %
  buyHoldReturn: number;  // %
  alpha: number;          // strategyReturn - buyHoldReturn
  trades: number;         // number of entries
  winRate: number;        // % of closed round-trips that were profitable
  maxDrawdown: number;    // % (strategy, positive number)
  exposure: number;       // % of sessions spent in the market
  equityCurve: EquityPoint[];
}

const WARMUP = 200; // need MA200 before the first signal is meaningful

function isBuy(type: string): boolean {
  return type === 'strong_buy' || type === 'buy' || type === 'weak_buy';
}
function isSell(type: string): boolean {
  return type === 'strong_sell' || type === 'sell' || type === 'weak_sell';
}

/**
 * Run the signal backtest. Returns null when there isn't enough history
 * (need the MA200 warm-up plus a meaningful test window).
 */
export function runSignalBacktest(data: StockData[]): SignalBacktestResult | null {
  if (!data || data.length < WARMUP + 30) return null;

  const indicators = calculateAllIndicators(data);
  const n = data.length;

  let equity = 1;
  let buyHold = 1;
  let position = 0; // 0 = cash, 1 = invested
  let entryPrice: number | null = null;

  let trades = 0;
  let closedTrades = 0;
  let wins = 0;
  let daysInMarket = 0;

  let peak = 1;
  let maxDrawdown = 0;

  const equityCurve: EquityPoint[] = [
    { date: data[WARMUP].date, strategy: 100, buyHold: 100 },
  ];

  for (let i = WARMUP; i < n - 1; i++) {
    const signal = generateTradingSignal(data.slice(0, i + 1), indicators);
    const close = data[i].close;

    // Decide desired position from the signal (hold keeps the current one).
    let desired = position;
    if (isBuy(signal.type)) desired = 1;
    else if (isSell(signal.type)) desired = 0;

    if (desired === 1 && position === 0) {
      position = 1;
      entryPrice = close;
      trades += 1;
    } else if (desired === 0 && position === 1) {
      closedTrades += 1;
      if (entryPrice !== null && close > entryPrice) wins += 1;
      position = 0;
      entryPrice = null;
    }

    // Realise the next session's return.
    const next = data[i + 1].close;
    const ret = close > 0 ? (next - close) / close : 0;

    if (position === 1) {
      equity *= 1 + ret;
      daysInMarket += 1;
    }
    buyHold *= 1 + ret;

    if (equity > peak) peak = equity;
    const drawdown = (equity - peak) / peak;
    if (drawdown < maxDrawdown) maxDrawdown = drawdown;

    equityCurve.push({
      date: data[i + 1].date,
      strategy: equity * 100,
      buyHold: buyHold * 100,
    });
  }

  // Close any open position at the final price.
  if (position === 1 && entryPrice !== null) {
    closedTrades += 1;
    if (data[n - 1].close > entryPrice) wins += 1;
  }

  const totalSessions = n - 1 - WARMUP;

  return {
    startDate: data[WARMUP].date,
    endDate: data[n - 1].date,
    bars: n,
    strategyReturn: (equity - 1) * 100,
    buyHoldReturn: (buyHold - 1) * 100,
    alpha: (equity - buyHold) * 100,
    trades,
    winRate: closedTrades > 0 ? (wins / closedTrades) * 100 : 0,
    maxDrawdown: Math.abs(maxDrawdown) * 100,
    exposure: totalSessions > 0 ? (daysInMarket / totalSessions) * 100 : 0,
    equityCurve,
  };
}
