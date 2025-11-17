'use client';

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  BarChart,
  Bar,
} from 'recharts';
import { ChartDataPoint } from '@/types';

interface TechnicalIndicatorsChartProps {
  data: ChartDataPoint[];
  indicator: 'rsi' | 'macd';
}

export default function TechnicalIndicatorsChart({
  data,
  indicator,
}: TechnicalIndicatorsChartProps) {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  // Get latest values for analysis
  const latestData = data[data.length - 1];
  const previousData = data[data.length - 2];

  // RSI Analysis (14-period calculation)
  const getRSIConclusion = () => {
    if (!latestData?.rsi) return null;

    const rsi = latestData.rsi;
    const prevRsi = previousData?.rsi || rsi;
    const rsiTrend = rsi > prevRsi ? 'rising' : rsi < prevRsi ? 'falling' : 'stable';
    const rsiChange = Math.abs(rsi - prevRsi);

    let condition = '';
    let meaning = '';
    let momentumStrength = '';
    let color = '';
    let icon = '';
    let interpretation = '';

    if (rsi >= 80) {
      condition = 'EXTREMELY OVERBOUGHT';
      meaning = 'Exceptional buying pressure';
      momentumStrength = 'Very Strong';
      color = 'var(--danger)';
      icon = '🔴';
      interpretation = `Price is in extreme overbought territory (RSI: ${rsi.toFixed(1)}). Buyers are dominant but exhaustion is imminent. Strong chance of reversal or consolidation. ${rsiTrend === 'rising' ? 'Momentum still strengthening - classic parabolic move, extremely risky to buy here.' : 'Momentum peaked and turning down - reversal may be starting. Consider profit-taking or short positions.'}`;
    } else if (rsi >= 70) {
      condition = 'OVERBOUGHT ZONE';
      meaning = 'Strong buying pressure';
      momentumStrength = 'Strong';
      color = 'var(--danger)';
      icon = '⚠️';
      interpretation = `Stock is overbought (RSI: ${rsi.toFixed(1)}). Uptrend is mature and vulnerable to pullback. ${rsiTrend === 'rising' ? `Momentum accelerating (Δ${rsiChange.toFixed(1)}) - trend continuation possible but risky. Watch for divergence.` : `Momentum slowing (Δ${rsiChange.toFixed(1)}) - likely topping pattern forming. Tighten stops on longs.`}`;
    } else if (rsi >= 60) {
      condition = 'BULLISH MOMENTUM';
      meaning = 'Healthy upward trend';
      momentumStrength = 'Moderate to Strong';
      color = 'var(--success)';
      icon = '📈';
      interpretation = `Solid upward momentum (RSI: ${rsi.toFixed(1)}). Price trend is rising with buyers in control. ${rsiTrend === 'rising' ? `Trend strengthening (Δ${rsiChange.toFixed(1)}) - good for swing trades and trend following. Still room to run before overbought.` : `Slight consolidation (Δ${rsiChange.toFixed(1)}) within uptrend - healthy pullback or early warning? Monitor price action.`}`;
    } else if (rsi >= 50) {
      condition = 'NEUTRAL-BULLISH BIAS';
      meaning = 'Slight buyer advantage';
      momentumStrength = 'Weak to Moderate';
      color = 'var(--info)';
      icon = '↗️';
      interpretation = `Slightly bullish conditions (RSI: ${rsi.toFixed(1)}). Price in equilibrium with mild upward bias. ${rsiTrend === 'rising' ? `Building momentum (Δ${rsiChange.toFixed(1)}) - potential early stage of uptrend. Wait for confirmation above 60.` : `Losing steam (Δ${rsiChange.toFixed(1)}) - transition phase, could go either way. Neutral stance recommended.`}`;
    } else if (rsi >= 40) {
      condition = 'NEUTRAL-BEARISH BIAS';
      meaning = 'Slight seller advantage';
      momentumStrength = 'Weak to Moderate';
      color = 'var(--text-4)';
      icon = '↘️';
      interpretation = `Slightly bearish conditions (RSI: ${rsi.toFixed(1)}). Price in equilibrium with mild downward bias. ${rsiTrend === 'falling' ? `Weakening momentum (Δ${rsiChange.toFixed(1)}) - potential early stage of downtrend. Watch for break below 40.` : `Attempting recovery (Δ${rsiChange.toFixed(1)}) - transition phase, stabilization possible. Cautiously optimistic.`}`;
    } else if (rsi >= 30) {
      condition = 'BEARISH MOMENTUM';
      meaning = 'Downward trend active';
      momentumStrength = 'Moderate to Strong';
      color = 'var(--warning)';
      icon = '📉';
      interpretation = `Downward momentum present (RSI: ${rsi.toFixed(1)}). Price trend declining with sellers in control. ${rsiTrend === 'falling' ? `Selling pressure increasing (Δ${rsiChange.toFixed(1)}) - downtrend intact. Approaching oversold zone where bounce possible.` : `Selling exhausting (Δ${rsiChange.toFixed(1)}) - potential bottoming process. Early reversal signal if confirmed.`}`;
    } else if (rsi >= 20) {
      condition = 'OVERSOLD ZONE';
      meaning = 'Heavy selling pressure';
      momentumStrength = 'Strong';
      color = 'var(--success)';
      icon = '✅';
      interpretation = `Stock is oversold (RSI: ${rsi.toFixed(1)}). Downtrend is extended and due for relief rally. ${rsiTrend === 'falling' ? `Capitulation ongoing (Δ${rsiChange.toFixed(1)}) - extreme fear, but catching falling knife risk. Wait for RSI to turn up.` : `Bottom forming (Δ${rsiChange.toFixed(1)}) - sellers exhausted, buyers stepping in. Good risk/reward for contrarian entry.`}`;
    } else {
      condition = 'EXTREMELY OVERSOLD';
      meaning = 'Panic selling / capitulation';
      momentumStrength = 'Very Strong';
      color = 'var(--success)';
      icon = '🟢';
      interpretation = `Extreme oversold levels (RSI: ${rsi.toFixed(1)}). Panic selling evident - historically strong reversal zone. ${rsiTrend === 'falling' ? 'Final capitulation phase - maximum pain point. High probability bounce imminent but timing uncertain.' : 'Reversal initiated - smart money accumulating. Strong recovery potential but scale in gradually.'}`;
    }

    return {
      condition,
      meaning,
      momentumStrength,
      color,
      icon,
      value: rsi.toFixed(1),
      trend: rsiTrend,
      interpretation,
      period: '14-period'
    };
  };

  // MACD Analysis (12, 26, 9 configuration)
  const getMACDConclusion = () => {
    if (!latestData?.macd || !latestData?.macdSignal) return null;

    const macd = latestData.macd;
    const signal = latestData.macdSignal;
    const histogram = macd - signal;

    const prevMacd = previousData?.macd || 0;
    const prevSignal = previousData?.macdSignal || 0;
    const prevHistogram = prevMacd - prevSignal;

    let condition = '';
    let meaning = '';
    let momentumStrength = '';
    let color = '';
    let icon = '';
    let interpretation = '';

    // Crossover detection
    const bullishCrossover = prevMacd <= prevSignal && macd > signal;
    const bearishCrossover = prevMacd >= prevSignal && macd < signal;

    // Histogram analysis
    const histogramExpanding = Math.abs(histogram) > Math.abs(prevHistogram);
    const histogramContracting = Math.abs(histogram) < Math.abs(prevHistogram);
    const separation = Math.abs(histogram);

    // Zero line position
    const aboveZero = macd > 0;
    const signalAboveZero = signal > 0;

    // Determine if overbought/oversold based on histogram extremes
    const isOverbought = histogram > 2.0 && macd > signal;
    const isOversold = histogram < -2.0 && macd < signal;

    if (bullishCrossover) {
      condition = 'BULLISH CROSSOVER';
      meaning = 'MACD crossed above Signal line';
      momentumStrength = aboveZero ? 'Strong' : 'Moderate';
      color = 'var(--success)';
      icon = '🚀';
      if (aboveZero && signalAboveZero) {
        interpretation = `Bullish crossover in positive territory (MACD: ${macd.toFixed(3)} > Signal: ${signal.toFixed(3)}). Strong continuation signal - uptrend accelerating. Both lines above zero confirms bullish environment. Price likely to rise; buyers dominant. ${histogramExpanding ? 'Momentum expanding rapidly.' : 'Initial crossover - monitor for follow-through.'}`;
      } else {
        interpretation = `Bullish crossover from negative zone (MACD: ${macd.toFixed(3)} > Signal: ${signal.toFixed(3)}). Early reversal signal - downtrend may be ending. Market transitioning from bearish to bullish. Wait for MACD to cross zero line for full confirmation. ${histogramExpanding ? 'Positive momentum building.' : 'Tentative recovery beginning.'}`;
      }
    } else if (bearishCrossover) {
      condition = 'BEARISH CROSSOVER';
      meaning = 'MACD crossed below Signal line';
      momentumStrength = !aboveZero ? 'Strong' : 'Moderate';
      color = 'var(--danger)';
      icon = '⚠️';
      if (!aboveZero && !signalAboveZero) {
        interpretation = `Bearish crossover in negative territory (MACD: ${macd.toFixed(3)} < Signal: ${signal.toFixed(3)}). Strong continuation signal - downtrend accelerating. Both lines below zero confirms bearish environment. Price likely to fall; sellers dominant. ${histogramExpanding ? 'Selling pressure intensifying.' : 'Initial breakdown - monitor for acceleration.'}`;
      } else {
        interpretation = `Bearish crossover from positive zone (MACD: ${macd.toFixed(3)} < Signal: ${signal.toFixed(3)}). Reversal warning - uptrend weakening. Market transitioning from bullish to bearish. Consider reducing long positions or tightening stops. ${histogramExpanding ? 'Negative momentum building.' : 'Early warning - confirmation needed.'}`;
      }
    } else if (isOverbought) {
      condition = 'OVERBOUGHT (Far Above Signal)';
      meaning = 'Extreme bullish momentum';
      momentumStrength = 'Very Strong';
      color = 'var(--warning)';
      icon = '🔥';
      interpretation = `MACD far above signal line (Histogram: ${histogram.toFixed(3)}). Extreme bullish momentum - possible exhaustion ahead. ${histogramExpanding ? 'Parabolic move - momentum still accelerating but unsustainable. High risk of sharp reversal.' : 'Momentum peaked and weakening - possible top forming. Bullish momentum may soon exhaust; consider taking profits.'}`;
    } else if (isOversold) {
      condition = 'OVERSOLD (Far Below Signal)';
      meaning = 'Extreme bearish momentum';
      momentumStrength = 'Very Strong';
      color = 'var(--info)';
      icon = '💎';
      interpretation = `MACD far below signal line (Histogram: ${histogram.toFixed(3)}). Extreme bearish momentum - possible bounce ahead. ${histogramContracting ? 'Selling pressure easing - potential reversal zone. Bearish momentum may soon exhaust; possible rebound opportunity.' : 'Capitulation ongoing - downside momentum extreme. Wait for histogram contraction before considering entries.'}`;
    } else if (Math.abs(histogram) < 0.1) {
      condition = 'NEUTRAL / INDECISIVE';
      meaning = 'MACD ≈ Signal (convergence)';
      momentumStrength = 'Very Low';
      color = 'var(--text-4)';
      icon = '🔄';
      interpretation = `MACD and Signal nearly equal (Histogram: ${histogram.toFixed(3)}). Market in consolidation or transition phase. Directional momentum absent - waiting for catalyst. ${aboveZero ? 'In positive zone - slight bullish bias but needs confirmation.' : 'In negative zone - slight bearish bias but indecisive.'} Watch for breakout in either direction.`;
    } else if (macd > signal) {
      const strength = separation > 1.0 ? 'Strong' : separation > 0.3 ? 'Moderate' : 'Weak';

      if (histogramExpanding) {
        condition = 'STRONG BULLISH TREND FORMING';
        meaning = 'MACD rising & above Signal';
        momentumStrength = strength;
        color = 'var(--success)';
        icon = '📈';
        interpretation = `MACD above signal with expanding histogram (${histogram.toFixed(3)}). Positive momentum accelerating - trend strengthening. ${aboveZero ? 'Both above zero - established uptrend gaining strength. Price trend rising; buyers dominant.' : 'Rising from negative levels - recovery gaining traction. Building bullish momentum.'} Histogram widening = ${strength.toLowerCase()} momentum expansion.`;
      } else if (histogramContracting) {
        condition = 'BULLISH WEAKENING';
        meaning = 'MACD falling but above Signal';
        momentumStrength = separation > 0.3 ? 'Moderate' : 'Weak';
        color = 'var(--warning)';
        icon = '↗️';
        interpretation = `MACD above signal but histogram contracting (${histogram.toFixed(3)}). Upward momentum decelerating - possible upcoming bearish crossover. ${aboveZero ? 'Still in bullish territory but losing steam. Potential top forming - monitor closely.' : 'Recovery stalling - may need consolidation before next move.'} Possible trend reversal if crossover occurs.`;
      } else {
        condition = 'BULLISH TREND';
        meaning = 'MACD > Signal (stable)';
        momentumStrength = separation > 0.5 ? 'Moderate' : 'Weak';
        color = 'var(--info)';
        icon = '↗️';
        interpretation = `MACD maintaining position above signal (Histogram: ${histogram.toFixed(3)}). ${aboveZero ? 'Steady uptrend in positive territory. Price likely rising at consistent pace.' : 'Gradual recovery from negative levels - early improvement phase.'} Momentum stable but watch for changes in histogram direction.`;
      }
    } else {
      const strength = separation > 1.0 ? 'Strong' : separation > 0.3 ? 'Moderate' : 'Weak';

      if (histogramExpanding) {
        condition = 'STRONG BEARISH TREND FORMING';
        meaning = 'MACD falling & below Signal';
        momentumStrength = strength;
        color = 'var(--danger)';
        icon = '📉';
        interpretation = `MACD below signal with expanding histogram (${histogram.toFixed(3)}). Negative momentum accelerating - trend strengthening. ${!aboveZero ? 'Both below zero - established downtrend intensifying. Price trend falling; sellers dominant.' : 'Weakening from positive levels - concerning deterioration. Building bearish pressure.'} Histogram widening = ${strength.toLowerCase()} momentum expansion.`;
      } else if (histogramContracting) {
        condition = 'BEARISH WEAKENING';
        meaning = 'MACD rising but below Signal';
        momentumStrength = separation > 0.3 ? 'Moderate' : 'Weak';
        color = 'var(--info)';
        icon = '↘️';
        interpretation = `MACD below signal but histogram contracting (${histogram.toFixed(3)}). Downward momentum decelerating - possible upcoming bullish crossover. ${!aboveZero ? 'Still in bearish territory but bottoming process may be starting. Potential support forming.' : 'Attempting to stabilize after decline - early warning of reversal.'} Watch for bullish crossover confirmation.`;
      } else {
        condition = 'BEARISH TREND';
        meaning = 'MACD < Signal (stable)';
        momentumStrength = separation > 0.5 ? 'Moderate' : 'Weak';
        color = 'var(--warning)';
        icon = '↘️';
        interpretation = `MACD maintaining position below signal (Histogram: ${histogram.toFixed(3)}). ${!aboveZero ? 'Steady downtrend in negative territory. Price likely falling at consistent pace.' : 'Gradual deterioration from positive levels - concerning but not accelerating.'} Momentum stable but in negative direction.`;
      }
    }

    return {
      condition,
      meaning,
      momentumStrength,
      color,
      icon,
      macd: macd.toFixed(3),
      signal: signal.toFixed(3),
      histogram: histogram.toFixed(3),
      interpretation,
      period: '12, 26, 9'
    };
  };

  if (indicator === 'rsi') {
    const rsiAnalysis = getRSIConclusion();

    return (
      <div className="w-full">
        <div className="h-[150px] sm:h-[180px] md:h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 5, right: 15, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(31% 0 0)" opacity={0.3} />
              <XAxis
                dataKey="date"
                tickFormatter={formatDate}
                stroke="oklch(75% 0 0)"
                style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace' }}
              />
              <YAxis
                domain={[0, 100]}
                stroke="oklch(75% 0 0)"
                style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace' }}
              />
              <Tooltip
                labelFormatter={formatDate}
                contentStyle={{
                  backgroundColor: 'oklch(23% 0 0)',
                  border: '2px solid oklch(70% 0.12 170)',
                  borderRadius: '0',
                  color: 'oklch(85% 0 0)',
                  fontFamily: 'DM Mono, monospace',
                  fontSize: '12px',
                }}
                labelStyle={{
                  color: 'oklch(70% 0.12 170)',
                  fontWeight: 'bold',
                }}
              />
              <Legend
                wrapperStyle={{
                  fontFamily: 'DM Mono, monospace',
                  fontSize: '11px'
                }}
              />
              <ReferenceLine y={70} stroke="oklch(70% 0.13 0)" strokeDasharray="3 3" label="Overbought" />
              <ReferenceLine y={30} stroke="oklch(70% 0.12 170)" strokeDasharray="3 3" label="Oversold" />
              <ReferenceLine y={50} stroke="oklch(60% 0 0)" strokeDasharray="2 2" />
              <Line
                type="monotone"
                dataKey="rsi"
                stroke="oklch(75% 0.12 90)"
                strokeWidth={2}
                dot={false}
                name="RSI"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* RSI Conclusion */}
        {rsiAnalysis && (
          <div className="mt-4 p-4 border-2" style={{
            background: 'var(--bg-2)',
            borderColor: rsiAnalysis.color,
            borderLeftWidth: '3px'
          }}>
            <div className="flex items-start gap-3">
              <span className="text-2xl">{rsiAnalysis.icon}</span>
              <div className="flex-1">
                {/* Header */}
                <div className="mb-3">
                  <div className="flex items-baseline gap-2 mb-1">
                    <span className="text-sm font-mono font-bold" style={{ color: rsiAnalysis.color }}>
                      {rsiAnalysis.condition}
                    </span>
                    <span className="text-xs font-mono" style={{ color: 'var(--text-5)' }}>
                      ({rsiAnalysis.period})
                    </span>
                  </div>
                  {/* Info Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs font-mono mb-2">
                    <div>
                      <span style={{ color: 'var(--text-5)' }}>Value: </span>
                      <strong style={{ color: rsiAnalysis.color }}>{rsiAnalysis.value}</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-5)' }}>Trend: </span>
                      <strong style={{ color: 'var(--text-3)' }}>{rsiAnalysis.trend}</strong>
                    </div>
                    <div>
                      <span style={{ color: 'var(--text-5)' }}>Strength: </span>
                      <strong style={{ color: 'var(--text-3)' }}>{rsiAnalysis.momentumStrength}</strong>
                    </div>
                  </div>
                  <div className="text-xs italic" style={{ color: 'var(--text-4)' }}>
                    {rsiAnalysis.meaning}
                  </div>
                </div>
                {/* Interpretation */}
                <p className="text-xs leading-relaxed" style={{ color: 'var(--text-3)' }}>
                  {rsiAnalysis.interpretation}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // MACD Chart
  const macdAnalysis = getMACDConclusion();

  return (
    <div className="w-full">
      <div className="h-[150px] sm:h-[180px] md:h-[200px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 15, left: 10, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="oklch(31% 0 0)" opacity={0.3} />
            <XAxis
              dataKey="date"
              tickFormatter={formatDate}
              stroke="oklch(75% 0 0)"
              style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace' }}
            />
            <YAxis
              stroke="oklch(75% 0 0)"
              style={{ fontSize: '11px', fontFamily: 'DM Mono, monospace' }}
            />
            <Tooltip
              labelFormatter={formatDate}
              contentStyle={{
                backgroundColor: 'oklch(23% 0 0)',
                border: '2px solid oklch(70% 0.12 170)',
                borderRadius: '0',
                color: 'oklch(85% 0 0)',
                fontFamily: 'DM Mono, monospace',
                fontSize: '12px',
              }}
              labelStyle={{
                color: 'oklch(70% 0.12 170)',
                fontWeight: 'bold',
              }}
            />
            <Legend
              wrapperStyle={{
                fontFamily: 'DM Mono, monospace',
                fontSize: '11px'
              }}
            />
            <ReferenceLine y={0} stroke="oklch(60% 0 0)" strokeDasharray="2 2" />
            <Line
              type="monotone"
              dataKey="macd"
              stroke="oklch(70% 0.11 215)"
              strokeWidth={2}
              dot={false}
              name="MACD"
            />
            <Line
              type="monotone"
              dataKey="macdSignal"
              stroke="oklch(70% 0.13 0)"
              strokeWidth={2}
              dot={false}
              name="Signal"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* MACD Conclusion */}
      {macdAnalysis && (
        <div className="mt-4 p-4 border-2" style={{
          background: 'var(--bg-2)',
          borderColor: macdAnalysis.color,
          borderLeftWidth: '3px'
        }}>
          <div className="flex items-start gap-3">
            <span className="text-2xl">{macdAnalysis.icon}</span>
            <div className="flex-1">
              {/* Header */}
              <div className="mb-3">
                <div className="flex items-baseline gap-2 mb-1">
                  <span className="text-sm font-mono font-bold" style={{ color: macdAnalysis.color }}>
                    {macdAnalysis.condition}
                  </span>
                  <span className="text-xs font-mono" style={{ color: 'var(--text-5)' }}>
                    ({macdAnalysis.period})
                  </span>
                </div>
                {/* Info Grid */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs font-mono mb-2">
                  <div>
                    <span style={{ color: 'var(--text-5)' }}>MACD: </span>
                    <strong style={{ color: 'var(--info)' }}>{macdAnalysis.macd}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-5)' }}>Signal: </span>
                    <strong style={{ color: 'var(--warning)' }}>{macdAnalysis.signal}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-5)' }}>Histogram: </span>
                    <strong style={{ color: macdAnalysis.color }}>{macdAnalysis.histogram}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-5)' }}>Strength: </span>
                    <strong style={{ color: 'var(--text-3)' }}>{macdAnalysis.momentumStrength}</strong>
                  </div>
                </div>
                <div className="text-xs italic" style={{ color: 'var(--text-4)' }}>
                  {macdAnalysis.meaning}
                </div>
              </div>
              {/* Interpretation */}
              <p className="text-xs leading-relaxed" style={{ color: 'var(--text-3)' }}>
                {macdAnalysis.interpretation}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
