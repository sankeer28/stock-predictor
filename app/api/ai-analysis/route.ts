import { NextRequest, NextResponse } from 'next/server';

function buildAnalysisPrompt(data: any): string {
  const {
    symbol, companyName, currentPrice,
    companyInfo, fundamentalsData,
    tradingSignal, forecastInsights,
    mlPredictions, newsArticles, newsSentiments,
    chartData, chartPatterns,
    extra,
  } = data;

  const recentChart = chartData?.[chartData.length - 1] || {};
  const rsi = recentChart.rsi != null ? Number(recentChart.rsi).toFixed(1) : 'N/A';
  const macd = recentChart.macd != null ? Number(recentChart.macd).toFixed(3) : 'N/A';
  const macdSignal = recentChart.macdSignal != null ? Number(recentChart.macdSignal).toFixed(3) : 'N/A';
  const ma20 = recentChart.ma20 != null ? Number(recentChart.ma20).toFixed(2) : 'N/A';
  const ma50 = recentChart.ma50 != null ? Number(recentChart.ma50).toFixed(2) : 'N/A';
  const bbUpper = recentChart.bbUpper != null ? Number(recentChart.bbUpper).toFixed(2) : 'N/A';
  const bbLower = recentChart.bbLower != null ? Number(recentChart.bbLower).toFixed(2) : 'N/A';
  const aboveMA20 = recentChart.ma20 != null ? currentPrice > recentChart.ma20 : null;
  const aboveMA50 = recentChart.ma50 != null ? currentPrice > recentChart.ma50 : null;
  const ma20Diff = recentChart.ma20 != null ? (currentPrice - recentChart.ma20).toFixed(2) : 'N/A';
  const ma50Diff = recentChart.ma50 != null ? (currentPrice - recentChart.ma50).toFixed(2) : 'N/A';
  const ma20DiffPct = recentChart.ma20 != null ? ((currentPrice - recentChart.ma20) / recentChart.ma20 * 100).toFixed(1) : 'N/A';
  const ma50DiffPct = recentChart.ma50 != null ? ((currentPrice - recentChart.ma50) / recentChart.ma50 * 100).toFixed(1) : 'N/A';

  const recentPrices = (chartData || []).slice(-10).map((d: any) =>
    `${d.date?.slice(0, 10)}:$${Number(d.close).toFixed(2)}`
  ).join(' ');

  const mlLines = Object.entries(mlPredictions || {})
    .filter(([, v]) => Array.isArray(v) && (v as any[]).length > 0)
    .map(([model, predictions]) => {
      const arr = predictions as any[];
      const last = arr[arr.length - 1];
      // Models use `predicted` field (MLPrediction interface), not `price`
      const price = last?.predicted ?? last?.price ?? null;
      const pct = price != null ? ((price - currentPrice) / currentPrice * 100).toFixed(1) : null;
      return `${model}:$${price != null ? Number(price).toFixed(2) : 'N/A'}(${pct != null ? (Number(pct) >= 0 ? '+' : '') + pct + '%' : 'N/A'})`;
    });
  const mlSummary = mlLines.length ? mlLines.join(' ') : 'none';

  const newsSummary = (newsArticles || []).slice(0, 6).map((article: any, i: number) => {
    const s = newsSentiments?.[i];
    const tag = s?.sentiment && s.sentiment !== 'neutral' ? `[${s.sentiment}]` : '';
    return `"${article.title}"${tag}`;
  }).join(' | ') || 'none';

  const patternsSummary = (chartPatterns || []).slice(0, 5).map((p: any) =>
    `${p.type}(${(p.confidence * 100).toFixed(0)}%)`
  ).join(' ') || 'none';

  const ov = fundamentalsData?.overview || {};
  const pe = companyInfo?.pe || ov.PERatio || 'N/A';
  const forwardPE = companyInfo?.forwardPE || ov.ForwardPE || 'N/A';
  const eps = companyInfo?.eps || ov.EPS || 'N/A';
  const marketCap = companyInfo?.marketCap || ov.MarketCapitalization || 'N/A';
  const sector = companyInfo?.sector || ov.Sector || 'N/A';
  const weekHigh = companyInfo?.fiftyTwoWeekHigh || ov['52WeekHigh'] || 'N/A';
  const weekLow = companyInfo?.fiftyTwoWeekLow || ov['52WeekLow'] || 'N/A';
  const dividendYield = companyInfo?.dividendYield || ov.DividendYield || 'N/A';
  const profitMargin = ov.ProfitMargin || companyInfo?.profitMargins || 'N/A';
  const revenueGrowth = ov.QuarterlyRevenueGrowthYOY || companyInfo?.revenueGrowth || 'N/A';
  const debtToEquity = ov.DebtToEquityRatio || companyInfo?.debtToEquity || 'N/A';
  // Analyst target comes from Finnhub via extra.priceTarget — don't duplicate here
  const returnOnEquity = ov.ReturnOnEquityTTM || 'N/A';
  const beta = companyInfo?.beta || ov.Beta || 'N/A';
  const change = companyInfo?.change != null ? Number(companyInfo.change).toFixed(2) : 'N/A';
  const changePct = companyInfo?.changePercent != null ? Number(companyInfo.changePercent).toFixed(2) : 'N/A';

  const signalType = tradingSignal?.type?.replace(/_/g, ' ').toUpperCase() || 'N/A';
  const signalScore = tradingSignal?.score ?? 'N/A';
  const signalReasons = (tradingSignal?.reasons || []).slice(0, 4).join('; ');

  const stPrice = forecastInsights?.shortTerm?.price?.toFixed(2) ?? 'N/A';
  const stChange = forecastInsights?.shortTerm?.change?.toFixed(2) ?? 'N/A';
  const mtPrice = forecastInsights?.mediumTerm?.price?.toFixed(2) ?? 'N/A';
  const mtChange = forecastInsights?.mediumTerm?.change?.toFixed(2) ?? 'N/A';
  const trendDir = forecastInsights?.trend?.direction ?? 'N/A';
  const trendStr = forecastInsights?.trend?.strength?.toFixed(0) ?? 'N/A';

  // 52w range position %
  const rangePct = (weekHigh !== 'N/A' && weekLow !== 'N/A')
    ? (((currentPrice - Number(weekLow)) / (Number(weekHigh) - Number(weekLow))) * 100).toFixed(0)
    : 'N/A';

  return `You are a financial analyst. Analyze the following stock data and return ONLY a valid JSON object. No markdown, no code fences, no explanation — raw JSON only.

DATA:
${symbol} (${companyName}) | Sector: ${sector} | Market Cap: ${marketCap}
Price: $${currentPrice} | Change: $${change} (${changePct}%) | Beta: ${beta}
52W High: $${weekHigh} | 52W Low: $${weekLow} | Range Position: ${rangePct}% from low
Dividend Yield: ${dividendYield}
P/E: ${pe} | Fwd P/E: ${forwardPE} | EPS: $${eps} | Profit Margin: ${profitMargin}
ROE: ${returnOnEquity} | Revenue Growth: ${revenueGrowth} | Debt/Equity: ${debtToEquity}
RSI(14): ${rsi} | MACD: ${macd} | Signal: ${macdSignal} | MACD diff: ${(Number(macd) - Number(macdSignal)).toFixed(3)}
MA20: $${ma20} (price is ${ma20Diff >= '0' ? '+' : ''}$${ma20Diff}, ${ma20DiffPct}% ${aboveMA20 ? 'above' : 'below'})
MA50: $${ma50} (price is ${ma50Diff >= '0' ? '+' : ''}$${ma50Diff}, ${ma50DiffPct}% ${aboveMA50 ? 'above' : 'below'})
BB Upper: $${bbUpper} | BB Lower: $${bbLower}
Algo Signal: ${signalType} (${signalScore}/100) | ${signalReasons}
7D Forecast: $${stPrice} (${stChange}%) | 30D Forecast: $${mtPrice} (${mtChange}%) | Trend: ${trendDir} ${trendStr}%
ML Models: ${mlSummary}
Patterns: ${patternsSummary}
Recent prices: ${recentPrices}
News: ${newsSummary}
Analyst Recommendations: ${extra?.recommendations ?? 'N/A'}
Analyst Price Target: ${extra?.priceTarget ?? 'N/A'}
Insider Transactions (last 90 days): ${extra?.insiderSummary ?? 'N/A'}
Earnings History: ${extra?.earningsSummary ?? 'N/A'}
Reddit/WSB Sentiment: ${extra?.redditSentiment ?? 'N/A'}
Social Media Mentions (ApeWisdom): ${extra?.apeSentiment ?? 'N/A'}

RETURN EXACTLY THIS JSON STRUCTURE (fill in all values, use null for truly unknown numbers):
{
  "summary": "2-3 sentences using specific numbers from the data. State current price, key trend, and the most critical signal.",
  "valuation": {
    "pe_note": "State P/E value, compare to S&P500 avg (~25x) and sector avg if known. Mention forward P/E. Conclude cheap/fair/expensive.",
    "range_note": "State exact % position in 52w range, dollar distance from high and low. What does this position suggest?",
    "target_note": "Use the Analyst Price Target from the Finnhub data (see 'Analyst Price Target' line above). Calculate upside/downside % from current price $${currentPrice}. If unavailable, say so.",
    "verdict": "undervalued|fair|overvalued|unknown"
  },
  "technicals": {
    "rsi_note": "State RSI value (${rsi}), explain what this level means (oversold <30, overbought >70, neutral 30-70), and historical reliability of this signal.",
    "macd_note": "State exact MACD (${macd}) and signal (${macdSignal}) values and the gap between them. Is momentum accelerating or decelerating?",
    "ma_note": "State exact $ and % gap between price and MA20/MA50. What pattern do these moving averages form (e.g., death cross, golden cross)?",
    "bb_note": "State where price sits relative to upper ($${bbUpper}) and lower ($${bbLower}) bands. What does this imply about volatility and mean reversion?",
    "outlook": "bullish|bearish|mixed",
    "outlook_reason": "1-2 sentences citing specific indicators as evidence."
  },
  "forecasts": {
    "note": "State all available model predictions with exact prices and % changes. Describe consensus direction and spread between most bullish and most bearish models.",
    "ml_consensus": "bullish|bearish|mixed|inconclusive"
  },
  "risks": [
    "Specific risk citing actual data (e.g. insider sells, missed earnings, technical levels, debt ratio)",
    "Specific risk with data reference",
    "Specific risk with data reference",
    "Specific risk with data reference"
  ],
  "opportunities": [
    "Specific opportunity citing actual data (e.g. earnings beats, analyst upgrades, social momentum, oversold RSI)",
    "Specific opportunity with data reference",
    "Specific opportunity with data reference",
    "Specific opportunity with data reference"
  ],
  "verdict": {
    "rating": "STRONG BUY|BUY|HOLD|SELL|STRONG SELL",
    "confidence": "Low|Medium|High",
    "target_low": 0.00,
    "target_high": 0.00,
    "stop_loss": 0.00,
    "reasoning": "2-3 sentences citing specific numbers: RSI, MACD, price vs MAs, forecast. Explain the primary catalyst for your rating.",
    "suitable_for": "Who this suits (e.g., long-term investor, short-term trader, income investor, not recommended)"
  }
}`;
}

// Fetch all missing container data server-side
async function fetchAdditionalData(symbol: string) {
  const FINNHUB = process.env.FINN_HUB;
  const sym = symbol.toUpperCase();

  const results = await Promise.allSettled([
    // Analyst recommendations
    FINNHUB
      ? fetch(`https://finnhub.io/api/v1/stock/recommendation?symbol=${sym}&token=${FINNHUB}`)
          .then(r => r.ok ? r.json() : null).catch(() => null)
      : Promise.resolve(null),

    // Price target
    FINNHUB
      ? fetch(`https://finnhub.io/api/v1/stock/price-target?symbol=${sym}&token=${FINNHUB}`)
          .then(r => r.ok ? r.json() : null).catch(() => null)
      : Promise.resolve(null),

    // Insider transactions (last 3 months)
    FINNHUB
      ? (() => {
          const to = new Date().toISOString().split('T')[0];
          const from = new Date(Date.now() - 90 * 86400000).toISOString().split('T')[0];
          return fetch(`https://finnhub.io/api/v1/stock/insider-transactions?symbol=${sym}&from=${from}&to=${to}&token=${FINNHUB}`)
            .then(r => r.ok ? r.json() : null).catch(() => null);
        })()
      : Promise.resolve(null),

    // Earnings calendar
    FINNHUB
      ? fetch(`https://finnhub.io/api/v1/stock/earnings?symbol=${sym}&token=${FINNHUB}`)
          .then(r => r.ok ? r.json() : null).catch(() => null)
      : Promise.resolve(null),

    // Reddit/WallStreetBets sentiment (filter for this symbol)
    fetch('https://api.tradestie.com/v1/apps/reddit')
      .then(r => r.ok ? r.json() : null).catch(() => null),

    // ApeWisdom social mentions
    fetch('https://apewisdom.io/api/v1.0/filter/all-stocks/page/1')
      .then(r => r.ok ? r.json() : null).catch(() => null),
  ]);

  const [recsRaw, priceTargetRaw, insiderRaw, earningsRaw, redditRaw, apeRaw] =
    results.map(r => r.status === 'fulfilled' ? r.value : null);

  // Latest analyst recs row
  const latestRec = Array.isArray(recsRaw) ? recsRaw[0] : null;
  const recommendations = latestRec
    ? `Strong Buy: ${latestRec.strongBuy}, Buy: ${latestRec.buy}, Hold: ${latestRec.hold}, Sell: ${latestRec.sell}, Strong Sell: ${latestRec.strongSell} (period: ${latestRec.period})`
    : 'N/A';

  const priceTarget = priceTargetRaw
    ? `Mean: $${priceTargetRaw.targetMean ?? 'N/A'}, High: $${priceTargetRaw.targetHigh ?? 'N/A'}, Low: $${priceTargetRaw.targetLow ?? 'N/A'} (${priceTargetRaw.numberOfAnalysts ?? '?'} analysts)`
    : 'N/A';

  // Insider transactions summary
  const transactions: any[] = insiderRaw?.data ?? [];
  const insiderSummary = transactions.length === 0 ? 'No recent insider activity' : (() => {
    const buys = transactions.filter((t: any) => t.transactionType === 'P' || (t.change ?? 0) > 0);
    const sells = transactions.filter((t: any) => t.transactionType === 'S' || (t.change ?? 0) < 0);
    const topBuy = buys[0];
    const topSell = sells[0];
    const lines = [];
    if (buys.length) lines.push(`${buys.length} buys (e.g. ${topBuy?.name ?? 'insider'} bought ${topBuy?.share ?? '?'} shares)`);
    if (sells.length) lines.push(`${sells.length} sells (e.g. ${topSell?.name ?? 'insider'} sold ${topSell?.share ?? '?'} shares)`);
    return lines.join('; ') || 'No notable transactions';
  })();

  // Last 4 earnings
  const earningsArr: any[] = Array.isArray(earningsRaw) ? earningsRaw.slice(0, 4) : [];
  const earningsSummary = earningsArr.length === 0 ? 'N/A' : earningsArr.map((e: any) =>
    `${e.period}: EPS actual $${e.actual ?? '?'} vs est $${e.estimate ?? '?'} (${e.actual != null && e.estimate != null ? (e.actual >= e.estimate ? 'BEAT' : 'MISS') : '?'})`
  ).join(' | ');

  // Reddit sentiment for this symbol
  const redditStocks: any[] = Array.isArray(redditRaw) ? redditRaw : [];
  const redditEntry = redditStocks.find((s: any) => s.ticker?.toUpperCase() === sym);
  const redditSentiment = redditEntry
    ? `Rank #${redditStocks.indexOf(redditEntry) + 1} on WallStreetBets — Sentiment: ${redditEntry.sentiment}, Score: ${redditEntry.sentiment_score}, Comments: ${redditEntry.no_of_comments}`
    : 'Not trending on WallStreetBets today';

  // ApeWisdom social mentions
  const apeStocks: any[] = apeRaw?.results ?? [];
  const apeEntry = apeStocks.find((s: any) => s.ticker?.toUpperCase() === sym);
  const apeSentiment = apeEntry
    ? `Rank #${apeEntry.rank} (was #${apeEntry.rank_24h_ago} 24h ago) — Mentions: ${apeEntry.mentions} (+${apeEntry.mentions - (apeEntry.mentions_24h_ago ?? 0)} vs 24h ago), Upvotes: ${apeEntry.upvotes}`
    : 'Not in top social mentions today';

  return { recommendations, priceTarget, insiderSummary, earningsSummary, redditSentiment, apeSentiment };
}

export async function POST(request: NextRequest) {
  try {
    const data = await request.json();
    const OLLAMA_API_KEY = process.env.OLLAMA_API_KEY;

    if (!OLLAMA_API_KEY) {
      return NextResponse.json({ error: 'Ollama API key not configured. Add OLLAMA_API_KEY to your .env.local file.' }, { status: 500 });
    }

    // Fetch all additional container data in parallel before building prompt
    const extra = await fetchAdditionalData(data.symbol || '');
    const prompt = buildAnalysisPrompt({ ...data, extra });

    const ollamaResponse = await fetch('https://ollama.com/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OLLAMA_API_KEY}`,
      },
      body: JSON.stringify({
        model: 'deepseek-v3.1:671b-cloud',
        messages: [
          {
            role: 'system',
            content: 'You are a financial analyst. Output ONLY raw valid JSON. No markdown, no code fences, no text before or after the JSON object.',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
        stream: true,
      }),
    });

    if (!ollamaResponse.ok) {
      const errorText = await ollamaResponse.text();
      return NextResponse.json(
        { error: `Ollama API returned ${ollamaResponse.status}: ${errorText.slice(0, 200)}` },
        { status: ollamaResponse.status }
      );
    }

    const encoder = new TextEncoder();
    const decoder = new TextDecoder();

    const stream = new ReadableStream({
      async start(controller) {
        const reader = ollamaResponse.body!.getReader();
        try {
          let buffer = '';
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed) continue;
              try {
                const json = JSON.parse(trimmed);
                if (json.message?.content) {
                  controller.enqueue(encoder.encode(json.message.content));
                }
              } catch { /* skip malformed */ }
            }
          }
          if (buffer.trim()) {
            try {
              const json = JSON.parse(buffer.trim());
              if (json.message?.content) {
                controller.enqueue(encoder.encode(json.message.content));
              }
            } catch { /* skip */ }
          }
        } finally {
          controller.close();
          reader.releaseLock();
        }
      },
    });

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
      },
    });
  } catch (error: any) {
    return NextResponse.json(
      { error: error.message || 'Failed to generate analysis' },
      { status: 500 }
    );
  }
}
