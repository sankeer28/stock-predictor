import { NextRequest, NextResponse } from 'next/server';

/**
 * Fetch related/similar stocks from Yahoo Finance
 * GET /api/related-stocks?symbol=AAPL
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol');

  if (!symbol) {
    return NextResponse.json(
      { error: 'symbol parameter is required' },
      { status: 400 }
    );
  }

  try {
    console.log(`[Related Stocks API] Fetching related stocks for ${symbol}`);

    // Yahoo Finance has a "recommendationsBySymbol" endpoint
    const url = `https://query2.finance.yahoo.com/v6/finance/recommendationsbysymbol/${symbol}`;

    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
      },
      cache: 'force-cache', // Cache related stocks as they don't change often
      signal: AbortSignal.timeout(10000),
    });

    if (!response.ok) {
      console.error(`[Related Stocks API] Yahoo Finance error: ${response.status}`);
      // Fallback to industry-based defaults
      return getFallbackRelatedStocks(symbol);
    }

    const data = await response.json();

    // Extract recommended symbols
    const relatedSymbols: string[] = [];

    if (data.finance?.result?.[0]?.recommendedSymbols) {
      const recommended = data.finance.result[0].recommendedSymbols;
      for (const rec of recommended) {
        if (rec.symbol && rec.symbol !== symbol) {
          relatedSymbols.push(rec.symbol);
        }
      }
    }

    // If we got related symbols, return them
    if (relatedSymbols.length > 0) {
      console.log(`[Related Stocks API] Found ${relatedSymbols.length} related stocks: ${relatedSymbols.slice(0, 8).join(', ')}`);
      return NextResponse.json({
        success: true,
        symbol,
        relatedStocks: relatedSymbols.slice(0, 8), // Limit to 8
        source: 'yahoo-recommendations'
      });
    }

    // Fallback if no recommendations found
    return getFallbackRelatedStocks(symbol);

  } catch (error: any) {
    console.error('[Related Stocks API] Error:', error);
    return getFallbackRelatedStocks(symbol);
  }
}

/**
 * Fallback function to return related stocks based on common industry groups
 */
function getFallbackRelatedStocks(symbol: string): NextResponse {
  // Common industry groups
  const industryGroups: Record<string, string[]> = {
    // Mega Tech
    'AAPL': ['MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA', 'AMD'],
    'MSFT': ['AAPL', 'GOOGL', 'META', 'AMZN', 'ORCL', 'CRM', 'ADBE'],
    'GOOGL': ['META', 'AAPL', 'MSFT', 'AMZN', 'NFLX', 'SNAP', 'PINS'],
    'META': ['GOOGL', 'SNAP', 'PINS', 'TWTR', 'AAPL', 'MSFT', 'NFLX'],
    'AMZN': ['AAPL', 'MSFT', 'GOOGL', 'WMT', 'EBAY', 'SHOP', 'BABA'],

    // Semiconductors
    'NVDA': ['AMD', 'INTC', 'TSM', 'QCOM', 'AVGO', 'MU', 'ASML'],
    'AMD': ['NVDA', 'INTC', 'TSM', 'QCOM', 'AVGO', 'MU', 'ARM'],
    'INTC': ['AMD', 'NVDA', 'TSM', 'QCOM', 'AVGO', 'MU', 'TXN'],

    // EVs & Auto
    'TSLA': ['RIVN', 'LCID', 'NIO', 'F', 'GM', 'NVDA', 'AAPL'],
    'RIVN': ['TSLA', 'LCID', 'NIO', 'F', 'GM', 'FORD', 'XPEV'],
    'NIO': ['TSLA', 'XPEV', 'LI', 'RIVN', 'LCID', 'F', 'GM'],

    // Finance
    'JPM': ['BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC'],
    'BAC': ['JPM', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'SCHW'],
    'GS': ['MS', 'JPM', 'C', 'BAC', 'BLK', 'SCHW', 'AXP'],

    // Healthcare/Pharma
    'JNJ': ['PFE', 'UNH', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT'],
    'PFE': ['JNJ', 'MRK', 'ABBV', 'LLY', 'BMY', 'GILD', 'AMGN'],
    'UNH': ['CVS', 'CI', 'HUM', 'ANTM', 'ELV', 'CNC', 'MOH'],

    // Retail/Consumer
    'WMT': ['TGT', 'COST', 'HD', 'LOW', 'AMZN', 'DG', 'DLTR'],
    'TGT': ['WMT', 'COST', 'KR', 'DG', 'DLTR', 'BBY', 'AMZN'],
    'COST': ['WMT', 'TGT', 'BJ', 'AMZN', 'HD', 'LOW', 'KR'],

    // Energy
    'XOM': ['CVX', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'VLO'],
    'CVX': ['XOM', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'MPC'],

    // Streaming/Entertainment
    'NFLX': ['DIS', 'PARA', 'WBD', 'CMCSA', 'SPOT', 'ROKU', 'GOOGL'],
    'DIS': ['NFLX', 'PARA', 'WBD', 'CMCSA', 'LYV', 'SONY', 'FOX'],
  };

  const upperSymbol = symbol.toUpperCase();
  const related = industryGroups[upperSymbol] || ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'TSLA'];

  console.log(`[Related Stocks API] Using fallback for ${symbol}: ${related.slice(0, 8).join(', ')}`);

  return NextResponse.json({
    success: true,
    symbol,
    relatedStocks: related.slice(0, 8),
    source: 'fallback'
  });
}
