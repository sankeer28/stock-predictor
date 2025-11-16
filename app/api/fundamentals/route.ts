import { NextRequest, NextResponse } from 'next/server';

/**
 * Get stock fundamentals from Alpha Vantage
 * GET /api/fundamentals?symbol=AAPL
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol');
  
  if (!symbol) {
    return NextResponse.json(
      { error: 'Symbol parameter is required' },
      { status: 400 }
    );
  }

  const API_KEY = process.env.ALPHA_VANTAGE_API_KEY;
  
  if (!API_KEY) {
    return NextResponse.json(
      { error: 'Alpha Vantage API key not configured' },
      { status: 500 }
    );
  }

  try {
    console.log(`[Fundamentals API] Fetching data for ${symbol}`);

    // Fetch company overview from Alpha Vantage
    const response = await fetch(
      `https://www.alphavantage.co/query?function=OVERVIEW&symbol=${symbol}&apikey=${API_KEY}`,
      { 
        cache: 'no-store',
        signal: AbortSignal.timeout(10000) 
      }
    );

    if (!response.ok) {
      throw new Error(`Alpha Vantage API error: ${response.status}`);
    }

    const data = await response.json();

    // Check if we got rate limited or invalid symbol
    if (data.Note) {
      return NextResponse.json(
        { error: 'API rate limit reached. Please try again later.' },
        { status: 429 }
      );
    }

    if (!data.Symbol || Object.keys(data).length < 5) {
      return NextResponse.json(
        { error: 'Invalid symbol or no data available' },
        { status: 404 }
      );
    }

    // Parse and structure the data
    const fundamentals = {
      // Valuation Metrics
      marketCap: data.MarketCapitalization ? parseInt(data.MarketCapitalization) : null,
      peRatio: data.PERatio ? parseFloat(data.PERatio) : null,
      pegRatio: data.PEGRatio ? parseFloat(data.PEGRatio) : null,
      priceToBook: data.PriceToBookRatio ? parseFloat(data.PriceToBookRatio) : null,
      priceToSales: data.PriceToSalesRatioTTM ? parseFloat(data.PriceToSalesRatioTTM) : null,
      evToRevenue: data.EVToRevenue ? parseFloat(data.EVToRevenue) : null,
      evToEbitda: data.EVToEBITDA ? parseFloat(data.EVToEBITDA) : null,
      
      // Profitability Metrics
      profitMargin: data.ProfitMargin ? parseFloat(data.ProfitMargin) : null,
      operatingMargin: data.OperatingMarginTTM ? parseFloat(data.OperatingMarginTTM) : null,
      grossMargin: data.GrossProfitTTM && data.RevenueTTM 
        ? parseFloat(data.GrossProfitTTM) / parseFloat(data.RevenueTTM) 
        : null,
      roe: data.ReturnOnEquityTTM ? parseFloat(data.ReturnOnEquityTTM) : null,
      roa: data.ReturnOnAssetsTTM ? parseFloat(data.ReturnOnAssetsTTM) : null,
      
      // Per Share Metrics
      eps: data.EPS ? parseFloat(data.EPS) : null,
      bookValue: data.BookValue ? parseFloat(data.BookValue) : null,
      revenuePerShare: data.RevenuePerShareTTM ? parseFloat(data.RevenuePerShareTTM) : null,
      
      // Financial Health
      debtToEquity: data.DebtToEquity ? parseFloat(data.DebtToEquity) : null,
      currentRatio: data.CurrentRatio ? parseFloat(data.CurrentRatio) : null,
      quickRatio: data.QuickRatio ? parseFloat(data.QuickRatio) : null,
      
      // Growth Metrics
      revenueGrowth: data.QuarterlyRevenueGrowthYOY ? parseFloat(data.QuarterlyRevenueGrowthYOY) : null,
      earningsGrowth: data.QuarterlyEarningsGrowthYOY ? parseFloat(data.QuarterlyEarningsGrowthYOY) : null,
      
      // Dividend Info
      dividendYield: data.DividendYield ? parseFloat(data.DividendYield) : null,
      dividendPerShare: data.DividendPerShare ? parseFloat(data.DividendPerShare) : null,
      payoutRatio: data.PayoutRatio ? parseFloat(data.PayoutRatio) : null,
      exDividendDate: data.ExDividendDate || null,
      dividendDate: data.DividendDate || null,
      
      // Trading Metrics
      beta: data.Beta ? parseFloat(data.Beta) : null,
      week52High: data['52WeekHigh'] ? parseFloat(data['52WeekHigh']) : null,
      week52Low: data['52WeekLow'] ? parseFloat(data['52WeekLow']) : null,
      day50MA: data['50DayMovingAverage'] ? parseFloat(data['50DayMovingAverage']) : null,
      day200MA: data['200DayMovingAverage'] ? parseFloat(data['200DayMovingAverage']) : null,
      
      // Volume
      sharesOutstanding: data.SharesOutstanding ? parseInt(data.SharesOutstanding) : null,
      
      // Analyst Info
      analystTargetPrice: data.AnalystTargetPrice ? parseFloat(data.AnalystTargetPrice) : null,
    };

    // Company Info
    const companyInfo = {
      symbol: data.Symbol,
      name: data.Name,
      description: data.Description,
      sector: data.Sector,
      industry: data.Industry,
      exchange: data.Exchange,
      currency: data.Currency,
      country: data.Country,
      fiscalYearEnd: data.FiscalYearEnd,
      latestQuarter: data.LatestQuarter,
    };

    console.log(`[Fundamentals API] Successfully fetched data for ${symbol}`);

    return NextResponse.json({
      ...companyInfo,
      fundamentals,
      timestamp: new Date().toISOString(),
    });

  } catch (error: any) {
    console.error('[Fundamentals API] Error:', error);
    return NextResponse.json(
      { 
        error: error instanceof Error ? error.message : 'Failed to fetch fundamentals',
        details: error instanceof Error ? error.stack : String(error)
      },
      { status: 500 }
    );
  }
}

