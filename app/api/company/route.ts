import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbol = searchParams.get('symbol');

  if (!symbol) {
    return NextResponse.json({ error: 'Symbol is required' }, { status: 400 });
  }

  const MASSIVE_API_KEY = process.env.MASSIVE;

  if (!MASSIVE_API_KEY) {
    console.error('MASSIVE API key not configured');
    return NextResponse.json({ error: 'API key not configured' }, { status: 500 });
  }

  try {
    // Fetch ticker details from Massive API
    const url = `https://api.massive.com/v3/reference/tickers/${symbol.toUpperCase()}?apiKey=${MASSIVE_API_KEY}`;

    const response = await fetch(url, {
      headers: {
        'Accept': 'application/json',
      },
    });

    if (!response.ok) {
      console.error(`Massive API error: ${response.status} ${response.statusText}`);
      return NextResponse.json({ error: 'Failed to fetch company data' }, { status: response.status });
    }

    const data = await response.json();

    if (data.status !== 'OK' || !data.results) {
      console.error('Massive API returned error or no results:', data);
      return NextResponse.json({ error: 'No company data found' }, { status: 404 });
    }

    // Transform the data to match our CompanyInfo interface
    // Convert branding URLs to use our proxy endpoint
    const logoUrl = data.results.branding?.logo_url
      ? `/api/company-logo?url=${encodeURIComponent(data.results.branding.logo_url)}`
      : undefined;
    const iconUrl = data.results.branding?.icon_url
      ? `/api/company-logo?url=${encodeURIComponent(data.results.branding.icon_url)}`
      : undefined;

    const companyInfo = {
      ticker: data.results.ticker,
      name: data.results.name,
      description: data.results.description,
      website: data.results.homepage_url,
      marketCap: data.results.market_cap,
      sector: data.results.sic_description,
      industry: data.results.sic_description,
      sicCode: data.results.sic_code,

      // Contact & Location
      phone: data.results.phone_number,
      address: data.results.address,

      // Company Details
      primaryExchange: data.results.primary_exchange,
      totalEmployees: data.results.total_employees,
      listDate: data.results.list_date,
      active: data.results.active,
      delistedDate: data.results.delisted_utc,

      // Shares & Identifiers
      sharesOutstanding: data.results.share_class_shares_outstanding,
      weightedSharesOutstanding: data.results.weighted_shares_outstanding,
      roundLot: data.results.round_lot,
      cik: data.results.cik,
      compositeFigi: data.results.composite_figi,
      shareClassFigi: data.results.share_class_figi,

      // Branding (proxied through our endpoint to add API key)
      logoUrl,
      iconUrl,

      // Market Info
      locale: data.results.locale,
      market: data.results.market,
      type: data.results.type,
      currencyName: data.results.currency_name,
      tickerRoot: data.results.ticker_root,
      tickerSuffix: data.results.ticker_suffix,
    };

    return NextResponse.json({
      success: true,
      companyInfo,
    });

  } catch (error: any) {
    console.error('Error fetching company data from Massive:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch company data' },
      { status: 500 }
    );
  }
}
