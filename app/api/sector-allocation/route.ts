import { NextRequest, NextResponse } from 'next/server';

/**
 * Fetch sector allocation for a portfolio of stocks
 * GET /api/sector-allocation?symbols=AAPL,MSFT,JPM,JNJ
 */
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const symbolsParam = searchParams.get('symbols');

  if (!symbolsParam) {
    return NextResponse.json(
      { error: 'symbols parameter is required (comma-separated)' },
      { status: 400 }
    );
  }

  const symbols = symbolsParam.split(',').map(s => s.trim().toUpperCase()).filter(Boolean);

  if (symbols.length === 0) {
    return NextResponse.json(
      { error: 'At least 1 symbol is required' },
      { status: 400 }
    );
  }

  const MASSIVE_API_KEY = process.env.MASSIVE;

  if (!MASSIVE_API_KEY) {
    console.error('MASSIVE API key not configured');
    return NextResponse.json(
      { error: 'API key not configured' },
      { status: 500 }
    );
  }

  try {
    console.log(`[Sector Allocation API] Analyzing symbols: ${symbols.join(', ')}`);

    // Fetch company info for all symbols in parallel
    const companyPromises = symbols.map(async (symbol) => {
      try {
        const url = `https://api.massive.com/v3/reference/tickers/${symbol}?apiKey=${MASSIVE_API_KEY}`;

        const response = await fetch(url, {
          headers: {
            'Accept': 'application/json',
          },
          cache: 'force-cache', // Cache sector data as it rarely changes
        });

        if (!response.ok) {
          console.error(`[Sector Allocation API] Failed to fetch ${symbol}: ${response.status}`);
          return { symbol, sector: 'Other', error: `HTTP ${response.status}` };
        }

        const data = await response.json();

        if (data.status !== 'OK' || !data.results) {
          console.error(`[Sector Allocation API] No data for ${symbol}`);
          return { symbol, sector: 'Other', error: 'No data' };
        }

        // Extract sector from SIC description
        const sicDescription = data.results.sic_description || 'Other';
        const sector = categorizeSector(sicDescription);

        return {
          symbol,
          sector,
          companyName: data.results.name,
          sicDescription,
        };
      } catch (error) {
        console.error(`[Sector Allocation API] Error fetching ${symbol}:`, error);
        return { symbol, sector: 'Other', error: 'Fetch failed' };
      }
    });

    const companies = await Promise.all(companyPromises);

    // Calculate sector allocation
    const sectorAllocation: Record<string, number> = {};

    companies.forEach(company => {
      sectorAllocation[company.sector] = (sectorAllocation[company.sector] || 0) + 1;
    });

    // Sort by count descending
    const sortedSectors = Object.entries(sectorAllocation)
      .sort(([, a], [, b]) => b - a)
      .reduce((acc, [key, value]) => ({ ...acc, [key]: value }), {});

    return NextResponse.json({
      success: true,
      sectorAllocation: sortedSectors,
      companies,
      totalCompanies: companies.length,
    });

  } catch (error: any) {
    console.error('[Sector Allocation API] Error:', error);
    return NextResponse.json(
      { error: error.message || 'Failed to fetch sector allocation' },
      { status: 500 }
    );
  }
}

/**
 * Categorize SIC description into broader sector categories
 */
function categorizeSector(sicDescription: string): string {
  const desc = sicDescription.toLowerCase();

  // Technology
  if (
    desc.includes('computer') ||
    desc.includes('software') ||
    desc.includes('semiconductor') ||
    desc.includes('electronic') ||
    desc.includes('internet') ||
    desc.includes('technology') ||
    desc.includes('data processing')
  ) {
    return 'Technology';
  }

  // Finance
  if (
    desc.includes('bank') ||
    desc.includes('financial') ||
    desc.includes('insurance') ||
    desc.includes('investment') ||
    desc.includes('credit') ||
    desc.includes('securities')
  ) {
    return 'Finance';
  }

  // Healthcare
  if (
    desc.includes('pharmaceutical') ||
    desc.includes('medical') ||
    desc.includes('health') ||
    desc.includes('hospital') ||
    desc.includes('biotech') ||
    desc.includes('drug')
  ) {
    return 'Healthcare';
  }

  // Consumer
  if (
    desc.includes('retail') ||
    desc.includes('restaurant') ||
    desc.includes('food') ||
    desc.includes('beverage') ||
    desc.includes('apparel') ||
    desc.includes('consumer')
  ) {
    return 'Consumer';
  }

  // Energy
  if (
    desc.includes('oil') ||
    desc.includes('gas') ||
    desc.includes('energy') ||
    desc.includes('petroleum') ||
    desc.includes('coal')
  ) {
    return 'Energy';
  }

  // Industrial
  if (
    desc.includes('manufacturing') ||
    desc.includes('industrial') ||
    desc.includes('machinery') ||
    desc.includes('aerospace') ||
    desc.includes('defense') ||
    desc.includes('construction')
  ) {
    return 'Industrial';
  }

  // Telecom
  if (
    desc.includes('telecommunication') ||
    desc.includes('telecom') ||
    desc.includes('wireless') ||
    desc.includes('communication')
  ) {
    return 'Telecom';
  }

  // Utilities
  if (
    desc.includes('electric') ||
    desc.includes('utility') ||
    desc.includes('water') ||
    desc.includes('power')
  ) {
    return 'Utilities';
  }

  // Real Estate
  if (
    desc.includes('real estate') ||
    desc.includes('reit') ||
    desc.includes('property')
  ) {
    return 'Real Estate';
  }

  // Materials
  if (
    desc.includes('mining') ||
    desc.includes('metal') ||
    desc.includes('chemical') ||
    desc.includes('materials') ||
    desc.includes('steel') ||
    desc.includes('paper')
  ) {
    return 'Materials';
  }

  return 'Other';
}
