import { StockData } from '@/types';

/**
 * Calculate Pearson correlation coefficient between two arrays
 */
export function calculateCorrelation(x: number[], y: number[]): number {
  const n = Math.min(x.length, y.length);

  if (n === 0) return 0;

  // Calculate means
  const meanX = x.slice(0, n).reduce((a, b) => a + b, 0) / n;
  const meanY = y.slice(0, n).reduce((a, b) => a + b, 0) / n;

  // Calculate correlation coefficient
  let numerator = 0;
  let sumXSquared = 0;
  let sumYSquared = 0;

  for (let i = 0; i < n; i++) {
    const deltaX = x[i] - meanX;
    const deltaY = y[i] - meanY;
    numerator += deltaX * deltaY;
    sumXSquared += deltaX * deltaX;
    sumYSquared += deltaY * deltaY;
  }

  const denominator = Math.sqrt(sumXSquared * sumYSquared);

  if (denominator === 0) return 0;

  return numerator / denominator;
}

/**
 * Calculate correlation matrix for multiple stocks
 */
export interface CorrelationMatrix {
  symbols: string[];
  matrix: number[][];
  startDate: string;
  endDate: string;
}

export function calculateCorrelationMatrix(
  stocksData: { symbol: string; data: StockData[] }[],
  startDate?: string,
  endDate?: string
): CorrelationMatrix {
  // Filter data by date range if specified
  const filteredData = stocksData.map(({ symbol, data }) => {
    let filtered = [...data];

    if (startDate) {
      filtered = filtered.filter(d => new Date(d.date) >= new Date(startDate));
    }

    if (endDate) {
      filtered = filtered.filter(d => new Date(d.date) <= new Date(endDate));
    }

    return { symbol, data: filtered };
  });

  // Find common date range across all stocks
  const allDates = new Set<string>();
  filteredData.forEach(({ data }) => {
    data.forEach(d => allDates.add(d.date));
  });

  const commonDates = Array.from(allDates).sort();

  // Create aligned price arrays for each stock
  const priceArrays = filteredData.map(({ symbol, data }) => {
    const dateMap = new Map(data.map(d => [d.date, d.close]));
    return {
      symbol,
      prices: commonDates.map(date => dateMap.get(date) || 0).filter(p => p > 0)
    };
  });

  // Calculate correlation matrix
  const n = priceArrays.length;
  const matrix: number[][] = [];

  for (let i = 0; i < n; i++) {
    const row: number[] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        row.push(1); // Perfect correlation with itself
      } else {
        const correlation = calculateCorrelation(
          priceArrays[i].prices,
          priceArrays[j].prices
        );
        row.push(correlation);
      }
    }
    matrix.push(row);
  }

  return {
    symbols: priceArrays.map(p => p.symbol),
    matrix,
    startDate: commonDates[0] || '',
    endDate: commonDates[commonDates.length - 1] || ''
  };
}

/**
 * Categorize SIC description into broader sector categories
 */
export function categorizeSector(sicDescription: string): string {
  if (!sicDescription) return 'Technology'; // Default

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

  return 'Technology'; // Default fallback
}

/**
 * Get similar companies in the same sector (placeholder - you can enhance this)
 */
export const SECTOR_COMPANIES: Record<string, string[]> = {
  'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC'],
  'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW'],
  'Healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'DHR'],
  'Consumer': ['AMZN', 'WMT', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'COST'],
  'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO'],
  'Industrial': ['BA', 'HON', 'UPS', 'CAT', 'GE', 'MMM', 'LMT', 'RTX'],
  'Telecom': ['T', 'VZ', 'TMUS', 'CMCSA'],
  'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL'],
  'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'WELL', 'DLR'],
  'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'NUE']
};

export function getSimilarCompanies(sector: string, limit: number = 8): string[] {
  const companies = SECTOR_COMPANIES[sector] || [];
  return companies.slice(0, limit);
}

/**
 * Calculate returns for each stock
 */
export function calculateReturns(data: StockData[]): number[] {
  const returns: number[] = [];

  for (let i = 1; i < data.length; i++) {
    const prevClose = data[i - 1].close;
    const currentClose = data[i].close;

    if (prevClose > 0) {
      const returnPct = ((currentClose - prevClose) / prevClose) * 100;
      returns.push(returnPct);
    }
  }

  return returns;
}
