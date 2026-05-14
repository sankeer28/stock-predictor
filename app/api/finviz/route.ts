import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';

type ScreenerTable = 'Overview' | 'Valuation' | 'Financial' | 'Ownership' | 'Performance' | 'Technical';

const FINVIZ_BASE = 'https://finviz.com';
const TABLE_CODES: Record<ScreenerTable, string> = {
  Overview: '111',
  Valuation: '121',
  Ownership: '131',
  Performance: '141',
  Financial: '161',
  Technical: '171',
};

const USER_AGENT =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36';

const decodeHtml = (value: string) =>
  value
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(Number(code)))
    .replace(/&#x([a-fA-F0-9]+);/g, (_, code) => String.fromCharCode(parseInt(code, 16)));

const textFromHtml = (html: string) =>
  decodeHtml(
    html
      .replace(/<script[\s\S]*?<\/script>/gi, '')
      .replace(/<style[\s\S]*?<\/style>/gi, '')
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
  );

const absolutizeUrl = (url: string) => {
  const clean = decodeHtml(url.trim());
  if (!clean) return '';
  if (clean.startsWith('//')) return `https:${clean}`;
  if (clean.startsWith('http')) return clean;
  if (clean.startsWith('/')) return `${FINVIZ_BASE}${clean}`;
  return clean;
};

const matchAll = (html: string, regex: RegExp) => Array.from(html.matchAll(regex));

const getAttr = (tag: string, attr: string) => {
  const match = tag.match(new RegExp(`${attr}=["']([^"']+)["']`, 'i'));
  return match?.[1] || '';
};

async function fetchFinviz(path: string) {
  const response = await fetch(`${FINVIZ_BASE}${path}`, {
    headers: {
      'User-Agent': USER_AGENT,
      Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
    },
    cache: 'no-store',
    signal: AbortSignal.timeout(12000),
  });

  if (!response.ok) {
    throw new Error(`Finviz returned ${response.status}`);
  }

  return response.text();
}

function parseStockSnapshot(html: string, symbol: string) {
  const data: Record<string, string | null> = { Ticker: symbol };
  const ticker = html.match(/<h1[^>]*quote-header_ticker-wrapper_ticker[^>]*>([\s\S]*?)<\/h1>/i);
  if (ticker) data.Ticker = textFromHtml(ticker[1]) || symbol;

  const company = html.match(/<h2[^>]*quote-header_ticker-wrapper_company[^>]*>[\s\S]*?<a[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/i);
  if (company) {
    data.Company = textFromHtml(company[2]);
    data.Website = absolutizeUrl(company[1]);
  }

  const quoteLinks = html.match(/<div[^>]*quote-links[^>]*>([\s\S]*?)<\/div>/i)?.[1] || '';
  const profileLinks = matchAll(quoteLinks, /<a[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/gi)
    .filter(match => /f=(sec_|ind_|geo_)/.test(match[1]))
    .map(match => textFromHtml(match[2]));

  if (profileLinks[0]) data.Sector = profileLinks[0];
  if (profileLinks[1]) data.Industry = profileLinks[1];
  if (profileLinks[2]) data.Country = profileLinks[2];

  const cells = matchAll(html, /<td[^>]*snapshot-td2[^>]*>([\s\S]*?)<\/td>/gi).map(match => textFromHtml(match[1]));
  for (let i = 0; i < cells.length - 1; i += 2) {
    const label = cells[i];
    const value = cells[i + 1];
    if (!label) continue;

    if (label === 'EPS next Y' && data['EPS next Y']) {
      data['EPS growth next Y'] = value;
      continue;
    }

    if (label === 'Volatility') {
      const [week, month] = value.split(/\s+/);
      data['Volatility (Week)'] = week || value;
      data['Volatility (Month)'] = month || week || value;
      continue;
    }

    data[label] = value || null;
  }

  return data;
}

function parseNews(html: string) {
  const table = html.match(/<table[^>]*id=["']news-table["'][^>]*>([\s\S]*?)<\/table>/i)?.[1] || '';
  return matchAll(table, /<tr[^>]*>([\s\S]*?)<\/tr>/gi).map(match => {
    const cells = matchAll(match[1], /<td[^>]*>([\s\S]*?)<\/td>/gi).map(cell => cell[1]);
    const link = cells[1]?.match(/<a[^>]*href=["']([^"']+)["'][^>]*>([\s\S]*?)<\/a>/i);
    const source = cells[1]?.match(/news-link-right[\s\S]*?<span[^>]*>([\s\S]*?)<\/span>/i);
    return {
      timestamp: textFromHtml(cells[0] || ''),
      headline: link ? textFromHtml(link[2]) : '',
      url: link ? absolutizeUrl(link[1]) : '',
      source: source ? textFromHtml(source[1]).replace(/^\(|\)$/g, '') : '',
    };
  }).filter(item => item.headline);
}

function parseTableByText(html: string, marker: string) {
  const tables = matchAll(html, /<table\b[^>]*>([\s\S]*?)<\/table>/gi);
  return tables.find(match => textFromHtml(match[0]).includes(marker))?.[0] || '';
}

function parseGenericTable(tableHtml: string) {
  if (!tableHtml) return { headers: [] as string[], rows: [] as Record<string, string>[] };

  let headers = matchAll(tableHtml, /<th[^>]*>([\s\S]*?)<\/th>/gi).map(match => textFromHtml(match[1])).filter(Boolean);
  const allRows = matchAll(tableHtml, /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi);

  if (!headers.length && allRows.length) {
    headers = matchAll(allRows[0][1], /<td\b[^>]*>([\s\S]*?)<\/td>/gi).map(match => textFromHtml(match[1])).filter(Boolean);
  }

  const rows = allRows.map((row, index) => {
    const values = matchAll(row[1], /<td\b[^>]*>([\s\S]*?)<\/td>/gi).map(match => textFromHtml(match[1]));
    if (!values.length || (index === 0 && values.join('|') === headers.join('|'))) return null;
    return headers.reduce<Record<string, string>>((acc, header, cellIndex) => {
      acc[header || `Column ${cellIndex + 1}`] = values[cellIndex] || '';
      return acc;
    }, {});
  }).filter(Boolean) as Record<string, string>[];

  return { headers, rows };
}

function parseScreenerTable(html: string) {
  const tableMatches = matchAll(html, /<table\b([^>]*)>([\s\S]*?)<\/table>/gi);
  const tableHtml =
    tableMatches.find(match => getAttr(match[1], 'class').split(/\s+/).includes('table-light'))?.[0] ||
    tableMatches
      .map(match => match[0])
      .find(table => {
        const text = textFromHtml(table);
        return /\bTicker\b/.test(text) && /\bCompany\b/.test(text) && /\bPrice\b/.test(text);
      }) ||
    '';

  if (!tableHtml) return { headers: [] as string[], rows: [] as Record<string, string>[] };

  const rowsHtml = matchAll(tableHtml, /<tr\b[^>]*>([\s\S]*?)<\/tr>/gi);
  let headers: string[] = [];

  for (const row of rowsHtml) {
    const cells = matchAll(row[1], /<t[hd]\b[^>]*>([\s\S]*?)<\/t[hd]>/gi)
      .map(match => textFromHtml(match[1]))
      .filter(Boolean);

    if (cells.includes('Ticker') && cells.includes('Company')) {
      headers = cells;
      break;
    }
  }

  const rows = rowsHtml.map(row => {
    const cells = matchAll(row[1], /<td\b[^>]*>([\s\S]*?)<\/td>/gi)
      .map(match => textFromHtml(match[1]))
      .filter(value => value !== '');

    if (!cells.length) return null;
    if (!headers.length && cells.includes('Ticker') && cells.includes('Company')) {
      headers = cells;
      return null;
    }
    if (!headers.length || cells.length !== headers.length) return null;
    if (cells.join('|') === headers.join('|')) return null;

    const rowData = headers.reduce<Record<string, string>>((acc, header, index) => {
      acc[header || `Column ${index + 1}`] = cells[index] || '';
      return acc;
    }, {});

    return rowData.Ticker || rowData.Company ? rowData : null;
  }).filter(Boolean) as Record<string, string>[];

  return { headers, rows };
}

function parseAnalystTargets(html: string) {
  const table =
    html.match(/<table[^>]*(?:js-table-ratings|fullview-ratings-outer)[^>]*>([\s\S]*?)<\/table>/i)?.[0] ||
    parseTableByText(html, 'Analyst');

  return parseGenericTable(table).rows.slice(0, 20).map(row => {
    const values = Object.values(row).filter(Boolean);
    return {
      date: values[0] || '',
      category: values[1] || '',
      analyst: values[2] || '',
      rating: values[3] || '',
      target: values[4] || '',
    };
  }).filter(row => row.date && row.analyst);
}

async function parseScreener(symbol: string) {
  const entries = await Promise.all(
    Object.entries(TABLE_CODES).map(async ([table, code]) => {
      try {
        const html = await fetchFinviz(`/screener.ashx?v=${code}&t=${encodeURIComponent(symbol)}`);
        return [table, parseScreenerTable(html)] as const;
      } catch {
        return [table, { headers: [], rows: [] }] as const;
      }
    })
  );

  return Object.fromEntries(entries);
}

export async function GET(request: NextRequest) {
  const symbol = request.nextUrl.searchParams.get('symbol')?.trim().toUpperCase();

  if (!symbol) {
    return NextResponse.json({ success: false, error: 'Symbol parameter is required' }, { status: 400 });
  }

  try {
    const quoteHtml = await fetchFinviz(`/quote.ashx?t=${encodeURIComponent(symbol)}`);
    const stock = parseStockSnapshot(quoteHtml, symbol);
    const screener = await parseScreener(symbol);

    return NextResponse.json({
      success: true,
      symbol,
      source: 'finviz',
      delayedQuoteNotice: 'Finviz quote data is delayed and is not for live trading.',
      stock,
      news: parseNews(quoteHtml).slice(0, 30),
      insider: parseGenericTable(parseTableByText(quoteHtml, 'Insider Trading')).rows.slice(0, 20),
      analystTargets: parseAnalystTargets(quoteHtml),
      screener,
      charts: {
        dailyCandle: `${FINVIZ_BASE}/chart.ashx?t=${encodeURIComponent(symbol)}&ty=c&ta=1&p=d&s=l`,
        weeklyCandle: `${FINVIZ_BASE}/chart.ashx?t=${encodeURIComponent(symbol)}&ty=c&ta=1&p=w&s=l`,
        monthlyCandle: `${FINVIZ_BASE}/chart.ashx?t=${encodeURIComponent(symbol)}&ty=c&ta=1&p=m&s=l`,
        dailyLine: `${FINVIZ_BASE}/chart.ashx?t=${encodeURIComponent(symbol)}&ty=l&ta=1&p=d&s=l`,
      },
      links: {
        quote: `${FINVIZ_BASE}/quote.ashx?t=${encodeURIComponent(symbol)}`,
        screener: `${FINVIZ_BASE}/screener.ashx?t=${encodeURIComponent(symbol)}`,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[Finviz API] Error:', error);
    return NextResponse.json(
      { success: false, error: error.message || 'Failed to fetch Finviz data' },
      { status: 500 }
    );
  }
}
