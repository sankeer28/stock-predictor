import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';

const FINVIZ_BASE = 'https://finviz.com';
const USER_AGENT =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';

// ─── HTML helpers ─────────────────────────────────────────────────────────────

function decodeHtml(v: string): string {
  return v
    .replace(/&nbsp;/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&quot;/g, '"')
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&#(\d+);/g, (_, c) => String.fromCharCode(Number(c)))
    .replace(/&#x([a-fA-F0-9]+);/g, (_, c) => String.fromCharCode(parseInt(c, 16)));
}

function textFromHtml(html: string): string {
  return decodeHtml(
    html
      .replace(/<script[\s\S]*?<\/script>/gi, '')
      .replace(/<style[\s\S]*?<\/style>/gi, '')
      .replace(/<[^>]+>/g, ' ')
      .replace(/\s+/g, ' ')
      .trim()
  );
}

/**
 * Find the COMPLETE outer <table> that contains `className` in its class attribute.
 * Uses an explicit depth counter instead of regex to handle nested tables correctly.
 */
function findTableByClass(html: string, className: string): string {
  // Match class attribute containing the className (escaped for regex)
  const escaped = className.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/-/g, '[\\-]');
  const re = new RegExp(`<table\\b[^>]*class=["'][^"']*${escaped}[^"']*["'][^>]*>`, 'i');
  const startIdx = html.search(re);
  if (startIdx === -1) return '';

  let depth = 0;
  let pos = startIdx;

  while (pos < html.length) {
    const open = html.indexOf('<table', pos);
    const close = html.indexOf('</table', pos);

    if (open === -1 && close === -1) break;

    if (open !== -1 && (close === -1 || open < close)) {
      depth++;
      pos = open + 6;
    } else {
      depth--;
      if (depth === 0) {
        const end = html.indexOf('>', close) + 1;
        return html.slice(startIdx, end);
      }
      pos = close + 8;
    }
  }
  return '';
}

/**
 * Extract complete outer table HTML starting at `startPos`.
 */
function extractTableAt(html: string, startPos: number): { html: string; end: number } {
  let depth = 0;
  let pos = startPos;

  while (pos < html.length) {
    const open = html.indexOf('<table', pos);
    const close = html.indexOf('</table', pos);

    if (open === -1 && close === -1) break;

    if (open !== -1 && (close === -1 || open < close)) {
      depth++;
      pos = open + 6;
    } else {
      depth--;
      if (depth === 0) {
        const end = html.indexOf('>', close) + 1;
        return { html: html.slice(startPos, end), end };
      }
      pos = close + 8;
    }
  }
  return { html: '', end: startPos };
}

/**
 * Extract all <tr>...</tr> segments linearly (avoids nested-table regex issues).
 */
function extractRows(html: string): string[] {
  const rows: string[] = [];
  let pos = 0;

  while (pos < html.length) {
    const start = html.indexOf('<tr', pos);
    if (start === -1) break;
    const end = html.indexOf('</tr>', start);
    if (end === -1) break;
    rows.push(html.slice(start, end + 5));
    pos = end + 5;
  }

  return rows;
}

// ─── Fetch ────────────────────────────────────────────────────────────────────

async function fetchFinviz(path: string): Promise<string> {
  const res = await fetch(`${FINVIZ_BASE}${path}`, {
    headers: {
      'User-Agent': USER_AGENT,
      Accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
      Referer: 'https://finviz.com/screener.ashx',
      'Cache-Control': 'no-cache',
      Pragma: 'no-cache',
    },
    cache: 'no-store',
    signal: AbortSignal.timeout(20000),
  });
  if (!res.ok) throw new Error(`FinViz returned ${res.status}`);
  return res.text();
}

// ─── Parsing ──────────────────────────────────────────────────────────────────

function parseTotal(html: string): number {
  const patterns = [
    /Total:\s*#?([\d,]+)/i,
    /class=["'][^"']*count-text[^"']*["'][^>]*>[^<]*#?([\d,]+)/i,
    /(\d[\d,]+)\s+(?:stocks?|results?|found)/i,
    /#?([\d,]+)\s+results?/i,
    /of\s+([\d,]+)\s+stocks?/i,
  ];
  for (const p of patterns) {
    const m = html.match(p);
    if (m) {
      const n = parseInt(m[1].replace(/,/g, ''), 10);
      if (n > 20) return n; // ignore page-size sized numbers (≤20) to avoid false positives
    }
  }
  return 0;
}

function findHeaderRow(rows: string[]): { headers: string[]; idx: number } {
  for (let i = 0; i < rows.length; i++) {
    const cells = Array.from(rows[i].matchAll(/<t[hd]\b[^>]*>([\s\S]*?)<\/t[hd]>/gi))
      .map(m => textFromHtml(m[1]))
      .filter(Boolean);

    // "Ticker" appears in every FinViz view's header row.
    // Require at least 3 cells so we don't accidentally match a sparse data row.
    if (cells.includes('Ticker') && cells.length >= 3) {
      return { headers: cells, idx: i };
    }
  }
  return { headers: [], idx: -1 };
}

function extractDataRows(
  allRows: string[],
  headers: string[],
  startFrom: number
): { headers: string[]; rows: Record<string, string>[] } {
  const rows: Record<string, string>[] = [];
  const tickerColIdx = headers.indexOf('Ticker');

  for (let i = startFrom; i < allRows.length; i++) {
    const row = allRows[i];

    const cells = Array.from(row.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/gi)).map(c =>
      textFromHtml(c[1])
    );

    if (!cells.length) continue;

    // Primary check: FinViz data rows link to a stock quote page
    const hasQuoteLink =
      row.includes('quote.ashx') ||
      row.includes('/quote?t=') ||
      row.includes('/quote/');

    // Fallback: the Ticker cell looks like a valid stock symbol
    const tickerVal = tickerColIdx >= 0 ? (cells[tickerColIdx] ?? '') : (cells[0] ?? '');
    const looksLikeTicker = /^[A-Z]{1,5}[A-Z.]?$/.test(tickerVal);

    if (!hasQuoteLink && !looksLikeTicker) continue;

    const obj: Record<string, string> = {};
    headers.forEach((h, idx) => {
      obj[h] = cells[idx] ?? '';
    });

    if (!obj['Ticker'] || obj['Ticker'] === 'Ticker') continue;
    rows.push(obj);
    if (rows.length >= 20) break;
  }

  return { headers, rows };
}

function parseScreenerRows(html: string): { headers: string[]; rows: Record<string, string>[] } {
  // Strategy 1: Try known FinViz screener table CSS classes
  for (const cls of ['table-light', 'screener_table', 'screener-content', 'styled-table-new']) {
    const tableHtml = findTableByClass(html, cls);
    if (!tableHtml) continue;
    const allRows = extractRows(tableHtml);
    const { headers, idx } = findHeaderRow(allRows);
    if (headers.length && idx >= 0) {
      const result = extractDataRows(allRows, headers, idx + 1);
      if (result.rows.length > 0) return result;
    }
  }

  // Strategy 2: Scan every top-level table for one with a "Ticker" header row
  let pos = 0;
  while (pos < html.length) {
    const tableStart = html.indexOf('<table', pos);
    if (tableStart === -1) break;

    const { html: tableHtml, end } = extractTableAt(html, tableStart);
    if (!tableHtml) { pos = tableStart + 6; continue; }

    const allRows = extractRows(tableHtml);
    const { headers, idx } = findHeaderRow(allRows);
    if (headers.length && idx >= 0) {
      const result = extractDataRows(allRows, headers, idx + 1);
      if (result.rows.length > 0) return result;
    }

    pos = end;
  }

  // Strategy 3: Last resort — extract rows from full HTML
  const allRows = extractRows(html);
  const { headers, idx } = findHeaderRow(allRows);
  if (headers.length && idx >= 0) {
    return extractDataRows(allRows, headers, idx + 1);
  }

  return { headers: [], rows: [] };
}

// ─── Route handler ────────────────────────────────────────────────────────────

export async function GET(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const filters = (sp.get('filters') || '').trim();
  const page = Math.max(1, parseInt(sp.get('page') || '1'));
  const view = sp.get('view') || '111'; // 111 = Overview
  const rowStart = (page - 1) * 20 + 1;

  try {
    const filterPart = filters ? `&f=${filters}` : '';
    const path = `/screener.ashx?v=${view}${filterPart}&r=${rowStart}`;
    const html = await fetchFinviz(path);

    // Detect bot-blocking pages
    if (
      html.includes('Access Denied') ||
      (html.includes('Please enable JavaScript') && !html.includes('Ticker'))
    ) {
      return NextResponse.json(
        { success: false, error: 'FinViz blocked the request. Try again in a moment.' },
        { status: 503 }
      );
    }

    const total = parseTotal(html);
    const { headers, rows } = parseScreenerRows(html);

    return NextResponse.json(
      {
        success: true,
        filters,
        view,
        page,
        pageSize: 20,
        total,
        totalPages: total > 0 ? Math.ceil(total / 20) : rows.length > 0 ? 1 : 0,
        headers,
        rows,
      },
      { headers: { 'Cache-Control': 'no-store' } }
    );
  } catch (err: any) {
    console.error('[screener]', err);
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
