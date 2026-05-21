import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';
export const maxDuration = 30;

const FINVIZ_BASE = 'https://finviz.com';
const USER_AGENT =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';

const COMMON_HEADERS = {
  'User-Agent': USER_AGENT,
  'Accept-Language': 'en-US,en;q=0.9',
  'Accept-Encoding': 'gzip, deflate, br',
  'Referer': 'https://finviz.com/',
  'Cache-Control': 'no-cache',
  'Pragma': 'no-cache',
  'Upgrade-Insecure-Requests': '1',
  'Sec-Fetch-Dest': 'document',
  'Sec-Fetch-Mode': 'navigate',
  'Sec-Fetch-Site': 'same-origin',
  'Sec-Fetch-User': '?1',
};

// ─── CSV parsing (primary strategy) ──────────────────────────────────────────

function parseCSVRow(line: string): string[] {
  const cells: string[] = [];
  let cur = '';
  let inQ = false;
  for (const ch of line) {
    if (ch === '"') { inQ = !inQ; }
    else if (ch === ',' && !inQ) { cells.push(cur); cur = ''; }
    else { cur += ch; }
  }
  cells.push(cur);
  return cells.map(c => c.trim().replace(/^"|"$/g, ''));
}

function parseCSVData(text: string): { headers: string[]; rows: Record<string, string>[] } {
  const lines = text.trim().split('\n').filter(Boolean);
  if (lines.length < 2) return { headers: [], rows: [] };

  const rawHeaders = parseCSVRow(lines[0]);
  const headers = rawHeaders.filter(h => h !== 'No.' && h !== 'No' && h !== '#');
  const tickerIdx = rawHeaders.indexOf('Ticker');
  if (tickerIdx === -1) return { headers: [], rows: [] };

  const rows: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = parseCSVRow(lines[i]);
    const ticker = cells[tickerIdx] ?? '';
    if (!ticker || ticker === 'Ticker') continue;

    const obj: Record<string, string> = {};
    rawHeaders.forEach((h, idx) => { obj[h] = cells[idx] ?? ''; });
    rows.push(obj);
  }

  return { headers, rows };
}

// ─── HTML helpers (fallback) ──────────────────────────────────────────────────

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

function findTableByClass(html: string, className: string): string {
  const escaped = className.replace(/[.*+?^${}()|[\]\\]/g, '\\$&').replace(/-/g, '[\\-]');
  const re = new RegExp(`<table\\b[^>]*class=["'][^"']*${escaped}[^"']*["'][^>]*>`, 'i');
  const startIdx = html.search(re);
  if (startIdx === -1) return '';

  let depth = 0, pos = startIdx;
  while (pos < html.length) {
    const open = html.indexOf('<table', pos);
    const close = html.indexOf('</table', pos);
    if (open === -1 && close === -1) break;
    if (open !== -1 && (close === -1 || open < close)) { depth++; pos = open + 6; }
    else {
      depth--;
      if (depth === 0) { const end = html.indexOf('>', close) + 1; return html.slice(startIdx, end); }
      pos = close + 8;
    }
  }
  return '';
}

function extractTableAt(html: string, startPos: number): { html: string; end: number } {
  let depth = 0, pos = startPos;
  while (pos < html.length) {
    const open = html.indexOf('<table', pos);
    const close = html.indexOf('</table', pos);
    if (open === -1 && close === -1) break;
    if (open !== -1 && (close === -1 || open < close)) { depth++; pos = open + 6; }
    else {
      depth--;
      if (depth === 0) { const end = html.indexOf('>', close) + 1; return { html: html.slice(startPos, end), end }; }
      pos = close + 8;
    }
  }
  return { html: '', end: startPos };
}

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

function findHeaderRow(rows: string[]): { headers: string[]; idx: number } {
  for (let i = 0; i < rows.length; i++) {
    const cells = Array.from(rows[i].matchAll(/<t[hd]\b[^>]*>([\s\S]*?)<\/t[hd]>/gi))
      .map(m => textFromHtml(m[1])).filter(Boolean);
    if (cells.includes('Ticker') && cells.length >= 3) return { headers: cells, idx: i };
  }
  return { headers: [], idx: -1 };
}

function extractDataRows(
  allRows: string[], headers: string[], startFrom: number
): { headers: string[]; rows: Record<string, string>[] } {
  const rows: Record<string, string>[] = [];
  const tickerColIdx = headers.indexOf('Ticker');

  for (let i = startFrom; i < allRows.length; i++) {
    const row = allRows[i];
    const cells = Array.from(row.matchAll(/<td\b[^>]*>([\s\S]*?)<\/td>/gi)).map(c => textFromHtml(c[1]));
    if (!cells.length) continue;

    const hasQuoteLink = row.includes('quote.ashx') || row.includes('/quote?t=') || row.includes('/quote/');
    const tickerVal = tickerColIdx >= 0 ? (cells[tickerColIdx] ?? '') : (cells[0] ?? '');
    const looksLikeTicker = /^[A-Z]{1,5}[A-Z.]?$/.test(tickerVal);
    if (!hasQuoteLink && !looksLikeTicker) continue;

    const obj: Record<string, string> = {};
    headers.forEach((h, idx) => { obj[h] = cells[idx] ?? ''; });
    if (!obj['Ticker'] || obj['Ticker'] === 'Ticker') continue;
    rows.push(obj);
    if (rows.length >= 20) break;
  }
  return { headers, rows };
}

function parseScreenerRows(html: string): { headers: string[]; rows: Record<string, string>[] } {
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

  const allRows = extractRows(html);
  const { headers, idx } = findHeaderRow(allRows);
  if (headers.length && idx >= 0) return extractDataRows(allRows, headers, idx + 1);

  return { headers: [], rows: [] };
}

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
      if (n > 20) return n;
    }
  }
  return 0;
}

// ─── Route handler ────────────────────────────────────────────────────────────

export async function GET(request: NextRequest) {
  const sp = request.nextUrl.searchParams;
  const filters = (sp.get('filters') || '').trim();
  const page    = Math.max(1, parseInt(sp.get('page') || '1'));
  const rowStart = (page - 1) * 20 + 1;
  const filterPart = filters ? `&f=${filters}` : '';

  // ── Strategy 1: CSV export (lighter, works better from cloud IPs) ─────────
  try {
    const csvRes = await fetch(`${FINVIZ_BASE}/export.ashx?v=111${filterPart}`, {
      headers: { ...COMMON_HEADERS, Accept: 'text/csv,text/plain,*/*' },
      cache: 'no-store',
      signal: AbortSignal.timeout(9000),
    });

    if (csvRes.ok) {
      const text = await csvRes.text();
      if (text.includes('Ticker') && !text.includes('<html')) {
        const { headers, rows: allRows } = parseCSVData(text);
        if (allRows.length > 0) {
          const pageRows = allRows.slice(rowStart - 1, rowStart - 1 + 20);
          return NextResponse.json(
            {
              success: true, filters, page, pageSize: 20,
              total: allRows.length,
              totalPages: Math.ceil(allRows.length / 20),
              headers, rows: pageRows,
            },
            { headers: { 'Cache-Control': 'no-store' } }
          );
        }
      }
    }
  } catch {
    // CSV failed — fall through to HTML scraping
  }

  // ── Strategy 2: HTML scraping ─────────────────────────────────────────────
  try {
    const htmlRes = await fetch(
      `${FINVIZ_BASE}/screener.ashx?v=111${filterPart}&r=${rowStart}`,
      {
        headers: { ...COMMON_HEADERS, Accept: 'text/html,application/xhtml+xml,*/*;q=0.8' },
        cache: 'no-store',
        signal: AbortSignal.timeout(9000),
      }
    );

    if (!htmlRes.ok) throw new Error(`FinViz returned ${htmlRes.status}`);
    const html = await htmlRes.text();

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
        success: true, filters, page, pageSize: 20,
        total,
        totalPages: total > 0 ? Math.ceil(total / 20) : rows.length > 0 ? 1 : 0,
        headers, rows,
      },
      { headers: { 'Cache-Control': 'no-store' } }
    );
  } catch (err: any) {
    console.error('[screener]', err);
    return NextResponse.json({ success: false, error: err.message }, { status: 500 });
  }
}
