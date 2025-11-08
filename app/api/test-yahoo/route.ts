import { NextRequest, NextResponse } from 'next/server';

export async function GET(request: NextRequest) {
  const symbol = 'AAPL';
  const results: any = {};

  try {
    // Test 1: Chart API
    console.log('\n=== TESTING CHART API ===');
    const chartUrl = `https://query1.finance.yahoo.com/v8/finance/chart/${symbol}`;
    const chartRes = await fetch(chartUrl);
    const chartData = await chartRes.json();
    console.log('Chart status:', chartRes.status);
    console.log('Chart meta keys:', Object.keys(chartData.chart?.result?.[0]?.meta || {}));
    results.chartMeta = chartData.chart?.result?.[0]?.meta;
    results.chartStatus = chartRes.status;
  } catch (err: any) {
    results.chartError = err.message;
    console.error('Chart error:', err);
  }

  try {
    // Test 2: Quote API v7
    console.log('\n=== TESTING QUOTE API V7 ===');
    const quoteUrl = `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${symbol}`;
    const quoteRes = await fetch(quoteUrl, {
      headers: { 'User-Agent': 'Mozilla/5.0' }
    });
    console.log('Quote status:', quoteRes.status);
    const quoteData = await quoteRes.json();
    console.log('Quote full response:', JSON.stringify(quoteData, null, 2));

    const quote = quoteData.quoteResponse?.result?.[0];
    console.log('Quote keys:', quote ? Object.keys(quote) : 'NO QUOTE DATA');
    console.log('Quote sector:', quote?.sector);
    console.log('Quote industry:', quote?.industry);

    results.quoteData = quote;
    results.quoteStatus = quoteRes.status;
    results.quoteRaw = quoteData;
  } catch (err: any) {
    results.quoteError = err.message;
    console.error('Quote error:', err);
  }

  try {
    // Test 3: Quote Summary v10
    console.log('\n=== TESTING QUOTE SUMMARY V10 ===');
    const summaryUrl = `https://query1.finance.yahoo.com/v10/finance/quoteSummary/${symbol}?modules=summaryDetail,financialData,defaultKeyStatistics,summaryProfile,price`;
    const summaryRes = await fetch(summaryUrl, {
      headers: { 'User-Agent': 'Mozilla/5.0' }
    });
    console.log('Summary status:', summaryRes.status);
    const summaryData = await summaryRes.json();
    console.log('Summary full response:', JSON.stringify(summaryData, null, 2));

    results.summaryData = summaryData.quoteSummary?.result?.[0];
    results.summaryStatus = summaryRes.status;
    results.summaryRaw = summaryData;
  } catch (err: any) {
    results.summaryError = err.message;
    console.error('Summary error:', err);
  }

  return NextResponse.json(results);
}
