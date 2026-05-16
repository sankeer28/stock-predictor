import { ChartDataPoint } from '@/types';

export function exportToCSV(symbol: string, data: ChartDataPoint[]): void {
  if (!data.length) return;

  const headers = [
    'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
    'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'BB_Upper', 'BB_Middle', 'BB_Lower',
  ];

  const rows = data.map(d => [
    d.date ? new Date(d.date).toISOString().slice(0, 10) : '',
    d.open?.toFixed(4) ?? '',
    d.high?.toFixed(4) ?? '',
    d.low?.toFixed(4) ?? '',
    d.close?.toFixed(4) ?? '',
    d.volume ?? '',
    d.ma20?.toFixed(4) ?? '',
    d.ma50?.toFixed(4) ?? '',
    (d as any).ma200?.toFixed(4) ?? '',
    d.rsi?.toFixed(2) ?? '',
    d.macd?.toFixed(6) ?? '',
    d.macdSignal?.toFixed(6) ?? '',
    ((d.macd ?? 0) - (d.macdSignal ?? 0)).toFixed(6),
    d.bbUpper?.toFixed(4) ?? '',
    d.bbMiddle?.toFixed(4) ?? '',
    d.bbLower?.toFixed(4) ?? '',
  ]);

  const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${symbol}_${new Date().toISOString().slice(0, 10)}.csv`;
  link.click();
  URL.revokeObjectURL(url);
}

export function exportToJSON(symbol: string, data: ChartDataPoint[]): void {
  if (!data.length) return;
  const json = JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${symbol}_${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(url);
}
