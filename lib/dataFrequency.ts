// Data frequency (chart interval / lookback) presets shared by the dashboard.
// Extracted from app/page.tsx to keep that component focused on view logic.

export const DATA_FREQUENCY_OPTIONS = [
  {
    id: '5m' as const,
    label: '5m',
    interval: '5m',
    days: 25,
    description: '5-minute bars • ~1 month',
    category: 'intraday' as const,
  },
  {
    id: '15m' as const,
    label: '15m',
    interval: '15m',
    days: 60,
    description: '15-minute bars • last 3 months',
    category: 'intraday' as const,
  },
  {
    id: '1h' as const,
    label: '1H',
    interval: '60m',
    days: 365,
    description: 'Hourly bars • last year',
    category: 'intraday' as const,
  },
  {
    id: '1d' as const,
    label: '1D',
    interval: '1d',
    days: 1825,
    description: 'Daily bars • 5 years',
    category: 'session' as const,
  },
  {
    id: '1wk' as const,
    label: '1W',
    interval: '1wk',
    days: 1825,
    description: 'Weekly bars • 5 years',
    category: 'session' as const,
  },
  {
    id: '1mo' as const,
    label: '1M',
    interval: '1mo',
    days: 1825,
    description: 'Monthly bars • 5 years',
    category: 'session' as const,
  },
] as const;

export type DataFrequencyOption = typeof DATA_FREQUENCY_OPTIONS[number];
export type DataFrequencyId = DataFrequencyOption['id'];

export const DEFAULT_DATA_FREQUENCY_ID: DataFrequencyId = '1d';
export const DEFAULT_FREQUENCY_OPTION =
  DATA_FREQUENCY_OPTIONS.find(option => option.id === DEFAULT_DATA_FREQUENCY_ID)!;

export const getFrequencyOption = (id?: DataFrequencyId): DataFrequencyOption =>
  id
    ? DATA_FREQUENCY_OPTIONS.find(option => option.id === id) ?? DEFAULT_FREQUENCY_OPTION
    : DEFAULT_FREQUENCY_OPTION;
