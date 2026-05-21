'use client';

import React, { useState, useEffect, startTransition } from 'react';
import dynamic from 'next/dynamic';
import { Search, TrendingUp, Loader2, AlertCircle, Github, Clock, BarChart2, Brain, Sparkles, Download, Filter } from 'lucide-react';
import { calculateAllIndicators } from '@/lib/technicalIndicators';
import { generateForecast, getForecastInsights } from '@/lib/forecasting';
import { generateTradingSignal } from '@/lib/tradingSignals';
import { StockData, NewsArticle, ChartDataPoint, ChartPattern } from '@/types';
import { getCachedPredictions, savePredictionsToCache, CachedPrediction } from '@/lib/predictionsCache';
import { MLSettings, MLPreset, DEFAULT_ML_SETTINGS } from '@/types/mlSettings';
import { PatternSettings, PatternPreset, DEFAULT_PATTERN_SETTINGS } from '@/types/patternSettings';
import type { SearchHistoryItem } from '@/components/Sidebar';
import { exportToCSV } from '@/lib/exportData';

// Lazy load heavy components with dynamic imports
const LightweightChartWrapper = dynamic(() => import('@/components/LightweightChartWrapper'), { ssr: false });
const StockChart = dynamic(() => import('@/components/StockChart'), {
  loading: () => <div className="h-96 flex items-center justify-center" style={{ background: 'var(--bg-3)' }}>
    <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
  </div>,
  ssr: false
});

const TechnicalIndicatorsChart = dynamic(() => import('@/components/TechnicalIndicatorsChart'), {
  loading: () => <div className="h-64 flex items-center justify-center" style={{ background: 'var(--bg-3)' }}>
    <Loader2 className="w-6 h-6 animate-spin" style={{ color: 'var(--accent)' }} />
  </div>,
  ssr: false
});

const NewsPanel = dynamic(() => import('@/components/NewsPanel'), { ssr: false });
const TradingSignals = dynamic(() => import('@/components/TradingSignals'), { ssr: false });
const Sidebar = dynamic(() => import('@/components/Sidebar'), { ssr: false });
const CompanyInfo = dynamic(() => import('@/components/CompanyInfo'), { ssr: false });
const MLPredictions = dynamic(() => import('@/components/MLPredictions'), { ssr: false });
const PatternPanel = dynamic(() => import('@/components/PatternPanel'), { ssr: false });
const CorrelationHeatmap = dynamic(() => import('@/components/CorrelationHeatmap'), { ssr: false });
const RedditSentiment = dynamic(() => import('@/components/RedditSentiment'), { ssr: false });
const ApeWisdomMentions = dynamic(() => import('@/components/ApeWisdomMentions'), { ssr: false });
const InsiderTransactions = dynamic(() => import('@/components/InsiderTransactions'), { ssr: false });
const EarningsCalendar = dynamic(() => import('@/components/EarningsCalendar'), { ssr: false });
const AnalystRecommendations = dynamic(() => import('@/components/AnalystRecommendations'), { ssr: false });
const PeerStocks = dynamic(() => import('@/components/PeerStocks'), { ssr: false });
const AIAnalysis = dynamic(() => import('@/components/AIAnalysis'), { ssr: false });
const FinvizPanel = dynamic(() => import('@/components/FinvizPanel'), { ssr: false });
const LivePredictionChart = dynamic(() => import('@/components/LivePredictionChart'), { ssr: false });
const FearGreedIndex = dynamic(() => import('@/components/FearGreedIndex'), { ssr: false });
const MarketMovers = dynamic(() => import('@/components/MarketMovers'), { ssr: false });
const Watchlist = dynamic(() => import('@/components/Watchlist'), { ssr: false });
const OptionsChain = dynamic(() => import('@/components/OptionsChain'), { ssr: false });
const PriceAlerts = dynamic(() => import('@/components/PriceAlerts'), { ssr: false });
const StockScreener = dynamic(() => import('@/components/StockScreener'), { ssr: false });

// Lazy load heavy ML libraries only when needed
const loadMLLibraries = async () => {
  const [
    { generateMLForecast },
    { generateProphetWithChangepoints },
    { generateLinearRegression, generateEMAForecast, generateARIMAForecast, generateProphetLiteForecast },
    { generateGRUForecast, generateCNNLSTMForecast, generateEnsembleFromPredictions },
    { detectChartPatterns }
  ] = await Promise.all([
    import('@/lib/mlForecasting'),
    import('@/lib/prophetForecast'),
    import('@/lib/mlAlgorithms'),
    import('@/lib/advancedMLModels'),
    import('@/lib/chartPatterns')
  ]);

  return {
    generateMLForecast,
    generateProphetWithChangepoints,
    generateLinearRegression,
    generateEMAForecast,
    generateARIMAForecast,
    generateProphetLiteForecast,
    generateGRUForecast,
    generateCNNLSTMForecast,
    generateEnsembleFromPredictions,
    detectChartPatterns
  };
};

const DATA_FREQUENCY_OPTIONS = [
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

type DataFrequencyOption = typeof DATA_FREQUENCY_OPTIONS[number];
type DataFrequencyId = DataFrequencyOption['id'];

const DEFAULT_DATA_FREQUENCY_ID: DataFrequencyId = '1d';
const DEFAULT_FREQUENCY_OPTION =
  DATA_FREQUENCY_OPTIONS.find(option => option.id === DEFAULT_DATA_FREQUENCY_ID)!;

const getFrequencyOption = (id?: DataFrequencyId): DataFrequencyOption =>
  id ? DATA_FREQUENCY_OPTIONS.find(option => option.id === id) ?? DEFAULT_FREQUENCY_OPTION : DEFAULT_FREQUENCY_OPTION;

type FetchDataOptions = {
  forceRecalc?: boolean;
  skipMLCalculations?: boolean;
  frequencyId?: DataFrequencyId;
  chartOnly?: boolean;
};

export default function Home() {
  const [symbol, setSymbol] = useState('AAPL');
  const [inputSymbol, setInputSymbol] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [chartRefreshing, setChartRefreshing] = useState(false);
  const [error, setError] = useState('');

  const [stockData, setStockData] = useState<StockData[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [companyName, setCompanyName] = useState<string>('');
  const [marketState, setMarketState] = useState<string>('');
  const [companyInfo, setCompanyInfo] = useState<any>(null);
  const [fundamentalsData, setFundamentalsData] = useState<any>(null);
  const [fundamentalsLoading, setFundamentalsLoading] = useState(false);
  const [finvizStock, setFinvizStock] = useState<Record<string, string | null> | null>(null);
  const [finvizAnalystTargets, setFinvizAnalystTargets] = useState<Array<{ date: string; category: string; analyst: string; rating: string; target: string }> | null>(null);
  const [finvizCharts, setFinvizCharts] = useState<Record<string, string> | null>(null);
  const [finvizLinks, setFinvizLinks] = useState<Record<string, string>>({});
  const [showFinvizChart, setShowFinvizChart] = useState(false);
  const [activeFinvizChart, setActiveFinvizChart] = useState('dailyCandle');
  const sentimentByUrl = React.useRef<Map<string, any>>(new Map());
  const newsArticlesRef = React.useRef<NewsArticle[]>([]);
  const [finvizNewsLoading, setFinvizNewsLoading] = useState(false);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [chartPatterns, setChartPatterns] = useState<ChartPattern[]>([]);
  const [forecastData, setForecastData] = useState<any[]>([]);
  const [prophetForecastData, setProphetForecastData] = useState<any[]>([]);
  const [useProphetForecast, setUseProphetForecast] = useState(false);
  const [newsArticles, setNewsArticles] = useState<NewsArticle[]>([]);
  const [newsSentiments, setNewsSentiments] = useState<any[]>([]);
  const [isAnalyzingSentiment, setIsAnalyzingSentiment] = useState(false);
  const [tradingSignal, setTradingSignal] = useState<any>(null);
  const [forecastInsights, setForecastInsights] = useState<any>(null);
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([]);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [showScreener, setShowScreener] = useState(false);
  const historyRef = React.useRef<HTMLDivElement>(null);

  // Chart display options
  const [showMA20, setShowMA20] = useState(true);
  const [showMA50, setShowMA50] = useState(true);
  const [showBB, setShowBB] = useState(false);
  const [showIndicators, setShowIndicators] = useState(true);
  const [forecastHorizon, setForecastHorizon] = useState(30);
  const [chartType, setChartType] = useState<'line' | 'candlestick'>('candlestick');
  const [useLightweightChart, setUseLightweightChart] = useState(true);
  const [showVolume, setShowVolume] = useState(true);
  const [showPatterns, setShowPatterns] = useState(true);
  const [dataFrequencyId, setDataFrequencyId] = useState<DataFrequencyId>(DEFAULT_DATA_FREQUENCY_ID);
  const [dataInterval, setDataInterval] = useState<string>(DEFAULT_FREQUENCY_OPTION.interval);
  const [visibleDateRange, setVisibleDateRange] = useState<{ startDate: string; endDate: string } | null>(null);
  const currentFrequency = React.useMemo(
    () => getFrequencyOption(dataFrequencyId),
    [dataFrequencyId]
  );

  const handleChartTypeChange = (type: 'line' | 'candlestick') => {
    startTransition(() => setChartType(type));
  };

  // State for pattern detection loading
  const [patternDetecting, setPatternDetecting] = useState(false);

  // ML predictions state
  const [mlPredictions, setMlPredictions] = useState<{
    lstm?: any[];
    arima?: any[];
    prophetLite?: any[];
    gru?: any[];
    ensemble?: any[];
    cnnLstm?: any[];
    linearRegression?: any[];
    ema?: any[];
  }>({});
  const [mlTraining, setMlTraining] = useState(false);
  const [mlFromCache, setMlFromCache] = useState(false);
  const isLoadingFromCacheTable = React.useRef(false);
  const suggestionsDebounce = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  // ML Settings
  const [mlSettings, setMlSettings] = useState<MLSettings>(DEFAULT_ML_SETTINGS);
  const [mlPreset, setMlPreset] = useState<MLPreset>('balanced');

  // Pattern Settings
  const [patternSettings, setPatternSettings] = useState<PatternSettings>(DEFAULT_PATTERN_SETTINGS);
  const [patternPreset, setPatternPreset] = useState<PatternPreset>('balanced');

  // Memoize ML settings callbacks to prevent unnecessary re-renders
  const handleMlSettingsChange = React.useCallback((newSettings: MLSettings) => {
    setMlSettings(newSettings);
  }, []);

  const handleMlPresetChange = React.useCallback((newPreset: MLPreset) => {
    setMlPreset(newPreset);
  }, []);

  // Memoize Pattern settings callbacks to prevent unnecessary re-renders
  const handlePatternSettingsChange = React.useCallback((newSettings: PatternSettings) => {
    setPatternSettings(newSettings);
  }, []);

  const handlePatternPresetChange = React.useCallback((newPreset: PatternPreset) => {
    setPatternPreset(newPreset);
  }, []);

  // Function to manually detect patterns
  const detectPatterns = React.useCallback(() => {
    if (!chartData.length) {
      setChartPatterns([]);
      return;
    }

    setPatternDetecting(true);
    loadMLLibraries().then(({ detectChartPatterns }) => {
      // Use requestIdleCallback to run pattern detection when browser is idle
      if (typeof requestIdleCallback !== 'undefined') {
        requestIdleCallback(() => {
          const patterns = detectChartPatterns(chartData, patternSettings);
          setChartPatterns(patterns);
          setPatternDetecting(false);
        }, { timeout: 2000 });
      } else {
        // Fallback for browsers without requestIdleCallback
        setTimeout(() => {
          const patterns = detectChartPatterns(chartData, patternSettings);
          setChartPatterns(patterns);
          setPatternDetecting(false);
        }, 0);
      }
    }).catch(error => {
      console.error('[ERROR] Pattern detection error:', error);
      setPatternDetecting(false);
    });
  }, [chartData, patternSettings]);

  // Auto-detect patterns when data or settings change
  useEffect(() => {
    // Only detect patterns when the toggle is enabled
    if (!showPatterns || !chartData.length) {
      setChartPatterns([]);
      return;
    }

    // Debounce pattern detection
    const timer = setTimeout(() => {
      detectPatterns();
    }, 300);

    return () => clearTimeout(timer);
  }, [chartData, showPatterns, patternSettings, detectPatterns]);

  // Load search history and ML settings from localStorage on mount
  useEffect(() => {
    // Initialize TensorFlow.js with WebGPU for better performance
    const initTF = async () => {
      try {
        const { initializeTensorFlow, warmupGPU } = await import('@/lib/tfConfig');
        await initializeTensorFlow();
        await warmupGPU();
      } catch (error) {
        console.error('Failed to initialize TensorFlow.js:', error);
      }
    };
    initTF();

    try {
      const savedHistory = localStorage.getItem('stockSearchHistory');
      if (savedHistory) {
        setSearchHistory(JSON.parse(savedHistory));
      }

      // Load ML settings
      const savedMLSettings = localStorage.getItem('mlSettings');
      const savedMLPreset = localStorage.getItem('mlPreset');
      if (savedMLSettings) {
        setMlSettings(JSON.parse(savedMLSettings));
      }
      if (savedMLPreset) {
        setMlPreset(savedMLPreset as MLPreset);
      }

      // Load Pattern settings
      const savedPatternSettings = localStorage.getItem('patternSettings');
      const savedPatternPreset = localStorage.getItem('patternPreset');
      if (savedPatternSettings) {
        setPatternSettings(JSON.parse(savedPatternSettings));
      }
      if (savedPatternPreset) {
        setPatternPreset(savedPatternPreset as PatternPreset);
      }
    } catch (e) {
      console.error('Failed to load from localStorage:', e);
    }
  }, []);

  // Save ML settings to localStorage when they change
  useEffect(() => {
    try {
      localStorage.setItem('mlSettings', JSON.stringify(mlSettings));
      localStorage.setItem('mlPreset', mlPreset);
    } catch (error) {
      console.error('Error saving ML settings:', error);
    }
  }, [mlSettings, mlPreset]);

  // Save Pattern settings to localStorage when they change
  useEffect(() => {
    try {
      localStorage.setItem('patternSettings', JSON.stringify(patternSettings));
      localStorage.setItem('patternPreset', patternPreset);
    } catch (error) {
      console.error('Error saving Pattern settings:', error);
    }
  }, [patternSettings, patternPreset]);

  // Add to search history
  const addToHistory = (stockSymbol: string, price?: number, companyName?: string) => {
    const newItem: SearchHistoryItem = {
      symbol: stockSymbol,
      timestamp: Date.now(),
      price,
      companyName
    };

    // Use functional update to ensure we have the latest state
    setSearchHistory(prevHistory => {
      // Remove duplicates and add to top
      const filteredHistory = prevHistory.filter(item => item.symbol !== stockSymbol);
      const newHistory = [newItem, ...filteredHistory].slice(0, 20); // Keep last 20

      // Save to localStorage
      try {
        localStorage.setItem('stockSearchHistory', JSON.stringify(newHistory));
      } catch (e) {
        console.error('Failed to save search history to localStorage:', e);
      }

      return newHistory;
    });
  };

  // Clear search history
  const clearHistory = () => {
    try {
      localStorage.removeItem('stockSearchHistory');
      setSearchHistory([]);
    } catch (e) {
      console.error('Failed to clear search history from localStorage:', e);
    }
  };

  // Determine market status in Eastern Time (ET). Uses Intl timeZone when available.
  const getMarketStatus = () => {
    const now = new Date();
    try {
      const parts = new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York',
        hour12: false,
        weekday: 'short',
        hour: '2-digit',
        minute: '2-digit',
      }).formatToParts(now);

      const map: Record<string, string> = {};
      for (const p of parts) if (p.type && p.value) map[p.type] = p.value;

      const hour = parseInt(map.hour || '0', 10);
      const minute = parseInt(map.minute || '0', 10);
      const weekdayShort = map.weekday || '';

      const weekdayIndex = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].indexOf(weekdayShort);
      const currentMinutes = hour * 60 + minute;
      const openMinutes = 9 * 60 + 30;
      const closeMinutes = 16 * 60;

      const status = (weekdayIndex === 0 || weekdayIndex === 6)
        ? 'CLOSED'
        : (currentMinutes < openMinutes ? 'PRE' : currentMinutes >= closeMinutes ? 'POST' : 'REGULAR');

      return status;
    } catch (e) {
      const monthIdx = now.getMonth();
      const isDSTInNY = monthIdx >= 2 && monthIdx <= 10;
      const etOffset = isDSTInNY ? -4 : -5;
      const etTime = new Date(now.getTime() + etOffset * 60 * 60 * 1000);
      const dayIdx = etTime.getDay();
      const hour = etTime.getHours();
      const minute = etTime.getMinutes();

      if (dayIdx === 0 || dayIdx === 6) return 'CLOSED';
      const currentMinutes = hour * 60 + minute;
      if (currentMinutes < 9 * 60 + 30) return 'PRE';
      if (currentMinutes >= 16 * 60) return 'POST';
      return 'REGULAR';
    }
  };

  // Shared ML training runner — used by fetchData and the forecastHorizon effect.
  // Trains all models sequentially (to avoid browser freeze), updating predictions
  // incrementally, then saves the full set to cache.
  const runMLTraining = (
    data: StockData[],
    sym: string,
    horizon: number,
    settings: MLSettings,
    delay: number
  ) => {
    setMlFromCache(false);
    setMlTraining(true);

    setTimeout(async () => {
      try {
        const mlLibs = await loadMLLibraries();

        // Fast algorithms first — display immediately
        const basePredictions = {
          linearRegression: mlLibs.generateLinearRegression(data, horizon),
          ema: mlLibs.generateEMAForecast(data, horizon),
          arima: mlLibs.generateARIMAForecast(data, horizon),
          prophetLite: mlLibs.generateProphetLiteForecast(data, horizon),
        };
        setMlPredictions(basePredictions);

        // Neural networks trained one at a time to keep the UI responsive
        let lstm = null;
        try {
          lstm = await mlLibs.generateMLForecast(data, horizon, settings);
          setMlPredictions(prev => ({ ...prev, lstm }));
        } catch (err) { console.error('LSTM failed:', err); }

        let gru = null;
        try {
          gru = await mlLibs.generateGRUForecast(data, horizon, settings);
          setMlPredictions(prev => ({ ...prev, gru }));
        } catch (err) { console.error('GRU failed:', err); }

        let cnnLstm = null;
        try {
          cnnLstm = await mlLibs.generateCNNLSTMForecast(data, horizon, settings);
          setMlPredictions(prev => ({ ...prev, cnnLstm }));
        } catch (err) { console.error('CNN-LSTM failed:', err); }

        const ensemble = mlLibs.generateEnsembleFromPredictions({ gru, cnnLstm, lstm }, horizon);
        if (ensemble) setMlPredictions(prev => ({ ...prev, ensemble }));

        savePredictionsToCache(sym, {
          ...basePredictions,
          ...(lstm && { lstm }),
          ...(gru && { gru }),
          ...(cnnLstm && { cnnLstm }),
          ...(ensemble && { ensemble }),
        }, horizon);

        setMlTraining(false);
      } catch (mlError) {
        console.error('ML algorithms error:', mlError);
        setMlTraining(false);
      }
    }, delay);
  };

  const fetchData = async (
    stockSymbol: string,
    options: FetchDataOptions = {}
  ) => {
    const { forceRecalc = false, skipMLCalculations = false, frequencyId, chartOnly = false } = options;
    const targetFrequency = getFrequencyOption(frequencyId ?? dataFrequencyId);
    const intervalParam = targetFrequency.interval;
    const rangeDays = targetFrequency.days;

    // Fast path: frequency change only — refetch OHLCV and recalculate chart data without
    // resetting news, ML predictions, Finviz, or company info.
    if (chartOnly) {
      setChartRefreshing(true);
      try {
        const params = new URLSearchParams({ symbol: stockSymbol, days: String(rangeDays), interval: intervalParam });
        const stockResponse = await fetch(`/api/stock?${params.toString()}`);
        if (!stockResponse.ok) throw new Error('Failed to fetch stock data');
        const stockResult = await stockResponse.json();
        if (stockResult.error) throw new Error(stockResult.error);

        setStockData(stockResult.data);
        const price = stockResult.currentPrice || stockResult.data[stockResult.data.length - 1].close;
        setCurrentPrice(price);
        setDataInterval(stockResult.interval || intervalParam);

        const indicators = calculateAllIndicators(stockResult.data);
        const preparedChartData: ChartDataPoint[] = stockResult.data.map((d: StockData, i: number) => ({
          date: d.date, open: d.open, high: d.high, low: d.low, close: d.close, volume: d.volume,
          ma20: indicators.ma20[i], ma50: indicators.ma50[i], ma200: indicators.ma200[i],
          bbUpper: indicators.bbUpper[i], bbMiddle: indicators.bbMiddle[i], bbLower: indicators.bbLower[i],
          rsi: indicators.rsi[i], macd: indicators.macd[i], macdSignal: indicators.macdSignal[i],
        }));
        setChartData(preparedChartData);

        setTimeout(async () => {
          try {
            const simpleForecast = generateForecast(stockResult.data, forecastHorizon);
            const simpleInsights = getForecastInsights(price, simpleForecast);
            setForecastData(simpleForecast);
            setForecastInsights(simpleInsights);
            try {
              const { generateProphetWithChangepoints } = await loadMLLibraries();
              setProphetForecastData(generateProphetWithChangepoints(stockResult.data, forecastHorizon, 5, mlSettings));
            } catch {}
            setTradingSignal(generateTradingSignal(stockResult.data, indicators, simpleInsights?.mediumTerm.change));
          } catch {}
        }, 100);
      } catch (err: any) {
        setError(err.message || 'Failed to refresh chart');
      } finally {
        setChartRefreshing(false);
      }
      return;
    }

    setLoading(true);
    setError('');
    setNewsArticles([]);
    setNewsSentiments([]);
    setIsAnalyzingSentiment(false);
    setMlPredictions({});
    setMlTraining(false);
    setFinvizStock(null);
    setFinvizAnalystTargets(null);
    setFinvizCharts(null);
    setShowFinvizChart(false);
    sentimentByUrl.current.clear();
    newsArticlesRef.current = [];
    setFinvizNewsLoading(true);

    try {
      // Fetch stock data for the selected frequency/range
      const params = new URLSearchParams({
        symbol: stockSymbol,
        days: String(rangeDays),
        interval: intervalParam,
      });
      const stockResponse = await fetch(`/api/stock?${params.toString()}`);
      if (!stockResponse.ok) {
        throw new Error('Failed to fetch stock data');
      }
      const stockResult = await stockResponse.json();

      if (stockResult.error) {
        throw new Error(stockResult.error);
      }

  setStockData(stockResult.data);
      const price = stockResult.currentPrice || stockResult.data[stockResult.data.length - 1].close;
      setCurrentPrice(price);
      setCompanyName(stockResult.companyName || stockSymbol);
  // Use client-side ET calculation to determine market status immediately
  setMarketState(getMarketStatus());
      setDataInterval(stockResult.interval || intervalParam);

      // Fetch detailed company info from Massive API
      let massiveCompanyInfo: any = {};
      try {
        const companyResponse = await fetch(`/api/company?symbol=${stockSymbol}`);
        if (companyResponse.ok) {
          const companyResult = await companyResponse.json();
          if (companyResult.success && companyResult.companyInfo) {
            massiveCompanyInfo = companyResult.companyInfo;
            // Update company name from Massive if available
            if (companyResult.companyInfo.name) {
              setCompanyName(companyResult.companyInfo.name);
            }
          }
        }
      } catch (companyError) {
        console.error('Error fetching Massive company info:', companyError);
        // Continue with stock data even if company info fails
      }

      // Merge Yahoo Finance data with Massive API data
      setCompanyInfo({
        ...(stockResult.companyInfo || {}),
        ...massiveCompanyInfo,
        change: typeof stockResult.change !== 'undefined' ? stockResult.change : null,
        changePercent: typeof stockResult.changePercent !== 'undefined' ? stockResult.changePercent : null,
      });

      // Fetch fundamentals data from Alpha Vantage (cached for 24 hours)
      const cachedFundamentals = localStorage.getItem(`fundamentals_${stockSymbol}`);
      const cachedTime = localStorage.getItem(`fundamentals_time_${stockSymbol}`);
      
      if (cachedFundamentals && cachedTime) {
        const age = Date.now() - parseInt(cachedTime);
        if (age < 24 * 60 * 60 * 1000) { // 24 hours
          setFundamentalsData(JSON.parse(cachedFundamentals));
        } else {
          // Cache expired, fetch new data
          fetchFundamentals(stockSymbol);
        }
      } else {
        // No cache, fetch new data
        fetchFundamentals(stockSymbol);
      }

      // Add to search history
      const nameForHistory = massiveCompanyInfo?.name || stockResult.companyInfo?.name || stockResult.symbol;
      addToHistory(stockSymbol, price, nameForHistory);

      // Calculate technical indicators
      const indicators = calculateAllIndicators(stockResult.data);

      // Prepare chart data
      const preparedChartData: ChartDataPoint[] = stockResult.data.map((d: StockData, i: number) => ({
        date: d.date,
        open: d.open,
        high: d.high,
        low: d.low,
        close: d.close,
        volume: d.volume,
        ma20: indicators.ma20[i],
        ma50: indicators.ma50[i],
        ma200: indicators.ma200[i],
        bbUpper: indicators.bbUpper[i],
        bbMiddle: indicators.bbMiddle[i],
        bbLower: indicators.bbLower[i],
        rsi: indicators.rsi[i],
        macd: indicators.macd[i],
        macdSignal: indicators.macdSignal[i],
      }));

      setChartData(preparedChartData);

      setSymbol(stockSymbol);

      // Generate forecasts in background (non-blocking)
      setTimeout(async () => {
        try {
          // Generate custom forecast
          const simpleForecast = generateForecast(stockResult.data, forecastHorizon);
          const simpleInsights = getForecastInsights(
            stockResult.currentPrice || stockResult.data[stockResult.data.length - 1].close,
            simpleForecast
          );

          setForecastData(simpleForecast);
          setForecastInsights(simpleInsights);

          // Lazy load and generate Prophet forecast
          try {
            const { generateProphetWithChangepoints } = await loadMLLibraries();
            const prophetForecast = generateProphetWithChangepoints(stockResult.data, forecastHorizon, 5, mlSettings);
            setProphetForecastData(prophetForecast);
          } catch (prophetError) {
            console.error('Prophet forecast error:', prophetError);
          }

          // Generate trading signal with simple forecast
          const signal = generateTradingSignal(
            stockResult.data,
            indicators,
            simpleInsights?.mediumTerm.change
          );
          setTradingSignal(signal);
        } catch (forecastError) {
          console.error('Forecast error:', forecastError);
          // Generate trading signal without forecast
          const signal = generateTradingSignal(stockResult.data, indicators);
          setTradingSignal(signal);
        }
      }, 100);

      if (skipMLCalculations) return;

      // Check cache first unless force recalculate is true
      const cached = !forceRecalc ? getCachedPredictions(stockSymbol, forecastHorizon) : null;

      if (cached && !forceRecalc) {
        setMlPredictions(cached.predictions);
        setMlFromCache(true);
        setMlTraining(false);
      } else {
        runMLTraining(stockResult.data, stockSymbol, forecastHorizon, mlSettings, 2000);
      }

      // Fetch news asynchronously (fast, no sentiment)
      setTimeout(async () => {
        try {
          const newsResponse = await fetch(`/api/news?symbol=${stockSymbol}`);
          if (newsResponse.ok) {
            const newsResult = await newsResponse.json();
            const articles = newsResult.articles || [];
            newsArticlesRef.current = articles;
            setNewsArticles(articles);

            const neutral = { sentiment: 'neutral' as const, score: 0, confidence: 0 };
            setNewsSentiments(articles.map(() => neutral));

            // Analyze sentiment in background
            setTimeout(async () => {
              try {
                setIsAnalyzingSentiment(true);
                const sentimentResponse = await fetch('/api/sentiment', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ articles }),
                });

                if (sentimentResponse.ok) {
                  const sentimentResult = await sentimentResponse.json();
                  const sentiments = sentimentResult.sentiments || articles.map(() => neutral);
                  sentiments.forEach((s: any, i: number) => {
                    if (articles[i]) sentimentByUrl.current.set(articles[i].url, s);
                  });
                  // Rebuild aligned to current article order (may include finviz articles by now)
                  const current = newsArticlesRef.current;
                  setNewsSentiments(current.map(a => sentimentByUrl.current.get(a.url) || neutral));
                }
              } catch (sentimentError) {
                console.error('Error analyzing sentiment:', sentimentError);
              } finally {
                setIsAnalyzingSentiment(false);
              }
            }, 1000);
          }
        } catch (newsError) {
          console.error('Error fetching news:', newsError);
        }
      }, 100);

    } catch (err: any) {
      setError(err.message || 'Failed to load data');
      console.error('Error fetching data:', err);
    } finally {
      setLoading(false);
    }
  };

  // Load initial data
  useEffect(() => {
    fetchData(symbol);
  }, []);

  // Update forecast when horizon changes
  useEffect(() => {
    if (stockData.length > 0 && !isLoadingFromCacheTable.current) {
      const updateForecast = async () => {
        try {
          // Generate custom forecast
          const simpleForecast = generateForecast(stockData, forecastHorizon);
          const simpleInsights = getForecastInsights(currentPrice, simpleForecast);

          setForecastData(simpleForecast);
          setForecastInsights(simpleInsights);

          // Lazy load and generate Prophet forecast
          try {
            const { generateProphetWithChangepoints } = await loadMLLibraries();
            const prophetForecast = generateProphetWithChangepoints(stockData, forecastHorizon, 5, mlSettings);
            setProphetForecastData(prophetForecast);
          } catch (prophetError) {
            console.error('Prophet forecast error:', prophetError);
          }

          const indicators = calculateAllIndicators(stockData);
          const simpleSignal = generateTradingSignal(
            stockData,
            indicators,
            simpleInsights?.mediumTerm.change
          );
          setTradingSignal(simpleSignal);

          // Check cache for this forecast horizon
          const cached = getCachedPredictions(symbol, forecastHorizon);

          if (cached) {
            setMlPredictions(cached.predictions);
            setMlFromCache(true);
            setMlTraining(false);
          } else {
            runMLTraining(stockData, symbol, forecastHorizon, mlSettings, 1500);
          }
        } catch (forecastError) {
          console.error('Forecast update error:', forecastError);
          setMlTraining(false);
        }
      };

      updateForecast();
    }
  }, [forecastHorizon, mlSettings]);

  // Fetch fundamentals data
  const fetchFundamentals = async (stockSymbol: string) => {
    setFundamentalsLoading(true);
    try {
      const response = await fetch(`/api/fundamentals?symbol=${stockSymbol}`);
      if (response.ok) {
        const data = await response.json();
        setFundamentalsData(data);
        
        // Cache for 24 hours
        localStorage.setItem(`fundamentals_${stockSymbol}`, JSON.stringify(data));
        localStorage.setItem(`fundamentals_time_${stockSymbol}`, Date.now().toString());
      } else {
        const error = await response.json();
        console.error('Fundamentals fetch error:', error);
        setFundamentalsData(null);
      }
    } catch (error) {
      console.error('Error fetching fundamentals:', error);
      setFundamentalsData(null);
    } finally {
      setFundamentalsLoading(false);
    }
  };

  const handleFinvizNews = React.useCallback((finvizNews: Array<{ timestamp: string; headline: string; url: string; source: string }>) => {
    const parseTs = (ts: string): string => {
      const now = new Date();
      if (/^today/i.test(ts)) {
        const m = (ts.split(' ')[1] || '').match(/^(\d{1,2}):(\d{2})(AM|PM)$/i);
        if (m) {
          let h = parseInt(m[1]);
          if (m[3].toUpperCase() === 'PM' && h !== 12) h += 12;
          if (m[3].toUpperCase() === 'AM' && h === 12) h = 0;
          return new Date(now.getFullYear(), now.getMonth(), now.getDate(), h, parseInt(m[2])).toISOString();
        }
        return now.toISOString();
      }
      const m = ts.match(/^([A-Za-z]{3})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})(AM|PM)$/i);
      if (m) {
        const mo: Record<string, number> = { jan:0,feb:1,mar:2,apr:3,may:4,jun:5,jul:6,aug:7,sep:8,oct:9,nov:10,dec:11 };
        let h = parseInt(m[4]);
        if (m[6].toUpperCase() === 'PM' && h !== 12) h += 12;
        if (m[6].toUpperCase() === 'AM' && h === 12) h = 0;
        return new Date(2000 + parseInt(m[3]), mo[m[1].toLowerCase()] ?? 0, parseInt(m[2]), h, parseInt(m[5])).toISOString();
      }
      const p = new Date(ts);
      return isNaN(p.getTime()) ? now.toISOString() : p.toISOString();
    };

    const neutral = { sentiment: 'neutral' as const, score: 0, confidence: 0 };
    const converted = finvizNews.map(item => ({
      title: item.headline,
      description: '',
      url: item.url,
      publishedAt: parseTs(item.timestamp),
      source: item.source || 'Finviz',
    }));

    setFinvizNewsLoading(false);
    const existingUrls = new Set(newsArticlesRef.current.map(a => a.url));
    const newOnes = converted.filter(a => !existingUrls.has(a.url));
    if (!newOnes.length) return;

    const merged = [...newsArticlesRef.current, ...newOnes].sort(
      (a, b) => new Date(b.publishedAt).getTime() - new Date(a.publishedAt).getTime()
    );
    newsArticlesRef.current = merged;
    setNewsArticles(merged);
    setNewsSentiments(merged.map(a => sentimentByUrl.current.get(a.url) || neutral));

    // Run sentiment on the new Finviz articles
    const toAnalyze = newOnes.filter(a => !sentimentByUrl.current.has(a.url));
    if (!toAnalyze.length) return;

    setIsAnalyzingSentiment(true);
    fetch('/api/sentiment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ articles: toAnalyze }),
    })
      .then(r => r.json())
      .then(result => {
        (result.sentiments || []).forEach((s: any, i: number) => {
          if (toAnalyze[i]) sentimentByUrl.current.set(toAnalyze[i].url, s);
        });
        setNewsSentiments(newsArticlesRef.current.map(a => sentimentByUrl.current.get(a.url) || neutral));
      })
      .catch(console.error)
      .finally(() => setIsAnalyzingSentiment(false));
  }, []);

  const handleSearch = () => {
    if (inputSymbol.trim()) {
      fetchData(inputSymbol.trim().toUpperCase());
    }
  };

  const getTickerFromAPi = async (e: string) => {
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(e)}`);
      const data = await response.json();
      const suggestions = data.quotes.map((item: any) => `${item.symbol} , ${item.shortname}`);
      setSuggestions(suggestions);
    } catch (error) {
      console.error('Error fetching tickers:', error);
      setSuggestions([]);
    }
  };

  const handleLoadCachedPrediction = async (cachedPred: CachedPrediction) => {
    // Set flag to prevent useEffect from interfering
    isLoadingFromCacheTable.current = true;

    try {
      // Load the stock data for this symbol if different (skip ML calculations since we'll load from cache)
      if (cachedPred.symbol !== symbol) {
        await fetchData(cachedPred.symbol, { skipMLCalculations: true }); // Skip ML, just load data
      }

      // Update forecast horizon if different
      if (cachedPred.forecastHorizon !== forecastHorizon) {
        setForecastHorizon(cachedPred.forecastHorizon);
      }

      // Load the cached predictions
      setMlPredictions(cachedPred.predictions);
      setMlFromCache(true);
      setMlTraining(false);

    } finally {
      // Reset flag after a short delay to allow state updates to complete
      setTimeout(() => {
        isLoadingFromCacheTable.current = false;
      }, 500);
    }
  };

  const handleFrequencyChange = (nextId: DataFrequencyId) => {
    if (nextId === dataFrequencyId && stockData.length > 0 && !error) return;
    setDataFrequencyId(nextId);
    fetchData(symbol, { chartOnly: true, frequencyId: nextId });
  };

  const handleVisibleRangeChange = React.useCallback((startDate: string, endDate: string) => {
    // Only update visible range if patterns are enabled
    if (showPatterns) {
      setVisibleDateRange({ startDate, endDate });
    }
  }, [showPatterns]);

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  // Close history dropdown when clicking outside
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (historyRef.current && !historyRef.current.contains(e.target as Node)) {
        setShowHistory(false);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  // Update market status on interval
  useEffect(() => {
    const interval = setInterval(() => {
      setMarketState(getMarketStatus());
    }, 60000); // Update every minute

    // Initial set
    setMarketState(getMarketStatus());

    return () => clearInterval(interval);
  }, []);

  // Live price polling — every 15s during market hours, every 60s otherwise
  useEffect(() => {
    if (!symbol) return;

    const poll = async () => {
      try {
        const res = await fetch(`/api/price?symbol=${encodeURIComponent(symbol)}`);
        if (!res.ok) return;
        const data = await res.json();
        if (typeof data.price === 'number') {
          setCurrentPrice(data.price);
          setCompanyInfo((prev: any) => prev ? { ...prev, change: data.change, changePercent: data.changePercent } : prev);
          if (data.marketState) setMarketState(data.marketState);
        }
      } catch { /* silently ignore poll errors */ }
    };

    poll(); // immediate first poll
    const ms = getMarketStatus() === 'REGULAR' ? 15000 : 60000;
    const id = setInterval(poll, ms);
    return () => clearInterval(id);
  }, [symbol]);

  // Helper to show ET clock next to the market status for easier debugging/visibility
  const getETTimeString = () => {
    try {
      return new Intl.DateTimeFormat('en-US', {
        timeZone: 'America/New_York',
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
      }).format(new Date());
    } catch (e) {
      const now = new Date();
      const month = now.getMonth();
      const isDSTInNY = month >= 2 && month <= 10;
      const etOffset = isDSTInNY ? -4 : -5;
      const et = new Date(now.getTime() + etOffset * 60 * 60 * 1000);
      const hh = String(et.getHours()).padStart(2, '0');
      const mm = String(et.getMinutes()).padStart(2, '0');
      return `${hh}:${mm}`;
    }
  };

  return (
    <main className="min-h-screen p-2 sm:p-4" style={{ background: 'var(--bg-4)' }}>
      <div className="flex gap-4 items-start">
        {/* Main Content Area - Now Full Width */}
        <div className="flex-1 min-w-0">
          {/* Header with Search */}
          <div className="card mb-4" style={{ position: 'relative', zIndex: 20 }}>
            <span className="card-label">Stock Predictor</span>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-6 h-6" style={{ color: 'var(--accent)' }} />
              <h1 className="text-lg font-bold hidden sm:block" style={{ color: 'var(--text-1)' }}>
                Stock Predictor
              </h1>
            </div>
            <a
              href="https://github.com/sankeer28/stock-predictor"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 transition-colors px-3 py-1 border"
              style={{
                color: 'var(--text-3)',
                borderColor: 'var(--bg-1)',
                background: 'var(--bg-2)'
              }}
            >
              <Github className="w-4 h-4" />
              <span className="hidden sm:inline text-sm">GitHub</span>
            </a>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3">
              {/* Search Input */}
              <div className="flex gap-2 items-center flex-1 min-w-0">
                <div className="relative flex-1 min-w-0">
                  <input
                    type="text"
                    value={inputSymbol}
                    onChange={(e) => {
                      const value = e.target.value.toUpperCase();
                      setInputSymbol(value);
                      if (suggestionsDebounce.current) clearTimeout(suggestionsDebounce.current);
                      if (value) {
                        suggestionsDebounce.current = setTimeout(() => getTickerFromAPi(value), 300);
                      } else {
                        setSuggestions([]);
                      }
                    }}
                    onKeyPress={handleKeyPress}
                    placeholder="Symbol (e.g., AAPL)"
                    className="w-full px-4 py-3 border transition-all font-mono"
                    style={{
                      background: 'var(--bg-3)',
                      borderColor: 'var(--bg-1)',
                      borderLeftColor: 'var(--accent)',
                      borderLeftWidth: '3px',
                      color: 'var(--text-2)',
                      outline: 'none'
                    }}
                  />
                  {suggestions.length > 0 && (
                    <div
                      className="absolute top-full left-0 right-0 z-50 max-h-40 overflow-y-auto border-2"
                      style={{
                        background: 'var(--bg-2)',
                        borderColor: 'var(--bg-1)',
                      }}
                    >
                      {suggestions.map((suggestion, index) => (
                        <button
                          key={index}
                          onClick={() => {
                            const symbol = suggestion.split(' ,')[0];
                            setInputSymbol(symbol);
                            setSuggestions([]);
                            fetchData(symbol);
                          }}
                          className="w-full text-left px-4 py-2 hover:opacity-80 transition-all"
                          style={{ color: 'var(--text-2)' }}
                        >
                          {suggestion}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="px-6 py-3 font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 border"
                  style={{
                    background: 'var(--accent)',
                    borderColor: 'var(--accent)',
                    color: 'var(--text-0'
                  }}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Loading...
                    </>
                  ) : (
                    <>
                      <Search className="w-5 h-5" />
                      Analyze
                    </>
                  )}
                </button>

              </div>

              {/* Utility row: history, screener, market status */}
              <div className="flex items-center gap-2">
                {/* Search History Dropdown */}
                {searchHistory.length > 0 && (
                  <div className="relative" ref={historyRef}>
                    <button
                      onClick={() => setShowHistory(prev => !prev)}
                      className="px-3 py-2.5 font-medium transition-all cursor-pointer flex items-center gap-2 border"
                      style={{
                        background: 'var(--bg-3)',
                        borderColor: 'var(--bg-1)',
                        color: 'var(--text-3)'
                      }}
                    >
                      <Clock className="w-4 h-4" />
                      <span className="hidden sm:inline text-sm">History ({searchHistory.length})</span>
                      <span className="sm:hidden text-xs">{searchHistory.length}</span>
                    </button>
                    {showHistory && (
                      <div className="absolute left-0 sm:right-0 sm:left-auto mt-2 w-64 border-2 shadow-lg z-50 max-h-80 overflow-y-auto"
                        style={{
                          background: 'var(--bg-2)',
                          borderColor: 'var(--bg-1)'
                        }}>
                        <div className="p-3 border-b flex items-center justify-between" style={{ borderColor: 'var(--bg-1)' }}>
                          <span className="text-xs font-semibold" style={{ color: 'var(--text-4)' }}>Recent Searches</span>
                          <button
                            onClick={clearHistory}
                            className="text-xs px-2 py-1 border transition-all"
                            style={{
                              background: 'var(--bg-3)',
                              borderColor: 'var(--bg-1)',
                              color: 'var(--danger)'
                            }}
                          >
                            Clear All
                          </button>
                        </div>
                        {searchHistory.map((item, index) => (
                          <button
                            key={index}
                            onClick={() => {
                              setInputSymbol(item.symbol);
                              setShowHistory(false);
                              fetchData(item.symbol);
                            }}
                            className="w-full text-left px-3 py-2 transition-all border-b hover:opacity-80"
                            style={{
                              borderColor: 'var(--bg-1)',
                              background: item.symbol === symbol ? 'var(--bg-3)' : 'transparent'
                            }}
                          >
                            <div className="flex items-center justify-between">
                              <div>
                                <div className="font-mono font-bold text-sm" style={{ color: 'var(--text-2)' }}>
                                  {item.symbol}
                                </div>
                                {item.companyName && (
                                  <div className="text-xs truncate" style={{ color: 'var(--text-4)' }}>
                                    {item.companyName}
                                  </div>
                                )}
                              </div>
                              <div className="text-xs" style={{ color: 'var(--text-5)' }}>
                                {new Date(item.timestamp).toLocaleDateString()}
                              </div>
                            </div>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                )}

                {/* Screener Button */}
                <button
                  onClick={() => setShowScreener(prev => !prev)}
                  className="px-3 py-2.5 font-medium transition-all cursor-pointer flex items-center gap-2 border"
                  style={{
                    background: showScreener ? 'var(--purple-1)' : 'var(--bg-3)',
                    borderColor: showScreener ? 'var(--purple-1)' : 'var(--bg-1)',
                    color: showScreener ? 'var(--text-0)' : 'var(--text-3)',
                  }}
                >
                  <Filter className="w-4 h-4" />
                  <span className="hidden sm:inline text-sm">Screener</span>
                </button>

                {/* Market Status */}
                {!loading && stockData.length > 0 && (
                  <div className="flex items-center gap-1.5 px-2 py-2 border-2 ml-auto" style={{
                    borderColor: marketState === 'REGULAR' ? 'var(--success)' : 'var(--text-4)',
                    background: 'var(--bg-3)',
                  }}>
                    <div className={`w-2 h-2 rounded-full flex-shrink-0 ${marketState === 'REGULAR' ? 'animate-pulse' : ''}`} style={{
                      background: marketState === 'REGULAR' ? 'var(--success)' : 'var(--text-4)'
                    }} />
                    <span className="text-xs font-semibold" style={{
                      color: marketState === 'REGULAR' ? 'var(--success)' : 'var(--text-4)'
                    }}>
                      {marketState === 'REGULAR' ? 'OPEN' : marketState === 'CLOSED' ? 'CLOSED' : marketState === 'PRE' ? 'PRE' : 'POST'}
                    </span>
                    <span className="text-xs hidden sm:inline" style={{ color: 'var(--text-4)' }}>{getETTimeString()} ET</span>
                  </div>
                )}
              </div>
            </div>
          </div>

        {/* Stock Screener Panel */}
        {showScreener && (
          <div className="mb-6">
            <StockScreener
              onSelectTicker={(ticker) => {
                setInputSymbol(ticker);
                setShowScreener(false);
                fetchData(ticker);
              }}
            />
          </div>
        )}

        {error && (
          <div className="mb-6 p-4 border-2 flex items-start gap-3" style={{
            background: 'var(--bg-2)',
            borderColor: 'var(--danger)',
            borderLeftWidth: '3px'
          }}>
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" style={{ color: 'var(--danger)' }} />
            <div>
              <h3 className="font-semibold" style={{ color: 'var(--danger)' }}>Error</h3>
              <p className="text-sm" style={{ color: 'var(--text-3)' }}>{error}</p>
            </div>
          </div>
        )}

        {loading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-12 h-12 animate-spin" style={{ color: 'var(--accent)' }} />
          </div>
        )}

        {!loading && stockData.length > 0 && (
          <>
            {/* Company Overview */}
            {companyInfo && (
              <div className="mb-6">
                <CompanyInfo
                  symbol={symbol}
                  companyName={companyName}
                  currentPrice={currentPrice}
                  // forward today's change if available from API
                  currentChange={companyInfo?.change ?? (stockData.length > 1 ? undefined : undefined)}
                  currentChangePercent={companyInfo?.changePercent ?? undefined}
                  companyInfo={companyInfo}
                  fundamentalsData={fundamentalsData}
                  fundamentalsLoading={fundamentalsLoading}
                  finvizStock={finvizStock}
                />
              </div>
            )}

            <FinvizPanel
              symbol={symbol}
              onStockData={setFinvizStock}
              onAnalystTargets={setFinvizAnalystTargets}
              onNewsData={handleFinvizNews}
              onChartsData={(charts, links) => { setFinvizCharts(charts); setFinvizLinks(links); }}
            />

            {/* Main Chart with Integrated Controls */}
            <div className="card mb-4 sm:mb-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-3 flex-wrap">
                  <span className="card-label">Price Chart with Forecast</span>
                  {!showFinvizChart && <div className="flex flex-wrap gap-1">
                    {([
                      { label: 'MA20', checked: showMA20, set: setShowMA20 },
                      { label: 'MA50', checked: showMA50, set: setShowMA50 },
                      { label: 'Bollinger', checked: showBB, set: setShowBB },
                      { label: 'Volume', checked: showVolume, set: setShowVolume },
                      { label: 'RSI/MACD', checked: showIndicators, set: setShowIndicators },
                      { label: 'Patterns', checked: showPatterns, set: setShowPatterns },
                    ] as const).map(({ label, checked, set }) => (
                      <label key={label} className="flex items-center gap-1 cursor-pointer px-2 py-1 border transition-all" style={{
                        background: checked ? 'var(--bg-3)' : 'var(--bg-4)',
                        borderColor: checked ? 'var(--accent)' : 'var(--bg-1)',
                        borderLeftWidth: checked ? '2px' : '1px',
                      }}>
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(e) => { const v = e.target.checked; startTransition(() => set(v)); }}
                          className="w-3 h-3 cursor-pointer border appearance-none transition-all flex-shrink-0"
                          style={{ borderColor: 'var(--bg-1)', background: checked ? 'var(--accent)' : 'var(--bg-4)' }}
                        />
                        <span className="text-xs font-medium" style={{ color: checked ? 'var(--text-2)' : 'var(--text-4)' }}>{label}</span>
                      </label>
                    ))}
                  </div>}
                </div>

                {/* Compact Controls */}
                <div className="overflow-x-auto pb-0.5 -mb-0.5">
                <div className="flex items-center gap-2 min-w-max">
                  {/* Finviz toggle */}
                  {finvizCharts && (
                    <>
                      <button
                        onClick={() => setShowFinvizChart(v => !v)}
                        className="px-2 py-1 text-[10px] font-semibold border transition-all"
                        style={{
                          background: showFinvizChart ? 'var(--info)' : 'var(--bg-4)',
                          borderColor: showFinvizChart ? 'var(--info)' : 'var(--bg-1)',
                          color: showFinvizChart ? 'var(--text-0)' : 'var(--text-3)',
                        }}
                      >
                        <div className="flex items-center gap-1.5"><BarChart2 className="w-3.5 h-3.5" /> FINVIZ</div>
                      </button>
                      <div className="h-4 w-px" style={{ background: 'var(--bg-1)' }} />
                    </>
                  )}

                  {/* Finviz chart type buttons (shown only in Finviz mode) */}
                  {showFinvizChart && finvizCharts ? (
                    <>
                      {([
                        ['dailyCandle','6M','~6 months · daily candles'],
                        ['weeklyCandle','2Y','~2 years · weekly candles'],
                        ['monthlyCandle','All','10+ years · monthly candles'],
                        ['dailyLine','Line','~6 months · daily line'],
                      ] as const).map(([id, label, tip]) => (
                        <button
                          key={id}
                          onClick={() => setActiveFinvizChart(id)}
                          title={tip}
                          className="px-2 py-1 text-[10px] font-semibold border transition-all"
                          style={{
                            background: activeFinvizChart === id ? 'var(--accent)' : 'var(--bg-4)',
                            borderColor: activeFinvizChart === id ? 'var(--accent)' : 'var(--bg-1)',
                            color: activeFinvizChart === id ? 'var(--text-0)' : 'var(--text-3)',
                          }}
                        >
                          {label}
                        </button>
                      ))}
                    </>
                  ) : (
                  <>
                  {/* Chart Type */}
                  <div className="flex gap-1">
                    <button
                      onClick={() => handleChartTypeChange('line')}
                      className="px-2 py-1 text-[10px] font-semibold border transition-all"
                      style={{
                        background: chartType === 'line' ? 'var(--accent)' : 'var(--bg-4)',
                        borderColor: chartType === 'line' ? 'var(--accent)' : 'var(--bg-1)',
                        color: chartType === 'line' ? 'var(--text-0)' : 'var(--text-3)',
                      }}
                    >
                      <div className="flex items-center gap-1.5"><TrendingUp className="w-3.5 h-3.5" /> LINE</div>
                    </button>
                    <button
                      onClick={() => handleChartTypeChange('candlestick')}
                      className="px-2 py-1 text-[10px] font-semibold border transition-all"
                      style={{
                        background: chartType === 'candlestick' ? 'var(--accent)' : 'var(--bg-4)',
                        borderColor: chartType === 'candlestick' ? 'var(--accent)' : 'var(--bg-1)',
                        color: chartType === 'candlestick' ? 'var(--text-0)' : 'var(--text-3)',
                      }}
                    >
                      <div className="flex items-center gap-1.5"><BarChart2 className="w-3.5 h-3.5" /> CANDLE</div>
                    </button>
                    <button
                      onClick={() => setUseLightweightChart(v => !v)}
                      className="px-2 py-1 text-[10px] font-semibold border transition-all"
                      style={{
                        background: useLightweightChart ? 'var(--info)' : 'var(--bg-4)',
                        borderColor: useLightweightChart ? 'var(--info)' : 'var(--bg-1)',
                        color: useLightweightChart ? 'var(--text-0)' : 'var(--text-3)',
                      }}
                      title="Toggle TradingView lightweight-charts renderer"
                    >
                      <div className="flex items-center gap-1.5"><TrendingUp className="w-3.5 h-3.5" /> TW</div>
                    </button>
                  </div>

                  <div className="h-4 w-px" style={{ background: 'var(--bg-1)' }} />

                  {/* Forecast Type */}
                  <div className="flex gap-1">
                    <button
                      onClick={() => setUseProphetForecast(false)}
                      className="px-2 py-1 text-[10px] font-semibold border transition-all"
                      style={{
                        background: !useProphetForecast ? 'var(--success)' : 'var(--bg-4)',
                        borderColor: !useProphetForecast ? 'var(--success)' : 'var(--bg-1)',
                        color: !useProphetForecast ? 'var(--text-0)' : 'var(--text-3)',
                      }}
                    >
                      <div className="flex items-center gap-1.5"><Brain className="w-3.5 h-3.5" /> ML</div>
                    </button>
                    <button
                      onClick={() => setUseProphetForecast(true)}
                      className="px-2 py-1 text-[10px] font-semibold border transition-all"
                      style={{
                        background: useProphetForecast ? 'var(--info)' : 'var(--bg-4)',
                        borderColor: useProphetForecast ? 'var(--info)' : 'var(--bg-1)',
                        color: useProphetForecast ? 'var(--text-0)' : 'var(--text-3)',
                      }}
                    >
                      <div className="flex items-center gap-1.5"><Sparkles className="w-3.5 h-3.5" /> PROPHET</div>
                    </button>
                  </div>

                  <div className="h-4 w-px" style={{ background: 'var(--bg-1)' }} />

                  {/* Forecast Days */}
                  <div className="flex items-center gap-1">
                    <label htmlFor="forecast-horizon" className="text-[10px] font-semibold" style={{ color: 'var(--text-4)' }}>
                      DAYS:
                    </label>
                    <input
                      id="forecast-horizon"
                      type="number"
                      min="7"
                      max="90"
                      value={forecastHorizon}
                      onChange={(e) => setForecastHorizon(parseInt(e.target.value) || 30)}
                      className="w-16 border font-mono text-[10px]"
                      style={{
                        background: 'var(--bg-4)',
                        borderColor: 'var(--bg-1)',
                        color: 'var(--text-2)',
                        outline: 'none',
                        padding: '4px 6px',
                      }}
                    />
                  </div>

                  <div className="h-4 w-px" style={{ background: 'var(--bg-1)' }} />

                  {/* Frequency */}
                  <span className="text-[10px] font-semibold" style={{ color: 'var(--text-4)' }}>FREQ:</span>
                  {DATA_FREQUENCY_OPTIONS.map(option => {
                    const isActive = option.id === dataFrequencyId;
                    return (
                      <button
                        key={option.id}
                        onClick={() => handleFrequencyChange(option.id)}
                        className="px-2 py-1 text-[10px] font-semibold border transition-all"
                        style={{
                          background: isActive ? 'var(--accent)' : 'var(--bg-4)',
                          borderColor: isActive ? 'var(--accent)' : 'var(--bg-1)',
                          color: isActive ? 'var(--text-0)' : 'var(--text-3)',
                          opacity: loading && isActive ? 0.7 : 1,
                        }}
                        disabled={loading && isActive}
                        title={option.description}
                      >
                        {option.label}
                      </button>
                    );
                  })}
                  {/* Export CSV */}
                  {chartData.length > 0 && (
                    <>
                      <div className="h-4 w-px" style={{ background: 'var(--bg-1)' }} />
                      <button
                        onClick={() => exportToCSV(symbol, chartData)}
                        className="px-2 py-1 text-[10px] font-semibold border transition-all flex items-center gap-1"
                        style={{
                          background: 'var(--bg-4)',
                          borderColor: 'var(--bg-1)',
                          color: 'var(--text-3)',
                        }}
                        title="Export chart data to CSV"
                      >
                        <Download className="w-3 h-3" /> CSV
                      </button>
                    </>
                  )}
                  </>
                  )}
                </div>
                </div>
              </div>

              {showFinvizChart && finvizCharts ? (
                <img
                  src={finvizCharts[activeFinvizChart]}
                  alt={`${symbol} ${activeFinvizChart} Finviz chart`}
                  className="w-full border"
                  style={{ borderColor: 'var(--bg-1)', background: 'var(--bg-4)' }}
                />
              ) : (
              <div className="relative">
                {useLightweightChart ? (
                  <LightweightChartWrapper
                    data={chartData}
                    chartType={chartType}
                    showVolume={showVolume}
                    showMA20={showMA20}
                    showMA50={showMA50}
                    showBB={showBB}
                    dataInterval={dataInterval}
                    patterns={showPatterns ? chartPatterns : []}
                    enablePatterns={showPatterns}
                    forecastData={useProphetForecast ? prophetForecastData : forecastData}
                    showForecast={true}
                  />
                ) : (
                  <StockChart
                    data={chartData}
                    showMA20={showMA20}
                    showMA50={showMA50}
                    showBB={showBB}
                    forecastData={useProphetForecast ? prophetForecastData : forecastData}
                    chartType={chartType}
                    showVolume={showVolume}
                    patterns={showPatterns ? chartPatterns : []}
                    enablePatterns={showPatterns}
                    dataInterval={dataInterval}
                    onVisibleRangeChange={handleVisibleRangeChange}
                  />
                )}
                {chartRefreshing && (
                  <div className="absolute inset-0 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.25)', zIndex: 10 }}>
                    <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
                  </div>
                )}
              </div>
              )}
            </div>

            {/* Technical Indicators Charts */}
            {showIndicators && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6 mb-4 sm:mb-6">
                <div className="card">
                  <span className="card-label">RSI (Relative Strength Index)</span>
                  <TechnicalIndicatorsChart data={chartData} indicator="rsi" />
                </div>
                <div className="card">
                  <span className="card-label">MACD</span>
                  <TechnicalIndicatorsChart data={chartData} indicator="macd" />
                </div>
              </div>
            )}

            {/* Forecast Insights */}
            {forecastInsights && (
              <div className="card mb-6">
                <span className="card-label">Forecast Insights</span>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="p-4 border-2" style={{
                    background: 'var(--bg-2)',
                    borderColor: 'var(--info)',
                    borderLeftWidth: '3px'
                  }}>
                    <div className="text-sm mb-1" style={{ color: 'var(--text-3)' }}>7-Day Forecast</div>
                    <div className="text-2xl font-bold" style={{ color: 'var(--info)' }}>
                      ${forecastInsights.shortTerm.price.toFixed(2)}
                    </div>
                    <div
                      className="text-sm font-medium"
                      style={{
                        color: forecastInsights.shortTerm.change > 0 ? 'var(--success)' : 'var(--danger)'
                      }}
                    >
                      {forecastInsights.shortTerm.change > 0 ? '+' : ''}
                      {forecastInsights.shortTerm.change.toFixed(2)}%
                    </div>
                  </div>
                  <div className="p-4 border-2" style={{
                    background: 'var(--bg-2)',
                    borderColor: 'var(--accent)',
                    borderLeftWidth: '3px'
                  }}>
                    <div className="text-sm mb-1" style={{ color: 'var(--text-3)' }}>30-Day Forecast</div>
                    <div className="text-2xl font-bold" style={{ color: 'var(--accent)' }}>
                      ${forecastInsights.mediumTerm.price.toFixed(2)}
                    </div>
                    <div
                      className="text-sm font-medium"
                      style={{
                        color: forecastInsights.mediumTerm.change > 0 ? 'var(--success)' : 'var(--danger)'
                      }}
                    >
                      {forecastInsights.mediumTerm.change > 0 ? '+' : ''}
                      {forecastInsights.mediumTerm.change.toFixed(2)}%
                    </div>
                  </div>
                  <div className="p-4 border-2" style={{
                    background: 'var(--bg-2)',
                    borderColor: 'var(--purple-2)',
                    borderLeftWidth: '3px'
                  }}>
                    <div className="text-sm mb-1" style={{ color: 'var(--text-3)' }}>Trend</div>
                    <div className="text-2xl font-bold capitalize" style={{ color: 'var(--purple-2)' }}>
                      {forecastInsights.trend.direction}
                    </div>
                    <div className="text-sm font-medium" style={{ color: 'var(--purple-2)' }}>
                      Strength: {forecastInsights.trend.strength.toFixed(0)}%
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* AI Analysis - Full Width in Main Content */}
            <div className="mb-6">
              <AIAnalysis
                symbol={symbol}
                companyName={companyName}
                currentPrice={currentPrice}
                companyInfo={companyInfo}
                fundamentalsData={fundamentalsData}
                tradingSignal={tradingSignal}
                forecastInsights={forecastInsights}
                mlPredictions={mlPredictions}
                newsArticles={newsArticles}
                newsSentiments={newsSentiments}
                chartData={chartData}
                chartPatterns={chartPatterns}
              />
            </div>

            {/* Two Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
              {/* Trading Signals */}
              {tradingSignal && (
                <TradingSignals signal={tradingSignal} currentPrice={currentPrice} />
              )}

              {/* News & Sentiment */}
              <NewsPanel
                articles={newsArticles}
                sentiments={newsSentiments}
                isAnalyzingSentiment={isAnalyzingSentiment}
                finvizNewsLoading={finvizNewsLoading}
              />
            </div>

            {/* Correlation Heatmap */}
            <div className="mt-6">
              <CorrelationHeatmap
                symbol={symbol}
              />
            </div>

            {/* Live Prediction Lab */}
            <div className="mt-6">
              <LivePredictionChart symbol={symbol} />
            </div>

            {/* Mobile: stack ML panels below News */}
            <div className="block xl:hidden mt-6 space-y-4">
              <MLPredictions
                currentPrice={currentPrice}
                predictions={mlPredictions}
                isTraining={mlTraining}
                fromCache={mlFromCache}
                onRecalculate={() => fetchData(symbol, { forceRecalc: true })}
                inlineMobile={true}
                onLoadPrediction={handleLoadCachedPrediction}
                mlSettings={mlSettings}
                onSettingsChange={handleMlSettingsChange}
                onPresetChange={handleMlPresetChange}
                currentPreset={mlPreset}
              />

              {/* Pattern Panel - Mobile */}
              {showPatterns && (
                <PatternPanel
                  patterns={chartPatterns}
                  startDate={visibleDateRange?.startDate}
                  endDate={visibleDateRange?.endDate}
                  onRefreshPatterns={detectPatterns}
                  isDetecting={patternDetecting}
                  settings={patternSettings}
                  onSettingsChange={handlePatternSettingsChange}
                  onPresetChange={handlePatternPresetChange}
                  currentPreset={patternPreset}
                  inlineMobile={true}
                />
              )}

              {/* Reddit Sentiment - Mobile */}
              <RedditSentiment
                onTickerClick={(ticker) => {
                  setInputSymbol(ticker);
                  fetchData(ticker);
                }}
                inlineMobile={true}
              />

              {/* ApeWisdom Mentions - Mobile */}
              <ApeWisdomMentions
                onTickerClick={(ticker) => {
                  setInputSymbol(ticker);
                  fetchData(ticker);
                }}
                inlineMobile={true}
              />

              {/* FinnHub Features - Mobile */}
              <InsiderTransactions symbol={symbol} inlineMobile={true} />

              <EarningsCalendar symbol={symbol} inlineMobile={true} />

              <AnalystRecommendations symbol={symbol} inlineMobile={true} finvizTargets={finvizAnalystTargets} />

              <PeerStocks
                symbol={symbol}
                onPeerClick={(peer) => {
                  setInputSymbol(peer);
                  fetchData(peer);
                }}
                inlineMobile={true}
              />

              {/* Correlation Heatmap - Mobile */}
              <CorrelationHeatmap
                symbol={symbol}
                inlineMobile={true}
              />

              {/* Watchlist - Mobile */}
              <Watchlist
                currentSymbol={symbol}
                onSymbolClick={(s) => { setInputSymbol(s); fetchData(s); }}
                inlineMobile={true}
              />

              {/* Options Chain - Mobile */}
              <OptionsChain symbol={symbol} />

              {/* Market Movers - Mobile */}
              <MarketMovers
                onTickerClick={(ticker) => { setInputSymbol(ticker); fetchData(ticker); }}
                inlineMobile={true}
              />

              {/* Fear & Greed - Mobile */}
              <FearGreedIndex />
            </div>
          </>
        )}
        </div>

        {/* Right Sidebar - Desktop */}
        <div className="hidden xl:block flex-shrink-0 w-80">
          {/* Watchlist - Always visible */}
          <div>
            <Watchlist
              currentSymbol={symbol}
              onSymbolClick={(s) => { setInputSymbol(s); fetchData(s); }}
            />
          </div>

          {/* Fear & Greed Index */}
          <div className="mt-4">
            <FearGreedIndex />
          </div>

          {/* Market Movers */}
          <div className="mt-4">
            <MarketMovers
              onTickerClick={(ticker) => { setInputSymbol(ticker); fetchData(ticker); }}
            />
          </div>

          {/* Stock-specific panels */}
          {!loading && stockData.length > 0 && (
            <>
            <div className="mt-4">
              <MLPredictions
                currentPrice={currentPrice}
                predictions={mlPredictions}
                isTraining={mlTraining}
                fromCache={mlFromCache}
                onRecalculate={() => {
                  fetchData(symbol, { forceRecalc: true });
                }}
                onLoadPrediction={handleLoadCachedPrediction}
                mlSettings={mlSettings}
                onSettingsChange={handleMlSettingsChange}
                onPresetChange={handleMlPresetChange}
                currentPreset={mlPreset}
              />
            </div>

            {/* Pattern Panel - Combined settings + analysis */}
            {showPatterns && (
              <div className="mt-4">
                <PatternPanel
                  patterns={chartPatterns}
                  startDate={visibleDateRange?.startDate}
                  endDate={visibleDateRange?.endDate}
                  onRefreshPatterns={detectPatterns}
                  isDetecting={patternDetecting}
                  settings={patternSettings}
                  onSettingsChange={handlePatternSettingsChange}
                  onPresetChange={handlePatternPresetChange}
                  currentPreset={patternPreset}
                />
              </div>
            )}

            {/* Reddit Sentiment - Desktop Sidebar */}
            <div className="mt-4">
              <RedditSentiment
                onTickerClick={(ticker) => {
                  setInputSymbol(ticker);
                  fetchData(ticker);
                }}
              />
            </div>

            {/* ApeWisdom Mentions - Desktop Sidebar */}
            <div className="mt-4">
              <ApeWisdomMentions
                onTickerClick={(ticker) => {
                  setInputSymbol(ticker);
                  fetchData(ticker);
                }}
              />
            </div>

            {/* FinnHub Features - Desktop Sidebar */}
            <div className="mt-4">
              <InsiderTransactions symbol={symbol} />
            </div>

            <div className="mt-4">
              <EarningsCalendar symbol={symbol} />
            </div>

            <div className="mt-4">
              <AnalystRecommendations symbol={symbol} finvizTargets={finvizAnalystTargets} />
            </div>

            <div className="mt-4">
              <PeerStocks
                symbol={symbol}
                onPeerClick={(peer) => {
                  setInputSymbol(peer);
                  fetchData(peer);
                }}
              />
            </div>

            {/* Options Chain */}
            <div className="mt-4">
              <OptionsChain symbol={symbol} />
            </div>
            </>
          )}
        </div>
      </div>
    </main>
  );
}
