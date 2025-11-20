'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { Search, TrendingUp, Loader2, AlertCircle, Github, Clock } from 'lucide-react';
import { calculateAllIndicators } from '@/lib/technicalIndicators';
import { generateForecast, getForecastInsights } from '@/lib/forecasting';
import { generateTradingSignal } from '@/lib/tradingSignals';
import { StockData, NewsArticle, ChartDataPoint, ChartPattern } from '@/types';
import { getCachedPredictions, savePredictionsToCache, CachedPrediction } from '@/lib/predictionsCache';
import { MLSettings, MLPreset, DEFAULT_ML_SETTINGS } from '@/types/mlSettings';
import { PatternSettings, PatternPreset, DEFAULT_PATTERN_SETTINGS } from '@/types/patternSettings';
import type { SearchHistoryItem } from '@/components/Sidebar';

// Lazy load heavy components with dynamic imports
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
const PatternAnalysis = dynamic(() => import('@/components/PatternAnalysis'), { ssr: false });
const MLSettingsPanel = dynamic(() => import('@/components/MLSettingsPanel'), { ssr: false });
const PatternSettingsPanel = dynamic(() => import('@/components/PatternSettingsPanel'), { ssr: false });
const CorrelationHeatmap = dynamic(() => import('@/components/CorrelationHeatmap'), { ssr: false });
const PredictionsCache = dynamic(() => import('@/components/PredictionsCache'), { ssr: false });
const RedditSentiment = dynamic(() => import('@/components/RedditSentiment'), { ssr: false });
const ApeWisdomMentions = dynamic(() => import('@/components/ApeWisdomMentions'), { ssr: false });
const InsiderTransactions = dynamic(() => import('@/components/InsiderTransactions'), { ssr: false });
const EarningsCalendar = dynamic(() => import('@/components/EarningsCalendar'), { ssr: false });
const AnalystRecommendations = dynamic(() => import('@/components/AnalystRecommendations'), { ssr: false });
const PeerStocks = dynamic(() => import('@/components/PeerStocks'), { ssr: false });

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

const DEFAULT_DATA_FREQUENCY_ID: DataFrequencyId = '1h';
const DEFAULT_FREQUENCY_OPTION =
  DATA_FREQUENCY_OPTIONS.find(option => option.id === DEFAULT_DATA_FREQUENCY_ID)!;

const getFrequencyOption = (id?: DataFrequencyId): DataFrequencyOption =>
  id ? DATA_FREQUENCY_OPTIONS.find(option => option.id === id) ?? DEFAULT_FREQUENCY_OPTION : DEFAULT_FREQUENCY_OPTION;

type FetchDataOptions = {
  forceRecalc?: boolean;
  skipMLCalculations?: boolean;
  frequencyId?: DataFrequencyId;
};

export default function Home() {
  const [symbol, setSymbol] = useState('AAPL');
  const [inputSymbol, setInputSymbol] = useState('AAPL');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [tfReady, setTfReady] = useState(false);

  const [stockData, setStockData] = useState<StockData[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [companyName, setCompanyName] = useState<string>('');
  const [marketState, setMarketState] = useState<string>('');
  const [companyInfo, setCompanyInfo] = useState<any>(null);
  const [fundamentalsData, setFundamentalsData] = useState<any>(null);
  const [fundamentalsLoading, setFundamentalsLoading] = useState(false);
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
  const [debouncedQuery, setDebouncedQuery] = useState('');

  // Chart display options
  const [showMA20, setShowMA20] = useState(true);
  const [showMA50, setShowMA50] = useState(true);
  const [showBB, setShowBB] = useState(false);
  const [showIndicators, setShowIndicators] = useState(true);
  const [forecastHorizon, setForecastHorizon] = useState(30);
  const [chartType, setChartType] = useState<'line' | 'candlestick'>('line');
  const [showVolume, setShowVolume] = useState(true);
  const [showPatterns, setShowPatterns] = useState(false);
  const [dataFrequencyId, setDataFrequencyId] = useState<DataFrequencyId>(DEFAULT_DATA_FREQUENCY_ID);
  const [dataInterval, setDataInterval] = useState<string>(DEFAULT_FREQUENCY_OPTION.interval);
  const [visibleDateRange, setVisibleDateRange] = useState<{ startDate: string; endDate: string } | null>(null);
  const currentFrequency = React.useMemo(
    () => getFrequencyOption(dataFrequencyId),
    [dataFrequencyId]
  );

  // Optimize chart type changes to avoid blocking UI
  const handleChartTypeChange = (type: 'line' | 'candlestick') => {
    // Use setTimeout to make the state update non-blocking
    setTimeout(() => {
      setChartType(type);
    }, 0);
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
    console.log('🔄 Pattern settings updated:', newSettings);
    setPatternSettings(newSettings);
  }, []);

  const handlePatternPresetChange = React.useCallback((newPreset: PatternPreset) => {
    console.log('🔄 Pattern preset updated:', newPreset);
    setPatternPreset(newPreset);
  }, []);

  // Function to manually detect patterns
  const detectPatterns = React.useCallback(() => {
    if (!chartData.length) {
      setChartPatterns([]);
      return;
    }

    console.log('🔍 Starting pattern detection with settings:', patternSettings);
    console.log('📊 Chart data points:', chartData.length);

    setPatternDetecting(true);
    loadMLLibraries().then(({ detectChartPatterns }) => {
      // Use requestIdleCallback to run pattern detection when browser is idle
      if (typeof requestIdleCallback !== 'undefined') {
        requestIdleCallback(() => {
          const startTime = performance.now();
          const patterns = detectChartPatterns(chartData, patternSettings);
          const endTime = performance.now();

          console.log(`✅ Pattern detection completed in ${(endTime - startTime).toFixed(2)}ms`);
          console.log(`📈 Detected ${patterns.length} patterns:`, patterns.map(p => `${p.type} (${(p.confidence * 100).toFixed(0)}%)`));

          setChartPatterns(patterns);
          setPatternDetecting(false);
        }, { timeout: 2000 });
      } else {
        // Fallback for browsers without requestIdleCallback
        setTimeout(() => {
          const startTime = performance.now();
          const patterns = detectChartPatterns(chartData, patternSettings);
          const endTime = performance.now();

          console.log(`✅ Pattern detection completed in ${(endTime - startTime).toFixed(2)}ms`);
          console.log(`📈 Detected ${patterns.length} patterns:`, patterns.map(p => `${p.type} (${(p.confidence * 100).toFixed(0)}%)`));

          setChartPatterns(patterns);
          setPatternDetecting(false);
        }, 0);
      }
    }).catch(error => {
      console.error('❌ Pattern detection error:', error);
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
        const parsed = JSON.parse(savedHistory);
        console.log('Loaded search history from localStorage:', parsed);
        setSearchHistory(parsed);
      } else {
        console.log('No search history found in localStorage');
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

  // Save search history to localStorage
  const saveSearchHistory = (history: SearchHistoryItem[]) => {
    localStorage.setItem('stockSearchHistory', JSON.stringify(history));
    setSearchHistory(history);
  };

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
        console.log('Saved search history to localStorage:', newHistory);
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
      console.log('Cleared search history from localStorage');
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

      // eslint-disable-next-line no-console
      console.debug('[MarketStatus]', { now: now.toString(), et: { hour, minute, weekdayShort, weekdayIndex }, currentMinutes, status });

      return status;
    } catch (e) {
      const monthIdx = now.getMonth();
      const isDSTInNY = monthIdx >= 2 && monthIdx <= 10;
      const etOffset = isDSTInNY ? -4 : -5;
      const etTime = new Date(now.getTime() + etOffset * 60 * 60 * 1000);
      const dayIdx = etTime.getDay();
      const hour = etTime.getHours();
      const minute = etTime.getMinutes();

      // eslint-disable-next-line no-console
      console.debug('[MarketStatus-fallback]', { now: now.toString(), etTime: etTime.toString(), dayIdx, hour, minute });

      if (dayIdx === 0 || dayIdx === 6) return 'CLOSED';
      const currentMinutes = hour * 60 + minute;
      if (currentMinutes < 9 * 60 + 30) return 'PRE';
      if (currentMinutes >= 16 * 60) return 'POST';
      return 'REGULAR';
    }
  };

  const fetchData = async (
    stockSymbol: string,
    options: FetchDataOptions = {}
  ) => {
    const { forceRecalc = false, skipMLCalculations = false, frequencyId } = options;
    const targetFrequency = getFrequencyOption(frequencyId ?? dataFrequencyId);
    const intervalParam = targetFrequency.interval;
    const rangeDays = targetFrequency.days;

    setLoading(true);
    setError('');
    setNewsArticles([]); // Reset news to show loading state
    setNewsSentiments([]);
    setIsAnalyzingSentiment(false);
    setMlPredictions({}); // Reset ML predictions for new stock
    setMlTraining(false); // Reset training state

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
          console.log(`Using cached fundamentals for ${stockSymbol}`);
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

      // Skip ML if requested (e.g., when loading from cache table)
      if (skipMLCalculations) {
        console.log('Skipping ML calculations as requested');
        return;
      }

      // Check cache first unless force recalculate is true
      const cached = !forceRecalc ? getCachedPredictions(stockSymbol, forecastHorizon) : null;

      if (cached && !forceRecalc) {
        // Use cached predictions
        console.log(`Loading cached ML predictions for ${stockSymbol}`);
        setMlPredictions(cached.predictions);
        setMlFromCache(true);
        setMlTraining(false);
      } else {
        // Run ML algorithms in background (much lower priority - wait for chart to render)
        setMlFromCache(false);
        setMlTraining(true);
        setTimeout(async () => {
          try {
            // Lazy load ML libraries first
            const mlLibs = await loadMLLibraries();

            // Fast algorithms (non-neural network) - run first for immediate display
            const linearReg = mlLibs.generateLinearRegression(stockResult.data, forecastHorizon);
            const emaForecast = mlLibs.generateEMAForecast(stockResult.data, forecastHorizon);
            const arimaForecast = mlLibs.generateARIMAForecast(stockResult.data, forecastHorizon);
            const prophetLite = mlLibs.generateProphetLiteForecast(stockResult.data, forecastHorizon);

            const predictions = {
              linearRegression: linearReg,
              ema: emaForecast,
              arima: arimaForecast,
              prophetLite,
            };

            setMlPredictions(predictions);

            // Neural network models - TRAIN SEQUENTIALLY to prevent browser freeze
            console.log('Starting neural network models (sequential training)...');

            // Train models ONE AT A TIME to avoid freezing the browser
            let lstm = null;
            try {
              console.log('Training LSTM...');
              lstm = await mlLibs.generateMLForecast(stockResult.data, forecastHorizon, mlSettings);
              setMlPredictions(prev => ({ ...prev, lstm }));
              console.log('✅ LSTM complete');
            } catch (err) {
              console.error('LSTM failed:', err);
            }

            let gru = null;
            try {
              console.log('Training GRU...');
              gru = await mlLibs.generateGRUForecast(stockResult.data, forecastHorizon, mlSettings);
              setMlPredictions(prev => ({ ...prev, gru }));
              console.log('✅ GRU complete');
            } catch (err) {
              console.error('GRU failed:', err);
            }

            let cnnLstm = null;
            try {
              console.log('Training CNN-LSTM...');
              cnnLstm = await mlLibs.generateCNNLSTMForecast(stockResult.data, forecastHorizon, mlSettings);
              setMlPredictions(prev => ({ ...prev, cnnLstm }));
              console.log('✅ CNN-LSTM complete');
            } catch (err) {
              console.error('CNN-LSTM failed:', err);
            }

            // Generate ensemble INSTANTLY from already-trained models (no retraining!)
            const ensemble = mlLibs.generateEnsembleFromPredictions(
              { gru, cnnLstm, lstm },
              forecastHorizon
            );

            if (ensemble) {
              setMlPredictions(prev => ({ ...prev, ensemble }));
            }

            // Save all predictions to cache
            const allPredictions = {
              ...predictions,
              ...(gru && { gru }),
              ...(cnnLstm && { cnnLstm }),
              ...(lstm && { lstm }),
              ...(ensemble && { ensemble }),
            };

            savePredictionsToCache(stockSymbol, allPredictions, forecastHorizon);

            setMlTraining(false);
            console.log('All ML algorithms completed and cached!');
          } catch (mlError) {
            console.error('ML algorithms error:', mlError);
            setMlTraining(false);
          }
        }, 2000); // Wait 2 seconds to let chart fully render first
      }

      // Fetch news asynchronously (fast, no sentiment)
      setTimeout(async () => {
        try {
          const newsResponse = await fetch(`/api/news?symbol=${stockSymbol}`);
          if (newsResponse.ok) {
            const newsResult = await newsResponse.json();
            const articles = newsResult.articles || [];
            setNewsArticles(articles);

            // Set placeholder sentiments
            const placeholders = articles.map(() => ({
              sentiment: 'neutral' as const,
              score: 0,
              confidence: 0
            }));
            setNewsSentiments(placeholders);

            // Analyze sentiment in background (slower)
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
                  setNewsSentiments(sentimentResult.sentiments || placeholders);
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
            console.log(`Loading cached ML predictions for ${symbol} (horizon: ${forecastHorizon})`);
            setMlPredictions(cached.predictions);
            setMlFromCache(true);
            setMlTraining(false);
          } else {
            // Run all ML algorithms in background (delay to avoid blocking UI)
            setMlFromCache(false);
            setMlTraining(true);
            setTimeout(async () => {
            try {
              // Lazy load ML libraries first
              const mlLibs = await loadMLLibraries();

              // Fast algorithms (non-neural network)
              const linearReg = mlLibs.generateLinearRegression(stockData, forecastHorizon);
              const emaForecast = mlLibs.generateEMAForecast(stockData, forecastHorizon);
              const arimaForecast = mlLibs.generateARIMAForecast(stockData, forecastHorizon);
              const prophetLite = mlLibs.generateProphetLiteForecast(stockData, forecastHorizon);

              const predictions = {
                linearRegression: linearReg,
                ema: emaForecast,
                arima: arimaForecast,
                prophetLite,
              };

              setMlPredictions(predictions);

              // Neural network models - TRAIN SEQUENTIALLY to prevent browser freeze
              console.log('Starting neural network models (sequential training)...');

              // Train models ONE AT A TIME to avoid freezing the browser
              let lstm = null;
              try {
                console.log('Training LSTM...');
                lstm = await mlLibs.generateMLForecast(stockData, forecastHorizon, mlSettings);
                setMlPredictions(prev => ({ ...prev, lstm }));
                console.log('✅ LSTM complete');
              } catch (err) {
                console.error('LSTM failed:', err);
              }

              let gru = null;
              try {
                console.log('Training GRU...');
                gru = await mlLibs.generateGRUForecast(stockData, forecastHorizon, mlSettings);
                setMlPredictions(prev => ({ ...prev, gru }));
                console.log('✅ GRU complete');
              } catch (err) {
                console.error('GRU failed:', err);
              }

              let cnnLstm = null;
              try {
                console.log('Training CNN-LSTM...');
                cnnLstm = await mlLibs.generateCNNLSTMForecast(stockData, forecastHorizon, mlSettings);
                setMlPredictions(prev => ({ ...prev, cnnLstm }));
                console.log('✅ CNN-LSTM complete');
              } catch (err) {
                console.error('CNN-LSTM failed:', err);
              }

              // Generate ensemble INSTANTLY from already-trained models (no retraining!)
              const ensemble = mlLibs.generateEnsembleFromPredictions(
                { gru, cnnLstm, lstm },
                forecastHorizon
              );

              if (ensemble) {
                setMlPredictions(prev => ({ ...prev, ensemble }));
              }

              // Save all predictions to cache
              const allPredictions = {
                ...predictions,
                ...(gru && { gru }),
                ...(cnnLstm && { cnnLstm }),
                ...(lstm && { lstm }),
                ...(ensemble && { ensemble }),
              };

              savePredictionsToCache(symbol, allPredictions, forecastHorizon);

              setMlTraining(false);
            } catch (mlError) {
              console.error('ML algorithms error:', mlError);
              setMlTraining(false);
            }
          }, 1500); // Delay ML to let chart render smoothly
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
    console.log('Loading cached prediction:', cachedPred);

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

      console.log('Successfully loaded cached prediction from table');
    } finally {
      // Reset flag after a short delay to allow state updates to complete
      setTimeout(() => {
        isLoadingFromCacheTable.current = false;
      }, 500);
    }
  };

  const handleFrequencyChange = (nextId: DataFrequencyId) => {
    if (nextId === dataFrequencyId && stockData.length > 0 && !error) {
      return;
    }
    setDataFrequencyId(nextId);
    fetchData(symbol, { forceRecalc: true, frequencyId: nextId });
  };

  const handleVisibleRangeChange = (startDate: string, endDate: string) => {
    // Only update visible range if patterns are enabled
    if (showPatterns) {
      setVisibleDateRange({ startDate, endDate });
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  // Update market status on interval
  useEffect(() => {
    const interval = setInterval(() => {
      setMarketState(getMarketStatus());
    }, 60000); // Update every minute

    // Initial set
    setMarketState(getMarketStatus());

    return () => clearInterval(interval);
  }, []);

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
    <main className="min-h-screen p-4" style={{ background: 'var(--bg-4)' }}>
      <div className="flex gap-4 items-start">
        {/* Main Content Area - Now Full Width */}
        <div className="flex-1 min-w-0">
          {/* Header with Search */}
          <div className="card mb-4">
            <span className="card-label">Stock Predictor</span>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-6 h-6" style={{ color: 'var(--accent)' }} />
              <h1 className="text-lg font-bold" style={{ color: 'var(--text-1)' }}>
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

          <div className="flex items-center gap-3 flex-wrap">
              {/* Search Input */}
              <div className="flex gap-3 items-center flex-1">
                <div className="relative flex-1 max-w-md">
                  <input
                    type="text"
                    value={inputSymbol}
                    onChange={(e) => {
                      const value = e.target.value.toUpperCase();
                      setInputSymbol(value);
                      if (value) {
                        getTickerFromAPi(value);
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

                {/* Search History Dropdown */}
                {searchHistory.length > 0 && (
                  <details className="relative">
                    <summary className="px-4 py-3 font-medium transition-all cursor-pointer flex items-center gap-2 border list-none"
                      style={{
                        background: 'var(--bg-3)',
                        borderColor: 'var(--bg-1)',
                        color: 'var(--text-3)'
                      }}>
                      <Clock className="w-5 h-5" />
                      <span className="hidden sm:inline">History ({searchHistory.length})</span>
                      <span className="sm:hidden">{searchHistory.length}</span>
                    </summary>
                    <div className="absolute right-0 mt-2 w-64 border-2 shadow-lg z-50 max-h-96 overflow-y-auto"
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
                  </details>
                )}
              </div>

              {/* Market Status */}
              {!loading && stockData.length > 0 && (
                <div className="hidden md:flex items-center gap-2 px-3 py-2 border-2" style={{
                  borderColor: marketState === 'REGULAR' ? 'var(--success)' : 'var(--text-4)',
                  background: 'var(--bg-3)',
                }}>
                  <div className={`w-2 h-2 rounded-full ${marketState === 'REGULAR' ? 'animate-pulse' : ''}`} style={{
                    background: marketState === 'REGULAR' ? 'var(--success)' : 'var(--text-4)'
                  }} />
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold" style={{
                      color: marketState === 'REGULAR' ? 'var(--success)' : 'var(--text-4)'
                    }}>
                      {marketState === 'REGULAR' ? 'OPEN' : marketState === 'CLOSED' ? 'CLOSED' : marketState === 'PRE' ? 'PRE' : marketState === 'POST' ? 'POST' : 'CLOSED'}
                    </span>
                    <span className="text-xs" style={{ color: 'var(--text-4)' }}>{getETTimeString()} ET</span>
                  </div>
                </div>
              )}
            </div>
          </div>

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
                />
              </div>
            )}

            {/* Technical Indicators - Compact Bar */}
            <div className="card mb-4">
              <div className="flex items-center justify-between mb-3">
                <span className="card-label">Technical Indicators</span>
                <span className="text-xs" style={{ color: 'var(--text-5)' }}>Toggle overlays on chart</span>
              </div>
              
              <div className="grid grid-cols-3 md:grid-cols-6 gap-1">
                <label className="flex items-center gap-1.5 cursor-pointer group px-2 py-1.5 border transition-all" style={{
                  background: showMA20 ? 'var(--bg-3)' : 'var(--bg-4)',
                  borderColor: showMA20 ? 'var(--accent)' : 'var(--bg-1)',
                  borderLeftWidth: showMA20 ? '3px' : '1px'
                }}>
                  <input
                    type="checkbox"
                    checked={showMA20}
                    onChange={(e) => setShowMA20(e.target.checked)}
                    className="w-4 h-4 cursor-pointer border-2 appearance-none transition-all flex-shrink-0"
                    style={{
                      borderColor: 'var(--bg-1)',
                      background: showMA20 ? 'var(--accent)' : 'var(--bg-4)',
                    }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate" style={{ color: 'var(--text-2)' }}>20-Day MA</div>
                    <div className="text-xs truncate" style={{ color: 'var(--text-5)' }}>Short-term</div>
                  </div>
                </label>
                
                <label className="flex items-center gap-1.5 cursor-pointer group px-2 py-1.5 border transition-all" style={{
                  background: showMA50 ? 'var(--bg-3)' : 'var(--bg-4)',
                  borderColor: showMA50 ? 'var(--accent)' : 'var(--bg-1)',
                  borderLeftWidth: showMA50 ? '3px' : '1px'
                }}>
                  <input
                    type="checkbox"
                    checked={showMA50}
                    onChange={(e) => setShowMA50(e.target.checked)}
                    className="w-4 h-4 cursor-pointer border-2 appearance-none transition-all flex-shrink-0"
                    style={{
                      borderColor: 'var(--bg-1)',
                      background: showMA50 ? 'var(--accent)' : 'var(--bg-4)',
                    }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate" style={{ color: 'var(--text-2)' }}>50-Day MA</div>
                    <div className="text-xs truncate" style={{ color: 'var(--text-5)' }}>Mid-term</div>
                  </div>
                </label>

                <label className="flex items-center gap-1.5 cursor-pointer group px-2 py-1.5 border transition-all" style={{
                  background: showBB ? 'var(--bg-3)' : 'var(--bg-4)',
                  borderColor: showBB ? 'var(--accent)' : 'var(--bg-1)',
                  borderLeftWidth: showBB ? '3px' : '1px'
                }}>
                  <input
                    type="checkbox"
                    checked={showBB}
                    onChange={(e) => setShowBB(e.target.checked)}
                    className="w-4 h-4 cursor-pointer border-2 appearance-none transition-all flex-shrink-0"
                    style={{
                      borderColor: 'var(--bg-1)',
                      background: showBB ? 'var(--accent)' : 'var(--bg-4)',
                    }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate" style={{ color: 'var(--text-2)' }}>Bollinger</div>
                    <div className="text-xs truncate" style={{ color: 'var(--text-5)' }}>Volatility</div>
                  </div>
                </label>

                <label className="flex items-center gap-1.5 cursor-pointer group px-2 py-1.5 border transition-all" style={{
                  background: showVolume ? 'var(--bg-3)' : 'var(--bg-4)',
                  borderColor: showVolume ? 'var(--accent)' : 'var(--bg-1)',
                  borderLeftWidth: showVolume ? '3px' : '1px'
                }}>
                  <input
                    type="checkbox"
                    checked={showVolume}
                    onChange={(e) => setShowVolume(e.target.checked)}
                    className="w-4 h-4 cursor-pointer border-2 appearance-none transition-all flex-shrink-0"
                    style={{
                      borderColor: 'var(--bg-1)',
                      background: showVolume ? 'var(--accent)' : 'var(--bg-4)',
                    }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate" style={{ color: 'var(--text-2)' }}>Volume</div>
                    <div className="text-xs truncate" style={{ color: 'var(--text-5)' }}>Strength</div>
                  </div>
                </label>

                <label className="flex items-center gap-1.5 cursor-pointer group px-2 py-1.5 border transition-all" style={{
                  background: showIndicators ? 'var(--bg-3)' : 'var(--bg-4)',
                  borderColor: showIndicators ? 'var(--accent)' : 'var(--bg-1)',
                  borderLeftWidth: showIndicators ? '3px' : '1px'
                }}>
                  <input
                    type="checkbox"
                    checked={showIndicators}
                    onChange={(e) => setShowIndicators(e.target.checked)}
                    className="w-4 h-4 cursor-pointer border-2 appearance-none transition-all flex-shrink-0"
                    style={{
                      borderColor: 'var(--bg-1)',
                      background: showIndicators ? 'var(--accent)' : 'var(--bg-4)',
                    }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-medium truncate" style={{ color: 'var(--text-2)' }}>RSI/MACD</div>
                    <div className="text-xs truncate" style={{ color: 'var(--text-5)' }}>Momentum</div>
                  </div>
                </label>

              <label className="flex items-center gap-1.5 cursor-pointer group px-2 py-1.5 border transition-all" style={{
                background: showPatterns ? 'var(--bg-3)' : 'var(--bg-4)',
                borderColor: showPatterns ? 'var(--accent)' : 'var(--bg-1)',
                borderLeftWidth: showPatterns ? '3px' : '1px'
              }}>
                <input
                  type="checkbox"
                  checked={showPatterns}
                  onChange={(e) => setShowPatterns(e.target.checked)}
                  className="w-4 h-4 cursor-pointer border-2 appearance-none transition-all flex-shrink-0"
                  style={{
                    borderColor: 'var(--bg-1)',
                    background: showPatterns ? 'var(--accent)' : 'var(--bg-4)',
                  }}
                />
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium truncate" style={{ color: 'var(--text-2)' }}>Chart Patterns</div>
                  <div className="text-xs truncate" style={{ color: 'var(--text-5)' }}>(beta)</div>
                </div>
              </label>
              </div>
            </div>

            {/* Main Chart with Integrated Controls */}
            <div className="card mb-6">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <span className="card-label">Price Chart with Forecast</span>

                {/* Compact Controls */}
                <div className="flex flex-wrap items-center gap-2">
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
                      📈 LINE
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
                      📊 CANDLE
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
                      🤖 ML
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
                      🔮 PROPHET
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
                      className="w-20 px-2 py-1 border font-mono text-[10px]"
                      style={{
                        background: 'var(--bg-4)',
                        borderColor: 'var(--bg-1)',
                        color: 'var(--text-2)',
                        outline: 'none'
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
                </div>
              </div>

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
            </div>

            {/* Technical Indicators Charts */}
            {showIndicators && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
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

            {/* Two Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Trading Signals */}
              {tradingSignal && (
                <TradingSignals signal={tradingSignal} currentPrice={currentPrice} />
              )}

              {/* News & Sentiment */}
              <NewsPanel
                articles={newsArticles}
                sentiments={newsSentiments}
                isAnalyzingSentiment={isAnalyzingSentiment}
              />
            </div>

            {/* Correlation Heatmap */}
            <div className="mt-6">
              <CorrelationHeatmap
                symbol={symbol}
              />
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
              />

              {/* ML Cache - Mobile */}
              <PredictionsCache onLoadPrediction={handleLoadCachedPrediction} />

              {/* ML Settings Panel - Mobile */}
              <MLSettingsPanel
                settings={mlSettings}
                onSettingsChange={handleMlSettingsChange}
                onPresetChange={handleMlPresetChange}
                currentPreset={mlPreset}
                inlineMobile={true}
              />

              {/* Pattern Settings Panel - Mobile */}
              {showPatterns && (
                <PatternSettingsPanel
                  settings={patternSettings}
                  onSettingsChange={handlePatternSettingsChange}
                  onPresetChange={handlePatternPresetChange}
                  currentPreset={patternPreset}
                  inlineMobile={true}
                  patternCount={chartPatterns.length}
                  isDetecting={patternDetecting}
                />
              )}

              {/* Pattern Analysis - Mobile */}
              {showPatterns && chartPatterns.length > 0 && (
                <PatternAnalysis
                  patterns={chartPatterns}
                  startDate={visibleDateRange?.startDate}
                  endDate={visibleDateRange?.endDate}
                  inlineMobile={true}
                  onRefreshPatterns={detectPatterns}
                  isDetecting={patternDetecting}
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

              <AnalystRecommendations symbol={symbol} inlineMobile={true} />

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
            </div>
          </>
        )}
        </div>

        {/* ML Predictions Sidebar - Right Side (Desktop) */}
        {!loading && stockData.length > 0 && (
          <div className="hidden xl:block flex-shrink-0">
            <MLPredictions
              currentPrice={currentPrice}
              predictions={mlPredictions}
              isTraining={mlTraining}
              fromCache={mlFromCache}
              onRecalculate={() => {
                fetchData(symbol, { forceRecalc: true });
              }}
            />

            {/* ML Cache - Below ML Predictions */}
            <div className="mt-4">
              <PredictionsCache onLoadPrediction={handleLoadCachedPrediction} />
            </div>

            {/* ML Settings Panel - Below ML Cache */}
            <div className="mt-4">
              <MLSettingsPanel
                settings={mlSettings}
                onSettingsChange={handleMlSettingsChange}
                onPresetChange={handleMlPresetChange}
                currentPreset={mlPreset}
              />
            </div>

            {/* Pattern Settings Panel - Below ML Settings */}
            {showPatterns && (
              <div className="mt-4">
                <PatternSettingsPanel
                  settings={patternSettings}
                  onSettingsChange={handlePatternSettingsChange}
                  onPresetChange={handlePatternPresetChange}
                  currentPreset={patternPreset}
                  patternCount={chartPatterns.length}
                  isDetecting={patternDetecting}
                />
              </div>
            )}

            {/* Pattern Analysis - Below Pattern Settings */}
            {showPatterns && chartPatterns.length > 0 && (
              <div className="mt-4">
                <PatternAnalysis
                  patterns={chartPatterns}
                  startDate={visibleDateRange?.startDate}
                  endDate={visibleDateRange?.endDate}
                  onRefreshPatterns={detectPatterns}
                  isDetecting={patternDetecting}
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
              <AnalystRecommendations symbol={symbol} />
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
          </div>
        )}
      </div>
    </main>
  );
}
