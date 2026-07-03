'use client';

// ═══════════════════════════════════════════════════════════════════════════
// PRO DASHBOARD — a from-scratch second interface over the same app.
//
// This is NOT a restyle of app/page.tsx: it is its own shell (fixed sidebar
// navigation + command topbar + symbol hero + sectioned workspaces) built new.
// The data/ML pipeline is intentionally kept identical to the classic
// dashboard (same lib calls, same caches, same localStorage keys) so both UIs
// stay interchangeable, and every feature component from the classic UI is
// mounted here — organized into workspaces instead of one endless column.
// Reached at /pro (npm run dev:pro lands here via redirect).
// ═══════════════════════════════════════════════════════════════════════════

import React, { useState, useEffect, startTransition } from 'react';
import dynamic from 'next/dynamic';
import {
  Search, Loader2, AlertCircle, Github, Clock, BarChart2, Brain, Sparkles,
  Download, Briefcase, TrendingUp, TrendingDown, LayoutDashboard, Activity,
  Building2, Newspaper, Globe2, Star, X,
} from 'lucide-react';
import { calculateAllIndicators } from '@/lib/technicalIndicators';
import { generateForecast, getForecastInsights } from '@/lib/forecasting';
import { generateTradingSignal } from '@/lib/tradingSignals';
import { StockData, NewsArticle, ChartDataPoint, ChartPattern } from '@/types';
import { getCachedPredictions, savePredictionsToCache, CachedPrediction } from '@/lib/predictionsCache';
import { logPredictionRun } from '@/lib/predictionLog';
import { ErrorBoundary } from '@/components/ErrorBoundary';
import { MLSettings, MLPreset, DEFAULT_ML_SETTINGS } from '@/types/mlSettings';
import { PatternSettings, PatternPreset, DEFAULT_PATTERN_SETTINGS } from '@/types/patternSettings';
import type { SearchHistoryItem } from '@/components/Sidebar';
import { exportToCSV } from '@/lib/exportData';
import { loadMLLibraries } from '@/lib/loadMLLibraries';
import {
  DATA_FREQUENCY_OPTIONS,
  DataFrequencyId,
  DEFAULT_DATA_FREQUENCY_ID,
  DEFAULT_FREQUENCY_OPTION,
  getFrequencyOption,
} from '@/lib/dataFrequency';

// Feature components — the same set the classic dashboard uses, lazily loaded.
const LightweightChartWrapper = dynamic(() => import('@/components/LightweightChartWrapper'), { ssr: false });
const StockChart = dynamic(() => import('@/components/StockChart'), {
  loading: () => (
    <div className="h-96 flex items-center justify-center">
      <Loader2 className="w-8 h-8 animate-spin" style={{ color: 'var(--accent)' }} />
    </div>
  ),
  ssr: false,
});
const TechnicalIndicatorsChart = dynamic(() => import('@/components/TechnicalIndicatorsChart'), { ssr: false });
const NewsPanel = dynamic(() => import('@/components/NewsPanel'), { ssr: false });
const TradingSignals = dynamic(() => import('@/components/TradingSignals'), { ssr: false });
const CompanyInfo = dynamic(() => import('@/components/CompanyInfo'), { ssr: false });
const MLPredictions = dynamic(() => import('@/components/MLPredictions'), { ssr: false });
const PatternPanel = dynamic(() => import('@/components/PatternPanel'), { ssr: false });
const CorrelationHeatmap = dynamic(() => import('@/components/CorrelationHeatmap'), { ssr: false });
const RedditSentiment = dynamic(() => import('@/components/RedditSentiment'), { ssr: false });
const ApeWisdomMentions = dynamic(() => import('@/components/ApeWisdomMentions'), { ssr: false });
const InsiderTransactions = dynamic(() => import('@/components/InsiderTransactions'), { ssr: false });
const CongressionalTrading = dynamic(() => import('@/components/CongressionalTrading'), { ssr: false });
const EarningsCalendar = dynamic(() => import('@/components/EarningsCalendar'), { ssr: false });
const EconomicCalendar = dynamic(() => import('@/components/EconomicCalendar'), { ssr: false });
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
const DailyReturnHeatmap = dynamic(() => import('@/components/DailyReturnHeatmap'), { ssr: false });
const VolumeProfile = dynamic(() => import('@/components/VolumeProfile'), { ssr: false });
const EarningsHistory = dynamic(() => import('@/components/EarningsHistory'), { ssr: false });
const PredictionScorecard = dynamic(() => import('@/components/PredictionScorecard'), { ssr: false });
const SignalBacktest = dynamic(() => import('@/components/SignalBacktest'), { ssr: false });
// The standalone portfolio page mounted as a workspace (route stays available)
const PortfolioWorkspace = dynamic(() => import('@/app/portfolio/page'), { ssr: false });

// ─── Workspace navigation ────────────────────────────────────────────────────

type SectionId = 'overview' | 'ai' | 'technicals' | 'intel' | 'news' | 'markets' | 'tools' | 'portfolio';

const SECTIONS: { id: SectionId; label: string; hint: string; icon: React.ComponentType<any> }[] = [
  { id: 'overview',   label: 'Overview',       hint: 'Chart, signal & company',        icon: LayoutDashboard },
  { id: 'ai',         label: 'Predictions',    hint: 'Models, accuracy & backtests',   icon: Brain },
  { id: 'technicals', label: 'Technicals',     hint: 'RSI, MACD, patterns, volume',    icon: Activity },
  { id: 'intel',      label: 'Market Intel',   hint: 'Analysts, insiders, earnings',   icon: Building2 },
  { id: 'news',       label: 'News & Social',  hint: 'Headlines & crowd sentiment',    icon: Newspaper },
  { id: 'markets',    label: 'Markets',        hint: 'Movers, screener, fear & greed', icon: Globe2 },
  { id: 'tools',      label: 'Watchlist',      hint: 'Saved symbols & price alerts',   icon: Star },
  { id: 'portfolio',  label: 'Portfolio',      hint: 'Holdings, dividends & risk',     icon: Briefcase },
];

type FetchDataOptions = {
  forceRecalc?: boolean;
  skipMLCalculations?: boolean;
  frequencyId?: DataFrequencyId;
  chartOnly?: boolean;
};

export default function ProDashboard() {
  // ─── Core state (mirrors the classic dashboard 1:1) ───────────────────────
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
  const [, setFinvizLinks] = useState<Record<string, string>>({});
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
  const historyRef = React.useRef<HTMLDivElement>(null);
  const searchInputRef = React.useRef<HTMLInputElement>(null);

  // Shell state (pro-only)
  const [activeSection, setActiveSection] = useState<SectionId>('overview');
  // The portfolio workspace is heavy (own quotes/history fetches), so it
  // mounts on first visit and stays mounted after.
  const [portfolioMounted, setPortfolioMounted] = useState(false);
  useEffect(() => {
    if (activeSection === 'portfolio') setPortfolioMounted(true);
  }, [activeSection]);

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
  const [showFibonacci, setShowFibonacci] = useState(false);
  const [dataFrequencyId, setDataFrequencyId] = useState<DataFrequencyId>(DEFAULT_DATA_FREQUENCY_ID);
  const [dataInterval, setDataInterval] = useState<string>(DEFAULT_FREQUENCY_OPTION.interval);
  const [visibleDateRange, setVisibleDateRange] = useState<{ startDate: string; endDate: string } | null>(null);

  const handleChartTypeChange = (type: 'line' | 'candlestick') => {
    startTransition(() => setChartType(type));
  };

  const [patternDetecting, setPatternDetecting] = useState(false);

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

  const [mlSettings, setMlSettings] = useState<MLSettings>(DEFAULT_ML_SETTINGS);
  const [mlPreset, setMlPreset] = useState<MLPreset>('balanced');
  const [patternSettings, setPatternSettings] = useState<PatternSettings>(DEFAULT_PATTERN_SETTINGS);
  const [patternPreset, setPatternPreset] = useState<PatternPreset>('balanced');

  const handleMlSettingsChange = React.useCallback((newSettings: MLSettings) => {
    setMlSettings(newSettings);
  }, []);
  const handleMlPresetChange = React.useCallback((newPreset: MLPreset) => {
    setMlPreset(newPreset);
  }, []);
  const handlePatternSettingsChange = React.useCallback((newSettings: PatternSettings) => {
    setPatternSettings(newSettings);
  }, []);
  const handlePatternPresetChange = React.useCallback((newPreset: PatternPreset) => {
    setPatternPreset(newPreset);
  }, []);

  // ─── Pattern detection (identical pipeline to classic) ─────────────────────
  const detectPatterns = React.useCallback(() => {
    if (!chartData.length) {
      setChartPatterns([]);
      return;
    }

    setPatternDetecting(true);
    loadMLLibraries().then(({ detectChartPatterns }) => {
      if (typeof requestIdleCallback !== 'undefined') {
        requestIdleCallback(() => {
          const patterns = detectChartPatterns(chartData, patternSettings);
          setChartPatterns(patterns);
          setPatternDetecting(false);
        }, { timeout: 2000 });
      } else {
        setTimeout(() => {
          const patterns = detectChartPatterns(chartData, patternSettings);
          setChartPatterns(patterns);
          setPatternDetecting(false);
        }, 0);
      }
    }).catch(err => {
      console.error('[ERROR] Pattern detection error:', err);
      setPatternDetecting(false);
    });
  }, [chartData, patternSettings]);

  useEffect(() => {
    if (!showPatterns || !chartData.length) {
      setChartPatterns([]);
      return;
    }
    const timer = setTimeout(() => {
      detectPatterns();
    }, 300);
    return () => clearTimeout(timer);
  }, [chartData, showPatterns, patternSettings, detectPatterns]);

  // ─── Boot: TF init + persisted settings (same localStorage keys as classic,
  // so history/settings carry across both UIs) ────────────────────────────────
  useEffect(() => {
    const initTF = async () => {
      try {
        const { initializeTensorFlow, warmupGPU } = await import('@/lib/tfConfig');
        await initializeTensorFlow();
        await warmupGPU();
      } catch (err) {
        console.error('Failed to initialize TensorFlow.js:', err);
      }
    };
    initTF();

    try {
      const savedHistory = localStorage.getItem('stockSearchHistory');
      if (savedHistory) setSearchHistory(JSON.parse(savedHistory));

      const savedMLSettings = localStorage.getItem('mlSettings');
      const savedMLPreset = localStorage.getItem('mlPreset');
      if (savedMLSettings) setMlSettings(JSON.parse(savedMLSettings));
      if (savedMLPreset) setMlPreset(savedMLPreset as MLPreset);

      const savedPatternSettings = localStorage.getItem('patternSettings');
      const savedPatternPreset = localStorage.getItem('patternPreset');
      if (savedPatternSettings) setPatternSettings(JSON.parse(savedPatternSettings));
      if (savedPatternPreset) setPatternPreset(savedPatternPreset as PatternPreset);
    } catch (e) {
      console.error('Failed to load from localStorage:', e);
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem('mlSettings', JSON.stringify(mlSettings));
      localStorage.setItem('mlPreset', mlPreset);
    } catch (err) {
      console.error('Error saving ML settings:', err);
    }
  }, [mlSettings, mlPreset]);

  useEffect(() => {
    try {
      localStorage.setItem('patternSettings', JSON.stringify(patternSettings));
      localStorage.setItem('patternPreset', patternPreset);
    } catch (err) {
      console.error('Error saving Pattern settings:', err);
    }
  }, [patternSettings, patternPreset]);

  const addToHistory = (stockSymbol: string, price?: number, name?: string) => {
    const newItem: SearchHistoryItem = { symbol: stockSymbol, timestamp: Date.now(), price, companyName: name };
    setSearchHistory(prevHistory => {
      const filteredHistory = prevHistory.filter(item => item.symbol !== stockSymbol);
      const newHistory = [newItem, ...filteredHistory].slice(0, 20);
      try {
        localStorage.setItem('stockSearchHistory', JSON.stringify(newHistory));
      } catch (e) {
        console.error('Failed to save search history to localStorage:', e);
      }
      return newHistory;
    });
  };

  const clearHistory = () => {
    try {
      localStorage.removeItem('stockSearchHistory');
      setSearchHistory([]);
    } catch (e) {
      console.error('Failed to clear search history from localStorage:', e);
    }
  };

  // ─── Market status in ET (identical to classic) ────────────────────────────
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

      return (weekdayIndex === 0 || weekdayIndex === 6)
        ? 'CLOSED'
        : (currentMinutes < openMinutes ? 'PRE' : currentMinutes >= closeMinutes ? 'POST' : 'REGULAR');
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

  // ─── ML training runner (identical to classic incl. cache + accuracy log) ──
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

        const basePredictions = {
          linearRegression: mlLibs.generateLinearRegression(data, horizon),
          ema: mlLibs.generateEMAForecast(data, horizon),
          arima: mlLibs.generateARIMAForecast(data, horizon),
          prophetLite: mlLibs.generateProphetLiteForecast(data, horizon),
        };
        setMlPredictions(basePredictions);

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

        const allPredictions = {
          ...basePredictions,
          ...(lstm && { lstm }),
          ...(gru && { gru }),
          ...(cnnLstm && { cnnLstm }),
          ...(ensemble && { ensemble }),
        };
        savePredictionsToCache(sym, allPredictions, horizon);

        const basePrice = data.length > 0 ? data[data.length - 1].close : 0;
        logPredictionRun(sym, basePrice, horizon, allPredictions);

        setMlTraining(false);
      } catch (mlError) {
        console.error('ML algorithms error:', mlError);
        setMlTraining(false);
      }
    }, delay);
  };

  // ─── Data pipeline (identical to classic fetchData) ────────────────────────
  const fetchData = async (
    stockSymbol: string,
    options: FetchDataOptions = {}
  ) => {
    const { forceRecalc = false, skipMLCalculations = false, frequencyId, chartOnly = false } = options;
    const targetFrequency = getFrequencyOption(frequencyId ?? dataFrequencyId);
    const intervalParam = targetFrequency.interval;
    const rangeDays = targetFrequency.days;

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
      setMarketState(getMarketStatus());
      setDataInterval(stockResult.interval || intervalParam);

      let massiveCompanyInfo: any = {};
      try {
        const companyResponse = await fetch(`/api/company?symbol=${stockSymbol}`);
        if (companyResponse.ok) {
          const companyResult = await companyResponse.json();
          if (companyResult.success && companyResult.companyInfo) {
            massiveCompanyInfo = companyResult.companyInfo;
            if (companyResult.companyInfo.name) {
              setCompanyName(companyResult.companyInfo.name);
            }
          }
        }
      } catch (companyError) {
        console.error('Error fetching Massive company info:', companyError);
      }

      setCompanyInfo({
        ...(stockResult.companyInfo || {}),
        ...massiveCompanyInfo,
        change: typeof stockResult.change !== 'undefined' ? stockResult.change : null,
        changePercent: typeof stockResult.changePercent !== 'undefined' ? stockResult.changePercent : null,
      });

      const cachedFundamentals = localStorage.getItem(`fundamentals_${stockSymbol}`);
      const cachedTime = localStorage.getItem(`fundamentals_time_${stockSymbol}`);

      if (cachedFundamentals && cachedTime) {
        const age = Date.now() - parseInt(cachedTime);
        if (age < 24 * 60 * 60 * 1000) {
          setFundamentalsData(JSON.parse(cachedFundamentals));
        } else {
          fetchFundamentals(stockSymbol);
        }
      } else {
        fetchFundamentals(stockSymbol);
      }

      const nameForHistory = massiveCompanyInfo?.name || stockResult.companyInfo?.name || stockResult.symbol;
      addToHistory(stockSymbol, price, nameForHistory);

      const indicators = calculateAllIndicators(stockResult.data);

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

      setTimeout(async () => {
        try {
          const simpleForecast = generateForecast(stockResult.data, forecastHorizon);
          const simpleInsights = getForecastInsights(
            stockResult.currentPrice || stockResult.data[stockResult.data.length - 1].close,
            simpleForecast
          );

          setForecastData(simpleForecast);
          setForecastInsights(simpleInsights);

          try {
            const { generateProphetWithChangepoints } = await loadMLLibraries();
            const prophetForecast = generateProphetWithChangepoints(stockResult.data, forecastHorizon, 5, mlSettings);
            setProphetForecastData(prophetForecast);
          } catch (prophetError) {
            console.error('Prophet forecast error:', prophetError);
          }

          const signal = generateTradingSignal(
            stockResult.data,
            indicators,
            simpleInsights?.mediumTerm.change
          );
          setTradingSignal(signal);
        } catch (forecastError) {
          console.error('Forecast error:', forecastError);
          const signal = generateTradingSignal(stockResult.data, indicators);
          setTradingSignal(signal);
        }
      }, 100);

      if (skipMLCalculations) return;

      const cached = !forceRecalc ? getCachedPredictions(stockSymbol, forecastHorizon) : null;

      if (cached && !forceRecalc) {
        setMlPredictions(cached.predictions);
        setMlFromCache(true);
        setMlTraining(false);
      } else {
        runMLTraining(stockResult.data, stockSymbol, forecastHorizon, mlSettings, 2000);
      }

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

  useEffect(() => {
    fetchData(symbol);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update forecast when horizon or ML settings change (identical to classic)
  useEffect(() => {
    if (stockData.length > 0 && !isLoadingFromCacheTable.current) {
      const updateForecast = async () => {
        try {
          const simpleForecast = generateForecast(stockData, forecastHorizon);
          const simpleInsights = getForecastInsights(currentPrice, simpleForecast);

          setForecastData(simpleForecast);
          setForecastInsights(simpleInsights);

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [forecastHorizon, mlSettings]);

  const fetchFundamentals = async (stockSymbol: string) => {
    setFundamentalsLoading(true);
    try {
      const response = await fetch(`/api/fundamentals?symbol=${stockSymbol}`);
      if (response.ok) {
        const data = await response.json();
        setFundamentalsData(data);
        localStorage.setItem(`fundamentals_${stockSymbol}`, JSON.stringify(data));
        localStorage.setItem(`fundamentals_time_${stockSymbol}`, Date.now().toString());
      } else {
        const err = await response.json();
        console.error('Fundamentals fetch error:', err);
        setFundamentalsData(null);
      }
    } catch (err) {
      console.error('Error fetching fundamentals:', err);
      setFundamentalsData(null);
    } finally {
      setFundamentalsLoading(false);
    }
  };

  // ─── Finviz news merge (identical to classic) ──────────────────────────────
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

  // ─── Search / navigation helpers ───────────────────────────────────────────
  const handleSearch = () => {
    if (inputSymbol.trim()) {
      fetchData(inputSymbol.trim().toUpperCase());
    }
  };

  // Cross-panel symbol jump: any panel picking a ticker returns to Overview.
  const loadSymbol = (ticker: string) => {
    setInputSymbol(ticker);
    setActiveSection('overview');
    fetchData(ticker);
  };

  const getTickerFromAPi = async (e: string) => {
    try {
      const response = await fetch(`/api/search?q=${encodeURIComponent(e)}`);
      const data = await response.json();
      setSuggestions(data.quotes.map((item: any) => `${item.symbol} , ${item.shortname}`));
    } catch (err) {
      console.error('Error fetching tickers:', err);
      setSuggestions([]);
    }
  };

  const handleLoadCachedPrediction = async (cachedPred: CachedPrediction) => {
    isLoadingFromCacheTable.current = true;
    try {
      if (cachedPred.symbol !== symbol) {
        await fetchData(cachedPred.symbol, { skipMLCalculations: true });
      }
      if (cachedPred.forecastHorizon !== forecastHorizon) {
        setForecastHorizon(cachedPred.forecastHorizon);
      }
      setMlPredictions(cachedPred.predictions);
      setMlFromCache(true);
      setMlTraining(false);
    } finally {
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

  // "/" focuses the command search from anywhere
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key !== '/') return;
      const target = e.target as HTMLElement | null;
      const tag = target?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target?.isContentEditable) return;
      e.preventDefault();
      searchInputRef.current?.focus();
      searchInputRef.current?.select();
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, []);

  // Market status refresh + live price polling (identical to classic)
  useEffect(() => {
    const interval = setInterval(() => {
      setMarketState(getMarketStatus());
    }, 60000);
    setMarketState(getMarketStatus());
    return () => clearInterval(interval);
  }, []);

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

    poll();
    const ms = getMarketStatus() === 'REGULAR' ? 15000 : 60000;
    const id = setInterval(poll, ms);
    return () => clearInterval(id);
  }, [symbol]);

  // ─── Derived hero values ───────────────────────────────────────────────────
  const change = companyInfo?.change ?? null;
  const changePercent = companyInfo?.changePercent ?? null;
  const changeUp = typeof changePercent === 'number' ? changePercent >= 0 : null;
  const activeMeta = SECTIONS.find(s => s.id === activeSection)!;

  // ─── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="pro-shell">
      {/* ══ Sidebar ══ */}
      <aside className="pro-side">
        <div className="pro-brand">
          {/* Same mark as the browser tab icon */}
          <span className="pro-brand-mark">
            <img src="/icon.svg" alt="" width={24} height={24} />
          </span>
          <span className="pro-brand-name">Stock Predictor</span>
        </div>

        <nav className="pro-nav" aria-label="Workspaces">
          {SECTIONS.map(({ id, label, hint, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={`pro-nav-item${activeSection === id ? ' is-active' : ''}`}
              onClick={() => setActiveSection(id)}
              aria-current={activeSection === id ? 'page' : undefined}
            >
              <Icon className="pro-nav-icon" />
              <span className="pro-nav-text">
                <span className="pro-nav-label">{label}</span>
                <span className="pro-nav-hint">{hint}</span>
              </span>
            </button>
          ))}
        </nav>

        <div className="pro-side-footer">
          <a
            href="https://github.com/sankeer28/stock-predictor"
            target="_blank"
            rel="noopener noreferrer"
            className="pro-side-link"
          >
            <Github /> GitHub
          </a>
        </div>
      </aside>

      {/* ══ Main column ══ */}
      <div className="pro-main">
        {/* ── Topbar: command search ── */}
        <header className="pro-topbar">
          <div className="pro-search">
            <Search className="pro-search-icon" />
            <input
              ref={searchInputRef}
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
              placeholder="Search any ticker — AAPL, NVDA, TSLA…"
              className="pro-search-input font-mono"
              aria-label="Search ticker"
            />
            <kbd className="pro-search-kbd">/</kbd>
            {suggestions.length > 0 && (
              <div className="pro-popover pro-suggest" role="listbox">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    type="button"
                    className="pro-popover-item font-mono"
                    onClick={() => {
                      const sym = suggestion.split(' ,')[0];
                      setInputSymbol(sym);
                      setSuggestions([]);
                      setActiveSection('overview');
                      fetchData(sym);
                    }}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            )}
          </div>

          <button
            type="button"
            onClick={handleSearch}
            disabled={loading}
            className="pro-analyze-btn"
          >
            {loading ? <Loader2 className="animate-spin" /> : <Sparkles />}
            <span>{loading ? 'Loading…' : 'Analyze'}</span>
          </button>

          {searchHistory.length > 0 && (
            <div className="pro-history" ref={historyRef}>
              <button
                type="button"
                className="pro-icon-btn"
                onClick={() => setShowHistory(prev => !prev)}
                title="Recent searches"
                aria-label={`Recent searches (${searchHistory.length})`}
              >
                <Clock />
              </button>
              {showHistory && (
                <div className="pro-popover pro-history-menu">
                  <div className="pro-popover-head">
                    <span>Recent searches</span>
                    <button type="button" className="pro-popover-clear" onClick={clearHistory}>
                      <X /> Clear
                    </button>
                  </div>
                  {searchHistory.map((item, index) => (
                    <button
                      key={index}
                      type="button"
                      className={`pro-popover-item${item.symbol === symbol ? ' is-current' : ''}`}
                      onClick={() => {
                        setShowHistory(false);
                        loadSymbol(item.symbol);
                      }}
                    >
                      <span className="font-mono pro-popover-sym">{item.symbol}</span>
                      <span className="pro-popover-name">{item.companyName || ''}</span>
                      <span className="pro-popover-date">{new Date(item.timestamp).toLocaleDateString()}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}
        </header>

        {/* ── Mobile workspace nav ── */}
        <nav className="pro-mobilenav" aria-label="Workspaces">
          {SECTIONS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              className={`pro-mobilenav-item${activeSection === id ? ' is-active' : ''}`}
              onClick={() => setActiveSection(id)}
            >
              <Icon /> {label}
            </button>
          ))}
        </nav>

        {/* ── Symbol hero ── */}
        <section className="pro-hero">
          <div className="pro-hero-id">
            <div className="pro-hero-symbol font-mono">{symbol}</div>
            <div className="pro-hero-company">{companyName || '—'}</div>
          </div>

          <div className="pro-hero-price">
            <span className="pro-hero-value font-mono">
              {currentPrice > 0 ? `$${currentPrice.toFixed(2)}` : '—'}
            </span>
            {typeof changePercent === 'number' && (
              <span className={`pro-hero-change ${changeUp ? 'is-up' : 'is-down'}`}>
                {changeUp ? <TrendingUp /> : <TrendingDown />}
                {typeof change === 'number' ? `${change >= 0 ? '+' : ''}${change.toFixed(2)}` : ''}
                {' '}({changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%)
              </span>
            )}
          </div>

          <div className="pro-hero-right">
            <div className={`pro-status ${marketState === 'REGULAR' ? 'is-open' : ''}`}>
              <span className="pro-status-dot" />
              {marketState === 'REGULAR' ? 'Market open' : marketState === 'PRE' ? 'Pre-market' : marketState === 'POST' ? 'After hours' : 'Market closed'}
              <span className="pro-status-time">{getETTimeString()} ET</span>
            </div>
            <div className="pro-seg" role="group" aria-label="Data frequency">
              {DATA_FREQUENCY_OPTIONS.map(option => (
                <button
                  key={option.id}
                  type="button"
                  title={option.description}
                  className={`pro-seg-btn${dataFrequencyId === option.id ? ' is-active' : ''}`}
                  onClick={() => handleFrequencyChange(option.id)}
                  disabled={loading || chartRefreshing}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        </section>

        {/* ── Error / loading states ── */}
        {error && (
          <div className="pro-error" role="alert">
            <AlertCircle />
            <div>
              <strong>Couldn&apos;t load {inputSymbol || symbol}</strong>
              <p>{error}</p>
            </div>
          </div>
        )}

        {loading && (
          <div className="pro-loading">
            <div className="pro-skeleton pro-skeleton-chart" />
            <div className="pro-skeleton-row">
              <div className="pro-skeleton" />
              <div className="pro-skeleton" />
              <div className="pro-skeleton" />
            </div>
            <div className="pro-loading-note">
              <Loader2 className="animate-spin" /> Crunching {inputSymbol || symbol} market data…
            </div>
          </div>
        )}

        {!loading && stockData.length > 0 && (
          <div className="pro-sections">
            <div className="pro-section-head">
              <activeMeta.icon className="pro-section-head-icon" />
              <div>
                <h2 className="pro-section-title">{activeMeta.label}</h2>
                <p className="pro-section-sub">{activeMeta.hint}</p>
              </div>
            </div>

            {/* ══ OVERVIEW ══ */}
            <section className="pro-section" hidden={activeSection !== 'overview'}>
              <div className="pro-chart-card">
                <div className="pro-toolbar">
                  <div className="pro-toolbar-group">
                    {finvizCharts && (
                      <button
                        type="button"
                        onClick={() => setShowFinvizChart(v => !v)}
                        className={`pro-seg-btn pro-seg-solo${showFinvizChart ? ' is-active' : ''}`}
                        title="Toggle Finviz static charts"
                      >
                        <BarChart2 /> Finviz
                      </button>
                    )}
                    {showFinvizChart && finvizCharts ? (
                      <div className="pro-seg">
                        {([
                          ['dailyCandle', '6M', '~6 months · daily candles'],
                          ['weeklyCandle', '2Y', '~2 years · weekly candles'],
                          ['monthlyCandle', 'All', '10+ years · monthly candles'],
                          ['dailyLine', 'Line', '~6 months · daily line'],
                        ] as const).map(([id, label, tip]) => (
                          <button
                            key={id}
                            type="button"
                            onClick={() => setActiveFinvizChart(id)}
                            title={tip}
                            className={`pro-seg-btn${activeFinvizChart === id ? ' is-active' : ''}`}
                          >
                            {label}
                          </button>
                        ))}
                      </div>
                    ) : (
                      <>
                        <div className="pro-seg">
                          <button
                            type="button"
                            onClick={() => handleChartTypeChange('line')}
                            className={`pro-seg-btn${chartType === 'line' ? ' is-active' : ''}`}
                          >
                            <TrendingUp /> Line
                          </button>
                          <button
                            type="button"
                            onClick={() => handleChartTypeChange('candlestick')}
                            className={`pro-seg-btn${chartType === 'candlestick' ? ' is-active' : ''}`}
                          >
                            <BarChart2 /> Candles
                          </button>
                          <button
                            type="button"
                            onClick={() => setUseLightweightChart(v => !v)}
                            className={`pro-seg-btn${useLightweightChart ? ' is-active' : ''}`}
                            title="Toggle TradingView lightweight-charts renderer"
                          >
                            TV
                          </button>
                        </div>
                        <div className="pro-seg">
                          <button
                            type="button"
                            onClick={() => setUseProphetForecast(false)}
                            className={`pro-seg-btn${!useProphetForecast ? ' is-active' : ''}`}
                          >
                            <Brain /> ML
                          </button>
                          <button
                            type="button"
                            onClick={() => setUseProphetForecast(true)}
                            className={`pro-seg-btn${useProphetForecast ? ' is-active' : ''}`}
                          >
                            <Sparkles /> Prophet
                          </button>
                        </div>
                        <label className="pro-days">
                          Horizon
                          <input
                            type="number"
                            min="7"
                            max="90"
                            value={forecastHorizon}
                            onChange={(e) => setForecastHorizon(parseInt(e.target.value) || 30)}
                            className="font-mono"
                            aria-label="Forecast horizon in days"
                          />
                          d
                        </label>
                        {chartData.length > 0 && (
                          <button
                            type="button"
                            onClick={() => exportToCSV(symbol, chartData)}
                            className="pro-seg-btn pro-seg-solo"
                            title="Export chart data to CSV"
                          >
                            <Download /> CSV
                          </button>
                        )}
                      </>
                    )}
                  </div>

                  {!showFinvizChart && (
                    <div className="pro-toolbar-group pro-overlays">
                      {([
                        { label: 'MA20', checked: showMA20, set: setShowMA20 },
                        { label: 'MA50', checked: showMA50, set: setShowMA50 },
                        { label: 'Bollinger', checked: showBB, set: setShowBB },
                        { label: 'Volume', checked: showVolume, set: setShowVolume },
                        { label: 'RSI/MACD', checked: showIndicators, set: setShowIndicators },
                        { label: 'Patterns', checked: showPatterns, set: setShowPatterns },
                        { label: 'Fibonacci', checked: showFibonacci, set: setShowFibonacci },
                      ] as const).map(({ label, checked, set }) => (
                        <label key={label} className={`pro-check${checked ? ' is-on' : ''}`}>
                          <input
                            type="checkbox"
                            checked={checked}
                            onChange={(e) => { const v = e.target.checked; startTransition(() => set(v)); }}
                          />
                          {label}
                        </label>
                      ))}
                    </div>
                  )}
                </div>

                {showFinvizChart && finvizCharts ? (
                  <img
                    src={finvizCharts[activeFinvizChart]}
                    alt={`${symbol} ${activeFinvizChart} Finviz chart`}
                    className="pro-finviz-img"
                  />
                ) : (
                  <div className="pro-chart-body">
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
                        showFibonacci={showFibonacci}
                        freqOptions={DATA_FREQUENCY_OPTIONS.map(o => ({ id: o.id, label: o.label, description: o.description }))}
                        activeFreqId={dataFrequencyId}
                        onFreqChange={(id) => handleFrequencyChange(id as typeof dataFrequencyId)}
                        freqLoading={loading}
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
                      <div className="pro-chart-refresh">
                        <Loader2 className="animate-spin" />
                      </div>
                    )}
                  </div>
                )}
              </div>

              {forecastInsights && (
                <div className="pro-stat-row">
                  <div className="pro-stat">
                    <span className="pro-stat-label">7-day forecast</span>
                    <span className="pro-stat-value font-mono">${forecastInsights.shortTerm.price.toFixed(2)}</span>
                    <span className={`pro-stat-delta ${forecastInsights.shortTerm.change > 0 ? 'is-up' : 'is-down'}`}>
                      {forecastInsights.shortTerm.change > 0 ? '+' : ''}{forecastInsights.shortTerm.change.toFixed(2)}%
                    </span>
                  </div>
                  <div className="pro-stat">
                    <span className="pro-stat-label">30-day forecast</span>
                    <span className="pro-stat-value font-mono">${forecastInsights.mediumTerm.price.toFixed(2)}</span>
                    <span className={`pro-stat-delta ${forecastInsights.mediumTerm.change > 0 ? 'is-up' : 'is-down'}`}>
                      {forecastInsights.mediumTerm.change > 0 ? '+' : ''}{forecastInsights.mediumTerm.change.toFixed(2)}%
                    </span>
                  </div>
                  <div className="pro-stat">
                    <span className="pro-stat-label">Trend</span>
                    <span className="pro-stat-value pro-stat-cap">{forecastInsights.trend.direction}</span>
                    <span className="pro-stat-delta">strength {forecastInsights.trend.strength.toFixed(0)}%</span>
                  </div>
                </div>
              )}

              <div className="pro-grid pro-grid-2">
                {tradingSignal && (
                  <ErrorBoundary label="Trading Signals">
                    <TradingSignals signal={tradingSignal} currentPrice={currentPrice} />
                  </ErrorBoundary>
                )}
                {companyInfo && (
                  <ErrorBoundary label="Company Info">
                    <CompanyInfo
                      symbol={symbol}
                      companyName={companyName}
                      currentPrice={currentPrice}
                      currentChange={companyInfo?.change ?? undefined}
                      currentChangePercent={companyInfo?.changePercent ?? undefined}
                      companyInfo={companyInfo}
                      fundamentalsData={fundamentalsData}
                      fundamentalsLoading={fundamentalsLoading}
                      finvizStock={finvizStock}
                    />
                  </ErrorBoundary>
                )}
              </div>

              {/* Also feeds finviz fundamentals/targets/news into shared state.
                  Sections hide via CSS (stay mounted), so callbacks keep firing
                  while other workspaces are active. */}
              <ErrorBoundary label="Finviz Snapshot">
                <FinvizPanel
                  symbol={symbol}
                  onStockData={setFinvizStock}
                  onAnalystTargets={setFinvizAnalystTargets}
                  onNewsData={handleFinvizNews}
                  onChartsData={(charts, links) => { setFinvizCharts(charts); setFinvizLinks(links); }}
                />
              </ErrorBoundary>
            </section>

            {/* ══ PREDICTIONS ══ */}
            <section className="pro-section" hidden={activeSection !== 'ai'}>
              {/* Strict 50/50: model settings left, AI analysis right */}
              <div className="pro-grid pro-grid-half">
                <ErrorBoundary label="ML Predictions">
                  <MLPredictions
                    currentPrice={currentPrice}
                    predictions={mlPredictions}
                    isTraining={mlTraining}
                    fromCache={mlFromCache}
                    onRecalculate={() => fetchData(symbol, { forceRecalc: true })}
                    onLoadPrediction={handleLoadCachedPrediction}
                    mlSettings={mlSettings}
                    onSettingsChange={handleMlSettingsChange}
                    onPresetChange={handleMlPresetChange}
                    currentPreset={mlPreset}
                  />
                </ErrorBoundary>
                <ErrorBoundary label="AI Analysis">
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
                </ErrorBoundary>
              </div>
              <ErrorBoundary label="Live Prediction Lab">
                <LivePredictionChart symbol={symbol} />
              </ErrorBoundary>
              <ErrorBoundary label="Prediction Scorecard">
                <PredictionScorecard symbol={symbol} />
              </ErrorBoundary>
              <ErrorBoundary label="Signal Backtest">
                <SignalBacktest symbol={symbol} />
              </ErrorBoundary>
            </section>

            {/* ══ TECHNICALS ══ */}
            <section className="pro-section" hidden={activeSection !== 'technicals'}>
              <div className="pro-grid pro-grid-2">
                <div className="pro-panel">
                  <h3 className="pro-panel-title">RSI — Relative Strength Index</h3>
                  <TechnicalIndicatorsChart data={chartData} indicator="rsi" />
                </div>
                <div className="pro-panel">
                  <h3 className="pro-panel-title">MACD</h3>
                  <TechnicalIndicatorsChart data={chartData} indicator="macd" />
                </div>
              </div>
              <div className="pro-grid pro-grid-2">
                <ErrorBoundary label="Volume Profile">
                  <VolumeProfile chartData={chartData} currentPrice={currentPrice} />
                </ErrorBoundary>
                <ErrorBoundary label="Daily Return Heatmap">
                  <DailyReturnHeatmap chartData={chartData} />
                </ErrorBoundary>
              </div>
              {showPatterns && (
                <ErrorBoundary label="Pattern Analysis">
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
                </ErrorBoundary>
              )}
            </section>

            {/* ══ MARKET INTEL ══ */}
            <section className="pro-section" hidden={activeSection !== 'intel'}>
              <div className="pro-grid pro-grid-2">
                <ErrorBoundary label="Analyst Recommendations">
                  <AnalystRecommendations symbol={symbol} finvizTargets={finvizAnalystTargets} />
                </ErrorBoundary>
                <ErrorBoundary label="Peer Stocks">
                  <PeerStocks symbol={symbol} onPeerClick={loadSymbol} />
                </ErrorBoundary>
                <ErrorBoundary label="Insider Transactions">
                  <InsiderTransactions symbol={symbol} />
                </ErrorBoundary>
                <ErrorBoundary label="Congressional Trading">
                  <CongressionalTrading symbol={symbol} />
                </ErrorBoundary>
                <ErrorBoundary label="Earnings Calendar">
                  <EarningsCalendar symbol={symbol} />
                </ErrorBoundary>
                <ErrorBoundary label="Earnings History">
                  <EarningsHistory symbol={symbol} />
                </ErrorBoundary>
              </div>
              <ErrorBoundary label="Options Chain">
                <OptionsChain symbol={symbol} />
              </ErrorBoundary>
              <ErrorBoundary label="Correlation Heatmap">
                <CorrelationHeatmap symbol={symbol} />
              </ErrorBoundary>
              <ErrorBoundary label="Economic Calendar">
                <EconomicCalendar />
              </ErrorBoundary>
            </section>

            {/* ══ NEWS & SOCIAL ══ */}
            <section className="pro-section" hidden={activeSection !== 'news'}>
              <ErrorBoundary label="News">
                <NewsPanel
                  articles={newsArticles}
                  sentiments={newsSentiments}
                  isAnalyzingSentiment={isAnalyzingSentiment}
                  finvizNewsLoading={finvizNewsLoading}
                />
              </ErrorBoundary>
              <div className="pro-grid pro-grid-2">
                <ErrorBoundary label="Reddit Sentiment">
                  <RedditSentiment onTickerClick={loadSymbol} />
                </ErrorBoundary>
                <ErrorBoundary label="ApeWisdom Mentions">
                  <ApeWisdomMentions onTickerClick={loadSymbol} />
                </ErrorBoundary>
              </div>
            </section>

            {/* ══ MARKETS ══ */}
            <section className="pro-section" hidden={activeSection !== 'markets'}>
              <div className="pro-grid pro-grid-2">
                <ErrorBoundary label="Market Movers">
                  <MarketMovers onTickerClick={loadSymbol} />
                </ErrorBoundary>
                <ErrorBoundary label="Fear & Greed Index">
                  <FearGreedIndex />
                </ErrorBoundary>
              </div>
              <ErrorBoundary label="Stock Screener">
                <StockScreener onSelectTicker={loadSymbol} />
              </ErrorBoundary>
            </section>

            {/* ══ WATCHLIST & TOOLS ══ */}
            <section className="pro-section" hidden={activeSection !== 'tools'}>
              <div className="pro-grid pro-grid-2">
                <ErrorBoundary label="Watchlist">
                  <Watchlist currentSymbol={symbol} onSymbolClick={loadSymbol} />
                </ErrorBoundary>
                <ErrorBoundary label="Price Alerts">
                  <PriceAlerts symbol={symbol} currentPrice={currentPrice} />
                </ErrorBoundary>
              </div>
            </section>

            {/* ══ PORTFOLIO ══ */}
            <section className="pro-section" hidden={activeSection !== 'portfolio'}>
              {portfolioMounted && (
                <ErrorBoundary label="Portfolio">
                  <PortfolioWorkspace />
                </ErrorBoundary>
              )}
            </section>
          </div>
        )}
      </div>
    </div>
  );
}
