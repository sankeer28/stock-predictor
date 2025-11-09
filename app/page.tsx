'use client';

import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, Loader2, AlertCircle, Github } from 'lucide-react';
import StockChart from '@/components/StockChart';
import TechnicalIndicatorsChart from '@/components/TechnicalIndicatorsChart';
import NewsPanel from '@/components/NewsPanel';
import TradingSignals from '@/components/TradingSignals';
import Sidebar, { SearchHistoryItem } from '@/components/Sidebar';
import CompanyInfo from '@/components/CompanyInfo';
import MLPredictions from '@/components/MLPredictions';
import { calculateAllIndicators } from '@/lib/technicalIndicators';
import { generateForecast, getForecastInsights } from '@/lib/forecasting';
import { generateMLForecast, getMLForecastInsights, MLForecast } from '@/lib/mlForecasting';
import {
  generateLinearRegression,
  generatePolynomialRegression,
  generateMovingAverageForecast,
  generateEMAForecast,
  MLPrediction
} from '@/lib/mlAlgorithms';
import { analyzeSentiment } from '@/lib/sentiment';
import { generateTradingSignal } from '@/lib/tradingSignals';
import { StockData, NewsArticle, ChartDataPoint } from '@/types';

export default function Home() {
  const [symbol, setSymbol] = useState('AAPL');
  const [inputSymbol, setInputSymbol] = useState('AAPL');
  const [days, setDays] = useState(365);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const [stockData, setStockData] = useState<StockData[]>([]);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [companyName, setCompanyName] = useState<string>('');
  const [marketState, setMarketState] = useState<string>('');
  const [companyInfo, setCompanyInfo] = useState<any>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [forecastData, setForecastData] = useState<any[]>([]);
  const [newsArticles, setNewsArticles] = useState<NewsArticle[]>([]);
  const [newsSentiments, setNewsSentiments] = useState<any[]>([]);
  const [tradingSignal, setTradingSignal] = useState<any>(null);
  const [forecastInsights, setForecastInsights] = useState<any>(null);
  const [searchHistory, setSearchHistory] = useState<SearchHistoryItem[]>([]);

  // Chart display options
  const [showMA20, setShowMA20] = useState(true);
  const [showMA50, setShowMA50] = useState(true);
  const [showMA200, setShowMA200] = useState(false);
  const [showBB, setShowBB] = useState(false);
  const [showIndicators, setShowIndicators] = useState(true);
  const [forecastHorizon, setForecastHorizon] = useState(30);

  // ML predictions state
  const [mlPredictions, setMlPredictions] = useState<{
    lstm?: MLForecast[];
    linearRegression?: MLPrediction[];
    polynomialRegression?: MLPrediction[];
    movingAverage?: MLPrediction[];
    ema?: MLPrediction[];
  }>({});
  const [mlTraining, setMlTraining] = useState(false);

  // Load search history from localStorage on mount
  useEffect(() => {
    try {
      const savedHistory = localStorage.getItem('stockSearchHistory');
      if (savedHistory) {
        const parsed = JSON.parse(savedHistory);
        console.log('Loaded search history from localStorage:', parsed);
        setSearchHistory(parsed);
      } else {
        console.log('No search history found in localStorage');
      }
    } catch (e) {
      console.error('Failed to load search history from localStorage:', e);
    }
  }, []);

  // Save search history to localStorage
  const saveSearchHistory = (history: SearchHistoryItem[]) => {
    localStorage.setItem('stockSearchHistory', JSON.stringify(history));
    setSearchHistory(history);
  };

  // Add to search history
  const addToHistory = (stockSymbol: string, price?: number) => {
    const newItem: SearchHistoryItem = {
      symbol: stockSymbol,
      timestamp: Date.now(),
      price
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

  const fetchData = async (stockSymbol: string) => {
    setLoading(true);
    setError('');

    try {
      // Fetch stock data
      const stockResponse = await fetch(`/api/stock?symbol=${stockSymbol}&days=${days}`);
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
      setMarketState(stockResult.marketState || 'UNKNOWN');
      setCompanyInfo(stockResult.companyInfo || null);

      // Add to search history
      addToHistory(stockSymbol, price);

      // Calculate technical indicators
      const indicators = calculateAllIndicators(stockResult.data);

      // Prepare chart data
      const preparedChartData: ChartDataPoint[] = stockResult.data.map((d: StockData, i: number) => ({
        date: d.date,
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

      // Generate simple forecast first (non-blocking)
      try {
        const simpleForecast = generateForecast(stockResult.data, forecastHorizon);
        const simpleInsights = getForecastInsights(
          stockResult.currentPrice || stockResult.data[stockResult.data.length - 1].close,
          simpleForecast
        );

        setForecastData(simpleForecast);
        setForecastInsights(simpleInsights);

        // Generate trading signal with simple forecast
        const signal = generateTradingSignal(
          stockResult.data,
          indicators,
          simpleInsights?.mediumTerm.change
        );
        setTradingSignal(signal);

        // Run all ML algorithms in background
        setMlTraining(true);
        setTimeout(async () => {
          try {
            // Fast algorithms (run immediately)
            const linearReg = generateLinearRegression(stockResult.data, forecastHorizon);
            const polyReg = generatePolynomialRegression(stockResult.data, forecastHorizon);
            const maForecast = generateMovingAverageForecast(stockResult.data, forecastHorizon);
            const emaForecast = generateEMAForecast(stockResult.data, forecastHorizon);

            setMlPredictions({
              linearRegression: linearReg,
              polynomialRegression: polyReg,
              movingAverage: maForecast,
              ema: emaForecast,
            });

            // LSTM (slower, train last)
            console.log('Starting LSTM training...');
            const lstmForecast = await generateMLForecast(stockResult.data, forecastHorizon);

            setMlPredictions(prev => ({
              ...prev,
              lstm: lstmForecast,
            }));

            setMlTraining(false);
            console.log('All ML algorithms completed!');
          } catch (mlError) {
            console.error('ML algorithms error:', mlError);
            setMlTraining(false);
          }
        }, 100);
      } catch (forecastError) {
        console.error('Forecast error:', forecastError);
        setMlTraining(false);
        // Generate trading signal without forecast
        const signal = generateTradingSignal(stockResult.data, indicators);
        setTradingSignal(signal);
      }

      // Fetch news
      const newsResponse = await fetch(`/api/news?symbol=${stockSymbol}`);
      if (newsResponse.ok) {
        const newsResult = await newsResponse.json();
        setNewsArticles(newsResult.articles || []);

        // Analyze sentiment for each article
        const sentiments = (newsResult.articles || []).map((article: NewsArticle) => {
          const combinedText = `${article.title} ${article.description}`;
          return analyzeSentiment(combinedText);
        });
        setNewsSentiments(sentiments);
      }

      setSymbol(stockSymbol);
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
    if (stockData.length > 0) {
      const updateForecast = async () => {
        try {
          // Always generate simple forecast first
          const simpleForecast = generateForecast(stockData, forecastHorizon);
          const simpleInsights = getForecastInsights(currentPrice, simpleForecast);

          setForecastData(simpleForecast);
          setForecastInsights(simpleInsights);

          const indicators = calculateAllIndicators(stockData);
          const simpleSignal = generateTradingSignal(
            stockData,
            indicators,
            simpleInsights?.mediumTerm.change
          );
          setTradingSignal(simpleSignal);

          // Run all ML algorithms in background
          setMlTraining(true);
          setTimeout(async () => {
            try {
              // Fast algorithms
              const linearReg = generateLinearRegression(stockData, forecastHorizon);
              const polyReg = generatePolynomialRegression(stockData, forecastHorizon);
              const maForecast = generateMovingAverageForecast(stockData, forecastHorizon);
              const emaForecast = generateEMAForecast(stockData, forecastHorizon);

              setMlPredictions({
                linearRegression: linearReg,
                polynomialRegression: polyReg,
                movingAverage: maForecast,
                ema: emaForecast,
              });

              // LSTM (slower)
              const lstmForecast = await generateMLForecast(stockData, forecastHorizon);

              setMlPredictions(prev => ({
                ...prev,
                lstm: lstmForecast,
              }));

              setMlTraining(false);
            } catch (mlError) {
              console.error('ML algorithms error:', mlError);
              setMlTraining(false);
            }
          }, 100);
        } catch (forecastError) {
          console.error('Forecast update error:', forecastError);
          setMlTraining(false);
        }
      };

      updateForecast();
    }
  }, [forecastHorizon]);

  const handleSearch = () => {
    if (inputSymbol.trim()) {
      fetchData(inputSymbol.trim().toUpperCase());
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleSearch();
    }
  };

  return (
    <main className="min-h-screen p-4" style={{ background: 'var(--bg-4)' }}>
      <div className="flex gap-4 items-start">
        {/* Left Sidebar - Search History */}
        <div className="hidden xl:block flex-shrink-0">
          <Sidebar
            searchHistory={searchHistory}
            onSelectSymbol={(sym) => {
              setInputSymbol(sym);
              fetchData(sym);
            }}
            onClearHistory={clearHistory}
            currentSymbol={symbol}
          />
        </div>

        {/* Main Content Area */}
        <div className="flex-1 min-w-0">
          {/* Header with Search */}
          <div className="card mb-4">
            <span className="card-label">Stock Market Analysis</span>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-6 h-6" style={{ color: 'var(--accent)' }} />
              <div>
                <h1 className="text-lg font-bold" style={{ color: 'var(--text-1)' }}>
                  Stock Market Analysis
                </h1>
              </div>
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
                <input
                  type="text"
                  value={inputSymbol}
                  onChange={(e) => setInputSymbol(e.target.value.toUpperCase())}
                  onKeyPress={handleKeyPress}
                  placeholder="Symbol (e.g., AAPL)"
                  className="flex-1 max-w-md px-4 py-3 border transition-all font-mono"
                  style={{
                    background: 'var(--bg-3)',
                    borderColor: 'var(--bg-1)',
                    borderLeftColor: 'var(--accent)',
                    borderLeftWidth: '3px',
                    color: 'var(--text-2)',
                    outline: 'none'
                  }}
                />
                <button
                  onClick={handleSearch}
                  disabled={loading}
                  className="px-6 py-3 font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 border"
                  style={{
                    background: 'var(--accent)',
                    borderColor: 'var(--accent)',
                    color: 'var(--text-0)'
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

              {/* Market Status */}
              {!loading && stockData.length > 0 && (
                <div className="flex items-center gap-2 px-3 py-2 border-2" style={{
                  borderColor: marketState === 'REGULAR' ? 'var(--success)' : 'var(--text-4)',
                  background: 'var(--bg-3)',
                }}>
                  <div className={`w-2 h-2 rounded-full ${marketState === 'REGULAR' ? 'animate-pulse' : ''}`} style={{
                    background: marketState === 'REGULAR' ? 'var(--success)' : 'var(--text-4)'
                  }} />
                  <span className="text-xs font-semibold" style={{
                    color: marketState === 'REGULAR' ? 'var(--success)' : 'var(--text-4)'
                  }}>
                    {marketState === 'REGULAR' ? 'OPEN' : marketState === 'CLOSED' ? 'CLOSED' : marketState === 'PRE' ? 'PRE' : marketState === 'POST' ? 'POST' : 'CLOSED'}
                  </span>
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
                  companyInfo={companyInfo}
                />
              </div>
            )}

            {/* Chart Controls */}
            <div className="card mb-6">
              <span className="card-label">Chart Controls</span>
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-4">
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showMA20}
                    onChange={(e) => setShowMA20(e.target.checked)}
                    className="w-4 h-4 accent-[var(--accent)]"
                  />
                  <span className="text-sm" style={{ color: 'var(--text-3)' }}>20-Day MA</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showMA50}
                    onChange={(e) => setShowMA50(e.target.checked)}
                    className="w-4 h-4 accent-[var(--accent)]"
                  />
                  <span className="text-sm" style={{ color: 'var(--text-3)' }}>50-Day MA</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showMA200}
                    onChange={(e) => setShowMA200(e.target.checked)}
                    className="w-4 h-4 accent-[var(--accent)]"
                  />
                  <span className="text-sm" style={{ color: 'var(--text-3)' }}>200-Day MA</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showBB}
                    onChange={(e) => setShowBB(e.target.checked)}
                    className="w-4 h-4 accent-[var(--accent)]"
                  />
                  <span className="text-sm" style={{ color: 'var(--text-3)' }}>Bollinger Bands</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showIndicators}
                    onChange={(e) => setShowIndicators(e.target.checked)}
                    className="w-4 h-4 accent-[var(--accent)]"
                  />
                  <span className="text-sm" style={{ color: 'var(--text-3)' }}>RSI/MACD</span>
                </label>
                <div className="col-span-2 sm:col-span-3 md:col-span-1">
                  <label className="text-sm block mb-1" style={{ color: 'var(--text-3)' }}>
                    Forecast Days
                  </label>
                  <input
                    type="number"
                    min="7"
                    max="90"
                    value={forecastHorizon}
                    onChange={(e) => setForecastHorizon(parseInt(e.target.value) || 30)}
                    className="w-full px-3 py-2 border font-mono text-sm"
                    style={{
                      background: 'var(--bg-3)',
                      borderColor: 'var(--bg-1)',
                      borderLeftColor: 'var(--accent)',
                      borderLeftWidth: '3px',
                      color: 'var(--text-2)',
                      outline: 'none'
                    }}
                  />
                </div>
              </div>
            </div>

            {/* Main Chart */}
            <div className="card mb-6">
              <span className="card-label">Price Chart with Forecast</span>
              <StockChart
                data={chartData}
                showMA20={showMA20}
                showMA50={showMA50}
                showMA200={showMA200}
                showBB={showBB}
                forecastData={forecastData}
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
              <NewsPanel articles={newsArticles} sentiments={newsSentiments} />
            </div>
          </>
        )}
        </div>

        {/* ML Predictions Sidebar - Right Side */}
        {!loading && stockData.length > 0 && (
          <div className="hidden xl:block flex-shrink-0">
            <MLPredictions
              currentPrice={currentPrice}
              predictions={mlPredictions}
              isTraining={mlTraining}
            />
          </div>
        )}
      </div>
    </main>
  );
}
