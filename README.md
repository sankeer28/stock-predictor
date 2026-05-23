# Stock Predictor
Stock analysis platform with ML forecasting, live AI predictions, technical analysis, sentiment analysis, and a full portfolio tracker.

![demo](https://github.com/user-attachments/assets/7d1f79fb-8e0d-46c0-9b37-bc4879e283d7)
https://stock-predictor.sankeer28.workers.dev
## Features

### Dashboard

**Fear and Greed Index**
Real-time market sentiment indicator pulled from alternative.me. Displays the current fear/greed score, classification label, and historical trend so you can gauge overall market mood at a glance.

**Market Movers**
Shows the day's top gainers, top losers, most active stocks, and major market indices with live price and percentage change data.

**Economic Calendar**
Lists upcoming high-impact US economic events (CPI, FOMC, jobs reports, etc.) with scheduled release times and prior/estimate/actual values where available.

**Stock Screener**
Filter the market using built-in presets including Most Active, Top Gainers, Top Losers, Growth Tech, and Undervalued Growth. Returns a table of matching stocks with key metrics such as price, change, volume, P/E, and market cap.

---

### Stock Analysis

**Company Info**
Displays fundamental metrics for the selected stock: P/E ratio, EPS, profit margins, return on equity, debt-to-equity, dividend yield, sector, industry, employee count, and company website.

**Interactive Stock Chart**
Candlestick or line chart with zoom, brush scrubbing, multiple time intervals (5m, 15m, 1H, 1D, 1W, 1M), and optional overlay of detected chart patterns directly on the price series.

**Technical Indicators Chart**
Separate panel showing RSI and MACD plotted over time, with overbought/oversold threshold lines and momentum interpretation labels.

**Volume Profile**
Displays a horizontal histogram of trading volume at each price level over a selected period, highlighting the Point of Control (price with the most volume activity).

**Daily Return Heatmap**
Calendar grid showing each trading day coloured by its daily return percentage. Makes seasonal patterns, volatility clusters, and strong/weak months immediately visible.

**Trading Signals**
Aggregated buy and sell signals derived from moving average crossovers, RSI levels, MACD crossovers, and Bollinger Band position. Displays a composite rating from Strong Buy to Strong Sell.

**Pattern Analysis and Pattern Panel**
Automated detection of chart patterns including trendlines, wedges, triangles, channels, double tops and bottoms, and head and shoulders formations. Generates bullish or bearish signals with configurable detection windows and signal thresholds through a dedicated settings panel.

**Options Chain**
Full options table for calls and puts across all available expiration dates. Shows strike price, last price, bid, ask, volume, open interest, implied volatility, and delta/gamma Greeks.

**Price Alerts**
Set custom price thresholds for any ticker. The app notifies you when the price crosses above or below your specified level during an active browser session.

**Earnings Calendar and Earnings History**
Upcoming earnings dates and past earnings results with beat/miss classification, actual vs. estimated EPS, and surprise percentages.

**Insider Transactions**
Recent buy and sell activity by company insiders (executives and directors), including transaction date, shares traded, price, and aggregate net buying or selling direction.

**Congressional Trading**
Tracks stock trade disclosures filed by US congressional members, showing the politician, transaction type (buy/sell), ticker, and date.

**FinViz Panel**
Pulls supplementary stock data, news headlines, analyst targets, and technical chart snapshots directly from FinViz for a quick secondary reference.

**Price Targets**
Analyst price target summary showing the mean, median, high, and low targets alongside the implied upside or downside from the current price.

**Analyst Recommendations**
Historical analyst rating distribution (Strong Buy, Buy, Hold, Sell, Strong Sell) over time, with trend indicators showing whether conviction is increasing or decreasing.

**Peer and Similar Stocks**
Lists competitor or peer companies in the same industry with their current price and daily change, enabling quick side-by-side comparison.

**Correlation Heatmap**
Colour-coded matrix showing price return correlations between a user-selected group of stocks over a custom time window. Useful for identifying diversification or concentration.

**Sector Pie Chart**
Visualises the sector breakdown of a group of stocks or a portfolio as an interactive pie chart.

---

### ML Forecasting

**ML Predictions**
Multi-model price forecasting panel supporting eight algorithms run client-side via TensorFlow.js:
- LSTM (Long Short-Term Memory neural network)
- GRU (Gated Recurrent Unit neural network)
- CNN-LSTM (convolutional feature extraction feeding into LSTM)
- ARIMA (autoregressive integrated moving average)
- Prophet-Lite (trend and seasonality decomposition)
- Linear Regression
- Exponential Moving Average
- Ensemble (weighted combination of all models)

Users can select how many days ahead to forecast, and predictions are cached locally so previous runs can be reloaded without recomputing.

**ML Settings Panel**
Allows configuration of model parameters through preset profiles (Conservative, Balanced, Aggressive, Custom) and manual controls for detection thresholds and model weights.

**Predictions Cache**
Sidebar-accessible panel that lists all previously computed ML predictions stored in browser local storage, with timestamps and one-click reload.

---

### Live AI Prediction Chart

Intraday ensemble prediction engine that runs continuously in the browser. Four sub-models (Trend, Momentum, Reversal, Mean Reversion) vote on each prediction and are weighted by their recent accuracy using exponential decay. Features used include multi-scale momentum, RSI, Bollinger Bands, MACD, range position, volatility regime, VWAP deviation, and volume ratios.

Available prediction horizons: 1 minute, 5 minutes, 15 minutes, 30 minutes, and 1 hour.

Each resolved prediction is added to a replay buffer (up to the configured buffer size) and the models are retrained on mini-batches for incremental improvement during the session. Predictions run on existing historical data even when the market is outside regular trading hours.

---

### News and Sentiment

**News Panel**
Aggregates recent news articles for the selected stock from NewsAPI. Each article is scored by a rule-based NLP sentiment engine that handles negation ("not good" reverses polarity), intensity modifiers ("very", "extremely"), 100+ financial domain keywords (earnings beat, guidance raised, etc.), and comparative language ("better than expected", "missed estimates"). Displays an overall sentiment score alongside per-article classifications.

**Reddit Sentiment**
Analyses current Reddit discussion for the selected ticker across investing subreddits, returning a bullish/bearish/neutral classification and raw mention count.

**Ape Wisdom Mentions**
Tracks the selected stock's mention count and rank on ApeWisdom across communities including r/WallStreetBets and r/stocks, with trending direction indicators.

**AI Analysis**
Sends current stock data to the Groq API (Llama model) for a comprehensive written analysis covering valuation verdict, technical outlook, key risks, key opportunities, and an overall buy/sell/hold rating.

---

### Navigation and Watchlist

**Sidebar**
Collapsible left sidebar that shows recent search history for quick symbol switching, and hosts the Predictions Cache loader.

**Watchlist**
Persistent list of saved tickers with live price and daily change percentage. Clicking any entry loads that stock immediately.

---

### Portfolio Tracker

A dedicated page (`/portfolio`) for tracking a personal stock portfolio. All data is stored in browser local storage.

**Holdings Management**
- Add positions by ticker symbol (with autocomplete search), share quantity, and average cost
- Edit or delete any existing position inline
- Ticker autocomplete powered by Yahoo Finance search

**Real-Time Quotes**
Prices and dividend data are fetched from Yahoo Finance on load and on demand via the Refresh Prices button.

**Holdings Table**
Displays each position with: ticker, company name, shares held, average cost, current price, market value, unrealised gain/loss in dollars, and gain/loss percentage. A totals row summarises the full portfolio.

**USD/CAD Market Value Conversion**
Each holding has a per-row currency toggle that converts the displayed market value and gain/loss dollar amount between USD and CAD using a live exchange rate fetched from the USDCAD=X pair on Yahoo Finance. Toggling a holding also updates the portfolio total and projections to use the converted value.

**Estimated Dividends**
Always-visible card showing estimated monthly and yearly dividend income for each position based on actual dividend payments from the trailing twelve months (sourced from Yahoo Finance chart events). Totals are aggregated across all holdings.

**Projections**
Per-position growth projections over selectable time horizons (1, 3, 5, 10, 20, 30 years). Each position can be projected independently with:
- CAGR mode: uses the 5-year historical compound annual growth rate calculated from monthly close prices
- Custom mode: user-supplied annual growth rate percentage, pre-filled with the CAGR value for reference
- Annual contribution field: additional yearly investment to include in the future value calculation
- Include dividends toggle: optionally reinvests the estimated annual dividend as an additional contribution
- Warning indicator when the historical CAGR exceeds 30%, flagging the projection as based on unusually high recent performance

A portfolio total row sums projected future values across all positions for each selected horizon.

**Share Portfolio**
Generates a shareable URL by encoding the current holdings as a base64 JSON query parameter. Recipients who open the link see the full portfolio view in read-only mode with the add, edit, and delete controls hidden.

---

## Prerequisites

- Node.js 18+
- **NewsAPI key** (free from newsapi.org) - required for the news panel
- **Groq API key** (free from console.groq.com) - required for AI analysis
- **Alpha Vantage API key** (free from alphavantage.co) - used for supplementary fundamental data
- **MASSIVAPI key** (optional) - for company logos

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sankeer28/stock-predictor.git
cd stock-predictor
```

2. Install dependencies:
```bash
npm install
```

3. Create `.env.local` file:
```bash
cp .env.local.example .env.local
```

4. Add your API keys to `.env.local`:
```
NEWS_API_KEY=your_newsapi_key_here
GROQ_API_KEY=your_groq_key_here
```

5. Run the development server:
```bash
npm run dev
```

6. Open [http://localhost:3000](http://localhost:3000) in your browser

## Contributing

Contributions are welcome. Please feel free to submit a pull request.

## Disclaimer

This tool is for educational and research purposes only. Stock predictions are inherently uncertain and should not be considered financial advice. Always conduct your own research and consult with a financial advisor before making investment decisions.
