# Stock Predictor
Stock analysis with ML forecasting, technical analysis, and sentiment analysis.

## AI Models & Algorithms

### Machine Learning Forecasting
- **Deep Learning**: LSTM, GRU, CNN-LSTM (TensorFlow.js) - Neural networks for complex pattern recognition
- **Statistical**: ARIMA, Prophet-Lite - Time series forecasting with trend + seasonality detection
- **Ensemble**: Weighted combination of all models for optimal accuracy
- **Traditional**: Linear Regression, Exponential Moving Average

### Technical Analysis & Patterns
- **Indicators**: Moving Averages (20/50/200-day), RSI, MACD, Bollinger Bands
- **Pattern Detection**: Trendlines, wedges, triangles, channels, double tops/bottoms, head & shoulders
- **Chart Frequencies**: 5m, 15m, 1H, 1D, 1W, 1M intervals

### News Sentiment Analysis
- **Rule-based NLP**: Lightweight sentiment engine optimized for serverless
- **Negation Detection**: "not good", "no growth" → flips sentiment
- **Intensity Modifiers**: "very", "extremely" → boosts score strength
- **Financial Keywords**: 100+ domain-specific terms (earnings beat, stock surge, etc.)
- **Comparative Language**: "better than expected", "missed estimates"
- **Pattern Matching**: Percentage gains/losses detection

### Market Analysis
- **Trading Signals**: Buy/sell recommendations from MA crossovers, RSI, MACD, Bollinger Bands
- **Volume Analysis**: Identifies unusual trading activity
- **Momentum**: Price action and trend strength indicators

<img width="1865" height="4010" alt="image" src="https://github.com/user-attachments/assets/e1ade5ee-05e3-4c8f-a5b1-9b702e736d1b" />

### Prerequisites

- Node.js 18+ installed
- **NewsAPI key** (free from [newsapi.org](https://newsapi.org/))
- **Alpha Vantage API key** (free from [alphavantage.co](https://www.alphavantage.co/support/#api-key))
- **MASSIVE API key** (optional - for company logos and info)

### Installation

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

4. Add your NewsAPI key to `.env.local`:
```
NEWS_API_KEY=your_api_key_here
```

5. Run the development server:
```bash
npm run dev
```

6. Open [http://localhost:3000](http://localhost:3000) in your browser


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Disclaimer

This tool is for educational and research purposes only. Stock predictions are inherently uncertain and should not be considered financial advice. Always conduct your own research and consult with a financial advisor before making investment decisions.

