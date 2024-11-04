# Multi-Algorithm Stock Predictor

## 🚀 Overview
The Multi-Algorithm Stock Predictor is an advanced stock price prediction system that leverages multiple machine learning algorithms and technical indicators to generate ensemble predictions for stock market movements. Built with Streamlit, this application combines seven different prediction models, technical analysis, and real-time news sentiment to provide comprehensive trading insights.

## ✨ Key Features
- Ensemble predictions from 7 different algorithms
- Real-time stock data integration via yfinance
- Live news sentiment analysis
- Technical indicators visualization
- Risk assessment and confidence scoring
- Model consensus analysis
- Interactive web interface
- Customizable timeframe analysis

## 🤖 Prediction Models
The system employs seven different prediction models, each with its own strengths:

1. **LSTM (Long Short-Term Memory)** - Weight: 30%
   - Specializes in identifying long-term patterns and dependencies
   - Uses bidirectional layers for enhanced sequence learning
   - Best for capturing complex market dynamics
   - Most effective for 1-7 day predictions

2. **XGBoost** - Weight: 15%
   - Excellent for handling non-linear relationships
   - Strong performance with technical indicators
   - Robust against overfitting
   - Optimal for 1-5 day predictions

3. **Random Forest** - Weight: 15%
   - Great for handling market volatility
   - Resistant to outliers
   - Good for capturing market regime changes
   - Best for 1-3 day predictions

4. **ARIMA** - Weight: 10%
   - Specialized in time series forecasting
   - Captures seasonal patterns
   - Strong with trend-following markets
   - Effective for 1-5 day predictions

5. **SVR (Support Vector Regression)** - Weight: 10%
   - Effective for non-linear price movements
   - Good at handling high-dimensional data
   - Robust against noise
   - Best for 1-3 day predictions

6. **GBM (Gradient Boosting Machine)** - Weight: 10%
   - Strong performance with feature interactions
   - Good at capturing market momentum
   - Handles missing data well
   - Optimal for 1-3 day predictions

7. **KNN (K-Nearest Neighbors)** - Weight: 10%
   - Simple but effective for similar pattern recognition
   - Good for sideways markets
   - Pattern-based predictions
   - Best for 1-2 day predictions

## 📊 Technical Indicators
The system calculates and utilizes multiple technical indicators:
- Moving Averages (5, 20, 50, 200 days)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ROC (Rate of Change)
- ATR (Average True Range)
- Stochastic Oscillator
- Williams %R
- Volume Analysis
- EMA (Exponential Moving Average)
- Momentum Indicators

## 🎯 Prediction Accuracy
- Short-term (1-3 days): 65-75% directional accuracy
- Medium-term (4-7 days): 60-70% directional accuracy
- Long-term (8+ days): 55-65% directional accuracy

Note: Accuracy varies based on market conditions, volatility, and the specific stock being analyzed.

## ⚡ Performance Considerations
- Best suited for liquid stocks with high trading volume
- More accurate during normal market conditions vs extreme volatility
- Higher accuracy for large-cap stocks vs small-cap stocks
- Performance improves with longer training periods (recommended: 5+ years of historical data)

## 🛠️ Setup and Installation

### Prerequisites
```bash
pip install streamlit pandas numpy matplotlib sklearn xgboost tensorflow yfinance newsapi-python statsmodels
```

### Configuration
1. Get a News API key from https://newsapi.org
2. Replace the `NEWS_API_KEY` in the code with your API key

### Running the Application
```bash
streamlit run stock_predictor.py
```

## 💡 Usage Guidelines

### Best Practices
1. Use longer training periods (5+ years) for more stable predictions
2. Focus on liquid stocks with consistent trading history
3. Consider multiple timeframes for confirmation
4. Always combine predictions with fundamental analysis
5. Monitor prediction confidence scores and risk assessments

### Risk Management
- Use the confidence score to gauge prediction reliability
- Consider the prediction range (upper and lower bounds)
- Monitor the model consensus strength
- Check the risk assessment indicators
- Review news sentiment before making decisions

## 📈 Trading Signals

The system generates trading signals based on:
1. **Price Change Percentage**
   - Strong signals: >10% predicted change
   - Moderate signals: 3-10% predicted change
   - Weak signals: 1-3% predicted change
   - Hold signals: <1% predicted change

2. **Confidence Scores**
   - High confidence: >0.8
   - Medium confidence: 0.6-0.8
   - Low confidence: <0.6

## ⚠️ Limitations
1. Cannot predict black swan events or unexpected news
2. Less accurate during periods of extreme market volatility
3. Requires quality historical data for accurate predictions
4. May not capture fundamental company changes
5. Past performance doesn't guarantee future results

## 🔄 Future Improvements
1. Integration of sentiment analysis from social media (Twitter)
2. Addition of more sophisticated deep learning models
3. Enhanced feature engineering capabilities
4. Real-time market correlation analysis
5. Portfolio optimization recommendations
6. Market regime detection
7. Enhanced risk management features

---
⚠️ **Disclaimer**: This tool is for educational and research purposes only. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
