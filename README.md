# Multi-Algorithm Stock Predictor

## 🚀 Overview
The Multi-Algorithm Stock Predictor is an advanced stock price prediction system that leverages multiple machine learning algorithms and technical indicators to generate ensemble predictions for stock market movements. Built with Streamlit, this application combines seven different prediction models, technical analysis, and real-time news sentiment to provide comprehensive trading insights.
## ⚠️Stock price prediction is inherently difficult and no model can consistently predict market movements accurately
## ✨ Key Features
- Ensemble predictions from different algorithms
- Real-time stock data integration via yfinance
- Live news sentiment analysis
- Technical indicators visualization
- Risk assessment and confidence scoring
- Model consensus analysis
- Interactive web interface
- Customizable timeframe analysis

### Moving Average Controls
1. Use the checkboxes in the Chart Controls section to toggle 20-day and 50-day moving averages
2. SMAs help identify trends - when price crosses above an SMA it may indicate bullish momentum, below may indicate bearish momentum
3. The 20-day SMA responds faster to price changes while the 50-day shows longer-term trends

### Prophet Forecast Controls
1. Use the "Forecast Horizon" slider to adjust how far into the future the model predicts (7-365 days)
2. View the forecast line (red dashed line) and confidence interval (shaded red area)
3. Expand the "Prophet Forecast Details" section to see:
   - 7-day and 30-day price targets with expected percentage changes
   - Trend direction (upward/downward)
   - Weekly pattern information showing which day of the week historically performs best
   - Seasonal factor analysis

Note: Accuracy varies based on market conditions, volatility, and the specific stock being analyzed.

## 🛠️ Setup and Installation

### Prerequisites
```bash
pip install -r requirements.txt
```
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
