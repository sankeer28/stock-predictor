import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from newsapi import NewsApiClient
import yfinance as yf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

st.set_page_config(page_title="Multi-Algorithm Stock Predictor", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Multi-Algorithm Stock Predictor</h1>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <p style='text-align: center; color: gray; font-size: 14px;'>
    Disclaimer: This application provides stock predictions based on algorithms and is intended for informational purposes only. 
    Predictions may not be accurate, and users are encouraged to conduct their own research and consider consulting with a 
    financial advisor before making any investment decisions. This is not financial advice, and I am not responsible for any 
    outcomes resulting from the use of this application.
    </p>
    """,
    unsafe_allow_html=True
)
# API setup
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Cache functions remain the same as in original code
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = yf.download(symbol, start=start_date, end=end_date)
    return df

@st.cache_data(ttl=3600)
def get_news_headlines(symbol):
    try:
        news = newsapi.get_everything(
            q=symbol,
            language='en',
            sort_by='relevancy',
            page_size=5
        )
        return [(article['title'], article['description'], article['url']) 
                for article in news['articles']]
    except Exception as e:
        print(f"News API error: {str(e)}")
        return []
def calculate_technical_indicators_for_summary(df):
        analysis_df = df.copy()
        
        # Calculate Moving Averages
        analysis_df['MA20'] = analysis_df['Close'].rolling(window=20).mean()
        analysis_df['MA50'] = analysis_df['Close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = analysis_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        analysis_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Volume MA
        analysis_df['Volume_MA'] = analysis_df['Volume'].rolling(window=20).mean()
        
        # Calculate Bollinger Bands
        ma20 = analysis_df['Close'].rolling(window=20).mean()
        std20 = analysis_df['Close'].rolling(window=20).std()
        analysis_df['BB_upper'] = ma20 + (std20 * 2)
        analysis_df['BB_lower'] = ma20 - (std20 * 2)
        analysis_df['BB_middle'] = ma20
        
        return analysis_df

class MultiAlgorithmStockPredictor:
    def __init__(self, symbol, training_years=5, weights=None):
        self.symbol = symbol
        self.training_years = training_years
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.weights = weights if weights is not None else WEIGHT_CONFIGURATIONS["Default"]
        
    def fetch_historical_data(self):
        # Same as original EnhancedStockPredictor
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.training_years)
        
        try:
            df = yf.download(self.symbol, start=start_date, end=end_date)
            if df.empty:
                st.warning(f"Data for the last {self.training_years} years is unavailable. Fetching maximum available data instead.")
                df = yf.download(self.symbol, period="max")
            return df
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return yf.download(self.symbol, period="max")

    # Technical indicators calculation methods remain the same
    def calculate_technical_indicators(self, df):
        # Original technical indicators remain the same
        df['MA5'] = df['Close'].rolling(window=5).mean()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['ROC'] = df['Close'].pct_change(periods=10) * 100
        df['ATR'] = self.calculate_atr(df)
        df['BB_upper'], df['BB_lower'] = self.calculate_bollinger_bands(df['Close'])
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Rate'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
        
        # Additional technical indicators
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MOM'] = df['Close'].diff(10)
        df['STOCH_K'] = self.calculate_stochastic(df)
        df['WILLR'] = self.calculate_williams_r(df)
        
        return df.dropna()
    
    
    @staticmethod
    def calculate_stochastic(df, period=14):
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        return k

    @staticmethod
    def calculate_williams_r(df, period=14):
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        return -100 * ((high_max - df['Close']) / (high_max - low_min))

    # Original calculation methods remain the same
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2
    
    @staticmethod
    def calculate_atr(df, period=14):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band

    def prepare_data(self, df, seq_length=60):
        feature_columns = ['Close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                          'ROC', 'ATR', 'BB_upper', 'BB_lower', 'Volume_Rate',
                          'EMA12', 'EMA26', 'MOM', 'STOCH_K', 'WILLR']
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Prepare sequences for LSTM
        X_lstm, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X_lstm.append(scaled_data[i-seq_length:i])
            y.append(scaled_data[i, 0])  # 0 index represents Close price
            
        # Prepare data for other models
        X_other = scaled_data[seq_length:]
        
        return np.array(X_lstm), X_other, np.array(y)

    def build_lstm_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def train_arima(self, df):
        model = ARIMA(df['Close'], order=(5,1,0))
        return model.fit()

    def predict_with_all_models(self, prediction_days=30, sequence_length=60):
        try:
            # Fetch and prepare data
            df = self.fetch_historical_data()
            df = self.calculate_technical_indicators(df)
            
            X_lstm, X_other, y = self.prepare_data(df, sequence_length)
            
            # Split data
            split_idx = int(len(y) * 0.8)
            X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
            X_other_train, X_other_test = X_other[:split_idx], X_other[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            predictions = {}
            
            # Train and predict with LSTM
            lstm_model = self.build_lstm_model((sequence_length, X_lstm.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lstm_model.fit(X_lstm_train, y_train, epochs=50, batch_size=32,
                          validation_data=(X_lstm_test, y_test),
                          callbacks=[early_stopping], verbose=0)
            lstm_pred = lstm_model.predict(X_lstm_test[-1:], verbose=0)[0][0]
            predictions['LSTM'] = lstm_pred

            # Train and predict with SVR
            svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
            svr_model.fit(X_other_train, y_train)
            svr_pred = svr_model.predict(X_other_test[-1:])
            predictions['SVR'] = svr_pred[0]

            # Train and predict with Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_other_train, y_train)
            rf_pred = rf_model.predict(X_other_test[-1:])
            predictions['Random Forest'] = rf_pred[0]

            # Train and predict with XGBoost
            xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
            xgb_model.fit(X_other_train, y_train)
            xgb_pred = xgb_model.predict(X_other_test[-1:])
            predictions['XGBoost'] = xgb_pred[0]

            # Train and predict with KNN
            knn_model = KNeighborsRegressor(n_neighbors=5)
            knn_model.fit(X_other_train, y_train)
            knn_pred = knn_model.predict(X_other_test[-1:])
            predictions['KNN'] = knn_pred[0]

            # Train and predict with GBM
            gbm_model = GradientBoostingRegressor(random_state=42)
            gbm_model.fit(X_other_train, y_train)
            gbm_pred = gbm_model.predict(X_other_test[-1:])
            predictions['GBM'] = gbm_pred[0]

            # Train and predict with ARIMA
            try:
                close_prices = df['Close'].values
                arima_model = ARIMA(close_prices, order=(5,1,0))
                arima_fit = arima_model.fit()
                arima_pred = arima_fit.forecast(steps=1)[0]
                arima_scaled = (arima_pred - df['Close'].mean()) / df['Close'].std()
                predictions['ARIMA'] = arima_scaled
            except Exception as e:
                st.warning(f"ARIMA prediction failed: {str(e)}")

            weights = self.weights

            # Adjust weights if some models failed
            available_models = list(predictions.keys())
            total_weight = sum(weights[model] for model in available_models)
            adjusted_weights = {model: weights[model]/total_weight for model in available_models}

            ensemble_pred = sum(pred * adjusted_weights[model] 
                              for model, pred in predictions.items())
            
            # Inverse transform predictions
            dummy_array = np.zeros((1, X_other.shape[1]))
            dummy_array[0, 0] = ensemble_pred
            final_prediction = self.scaler.inverse_transform(dummy_array)[0, 0]

            # Calculate prediction range
            individual_predictions = []
            for pred in predictions.values():
                dummy = dummy_array.copy()
                dummy[0, 0] = pred
                individual_predictions.append(
                    self.scaler.inverse_transform(dummy)[0, 0]
                )
            
            std_dev = np.std(individual_predictions)
            
            return {
                'prediction': final_prediction,
                'lower_bound': final_prediction - std_dev,
                'upper_bound': final_prediction + std_dev,
                'confidence_score': 1 / (1 + std_dev / final_prediction),
                'individual_predictions': {
                    model: pred for model, pred in zip(predictions.keys(), individual_predictions)
                }
            }

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

# Streamlit interface

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL):", "AAPL")
display_days = st.slider("Select number of days to display", 30, 3650, 180)

# Define different weight configurations
WEIGHT_CONFIGURATIONS = {
    "Default": {
        'LSTM': 0.3,
        'XGBoost': 0.15,
        'Random Forest': 0.15,
        'ARIMA': 0.1,
        'SVR': 0.1,
        'GBM': 0.1,
        'KNN': 0.1
    },
    "Trend-Focused": {
        'LSTM': 0.35,
        'XGBoost': 0.20,
        'Random Forest': 0.15,
        'ARIMA': 0.10,
        'SVR': 0.08,
        'GBM': 0.07,
        'KNN': 0.05
    },
    "Statistical": {
        'LSTM': 0.20,
        'XGBoost': 0.15,
        'Random Forest': 0.15,
        'ARIMA': 0.20,
        'SVR': 0.15,
        'GBM': 0.10,
        'KNN': 0.05
    },
    "Tree-Ensemble": {
        'LSTM': 0.25,
        'XGBoost': 0.25,
        'Random Forest': 0.20,
        'ARIMA': 0.10,
        'SVR': 0.08,
        'GBM': 0.07,
        'KNN': 0.05
    },
    "Balanced": {
        'LSTM': 0.25,
        'XGBoost': 0.20,
        'Random Forest': 0.15,
        'ARIMA': 0.15,
        'SVR': 0.10,
        'GBM': 0.10,
        'KNN': 0.05
    },
    "Volatility-Focused": {
        'LSTM': 0.30,
        'XGBoost': 0.25,
        'Random Forest': 0.20,
        'ARIMA': 0.05,
        'SVR': 0.10,
        'GBM': 0.07,
        'KNN': 0.03
    }
}

WEIGHT_DESCRIPTIONS = {
    "Default": "Original configuration with balanced weights",
    "Trend-Focused": "Best for growth stocks, tech stocks, clear trend patterns",
    "Statistical": "Best for blue chip stocks, utilities, stable dividend stocks",
    "Tree-Ensemble": "Best for stocks with complex relationships to market factors",
    "Balanced": "Best for general purpose, unknown stock characteristics",
    "Volatility-Focused": "Best for small cap stocks, emerging market stocks, crypto-related stocks"
}

col1, col2 = st.columns([2, 1])

with col1:
    selected_weight = st.selectbox(
        "Select Weight Configuration:",
        options=list(WEIGHT_CONFIGURATIONS.keys()),
        help="Choose different weight configurations for the prediction models"
    )

with col2:
    st.info(WEIGHT_DESCRIPTIONS[selected_weight])

try:
    # Fetch data
    df = fetch_stock_data(symbol, display_days)
    
    # Display stock price chart
    st.subheader("Stock Price History")
    st.line_chart(df['Close'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("Generate Predictions"):
            with st.spinner("Training multiple models and generating predictions..."):
                predictor = MultiAlgorithmStockPredictor(
                    symbol, 
                    weights=WEIGHT_CONFIGURATIONS[selected_weight]
                )
                results = predictor.predict_with_all_models()
                
                if results is not None:
                    
                    
                    last_price = float(df['Close'].iloc[-1])
                    
                    st.subheader("Ensemble Prediction Results")
                    st.write(f"Current Price: ${abs(last_price):.2f}")
                    st.write(f"Predicted Price: ${abs(results['prediction']):.2f}")
                    st.write(f"Prediction Range: ${abs(results['lower_bound']):.2f} - ${abs(results['upper_bound']):.2f}")
                    st.write(f"Ensemble Confidence Score: {abs(results['confidence_score']):.2%}")
                    
                    # Individual model predictions
                    st.subheader("Individual Model Predictions")
                    model_predictions = pd.DataFrame({
                        'Model': results['individual_predictions'].keys(),
                        'Predicted Price': [abs(v) for v in results['individual_predictions'].values()]
                    })
                    model_predictions['Deviation from Ensemble'] = (
                        model_predictions['Predicted Price'] - abs(results['prediction'])
                    )
                    model_predictions = model_predictions.sort_values('Predicted Price', ascending=False)
                    st.dataframe(model_predictions.style.format({
                        'Predicted Price': '${:.2f}',
                        'Deviation from Ensemble': '${:.2f}'
                    }))
                    
                    # Trading signal with confidence
                    price_change = ((results['prediction'] - last_price) / last_price) * 100
                    
                    # Create a prediction distribution plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    predictions = list(results['individual_predictions'].values())
                    models = list(results['individual_predictions'].keys())
                    
                    # Horizontal bar chart showing predictions
                    y_pos = np.arange(len(models))
                    ax.barh(y_pos, predictions)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(models)
                    ax.axvline(x=last_price, color='r', linestyle='--', label='Current Price')
                    ax.axvline(x=results['prediction'], color='g', linestyle='--', label='Ensemble Prediction')
                    ax.set_xlabel('Price ($)')
                    ax.set_title('Model Predictions Comparison')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Trading signal box
                    signal_box = st.container()
                    if abs(price_change) > 10:  # For very large changes
                        if price_change > 0:
                            signal_box.success(f"💹 Strong BUY Signal (+{price_change:.1f}%)")
                        else:
                            signal_box.error(f"📉 Strong SELL Signal ({price_change:.1f}%)")
                    elif abs(price_change) > 3 and results['confidence_score'] > 0.8:
                        if price_change > 0:
                            signal_box.success(f"💹 BUY Signal (+{price_change:.1f}%)")
                        else:
                            signal_box.error(f"📉 SELL Signal ({price_change:.1f}%)")
                    elif abs(price_change) > 2 and results['confidence_score'] > 0.6:
                        if price_change > 0:
                            signal_box.warning(f"📈 Moderate BUY Signal (+{price_change:.1f}%)")
                        else:
                            signal_box.warning(f"📉 Moderate SELL Signal ({price_change:.1f}%)")
                    else:
                        if abs(price_change) < 1:
                            signal_box.info(f"⚖️ HOLD Signal ({price_change:.1f}%)")
                        else:
                            if price_change > 0:
                                signal_box.info(f"📈 Weak BUY Signal (+{price_change:.1f}%)")
                            else:
                                signal_box.info(f"📉 Weak SELL Signal ({price_change:.1f}%)")
                    
                    # Model consensus analysis
                    st.subheader("Model Consensus Analysis")
                    buy_signals = sum(1 for pred in predictions if pred > last_price)
                    sell_signals = sum(1 for pred in predictions if pred < last_price)
                    total_models = len(predictions)
                    
                    consensus_col1, consensus_col2, consensus_col3 = st.columns(3)
                    with consensus_col1:
                        st.metric("Buy Signals", f"{buy_signals}/{total_models}")
                    with consensus_col2:
                        st.metric("Sell Signals", f"{sell_signals}/{total_models}")
                    with consensus_col3:
                        consensus_strength = abs(buy_signals - sell_signals) / total_models
                        st.metric("Consensus Strength", f"{consensus_strength:.1%}")
                    
                    # Risk assessment
                    st.subheader("Risk Assessment")
                    prediction_std = np.std(predictions)
                    prediction_range = results['upper_bound'] - results['lower_bound']
                    risk_level = "Low" if prediction_std < last_price * 0.02 else \
                                "Medium" if prediction_std < last_price * 0.05 else "High"
                    
                    risk_col1, risk_col2 = st.columns(2)
                    with risk_col1:
                        st.metric("Prediction Volatility", f"${prediction_std:.2f}")
                    with risk_col2:
                        st.metric("Risk Level", risk_level)
    
    with col2:
        st.subheader("Latest News & Market Sentiment")
        news_headlines = get_news_headlines(symbol)
        
        if news_headlines:
            for title, description, url in news_headlines:
                with st.expander(title):
                    st.write(description)
                    st.markdown(f"[Read full article]({url})")
        else:
            st.write("No recent news available for this stock.")
        
        # Technical Analysis Summary
        st.subheader("Technical Analysis Summary")
        try:
            # Check if dataframe exists and has data
            if 'df' in locals() and isinstance(df, pd.DataFrame) and len(df) > 0:
                # Calculate technical indicators from historical data
                analysis_df = calculate_technical_indicators_for_summary(df)
                
                if len(analysis_df) >= 2:
                    latest = analysis_df.iloc[-1]
                    prev = analysis_df.iloc[-2]
                    
                    # Historical Data Analysis
                    st.write("📊 Historical Data Analysis")
                    
                    # Calculate indicator values first to avoid Series truth value ambiguity
                    ma_bullish = float(latest['MA20']) > float(latest['MA50'])
                    rsi_value = float(latest['RSI'])
                    volume_high = float(latest['Volume']) > float(latest['Volume_MA'])
                    close_price = float(latest['Close'])
                    bb_upper = float(latest['BB_upper'])
                    bb_lower = float(latest['BB_lower'])
                    
                    # Historical indicators
                    historical_indicators = {
                        "Moving Averages": {
                            "value": "Bullish" if ma_bullish else "Bearish",
                            "delta": f"{((float(latest['MA20']) - float(latest['MA50']))/float(latest['MA50']) * 100):.1f}% spread",
                            "description": "Based on 20 & 50-day moving averages"
                        },
                        "RSI (14)": {
                            "value": "Overbought" if rsi_value > 70 else "Oversold" if rsi_value < 30 else "Neutral",
                            "delta": f"{rsi_value:.1f}",
                            "description": "Current RSI value"
                        },
                        "Volume Trend": {
                            "value": "Above Average" if volume_high else "Below Average",
                            "delta": f"{((float(latest['Volume']) - float(latest['Volume_MA']))/float(latest['Volume_MA']) * 100):.1f}%",
                            "description": "Compared to 20-day average"
                        },
                        "Bollinger Bands": {
                            "value": "Upper Band" if close_price > bb_upper else 
                                    "Lower Band" if close_price < bb_lower else "Middle Band",
                            "delta": f"{((close_price - bb_lower)/(bb_upper - bb_lower) * 100):.1f}%",
                            "description": "Position within bands"
                        }
                    }
                    
                    # Display historical indicators
                    for indicator, data in historical_indicators.items():
                        with st.expander(f"{indicator}: {data['value']}"):
                            st.metric(
                                label=data['description'],
                                value=data['value'],
                                delta=data['delta']
                            )
                    
                    # Model Predictions Analysis
                    if 'results' in locals() and results is not None:
                        st.write("🤖 Model Predictions Analysis")
                        
                        # Calculate prediction metrics
                        current_price = float(df['Close'].iloc[-1])
                        pred_price = float(results['prediction'])
                        price_change_pct = ((pred_price - current_price) / current_price) * 100
                        predictions = results['individual_predictions']
                        bullish_models = sum(1 for p in predictions.values() if p > current_price)
                        bearish_models = len(predictions) - bullish_models
                        
                        prediction_indicators = {
                            "Price Prediction": {
                                "value": f"${pred_price:.2f}",
                                "delta": f"{price_change_pct:+.1f}% from current",
                                "description": "Ensemble model prediction"
                            },
                            "Model Consensus": {
                                "value": f"{bullish_models}/{len(predictions)} Bullish",
                                "delta": f"{(bullish_models/len(predictions)*100):.0f}% agreement",
                                "description": "Agreement among models"
                            },
                            "Prediction Range": {
                                "value": f"${results['lower_bound']:.2f} - ${results['upper_bound']:.2f}",
                                "delta": f"±{((results['upper_bound'] - results['lower_bound'])/2/pred_price*100):.1f}%",
                                "description": "Expected price range"
                            },
                            "Confidence Score": {
                                "value": f"{results['confidence_score']:.1%}",
                                "delta": "Based on model agreement",
                                "description": "Overall prediction confidence"
                            }
                        }
                        
                        # Display prediction indicators
                        for indicator, data in prediction_indicators.items():
                            with st.expander(f"{indicator}: {data['value']}"):
                                st.metric(
                                    label=data['description'],
                                    value=data['value'],
                                    delta=data['delta']
                                )
                        
                        # Overall Analysis
                        st.write("📈 Combined Signal Analysis")

                        technical_bullish = ma_bullish  
                        model_bullish = bullish_models > (len(predictions) / 2)  
                        model_confidence = results['confidence_score'] > 0.6

                        if technical_bullish and model_bullish:
                            st.success("Strong Buy Signal 💹: Both technical analysis and model predictions are bullish")
                        elif technical_bullish and not model_bullish:
                            st.warning("Mixed Signal 🤔: Technical analysis shows bullish trend but models predict decline")
                        elif not technical_bullish and model_bullish:
                            st.warning("Mixed Signal 🤔: Models predict upside but technical analysis shows weakness")
                        elif not technical_bullish and not model_bullish:
                            st.error("Strong Sell Signal 📉: Both technical analysis and models indicate bearish trend")

                        # Display confidence level
                        confidence_text = "High" if model_confidence else "Low"
                        st.info(f"Model Prediction Confidence: {confidence_text}")
                    
                else:
                    st.warning("Insufficient data points for technical analysis. Please ensure you have at least 50 days of historical data.")
            else:
                st.warning("No data available for technical analysis. Please enter a valid stock symbol.")
                        
        except Exception as e:
            st.error(f"Error in Technical Analysis: {str(e)}")
            st.write("Detailed error information:", str(e))
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Detailed error information:", str(e))
st.markdown("---")  
st.markdown(
    "🔗 [GitHub](https://github.com/sankeer28/stock-predictor)",
    unsafe_allow_html=True
)
