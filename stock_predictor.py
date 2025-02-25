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
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression

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

# Move the forecast_with_prophet function to be with the other cache functions
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

# Completely revise the Prophet forecast function
@st.cache_data(ttl=3600)
def forecast_with_prophet(df, forecast_days=30):
    try:
        # Check if we have enough data points
        if len(df) < 30:
            st.warning("Not enough historical data for reliable forecasting (< 30 data points)")
            return simple_forecast_fallback(df, forecast_days)
            
        # Make a copy to avoid modifying the original dataframe
        df_copy = df.copy()
        
        # Check for MultiIndex columns and handle appropriately
        has_multiindex = isinstance(df_copy.columns, pd.MultiIndex)
        
        # Reset index to make Date a column
        df_copy = df_copy.reset_index()
        
        # Find the date column
        date_col = None
        for col in df_copy.columns:
            # Handle both string and tuple column names
            col_str = col if isinstance(col, str) else col[0]
            if isinstance(col_str, str) and col_str.lower() in ['date', 'datetime', 'time', 'index']:
                date_col = col
                break
        
        if date_col is None:
            st.warning("No date column found - using simple forecast")
            return simple_forecast_fallback(df, forecast_days)
        
        # Prepare data for Prophet with careful handling of column types
        prophet_df = pd.DataFrame()
        
        # Extract the date and price columns safely
        date_values = df_copy[date_col]
        
        # For Close column, check if we're dealing with a MultiIndex
        if has_multiindex:
            # If MultiIndex, find the column with 'Close' as first element
            close_col = None
            for col in df_copy.columns:
                if isinstance(col, tuple) and col[0] == 'Close':
                    close_col = col
                    break
            
            if close_col is None:
                st.warning("No Close column found - using simple forecast")
                return simple_forecast_fallback(df, forecast_days)
            
            close_values = df_copy[close_col]
        else:
            # Standard columns
            close_values = df_copy['Close']
        
        # Assign to prophet dataframe
        prophet_df['ds'] = pd.to_datetime(date_values)
        prophet_df['y'] = close_values.astype(float)
        
        # Drop any NaN values
        prophet_df = prophet_df.dropna()
        
        # Create and fit the model
        model = Prophet(
            daily_seasonality=False, 
            yearly_seasonality=True,
            weekly_seasonality=True
        )
        model.fit(prophet_df)
        
        # Create future dataframe for prediction
        future = model.make_future_dataframe(periods=forecast_days)
        
        # Make predictions
        forecast = model.predict(future)
        
        return forecast
        
    except Exception as e:
        st.warning(f"Prophet model failed: {str(e)}. Using simple forecast instead.")
        return simple_forecast_fallback(df, forecast_days)

# Fix the simple forecast fallback
def simple_forecast_fallback(df, forecast_days=30):
    """A simple linear regression forecast as fallback when Prophet fails"""
    try:
        # Get the closing prices as a simple 1D array
        close_prices = df['Close'].values.flatten()
        
        # Create a sequence for x values (0, 1, 2, ...)
        x = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices
        
        # Fit a simple linear regression
        model = LinearRegression()
        model.fit(x, y)
        
        # Create future dates for forecasting
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
        
        # Historical dates and all dates together
        historical_dates = df.index
        all_dates = historical_dates.append(future_dates)
        
        # Predict future values
        future_x = np.arange(len(close_prices), len(close_prices) + forecast_days).reshape(-1, 1)
        future_y = model.predict(future_x)
        
        # Predict historical values for context
        historical_y = model.predict(x)
        
        # Calculate confidence interval (simple approach)
        mse = np.mean((y - historical_y) ** 2)
        sigma = np.sqrt(mse)
        
        # Create separate arrays for each column to ensure they're 1D
        ds_array = np.array(all_dates, dtype='datetime64')
        
        # Concatenate historical and future predictions
        yhat_array = np.concatenate([historical_y, future_y])
        yhat_lower_array = yhat_array - 1.96 * sigma
        yhat_upper_array = yhat_array + 1.96 * sigma
        
        # For trend/weekly/yearly, create simple placeholders
        trend_array = yhat_array.copy()  # Use the prediction as the trend
        weekly_array = np.zeros(len(yhat_array))  # No weekly component
        yearly_array = np.zeros(len(yhat_array))  # No yearly component
        
        # Create a forecast dataframe similar to Prophet's output
        forecast = pd.DataFrame({
            'ds': ds_array,
            'yhat': yhat_array,
            'yhat_lower': yhat_lower_array,
            'yhat_upper': yhat_upper_array,
            'trend': trend_array,
            'weekly': weekly_array,
            'yearly': yearly_array
        })
        
        return forecast
        
    except Exception as e:
        st.error(f"Simple forecast also failed: {str(e)}. No forecast will be shown.")
        return None

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
            
            # Check if we have enough data
            if len(df) < sequence_length + 20:  # Need extra days for technical indicators
                st.error(f"Insufficient historical data. Need at least {sequence_length + 20} days of data.")
                return None
                
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Check for NaN values and handle them
            if df.isnull().any().any():
                df = df.fillna(method='ffill').fillna(method='bfill')
                
            # Verify we have enough valid data after cleaning
            if len(df.dropna()) < sequence_length:
                st.error("Insufficient valid data after calculating indicators.")
                return None
                
            # Prepare features
            feature_columns = ['Close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                            'ROC', 'ATR', 'BB_upper', 'BB_lower', 'Volume_Rate',
                            'EMA12', 'EMA26', 'MOM', 'STOCH_K', 'WILLR']
                            
            # Verify all required features exist
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
                return None
                
            # Ensure we have valid data for all features
            df = df[feature_columns].dropna()
            if len(df) < sequence_length:
                st.error(f"Insufficient valid data points after cleaning. Need at least {sequence_length} points.")
                st.write(f"Available data points: {len(df)}")
                return None
                
            try:
                # Scale features
                scaled_data = self.scaler.fit_transform(df[feature_columns])
            except ValueError as e:
                st.error(f"Scaling error: {str(e)}")
                st.write("This usually happens with newly listed stocks or stocks with insufficient trading history.")
                return None
                
            # Prepare sequences for LSTM
            X_lstm, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X_lstm.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, 0])  # 0 index represents Close price
                
            # Verify we have enough sequences
            if len(X_lstm) == 0 or len(y) == 0:
                st.error("Could not create valid sequences for prediction.")
                return None
                
            # Prepare data for other models
            X_other = scaled_data[sequence_length:]
            
            # Convert to numpy arrays
            X_lstm = np.array(X_lstm)
            X_other = np.array(X_other)
            y = np.array(y)
            
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
    try:
        # Create a new DataFrame specifically for plotting
        plot_data = pd.DataFrame(index=df.index)
        
        # Add the Close price data
        plot_data['Close'] = df['Close'].values
        
        # Calculate and add SMA values if we have enough data
        if len(df) >= 20:
            plot_data['SMA_20'] = df['Close'].rolling(window=20).mean().values
        if len(df) >= 50:
            plot_data['SMA_50'] = df['Close'].rolling(window=50).mean().values
        
        # Add forecast days control under the chart controls
        st.write("#### Chart Controls")
        toggle_col1, toggle_col2, toggle_col3, toggle_col4, forecast_col = st.columns(5)

        with toggle_col1:
            show_sma20 = st.checkbox("Show 20-Day SMA", value=True)

        with toggle_col2:
            show_sma50 = st.checkbox("Show 50-Day SMA", value=True)
            
        with toggle_col3:
            show_bb = st.checkbox("Show Bollinger Bands", value=False)
            
        with toggle_col4:
            show_indicators = st.checkbox("Show RSI/MACD", value=False)

        with forecast_col:
            forecast_days = st.slider("Forecast Horizon (Days)", min_value=7, max_value=365, value=30, step=1)

        # Generate forecast with user-selected horizon
        with st.spinner("Generating forecast..."):
            forecast = forecast_with_prophet(df, forecast_days=forecast_days)
            
            # Calculate Bollinger Bands
            if show_bb and len(df) >= 20:
                ma20 = df['Close'].rolling(window=20).mean()
                std20 = df['Close'].rolling(window=20).std()
                df['BB_upper'] = ma20 + (std20 * 2)
                df['BB_lower'] = ma20 - (std20 * 2)
                df['BB_middle'] = ma20
            
            # Calculate RSI and MACD if needed
            if show_indicators:
                # Calculate RSI
                delta = df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # Calculate MACD
                df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
                df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA12'] - df['EMA26']
                df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Create plotly figure
            if show_indicators:
                # Create subplot with price and indicators
                fig = make_subplots(rows=3, cols=1, 
                                   shared_xaxes=True, 
                                   vertical_spacing=0.05,
                                   row_heights=[0.6, 0.2, 0.2],
                                   specs=[[{"secondary_y": False}],
                                          [{"secondary_y": False}],
                                          [{"secondary_y": False}]])
            else:
                fig = make_subplots(specs=[[{"secondary_y": False}]])
            
            # Add Close price (always shown)
            fig.add_trace(
                go.Scatter(x=plot_data.index, y=plot_data['Close'], name="Close Price", line=dict(color="blue"))
            )
            
            # Add SMA lines based on toggle state
            if 'SMA_20' in plot_data.columns and show_sma20:
                fig.add_trace(
                    go.Scatter(x=plot_data.index, y=plot_data['SMA_20'], name="20-Day SMA", line=dict(color="orange"))
                )
            if 'SMA_50' in plot_data.columns and show_sma50:
                fig.add_trace(
                    go.Scatter(x=plot_data.index, y=plot_data['SMA_50'], name="50-Day SMA", line=dict(color="green"))
                )
            
            # Add Bollinger Bands if enabled
            if show_bb and 'BB_upper' in df.columns:
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_upper'], name="BB Upper", line=dict(color="purple", width=1, dash='dash'))
                )
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['BB_lower'], name="BB Lower", 
                              line=dict(color="purple", width=1, dash='dash'),
                              fill='tonexty', fillcolor='rgba(128, 0, 128, 0.1)')
                )
            
            # Add RSI and MACD if enabled
            if show_indicators and 'RSI' in df.columns and 'MACD' in df.columns:
                # Add RSI trace to second subplot
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['RSI'], name="RSI", line=dict(color="orange")),
                    row=2, col=1
                )
                
                # Add reference lines for RSI
                fig.add_trace(
                    go.Scatter(x=[df.index[0], df.index[-1]], y=[70, 70], 
                              name="Overbought", line=dict(color="red", width=1, dash='dash'),
                              showlegend=False),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=[df.index[0], df.index[-1]], y=[30, 30], 
                              name="Oversold", line=dict(color="green", width=1, dash='dash'),
                              showlegend=False),
                    row=2, col=1
                )
                
                # Add MACD traces to third subplot
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['MACD'], name="MACD", line=dict(color="blue")),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(x=df.index, y=df['Signal'], name="Signal", line=dict(color="red")),
                    row=3, col=1
                )
                
                # Add MACD histogram
                fig.add_trace(
                    go.Bar(x=df.index, y=df['MACD']-df['Signal'], name="Histogram", 
                          marker=dict(color='rgba(0,0,255,0.5)')),
                    row=3, col=1
                )
            
            # Always add forecast if valid
            if forecast is not None and len(forecast) > 0:
                try:
                    # Add Prophet forecast
                    forecast_dates = pd.to_datetime(forecast['ds'])
                    historical_dates = plot_data.index
                    last_date = historical_dates[-1]
                    
                    # Create a boolean mask for future dates
                    future_mask = forecast_dates > last_date
                    
                    # Only proceed if we have future dates
                    if any(future_mask):
                        # Extract forecast values and convert to lists to avoid indexing issues
                        forecast_x = forecast_dates[future_mask].tolist()
                        forecast_y = forecast['yhat'][future_mask].tolist()
                        forecast_upper = forecast['yhat_upper'][future_mask].tolist()
                        forecast_lower = forecast['yhat_lower'][future_mask].tolist()
                        
                        # Add the forecast line
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_x, 
                                y=forecast_y,
                                name="Price Forecast", 
                                line=dict(color="red", dash="dash")
                            )
                        )
                        
                        # Add confidence interval
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_x,
                                y=forecast_upper,
                                name="Upper Bound",
                                line=dict(width=0),
                                showlegend=False
                            )
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_x,
                                y=forecast_lower,
                                name="Lower Bound",
                                fill='tonexty',
                                fillcolor='rgba(255, 0, 0, 0.1)',
                                line=dict(width=0),
                                showlegend=False
                            )
                        )
                except Exception as forecast_trace_err:
                    st.warning(f"Could not add forecast to chart: {str(forecast_trace_err)}")
            
            # Update layout for both chart types
            title = f"{symbol} Stock Price with Forecast"
            if show_indicators:
                # Add titles for subplots
                fig.update_yaxes(title_text="Price ($)", row=1, col=1)
                fig.update_yaxes(title_text="RSI", row=2, col=1)
                fig.update_yaxes(title_text="MACD", row=3, col=1)
                
                fig.update_layout(
                    title=title,
                    xaxis_title="Date",
                    hovermode="x unified",
                    legend=dict(y=0.99, x=0.01, orientation="h"),
                    template="plotly_white",
                    autosize=True,
                    height=700,  # Increase height for multiple subplots
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis=dict(
                        autorange=True,
                        rangeslider=dict(visible=False)
                    ),
                    yaxis=dict(
                        autorange=True,
                        fixedrange=False
                    ),
                    dragmode='pan'
                )
            else:
                fig.update_layout(
                    title=title,
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    hovermode="x unified",
                    legend=dict(y=0.99, x=0.01, orientation="h"),
                    template="plotly_white",
                    autosize=True,
                    height=500,
                    margin=dict(l=50, r=50, t=80, b=50),
                    xaxis=dict(
                        autorange=True,
                        rangeslider=dict(visible=False)
                    ),
                    yaxis=dict(
                        autorange=True,
                        fixedrange=False
                    ),
                    dragmode='pan'
                )
            
            # Update the chart configuration to fix zoom toggle issues
            st.plotly_chart(fig, use_container_width=True, config={
                'displayModeBar': True,
                'scrollZoom': True,
                'displaylogo': False,
                # Don't remove zoom buttons, but add a reset view button and make toggle possible
                'modeBarButtonsToRemove': ['autoScale2d', 'select2d', 'lasso2d'],
                'modeBarButtonsToAdd': ['resetScale2d', 'toImage'],
                'dragmode': 'pan'
            })
            
            # Display forecast metrics
            with st.expander("Prophet Forecast Details"):
                # Get last historical date and first forecast date
                if forecast is not None and len(forecast) > 0:
                    next_date_mask = forecast_dates > last_date
                    
                    if any(next_date_mask):
                        next_date_idx = next_date_mask.argmax()
                        last_close_price = float(plot_data['Close'].iloc[-1])
                        
                        # Calculate short-term forecast (7 days)
                        short_term_idx = min(next_date_idx + 7, len(forecast) - 1)
                        short_term_price = float(forecast['yhat'].iloc[short_term_idx])
                        short_term_change = (short_term_price - last_close_price) / last_close_price * 100
                        
                        # Calculate medium-term forecast (30 days)
                        medium_term_idx = min(next_date_idx + 30, len(forecast) - 1) 
                        medium_term_price = float(forecast['yhat'].iloc[medium_term_idx])
                        medium_term_change = (medium_term_price - last_close_price) / last_close_price * 100
                        
                        # Create metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                label="7-Day Forecast", 
                                value=f"${short_term_price:.2f}", 
                                delta=f"{short_term_change:.2f}%"
                            )
                        with col2:
                            st.metric(
                                label="30-Day Forecast", 
                                value=f"${medium_term_price:.2f}", 
                                delta=f"{medium_term_change:.2f}%"
                            )
                        
                        # Display trend and seasonality info
                        st.write("#### Forecast Components")
                        st.write("Prophet identifies the following patterns in the data:")
                        
                        try:
                            # Get components for analysis
                            trend_values = forecast['trend'][next_date_idx:medium_term_idx].values
                            weekly_values = forecast['weekly'][next_date_idx:medium_term_idx].values
                            yearly_values = forecast['yearly'][next_date_idx:medium_term_idx].values
                            
                            # Determine trend direction
                            trend_direction = "Upward" if np.mean(trend_values) > 0 else "Downward"
                            
                            # Find day with maximum weekly effect
                            forecast_subset = forecast.iloc[next_date_idx:medium_term_idx]
                            max_weekly_idx = forecast_subset['weekly'].idxmax()
                            max_weekly_day = pd.to_datetime(forecast_subset.loc[max_weekly_idx, 'ds']).strftime('%A')
                            
                            # Determine seasonal factor
                            seasonal_factor = "Positive" if np.mean(yearly_values) > 0 else "Negative"
                            
                            st.write(f"- **Trend**: {trend_direction}")
                            st.write(f"- **Weekly Pattern**: Most positive on {max_weekly_day}")
                            st.write(f"- **Seasonal Factor**: Currently {seasonal_factor}")
                        except Exception as component_err:
                            st.warning(f"Could not analyze forecast components: {str(component_err)}")
                            
    except Exception as e:
        st.error(f"Error creating enhanced chart: {str(e)}")
        # Fall back to simple chart
        try:
            st.line_chart(df['Close'])
        except:
            st.error("Unable to display chart. Please check your data.")
    
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
                    
                    
                    # Individual model predictions
                    st.subheader("Individual Model Predictions")
                    model_predictions = pd.DataFrame({
                        'Model': results['individual_predictions'].keys(),
                        'Predicted Price': [v for v in results['individual_predictions'].values()]
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
                                "value": f"${abs(results['lower_bound']):.2f} - ${abs(results['upper_bound']):.2f}",
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

                        # Get trading signal strength based on price_change
                        def get_trading_signal_strength(price_change, confidence_score):
                            if abs(price_change) > 10:
                                return "strong_buy" if price_change > 0 else "strong_sell"
                            elif abs(price_change) > 3 and confidence_score > 0.8:
                                return "buy" if price_change > 0 else "sell"
                            elif abs(price_change) > 2 and confidence_score > 0.6:
                                return "moderate_buy" if price_change > 0 else "moderate_sell"
                            elif abs(price_change) < 1:
                                return "hold"
                            else:
                                return "weak_buy" if price_change > 0 else "weak_sell"

                        # Get signals from different sources
                        technical_bullish = ma_bullish
                        trading_signal = get_trading_signal_strength(price_change, results['confidence_score'])
                        model_confidence = results['confidence_score'] > 0.6

                        # Determine overall signal
                        if technical_bullish and trading_signal in ['strong_buy', 'buy']:
                            st.success("🚀 Very Strong Buy Signal: Technical analysis is bullish and models show strong upward momentum")
                        elif technical_bullish and trading_signal in ['moderate_buy', 'weak_buy']:
                            st.success("💹 Strong Buy Signal: Technical analysis is bullish with moderate model support")
                        elif not technical_bullish and trading_signal in ['strong_buy', 'buy']:
                            st.warning("📈 Cautious Buy Signal: Models show strong upward potential but technical indicators suggest caution")
                        elif technical_bullish and trading_signal in ['hold']:
                            st.info("⚖️ Hold with Bullish Bias: Technical analysis is positive but models suggest consolidation")
                        elif not technical_bullish and trading_signal in ['hold']:
                            st.info("⚖️ Hold with Bearish Bias: Technical analysis is negative and models suggest consolidation")
                        elif technical_bullish and trading_signal in ['weak_sell', 'moderate_sell']:
                            st.warning("🤔 Mixed Signal: Technical analysis is bullish but models show weakness")
                        elif not technical_bullish and trading_signal in ['weak_sell', 'moderate_sell']:
                            st.error("📉 Strong Sell Signal: Both technical analysis and models show weakness")
                        elif not technical_bullish and trading_signal in ['strong_sell', 'sell']:
                            st.error("🔻 Very Strong Sell Signal: Technical analysis is bearish and models show strong downward momentum")
                        else:
                            st.warning("🔄 Mixed Signals: Conflicting indicators suggest caution")

                        # Display confidence metrics
                        confidence_text = "High" if model_confidence else "Low"
                        st.info(f"Model Prediction Confidence: {confidence_text}")

                        # Additional context based on signals
                        if model_confidence:
                            if technical_bullish:
                                st.write("💡 Technical indicators support the model predictions, suggesting higher reliability")
                            else:
                                st.write("💡 Technical indicators contrast with model predictions, suggesting careful monitoring")
                        else:
                            st.write("💡 Lower model confidence suggests waiting for clearer signals before making decisions")
                                            
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