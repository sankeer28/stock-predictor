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
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Download required NLTK data 
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

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
NEWS_API_KEY = '0de37ca8af9748898518daf699189abf'
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

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

@st.cache_data(ttl=300)  
def get_current_price(symbol):
    """Fetch the current live price of a stock"""
    try:
        ticker = yf.Ticker(symbol)
        todays_data = ticker.history(period='1d')
        
        if todays_data.empty:
            return None
            
        # If market is open, we can get the current price
        if 'Open' in todays_data.columns and len(todays_data) > 0:
            # For market hours, use current price if available
            if 'regularMarketPrice' in ticker.info:
                current_price = ticker.info['regularMarketPrice']
                is_live = True
            else:
                # Fallback to the most recent close
                current_price = float(todays_data['Close'].iloc[-1])
                is_live = False
            
            # Get last update time
            last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            return {
                "price": current_price,
                "is_live": is_live,
                "last_updated": last_updated
            }
        return None
    except Exception as e:
        st.error(f"Error fetching current price: {str(e)}")
        return None

@st.cache_data(ttl=3600)
def analyze_sentiment(text):
    """
    Analyze sentiment using both VADER and TextBlob, with financial context
    """
    # Check if text is None or empty
    if not text or not isinstance(text, str):
        return {
            'sentiment': "⚖️ Neutral",
            'confidence': 0,
            'color': "gray",
            'score': 0
        }
        
    # Clean the text
    text = re.sub(r'[^\w\s]', '', text)
    
    # VADER analysis
    sia = SentimentIntensityAnalyzer()
    vader_scores = sia.polarity_scores(text)
    
    # TextBlob analysis
    blob = TextBlob(text)
    textblob_polarity = blob.sentiment.polarity
    
    # Enhanced financial context keywords with weights
    financial_pos = {
        'strong': 1.2,
        'climbed': 1.3,
        'up': 1.1,
        'higher': 1.1,
        'beat': 1.2,
        'exceeded': 1.2,
        'growth': 1.1,
        'profit': 1.1,
        'gain': 1.1,
        'positive': 1.1,
        'bullish': 1.3,
        'outperform': 1.2,
        'buy': 1.1,
        'upgrade': 1.2,
        'recovers': 1.3,
        'rose': 1.3,
        'closed higher': 1.4
    }
    
    financial_neg = {
        'weak': 1.2,
        'fell': 1.3,
        'down': 1.1,
        'lower': 1.1,
        'miss': 1.2,
        'missed': 1.2,
        'decline': 1.1,
        'loss': 1.1,
        'negative': 1.1,
        'bearish': 1.3,
        'underperform': 1.2,
        'sell': 1.1,
        'downgrade': 1.2,
        'sell-off': 1.4,
        'rattled': 1.3,
        'correction': 1.3,
        'crossed below': 1.4,
        'pain': 1.3
    }
    
    # Add financial context with weighted scoring
    financial_score = 0
    words = text.lower().split()
    
    # Look for percentage changes with context
    percent_pattern = r'(\d+(?:\.\d+)?)\s*%'
    percentages = re.findall(percent_pattern, text)
    for pct in percentages:
        if any(term in text.lower() for term in ["rose", "up", "climb", "gain", "higher"]):
            financial_score += float(pct) * 0.15
        elif any(term in text.lower() for term in ["down", "fall", "drop", "lower", "decline"]):
            financial_score -= float(pct) * 0.15
    
    # Look for technical indicators
    if "moving average" in text.lower():
        if "crossed below" in text.lower() or "below" in text.lower():
            financial_score -= 1.2
        elif "crossed above" in text.lower() or "above" in text.lower():
            financial_score += 1.2
    
    # Look for market action terms
    if "sell-off" in text.lower() or "selloff" in text.lower():
        financial_score -= 1.3
    if "recovery" in text.lower() or "recovers" in text.lower():
        financial_score += 1.3
    
    # Calculate weighted keyword scores
    pos_score = sum(financial_pos.get(word, 0) for word in words)
    neg_score = sum(financial_neg.get(word, 0) for word in words)
    
    if pos_score or neg_score:
        financial_score += (pos_score - neg_score) / (pos_score + neg_score)
    
    # Combine scores with adjusted weights
    combined_score = (
        vader_scores['compound'] * 0.3 +     # VADER
        textblob_polarity * 0.2 +            # TextBlob
        financial_score * 0.5                 # Enhanced financial context (increased weight)
    )
    
    # Adjust thresholds and confidence calculation
    if combined_score >= 0.15:
        sentiment = "📈 Positive"
        confidence = min(abs(combined_score) * 150, 100)  # Increased multiplier
        color = "green"
    elif combined_score <= -0.15:
        sentiment = "📉 Negative"
        confidence = min(abs(combined_score) * 150, 100)
        color = "red"
    else:
        sentiment = "⚖️ Neutral"
        confidence = (1 - abs(combined_score)) * 100
        color = "gray"
        
    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'color': color,
        'score': combined_score
    }

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
        
        # Add additional features for regressors - even more comprehensive
        # Add volume as a regressor if available
        has_volume_regressor = False
        if 'Volume' in df_copy.columns:
            prophet_df['volume'] = df_copy['Volume'].astype(float)
            prophet_df['log_volume'] = np.log1p(prophet_df['volume'])  # log transform to handle skewness
            # Add volume momentum (rate of change)
            prophet_df['volume_roc'] = prophet_df['volume'].pct_change(periods=5).fillna(0)
            has_volume_regressor = True
        
        # Add price-based features
        # Volatility at different time windows
        prophet_df['volatility_5d'] = prophet_df['y'].rolling(window=5).std().fillna(0)
        prophet_df['volatility_10d'] = prophet_df['y'].rolling(window=10).std().fillna(0)
        prophet_df['volatility_20d'] = prophet_df['y'].rolling(window=20).std().fillna(0)
        
        # Relative strength indicator (simplified) 
        delta = prophet_df['y'].diff()
        gain = delta.mask(delta < 0, 0).rolling(window=14).mean()
        loss = -delta.mask(delta > 0, 0).rolling(window=14).mean()
        rs = gain / loss
        prophet_df['rsi'] = 100 - (100 / (1 + rs)).fillna(50)
        
        # Price momentum
        prophet_df['momentum_5d'] = prophet_df['y'].pct_change(periods=5).fillna(0)
        prophet_df['momentum_10d'] = prophet_df['y'].pct_change(periods=10).fillna(0)
        
        # Distance from moving averages
        prophet_df['ma10'] = prophet_df['y'].rolling(window=10).mean().fillna(method='bfill')
        prophet_df['ma20'] = prophet_df['y'].rolling(window=20).mean().fillna(method='bfill')
        prophet_df['ma10_dist'] = (prophet_df['y'] / prophet_df['ma10'] - 1)
        prophet_df['ma20_dist'] = (prophet_df['y'] / prophet_df['ma20'] - 1)
        
        # Bollinger band position
        bb_std = prophet_df['y'].rolling(window=20).std().fillna(0)
        prophet_df['bb_position'] = (prophet_df['y'] - prophet_df['ma20']) / (2 * bb_std + 1e-10)  # Avoid division by zero
        
        # Handle outliers by winsorizing extreme values
        # Helps with improving forecast accuracy by removing noise
        for col in prophet_df.columns:
            if col != 'ds' and prophet_df[col].dtype.kind in 'fc':  # if column is float or complex
                q1 = prophet_df[col].quantile(0.01)
                q3 = prophet_df[col].quantile(0.99)
                prophet_df[col] = prophet_df[col].clip(q1, q3)
        
        # Drop any NaN values
        prophet_df = prophet_df.dropna()
        
        # Determine appropriate seasonality based on data size
        daily_seasonality = len(prophet_df) > 90  # Only use daily seasonality with enough data
        weekly_seasonality = False  # Explicitly disable weekly seasonality for stocks
        yearly_seasonality = len(prophet_df) > 365
        
        # Adaptive parameter selection based on volatility
        recent_volatility = prophet_df['volatility_20d'].mean()
        avg_price = prophet_df['y'].mean()
        rel_volatility = recent_volatility / avg_price
        
        # Adjust changepoint_prior_scale based on volatility
        # Higher volatility -> more flexibility
        cp_prior_scale = min(0.05 + rel_volatility * 0.5, 0.5)  
        
        # Create and fit the model with optimized parameters
        model = Prophet(
            daily_seasonality=daily_seasonality,
            weekly_seasonality=weekly_seasonality,  # Disabled to prevent weekend spikes
            yearly_seasonality=yearly_seasonality,
            changepoint_prior_scale=cp_prior_scale,  # Adaptive to volatility
            seasonality_prior_scale=10.0,  # Increased to capture market seasonality better
            seasonality_mode='multiplicative',  # Better for stock data that tends to have proportional changes
            changepoint_range=0.95,  # Look at more recent changepoints for stocks
            interval_width=0.9  # 90% confidence interval
        )
        
        # Add US stock market holidays
        model.add_country_holidays(country_name='US')
        
        # Add custom regressors
        if has_volume_regressor:
            model.add_regressor('log_volume', mode='multiplicative')
            model.add_regressor('volume_roc', mode='additive')
            
        # Add technical indicators as regressors
        model.add_regressor('volatility_5d', mode='multiplicative')
        model.add_regressor('volatility_20d', mode='multiplicative')
        model.add_regressor('rsi', mode='additive')
        model.add_regressor('momentum_5d', mode='additive')
        model.add_regressor('momentum_10d', mode='additive')
        model.add_regressor('ma10_dist', mode='additive')
        model.add_regressor('ma20_dist', mode='additive')
        model.add_regressor('bb_position', mode='additive')
        
        # Add custom seasonality for common stock patterns
        if len(prophet_df) > 60:  # Only with enough data
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
            
        # Add beginning/end of month effects (common in stocks)
        if len(prophet_df) > 40:
            prophet_df['month_start'] = (prophet_df['ds'].dt.day <= 3).astype(int)
            prophet_df['month_end'] = (prophet_df['ds'].dt.day >= 28).astype(int)
            model.add_regressor('month_start', mode='additive')
            model.add_regressor('month_end', mode='additive')
        
        # For stocks with enough data, add quarterly earnings effect
        if len(prophet_df) > 250:
            # Approximate earnings seasonality (rough quarterly pattern)
            prophet_df['earnings_season'] = ((prophet_df['ds'].dt.month % 3 == 0) & 
                                           (prophet_df['ds'].dt.day >= 15) & 
                                           (prophet_df['ds'].dt.day <= 30)).astype(int)
        
        # Fit the model
        model.fit(prophet_df)
        
        # Create future dataframe for prediction using business days only
        # This is critical to avoid weekend predictions for stock markets
        last_date = prophet_df['ds'].max()
        # Use business day frequency (weekdays only)
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=forecast_days * 1.4,  # Add extra days to account for weekends
            freq='B'  # Business day frequency - weekdays only
        )[:forecast_days]  # Limit to requested forecast days
        
        # Create the future dataframe with correct dates
        future = pd.DataFrame({'ds': future_dates})
        
        # Add regressor values to future dataframe
        # Copy the last rows of data for future predictions
        last_values = prophet_df.iloc[-1].copy()
        future_start_idx = len(prophet_df)
        
        # Add volume regressors to future dataframe
        if has_volume_regressor:
            # For volume, use median of last 30 days as future values
            median_volume = prophet_df['volume'].tail(30).median()
            future['volume'] = median_volume
            future['log_volume'] = np.log1p(future['volume'])
            
            # For volume_roc, use last 5-day average
            future['volume_roc'] = prophet_df['volume_roc'].tail(5).mean()
        
        # Add technical indicators to future dataframe
        # Use recent averages for future values
        future['volatility_5d'] = prophet_df['volatility_5d'].tail(10).mean()
        future['volatility_20d'] = prophet_df['volatility_20d'].tail(10).mean()
        future['rsi'] = prophet_df['rsi'].tail(5).mean()
        future['momentum_5d'] = prophet_df['momentum_5d'].tail(5).mean()
        future['momentum_10d'] = prophet_df['momentum_10d'].tail(5).mean()
        future['ma10_dist'] = prophet_df['ma10_dist'].tail(5).mean()
        future['ma20_dist'] = prophet_df['ma20_dist'].tail(5).mean()
        future['bb_position'] = prophet_df['bb_position'].tail(5).mean()
        
        # Add month start/end flags if we calculated them
        if 'month_start' in prophet_df.columns:
            future['month_start'] = (future['ds'].dt.day <= 3).astype(int)
            future['month_end'] = (future['ds'].dt.day >= 28).astype(int)
            
        # Add earnings season flags if we calculated them
        if 'earnings_season' in prophet_df.columns:
            future['earnings_season'] = ((future['ds'].dt.month % 3 == 0) & 
                                        (future['ds'].dt.day >= 15) & 
                                        (future['ds'].dt.day <= 30)).astype(int)
        
        # Make predictions
        forecast = model.predict(future)
        
        # Post-processing for improved accuracy:
        # 1. Ensure forecasts don't go negative for stock prices
        forecast['yhat'] = np.maximum(forecast['yhat'], 0)
        forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], 0)
        
        # 2. Apply an exponential decay to prediction intervals for uncertainty growth
        if forecast_days > 7:
            future_dates = pd.to_datetime(forecast['ds']) > prophet_df['ds'].max()
            days_out = np.arange(1, sum(future_dates) + 1)
            uncertainty_multiplier = 1 + (np.sqrt(days_out) * 0.01)
            
            # Adjust confidence intervals for future dates
            future_indices = np.where(future_dates)[0]
            for i, idx in enumerate(future_indices):
                forecast.loc[idx, 'yhat_upper'] = (forecast.loc[idx, 'yhat'] + 
                                                  (forecast.loc[idx, 'yhat_upper'] - 
                                                   forecast.loc[idx, 'yhat']) * uncertainty_multiplier[i])
                forecast.loc[idx, 'yhat_lower'] = (forecast.loc[idx, 'yhat'] - 
                                                  (forecast.loc[idx, 'yhat'] - 
                                                   forecast.loc[idx, 'yhat_lower']) * uncertainty_multiplier[i])
        
        # Make sure there are no weekend forecasts by checking the day of week
        # 5 = Saturday, 6 = Sunday
        forecast = forecast[forecast['ds'].dt.dayofweek < 5]
        
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
        
        # Create future dates for forecasting - using business days only
        last_date = df.index[-1]
        
        # Generate business days only (exclude weekends)
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1), 
            periods=forecast_days * 1.4,  # Add extra days to account for weekends
            freq='B'  # Business day frequency - weekdays only
        )[:forecast_days]  # Limit to requested forecast days
        
        # Historical dates and all dates together
        historical_dates = df.index
        all_dates = historical_dates.append(future_dates)
        
        # Predict future values
        future_x = np.arange(len(close_prices), len(close_prices) + len(future_dates)).reshape(-1, 1)
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
    def __init__(self, symbol, training_years=2, weights=None):  # Reduced from 5 to 2 years
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
        # Enhanced feature selection and engineering
        feature_columns = ['Close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                          'ROC', 'ATR', 'BB_upper', 'BB_lower', 'Volume_Rate',
                          'EMA12', 'EMA26', 'MOM', 'STOCH_K', 'WILLR']
        
        # Add derivative features to capture momentum and acceleration
        df['Price_Momentum'] = df['Close'].pct_change(5)
        df['MA_Crossover'] = (df['MA5'] > df['MA20']).astype(int)
        df['RSI_Momentum'] = df['RSI'].diff(3)
        df['MACD_Signal'] = df['MACD'] - df['MACD'].ewm(span=9).mean()
        df['Volume_Shock'] = ((df['Volume'] - df['Volume'].shift(1)) / df['Volume'].shift(1)).clip(-1, 1)
        
        # Add market regime detection (trending vs range-bound)
        df['ADX'] = self.calculate_adx(df)
        df['Is_Trending'] = (df['ADX'] > 25).astype(int)
        
        # Calculate volatility features
        df['Volatility_20d'] = df['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Add day of week feature (market often behaves differently on different days)
        df['DayOfWeek'] = df.index.dayofweek
        
        # Create dummy variables for day of week
        for i in range(5):  # 0-4 for Monday-Friday
            df[f'Day_{i}'] = (df['DayOfWeek'] == i).astype(int)
        
        # Handle extreme outliers by winsorizing
        for col in df.columns:
            if col != 'DayOfWeek' and df[col].dtype in [np.float64, np.int64]:
                q1 = df[col].quantile(0.01)
                q3 = df[col].quantile(0.99)
                df[col] = df[col].clip(q1, q3)
        
        # Select the final set of features
        enhanced_features = feature_columns + ['Price_Momentum', 'MA_Crossover', 'RSI_Momentum', 
                                              'MACD_Signal', 'Volume_Shock', 'ADX', 'Is_Trending', 
                                              'Volatility_20d', 'Day_0', 'Day_1', 'Day_2', 'Day_3', 'Day_4']
        
        # Ensure all selected features exist and drop NaN values
        available_features = [col for col in enhanced_features if col in df.columns]
        df_cleaned = df[available_features].copy()
        df_cleaned = df_cleaned.dropna()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df_cleaned)
        
        # Prepare sequences for LSTM
        X_lstm, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X_lstm.append(scaled_data[i-seq_length:i])
            y.append(scaled_data[i, 0])  # 0 index represents Close price
            
        # Prepare data for other models
        X_other = scaled_data[seq_length:]
        
        return np.array(X_lstm), X_other, np.array(y), df_cleaned.columns.tolist()
    
    @staticmethod
    def calculate_adx(df, period=14):
        """Calculate Average Directional Index (ADX) to identify trend strength"""
        try:
            # Calculate True Range
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            
            # Use .values to get numpy arrays and avoid pandas alignment issues
            ranges = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close})
            tr = ranges.max(axis=1)
            atr = tr.rolling(period).mean()
            
            # Calculate Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
            plus_dm = df['High'].diff()
            minus_dm = df['Low'].diff()
            
            # Handle conditions separately to avoid multi-column assignment
            plus_dm_mask = (plus_dm > 0) & (plus_dm > minus_dm.abs())
            plus_dm = plus_dm.where(plus_dm_mask, 0)
            
            minus_dm_mask = (minus_dm < 0) & (minus_dm.abs() > plus_dm)
            minus_dm = minus_dm.abs().where(minus_dm_mask, 0)
            
            # Calculate Smoothed +DM and -DM
            smoothed_plus_dm = plus_dm.rolling(period).sum()
            smoothed_minus_dm = minus_dm.rolling(period).sum()
            
            # Replace zeros to avoid division issues
            atr_safe = atr.replace(0, np.nan)
            
            # Calculate Plus Directional Index (+DI) and Minus Directional Index (-DI)
            plus_di = 100 * smoothed_plus_dm / atr_safe
            minus_di = 100 * smoothed_minus_dm / atr_safe
            
            # Handle division by zero in DX calculation
            di_sum = plus_di + minus_di
            di_sum_safe = di_sum.replace(0, np.nan)
            
            # Calculate Directional Movement Index (DX)
            dx = 100 * abs(plus_di - minus_di) / di_sum_safe
            
            # Calculate Average Directional Index (ADX)
            adx = dx.rolling(period).mean()
            
            return adx
        except Exception as e:
            # If ADX calculation fails, return a series of zeros with same index as input
            return pd.Series(0, index=df.index)

    def build_lstm_model(self, input_shape):
        # Simplified LSTM architecture for faster training
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def train_arima(self, df):
        # Auto-optimize ARIMA parameters
        from pmdarima import auto_arima
        
        try:
            model = auto_arima(df['Close'], 
                              start_p=1, start_q=1,
                              max_p=5, max_q=5,
                              d=1, seasonal=False,
                              stepwise=True,
                              suppress_warnings=True,
                              error_action='ignore',
                              max_order=5)
            return model
        except:
            # Fallback to standard ARIMA
            model = ARIMA(df['Close'], order=(5,1,0))
            return model.fit()

    def predict_with_all_models(self, prediction_days=30, sequence_length=30):  # Reduced sequence length
        try:
            # Fetch and prepare data
            df = self.fetch_historical_data()
            
            # Check if we have enough data
            if len(df) < sequence_length + 20:  # Need extra days for technical indicators
                st.warning(f"Insufficient historical data. Need at least {sequence_length + 20} days of data.")
                # Use available data but reduce sequence length if necessary
                sequence_length = max(10, len(df) - 20)
                
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Check for NaN values and handle them
            if df.isnull().any().any():
                df = df.fillna(method='ffill').fillna(method='bfill')
                
            # Verify we have enough valid data after cleaning
            if len(df.dropna()) < sequence_length:
                st.error("Insufficient valid data after calculating indicators.")
                return None
                
            # Enhanced data preparation with more features
            X_lstm, X_other, y, feature_names = self.prepare_data(df, sequence_length)
            
            # Verify we have valid data for model training
            if len(X_lstm) == 0 or len(y) == 0:
                st.error("Could not create valid sequences for prediction.")
                return None
                
            # Convert to numpy arrays
            X_lstm = np.array(X_lstm)
            X_other = np.array(X_other)
            y = np.array(y)
            
            # Split data using our optimized function
            X_lstm_train, X_lstm_test = X_lstm[:int(len(X_lstm)*0.8)], X_lstm[int(len(X_lstm)*0.8):]
            X_other_train, X_other_test = X_other[:int(len(X_other)*0.8)], X_other[int(len(X_other)*0.8):]
            y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]

            predictions = {}
            
            # Train and predict with LSTM (with reduced epochs)
            lstm_model = self.build_lstm_model((sequence_length, X_lstm.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            lstm_model.fit(X_lstm_train, y_train, epochs=20, batch_size=32,  # Reduced from 50 to 20 epochs
                          validation_data=(X_lstm_test, y_test),
                          callbacks=[early_stopping], verbose=0)
            lstm_pred = lstm_model.predict(X_lstm_test[-1:], verbose=0)[0][0]
            predictions['LSTM'] = lstm_pred

            # Train and predict with SVR
            svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
            svr_model.fit(X_other_train, y_train)
            svr_pred = svr_model.predict(X_other_test[-1:])
            predictions['SVR'] = svr_pred[0]

            # Train and predict with Random Forest (reduced complexity)
            rf_model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # Reduced from 100 to 50 trees
            rf_model.fit(X_other_train, y_train)
            rf_pred = rf_model.predict(X_other_test[-1:])
            predictions['Random Forest'] = rf_pred[0]

            # Train and predict with XGBoost (reduced complexity)
            xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=50)  # Added n_estimators
            xgb_model.fit(X_other_train, y_train)
            xgb_pred = xgb_model.predict(X_other_test[-1:])
            predictions['XGBoost'] = xgb_pred[0]

            # Skip KNN and GBM for speed
            # Only include fast models when we have limited data
            if len(X_other_train) > 100:
                # Train and predict with GBM (reduced complexity)
                gbm_model = GradientBoostingRegressor(random_state=42, n_estimators=50)  # Reduced complexity
                gbm_model.fit(X_other_train, y_train)
                gbm_pred = gbm_model.predict(X_other_test[-1:])
                predictions['GBM'] = gbm_pred[0]

            # Simplified ARIMA - skip if we have other models
            if len(predictions) < 3:
                try:
                    close_prices = df['Close'].values
                    arima_model = ARIMA(close_prices, order=(2,1,0))  # Simplified from (5,1,0)
                    arima_fit = arima_model.fit()
                    arima_pred = arima_fit.forecast(steps=1)[0]
                    arima_scaled = (arima_pred - df['Close'].mean()) / df['Close'].std()
                    predictions['ARIMA'] = arima_scaled
                except Exception as e:
                    st.warning(f"ARIMA prediction failed: {str(e)}")

            weights = self.weights

            # Adjust weights if some models failed
            available_models = list(predictions.keys())
            total_weight = sum(weights.get(model, 0.1) for model in available_models)  # Default weight 0.1
            adjusted_weights = {model: weights.get(model, 0.1)/total_weight for model in available_models}

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

# Set default display days to 600
display_days = st.slider(
    "Select number of days to display", 
    min_value=30, 
    max_value=3650, 
    value=600,  # Default to 600 days
    help="Displaying more days provides the model with more information for predictions."
)

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

try:
    # Fetch data
    df = fetch_stock_data(symbol, display_days)
    
    # Get current live price
    current_price_data = get_current_price(symbol)
    
    # Display stock name and current price in big text
    if not df.empty:
        if current_price_data is not None:
            # Use live price if available
            last_price = current_price_data["price"]
            last_date = current_price_data["last_updated"]
            price_label = "LIVE" if current_price_data["is_live"] else "LAST CLOSE"
            price_color = "#0f9d58" if current_price_data["is_live"] else "#1E88E5"
        else:
            # Fallback to historical data
            last_price = float(df['Close'].iloc[-1])
            last_date = df.index[-1].strftime('%Y-%m-%d')
            price_label = "LAST CLOSE"
            price_color = "#1E88E5"
        
        st.markdown(f"""
        <div style="display: flex; align-items: baseline; margin-bottom: 20px;">
            <h2 style="margin-right: 15px;">{symbol}</h2>
            <h1 style="color: {price_color}; margin: 0;">${last_price:.2f}</h1>
            <div style="margin-left: 10px;">
                <span style="color: gray; font-size: 14px;">{price_label}</span>
                <p style="color: gray; margin: 0; font-size: 14px;">as of {last_date}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
                        
                        # Calculate long-term forecast (90 days)
                        long_term_idx = min(next_date_idx + 90, len(forecast) - 1)
                        long_term_price = float(forecast['yhat'].iloc[long_term_idx])
                        long_term_change = (long_term_price - last_close_price) / last_close_price * 100
                        
                        # Create metrics with 3 columns for different timeframes
                        col1, col2, col3 = st.columns(3)
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
                        with col3:
                            st.metric(
                                label="90-Day Forecast", 
                                value=f"${long_term_price:.2f}", 
                                delta=f"{long_term_change:.2f}%"
                            )
                        
                        # Display trend and seasonality info
                        st.write("#### Forecast Components")
                        st.write("Prophet identifies the following patterns in the data:")
                        
                        try:
                            # Get components for analysis
                            trend_values = forecast['trend'][next_date_idx:medium_term_idx].values
                            
                            # Check if we have weekly component (we disabled it, but check just in case)
                            has_weekly_component = 'weekly' in forecast.columns and not all(forecast['weekly'] == 0)
                            
                            # Check if we have yearly component
                            has_yearly_component = 'yearly' in forecast.columns and not all(forecast['yearly'] == 0)
                            
                            # Determine trend direction
                            trend_direction = "Upward" if np.mean(np.diff(trend_values)) > 0 else "Downward"
                            trend_strength = np.abs(np.mean(np.diff(trend_values))/np.mean(trend_values)*100)
                            
                            # Create a detailed insights section for trend analysis
                            st.markdown(f"""
                            **Trend Analysis:**
                            - Direction: {trend_direction} ({trend_strength:.2f}% per period)
                            - Strength: {"Strong" if trend_strength > 0.5 else "Moderate" if trend_strength > 0.1 else "Weak"}
                            """)
                            
                            # Only show weekly patterns if weekly component exists
                            if has_weekly_component:
                                weekly_values = forecast['weekly'][next_date_idx:medium_term_idx].values
                                # Find day with maximum weekly effect
                                forecast_subset = forecast.iloc[next_date_idx:medium_term_idx]
                                max_weekly_idx = forecast_subset['weekly'].idxmax()
                                min_weekly_idx = forecast_subset['weekly'].idxmin()
                                max_weekly_day = pd.to_datetime(forecast_subset.loc[max_weekly_idx, 'ds']).strftime('%A')
                                min_weekly_day = pd.to_datetime(forecast_subset.loc[min_weekly_idx, 'ds']).strftime('%A')
                                
                                st.markdown(f"""
                                **Weekly Patterns:**
                                - Most positive day: {max_weekly_day}
                                - Most negative day: {min_weekly_day}
                                """)
                            else:
                                # No weekly component was used (correctly disabled for stock prices)
                                st.markdown("""
                                **Weekly Patterns:**
                                - None detected (weekly seasonality disabled for stock market data)
                                - Stock markets are closed on weekends, so no trading patterns exist
                                """)
                            
                            # Only show yearly patterns if yearly component exists
                            if has_yearly_component:
                                yearly_values = forecast['yearly'][next_date_idx:medium_term_idx].values
                                # Determine seasonal factor
                                seasonal_factor = "Positive" if np.mean(yearly_values) > 0 else "Negative"
                                current_month = datetime.now().strftime('%B')
                                next_month = (datetime.now() + timedelta(days=30)).strftime('%B')
                                
                                st.markdown(f"""
                                **Seasonal Analysis:**
                                - Current seasonal effect: {seasonal_factor}
                                - Current month ({current_month}): {"Favorable" if np.mean(yearly_values) > 0 else "Unfavorable"} historically
                                - Next month ({next_month}): {"Likely favorable" if np.mean(yearly_values[15:]) > 0 else "Likely unfavorable"} based on patterns
                                """)
                            else:
                                # No yearly component or not enough data
                                st.markdown("""
                                **Seasonal Analysis:**
                                - No significant yearly patterns detected
                                - Not enough historical data for reliable yearly seasonality detection
                                """)
                            
                            # Add trading insights based on forecast
                            st.subheader("Forecast-Based Trading Insights")
                            
                            # Calculate volatility as the standard deviation of forecast values
                            forecast_volatility = np.std(forecast['yhat'][next_date_idx:medium_term_idx])/np.mean(forecast['yhat'][next_date_idx:medium_term_idx])
                            
                            # Calculate momentum (rate of change over forecast period)
                            momentum = (medium_term_price - last_close_price)/last_close_price
                            
                            # Calculate confidence as inverse of the width of prediction intervals
                            confidence = 1 - np.mean((forecast['yhat_upper'] - forecast['yhat_lower'])/forecast['yhat'])
                            
                            # Create trading signals based on multiple factors
                            signal_strength = abs(medium_term_change)
                            signal_confidence = confidence*100
                            
                            signal_col1, signal_col2 = st.columns(2)
                            with signal_col1:
                                if medium_term_change > 10:
                                    st.success("🚀 Strong Buy Signal")
                                elif medium_term_change > 5:
                                    st.success("💹 Buy Signal")
                                elif medium_term_change > 2:
                                    st.info("📈 Weak Buy Signal")
                                elif medium_term_change < -10:
                                    st.error("🔻 Strong Sell Signal")
                                elif medium_term_change < -5:
                                    st.error("📉 Sell Signal")
                                elif medium_term_change < -2:
                                    st.warning("📉 Weak Sell Signal")
                                else:
                                    st.info("⚖️ Hold/Neutral Signal")
                            
                            with signal_col2:
                                st.metric("Signal Strength", f"{signal_strength:.1f}/10", 
                                         delta=f"{signal_confidence:.0f}% confidence")
                            
                            # Add forecast-based scenarios
                            st.subheader("Possible Scenarios")
                            scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
                            
                            with scenario_col1:
                                st.markdown(f"""
                                **Bullish Case:**
                                - Target: ${forecast['yhat_upper'].iloc[medium_term_idx]:.2f}
                                - Gain: {((forecast['yhat_upper'].iloc[medium_term_idx] - last_close_price)/last_close_price*100):.1f}%
                                - Probability: {(confidence * (1 + medium_term_change/100) * 100):.0f}%
                                """)
                                
                            with scenario_col2:
                                st.markdown(f"""
                                **Base Case:**
                                - Target: ${medium_term_price:.2f}
                                - Change: {medium_term_change:.1f}%
                                - Probability: {(confidence * 100):.0f}%
                                """)
                                
                            with scenario_col3:
                                st.markdown(f"""
                                **Bearish Case:**
                                - Target: ${forecast['yhat_lower'].iloc[medium_term_idx]:.2f}
                                - Loss: {((forecast['yhat_lower'].iloc[medium_term_idx] - last_close_price)/last_close_price*100):.1f}%
                                - Probability: {(confidence * (1 - medium_term_change/100) * 100):.0f}%
                                """)
                            
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
        # First, add a subheader for the prediction section
        st.subheader("Model Configuration & Predictions")
        
        # Add the weight configuration selector and description
        selected_weight = st.selectbox(
            "Select Model Configuration:",
            options=list(WEIGHT_CONFIGURATIONS.keys()),
            help="Choose different weight configurations for the prediction models. This affects the predictions generated by the 'Generate Predictions' button."
        )
        
        # Show the description in an info box
        st.info(WEIGHT_DESCRIPTIONS[selected_weight])
        
        # Add some space
        st.write("")
        
        # Then add the Generate Predictions button
        if st.button("Generate Predictions"):
            with st.spinner("Training multiple models and generating predictions..."):
                predictor = MultiAlgorithmStockPredictor(
                    symbol, 
                    weights=WEIGHT_CONFIGURATIONS[selected_weight]
                )
                results = predictor.predict_with_all_models(prediction_days=30)
                
                if results is not None:
                    # Calculate target date here since it's not in results
                    target_date = datetime.now() + timedelta(days=30)
                    st.write(f"#### Predictions for {target_date.strftime('%B %d, %Y')}")
                    
                    last_price = float(df['Close'].iloc[-1])
                    
                    # Individual model predictions
                    st.subheader("Individual Model Predictions")
                    model_predictions = pd.DataFrame({
                        'Model': results['individual_predictions'].keys(),
                        'Predicted Price': [v for v in results['individual_predictions'].values()],
                        'Target Date': target_date.strftime('%Y-%m-%d')  # Add target date to DataFrame
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
        try:
            news_headlines = get_news_headlines(symbol)
            
            if news_headlines and len(news_headlines) > 0:
                # Initialize sentiment tracking
                sentiment_scores = []
                sentiment_weights = []
                
                for title, description, url in news_headlines:
                    # Ensure title and description are strings
                    title = str(title) if title else ""
                    description = str(description) if description else ""
                    
                    # Analyze both title and description with different weights
                    title_analysis = analyze_sentiment(title)
                    desc_analysis = analyze_sentiment(description)
                    
                    # Combined analysis (title has more weight)
                    combined_score = title_analysis['score'] * 0.6 + desc_analysis['score'] * 0.4
                    sentiment_scores.append(combined_score)
                    
                    # Weight more recent news higher
                    sentiment_weights.append(1.0)
                    
                    # Determine display properties
                    if combined_score >= 0.2:
                        sentiment = "📈 Positive"
                        color = "green"
                        confidence = min(abs(combined_score) * 100, 100)
                    elif combined_score <= -0.2:
                        sentiment = "📉 Negative"
                        color = "red"
                        confidence = min(abs(combined_score) * 100, 100)
                    else:
                        sentiment = "⚖️ Neutral"
                        color = "gray"
                        confidence = (1 - abs(combined_score)) * 100
                    
                    with st.expander(f"{title} ({sentiment})"):
                        st.write(description)
                        st.markdown(f"[Read full article]({url})")
                        st.markdown(
                            f"<span style='color: {color}'>Sentiment: {sentiment} "
                            f"(Confidence: {confidence:.1f}%)</span>",
                            unsafe_allow_html=True
                        )
                
                # Calculate weighted average sentiment
                total_weight = sum(sentiment_weights)
                weighted_sentiment = sum(score * weight for score, weight in zip(sentiment_scores, sentiment_weights)) / total_weight
                
                # Display overall sentiment consensus
                st.write("### News Sentiment Consensus")
                
                # Calculate sentiment distribution
                positive_scores = sum(1 for score in sentiment_scores if score >= 0.2)
                negative_scores = sum(1 for score in sentiment_scores if score <= -0.2)
                neutral_scores = len(sentiment_scores) - positive_scores - negative_scores
                
                # Create metrics columns
                consensus_col1, consensus_col2, consensus_col3 = st.columns(3)
                total_articles = len(sentiment_scores)
                
                with consensus_col1:
                    pos_pct = (positive_scores / total_articles) * 100
                    st.metric("Positive News", 
                             f"{positive_scores}/{total_articles}",
                             f"{pos_pct:.1f}%")
                    
                with consensus_col2:
                    neg_pct = (negative_scores / total_articles) * 100
                    st.metric("Negative News", 
                             f"{negative_scores}/{total_articles}",
                             f"{neg_pct:.1f}%")
                    
                with consensus_col3:
                    neu_pct = (neutral_scores / total_articles) * 100
                    st.metric("Neutral News", 
                             f"{neutral_scores}/{total_articles}",
                             f"{neu_pct:.1f}%")
                
                # Overall sentiment conclusion with confidence
                sentiment_strength = abs(weighted_sentiment)
                confidence = min(sentiment_strength * 100, 100)
                
                if weighted_sentiment >= 0.2:
                    st.success(
                        f"📈 Strong Bullish Sentiment "
                        f"(Confidence: {confidence:.1f}%)\n\n"
                        f"Market news suggests positive momentum with {positive_scores} supportive articles."
                    )
                elif weighted_sentiment >= 0.1:
                    st.success(
                        f"📈 Moderately Bullish Sentiment "
                        f"(Confidence: {confidence:.1f}%)\n\n"
                        f"Market news leans positive with mixed signals."
                    )
                elif weighted_sentiment <= -0.2:
                    st.error(
                        f"📉 Strong Bearish Sentiment "
                        f"(Confidence: {confidence:.1f}%)\n\n"
                        f"Market news suggests negative pressure with {negative_scores} concerning articles."
                    )
                elif weighted_sentiment <= -0.1:
                    st.error(
                        f"📉 Moderately Bearish Sentiment "
                        f"(Confidence: {confidence:.1f}%)\n\n"
                        f"Market news leans negative with mixed signals."
                    )
                else:
                    st.info(
                        f"⚖️ Neutral Market Sentiment "
                        f"(Confidence: {(1 - sentiment_strength) * 100:.1f}%)\n\n"
                        f"Market news shows balanced or unclear direction."
                    )
                    
            else:
                st.info("No recent news available for this stock.")
                
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            st.info("No recent news available for this stock.")
        
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
                        trading_signal = get_trading_signal_strength(price_change_pct, results['confidence_score'])
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
