import numpy as np
import pandas as pd 
import ta 

from utils.tools import load_csv_from_data, export_data_to_csv
import utils.db as db 

def load_crypto_ohlcv_from_db(crypto):
    '''Loading OHLCV data from db '''
    conn = db.DBConnection()
    conn.create_engine()
    data = conn.pull_data_to_dataframe(f"""
    SELECT oc.metric_date , oc.open, oc.high , oc.low , oc.close , oc.volume, oc.market_cap
    FROM ohlcv_cmc oc 
    WHERE oc.crypto = '{crypto}'
      AND oc.metric_date >= '2014-01-01'
    ORDER BY oc.metric_date ASC
    ;
    """)

    return data 

# Processed stablecoin OHLCV
def process_stablecoins_ohlcv():
    
    # Read the CSV
    csv_file = 'stablecoin_ohlcv'
    data = load_csv_from_data(csv_file)

    # Convert to timestamp object
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Sort the data by timestamp and crypto
    data = data.sort_values(by=['crypto', 'timestamp'])

    ''' # Price deviation and perc_deviation
    These directly measure how much the price deviates from the $1.00 peg. 
    '''
    data['price_deviation'] = (data['close'] - 1.0).abs()  # Absolute deviation from $1.00
    data['percentage_deviation'] = (data['close'] - 1.0).abs() / 1.0 * 100  # Percentage deviation from $1.00
    
    ''' # Rolling Standard Deviation (Volatility)
    This measures the volatility of the stablecoin's price over a rolling window (e.g., 10 days).
    '''
    data['rolling_std_dev'] = data.groupby('crypto')['close'].transform(lambda x: x.rolling(window=10, min_periods=1).std())

    ''' # True Range
    The difference between the high and low prices during a trading day.
    '''
    data['true_range'] = data['high'] - data['low']

    ''' # Volume-to-Market Cap Ratio
    A liquidity indicator that shows how active the trading is relative to the total market cap.
    '''
    data['volume_to_market_cap_ratio'] = data['volume'] / data['market_cap']

    ''' # Volume Change
    Percentage change in volume, useful for identifying spikes or drops in trading activity.
    '''
    data['volume_change'] = data.groupby('crypto')['volume'].pct_change()

    ''' # Market Cap Change
    Percentage change in market cap, which might indicate changes in market sentiment or supply.
    '''
    data['market_cap_change'] = data.groupby('crypto')['market_cap'].pct_change()

    # Export to CSV
    export_data_to_csv( data, f"{csv_file}_cleaned")

    print(f"Final OHLCV DataFrame::\n{data}")

    return data


# Method to calculate rsi -- Deprecated, use ta RSI instead
def compute_rsi(series, window=14):
    '''
    # Calculate gains (positive price changes) and losses (negative price changes)
    data['gain'] = data['price_change'].apply(lambda x: x if x > 0 else 0)
    data['loss'] = data['price_change'].apply(lambda x: -x if x < 0 else 0)

    # Calculate the rolling average of gains and losses, RSI is typically calculated over a 14-day period.
    data['avg_gain'] = data['gain'].rolling(window=14, min_periods=1).mean()
    data['avg_loss'] = data['loss'].rolling(window=14, min_periods=1).mean()

    # Calculate the Relative Strength (RS) = the ratio of the average gain to the average loss.
    data['rs'] = data['avg_gain'] / data['avg_loss']
    '''

    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# Method to compute wma -- Deprecated, use ta WMA instead
def calculate_wma(series, window):
    """Feature calculation of WMA """
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)


def add_indicators_daily(data):
    """Add commonly technical indicator for prediction model to OHLCV data."""

    '''price_change: price changes during the trading period '''
    data['price_change'] = data['close'] - data['open']

    '''price_range: price volatilty during the trading period. higher = more volatile '''
    data['price_range'] = data['high'] - data['low']

    '''price_momentum: day-over-day price changes. positive momentum = upward trend, negative=downward trend '''
    data['price_momentum'] = data['close'].diff()


    '''typical_price: average price of the period. '''
    data['typical_price'] = (data['open']+data['high']+data['low']+data['close']) / 4


    '''volume_change (VROC): day-over-day change of trading volume'''
    data['volume_change'] = data['volume'].diff()

    '''market_cap_change: day-over-day change of market cap '''
    data['mkcap_change'] = data['market_cap'].diff()

    '''Volume Weighted Average Price (VWAP)
    VWAP is a technical analysis indicator that represents the average price a asset has traded at
    throughout the day, weighted by volume. It is calculated by taking the sum of the product of price
    and volume for each trade, divided by the total volume traded for that day.

    Interpretation:
    - Fair value: VWAP can be seen as a benchmark for the "fair price" of an asset during the day.
    - Trend identification: If the current price is above the VWAP, it might suggest an upward trend;
      if below, a downward trend.
    - Trading strategy: Some traders use VWAP as a signal to buy or sell.
      For instance, buying when the price is below the VWAP and selling when it's above.

    Limitations:
    - Lagging indicator: VWAP is a lagging indicator, meaning it reflects past price and volume data.
    - Market conditions: VWAP might not be as effective in highly volatile or illiquid markets.
    '''
    data['vwap'] = ta.volume.volume_weighted_average_price(data['high'], data['low'], data['close'], data['volume'])


    '''daily_return: day-over-day price changes in % or rate of changes. positive=upward trend, negative=downward-trend
    cummulative_return : total return achieved over a period, considering the compounding of returns.
    calculate by taking the cumulative product of the daily returns plus one (to represent the return for each day).
    '''
    data['daily_return'] = data['close'].pct_change()
    data['cumulative_return'] = (1 + data['daily_return']).cumprod() - 1


    ''' True Range (TR) and Average True Range (ATR)
    The True Range (TR) is a technical analysis indicator that represents the largest price range between the current
    high, low, and previous close. A high TR indicates significant price volatility within a period.
    A low TR suggests a relatively stable price movement.

    The Average True Range (ATR) is a technical indicator that measures volatility by calculating the average True Range (TR)
    over a specific period (typically 14 days). A high ATR indicates a volatile market with large price swings.
    A low ATR suggests a less volatile market with smaller price movements.

    How to Use TR and ATR:
    - Volatility measurement: Both TR and ATR are essential tools for measuring market volatility.
    - Stop-loss and take-profit levels: Traders often use ATR to set stop-loss and take-profit levels based on a multiple of the ATR.
    - Identifying trend strength: A declining ATR can indicate a weakening trend, while a rising ATR might suggest a strengthening trend.
    - Position sizing: ATR can be used to determine appropriate position sizes based on risk tolerance.
    '''
    data['previous_close'] = data['close'].shift(1)
    data['atr'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)

    ''' Moving Averages (MA), Exponential Moving Average (EMA) and Weigthed Moving Average (WMA)
    Moving averages (MAs) are technical indicators that smooth out price data by calculating
    the average price over a specific period. They are widely used in technical analysis
    to identify trends, support and resistance levels, and potential reversal points.

    Types of Moving Averages:
    - Simple Moving Average (SMA or MA): Calculates the arithmetic mean of a given set of prices over a specified period.
    - Exponential Moving Average (EMA): Gives more weight to recent prices, making it more responsive to price changes.
    - Weighted Moving Average (WMA): Assigns weights to different data points, allowing for customization of the smoothing effect.

    Interpreting Moving Averages:
    - Trend Identification: When the price is above the moving average, it suggests an upward trend; when below, a downward trend.
    - Support and Resistance: Moving averages can act as support or resistance levels.
      A price breaking above a long-term moving average is often seen as a bullish signal.
    - Crossovers: The intersection of two moving averages (e.g., 50-day and 200-day) treated as reversal signal (BUY -> SELL vice versa)

    Limitations of Moving Averages:
    - Lagging Indicator: Moving averages are lagging indicators, meaning they react to price changes after they have occurred.
    - Sensitivity to Market Conditions: The effectiveness of moving averages can vary depending on market conditions.

    Common Moving Average Periods:
    - For Short-term (~6mths): commonly use 20, 50
    - For Medium-term (~1 year): commonly use 50,200
    - For Long-term (>2 yrs): 100, 200
    '''

    data['ma_20'] = data['close'].rolling(window=20, min_periods=1).mean()
    data['ma_50'] = data['close'].rolling(window=50, min_periods=1).mean()
    data['ma_100'] = data['close'].rolling(window=100, min_periods=1).mean()
    data['ma_200'] = data['close'].rolling(window=200, min_periods=1).mean()
    
    data['ema_7'] = data['close'].ewm(span=7, adjust=False).mean()
    data['ema_14'] = data['close'].ewm(span=14, adjust=False).mean()
    data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['ema_28'] = data['close'].ewm(span=28, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['ema_100'] = data['close'].ewm(span=100, adjust=False).mean()

    ''' WMA (Weighted Moving Average)
    Common used WMA window: 
    - Short-term (5) and Long-term (20): This combination is popular for capturing short-term trends and identifying potential buy or sell signals.
    - Short-term (12) and Long-term (26): This is often used in conjunction with the MACD indicator, which uses these same window sizes.
    '''
    data['wma_5'] = ta.trend.WMAIndicator(data['close'], window=5) # use for short-term crossover
    data['wma_7'] = ta.trend.WMAIndicator(data['close'], window=7) # use for short-term crossover
    data['wma_12'] = ta.trend.WMAIndicator(data['close'], window=12) # use for short-term crossover
    data['wma_14'] = ta.trend.WMAIndicator(data['close'], window=14) # use for short-term crossover
    data['wma_20'] = ta.trend.WMAIndicator(data['close'], window=20) # use for longer-term crossover
    data['wma_26'] = ta.trend.WMAIndicator(data['close'], window=26) # use for longer-term crossover

    ''' Standard Deviation
    Standard deviation is a statistical measure that quantifies the amount of variation or dispersion in a set of data values.
    It tells you how much the data points deviate from the mean (average) of the dataset.
    We will apply the standard deviation calculation over a set window (20, 50) to measure the volatility level.
    '''
    data['std_20'] = data['close'].rolling(window=20, min_periods=1).std()
    data['std_50'] = data['close'].rolling(window=50, min_periods=1).std()


    ''' Relative Strength Index (RSI)
    RSI is a momentum oscillator that measures the speed and change of price movements.
    It helps identify overbought or oversold conditions in an asset.

    How RSI Works:
    1. Calculate Average Gain and Loss: Over a specific period (typically 14 days),
       calculate the average of all price increases and the average of all price decreases.
    2. Relative Strength: Divide the average gain by the average loss.
    3. RSI Calculation: The RSI is calculated as 100 - (100 / (1 + Relative Strength)).

    Interpretation:
    - Oversold:An RSI value below 30 is generally considered oversold,
      suggesting a potential price reversal upwards.
    - Overbought: An RSI value above 70 is generally considered overbought,
      suggesting a potential price reversal downwards.
    - Divergence: When the price makes a new high, but the RSI fails to make a higher high
      (or vice versa for a lower low), it's called a divergence, which can be a potential reversal signal.

    Limitations:
    - Lagging Indicator: RSI is a lagging indicator, meaning it confirms price trends
      rather than predicting them.
    - Market Conditions: RSI levels can vary across different markets and timeframes.
    - False Signals: RSI can generate false signals, especially in trending markets.

    Using RSI:
    - Identify potential entry and exit points: Use RSI to spot potential buying opportunities
      when the indicator is oversold and selling opportunities when it's overbought.  
    - Confirm trend direction: RSI can help confirm the direction of a trend.
    - Identify divergences: Divergences between price and RSI can signal potential trend reversals.

    Common Used RSI Gauge: 
    - RSI >= 70 : Overbought, may be due for correction or pullback 
    - RSI <= 30 : Oversold, may be undervalued and poised for rebound
    '''
    data['rsi'] = ta.momentum.rsi(data['close'], window=14)

    ''' Bollinger Bands
    Bollinger Bands are a technical analysis tool that plots bands around a simple moving average (SMA)
    of an asset's price. They are based on standard deviations from the SMA. it creates a band of
    three lines (SMA, upper band, and lower band), which can indicate overbought or oversold conditions
    when prices move outside the bands. Typically using Window lenght of 20-days in SMA and Std.Dev
    more can be read here: https://www.britannica.com/money/bollinger-bands-indicator

    Components:
    - Middle Band: This is a simple moving average (SMA) of the closing price.
    - Upper Band: Typically 2 standard deviations above the middle band.
    - Lower Band: Typically 2 standard deviations below the middle band.

    Interpretation:
    - Volatility: As volatility increases, the bands widen. Volatility decreases, the bands contract.
    - Overbought/Oversold: Prices touching the upper band can signal an overbought condition,
      while prices touching the lower band might indicate an oversold condition.
    - Breakouts: Prices breaking above the upper band or below the lower band can signal
      potential trend reversals.
    - Contraction and Expansion: When the bands contract, it might indicate a period of low volatility,
      which can lead to a breakout in either direction.

    Important Considerations:
    - Timeframe: The choice of the moving average period (e.g., 20 days) and the standard deviation
      multiplier (typically 2) can affect the sensitivity of the bands.
    - False Signals: Like any technical indicator, Bollinger Bands can generate false signals.
      Combining them with other indicators can help improve accuracy.
    - Market Conditions: The effectiveness of Bollinger Bands can vary depending on market conditions.
    '''
    data['bb_upper'] = ta.volatility.bollinger_hband(data['close'], window=20, fillna=True)
    data['bb_lower'] = ta.volatility.bollinger_lband(data['close'], window=20, fillna=True)

    ''' MACD (Moving Average Convergence Divergence)
    MACD is a popular TI used in tech analysis to identify changes in momentum and potential trend reversals. 
    It consists of two exponential moving averages (EMAs) and their difference, known as the histogram.

    Components:
    - Signal Line: A shorter-term EMA (typically 9 periods) of the MACD line.
    - MACD Line: The difference between a longer-term EMA (typically 26 periods) and a shorter-term EMA (typically 12 periods).
    - Histogram: The difference between the MACD line and the signal line.

    How MACD is used: 
    1. Crossovers: When the MACD line crosses above or below the signal line, it can indicate potential trend reversals.
        - Bullish Signal: A bullish crossover occurs when the MACD line crosses above the signal line.
        - Bearish Signal: A bearish crossover occurs when the MACD line crosses below the signal line.
    2. Divergence: MACD can also be used to identify divergence between price and momentum. 
       If the price is making a new high, but the MACD is making a new low, 
       it could indicate a potential bearish divergence.
    3. Histogram: The histogram can provide additional insights into the strength of the trend. 
       A rising histogram indicates increasing bullish momentum, 
       while a falling histogram indicates decreasing bullish momentum.
    
    While there's no strict rule for interpreting MACD, some common gauges include:
    - Histogram: A histogram above the zero line suggests bullish momentum, while a histogram below the zero line suggests bearish momentum.
    - MACD Line: When the MACD line is far above the signal line, it indicates strong bullish momentum. Conversely, a MACD line far below the signal line suggests strong bearish momentum.
    - Crossovers: The strength of a crossover signal can be influenced by the angle of the crossover and the overall trend.

    Key Points:
    - MACD is a lagging indicator, meaning it signals potential reversals after a trend has already started.
    - MACD can be used in combination with other technical indicators for a more comprehensive analysis.
    - The effectiveness of MACD can vary depending on market conditions and the specific asset being analyzed.

    '''
    data['macd'] = ta.trend.macd(data['close'])

    return data

# Extract data to CSV
def extract_ohlcv(crypto):
    data = load_crypto_ohlcv_from_db(crypto)

    data = data.round(2)
    
    return data 

def extract_ohlcv_aggr_weekly(crypto):
    data = load_crypto_ohlcv_from_db(crypto)

    # data = data.round(2)

    data['metric_date'] = pd.to_datetime(data['metric_date'])

    data.set_index('metric_date', inplace=True)

    # Resample the data to weekly frequency ('W') and aggregate
    weekly_df = data.resample('W').agg({
        'open': 'first',           # First open of the week
        'high': 'max',             # Highest high of the week
        'low': 'min',              # Lowest low of the week
        'close': 'last',           # Last close of the week
        'volume': 'sum',           # Sum of the volume for the week
        'market_cap': 'last'       # Last market cap of the week
    })

    # Calculate weekly market cap growth (percentage change)
    weekly_df['market_cap_weekly_growth'] = weekly_df['market_cap'].pct_change() * 100

    # Calculate market cap change (difference between this week's and last week's market cap)
    weekly_df['market_cap_change'] = weekly_df['market_cap'].diff()

    weekly_df['crypto'] = crypto

    # Reset the index to have metric_date as a column again
    weekly_df.reset_index(inplace=True)

    export_data_to_csv(weekly_df, f"{crypto}-ohlcv-weekly")

    return weekly_df 

# Extract data with TI to CSV
def extract_ohlcv_with_ti(crypto):
    '''Extracting OHLCV and with computed OHLCV data'''
    data = load_crypto_ohlcv_from_db(crypto)
    data_ti = add_indicators_daily(data)

    # Fill those NA to 0
    data_ti.fillna(0, inplace=True)
    
    print(f"Columns:: {data_ti.columns}")
    print(f"Shape::{data_ti.shape}")
    print(data_ti)
    
    export_data_to_csv(data, 'bitcoin-ohlcv-ti')

