import pandas as pd
import numpy as np
from utils.tools import data_load_ohlcv


# Feature Engineering: Adding new features ...
def adding_features(data):
    """Add Features like MA7, MA14, EMA, etc to OHLCV data. Feature engineering to add commonly use TI to improve accuracy for prediction model. """

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
    data['mkcap_change'] = data['marketCap'].diff()

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
    data['vwap'] = (data['typical_price'] * data['volume']).cumsum() / data['volume'].cumsum()


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
    data['tr'] = data[['high', 'low', 'previous_close']].apply(
        lambda x: max(x['high'] - x['low'],
                      abs(x['high'] - x['previous_close']),
                      abs(x['low'] - x['previous_close'])), axis=1)
    data['atr_ma14'] = data['tr'].rolling(window=14).mean()
    data['atr_ema14'] = data['tr'].ewm(span=14, adjust=False).mean()
    data.drop(columns=['previous_close'], inplace=True)


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
    data['ema_20'] = data['close'].ewm(span=20, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['ema_100'] = data['close'].ewm(span=100, adjust=False).mean()
    data['ema_200'] = data['close'].ewm(span=200, adjust=False).mean()
    data['wma_20'] = calculate_wma(data['close'], 20)
    data['wma_50'] = calculate_wma(data['close'], 50)
    data['wma_100'] = calculate_wma(data['close'], 100)
    data['wma_200'] = calculate_wma(data['close'], 200)

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
    '''

    # Calculate gains (positive price changes) and losses (negative price changes)
    data['gain'] = data['price_change'].apply(lambda x: x if x > 0 else 0)
    data['loss'] = data['price_change'].apply(lambda x: -x if x < 0 else 0)

    # Calculate the rolling average of gains and losses, RSI is typically calculated over a 14-day period.
    data['avg_gain'] = data['gain'].rolling(window=14, min_periods=1).mean()
    data['avg_loss'] = data['loss'].rolling(window=14, min_periods=1).mean()

    # Calculate the Relative Strength (RS) = the ratio of the average gain to the average loss.
    data['rs'] = data['avg_gain'] / data['avg_loss']

    # Calculate the RSI
    data['rsi'] = 100 - (100 / (1 + data['rs']))


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
    # Calculate the upper and lower Bollinger Bands
    data['bollinger_upper'] = data['ma_20'] + (2 * data['std_20'])
    data['bollinger_lower'] = data['ma_20'] - (2 * data['std_20'])

    return data

def calculate_wma(series, window):
    """Feature calculation of WMA """
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)


if __name__ == '__main__':

    # load csv data to dataframe
    csvfile = 'ohlcv_bitcoin_20240821'
    data = data_load_ohlcv(f"{csvfile}.csv", start='2010-01-01')

    # add new feature to ohlcv
    data_with_features = adding_features(data)

    # Export the data to data folder.
    data_with_features.to_csv(f"data/processed/{csvfile}_with_features.csv")
