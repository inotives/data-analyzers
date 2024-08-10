import numpy as np
import pandas as pd 

from utils.tools import load_csv_from_data, export_data_to_csv

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


# This method ingest the OHLCV data and compute additonal Techinical Indicator to help further analysis
def enchanced_ohlcv_with_techinical_indicators(csvfile):

    # load the csv from 
    df = load_csv_from_data(csvfile)

    # Ensure the timestamp is a datetime object
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sort the data by timestamp and crypto
    df = df.sort_values(by=['crypto', 'timestamp'])
    
    ''' # 1. Price Change
    Shows the net change in price during the trading period. 
    Positive values indicate price gains, while negative values indicate losses.
    '''
    df['price_change'] = df['close'] - df['open']

    ''' # 2. Price Range
    Measures the volatility within a trading period. Larger ranges suggest higher volatility.
    '''
    df['price_range'] = df['high'] - df['low']

    ''' # 3. Price Momentum
    The difference in closing prices between consecutive periods, showing the direction 
    and strength of price movements. in essense the day-over-day price change for each cryptocurrency
    Positive momentum indicates an upward trend, while negative momentum suggests a downward trend. 
    Traders can use price momentum to create trading signals.
    '''
    df['price_momentum'] = df.groupby('crypto')['close'].diff()

    ''' # 4. Rate of Change (ROC) or daily return 
    calculates the daily percentage change in the closing price for each cryptocurrency. 
    calculate using ((Today's Value - Yesterday's Value) / Yesterday's Value) * 100
    Positive rate of change indicates an upward trend, while negative indicates a downward trend.
    can use rate of change to identify potential trading opportunities.
    '''
    df['rate_of_change'] = df.groupby('crypto')['close'].pct_change()

    ''' # 5. True Range (TR) and Average True Range (ATR)
    The True Range (TR) is a technical analysis indicator that represents the largest price range 
    between the current high, low, and previous close. A high TR indicates significant price volatility within a period.
    A low TR suggests a relatively stable price movement.

    The Average True Range (ATR) is a technical indicator that measures volatility by calculating 
    the average True Range (TR) over a specific period (typically 14 days).
    A high ATR indicates a volatile market with large price swings.
    A low ATR suggests a less volatile market with smaller price movements.

    How to Use TR and ATR:
    - Volatility measurement: Both TR and ATR are essential tools for measuring market volatility.
    - Stop-loss and take-profit levels: Traders often use ATR to set stop-loss and take-profit levels based on a multiple of the ATR.
    - Identifying trend strength: A declining ATR can indicate a weakening trend, while a rising ATR might suggest a strengthening trend.
    - Position sizing: ATR can be used to determine appropriate position sizes based on risk tolerance.
    '''
    df['tr'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close']), abs(x['low'] - x['close'])), axis=1)
    df['atr'] = df.groupby('crypto')['tr'].rolling(window=14).mean().reset_index(0, drop=True)

    ''' # 6. Cumulative Return
    The total change in the value of an investment over a specific period, expressed as a percentage. 
    It reflects the overall performance of an investment, taking into account all gains and losses.
    Cumulative Return = ((Ending Value / Beginning Value) - 1) * 100
    - Positive Cumulative Return: Indicates a profit over the investment period.
    - Negative Cumulative Return: Indicates a loss over the investment period.
    '''
    df['cumulative_return'] = df.groupby('crypto')['close'].apply(lambda x: (x / x.iloc[0]) - 1).reset_index(level=0, drop=True)

    ''' # 7. Typical Price or Average Price 
    In essense the avarage price of the period. typical_price = (open+high+low+close)/4.
    It can help to smooth out price fluctuations compared to using the closing price alone.
    Many technical indicators use typical price as a basis for their calculations.
    Limitations:
    - Equal weighting: It assigns equal weight to the open, high, low, and close prices, 
      which might not accurately reflect price behavior in all market conditions.
    - Sensitivity to outliers: Extreme high or low prices can significantly impact the typical price.
    '''
    df['typical_price'] = (df['open']+df['high']+df['low']+df['close']) / 4

    '''# 8. Volume-Weighted Average Price (VWAP)
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
    df['vwap'] = (df['volume'] * df['typical_price'] ).groupby(df['crypto']).cumsum() / df['volume'].groupby(df['crypto']).cumsum()

    ''' # 9. Moving Averages (MA) and Exponential Moving Average (EMA)
    Moving averages (MAs) are technical indicators that smooth out price data by calculating 
    the average price over a specific period. They are widely used in technical analysis 
    to identify trends, support and resistance levels, and potential reversal points.

    Types of Moving Averages:
    - Simple Moving Average (SMA or MA): Calculates the arithmetic mean of a given set of prices 
      over a specified period.
    - Exponential Moving Average (EMA): Gives more weight to recent prices, 
      making it more responsive to price changes.
    - Weighted Moving Average (WMA): Assigns weights to different data points, 
      allowing for customization of the smoothing effect.

    Interpreting Moving Averages:
    - Trend Identification: When the price is above the moving average, it suggests an upward trend; 
      when below, a downward trend.
    - Support and Resistance: Moving averages can act as support or resistance levels. 
      A price breaking above a long-term moving average is often seen as a bullish signal.
    - Crossovers: The intersection of two moving averages (e.g., 50-day and 200-day) 
      can generate buy or sell signals.

    Limitations of Moving Averages:
    - Lagging Indicator: Moving averages are lagging indicators, meaning they react to 
      price changes after they have occurred.
    - Sensitivity to Market Conditions: The effectiveness of moving averages can vary 
      depending on market conditions.
    
    Common Moving Average Periods:
    - For Short-term (~6mths): commonly use 20, 50
    - For Medium-term (~1 year): commonly use 50,200
    - For Long-term (>2 yrs): 100, 200 
    '''
    df['ma_20'] = df.groupby('crypto')['close'].rolling(window=20).mean().reset_index(0, drop=True)
    df['ma_50'] = df.groupby('crypto')['close'].rolling(window=50).mean().reset_index(0, drop=True)
    df['ma_100'] = df.groupby('crypto')['close'].rolling(window=100).mean().reset_index(0, drop=True)
    df['ma_200'] = df.groupby('crypto')['close'].rolling(window=200).mean().reset_index(0, drop=True)
    df['ema_20'] = df.groupby('crypto')['close'].apply(lambda x: x.ewm(span=20, adjust=False).mean()).reset_index(level=0, drop=True)
    df['ema_50'] = df.groupby('crypto')['close'].apply(lambda x: x.ewm(span=50, adjust=False).mean()).reset_index(level=0, drop=True)
    df['ema_100'] = df.groupby('crypto')['close'].apply(lambda x: x.ewm(span=100, adjust=False).mean()).reset_index(level=0, drop=True)
    df['ema_200'] = df.groupby('crypto')['close'].apply(lambda x: x.ewm(span=200, adjust=False).mean()).reset_index(level=0, drop=True)
    
    ''' # 10. Relative Strength Index (RSI)
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
    df['rsi'] = df.groupby('crypto')['close'].apply(lambda x: compute_rsi(x)).reset_index(level=0, drop=True)

    ''' # 11. Bollinger Bands
    Bollinger Bands are a technical analysis tool that plots bands around a simple moving average (SMA)
    of an asset's price. They are based on standard deviations from the SMA. it creates a band of 
    three lines (SMA, upper band, and lower band), which can indicate overbought or oversold conditions 
    when prices move outside the bands. 
    more can be read here: https://www.britannica.com/money/bollinger-bands-indicator

    Components:
    - Middle Band: This is a simple moving average (SMA) of the closing price.
    - Upper Band: Typically two standard deviations above the middle band. 
    - Lower Band: Typically two standard deviations below the middle band. 

    Interpretation:
    - Volatility: As volatility increases, the bands widen. 
      As volatility decreases, the bands contract.
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
    df['rolling_mean'] = df.groupby('crypto')['close'].rolling(window=20).mean().reset_index(0, drop=True)
    df['rolling_std'] = df.groupby('crypto')['close'].rolling(window=20).std().reset_index(0, drop=True)
    df['upper_band'] = df['rolling_mean'] + (df['rolling_std'] * 2)
    df['lower_band'] = df['rolling_mean'] - (df['rolling_std'] * 2)

    ''' # 12. Volume Change (VROC)
    Shows the change in trading volume from one period to the next, indicating shifts in market activity.
    It helps identify changes in trading activity and can be used to confirm or contradict price trends.
    VROC = ((Current Volume - Previous Volume) / Previous Volume) * 100

    Interpreting VROC:
    - Increasing VROC: Suggests growing buying or selling interest, which can confirm a price trend. 
    - Decreasing VROC: Indicates declining trading activity, potentially signaling a weakening trend 
      or a potential reversal.
    - Divergence: When the VROC diverges from price (e.g., price makes higher highs, but VROC makes lower highs), 
      it can be a warning sign of a potential trend reversal.
    
    Key Points:
    - Confirmation: VROC can confirm price trends indicated by other technical indicators.
    - Divergence: Divergence between price and VROC can be a valuable signal.
    - Volume Analysis: VROC complements traditional volume analysis by providing 
      a percentage-based view of volume changes.

    Limitations:
    - Lagging Indicator: Like many technical indicators, VROC is a lagging indicator.
    - Market Conditions: The effectiveness of VROC can vary depending on market conditions and asset type.
    '''
    df['volume_change'] = df.groupby('crypto')['volume'].diff()

    ''' # 13. Market Cap Change
    Measures the change in market capitalization, reflecting overall market sentiment.
    marketCap = Total Supply * Closing Price
    '''
    df['market_cap_change'] = df.groupby('crypto')['marketCap'].diff()

    ''' # 14. Volatility
    Measures the standard deviation of closing prices over a period (n days), 
    indicating the level of risk or uncertainty.
    The standard deviation is a measure of dispersion, indicating how much the data varies from the mean. 

    - High volatility can indicate increased market activity or uncertainty.
    - Volatility can be used to assess the risk associated with an investment.
    - Some trading strategies use volatility as a signal to enter or exit positions.
    
    '''
    df['volatility'] = df.groupby('crypto')['close'].rolling(window=20).std().reset_index(0, drop=True)

    # Drop unnecessary columns
    df = df.drop(columns=['tr', 'rolling_mean', 'rolling_std'])

    # Save the dataframe with new columns
    export_data_to_csv(df, f"{csvfile}_with_features")

    print(f"Final OHLCV DataFrame::\n{df}")

    return df


# Method to calculate rsi
def compute_rsi(series, window=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


