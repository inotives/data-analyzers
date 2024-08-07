import numpy as np
import pandas as pd 

from utils.tools import load_csv_from_data, export_data_to_csv

def data_cleanup():
    
    # Read the CSV 
    csv_file = 'stablecoin_ohlcv'
    data = load_csv_from_data(csv_file)

    # Convert to timestamp obj
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # Calculate 7-day and 30-day SMA for close price
    data['SMA_7'] = data['close'].rolling(window=7).mean()
    data['SMA_30'] = data['close'].rolling(window=30).mean()
    
    # Calculate Daily Return (percentage change from the previous day's close)
    data['Daily_Return'] = data['close'].pct_change() * 100

    # Calculate Absolute Price Change
    data['Price_Change'] = data['close'].diff()

    # Calculate Volume Change (percentage change from the previous day's volume)
    data['Volume_Change'] = data['volume'].pct_change() * 100

    # Calculate Rolling Average Volume (7-day and 30-day)
    data['Rolling_Avg_Volume_7'] = data['volume'].rolling(window=7).mean()
    data['Rolling_Avg_Volume_30'] = data['volume'].rolling(window=30).mean()

    # Calculate Volume Ratio (Current volume divided by the rolling average volume)
    data['Volume_Ratio_7'] = data['volume'] / data['Rolling_Avg_Volume_7']
    data['Volume_Ratio_30'] = data['volume'] / data['Rolling_Avg_Volume_30']

    # Calculate Daily Price Range
    data['Price_Range'] = data['close'] - data['close'].shift(1)

    # Calculate Rolling Volatility (Standard deviation of daily returns over a specified period)
    data['Rolling_Volatility_7'] = data['Daily_Return'].rolling(window=7).std()
    data['Rolling_Volatility_30'] = data['Daily_Return'].rolling(window=30).std()

    # Calculate Market Cap Change (percentage change from the previous day's market cap)
    data['Market_Cap_Change'] = data['market_cap'].pct_change() * 100

    # Calculate Rolling Average Market Cap (7-day and 30-day)
    data['Rolling_Avg_Market_Cap_7'] = data['market_cap'].rolling(window=7).mean()
    data['Rolling_Avg_Market_Cap_30'] = data['market_cap'].rolling(window=30).mean()

    # Export to CSV
    export_data_to_csv( data, f"{csv_file}_cleaned")


if __name__ == '__main__': 
    data_cleanup()
