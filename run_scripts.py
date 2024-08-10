import sys
from scripts.price_forecast import PriceForecast
from scripts.crypto_ohlcv import enchanced_ohlcv_with_techinical_indicators, process_stablecoins_ohlcv

if __name__ == '__main__':

    command = ''
    try:
        command = sys.argv[1]
    except IndexError:
        print ('after run_script.py need a command. Avail-CMD: ohlcv, forecast')
    
    if command == 'ohlcv':
        process_stablecoins_ohlcv()
        enchanced_ohlcv_with_techinical_indicators('cryptos_blw1_ohlcv')
    elif command == 'forecast':
        # try forecasting BTC price with OHLCV daily data 
        forecast = PriceForecast('btc_ohlcv')
        
        forecast_prices = forecast.run_forecast(model='SARIMAX', plot_chart=True, day=7)
        print(forecast_prices)
    else: 
        print('>> Not A valid command !\n Available Command: ohlcv, forecast')
