import sys
from scripts.price_forecast import PriceForecast
from scripts.crypto_ohlcv import enchanced_ohlcv_with_techinical_indicators, process_stablecoins_ohlcv
from scripts.sentimental_analysis import run_sa

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

        num_day = 7
        if len(sys.argv) >= 3: 
            num_day = int(sys.argv[2])
        
        forecast_prices = forecast.run_forecast(model='SARIMAX', plot_chart=False, day=num_day)
        print(forecast_prices)

    elif command == 'sa': 
        run_sa()
    else: 
        print('>> Not A valid command !\n Available Command: ohlcv, forecast')
