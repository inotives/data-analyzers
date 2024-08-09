from scripts.price_forecast import PriceForecast


if __name__ == '__main__':

    # try forecasting BTC price with OHLCV daily data 
    forecast = PriceForecast('btc_ohlcv')
    forecast_prices = forecast.run_simple_forecast(model='SARIMA', plot_chart=False)
    print(forecast_prices)