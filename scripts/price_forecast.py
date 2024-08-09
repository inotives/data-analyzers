import pandas as pd 
import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils.tools import load_csv_from_data, plotly_line_chart

class PriceForecast():
    
    def __init__(self, data_src):
        self.data_src = data_src
    
    def load_and_prep_data(self):
        # load bitcoin ohlcv 
        csv_file = self.data_src
        df = load_csv_from_data(csv_file)

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Sort the data by timestamp
        df.sort_index(inplace=True)

        # Add moving average feature
        df['moving_avg'] = df['close'].rolling(window=3).mean()

        # Prepare the data
        # Create lagged features for prediction
        df['prev_close'] = df['close'].shift(1)
        df.dropna(inplace=True)  # Drop rows with NaN values

        return df
    
    def run_simple_forecast(self, model='ARIMA', day=3, plot_chart=False):
        data = self.load_and_prep_data()

        if model == 'ARIMA': 
            forecast_price = self.price_predict_with_ARIMA(data, forecast_day=day, plot_chart=plot_chart)
            return forecast_price
        elif model == 'SARIMA': 
            forecast_price = self.price_predict_with_SARIMA(data, forecast_day=day, plot_chart=plot_chart)
            return forecast_price

    def price_predict_with_ARIMA(self, data, forecast_day=7, plot_chart=False):
        ''' ARIMA (Auto Regressive Integrated Moving Average)
        It is a class of models that explains a given time series based on its own past values, 
        its own past errors, and it integrates differencing to make the time series stationary.

        KEY COMPONENT of ARIMA: 
        1) AR (AutoRegressive) Component
        p: This represents the number of lag observations included in the model (i.e., the number of terms). 
            It shows the relationship between an observation and some number of lagged observations.
        
        2) I (Integrated) Component
        d: This represents the number of times the data have had past values subtracted to make 
            the series stationary (i.e., the number of differencing required to remove trends and seasonality). 
        
        3) MA (Moving Average) Component
        q: This represents the number of lagged forecast errors in the prediction equation.

        HOW ARIMA WORKS: 
        1. Identify if the time series is stationary. If not, apply differencing until it becomes stationary. 
        This determines the order d. Use autocorrelation function (ACF) and 
        partial autocorrelation function (PACF) plots to determine the appropriate values for p and q.
        2. Estimate the coefficients used in AR and MA Components using methods like maximum likelihood estimation.
        3. Check if the residuals (errors) of the model resemble white noise, 
        meaning they are independently and identically distributed with a mean of zero.
        If the residuals are not white noise, adjust the model accordingly.
        4. Use the fitted ARIMA model to forecast future values.
        '''

        # Ensure the index has a daily frequency
        data = data.asfreq('D')

        # We will use the closing prices for ARIMA modeling
        close_prices = data['close']

        # Fit ARIMA model
        p = 1  # the number of lag observations included in the model (AR component).
        d = 1  # the number of times that the raw observations are differenced (I component). number of differencing
        q = 30  #  the size of the moving average window (MA component).
        model = ARIMA(close_prices, order=(p, d, q))
        model_fit = model.fit()

        # Forecast
        forecast_horizon = forecast_day  # number of days you want to forecast
        forecast = model_fit.get_forecast(steps=forecast_horizon)
        forecasted_mean = forecast.predicted_mean
        forecasted_conf_int = forecast.conf_int()

        # Create a date range for the forecasted prices
        future_dates = pd.date_range(start=close_prices.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

        # Create DataFrame for forecasted prices
        forecast_df = pd.DataFrame({
            'predicted_price': forecasted_mean,
            'lower_conf_int': forecasted_conf_int.iloc[:, 0],
            'upper_conf_int': forecasted_conf_int.iloc[:, 1]
        }, index=future_dates)

        # Historical forecast to check model fit
        historical_forecast = model_fit.get_prediction(start=0, end=len(close_prices)-1)
        historical_forecast_mean = historical_forecast.predicted_mean
        historical_conf_int = historical_forecast.conf_int()
        # Create DataFrame for historical forecasted prices
        historical_forecast_df = pd.DataFrame({
            'predicted_price': historical_forecast_mean,
            'lower_conf_int': historical_conf_int.iloc[:, 0],
            'upper_conf_int': historical_conf_int.iloc[:, 1]
        }, index=close_prices.index)

        print(model_fit.summary())

        plot_data = [
            {"xvals": close_prices.index, 'yvals': close_prices, 'label': 'Actual Closing Price', 'marker': ',', 'plotly_mode': 'lines'},
            {"xvals": historical_forecast_df.index, 'yvals': historical_forecast_df['predicted_price'], 'label': 'Historical Forecasted Price', 'marker': 'x', 'plotly_mode': 'lines+markers'},
            {"xvals": forecast_df.index, 'yvals': forecast_df['predicted_price'], 'label': 'Forecasted Closing Price', 'marker': 'x', 'plotly_mode': 'lines+markers'}
        ]

        fig = plotly_line_chart(plot_data, 'ARIMA Model Forecast', 'Date', 'Price') if plot_chart else None
        if fig is not None: 
            fig.plot()

        return forecast_df


    def price_predict_with_SARIMA(self, data, forecast_day=7, plot_chart=False): 
        ''' SARIMA 

        The confidence interval (CI) is a range of values that is used to estimate the uncertainty surrounding 
        a predicted value or parameter estimate. In the context of time series forecasting 
        with models like ARIMA or SARIMA, the confidence interval provides a range within which 
        the true future values of the time series are expected to fall with a certain probability.
        
        Upper and Lower Bounds: The confidence interval is defined by two bounds:
            - Lower Bound (Lower Confidence Limit): The minimum value within the interval.
            - Upper Bound (Upper Confidence Limit): The maximum value within the interval.
        Interpretation: The wider the confidence interval, the greater the uncertainty in the prediction. 
        A narrow confidence interval indicates more precise predictions.
        '''

        # Ensure the index has a daily frequency
        data = data.asfreq('D')

        # We will use the closing prices for SARIMA modeling
        close_prices = data['close']

        # Define the SARIMA model parameters
        p, d, q = 1, 1, 1  # Non-seasonal parameters 
        P, D, Q, m = 1, 1, 1, 24  # Seasonal parameters (e.g., m=12 for monthly seasonality)

        # Fit SARIMA model
        model = SARIMAX(close_prices, order=(p, d, q), seasonal_order=(P, D, Q, m))
        model_fit = model.fit(disp=False, method='powell') # other method: powell, lbfgs 

        # Forecast
        forecast_horizon = forecast_day  # number of days you want to forecast
        forecast = model_fit.get_forecast(steps=forecast_horizon)
        forecasted_mean = forecast.predicted_mean
        forecasted_conf_int = forecast.conf_int()

        # Create a date range for the forecasted prices
        future_dates = pd.date_range(start=close_prices.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

        # Create DataFrame for forecasted prices
        forecast_df = pd.DataFrame({
            'predicted_price': forecasted_mean,
            'lower_conf_int': forecasted_conf_int.iloc[:, 0],
            'upper_conf_int': forecasted_conf_int.iloc[:, 1]
        }, index=future_dates)

        # Historical forecast to check model fit
        historical_forecast = model_fit.get_prediction(start=0, end=len(close_prices)-1)
        historical_forecast_mean = historical_forecast.predicted_mean
        historical_conf_int = historical_forecast.conf_int()

        # Create DataFrame for historical forecasted prices
        historical_forecast_df = pd.DataFrame({
            'predicted_price': historical_forecast_mean,
            'lower_conf_int': historical_conf_int.iloc[:, 0],
            'upper_conf_int': historical_conf_int.iloc[:, 1]
        }, index=close_prices.index)

        print(model_fit.summary())
        
        plot_data = [
            {"xvals": close_prices.index, 'yvals': close_prices, 'label': 'Actual Closing Price', 'marker': ',', 'plotly_mode': 'lines'},
            {"xvals": historical_forecast_df.index, 'yvals': historical_forecast_df['predicted_price'], 'label': 'Historical Forecasted Price', 'marker': 'x', 'plotly_mode': 'lines+markers'},
            {"xvals": forecast_df.index, 'yvals': forecast_df['predicted_price'], 'label': 'Forecasted Closing Price', 'marker': 'x', 'plotly_mode': 'lines+markers'}
        ]

        fig = plotly_line_chart(plot_data, 'SARIMA Model Forecast', 'Date', 'Price') if plot_chart else None

        # Additionally, you can plot confidence intervals
        if fig is not None: 
            fig.add_trace(go.Scatter(
                x=forecast_df.index, 
                y=forecast_df['lower_conf_int'], 
                fill=None, showlegend=False,
                mode='lines', line_color='rgba(255, 0, 0, 0.5)'            
            ))
            fig.add_trace(go.Scatter(
                x=forecast_df.index, 
                y=forecast_df['upper_conf_int'], 
                name='Confidence Interval', line_color='rgba(255, 0, 0, 0.5)', 
                mode='lines', fill='tonexty'  # fill area between lower and upper confidence interval
            ))
            fig.update_layout(
                title='SARIMA Model Forecast with Confidence Intervals', xaxis_title='Date', yaxis_title='Price'
            )
            # Plot the chart
            fig.show()

        return forecast_df

    
    
    # Simple Linear Regression too overfit. 
