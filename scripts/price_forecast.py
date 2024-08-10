import pandas as pd 

import plotly.graph_objects as go

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from arch import arch_model

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

        # Calculate price changes
        df['price_change'] = df['close'] - df['open']
        df['high_low_diff'] = df['high'] - df['low']

        # Add moving average feature
        df['moving_avg'] = df['close'].rolling(window=3).mean()

        # Calculate RSI (Relative Strength Index) using pandas
        delta = df['close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Calculate Bollinger Bands
        df['middle_band'] = df['close'].rolling(window=20).mean()
        df['std_dev'] = df['close'].rolling(window=20).std()
        df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
        df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)

        # Prepare the data
        # Create lagged features for prediction
        df['prev_close'] = df['close'].shift(1)
        df['prev_price_change'] = df['price_change'].shift(1)
        df['prev_volume'] = df['volume'].shift(1)

        df.dropna(inplace=True)  # Drop rows with NaN values

        return df
    
    def run_forecast(self, model='ARIMA', day=7, plot_chart=False):
        data = self.load_and_prep_data()

        if model == 'ARIMA': 
            forecast_price = self.price_forecast_with_ARIMA(data, forecast_day=day, plot_chart=plot_chart)
            return forecast_price
        elif model == 'SARIMAX': 
            forecast_price = self.price_forecast_with_SARIMAX(data, forecast_day=day, plot_chart=plot_chart)
            return forecast_price
        elif model == 'SARIMAX_GARCH':
            # still need more tweaking on the parameters ...
            forecast_price = self.price_forecast_with_SARIMAX_GARCH(data, forecast_day=day, plot_chart=plot_chart)
            return forecast_price
        else:
            return None



    '''
    --------------------------------------------------------------------------------------------------------
    Forecasting Models 
    --------------------------------------------------------------------------------------------------------
    '''

    def price_forecast_with_ARIMA(self, data, forecast_day=7, plot_chart=False):
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

        # Use closing prices, price changes, or other derived features
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


    def price_forecast_with_SARIMAX(self, data, forecast_day=7, plot_chart=False): 
        ''' SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)
        an extension of the ARIMA (AutoRegressive Integrated Moving Average) model that allows 
        for the inclusion of both seasonal components and external (exogenous) variables 
        in time series forecasting.

        KEY COMPONENTS: 
        1. ARIMA Components: 
            - The AR component captures the relationship between an observation and a number of lagged observations.
              represented as p. 
            - The I component deals with the differencing of observations to make the time series stationary.
              represent as d 
            - The MA component captures the relationship between an observation and a residual error 
              from a moving average model applied to lagged observations.
              represent as q
        2. Seasonal Components:
        - SARIMAX can model seasonality, which is the repeating patterns or cycles in time series data, 
          typically over a fixed period like a year, month, week, etc.
        - Seasonal AR (P), Seasonal I (D), and Seasonal MA (Q) are analogous to 
          the non-seasonal AR, I, and MA components but are applied to seasonal patterns.
        - Seasonal Period (S): The length of the seasonal cycle (e.g., 12 for monthly data with yearly seasonality).

        3. eXogenous Variables (X):
        - SARIMAX allows the inclusion of exogenous variables, which are external factors 
          that can influence the target time series but are not necessarily part of it.
        - These variables are represented by X, and could include metrics like GDP, interest rates, 
          weather data, or other related time series that might impact the target variable.
        - Including exogenous variables allows the model to account for these external influences, 
          potentially improving the accuracy of the forecast.
        
        HOW SARIMAX WORKS: 
        1. The model first difference the data if required (based on d and D) to achieve stationary. 
           This step removes trends and seasonality, leaving a more stable time series.
        2. The model then uses the AR and MA components to capture relationships between 
           the lagged observations and the residual errors.
        3. The seasonal components capture the repeating patterns in the data over 
           the defined seasonal period S
        4. The exogenous variables are included to account for their influence on the target time series, 
           providing additional predictive power.
        5. The SARIMAX model is then fitted to the historical data, learning the relationships 
           between the time series data and the exogenous variables.
        6. Once fitted, the model can be used to forecast future values, considering both 
           the inherent patterns in the time series and the influence of exogenous variables.


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
        # close_prices = data['close']

        # Exogenous variables (e.g., RSI, Bollinger Bands, Volume, etc.)
        exog = data[['rsi', 'upper_band', 'lower_band', 'volume']] 

        # Target variable (close prices)
        target_series = data['close']

        # Fit SARIMAX model with exogenous variables
        p = 1  # AR component
        d = 1  # I component
        q = 7 # MA component
        P, Q, D, S = (1, 1, 1, 12)
        model = SARIMAX(
            target_series, 
            exog=exog, 
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(
            method='powell',       # You can try different methods such as 'nm' or 'powell', 'lbfgs'
            maxiter=500,          # Increase the number of iterations
            disp=True             # Display convergence information
        )

        # Forecast
        forecast_horizon = forecast_day
        forecast = model_fit.get_forecast(steps=forecast_horizon, exog=exog.iloc[-forecast_horizon:])
        forecasted_mean = forecast.predicted_mean
        forecasted_conf_int = forecast.conf_int()

        # Create a date range for the forecasted prices
        future_dates = pd.date_range(start=target_series.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

        # Create DataFrame for forecasted prices
        forecast_df = pd.DataFrame({
            'predicted_price': forecasted_mean,
            'lower_conf_int': forecasted_conf_int.iloc[:, 0],
            'upper_conf_int': forecasted_conf_int.iloc[:, 1]
        }, index=future_dates)

        # Historical forecast to check model fit
        historical_forecast = model_fit.get_prediction(start=0, end=len(target_series)-1)
        historical_forecast_mean = historical_forecast.predicted_mean
        historical_conf_int = historical_forecast.conf_int()

        # Create DataFrame for historical forecasted prices
        historical_forecast_df = pd.DataFrame({
            'predicted_price': historical_forecast_mean,
            'lower_conf_int': historical_conf_int.iloc[:, 0],
            'upper_conf_int': historical_conf_int.iloc[:, 1]
        }, index=target_series.index)

        print(model_fit.summary())
        
        plot_data = [
            {"xvals": target_series.index, 'yvals': target_series, 'label': 'Actual Closing Price', 'marker': ',', 'plotly_mode': 'lines'},
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

    
    
    def price_forecast_with_SARIMAX_GARCH(self, data, forecast_day, plot_chart=False): 

        ''' SARIMAX + GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
        To enhance your current SARIMAX model with a GARCH model, 
        the idea is to first fit the SARIMAX model to capture the mean structure and 
        then use the residuals from the SARIMAX model as input to a GARCH model 
        to capture the volatility structure. Finally, the combined model can provide more 
        accurate predictions, especially in financial time series like cryptocurrency prices, 
        where volatility is a significant factor.

        GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is a class of statistical models 
        used to estimate and forecast the volatility of time series data. It is particularly useful 
        in financial markets where volatility (the degree of variation of a trading price series over time) 
        tends to fluctuate.

        Key Concepts
        -------------
        1. Volatility Clustering: Financial time series data often exhibit periods of high volatility 
           followed by periods of low volatility. GARCH models capture this phenomenon, 
           known as volatility clustering.
        2. Conditional Heteroskedasticity: GARCH models assume that volatility is not constant but 
           varies over time in a predictable manner. This means that the variance (volatility) of 
           the time series is conditional on past information.
        3. Autoregressive Structure: GARCH models use past values of the series and past values of 
           the conditional variance to model the current variance. This autoregressive structure 
           helps capture the dynamic nature of volatility.
        


        '''

        # Ensure the index has a daily frequency
        data = data.asfreq('D')

        # Exogenous variables (e.g., RSI, Bollinger Bands, Volume, etc.)
        exog = data[['rsi', 'upper_band', 'lower_band', 'volume']]

        # Target variable (close prices)
        target_series = data['close']

        # Step 1: Fit SARIMAX model with exogenous variables
        p = 1  # AR component
        d = 1  # Differencing component
        q = 7  # MA component
        model = SARIMAX(target_series, exog=exog, order=(p, d, q))
        sarimax_fit = model.fit()

        # Step 2: Extract residuals from SARIMAX model
        residuals = sarimax_fit.resid
        forecast_horizon=forecast_day

        # Rescale residuals for GARCH model fitting
        residuals_scaled = residuals / residuals.std()

        # Step 3: Fit GARCH model to the scaled residuals
        garch_model = arch_model(residuals_scaled, vol='Garch', p=1, q=1, rescale=True)
        try:
            garch_fit = garch_model.fit(disp='off')
        except Exception as e:
            print(f"GARCH model fitting failed: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of failure

        # Step 4: Generate volatility forecasts using GARCH model
        garch_forecast = garch_fit.forecast(horizon=forecast_horizon)
        volatility_forecast = garch_forecast.variance.iloc[-1] ** 0.5  # Standard deviation

        # Check for NaN in volatility forecast and handle it
        if volatility_forecast.isna().any():
            print("Warning: GARCH model returned NaN for volatility forecast.")
            volatility_forecast = pd.Series([0] * forecast_horizon)  # Fallback to zero volatility

        # Step 5: Generate forecasts using SARIMAX model
        # Rescale volatility forecast to original scale
        volatility_forecast = volatility_forecast * residuals.std()

        # Step 5: Generate forecasts using SARIMAX model
        forecast = sarimax_fit.get_forecast(steps=forecast_horizon, exog=exog.iloc[-forecast_horizon:])
        forecasted_mean = forecast.predicted_mean

        # Combine SARIMAX forecast with GARCH volatility forecast
        forecast_df = pd.DataFrame({
            'predicted_price': forecasted_mean,
            'lower_conf_int': forecasted_mean - 1.96 * volatility_forecast,
            'upper_conf_int': forecasted_mean + 1.96 * volatility_forecast
        }, index=pd.date_range(start=target_series.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D'))

        # Historical forecast to check model fit
        historical_forecast = sarimax_fit.get_prediction(start=0, end=len(target_series) - 1)
        historical_forecast_mean = historical_forecast.predicted_mean
        historical_conf_int = historical_forecast.conf_int()

        # Create DataFrame for historical forecasted prices
        historical_forecast_df = pd.DataFrame({
            'predicted_price': historical_forecast_mean,
            'lower_conf_int': historical_conf_int.iloc[:, 0],
            'upper_conf_int': historical_conf_int.iloc[:, 1]
        }, index=target_series.index)

        print(sarimax_fit.summary())
        print(garch_fit.summary())

        # Plot results
        plot_data = [
            {"xvals": target_series.index, 'yvals': target_series, 'label': 'Actual Closing Price', 'marker': ',', 'plotly_mode': 'lines'},
            {"xvals": historical_forecast_df.index, 'yvals': historical_forecast_df['predicted_price'], 'label': 'Historical Forecasted Price', 'marker': 'x', 'plotly_mode': 'lines+markers'},
            {"xvals": forecast_df.index, 'yvals': forecast_df['predicted_price'], 'label': 'Forecasted Closing Price', 'marker': 'x', 'plotly_mode': 'lines+markers'}
        ]
        fig = plotly_line_chart(plot_data, 'SARIMAX + GARCH Model Forecast', 'Date', 'Price') if plot_chart else None
        if fig is not None:
            # can add additional chart plotting here before display the chart.
            # Plot the chart
            fig.show()

        return forecast_df