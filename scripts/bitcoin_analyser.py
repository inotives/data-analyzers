import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
# from sklearn.preprocessing import MinMaxScaler

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from xgboost import XGBRegressor

from arch import arch_model

from utils.tools import load_csv_from_data, plotly_line_chart


def load_and_prep_data():
    # load bitcoin ohlcv 
    csv_file = 'btc_ohlcv'
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


def run_prediction():

    # load and prep data
    data = load_and_prep_data()

    forecast_price = price_predict_with_SARIMA(data, forecast_day=30, plot_chart=True)

    print(forecast_price)

    return 

def price_predict_with_SARIMA(data, forecast_day=7, plot_chart=False): 
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



def price_predict_with_ARIMA(data, forecast_day=7, plot_chart=False):
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


# W.I.P - Still have issue need more tweaking ...
def price_predict_with_GARCH(data, forecast_day=7, plot_chart=False): 
    '''

    GARCH (Generalized Autoregressive Conditional Heteroskedasticity) is a statistical model used to analyze and 
    forecast time series data where the variability (or volatility) of the data changes over time. 
    It is commonly used in financial markets to model the volatility of asset returns.

    KEY CONCEPT: 
    1) Volatility Clustering: Financial time series often exhibit periods of high volatility followed by periods of low volatility.
       GARCH models capture this clustering effect, 
       where periods of HIGH volatility are likely to be followed by more HIGH volatility, and vice versa.
    
    2) Conditional Variance: In a GARCH model, the variance of the time series is modeled as being dependent on 
       past values of the series (autoregressive part) and 
       past forecast errors (moving average part).
    
    3) GARCH(p, q) Model: 
        - p: Number of past squared returns (lagged terms) used to model the current variance.
        - q: Number of past forecast errors used to model the current variance.
       The model specifies that the current variance depends on past values and past errors, 
       incorporating both autoregressive (AR) and moving average (MA) components.
    
    '''
    p = 1
    q = 1

    # Drop non-numeric columns
    data = data.drop(columns=['name'])

    # Fit GARCH model
    returns = data['close'].pct_change().dropna()  # Calculate returns

    # Rescale the returns - scale of returns should be between 1 and 1000 for better convergence.
    returns_scaled = 1 * returns

    model = arch_model(returns_scaled, vol='Garch', p=p, q=q)
    model_fit = model.fit(disp='off')

    # Forecast
    forecast_horizon = forecast_day
    forecast = model_fit.forecast(horizon=forecast_horizon)

    # Extract forecasted returns and variances
    forecasted_returns = forecast.mean.iloc[-1]  # Forecasted returns for the future horizon
    forecasted_variances = forecast.variance.iloc[-1]  # Forecasted variances for the future horizon

    # Initial price for the prediction
    last_price = data['close'].iloc[-1]

    # Calculate forecasted prices
    forecasted_prices = [last_price * (1 + ret) for ret in forecasted_returns]

    # Create a date range for the forecasted prices
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon, freq='D')

    # Create DataFrame for forecasted prices
    forecast_df = pd.DataFrame({
        'predicted_price': forecasted_prices
    }, index=future_dates)

    # Past Historical Forecast
    forecast_historical = model_fit.forecast(start=0)
    forecasted_mean_historical = forecast_historical.mean
    forecasted_variance_historical = forecast_historical.variance

    # Align forecasted data with the original data index
    forecasted_mean_historical = forecasted_mean_historical.loc[data.index[1:]]
    forecasted_variance_historical = forecasted_variance_historical.loc[data.index[1:]]

    print(model_fit.summary())

    # Plotting the chart 
    plot_data = [
        {"xvals": data.index, 'yvals': data['close'], 'label': 'Actual Closing Price', 'marker': ',', 'plotly_mode': 'lines'},
        {"xvals": forecast_df.index, 'yvals': forecast_df['predicted_price'], 'label': 'Forecasted Closing Price', 'marker': 'x', 'plotly_mode': 'lines+markers'},
        {"xvals": forecasted_mean_historical.index, 'yvals': forecasted_mean_historical.values.flatten(), 'label': 'Predicted Price Hist', 'marker': 'x', 'plotly_mode': 'lines+markers'}
    ]

    plotly_line_chart(plot_data, 'GARCH Model Forecast', 'Date', 'Price') if plot_chart else None

    return



def simple_linear_regression_model(df, plot_eval=False): 

    # Define independent and dependent variables
    X = df[['open', 'high', 'low', 'volume', 'prev_close']]
    y = df['close']

    # Check for multicollinearity
    vif_data = check_multicollinearity(X)
    print(vif_data)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Perform RFE to select features
    selector, selected_features = perform_rfe(X_train, y_train, n_features_to_select=3)
    print("Selected Features: ", selected_features)

    # Train and evaluate the model with selected features
    model = train_model(X_train, X_test, y_train, y_test, selected_features)
    y_pred = model.predict(X_test[selected_features])
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error with selected features: {mse}")

    # Display coefficients of the model with selected features
    coefficients = pd.DataFrame(model.coef_, selected_features, columns=['Coefficient'])
    print(coefficients)

    plot_chart(X_test, y_test, y_pred) if plot_eval else None

    return model

def polynomial_regression_model(plot_eval=True, degree=1):
    '''
    Polynomial regression is a type of regression analysis where the relationship between the independent variable x 
    and the dependent variable y is modeled as an n-th degree polynomial. 
    It is used when the data shows a non-linear relationship.
    '''

    data = load_and_prep_data()

    # Define features and target
    features = ['open', 'high', 'low', 'volume']
    target = 'close'

    X = data[features]
    y = data[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Polynomial feature transformation
    poly = PolynomialFeatures(degree=degree)  # Adjust degree as needed
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Train polynomial regression model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # Make predictions
    y_pred = model.predict(X_test_poly)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # plot evaluation 
    plot_chart(X_test,y_test,y_pred) if plot_eval else None

    return model



def xgboost_model():
    """

    """

    df = load_and_prep_data()

    # Define independent and dependent variables
    X = df[['open', 'high', 'low', 'volume', 'prev_close']]
    y = df['close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error with XGBoost: {mse}")

    # Feature importance
    importances = model.feature_importances_
    for i, feature in enumerate(X.columns):
        print(f"{feature}: {importances[i]}")

    plot_chart(X_test, y_test, y_pred)


def simple_decision_tree_model():

    '''
    Simple Decision Tree Models
    Decision Trees are simple models that work by splitting the data into subsets based on feature values. They are easy to interpret but can overfit if not properly controlled.
    Base on MSE return, as expected the performance are bad ... 
    '''

    df = load_and_prep_data()

    # Define independent and dependent variables
    X = df[['open', 'high', 'low', 'volume', 'prev_close', 'moving_avg']]
    y = df['close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the Decision Tree model
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error with Decision Tree: {mse}")

    plot_chart(X_test, y_test, y_pred)


def random_forest_model():
    '''
    Random Forests are an ensemble method that combines multiple decision trees to improve accuracy and reduce overfitting.
    '''
    df = load_and_prep_data()

    # Define independent and dependent variables
    X = df[['open', 'high', 'low', 'volume', 'prev_close', 'moving_avg']]
    y = df['close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Initialize and train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error with Random Forest: {mse}")

    # Feature importance
    importances = model.feature_importances_
    for i, feature in enumerate(X.columns):
        print(f"{feature}: {importances[i]}")
    
    plot_chart(X_test, y_test, y_pred)


def ARIMA_model():
    '''

    '''

    df = load_and_prep_data()

    # Set the frequency of the index
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')  # 'D' for daily frequency, adjust if needed

    # Define the target variable
    y = df['close']

    # Apply differencing to make the series stationary
    y_diff = y.diff().dropna()


    # Check stationarity on differenced data
    result_diff = adfuller(y_diff)
    print('ADF Statistic after Differencing:', result_diff[0])
    print('p-value after Differencing:', result_diff[1])

    # Split the data into training and testing sets
    train_size = int(len(y) * 0.8)
    train, test = y[:train_size], y[train_size:]

    # Initialize and fit the ARIMA model
    model = ARIMA(train, order=(5, 1, 0))  # Example order (p, d, q)
    model_fit = model.fit()

    # Make predictions
    y_pred = model_fit.forecast(steps=len(test))
    print(y_pred)

    # Evaluate the model
    mse = mean_squared_error(test, y_pred)
    print(f"Mean Squared Error with ARIMA: {mse}")

    plot_chart(df, test, y_pred)

def is_stationary(timeseries):
    """
    This function checks for stationarity of a time series using the Dickey-Fuller test.
    """
    adf_result = adfuller(timeseries)
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")

    # Check for significance level (usually alpha=0.05)
    if adf_result[1] > 0.05:
        return False
    else:
        return True
  

def SARIMA_model(plot_eval=True):
    '''
    '''

    df = load_and_prep_data()
    print(df.head())

    # Set the frequency of the index
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')  # 'D' for daily frequency, adjust if needed

    # Define the target variable
    y = df['close']

    # Split the data into training and testing sets
    train_size = int(len(y) * 0.8)
    train, test = y[:train_size], y[train_size:]

    # Set the frequency of the index
    df.index = pd.to_datetime(df.index)
    df = df.asfreq('D')  # 'D' for daily frequency, adjust if needed

    # Define the target variable
    y = df['close']

      # Check stationarity
    d = 1  # Initial differencing parameter
    while not is_stationary(y.diff(d)):
        d += 1

    # SARIMA Model with differenced data
    sarima_order = (5, d, 3)  # Example order (p, d, q)
    seasonal_order = (1, 1, 1, 7)  # Example seasonal order (P, D, Q, S)
    model = SARIMAX(train, order=sarima_order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Make predictions
    y_pred = model_fit.forecast(steps=len(test))

    # Evaluate the model
    mse_sarima = mean_squared_error(test, y_pred)
    print(f"Mean Squared Error with SARIMA: {mse_sarima}")

    plot_chart(df, test, y_pred) if plot_eval else None


    
 
def plot_chart(x_test, y_test, y_pred):
    """
    Plot chart Actual Historical data vs Predictor
    """
    plt.figure(figsize=(14, 7))
    plt.plot(x_test.index[-len(y_test):], y_test, label='Actual Closing Price', marker='o')
    plt.plot(x_test.index[-len(y_test):], y_pred, label='Predicted Closing Price', linestyle='--', marker='x')
    plt.title('Actual vs Predicted Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def check_multicollinearity(X):
    """
    Check for multicollinearity using VIF.
    Variables with low coefficients might still be important in the presence of other variables. 
    High multicollinearity (when independent variables are highly correlated with each other) can affect the interpretability 
    and performance of the model. Checking the Variance Inflation Factor (VIF) can help identify multicollinearity issues.

    High VIF (>10) indicates high multicollinearity. Consider removing variables with high VIF.
    """
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
    return vif_data

def perform_rfe(X_train, y_train, n_features_to_select):
    """
    Perform Recursive Feature Elimination (RFE) to select features.
    You can use Recursive Feature Elimination (RFE) to select features based on their importance.
    Automatically selects a subset of features based on their importance.

    RFE selects the most important features. The selected features should improve or maintain model performance.
    """
    model = LinearRegression()
    selector = RFE(model, n_features_to_select=n_features_to_select)
    selector = selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.support_]
    return selector, selected_features

def train_model(X_train, X_test, y_train, y_test, selected_features):
    """Train the model and evaluate its performance."""
    model = LinearRegression()
    model.fit(X_train[selected_features], y_train)
    
    return model