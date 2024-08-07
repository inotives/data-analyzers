import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

from xgboost import XGBRegressor


from utils.tools import load_csv_from_data


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


def simple_linear_regression_model(chart=False): 
    
    df = load_and_prep_data()

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

    if chart: 
        plot_chart(X_test, y_test, y_pred)

    return model

def polynomial_regression_model():
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
    poly = PolynomialFeatures(degree=1)  # Adjust degree as needed
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

    # Plotting Actual vs Predicted Prices
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual Prices', marker='o')
    plt.plot(y_test.index, y_pred, label='Predicted Prices', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Actual vs Predicted Prices (Polynomial Regression)')
    plt.legend()
    plt.show()



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
    
def SARIMA_model():
    '''
    '''

    df = load_and_prep_data()

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

    # SARIMA Model with differenced data
    sarima_order = (5, 1, 0)  # Example order (p, d, q)
    seasonal_order = (1, 1, 1, 7)  # Example seasonal order (P, D, Q, S)
    model = SARIMAX(train, order=sarima_order, seasonal_order=seasonal_order)
    model_fit = model.fit()

    # Make predictions
    y_pred = model_fit.forecast(steps=len(test))

    # Evaluate the model
    mse_sarima = mean_squared_error(test, y_pred)
    print(f"Mean Squared Error with SARIMA: {mse_sarima}")

    plot_chart(df, test, y_pred)


    
 
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