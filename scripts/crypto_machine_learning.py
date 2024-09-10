from scripts.crypto_ohlcv import load_crypto_ohlcv_from_db
import pandas as pd
import ta
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to plot actual vs predicted close prices and the trend line
def plot_chart(df_test, trend, df_trend_line):
    fig = go.Figure()

    # Add actual close prices
    fig.add_trace(go.Scatter(
        x=df_test['metric_date'], y=df_test['actual_close'],
        mode='lines', name='Actual Close'
    ))

    # Add predicted close prices
    fig.add_trace(go.Scatter(
        x=df_test['metric_date'], y=df_test['predicted_close'],
        mode='lines', name='Predicted Close'
    ))

    # Add trend line
    fig.add_trace(go.Scatter(
        x=df_trend_line['metric_date'], y=df_trend_line['trend_line'],
        mode='lines', name='Trend Line',
        line=dict(color='firebrick', width=2, dash='dash')
    ))

    # Customize the chart
    fig.update_layout(
        title=f'Bitcoin Close Price Prediction with Trend Line ({trend.capitalize()} Trend)',
        xaxis_title='Date',
        yaxis_title='Close Price',
        hovermode='x'
    )

    fig.show()

# Function to perform linear regression on OHLCV data
def linear_regression():
    df = load_crypto_ohlcv_from_db('Bitcoin')

    # Convert metric_date to datetime format and use it as a numeric feature (days since start)
    df['metric_date'] = pd.to_datetime(df['metric_date'])
    df['days_since_start'] = (df['metric_date'] - df['metric_date'].min()).dt.days

    # Calculate moving averages using the `ta` library
    df['ma_7'] = ta.trend.sma_indicator(df['close'], window=7)
    df['ma_30'] = ta.trend.sma_indicator(df['close'], window=30)

    # Drop rows with NaN values (due to moving averages)
    df = df.dropna()

    # Prepare features (include days_since_start and moving averages)
    feature_columns = ['open', 'high', 'low', 'volume', 'days_since_start', 'ma_7', 'ma_30']
    X = df[feature_columns]
    y = df['close']

    # Split data into training and testing sets, preserving indices
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Scale the features while preserving the index
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=X_test.columns
    )

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Predict the close prices on the test set
    y_pred = model.predict(X_test_scaled)

    # Create a DataFrame for the test set
    df_test = df.loc[X_test.index].copy()
    df_test['actual_close'] = y_test.values  # Use the actual close prices from y_test
    df_test['predicted_close'] = y_pred

    # Get the coefficient for the time feature (days_since_start) to analyze the trend
    trend_slope = model.coef_[feature_columns.index('days_since_start')]

    # Determine the trend direction
    trend = 'upward' if trend_slope > 0 else 'downward'

    # Calculate the trend line over the entire date range
    df_trend_line = df[['metric_date', 'days_since_start']].copy()

    # Scale 'days_since_start' using the same scaler
    days_since_start_scaled = scaler.transform(df[feature_columns])[:, feature_columns.index('days_since_start')]

    # Calculate the trend line values
    trend_line_values = model.intercept_ + trend_slope * days_since_start_scaled

    df_trend_line['trend_line'] = trend_line_values

    # Print model evaluation metrics
    print("Model Evaluation Summary:")
    print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")
    print(f"R-squared (R²) Score: {r2_score(y_test, y_pred):.2f}")

    # Plot the actual vs predicted close prices and the trend line
    plot_chart(df_test, trend, df_trend_line)

    return trend, y_pred

