## DATA-ANALYZERS

This repository contains all the code and scripts I used for analyzing data collected from the [data-collectors](https://github.com/inotives/data-collectors) repository. Here, I experiment with the collected data using various machine learning techniques, technical indicators, statistical models, forecasting models, and more. It serves as a playground for me to deepen my understanding and familiarize myself with these techniques.

This repos also serve as data preprocessors to format various datasets i am using for creating dashboards in [Tableau](https://public.tableau.com/app/profile/toni.lim/vizzes) and EDA in [Kaggles](https://www.kaggle.com/inotives). 

### List of scripts 
Here is a list of scripts I am experimenting with on various collected data:

- Preprocessing various technical indicators (MA, EMA, VWAP, RSI, Bollinger Bands, etc.) with OHLCV data
- Crypto price forecasting using ARIMA, SARIMAX, GARCH
- Trading bots that utilize the forecasted prices
- Sentiment analysis using VADER on collected news articles
- Visualization with Plotly and Matplotlib+Seaborn
- Analyzing historical lottery draws using simple data mining techniques like FP-Growth and Weighted Random Occurrence to determine lottery numbers with higher chances of winning


#### INSTALLATION NOTES:
- For MacOS, since XGBoost need OpenMP runtime to works, you will need to make sure to install libomp. you can install this with homebrew. 
  ```
  brew install libomp
  ```