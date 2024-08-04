import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Fetch historical data for Apple (AAPL)
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-12-31')

# Display the first few rows of the dataset
print(data.head())

# Plot the historical closing prices
plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.title(f'{ticker} Historical Closing Prices')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()
# Calculate and plot moving averages
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

plt.figure(figsize=(14, 7))
plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA20'], label='20-Day Moving Average')
plt.plot(data['MA50'], label='50-Day Moving Average')
plt.title(f'{ticker} Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Decompose the time series
decomposition = seasonal_decompose(data['Close'], model='multiplicative', period=365)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(14, 10))
plt.subplot(411)
plt.plot(data['Close'], label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(14, 7))
plt.subplot(121)
plot_acf(data['Close'].dropna(), ax=plt.gca(), lags=50)
plt.subplot(122)
plot_pacf(data['Close'].dropna(), ax=plt.gca(), lags=50)
plt.show()
