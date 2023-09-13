from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from data_feed_01 import BinanceFuturesData

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
target_coin = 'BTC'
base_currency = 'USDT' # 'BUSD' # 
binance_timeframe = '1d'
start_date = dt.datetime.strptime(
            "2020-01-01 00:00:00", 
            "%Y-%m-%d %H:%M:%S")
end_date = dt.datetime.strptime(
            "2023-08-26 00:00:00",
             "%Y-%m-%d %H:%M:%S")

symbol = target_coin + base_currency
dataname = (f'{target_coin}/{base_currency}')

try:
    df = BinanceFuturesData.fetch_data(symbol=symbol, startdate=start_date, enddate=end_date, binance_timeframe=binance_timeframe)
    print('Fteching data...')
    print(df.head(3))
    print(df.tail(2))
except Exception as e:
    print(f"Error fetching data: {e}")

# Select only 'timestamp' and 'close' columns
df.reset_index(inplace=True)
df = df[['datetime', 'close']]

# Rename columns
df.rename(columns={'datetime': 'ds', 'close': 'y'}, inplace=True)

# Remove timezone information
df['ds'] = df['ds'].dt.tz_localize(None)

print(df.head(3))
print(df.tail(2))
print('Number of timesteps in the data: ', len(df))

# Initialize the Prophet model
model = Prophet(daily_seasonality=True)

# Fit the model to your data
model.fit(df)
print('Fitted the model...')

# Create a future dataframe for the next 30 days
future = model.make_future_dataframe(periods=30)#, include_history=True)

# Predict the future values
print('Predicting...')
forecast = model.predict(future)

# Plot the forecast and save the figure
fig = model.plot(forecast)
plt.savefig(f'./Prophet/{symbol}_{binance_timeframe}_forecast_plot.png')

# Plot the forecast components (trend, yearly seasonality, and weekly seasonality) and save the figure
fig2 = model.plot_components(forecast)
plt.savefig(f'./Prophet/{symbol}_{binance_timeframe}_forecast_components_plot.png')
