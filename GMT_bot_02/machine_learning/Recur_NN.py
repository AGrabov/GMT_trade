import tensorflow as tf
from data_feed import BinanceFuturesData
from GMT30_strat02_btest import HeikinAshiStrategy
import backtrader as bt
import datetime as dt



# Settings
target_coin = 'GMT'
base_currency = 'USDT' # 'BUSD' # 
symbol = target_coin + base_currency
dataname = (f'{target_coin}/{base_currency}')
start_date = dt.datetime.strptime("2023-07-01 00:00:00", "%Y-%m-%d %H:%M:%S")
# end_date = dt.datetime.strptime("2023-07-25 00:00:00", "%Y-%m-%d %H:%M:%S")
today = f"{dt.date.today()} 00:00:00"
print(today)
end_date = dt.datetime.strptime(today, "%Y-%m-%d %H:%M:%S")
timeframe =  'Minutes' # 'Hours' #  
compression = 5
binance_timeframe = f'{compression}m'


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mse',
              metrics=['mae'])

# Fetch the data for the specified symbol and time range    
df = BinanceFuturesData.fetch_data(symbol=symbol, start_date=start_date, end_date=end_date, binance_timeframe=binance_timeframe)

data = BinanceFuturesData(dataname=df,
                        start_date=start_date, 
                        end_date=end_date, 
                        timeframe=timeframe,
                        compression=compression)


y_train = df[['open', 'high', 'low', 'close', 'volume']]
x_train = df.drop(['open', 'high', 'low', 'close', 'volume'], axis=1)
x_train = x_train
x_train = x_train.values





model.fit(x_train, y_train, epochs=10)

predictions = model.predict(data)

print(predictions)

