import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
import datetime as dt
from data_feed_01 import BinanceFuturesData
from ta_indicators import TAIndicators
from data_normalizer import CryptoDataNormalizer
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
target_coin = 'GMT'
base_currency = 'USDT'
binance_timeframe = '30m'
start_date = dt.datetime.strptime("2023-05-01 00:00:00", "%Y-%m-%d %H:%M:%S")
end_date = dt.datetime.strptime("2023-08-01 00:00:00", "%Y-%m-%d %H:%M:%S")
use_fteched_data = False
seq_length = 48  # Consider experimenting with different sequence lengths
model_type = 'LSTM'  # Choose between 'Dense', 'LSTM' and 'GRU'

# Construct symbol and dataname
symbol = target_coin + base_currency
dataname = f'{target_coin}/{base_currency}'

# Initialize data normalizer
normalizer = CryptoDataNormalizer()

# Fetch or read data
if use_fteched_data:
    try:
        df = BinanceFuturesData.fetch_data(symbol=symbol, startdate=start_date, enddate=end_date, binance_timeframe=binance_timeframe)
        logger.info('Fetching data...')
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
else:
    csv_name = 'GMTUSDT - 30m_(since_2022-03-15).csv'
    csv_path = f'./data_csvs/{csv_name}'
    df = pd.read_csv('GMTUSDT_30m_data.csv', header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df = df.dropna()
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.set_index('timestamp')

# Ensure no missing values
assert not df.isnull().any().any(), "There are missing values in the dataset."

# Calculate technical indicators
df = TAIndicators(df)._calculate_indicators()
logger.info("Added TA Indicators")
print(df.head(3))
print(f"Data length: {len(df)}")

# Normalize the data
try:    
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)
except Exception as e:
    logger.error(f"Error normalizing data: {e}")

train_size = int(0.8 * len(df))
step_size = int(0.05 * len(df))

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, train_size=train_size, shuffle=False)

# Create sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:i+seq_length].values
        y = data.iloc[i+seq_length]['close']
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Define the model type with potential for stateful RNNs
def build_model(model_type='LSTM', stateful=False, batch_size=None):
    if model_type == 'Dense':
        model_structure = [
            tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ]
    elif model_type == 'LSTM':
        model_structure = [
            tf.keras.layers.LSTM(64, return_sequences=True, stateful=stateful, batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32, stateful=stateful),
            tf.keras.layers.Dense(1)
        ]
    elif model_type == 'GRU':
        model_structure = [
            tf.keras.layers.GRU(64, return_sequences=True, stateful=stateful, batch_input_shape=(batch_size, X_train.shape[1], X_train.shape[2])),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.GRU(32, stateful=stateful),
            tf.keras.layers.Dense(1)
        ]
    return tf.keras.models.Sequential(model_structure)

model = build_model(model_type=model_type, stateful=False)  # Set stateful=True for stateful RNNs

# Compile the model
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)  # You can adjust the learning rate if needed
model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f'./models/{model_type}_model_checkpoint_{symbol}_{binance_timeframe}.h5', save_best_only=True)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Evaluate the model
loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
logger.info(f'Test set Mean Abs Error: {mae}\nTest set Mean Squared Error: {mse}\nTest set Root Mean Squared Error: {np.sqrt(mse)}')

# Save the entire model
model.save(f'./models/{model_type}_model_{symbol}_{binance_timeframe}.h5')

# Load the model for future use
# loaded_model = tf.keras.models.load_model(f'./models/{model_type}_model_{symbol}_{binance_timeframe}.h5')

# Visualization of Model's Predictions
predictions = model.predict(X_test)
plt.figure(figsize=(15, 6))
plt.plot(y_test, label='True')
plt.plot(predictions, label='Predicted')
plt.title('Model Predictions vs True Values')
plt.legend()
plt.show()
