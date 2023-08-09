import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Load the data
ohlcv = pd.read_csv('GMTUSDT - 5m_(since_2021-01-01).csv')

df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
df.set_index('datetime', inplace=True)

# Ensure the data is in ascending order
df.sort_index(ascending=True, inplace=True)

# Calculate the 7 day moving average
df['7_day_avg'] = df['close'].rolling(window=7).mean()

# Calculate the 14 day moving average
df['14_day_avg'] = df['close'].rolling(window=14).mean()

# Calculate the 30 day moving average
df['30_day_avg'] = df['close'].rolling(window=30).mean()

# Drop the rows with NaN values (caused by the moving average calculation)
df.dropna(inplace=True)

# Normalize the data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

# Convert the scaled data back to a DataFrame
scaled_df = pd.DataFrame(scaled_df, columns=df.columns, index=df.index)

# Split the data into training and testing sets
# Let's say you want to use the last 1000 rows for testing
train_data = scaled_df.iloc[:-50000]
test_data = scaled_df.iloc[-50000:]

def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)].values
        y = data.iloc[i+seq_length]['close']
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_length = 60  # Choose sequence length
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)


# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(64, activation='relu', return_sequences=True,input_shape=(X_train.shape[1], X_train.shape[2])), 
  tf.keras.layers.LSTM(32, activation='relu', return_sequences=False),
  tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=['mae', 'mse'])

# Define callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('model_checkpoint.h5', save_best_only=True)

# Train the model
model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Use the trained model to make predictions in your trading strategy
predictions = model.predict(X_test)

# Evaluate the model
loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
print(f'Test set Mean Abs Error: {mae}\nTest set Mean Squared Error: {mse}')

# Save the entire model for future use
model.save('final_model.h5')

# Load the model for future use
# loaded_model = tf.keras.models.load_model('final_model.h5')


