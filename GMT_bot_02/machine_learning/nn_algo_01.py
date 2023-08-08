import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the data
ohlcv = pd.read_csv('GMTUSDT - 5m_(since_2021-01-01).csv')

df = pd.DataFrame(ohlcv, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
df['datetime'] = pd.to_datetime(df['datetime'], unit='ms', utc=True)
df.set_index('datetime', inplace=True)

# Ensure the data is in ascending order
df.sort_values('datetime', inplace=True)

# Calculate the moving averages
window_sizes = [7, 14, 30]
for window_size in window_sizes:
    df[f'{window_size}_day_avg'] = df['close'].rolling(window=window_size).mean()

# Drop the rows with NaN values (caused by the moving average calculation)
df.dropna(inplace=True)

# Normalize the data
scaler = StandardScaler()
df[df.columns] = scaler.fit_transform(df)

# Split the data into training and testing sets
train_data, test_data = train_test_split(df, test_size=50000, shuffle=False)

# Create sequences
def create_sequences(data, seq_length):
    xs = data.rolling(window=seq_length).apply(lambda x: x.tolist()).dropna().tolist()
    ys = data['close'].iloc[seq_length:].values
    return np.array(xs), np.array(ys)

seq_length = 60  # Choose sequence length
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Define the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=[X_train.shape[1]]),
  tf.keras.layers.Dense(32, activation='relu'),
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