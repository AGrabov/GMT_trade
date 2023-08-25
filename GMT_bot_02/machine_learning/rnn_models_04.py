import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from keras_tuner import RandomSearch
from tensorflow import keras
import os
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
import datetime as dt
from data_feed_01 import BinanceFuturesData
from ta_indicators import TAIndicators
from data_normalizer import CryptoDataNormalizer
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RNNModels:
    def __init__(self, settings):
        self.settings = settings
        self.model = None

    def fetch_or_read_data(self):
        if self.settings['use_fetched_data']:
            try:
                df = BinanceFuturesData.fetch_data(
                    symbol=self.settings['target_coin'] + self.settings['base_currency'],
                    startdate=self.settings['start_date'],
                    enddate=self.settings['end_date'],
                    binance_timeframe=self.settings['binance_timeframe']
                )
                logger.info('Fetching data...')
            except Exception as e:
                logger.error(f"Error fetching data: {e}")
        else:
            csv_name = 'GMTUSDT - 30m_(since_2022-03-15).csv'
            csv_path = f'./data_csvs/{csv_name}'
            df = pd.read_csv(csv_path, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.dropna()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        return df

    def preprocess_data(self, df):
        df.dropna(inplace=True)
        df = TAIndicators(df)._calculate_indicators()
        logger.info("Added TA Indicators")
        try:
            scalers = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler(),
                'quantile': QuantileTransformer(),
                'power': PowerTransformer()
            }
            scaler = scalers.get(self.settings['scaler_type'])
            df[df.columns] = scaler.fit_transform(df)
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
        return df

    def split_data(self, df):
        train_size = int(0.8 * len(df))
        train_data, test_data = train_test_split(df, train_size=train_size, shuffle=False)
        return train_data, test_data

    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:i+seq_length].values
            y = data.iloc[i+seq_length]['close']
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def build_model(self, model_type='LSTM', stateful=False, batch_size=None, input_shape=None):    
        if model_type == 'Dense':
            model_structure = [
                tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_shape=input_shape),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1)
            ]

        elif model_type == 'LSTM':
            model_structure = [
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, stateful=stateful, kernel_regularizer=tf.keras.regularizers.l2(0.01), recurrent_regularizer=tf.keras.regularizers.l2(0.01))),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, stateful=stateful, kernel_regularizer=tf.keras.regularizers.l2(0.01), recurrent_regularizer=tf.keras.regularizers.l2(0.01))),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1)
            ]    
        elif model_type == 'GRU':
            model_structure = [
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, kernel_regularizer=tf.keras.regularizers.l2(0.01))),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(1)
            ]

        return tf.keras.models.Sequential(model_structure)

    def compile_and_train_model(self, model, X_train, y_train):
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f'./models/RNN/{self.settings["model_type"]}_model_checkpoint_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.h5', save_best_only=True)
        callbacks = [early_stopping, model_checkpoint, lr_scheduler]
        history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=callbacks)
        return model, history

    def evaluate_model(self, model, X_test, y_test):
        loss, mae, mse = model.evaluate(X_test, y_test, verbose=2)
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        logger.info(f'R2 Score: {r2}')
        logger.info(f'Test set Mean Abs Error: {mae}\nTest set Mean Squared Error: {mse}\nTest set Root Mean Squared Error: {np.sqrt(mse)}')
        return predictions

    def visualize_predictions(self, y_test, predictions):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test, label='True')
        plt.plot(predictions, label='Predicted')
        plt.title('Model Predictions vs True Values')
        plt.legend()
        plt.show()

    def build_and_train_arima(self, y_train):
        # Fit ARIMA model
        model_arima = ARIMA(y_train, order=(5,1,0))
        model_arima_fit = model_arima.fit(disp=0)
        return model_arima_fit
    
    def build_transformer_model(self, X_train):
        def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
            # Normalization and Attention
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
            x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
            x = tf.keras.layers.Dropout(dropout)(x)
            res = x + inputs

            # Feed Forward Part
            x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
            x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
            x = tf.keras.layers.Dropout(dropout)(x)
            x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
            return x + res

        inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
        x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(30, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        outputs = tf.keras.layers.Dense(1)(x)

        model_transformer = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model_transformer


    def hyperparameter_tuning(self, X_train, y_train, X_test, y_test):
        if self.settings['model_type'] == 'ARIMA':
            # ARIMA hyperparameter tuning
            best_score, best_cfg = float("inf"), None
            p_values = [0, 1, 2, 4, 6, 8, 10]
            d_values = range(0, 3)
            q_values = range(0, 3)
            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        order = (p, d, q)
                        try:
                            mse = self.evaluate_arima_model(y_train, order)
                            if mse < best_score:
                                best_score, best_cfg = mse, order
                            logger.info(f'ARIMA{order} MSE={mse}')
                        except:
                            continue
            logger.info(f'Best ARIMA{best_cfg} MSE={best_score}')

        elif self.settings['model_type'] == 'Transformer':
            def build_transformer_model(hp):
                def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
                    # Normalization and Attention
                    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
                    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
                    x = tf.keras.layers.Dropout(dropout)(x)
                    res = x + inputs

                    # Feed Forward Part
                    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
                    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
                    x = tf.keras.layers.Dropout(dropout)(x)
                    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
                    return x + res

                inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))
                x = transformer_encoder(
                    inputs, 
                    head_size=hp.Int('head_size', min_value=32, max_value=256, step=32),
                    num_heads=hp.Int('num_heads', min_value=1, max_value=8, step=1),
                    ff_dim=hp.Int('ff_dim', min_value=64, max_value=512, step=32),
                    dropout=hp.Float('dropout', 0.1, 0.5, step=0.1)
                )
                x = tf.keras.layers.GlobalAveragePooling1D()(x)
                x = tf.keras.layers.Dense(30, activation='relu')(x)
                x = tf.keras.layers.Dropout(0.1)(x)
                outputs = tf.keras.layers.Dense(1)(x)

                model_transformer = tf.keras.Model(inputs=inputs, outputs=outputs)

                return model_transformer

            tuner = RandomSearch(
                build_transformer_model,
                objective='val_loss',
                max_trials=5,
                executions_per_trial=3,
            )
            tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

        else:
            def build_model(hp):
                model = tf.keras.models.Sequential()
                
                # Choose model type based on settings
                if self.settings['model_type'] == 'LSTM':
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                        units=hp.Int('units', min_value=32, max_value=512, step=32),
                        return_sequences=True,
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                        recurrent_regularizer=tf.keras.regularizers.l2(0.01)
                    )))
                    model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', 0.1, 0.5, step=0.1)))
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
                        units=hp.Int('units', min_value=32, max_value=512, step=32),
                        kernel_regularizer=tf.keras.regularizers.l2(0.01),
                        recurrent_regularizer=tf.keras.regularizers.l2(0.01)
                    )))
                elif self.settings['model_type'] == 'GRU':
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                        units=hp.Int('units', min_value=32, max_value=512, step=32),
                        return_sequences=True,
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)
                    )))
                    model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', 0.1, 0.5, step=0.1)))
                    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(
                        units=hp.Int('units', min_value=32, max_value=512, step=32),
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)
                    )))
                else:  # Dense model
                    model.add(tf.keras.layers.Dense(
                        units=hp.Int('units', min_value=32, max_value=512, step=32),
                        activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.01)
                    ))
                    model.add(tf.keras.layers.BatchNormalization())
                    model.add(tf.keras.layers.Dropout(rate=hp.Float('dropout', 0.1, 0.5, step=0.1)))
                
                model.add(tf.keras.layers.Dense(1))
                
                # Compile the model
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]))
                model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])
                
                return model

            tuner = RandomSearch(
                build_model,
                objective='val_loss',
                max_trials=5,
                executions_per_trial=3,
                
            )

            tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))


    def ensemble_methods(self, X_test, y_test):
        models_directory = f'./models/RNN/'
        if os.path.isdir(models_directory) and any(os.scandir(models_directory)):
            models = []
            for model_file in os.listdir(models_directory):
                if model_file.endswith('.h5'):
                    model_path = os.path.join(models_directory, model_file)
                    model = tf.keras.models.load_model(model_path)
                    models.append(model)
                    logger.info(f"Loaded model from {model_path}")
                elif model_file.endswith('.pkl'):
                    model_path = os.path.join(models_directory, model_file)
                    with open(model_path, 'rb') as pkl:
                        arima_model = pickle.load(pkl)
                        models.append(arima_model)
                    logger.info(f"Loaded ARIMA model from {model_path}")

            # Average the predictions from all models
            predictions = np.mean([model.predict(X_test) for model in models if hasattr(model, 'predict')], axis=0)

            # Evaluate ensemble predictions
            r2 = r2_score(y_test, predictions)
            mse = np.mean(np.square(y_test - predictions))
            mae = np.mean(np.abs(y_test - predictions))
            rmse = np.sqrt(mse)
            logger.info(f'Ensemble R2 Score: {r2}')
            logger.info(f'Ensemble Test set Mean Abs Error: {mae}\nEnsemble Test set Mean Squared Error: {mse}\nEnsemble Test set Root Mean Squared Error: {rmse}')
            return predictions
        else:
            logger.warning("No models found in the directory for ensemble.")
            return None
        
    def run(self):
        df = self.fetch_or_read_data()
        df = self.preprocess_data(df)
        train_data, test_data = self.split_data(df)
        X_train, y_train = self.create_sequences(train_data)
        X_test, y_test = self.create_sequences(test_data)
        
        if self.settings['model_type'] == 'ARIMA':
            arima_model = self.build_and_train_arima(y_train)
            arima_predictions = arima_model.forecast(steps=len(y_test))
            r2 = r2_score(y_test, arima_predictions)
            logger.info(f'ARIMA R2 Score: {r2}')
            
            # Save ARIMA model
            with open(f'./models/RNN/ARIMA_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.pkl', 'wb') as pkl:
                pickle.dump(arima_model, pkl)
                
        elif self.settings['model_type'] == 'Transformer':
            transformer_model = self.build_transformer_model(X_train)
            self.compile_and_train_model(transformer_model, X_train, y_train)
            transformer_predictions = self.evaluate_model(transformer_model, X_test, y_test)
            self.visualize_predictions(y_test, transformer_predictions)
            
            # Save Transformer model
            transformer_model.save(f'./models/RNN/Transformer_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.h5')
            
        else:
            self.model = self.build_model(model_type=self.settings['model_type'])
            self.compile_and_train_model(self.model, X_train, y_train)
            predictions = self.evaluate_model(self.model, X_test, y_test)
            self.visualize_predictions(y_test, predictions)
            
            # Save the model
            self.model.save(f'./models/RNN/{self.settings["model_type"]}_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.h5')
            
        # Hyperparameter Tuning
        if self.settings['use_hp_tuning']:
            self.hyperparameter_tuning(X_train, y_train, X_test, y_test)
        
        # Ensemble Methods
        models_directory = f'./models/RNN/'
        if os.path.isdir(models_directory) and any(os.scandir(models_directory)):
            ensemble_predictions = self.ensemble_methods(X_test, y_test)
            if ensemble_predictions is not None:
                self.visualize_predictions(y_test, ensemble_predictions)


if __name__ == '__main__':
    SETTINGS = {
        'target_coin': 'GMT',
        'base_currency': 'USDT',
        'binance_timeframe': '30m',
        'start_date': dt.datetime.strptime("2023-05-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'end_date': dt.datetime.strptime("2023-08-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'use_fetched_data': False,
        'seq_length': 48,
        'model_type': 'LSTM', # Dense, LSTM, GRU, Transformer, ARIMA
        'scaler_type': 'minmax',
        'use_hp_tuning': False,
        'use_ensemble': False,
    }
    rnn_models = RNNModels(SETTINGS)
    rnn_models.run()