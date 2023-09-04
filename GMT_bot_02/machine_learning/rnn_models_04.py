import tensorflow as tf
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split, TimeSeriesSplit
from keras_tuner import RandomSearch
from tensorflow import keras
import traceback  
import os  
import argparse
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import joblib 
import json
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
            # csv_path = '/root/gmt-bot/data_csvs/GMTUSDT - 30m_(since_2022-03-15).csv'
            df = pd.read_csv(csv_path, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = df.dropna()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        return df

    def preprocess_data(self, df):
        df.dropna(inplace=True)
        df = TAIndicators(df)._calculate_indicators()
        logger.info("Added TA Indicators")
        print(df.head(5))
        print()
        logger.info(f'Number of timesteps in the data: {len(df)}')
        
        return df

    def split_data(self, df):
        train_size = int(0.8 * len(df))
        train_data, test_data = train_test_split(df, train_size=train_size, shuffle=False)
        if self.settings['model_type'] == 'ARIMA':        
            # train_data = np.log(train_data + 1e-9)
            # test_data = np.log(test_data + 1e-9)
            return train_data, test_data
        
        try:
            scalers = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler(),
                'quantile': QuantileTransformer(),
                'power': PowerTransformer()
            }
            scaler = scalers.get(self.settings['scaler_type'])
            logger.info(f"Scaler: {self.settings['scaler_type']}")
            train_data[train_data.columns] = scaler.fit_transform(train_data)
            print(train_data.head(5))
            print()
            logger.info(f"Train data length: {len(train_data)}")
            test_data[test_data.columns] = scaler.fit_transform(test_data)
            print(test_data.head(5))
            print()
            logger.info(f"Test data length: {len(test_data)}")
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")

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
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(256, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.001))),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128, kernel_regularizer=tf.keras.regularizers.l2(0.001))),
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
        predictions = model.predict(X_test).flatten()  # Ensure it's a flat array
        y_test = y_test.flatten()  # Ensure it's a flat array
        
        # Check for shape mismatch
        if len(predictions) != len(y_test):
            logger.error("Shape mismatch between predictions and y_test.")
            return
        
        # Additional metrics
        epsilon = 1e-10  # small constant to avoid division by zero
        mape = np.mean(np.abs((y_test - predictions) / (y_test + epsilon))) * 100
        r2 = r2_score(y_test, predictions)
        evs = explained_variance_score(y_test, predictions)
        
        logger.info(f'R2 Score: {r2}')
        logger.info(f'Mean Absolute Percentage Error (MAPE): {mape}%')
        logger.info(f'Explained Variance Score: {evs}')
        logger.info(f'Test set Mean Abs Error: {mae}\nTest set Mean Squared Error: {mse}\nTest set Root Mean Squared Error: {np.sqrt(mse)}')
        
        # Visualizations
        self.visualize_predictions(y_test, predictions)
        self.plot_residuals(y_test, predictions)
        
        return predictions

    def visualize_predictions(self, y_test, predictions):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test, label='True')
        plt.plot(predictions, label='Predicted')
        plt.title('Model Predictions vs True Values')
        plt.legend()
        plt.savefig(f'./models/RNN/{self.settings["model_type"]}/Predictions.png')
        plt.show()

    def plot_residuals(self, y_test, predictions):
        residuals = y_test - predictions
        
        # Residual Plot
        plt.figure(figsize=(15, 6))
        plt.scatter(predictions, residuals, alpha=0.5)
        plt.title('Residuals vs Predicted Values')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.axhline(y=0, color='r', linestyle='--')        
        plt.savefig(f'./models/RNN/{self.settings["model_type"]}/Residuals_vs_Predictions.png')
        plt.show()
        
        
        # Histogram of Residuals
        plt.figure(figsize=(15, 6))
        plt.hist(residuals, bins=30, edgecolor='k')
        plt.title('Histogram of Residuals')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.savefig(f'./models/RNN/{self.settings["model_type"]}/Residuals_hist.png')
        plt.show()

    def build_and_train_arima(self, y_train):
        # Fit ARIMA model
        model_arima = ARIMA(y_train, order=(5,1,0))
        model_arima_fit = model_arima.fit()
        return model_arima_fit
    
    def evaluate_arima_model(self, model, y_test):
        predictions = model.forecast(steps=len(y_test))
        print("Predictions:", predictions)

        print("Length of y_test:", len(y_test))
        print("Length of predictions:", len(predictions))

        print("Any NaN in y_test:", np.isnan(y_test).any())
        print("Any Inf in y_test:", np.isinf(y_test).any())
        print("Any NaN in predictions:", np.isnan(predictions).any())
        print("Any Inf in predictions:", np.isinf(predictions).any())

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return mse, mae, r2   

    def check_stationarity(time_series):
        # Perform Augmented Dickey-Fuller test:
        print('Results of Augmented Dickey-Fuller Test:')
        dftest = adfuller(time_series, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)
        if dfoutput['p-value'] <= 0.05:
            print('The series is stationary.')
        else:
            print('The series is not stationary.') 

    def time_series_cross_validation(model, X, y, n_splits=5):
        tscv = TimeSeriesSplit(n_splits=n_splits)
        mse_scores = []
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            model.fit(X_train, y_train)
            predictions = model.predict(start=len(X_train), end=len(X_train) + len(X_test) - 1, dynamic=False)
            mse = mean_squared_error(y_test, predictions)
            mse_scores.append(mse)
        return np.mean(mse_scores)

    def tune_arima(self, y_train, y_test):
        # Initialize return variables to None
        model_arima, best_score = None, float('inf')

        print("Any NaN in y_train:", np.isnan(y_train).any())
        print("Any Inf in y_train:", np.isinf(y_train).any())
        # Debugging NaNs
        print("Min in y_train:", np.min(y_train))
        print("Max in y_train:", np.max(y_train))

        # # Applying log and checking for NaNs
        # series = np.log(y_train + 1e-9)
        series = y_train
        self.check_stationarity(series)
        if np.isnan(series).any():
            logger.error("NaN values detected in series after log transformation")
            
            return None, None  # or handle this appropriately        

        print("Series passed to adfuller:", series)

        try:
            result = adfuller(series)
            print('ADF Statistic: {}'.format(result[0]))
            print('p-value: {}'.format(result[1]))
            plot_acf(series)
            plot_pacf(series)
        except Exception as e:
            logger.error(f"Error in adfuller: {e}")
            logger.error(traceback.format_exc())  

        # series = y_train      

        p_values = range(0, 6) #[0, 1, 2, 4, 6, 8, 10]
        d_values = range(0, 6)
        q_values = range(0, 3)
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)
                    try:
                        model_arima = ARIMA(series, order=order)
                        model_arima_fit = model_arima.fit()
                        # print(model_arima_fit.summary())
                        mse, mae, r2 = self.evaluate_arima_model(model_arima_fit, y_test)  # Assuming y_test is available
                        if best_score is None or mse < best_score:
                            best_score, best_cfg = mse, order                        
                        print("Best Score:", best_score)
                        print("MSE:", mse)
                        
                        logger.info(f'ARIMA{order} MSE={mse}, MAE={mae}, R2={r2}')                        
                    except Exception as e:
                        logger.error(f"Error while fitting ARIMA model: {e}")
                        logger.error(traceback.format_exc())

        logger.info(f'Best ARIMA{best_cfg} MSE={best_score}')

        try:
            if best_cfg is not None:
                model_arima = ARIMA(series, order=best_cfg)
                model_arima_fit = model_arima.fit()
                mse, mae, r2 = self.evaluate_arima_model(model_arima_fit, y_test)
                logger.info(f'ARIMA{best_cfg} MSE={mse}')
                logger.info(f'ARIMA{best_cfg} MAE={mae}')
                logger.info(f'ARIMA{best_cfg} R2={r2}')

                # Save ARIMA model
                model_path = os.path.join('./models/RNN/', 
                                          f'{self.settings["model_type"]}/BEST_ARIMA_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.pkl')                
                joblib.dump(model_arima_fit, model_path)
                logger.info(f"Saved ARIMA model to {model_path}")

        except Exception as e:
            logger.error(f"Error saving ARIMA model: {e}")
            logger.error(traceback.format_exc())

        return model_arima, best_score
    
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
     
    def tune_transformer(self, X_train, y_train, X_test, y_test):
        epochs = self.settings.get('epochs', 5)  # Replace hardcoded epochs
        early_stopping = tf.keras.callbacks.EarlyStopping(patience=3)
        
        def build_transformer_model(hp):
            def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
                # Normalization and Attention
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
                x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
                x = tf.keras.layers.Dropout(dropout)(x)
                res = x + inputs

                # Feed Forward Part
                x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
                kernel_regularizer = tf.keras.regularizers.l2(hp.Float('kernel_regularizer', 0.001, 0.1, step=0.01))
            
                x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu', kernel_regularizer=kernel_regularizer)(x)
                x = tf.keras.layers.Dropout(dropout)(x)
                x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1, kernel_regularizer=kernel_regularizer)(x)
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

            learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 5e-3, 5e-4])
        
            model_transformer = tf.keras.Model(inputs=inputs, outputs=outputs)
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            model_transformer.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

            return model_transformer

        tuner = RandomSearch(
                build_transformer_model,
                objective='val_loss',
                max_trials=5,
                executions_per_trial=3,
                directory='./models/RNN',
                project_name='Transformer'
            )      
        
        tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test), callbacks=[early_stopping])

        best_model = tuner.get_best_models(num_models=1)[0]
        best_metrics = self.evaluate_model(best_model, X_test, y_test)
        
        return best_model, best_metrics

    def hyperparameter_tuning(self, X_train, y_train, X_test, y_test):
        best_model = None
        best_metrics = None
        
        if self.settings['model_type'] == 'ARIMA':
            best_model, best_metrics = self.tune_arima(y_train, y_test)
        elif self.settings['model_type'] == 'Transformer':
            best_model, best_metrics = self.tune_transformer(X_train, y_train, X_test, y_test)
    
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
                directory=f'./models/RNN/',
                project_name=f'{self.settings["model_type"]}'                
            )
            tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

            best_model = tuner.get_best_models(num_models=1)[0]
            best_metrics = self.evaluate_model(best_model, X_test, y_test)
        
        return best_model, best_metrics

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

            # Visualize ensemble predictions
            self.visualize_predictions(y_test, predictions)
            self.plot_residuals(y_test, predictions)

            return predictions
        else:
            logger.warning("No models found in the directory for ensemble.")
            return None
        
    def run(self):
        df = self.fetch_or_read_data()
        df = self.preprocess_data(df)
        train_data, test_data = self.split_data(df)
        X_train, y_train = self.create_sequences(train_data, self.settings['seq_length'])
        X_test, y_test = self.create_sequences(test_data, self.settings['seq_length'])

        # Create directory
        models_directory = f'./models/RNN/{self.settings["model_type"]}/'        
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)

        # Hyperparameter Tuning
        best_model, best_metrics = None, None
    
        if self.settings['use_hp_tuning']:
            best_model, best_metrics = self.hyperparameter_tuning(X_train, y_train, X_test, y_test)
        
            # Save the best model
            if best_model is not None and self.settings['model_type'] != 'ARIMA':
                best_model_path = f'./models/RNN/{self.settings["model_type"]}/best_{self.settings["model_type"]}_{self.settings["scaler_type"]}_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.h5'
                best_model.save(best_model_path)
                # Save the best parameters to a JSON file
                with open(f"./models/RNN/best_settings_for_{self.settings['model_type']}.json", "w") as f:
                    json.dump(best_metrics.best_params_, f)
                logger.info(f"Best model saved to {best_model_path}")
                # logger.info(f"Best hyperparameters saved")
        else: 
            if self.settings['model_type'] == 'ARIMA':
                arima_model = self.build_and_train_arima(y_train)
                arima_predictions = arima_model.forecast(steps=len(y_test))
                r2 = r2_score(y_test, arima_predictions)
                logger.info(f'ARIMA R2 Score: {r2}')
                
                # # Save ARIMA model
                # with open(f'./models/RNN/{self.settings["model_type"]}/ARIMA_{self.settings["scaler_type"]}_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.pkl', 'wb') as pkl:
                #     pickle.dump(arima_model, pkl)
                # Save ARIMA model
                model_path = os.path.join(f'./models/RNN/{self.settings["model_type"]}/ARIMA_{self.settings["scaler_type"]}_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.pkl')
                joblib.dump(arima_model, model_path)
                    
            elif self.settings['model_type'] == 'Transformer':
                transformer_model = self.build_transformer_model(X_train)
                self.compile_and_train_model(transformer_model, X_train, y_train)
                transformer_predictions = self.evaluate_model(transformer_model, X_test, y_test)
                self.visualize_predictions(y_test, transformer_predictions)
                
                # Save Transformer model
                transformer_model.save(f'./models/RNN/{self.settings["model_type"]}/Transformer_{self.settings["scaler_type"]}_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.h5')
                
            else:
                self.model = self.build_model(model_type=self.settings['model_type'])
                self.compile_and_train_model(self.model, X_train, y_train)
                predictions = self.evaluate_model(self.model, X_test, y_test)
                self.visualize_predictions(y_test, predictions)
                
                # Save the model
                self.model.save(f'./models/RNN/{self.settings["model_type"]}/{self.settings["model_type"]}_{self.settings["scaler_type"]}_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.h5')
                
           
        if self.settings['use_ensemble']:
            # Ensemble Methods
            models_directory = f'./models/RNN/'
            if os.path.isdir(models_directory) and any(os.scandir(models_directory)):
                ensemble_predictions = self.ensemble_methods(X_test, y_test)
                if ensemble_predictions is not None:
                    self.visualize_predictions(y_test, ensemble_predictions)

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='RNN models train.')
    parser.add_argument('--model_type', choices=['Dense', 'LSTM', 'GRU', 'Transformer', 'ARIMA'], type=str, default='GRU', help='Select model type')
    parser.add_argument('--use_hp_tuning', type=bool, default=True, help='Use hyperparameter tuning')
    parser.add_argument('--use_ensemble', type=bool, default=False, help='Use ensemble methods. Combines all models and evaluates on test set.')
    parser.add_argument('--scaler_type', choices=['standard', 'minmax', 'robust', 'quantile', 'power'], type=str, default='minmax', help='Select data scaler type')
    parser.add_argument('--use_fetched_data', type=bool, default=False, help='Use fetched data')
    parser.add_argument('--seq_length', type=int, default=48, help='Data segments length')

    args = parser.parse_args()

    SETTINGS = {
        'target_coin': 'GMT',
        'base_currency': 'USDT',
        'binance_timeframe': '30m',
        'start_date': dt.datetime.strptime("2023-05-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'end_date': dt.datetime.strptime("2023-08-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'use_fetched_data': args.use_fetched_data,
        'seq_length': args.seq_length,
        'model_type': args.model_type, # Dense, LSTM, GRU, Transformer, ARIMA
        'scaler_type': args.scaler_type,
        'use_hp_tuning': args.use_hp_tuning,
        'use_ensemble': args.use_ensemble,
    }
    rnn_models = RNNModels(SETTINGS)
    rnn_models.run()