import argparse
import datetime as dt
import logging
import os
import json
from cv2 import normalize

import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from data_feed_01 import BinanceFuturesData
from data_normalizer import CryptoDataNormalizer
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, mean_squared_log_error,
                             r2_score)
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)
from sklearn.svm import (SVC, SVR, LinearSVC, LinearSVR, NuSVC, NuSVR,
                         OneClassSVM)
from ta_indicators import TAIndicators

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SVMModels:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.normalizer = CryptoDataNormalizer()

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
            df = pd.read_csv(csv_path, header=None, 
                             names=['timestamp', 'open', 'high', 'low', 'close', 'volume'], 
                             ) #skip_blank_lines=True
            df = df.dropna()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        return df

    def preprocess_data(self, df):
        df.dropna(inplace=True)
        df = TAIndicators(df)._calculate_indicators()
        logger.info("Added TA Indicators")
        
        # try:
        #     scalers = {
        #         'standard': StandardScaler(),
        #         'minmax': MinMaxScaler(),
        #         'robust': RobustScaler(),
        #         'quantile': QuantileTransformer(),
        #         'power': PowerTransformer()
        #     }
        #     scaler = scalers.get(self.settings['scaler_type'])
        #     df[df.columns] = scaler.fit_transform(df)
        # except Exception as e:
        #     logger.error(f"Error normalizing data: {e}")
        
        return df

    def split_data(self, df):
        # Define features and target variable
        # X = df.drop('close', axis=1)
        # y = df['close']

        # Hyperparameter tuning and model training
        train_size = int(0.8 * len(df))
        train_data, test_data = train_test_split(df, train_size=train_size, shuffle=False)

        # # Split data and train model        
        # X_train = X[:train_size]
        # y_train = y[:train_size]
        # X_test = X[train_size:len(df)]
        # y_test = y[train_size:len(df)]

        try:
            scalers = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler(),
                'quantile': QuantileTransformer(),
                'power': PowerTransformer()
            }
            scaler = scalers.get(self.settings['scaler_type'])
            train_data[train_data.columns] = scaler.fit_transform(train_data)
            test_data[test_data.columns] = scaler.fit_transform(test_data)
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
        
        return train_data, test_data

    def extract_features_and_target(self, train_data, test_data, target_column='close'):
        # Extracting features and target for training data
        X_train = train_data.drop(columns=[target_column])
        y_train = train_data[target_column]

        # Extracting features and target for testing data
        X_test = test_data.drop(columns=[target_column])
        y_test = test_data[target_column]

        return X_train, y_train, X_test, y_test


    def train_svm(self, X_train, y_train, kernel='rbf'):
        if self.settings['model_type'] == 'SVR':
            model = SVR(kernel=kernel)
            
        elif self.settings['model_type'] == 'LinearSVR':
            model = LinearSVR()
            
        elif self.settings['model_type'] == 'NuSVR':
            model = NuSVR()

        elif self.settings['model_type'] == 'SVC':
            model = SVC()

        elif self.settings['model_type'] == 'LinearSVC':
            model = LinearSVC()

        elif self.settings['model_type'] == 'NuSVC':
            model = NuSVC()

        elif self.settings['model_type'] == 'OneClassSVM':
            model = OneClassSVM()
        
        model.fit(X_train, y_train)
        return model           
    
    def evaluate_svm(self, model, X_test, y_test):        
        predictions = model.predict(X_test)
        
        # Additional metrics
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        msle = mean_squared_log_error(y_test, predictions) 
        r2 = r2_score(y_test, predictions)
        evs = explained_variance_score(y_test, predictions)
        
        logger.info(f'R2 Score: {r2}')
        logger.info(f'Mean Absolute Percentage Error (MAPE): {mape}%')
        logger.info(f'Explained Variance Score: {evs}')
        logger.info(f'Test set Mean Abs Error: {mae}\nTest set Mean Squared Error: {mse}\nTest set Root Mean Squared Error: {np.sqrt(mse)}')
        logger.info(f'Mean Squared Log Error: {msle}')
        
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
        plt.show()
        
        # Histogram of Residuals
        plt.figure(figsize=(15, 6))
        plt.hist(residuals, bins=30, edgecolor='k')
        plt.title('Histogram of Residuals')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.show()    

    def cross_validate_svm(self, X, y, kernel='rbf'):
        """Perform cross-validation on the SVM model using TimeSeriesSplit."""
        
        # Initialize the model based on the model_type setting
        if self.settings['model_type'] == 'SVR':
            model = SVR(kernel=kernel)
        elif self.settings['model_type'] == 'LinearSVR':
            model = LinearSVR()
        elif self.settings['model_type'] == 'NuSVR':
            model = NuSVR()
        elif self.settings['model_type'] == 'SVC':
            model = SVC()
        elif self.settings['model_type'] == 'LinearSVC':
            model = LinearSVC()
        elif self.settings['model_type'] == 'NuSVC':
            model = NuSVC()
        elif self.settings['model_type'] == 'OneClassSVM':
            model = OneClassSVM()
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        
        # Print the scores
        logger.info(f"TimeSeriesSplit cross-validation scores (negative MSE): {scores}")
        logger.info(f"Mean score: {scores.mean()}")
        logger.info(f"Standard deviation: {scores.std()}")
        
        return scores

    def hp_tuning(self, X_train, y_train, X_test, y_test):
        # Define the hyperparameters and their possible values
        # parameters = {
        #     'C': [0.1, 1, 10],
        #     'kernel': ['linear', 'poly', 'rbf'], #, 'sigmoid'],
        #     'degree': [2, 3, 4],  # Only used when kernel is 'poly'
        #     'gamma': ['scale', 'auto']
        # }
        
        # Initialize the model based on the model_type setting
        if self.settings['model_type'] in ['SVR', 'LinearSVR', 'NuSVR']:
            parameters = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'degree': (1, 10),
                'gamma': ['scale', 'auto'],
                'coef0': (0.0, 1.0),
                'tol': (1e-4, 1e-2),
                'C': (0.1, 10),
                'epsilon': (0.01, 1),
                'shrinking': [False, True],                
            }
            model = SVR()
        elif self.settings['model_type'] in ['SVC', 'LinearSVC', 'NuSVC']:
            parameters = {
                'C': (0.1, 10),
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                'degree': (1, 10),
                'gamma': ['scale', 'auto'],
                'coef0': (0.0, 1.0),
                'shrinking': [False, True],
                'probability': [False, True],
                'tol': (1e-4, 1e-2),                
                'decision_function_shape': ['ovo', 'ovr'],
                'break_ties': [False, True],
                'random_state': [None, 1, 42]
            }
            model = SVC()
        else:
            parameters = {                    
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'degree': (1, 5),
                    'gamma': ['scale', 'auto'],
                    'coef0': (0.0, 1.0),
                    'tol': (0.001, 0.01),
                    'nu': (0.0, 1.0),
                    'shrinking': [False, True],
                    

            }
            model = OneClassSVM()

        # Grid search with cross-validation
        # grid_search = GridSearchCV(model, parameters, cv=3, scoring='r2', verbose=2) 
        # grid_search.fit(X_train, y_train)
        random_search = RandomizedSearchCV(model, parameters, cv=3, scoring='r2', verbose=2) 
        random_search.fit(X_train, y_train)
        
        # Set the best model
        self.model = random_search.best_estimator_
        
        # Evaluate the best model
        predictions = self.evaluate_svm(self.model, X_test, y_test)
        
        logger.info(f"Best Parameters: {random_search.best_params_}")
        logger.info(f"Best Score: {-random_search.best_score_}")

        # Save the best parameters to a JSON file
        with open(f"./models/SVM/best_params_for_model_{self.settings['model_type']}.json", "w") as f:
            json.dump(random_search.best_params_, f)

        return self.model
        
    def run(self):
        df = self.fetch_or_read_data()
        df = self.preprocess_data(df)
        print(f'Dataframe shape:{df.shape}')
        print(df.head(5))   

        train_data, test_data = self.split_data(df) 
        
        X_train, y_train, X_test, y_test = self.extract_features_and_target(train_data, test_data, target_column='close')
        
        # Create directory
        models_directory = f'./models/SVM/'        
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)

        model = None  # Initialize the model variable

        if self.settings['use_cross_validation']:
            self.cross_validate_svm(X_train, y_train)
        else:
            if self.settings['use_hp_tuning']:
                try:
                    model = self.hp_tuning(X_train, y_train, X_test, y_test)
                except Exception as e:
                    logger.error(f"Error tuning hyperparameters: {e}")
            else:
                try:
                    model = self.train_svm(X_train, y_train)
                    predictions = self.evaluate_svm(model, X_test, y_test)
                except Exception as e:
                    logger.error(f"Error training model: {e}")       

        # Save the model
        if model:  # Check if the model is not None
            try:
                joblib.dump(model, f'./models/SVM/{self.settings["model_type"]}_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.pkl')
            except Exception as e:
                logger.error(f"Error saving model: {e}")
        else:
            logger.error("Failed to save model. Model is None")
        


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SVM models train.')
    parser.add_argument('--model_type', choices=['SVR', 'LinearSVR', 'NuSVR', 'SVC', 'LinearSVC', 'NuSVC', 'OneClassSVM'], 
                        type=str, default='SVR', help='Select model type')
    parser.add_argument('--use_hp_tuning', type=bool, default=True, help='Use hyperparameter tuning') 
    parser.add_argument('--use_cross_validation', type=bool, default=False, help='Use cross-validation')   
    parser.add_argument('--scaler_type', choices=['standard', 'minmax', 'robust', 'quantile', 'power'], 
                        type=str, default='minmax', help='Select data scaler type')
    parser.add_argument('--use_fetched_data', type=bool, default=False, help='Use fetched data')    
    args = parser.parse_args()

    SETTINGS = {
        'target_coin': 'GMT',
        'base_currency': 'USDT',
        'binance_timeframe': '30m',
        'start_date': dt.datetime.strptime("2023-05-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'end_date': dt.datetime.strptime("2023-08-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'use_fetched_data': args.use_fetched_data,        
        'model_type': args.model_type, # SVR, LinearSVR, NuSVR, SVC, LinearSVC, NuSVC, OneClassSVM
        'scaler_type': args.scaler_type,
        'use_hp_tuning': args.use_hp_tuning,
        'use_cross_validation': args.use_cross_validation       
    }
    svm_models = SVMModels(SETTINGS)
    svm_models.run()