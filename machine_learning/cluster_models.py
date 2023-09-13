
import argparse
import datetime as dt
import json
import logging
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_feed_01 import BinanceFuturesData
from data_normalizer import CryptoDataNormalizer
from sklearn.cluster import (DBSCAN, AffinityPropagation,
                             AgglomerativeClustering, Birch, KMeans,
                             SpectralClustering)
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, mean_squared_log_error,
                             r2_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     TimeSeriesSplit, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)
from ta_indicators import TAIndicators
import traceback

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterModels:
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
    
    def create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:i+seq_length].values
            y = data.iloc[i+seq_length]['close']
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    # Apply Clustering
    def apply_clustering(self, X, n_clusters=3):
        if self.settings['model_type'] == 'KMeans':
            model = KMeans(n_clusters=n_clusters)

        elif self.settings['model_type'] == 'DBSCAN':
            model = DBSCAN()

        elif self.settings['model_type'] == 'Agglomerative':
            model = AgglomerativeClustering(n_clusters=n_clusters)

        elif self.settings['model_type'] == 'Spectral':
            model = SpectralClustering(n_clusters=n_clusters) 

        elif self.settings['model_type'] == 'AffinityPropagation':
            model = AffinityPropagation()

        elif self.settings['model_type'] == 'Birch':
            model = Birch(n_clusters=n_clusters)

        model.fit(X)
        return model    

    def evaluate_clustering(self, model, X_test, y_test):
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        evs = explained_variance_score(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        mape = mean_absolute_percentage_error(y_test, predictions)
        # msl = mean_squared_log_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)

        logger.info(f'R2 Score: {r2}')
        logger.info(f'Explained Variance Score: {evs}')
        logger.info(f'Mean Absolute Error: {mae}')
        logger.info(f'Mean Absolute Percentage Error: {mape}')
        # logger.info(f'Mean Squared Log Error: {msl}')
        logger.info(f'Mean Squared Error: {mse}')

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
        plt.savefig(f'./models/Cluster/{self.settings["model_type"]}/Predictions.png')
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
        plt.savefig(f'./models/Cluster/{self.settings["model_type"]}/Residuals_vs_Predictions.png')
        plt.show()
        
        # Histogram of Residuals
        plt.figure(figsize=(15, 6))
        plt.hist(residuals, bins=30, edgecolor='k')
        plt.title('Histogram of Residuals')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.savefig(f'./models/Cluster/{self.settings["model_type"]}/Residuals_hist.png')
        plt.show()    


    def hp_tuning(self, X_train, y_train, X_test, y_test):
        if self.settings['model_type'] == 'KMeans':
            parameters = {
                'n_clusters': range(1, 30),    
                'init': ['k-means++', 'random'],
                'n_init': ['auto'],
                'max_iter': range(200, 400),
                'tol': [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005],    
                'algorithm': ['lloyd'], # , 'elkan'
            }
            model = KMeans()

        elif self.settings['model_type'] == 'DBSCAN':
            parameters = {
                'eps': (0.1, 1.0),                
                'min_samples': (1, 10),                
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': (5, 50),
            }
            model = DBSCAN()

        elif self.settings['model_type'] == 'Agglomerative':
            parameters = {
                'n_clusters': range(1, 20), # [None, 5, 10, 15, 20],
                'linkage': ['ward', 'average', 'single'], #, 'complete'
                'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],                
                # 'compute_distances': [False, True],
            }
            model = AgglomerativeClustering()

        elif self.settings['model_type'] == 'Spectral':
            parameters = {
                'n_clusters': (1, 20),                
                'eigen_solver': ['arpack', 'lobpcg', 'amg', None],                
                'n_init': (0, 10),
                'gamma': (0, 1),
                'affinity': ['nearest_neighbors', 'rbf', 'precomputed', 'precomputed_nearest_neighbors'],
                'n_neighbors': (1, 10),                
                'assign_labels': ['kmeans', 'discretize', 'cluster_qr'],
                'degree': (1, 5),
                'coef0': (0.1, 1),                
            }
            model = SpectralClustering() 

        elif self.settings['model_type'] == 'AffinityPropagation':
            parameters = {
                'damping': (0.5, 1),
                'max_iter': (50, 300),
                'convergence_iter': (5, 20),                
                'affinity': ['euclidean', 'precomputed'],
            }
            model = AffinityPropagation()

        elif self.settings['model_type'] == 'Birch':
            parameters = {
                'threshold': (0.5),
                'branching_factor': range(10, 100),
                'n_clusters': [None, (1, 5)],
            }
            model = Birch()
        
        random_search = RandomizedSearchCV(model, parameters, n_iter=100, cv=5, scoring='explained_variance', verbose=2, n_jobs=-1, random_state=42) # neg_mean_absolute_error
        random_search.fit(X_train, y_train)            
            
        # Set the best model
        best_model = random_search.best_estimator_
            
        # Evaluate the best model
        predictions = self.evaluate_clustering(best_model, X_test, y_test)
            
        logger.info(f"Best Parameters: {random_search.best_params_}")
        logger.info(f"Best Score: {-random_search.best_score_}")

        # Save the best parameters to a JSON file
        with open(f"./models/Cluster/best_params_for_model_{self.settings['model_type']}.json", "w") as f:
            json.dump(random_search.best_params_, f)

        return best_model
        
        
    def run(self):
        model, best_params_ = None, None
        df = self.fetch_or_read_data()
        df = self.preprocess_data(df)
        train_data, test_data = self.split_data(df)
        X_train, y_train, X_test, y_test = self.extract_features_and_target(train_data, test_data, target_column='close')
        # X_train, y_train = self.create_sequences(train_data, self.settings['seq_length'])
        # X_test, y_test = self.create_sequences(test_data, self.settings['seq_length'])


        # Create directory
        models_directory = f'./models/Cluster/{self.settings["model_type"]}/'        
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)

        if self.settings['use_hp_tuning']:
            try:
                model = self.hp_tuning(X_train, y_train, X_test, y_test)
            except Exception as e:
                logger.error(f"Error tuning hyperparameters: {e}")
                logger.error(traceback.format_exc())  
        else:
            try:
                model = self.apply_clustering(X_train, n_clusters=self.settings['n_clusters'])
                predictions = self.evaluate_clustering(model, X_test, y_test)
            except Exception as e:
                logger.error(f"Error training model: {e}")
                logger.error(traceback.format_exc())  

        # Save the model
        if model:  # Check if the model is not None
            try:
                model_file = os.path.join(f'{models_directory}{self.settings["model_type"]}_{self.settings["scaler_type"]}_{self.settings["target_coin"] + self.settings["base_currency"]}_{self.settings["binance_timeframe"]}.pkl') 
                joblib.dump(model, model_file)
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                logger.error(traceback.format_exc())  
        else:
            logger.error("Failed to save model. Model is None")
            logger.error(traceback.format_exc())  

        

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Cluster models train.')
    parser.add_argument('--model_type', choices=['KMeans', 'DBSCAN', 'Agglomerative', 
                                                 'Spectral', 'AffinityPropagation', 'Birch'], 
                                                 type=str, default='KMeans', help='Select model type')
    parser.add_argument('--use_hp_tuning', type=bool, default=True, help='Use hyperparameter tuning')    
    parser.add_argument('--scaler_type', choices=['standard', 'minmax', 'robust', 'quantile', 'power'], 
                                        type=str, default='standard', help='Select data scaler type')
    parser.add_argument('--use_fetched_data', type=bool, default=False, help='Use fetched data')    
    args = parser.parse_args()

    SETTINGS = {
        'target_coin': 'GMT',
        'base_currency': 'USDT',
        'binance_timeframe': '30m',
        'start_date': dt.datetime.strptime("2023-05-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'end_date': dt.datetime.strptime("2023-08-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'use_fetched_data': args.use_fetched_data,        
        'model_type': args.model_type, # KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, AffinityPropagation, Birch
        'scaler_type': args.scaler_type,
        'use_hp_tuning': args.use_hp_tuning,
        'seq_length': 48,
        'n_clusters': 3,
    }
    cluster_models = ClusterModels(SETTINGS)
    cluster_models.run()