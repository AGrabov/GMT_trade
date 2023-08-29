from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_log_error
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, AffinityPropagation, Birch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
import datetime as dt
from data_feed_01 import BinanceFuturesData
from ta_indicators import TAIndicators
from data_normalizer import CryptoDataNormalizer
import logging

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

    # Apply Clustering
    def apply_clustering(self, X, n_clusters=3):
        if self.settings['model_type'] == 'KMeans':
            model = KMeans(n_clusters=n_clusters)

        elif self.settings['model_type'] == 'DBSCAN':
            model = DBSCAN()

        elif self.settings['model_type'] == 'AgglomerativeClustering':
            model = AgglomerativeClustering(n_clusters=n_clusters)

        elif self.settings['model_type'] == 'SpectralClustering':
            model = SpectralClustering(n_clusters=n_clusters) 

        elif self.settings['model_type'] == 'AffinityPropagation':
            model = AffinityPropagation()

        elif self.settings['model_type'] == 'Birch':
            model = Birch(n_clusters=n_clusters)

        model.fit(X)
        return model    

    def hp_tuning(self, X_train, y_train, X_test, y_test):
        pass    
        
    def run(self):
        df = self.fetch_or_read_data()
        df = self.preprocess_data(df)        

        # Create directory
        models_directory = f'./models/Cluster/'        
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)

        

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Cluster models train.')
    parser.add_argument('--model_type', choices=['KMeans', 'DBSCAN', 'AgglomerativeClustering', 
                                                 'SpectralClustering', 'AffinityPropagation', 'Birch'], 
                                                 type=str, default='KMeans', help='Select model type')
    parser.add_argument('--use_hp_tuning', type=bool, default=True, help='Use hyperparameter tuning')    
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
        'model_type': args.model_type, # KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, AffinityPropagation, Birch
        'scaler_type': args.scaler_type,
        'use_hp_tuning': args.use_hp_tuning,        
    }
    cluster_models = ClusterModels(SETTINGS)
    cluster_models.run()