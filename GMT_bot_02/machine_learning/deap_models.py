# DEAP models

import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
import datetime as dt
from data_feed_01 import BinanceFuturesData
from data_normalizer import CryptoDataNormalizer
from ta_indicators import TAIndicators
import joblib
import json
import argparse


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
target_coin = 'GMT'
base_currency = 'USDT'
binance_timeframe = '30m'
use_fteched_data = False
start_date = dt.datetime.strptime("2023-05-01 00:00:00", "%Y-%m-%d %H:%M:%S")
end_date = dt.datetime.strptime("2023-08-01 00:00:00", "%Y-%m-%d %H:%M:%S")

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Deap learning model trading with the optimized "best_params".')
parser.add_argument('--model_type', choices=['RF', 'GBM', 'ABR', 'ETR'], type=str, default='GBM', help='Select model type')
parser.add_argument('--search_type', choices=['randomized', 'grid'], type=str, default='randomized', help='Select best params search type')
parser.add_argument('--scaler_type', choices=['standard', 'minmax', 'robust', 'quantile', 'power'], type=str, default='minmax', help='Select data scaler type')
parser.add_argument('--use_fetched_data', type=bool, default=False, help='Use fetched data')
parser.add_argument('--debug', type=bool, default=True, help='Use fetched data')

args = parser.parse_args()

model_type = args.model_type
search_type = args.search_type
scaler_type = args.scaler_type
debug = args.debug

# Construct symbol and dataname
symbol = target_coin + base_currency
dataname = f'{target_coin}/{base_currency}'

# Initialize data normalizer
normalizer = CryptoDataNormalizer()
predictions = []  # Initialize predictions list



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

# Calculate technical indicators
df = TAIndicators(df)._calculate_indicators()
if debug:
    logger.info("Added TA Indicators")
    print(df.head(3))
    print(df.tail(1))
    logger.info(f"Data length: {len(df)}")

# Normalize the data
scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler(),
    'quantile': QuantileTransformer(),
    'power': PowerTransformer()
}
scaler = scalers.get(scaler_type)
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns, index=df.index)
if debug:
    logger.info(f"Scaler: {scaler_type}")
    print(f"Normalized data:")
    print(scaled_df.head(3))
    logger.info(f"Scaled data length: {len(scaled_df)}")

# Define features and target variable
X = scaled_df.drop('close', axis=1)
y = scaled_df['close']

# Split the data into training, validation, and test sets
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))
test_size = len(df) - train_size - val_size

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(val_size + test_size), random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=test_size, random_state=42)

# Early stopping parameters
early_stopping_rounds = 50
eval_set = [(X_val, y_val)]
eval_metric = 'rmse'  # Root Mean Squared Error

# Define hyperparameters grid for each model
param_grids = {
    'RF': {
        'n_estimators': [10, 50, 100, 200, 500],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [10, 20, 30, 40, 50, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'absolute_error', 'poisson', 'friedman_mse'],
        'warm_start': [True, False],
        'oob_score': [True, False]
    },
    'GBM': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'max_features': ['sqrt', 'log2', None]
    },
    'ABR': {        
        'n_estimators': [10, 50, 100, 200],
        'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0],
        'loss': ['linear', 'square', 'exponential']
    },
    'ETR': {
        'n_estimators': [10, 50, 100, 200],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
}

# Initialize the model based on model_type
models = {
    'RF': RandomForestRegressor(),
    'GBM': GradientBoostingRegressor(),
    'ABR': AdaBoostRegressor(),
    'ETR': ExtraTreesRegressor()
}

names = {
    'RF': 'Random Forest',
    'GBM': 'Gradient Boosting',
    'ABR': 'AdaBoost',
    'ETR': 'Extra Trees'
}
model = models.get(model_type)
name = names.get(model_type)

# Choose between RandomizedSearchCV and GridSearchCV
if search_type == 'randomized':
    search = RandomizedSearchCV(estimator=model, param_distributions=param_grids[model_type], n_iter=100, cv=5, verbose=2 if debug else 0, random_state=42, n_jobs=-1)    
elif search_type == 'grid':
    search = GridSearchCV(estimator=model, param_grid=param_grids[model_type], cv=5, verbose=2 if debug else 0, n_jobs=-1)    
else:
    logger.info(f"search_type {search_type} not supported. Please select from 'randomized' and 'grid'")

# Fit the model on training data and validate on validation set
search.fit(X_train, y_train)
model = search.best_estimator_

# Predict and evaluate on validation set
y_val_pred = model.predict(X_val)
mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
mape_val = mean_absolute_percentage_error(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)

if debug:
    logger.info(f"Validation Mean Squared Error: {mse_val}")
    logger.info(f"Validation Mean Absolute Error: {mae_val}")
    logger.info(f"Validation R2 Score: {r2_val}")
    logger.info(f"Validation Mean Absolute Percentage Error: {mape_val}")
    logger.info(f"Validation Root Mean Squared Error: {rmse_val}")

# Predict and evaluate on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
rmse = np.sqrt(mse)

if debug:
    logger.info(f"Test Mean Squared Error: {mse}")
    logger.info(f"Test Mean Absolute Error: {mae}")
    logger.info(f"Test R2 Score: {r2}")
    logger.info(f"Test Mean Absolute Percentage Error: {mape}")
    logger.info(f"Test Root Mean Squared Error: {rmse}")
    
predictions.append(y_pred)
predictions = np.concatenate(predictions, axis=0)

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.values, label='Actual', color='blue')
plt.plot(y_test.index, y_pred, label='Predicted', color='red', linestyle='--')
plt.title(f"{name} - Actual vs Predicted")
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Feature importance
feature_importances = model.feature_importances_
sorted_idx = np.argsort(feature_importances)
plt.barh(X.columns[sorted_idx], feature_importances[sorted_idx])
plt.xlabel(f"{name} Feature Importance")
plt.show()

# Save the model and scaler
joblib.dump(model, f'./models/{model_type}/{model_type}_{search_type}_{symbol}_{binance_timeframe}.pkl')
joblib.dump(scaler, f'./models/{model_type}/{model_type}_scaler_{scaler_type}_{symbol}_{binance_timeframe}.pkl')
