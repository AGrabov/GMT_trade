from gym import spaces
from matplotlib.pyplot import plot as plt
from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
from optuna import Trial, create_study
from hyperopt import hp, fmin, tpe, Trials, space_eval
from sympy import true
from env_001 import TradeEnv

import pandas as pd
import datetime as dt
import shap
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
import os
import numpy as np
import argparse
from data_feed_01 import BinanceFuturesData


import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Settings
target_coin = 'GMT'
base_currency = 'USDT' # 'BUSD' # 
binance_timeframe = '30m'
start_date = dt.datetime.strptime(
            "2023-05-01 00:00:00", 
            "%Y-%m-%d %H:%M:%S")
end_date = dt.datetime.strptime(
            "2023-08-01 00:00:00",
             "%Y-%m-%d %H:%M:%S")

symbol = target_coin + base_currency
dataname = (f'{target_coin}/{base_currency}')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Model trading with the optimized "best_params".')
parser.add_argument('--model_type', choices=['DQN', 'DDPG', 'PPO', 'A2C', 'SAC', 'TD3'], type=str, default='A2C', help='Select model type')
parser.add_argument('--param_optimization', type=bool, default=True, help='Hyper parameters optimization')
parser.add_argument('--hp_optimizer_type', choices=['optuna', 'hyperopt'], type=str, default='optuna', help='Select data scaler type')
parser.add_argument('--use_fetched_data', type=bool, default=False, help='Use fetched data')

args = parser.parse_args()

use_fetched_data = args.use_fetched_data
param_optimization = args.param_optimization
model_type = args.model_type
hp_optimizer_type = args.hp_optimizer_type


# This function optimizes hyperparameters using Optuna
def optuna_objective(trial: Trial, df):
    # global df  # Declare df as a global variable
        
    observation_window_length = 48*7
    if model_type == 'A2C':
        params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "n_steps": trial.suggest_int("n_steps", 1, 15),
            "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),
            "gae_lambda": trial.suggest_uniform("gae_lambda", 0.9, 1.0),
            "ent_coef": trial.suggest_uniform("ent_coef", 0.0, 0.1),
            "vf_coef": trial.suggest_uniform("vf_coef", 0.1, 1.0),
            "max_grad_norm": trial.suggest_uniform("max_grad_norm", 0.1, 1.0),
            "rms_prop_eps": trial.suggest_uniform("rms_prop_eps", 1e-6, 1e-4),
            "use_rms_prop": trial.suggest_categorical("use_rms_prop", [True, False]),
            "use_sde": trial.suggest_categorical("use_sde", [True, False]),
            "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False]),
            "policy_kwargs": {
                "net_arch": {
                    "pi": [trial.suggest_int("pi_1", 32, 512, step=32), trial.suggest_int("pi_2", 32, 512, step=32)],
                    "qf": [trial.suggest_int("qf_1", 32, 512, step=32), trial.suggest_int("qf_2", 32, 512, step=32)]
                }
            }
        }
        action_space = spaces.MultiDiscrete([5])
        ddqn = False        
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=observation_window_length)        
        model = A2C('MlpPolicy', env, verbose=2, **params)
        
    elif model_type == 'PPO':
        params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "n_steps": trial.suggest_int("n_steps", 1024, 4097, step=1024),
            "n_epochs": trial.suggest_int("n_epochs", 5, 20),
            "batch_size": trial.suggest_int("batch_size", 32, 257, step=32),
            "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),
            "gae_lambda": trial.suggest_uniform("gae_lambda", 0.9, 1.0),
            "ent_coef": trial.suggest_uniform("ent_coef", 0.0, 0.1),
            "vf_coef": trial.suggest_uniform("vf_coef", 0.1, 1.0, step=0.1),
            "max_grad_norm": trial.suggest_uniform("max_grad_norm", 0.1, 1.0, step=0.1),
            "use_rms_prop": trial.suggest_categorical("use_rms_prop", [True, False]),
            "use_sde": trial.suggest_categorical("use_sde", [True, False]),
            "normalize_advantage": trial.suggest_categorical("normalize_advantage", [True, False]),
            "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.4, step=0.1),
            "clip_range_vf": trial.suggest_uniform("clip_range_vf", 0.1, 0.4, step=0.1),
            "policy_kwargs": {
                "net_arch": {
                    "pi": [trial.suggest_int("pi_1", 32, 512, step=32), trial.suggest_int("pi_2", 32, 512, step=32)],
                    "qf": [trial.suggest_int("qf_1", 32, 512, step=32), trial.suggest_int("qf_2", 32, 512, step=32)]
                }
            }
        }
        action_space = spaces.MultiDiscrete([5])
        ddqn = False
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=observation_window_length)
        model = PPO('MlpPolicy', env, verbose=2, **params)
        
    elif model_type == 'DQN':
        params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-2),
            "buffer_size": trial.suggest_int("buffer_size", 500000, 2000000, step=500000),
            "batch_size": trial.suggest_int("batch_size", 32, 257, step=32),
            "learning_starts": trial.suggest_int("learning_starts", 10000, 100000, step=10000),
            "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),            
            "tau": trial.suggest_uniform("tau", 0.0, 1,0, step=0.1),
            "train_freq": trial.suggest_int("train_freq", 1, 10, step=1),
            "gradient_steps": trial.suggest_int("gradient_steps", 0, 15, step=1),
            "target_update_interval": trial.suggest_int("target_update_interval", 1000, 50000, step=1000),
            "exploration_fraction": trial.suggest_uniform("exploration_fraction", 0.0, 0.5),
            "exploration_initial_eps": trial.suggest_loguniform("exploration_initial_eps", 0.8, 1.0),
            "exploration_final_eps": trial.suggest_loguniform("exploration_final_eps", 0.01, 0.1, step=0.01),            
            "max_grad_norm": trial.suggest_uniform("max_grad_norm", 5, 20, step=0.1),            
            "policy_kwargs": {
                "net_arch": {
                    "pi": [trial.suggest_int("pi_1", 32, 512, step=32), trial.suggest_int("pi_2", 32, 512, step=32)],
                    "qf": [trial.suggest_int("qf_1", 32, 512, step=32), trial.suggest_int("qf_2", 32, 512, step=32)]
                }
            }
        }
        action_space = spaces.Discrete(5)
        ddqn = False
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=observation_window_length)
        model = DQN('MlpPolicy', env, verbose=2, **params)  
        
    elif model_type in ['DDPG', 'SAC', 'TD3']:
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        ddqn = True
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                        observation_window_length=observation_window_length)
        
        if model_type == 'DDPG':
            params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "buffer_size": trial.suggest_int("buffer_size", 500000, 2000000, step=500000),
            "batch_size": trial.suggest_int("batch_size", 50, 300, step=50),
            "learning_starts": trial.suggest_int("learning_starts", 50, 300, step=50),
            "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),            
            "tau": trial.suggest_uniform("tau", 0.0, 0.1, step=0.001),            
            "gradient_steps": trial.suggest_int("gradient_steps", -1, 10, step=1),                 
            "policy_kwargs": {
                "net_arch": {
                    "pi": [trial.suggest_int("pi_1", 32, 512, step=32), trial.suggest_int("pi_2", 32, 512, step=32)],
                    "qf": [trial.suggest_int("qf_1", 32, 512, step=32), trial.suggest_int("qf_2", 32, 512, step=32)]
                }
            }
        }
            model = DDPG('MlpPolicy', env, verbose=2, **params)
        elif model_type == 'SAC':
            params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "buffer_size": trial.suggest_int("buffer_size", 500000, 2000000, step=500000),
            "batch_size": trial.suggest_int("batch_size", 128, 1025, step=128),
            "learning_starts": trial.suggest_int("learning_starts", 50, 300, step=50),
            "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),            
            "tau": trial.suggest_uniform("tau", 0.0, 0.1, step=0.001),            
            "gradient_steps": trial.suggest_int("gradient_steps", -1, 10, step=1),                 
            "policy_kwargs": {
                "net_arch": {
                    "pi": [trial.suggest_int("pi_1", 32, 512, step=32), trial.suggest_int("pi_2", 32, 512, step=32)],
                    "qf": [trial.suggest_int("qf_1", 32, 512, step=32), trial.suggest_int("qf_2", 32, 512, step=32)]
                }
            }
        }
            model = SAC('MlpPolicy', env, verbose=2, **params)
        elif model_type == 'TD3':
            params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "buffer_size": trial.suggest_int("buffer_size", 500000, 2000000, step=500000),
            "batch_size": trial.suggest_int("batch_size", 50, 300, step=50),
            "learning_starts": trial.suggest_int("learning_starts", 50, 300, step=50),
            "gamma": trial.suggest_uniform("gamma", 0.9, 0.9999),            
            "tau": trial.suggest_uniform("tau", 0.0, 0.1, step=0.001),            
            "policy_delay": trial.suggest_int("policy_delay", 0, 10, step=1), 
            "target_policy_noise": trial.suggest_uniform("target_policy_noise", 0.1, 0.6),                
            "target_noise_clip": trial.suggest_uniform("target_noise_clip", 0.0, 1.0, step=0.1),                
            "policy_kwargs": {
                "net_arch": {
                    "pi": [trial.suggest_int("pi_1", 32, 512, step=32), trial.suggest_int("pi_2", 32, 512, step=32)],
                    "qf": [trial.suggest_int("qf_1", 32, 512, step=32), trial.suggest_int("qf_2", 32, 512, step=32)]
                }
            }
        }
            model = TD3('MlpPolicy', env, verbose=2, **params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    if model_type in ['DDPG', 'TD3', 'DQN', 'SAC']:
        try:
            model.learn(total_timesteps=10000, progress_bar=True)        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
    else:
        try:
            model.learn(total_timesteps=len(df), progress_bar=True)        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
        
    # Evaluate the trained model on a validation set and return the performance    
    obs = env.reset()
    total_reward = 0
    true_rewards = []
    predicted_rewards = []
    for _ in range(3000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        true_rewards.append(reward)
        predicted_rewards.append(model.predict(obs)[0])
        if done:
            obs = env.reset()

    logger.info(f"Total reward: {total_reward}, True rewards: {true_rewards}, Predicted rewards: {predicted_rewards}")

    mse = mean_squared_error(true_rewards, predicted_rewards)
    mae = mean_absolute_error(true_rewards, predicted_rewards)
    mape = mean_absolute_percentage_error(true_rewards, predicted_rewards)
    r2 = r2_score(true_rewards, predicted_rewards)
    explained_variance = explained_variance_score(true_rewards, predicted_rewards)

    logger.info(f"Mean squared error: {mse}")
    logger.info(f"Mean absolute error: {mae}")
    logger.info(f"Mean absolute percentage error: {mape}")
    logger.info(f"R2: {r2}")
    logger.info(f"Explained variance: {explained_variance}")

    score = r2
    
    return score

def objective_hyperopt(df, model_type, params):
    # global df  # Declare df as a global variable
        
    observation_window_length = 48*7
    if model_type == 'A2C':        
        action_space = spaces.MultiDiscrete([5])
        ddqn = False        
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=observation_window_length)        
        model = A2C('MlpPolicy', env, verbose=2, **params)
        
    elif model_type == 'PPO':        
        action_space = spaces.MultiDiscrete([5])
        ddqn = False
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=observation_window_length)
        model = PPO('MlpPolicy', env, verbose=2, **params)
        
    elif model_type == 'DQN':        
        action_space = spaces.Discrete(5)
        ddqn = False
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=observation_window_length)
        model = DQN('MlpPolicy', env, verbose=2, **params)  
        
    elif model_type in ['DDPG', 'SAC', 'TD3']:
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        ddqn = True
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                        observation_window_length=observation_window_length)
        
        if model_type == 'DDPG':            
            model = DDPG('MlpPolicy', env, verbose=2, **params)
        elif model_type == 'SAC':            
            model = SAC('MlpPolicy', env, verbose=2, **params)
        elif model_type == 'TD3':            
            model = TD3('MlpPolicy', env, verbose=2, **params)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    if model_type in ['DDPG', 'TD3', 'DQN', 'SAC']:
        try:
            model.learn(total_timesteps=10000, progress_bar=True)        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
    else:
        try:
            model.learn(total_timesteps=len(df), progress_bar=True)        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
        
    # Evaluate the trained model on a validation set and return the performance    
    obs = env.reset()
    total_reward = 0
    true_rewards = []
    predicted_rewards = []
    for _ in range(3000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        true_rewards.append(reward)
        predicted_rewards.append(model.predict(obs)[0])
        if done:
            obs = env.reset()

    logger.info(f"Total reward: {total_reward}, True rewards: {true_rewards}, Predicted rewards: {predicted_rewards}")

    mse = mean_squared_error(true_rewards, predicted_rewards)
    mae = mean_absolute_error(true_rewards, predicted_rewards)
    mape = mean_absolute_percentage_error(true_rewards, predicted_rewards)
    r2 = r2_score(true_rewards, predicted_rewards)
    explained_variance = explained_variance_score(true_rewards, predicted_rewards)

    logger.info(f"Mean squared error: {mse}")
    logger.info(f"Mean absolute error: {mae}")
    logger.info(f"Mean absolute percentage error: {mape}")
    logger.info(f"R2: {r2}")
    logger.info(f"Explained variance: {explained_variance}")  

    return mse  # Optimization is assumed to be minimization in Hyperopt

def train_model(df, model_type, params=None):
    print(f"Learning {model_type} model...")  
    observation_window_length = 48*7
    
    if model_type == 'A2C':        
        action_space = spaces.MultiDiscrete([5])
        ddqn = False        
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=observation_window_length)        
        model = A2C('MlpPolicy', env, verbose=2, **params)
        
    elif model_type == 'PPO':
        
        action_space = spaces.MultiDiscrete([5])
        ddqn = False
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=observation_window_length)
        model = PPO('MlpPolicy', env, verbose=2, **params)
        
    elif model_type == 'DQN':
        
        action_space = spaces.Discrete(5)
        ddqn = False
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=observation_window_length)
        model = DQN('MlpPolicy', env, verbose=2, **params)  
        
    elif model_type in ['DDPG', 'SAC', 'TD3']:
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        ddqn = True
        env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                        observation_window_length=observation_window_length)
        
        if model_type == 'DDPG':            
            model = DDPG('MlpPolicy', env, verbose=2, **params)
        elif model_type == 'SAC':            
            model = SAC('MlpPolicy', env, verbose=2, **params)
        elif model_type == 'TD3':            
            model = TD3('MlpPolicy', env, verbose=2, **params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    if model_type in ['DDPG', 'TD3', 'DQN', 'SAC']:
        try:
            model.learn(total_timesteps=10000, progress_bar=True)
        except Exception as e:
            logger.error(f"Error during model training: {e}")
    else:
        try:
            model.learn(total_timesteps=len(df), progress_bar=True)        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            
    logger.info(f"Learning {model_type} model finished.")
    
    return model, env

def evaluate_model(df, model, env):
    obs = env.reset()
    true_rewards = []
    total_reward = 0
    predicted_rewards = []
    for _ in range(5000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        true_rewards.append(reward)
        predicted_rewards.append(model.predict(obs)[0])
        if done:
            obs = env.reset()

    mae = mean_absolute_error(true_rewards, predicted_rewards)
    mse = mean_squared_error(true_rewards, predicted_rewards)
    mape = mean_absolute_percentage_error(true_rewards, predicted_rewards)
    r2 = r2_score(true_rewards, predicted_rewards)
    explained_variance = explained_variance_score(true_rewards, predicted_rewards)

    results = {
        "true_rewards": true_rewards,
        "predicted_rewards": predicted_rewards,
        "total_reward": total_reward,
        "mae": mae,
        "mse": mse,
        "mape": mape,
        "r2": r2,
        "explained_variance": explained_variance
    }
    
    logger.info(f'MAE: {mae}, MSE: {mse}')
    logger.info(f'MAPE: {mape}, R2: {r2}')
    logger.info(f'Explained Variance: {explained_variance}')

    return results

if use_fetched_data:
    # Fetch the data for the specified symbol and time range
    try:
        df = BinanceFuturesData.fetch_data(symbol=symbol, startdate=start_date, enddate=end_date, binance_timeframe=binance_timeframe)
        print('Fteching data...')
    except Exception as e:
        print(f"Error fetching data: {e}")

else:
    csv_name = 'GMTUSDT - 30m_(since_2022-03-15).csv' #GMTUSDT - 30m_(since_2022-03-15).csv
    csv_path = f'./data_csvs/{csv_name}'
    # csv_path = '/root/gmt-bot/data_csvs/GMTUSDT - 30m_(since_2022-03-15).csv'
    symbol = 'GMTUSDT'
    binance_timeframe = '30m'
    try:
        # Read the CSV file using more efficient parameters
        df = pd.read_csv(csv_path, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    except Exception as e:
        print(f"Error reading CSV file: {e}")

    # Create a new DataFrame without empty rows
    df = df.dropna()

    # Convert timestamp to datetime format using vectorized operation
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Create a new DataFrame with the index set to the timestamp column
    df = df.set_index('timestamp')

# Check if the dataframe is valid and not empty
if df is None or df.empty:
    raise ValueError("Failed to fetch data or the data is empty.")

# Create a new DataFrame without empty rows
df = df.dropna()

print(df.head(3))
print(df.tail(2))
print()
print(f'Number of timesteps in the data: {len(df)}')

if param_optimization:
    if args.hp_optimizer_type == 'optuna':
        try:
            if model_type in ['DDPG', 'DQN', 'TD3', 'SAC']:
                study = create_study(direction="maximize")
                study.optimize(optuna_objective, n_trials=10, show_progress_bar=True)        
            else:
                study = create_study(direction="maximize")
                study.optimize(optuna_objective, n_trials=100, show_progress_bar=True)
            logger.info(study.best_params)

            best_params = study.best_params
            best_score = study.best_value            

        except Exception as e:
            print(f"Error during optuna optimization: {e}")

    elif args.hp_optimizer_type == 'hyperopt':
        if model_type == 'A2C':
            space = {
                "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
                "n_steps": hp.quniform("n_steps", 1, 15, 1),
                "gamma": hp.uniform("gamma", 0.9, 0.9999),
                "gae_lambda": hp.uniform("gae_lambda", 0.9, 1.0),
                "ent_coef": hp.uniform("ent_coef", 0.0, 0.1),
                "vf_coef": hp.uniform("vf_coef", 0.1, 1.0),
                "max_grad_norm": hp.uniform("max_grad_norm", 0.1, 1.0),
                "rms_prop_eps": hp.uniform("rms_prop_eps", 1e-6, 1e-4),
                "use_rms_prop": hp.choice("use_rms_prop", [True, False]),
                "use_sde": hp.choice("use_sde", [True, False]),
                "normalize_advantage": hp.choice("normalize_advantage", [True, False]),                
                "policy_kwargs": {
                    "net_arch": {
                        "pi": [hp.quniform("pi_1", 32, 512, 32), hp.quniform("pi_2", 32, 512, 32)],
                        "qf": [hp.quniform("qf_1", 32, 512, 32), hp.quniform("qf_2", 32, 512, 32)]
                    }
                }             
            }
        elif model_type == 'PPO':
            space = {
                "learning_rate": hp.loguniform("learning_rate", np.log(1e-5), np.log(1e-2)),
                "n_steps": hp.quniform("n_steps", 1024, 4097, 1024),
                "n_epochs": hp.quniform("n_epochs", 5, 20),
                "batch_size": hp.quniform("batch_size", 32, 257, 32),
                "gamma": hp.uniform("gamma", 0.9, 0.9999),
                "gae_lambda": hp.uniform("gae_lambda", 0.9, 1.0),
                "ent_coef": hp.uniform("ent_coef", 0.0, 0.1),
                "vf_coef": hp.uniform("vf_coef", 0.1, 1.0, 0.1),
                "max_grad_norm": hp.uniform("max_grad_norm", 0.1, 1.0, 0.1),                
                "use_sde": hp.choice("use_sde", [True, False]),
                "normalize_advantage": hp.choice("normalize_advantage", [True, False]),
                "clip_range": hp.uniform("clip_range", 0.1, 0.5),
                "clip_range_vf": hp.uniform("clip_range_vf", 0.1, 0.5), 
                "policy_kwargs": {
                    "net_arch": {
                        "pi": [hp.quniform("pi_1", 32, 512, 32), hp.quniform("pi_2", 32, 512, 32)],
                        "qf": [hp.quniform("qf_1", 32, 512, 32), hp.quniform("qf_2", 32, 512, 32)]
                    }
                }            
            }
        elif model_type == 'DQN':
            space = {
                "learning_rate": hp.loguniform("learning_rate", np.log(1e-6), np.log(1e-2)),
                "buffer_size": hp.quniform("buffer_size", 500000, 2000000, 500000),
                "learning_starts": hp.quniform("learning_starts", 10000, 100000, 10000),
                "batch_size": hp.quniform("batch_size", 32, 257, 32),
                "train_freq": hp.quniform("train_freq", 1, 10, 1),
                "gradient_steps": hp.quniform("gradient_steps", 0, 15, 1),
                "target_update_interval": hp.quniform("target_update_interval", 1000, 50000, 1000),
                "gamma": hp.uniform("gamma", 0.9, 0.9999),
                "tau": hp.uniform("tau", 0.0, 1.0, 0.1),
                "exploration_fraction": hp.uniform("exploration_fraction", 0.0, 0.5),
                "exploration_initial_eps": hp.uniform("exploration_initial_eps", 0.8 , 1.0),
                "exploration_final_eps": hp.uniform("exploration_final_eps", 0.01, 0.1),
                "max_grad_norm": hp.uniform("max_grad_norm", 5, 20, step=0.1),
                "policy_kwargs": {
                    "net_arch": {
                        "pi": [hp.quniform("pi_1", 32, 512, 32), hp.quniform("pi_2", 32, 512, 32)],
                        "qf": [hp.quniform("qf_1", 32, 512, 32), hp.quniform("qf_2", 32, 512, 32)]
                    }
                }      
            }
        elif model_type == 'DDPG':
            space = {
                "learning_rate": hp.loguniform("learning_rate", np.log(1e-6), np.log(1e-2)),
                "buffer_size": hp.quniform("buffer_size", 500000, 2000000, 500000),
                "learning_starts": hp.quniform("learning_starts", 50, 300, 50),
                "batch_size": hp.quniform("batch_size", 50, 300, 50),                
                "gamma": hp.uniform("gamma", 0.9, 0.9999),
                "tau": hp.uniform("tau", 0.0, 1.0, 0.001),
                "gradient_steps": hp.quniform("gradient_steps", -1, 15, 1),
                "policy_kwargs": {
                    "net_arch": {
                        "pi": [hp.quniform("pi_1", 32, 512, 32), hp.quniform("pi_2", 32, 512, 32)],
                        "qf": [hp.quniform("qf_1", 32, 512, 32), hp.quniform("qf_2", 32, 512, 32)]
                    }
                }      
            }
        elif model_type == 'SAC':
            space = {
                "learning_rate": hp.loguniform("learning_rate", np.log(1e-6), np.log(1e-2)),
                "buffer_size": hp.quniform("buffer_size", 500000, 2000000, 500000),
                "learning_starts": hp.quniform("learning_starts", 50, 300, 50),
                "batch_size": hp.quniform("batch_size", 128, 1025, 128),                
                "gamma": hp.uniform("gamma", 0.9, 0.9999),
                "tau": hp.uniform("tau", 0.0, 1.0, 0.001),
                "gradient_steps": hp.quniform("gradient_steps", -1, 10, 1),
                "policy_kwargs": {
                    "net_arch": {
                        "pi": [hp.quniform("pi_1", 32, 512, 32), hp.quniform("pi_2", 32, 512, 32)],
                        "qf": [hp.quniform("qf_1", 32, 512, 32), hp.quniform("qf_2", 32, 512, 32)]
                    }
                }      
            }
        elif model_type == 'TD3':
            space = {
                "learning_rate": hp.loguniform("learning_rate", np.log(1e-6), np.log(1e-2)),
                "buffer_size": hp.quniform("buffer_size", 500000, 2000000, 500000),
                "learning_starts": hp.quniform("learning_starts", 50, 300, 50),
                "batch_size": hp.quniform("batch_size", 50, 300, 50),                
                "gamma": hp.uniform("gamma", 0.9, 0.9999),
                "tau": hp.uniform("tau", 0.0, 1.0, 0.001),
                "policy_delay": hp.quniform("policy_delay", 0, 10, 1),
                "target_policy_noise": hp.uniform("target_policy_noise", 0.1, 0.6),
                "target_noise_clip": hp.uniform("target_noise_clip", 0.0, 1.0, 0.1),                
                "policy_kwargs": {
                    "net_arch": {
                        "pi": [hp.quniform("pi_1", 32, 512, 32), hp.quniform("pi_2", 32, 512, 32)],
                        "qf": [hp.quniform("qf_1", 32, 512, 32), hp.quniform("qf_2", 32, 512, 32)]
                    }
                }
            }
        models_directory = f'./models/RLM/{model_type}/{binance_timeframe}/'        
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)
        trials_file = f"{models_directory}Hyperopt_trials_{model_type}.json"
        
        if model_type in ['DDPG', 'SAC', 'TD3', 'DQN']:
            num_trials = 15
        else:
            num_trials = 50
        
        best = fmin(objective_hyperopt, space, algo=tpe.suggest, 
                    max_evals=num_trials, show_progressbar=True,
                    trials_save_file=trials_file)
        best_params = space_eval(space, best)
        # 

    model, env = train_model(df, model_type, best_params)
    results = evaluate_model(df, model, env)


    # Create directory
    models_directory = f'./models/RLM/{model_type}/{binance_timeframe}/'        
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    # Save the best parameters to a JSON file
    with open(f"{models_directory}best_params_for_model_{model_type}.json", "w") as f:
        json.dump(best_params, f)

    # Save the agent
    model.save(f"{models_directory}{model_type}_trade_{symbol}_{binance_timeframe}.zip")

else:
    models_directory = f'./models/RLM/{model_type}/{binance_timeframe}/'
    if os.path.exists(models_directory):
        for filename in os.listdir(models_directory):
            if filename.endswith('.json'):
                file_path = os.path.join(models_directory, filename)
                if f'best_params_for_model_{model_type}.json' in filename:
                    with open(file_path, "r") as f:
                        best_params = json.load(f)
                else:
                    print(f'Error: "best_params_for_model_{model_type}.json" do not exist')    

    model, env = train_model(df, model_type, best_params)
    results = evaluate_model(df, model, env)
    # Save the agent
    model.save(f"{models_directory}{model_type}_trade_{symbol}_{binance_timeframe}.zip")

    
    


    
