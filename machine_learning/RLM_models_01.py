import argparse
import datetime as dt
import json
import logging
import os
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from bayes_opt import BayesianOptimization
from data_feed_01 import BinanceFuturesData
from machine_learning.ta_indicators import TAIndicators
from gym import spaces
from env_001 import TradeEnv
from numba import jit
from optuna import Trial, create_study
from pyswarm import pso
from scipy.stats import randint, uniform
from functools import partial
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3 import HER, HerReplayBuffer 
from data_normalizer import CryptoDataNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLM_Models:
    def __init__(self, settings):
        self.settings = settings
        self.model = None
        self.params = {}
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
            # csv_path = '/root/gmt-bot/data_csvs/GMTUSDT - 30m_(since_2022-03-15).csv'
            df = pd.read_csv(csv_path, header=None, 
                             names=['timestamp', 'open', 'high', 'low', 'close', 'volume'], 
                             ) #skip_blank_lines=True
            df = df.dropna()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        return df

    def get_params(self):
        pass

    def get_a2c_model(self, df):        
        action_space = spaces.MultiDiscrete([5])
        ddqn = False        
        params_ranges = {            
            # 'learning_rate': float, #| Schedule = 0.0007,
            # 'n_steps': int = 5,
            # 'gamma': float = 0.99,
            # 'gae_lambda': float = 1,
            # 'ent_coef': float = 0,
            # 'vf_coef': float = 0.5,
            # 'max_grad_norm': float = 0.5,
            # 'rms_prop_eps': float = 0.00001,
            # 'use_rms_prop': bool = True,
            # 'use_sde': bool = False,
            # 'sde_sample_freq': int = -1,
            # 'normalize_advantage': bool = False,
            # 'stats_window_size': int = 100,
            # 'tensorboard_log': str | None = None,
            # 'policy_kwargs': Dict[str, Any] | None = None,
        }
        a2c_env = TradeEnv(df, action_space=action_space, 
                           live=False, ddqn=ddqn,
                           observation_window_length=self.settings['observation_window_length'], 
                           debug=self.settings['debug'])
        a2c_model = A2C('MlpPolicy', a2c_env, verbose=2 if self.settings['debug'] else 0, params=params_ranges)

        return a2c_model, a2c_env

    def get_ppo_model(self, df):
        action_space = spaces.MultiDiscrete([5])
        ddqn = False                
        params_ranges = {            
            'learning_rate': "float | Schedule = 0.0003",
            'n_steps': "int = 2048",
            'batch_size': "int = 64",
            'n_epochs': "int = 10",
            'gamma': "float = 0.99",
            'gae_lambda': "float = 0.95",
            'clip_range': "float | Schedule = 0.2",
            'clip_range_vf': "float | Schedule | None = None",
            'normalize_advantage': "bool = True",
            'ent_coef': "float = 0",
            'vf_coef': "float = 0.5",
            'max_grad_norm': "float = 0.5",
            'use_sde': "bool = False",
            'sde_sample_freq': "int = -1",
            'target_kl': "float | None = None",            
            'policy_kwargs': "Dict[str, Any] | None = None",            
            'seed': "int | None = None",            
            '_init_setup_model': "bool = True"
        }
        ppo_env = TradeEnv(df, action_space=action_space, 
                           live=False, ddqn=ddqn, 
                           observation_window_length=self.settings['observation_window_length'], 
                           debug=self.settings['debug'])
                
        ppo_model = PPO('MlpPolicy', ppo_env, verbose=2 if self.settings['debug'] else 0, params=params_ranges)

        return ppo_model, ppo_env
    
    def get_ddpg_model(self, df):
        pass

    def get_dqn_model(self, df):
        pass

    def get_td3_model(self, df):
        pass

    def get_sac_model(self, df):
        pass

    def train_rlm_model(self, df, params):
        if self.settings['model_type'] == 'A2C':
            action_space = spaces.MultiDiscrete([5])
            ddqn = False
            env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=self.settings['observation_window_length'], 
                           debug=self.settings['debug'])
            model = A2C('MlpPolicy', env, params=params)
            
        elif self.settings['model_type'] == 'PPO':
            action_space = spaces.MultiDiscrete([5])
            ddqn = False
            env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=self.settings['observation_window_length'], 
                           debug=self.settings['debug'])
            model = PPO('MlpPolicy', env, params=params)

        elif self.settings['model_type'] == 'DQN':
            action_space = spaces.Discrete(5)
            env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=self.settings['observation_window_length'], 
                           debug=self.settings['debug'])
            model = DQN('MlpPolicy', env, params=params)
            
        elif self.settings['model_type'] in ['DDPG', 'SAC', 'TD3']:
            action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            ddqn = True
            env = TradeEnv(df, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=self.settings['observation_window_length'], 
                           debug=self.settings['debug'])
            if self.settings['model_type'] == 'DDPG':
                model = DDPG('MlpPolicy', env, params=params)
            elif self.settings['model_type'] == 'SAC':
                model = SAC('MlpPolicy', env, params=params)
            elif self.settings['model_type'] == 'TD3':
                model = TD3('MlpPolicy', env, params=params)

        if self.settings['model_type'] in ['DDPG', 'SAC', 'TD3']:
            model.learn(total_timesteps=len(df), progress_bar=True)            
        elif self.settings['model_type'] in ['A2C', 'PPO', 'DQN']:
            model.learn(total_timesteps=len(df)*5, progress_bar=True)

        return model, env
    
    def evaluate_rlm_model(self, df, model, env):
        obs = env.reset()
        true_rewards = []
        predicted_rewards = []
        for _ in range(5000):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            true_rewards.append(reward)
            predicted_rewards.append(model.predict(obs)[0])
            if done:
                obs = env.reset()
        
        mae = mean_absolute_error(true_rewards, predicted_rewards)
        mse = mean_squared_error(true_rewards, predicted_rewards)
        mape = mean_absolute_percentage_error(true_rewards, predicted_rewards)
        r2 = r2_score(true_rewards, predicted_rewards)
        explained_variance = explained_variance_score(true_rewards, predicted_rewards)
        logger.info(f'MAE: {mae}, MSE: {mse}')
        logger.info(f'MAPE: {mape}, R2: {r2}')
        logger.info(f'Explained Variance: {explained_variance}')
        
        return true_rewards, predicted_rewards, obs
    
    def get_shap(self, model, obs):
        if self.settings['model_type'] == 'DQN':
            explainer = shap.Explainer(model.q_net)
        elif self.settings['model_type'] in ['A2C', 'PPO']:
            explainer = shap.Explainer(model.policy.vf_net)
        elif self.settings['model_type'] in ['DDPG', 'TD3', 'SAC']:
            explainer = shap.Explainer(model.critic)

        shap_values = explainer.shap_values(obs)
        print(shap_values)
        shap.summary_plot(shap_values, obs)

    def visualize_predictions(self):
        pass
        

    def plot_residuals(self):
        pass

    def cross_validate_rlm(self):
        pass

    def hp_tuning(self, df):
        pass    

    def save_model(self, model):
        models_directory = f'./models/RLM/{self.settings["model_type"]}/'        
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)
        # Save the agent
        model_filename = f'{models_directory}{self.settings["model_type"]}_trade_{self.settings["symbol"]}_{self.settings["binance_timeframe"]}.zip'
        model.save(model_filename)
        logger.info(f"Model saved: {model_filename}")

    def save_params(self, params):
        params_filename = f'./params/RLM/{self.settings["model_type"]}/'        
        if not os.path.exists(params_filename):
            os.makedirs(params_filename)
        # Save the agent
        params_filename = f'{params_filename}{self.settings["model_type"]}_trade_{self.settings["symbol"]}_{self.settings["binance_timeframe"]}.json'
        with open(params_filename, 'w') as f:
            json.dump(params, f)
        logger.info(f"Params saved: {params_filename}")
        
    def run(self):
        df = self.fetch_or_read_data()
        params = self.get_params()
        model, env = self.train_rlm_model(df, params)
        true_rewards, predicted_rewards, obs = self.evaluate_rlm_model(df, model, env)
        self.get_shap(model, obs)
        self.save_params(params)
        self.save_model(model)

        
        


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Reinforcement learning models training and tuning.')
    parser.add_argument('--model_type', choices=['A2C', 'DDPG', 'DQN', 'PPO', 'SAC', 'TD3'], 
                        type=str, default='A2C', help='Select model type from A2C, DDPG, DQN, PPO, SAC, TD3')
    parser.add_argument('--use_hp_tuning', type=bool, default=True, help='Use hyperparameter tuning') 
    parser.add_argument('--use_cross_validation', type=bool, default=False, help='Use cross-validation')    
    parser.add_argument('--use_fetched_data', type=bool, default=False, help='Use fetched data')    
    args = parser.parse_args()

    SETTINGS = {
        'target_coin': 'GMT',
        'base_currency': 'USDT',
        'binance_timeframe': '30m',
        'start_date': dt.datetime.strptime("2023-05-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'end_date': dt.datetime.strptime("2023-08-01 00:00:00", "%Y-%m-%d %H:%M:%S"),
        'use_fetched_data': args.use_fetched_data,        
        'model_type': args.model_type, #         
        'use_hp_tuning': args.use_hp_tuning,
        'use_cross_validation': args.use_cross_validation,
        
    }
    svm_models = RLM_Models(SETTINGS)
    svm_models.run()