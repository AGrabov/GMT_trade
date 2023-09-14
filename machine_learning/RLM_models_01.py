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
from gym import spaces
from env_001 import TradeEnv
from numba import jit
from optuna import Trial, create_study
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import uniform, randint, loguniform

from functools import partial
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3, HER, HerReplayBuffer  
from data_normalizer import CryptoDataNormalizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RLM_Models:
    def __init__(self, settings):
        self.settings = settings
        self.models_dir = f'./models/RLM/{self.settings["model_type"]}/'
        self.model = None
        self.env = None
        self.data = None  # You can replace this with actual data initialization
        self.models_params = {
            "a2c": None,
            "ppo": None,
            "dqn": None,
            "sac": None,
            "td3": None,
            "ddpg": None,
            "her": None
        }
        self.results = {}
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

    def get_existing_best_params(self):        
        if os.path.exists(self.models_dir):
            for filename in os.listdir(self.models_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(self.models_dir, filename)
                    if 'params' in filename:
                        with open(filename, 'r') as f:
                            best_params = json.load(f)
                            logger.info(f'Loaded best params from: {file_path}')
                            return best_params
                else:                    
                    logger.info(f'No best params found for: {self.settings["model_type"]}')
                    return None

            
    def get_a2c_params(self):
        self.models_params["a2c"] = {
            "policy": ["MlpPolicy", "CnnPolicy"],
            "learning_rate": [(1e-5, 1e-2), "log-uniform"],
            "n_steps": [1, 5, 10, 20, 50],
            "gamma": [(0.9, 0.9999), "uniform"],
            "gae_lambda": [(0.9, 1.0), "uniform"],
            "ent_coef": [(0.0, 0.1), "uniform"],
            "vf_coef": [(0.1, 1.0), "uniform"],
            "max_grad_norm": [(0.1, 1.0), "uniform"],
            "rms_prop_eps": [(1e-6, 1e-4), "log-uniform"],
            "use_rms_prop": [True, False],
            "use_sde": [True, False],
            "sde_sample_freq": [-1, 1, 5, 10], 
            "normalize_advantage": [True, False],
            "stats_window_size": [10, 50, 100, 200],            
            "policy_kwargs": [
                        {"net_arch": [x, y], "activation_fn": fn} 
                        for x in np.arange(32, 257, 32) 
                        for y in np.arange(32, 513, 32) 
                        for fn in ["ReLU", "Tanh"]
                    ],
        }

    def get_ppo_params(self):
        self.models_params["ppo"] = {
            "policy": ["MlpPolicy", "CnnPolicy"],            
            "learning_rate": [(1e-5, 1e-2), "log-uniform"],
            "n_steps": [128, 256, 512, 1024, 2048],
            "batch_size": [64, 128, 256, 512],
            "gamma": [(0.9, 0.9999), "uniform"],
            "gae_lambda": [(0.9, 1.0), "uniform"],
            "ent_coef": [(0.0, 0.1), "uniform"],
            "vf_coef": [(0.1, 1.0), "uniform"],
            "max_grad_norm": [(0.1, 1.0), "uniform"],
            "clip_range": [(0.1, 0.3), "uniform"],
            "clip_range_vf": [None, (0.1, 0.3), "uniform"],
            "lam": [(0.8, 1.0), "uniform"],
            "policy_kwargs": [
                {"net_arch": [x, y], "activation_fn": fn} 
                for x in np.arange(32, 257, 32) 
                for y in np.arange(32, 513, 32) 
                for fn in ["ReLU", "Tanh"]
            ],
        }

    def get_dqn_params(self):
        self.models_params["dqn"] = {
            "policy": ["MlpPolicy", "CnnPolicy"],
            # "env": ["path/to/your/custom/env"],  # Replace with the actual path to your custom environment
            "learning_rate": [(1e-5, 1e-2), "log-uniform"],
            "buffer_size": [10000, 50000, 100000],
            "learning_starts": [1000, 5000, 10000],
            "batch_size": [32, 64, 128, 256],
            "tau": [(0.01, 0.1), "uniform"],
            "gamma": [(0.9, 0.9999), "uniform"],
            "train_freq": [1, 4, 8],
            "gradient_steps": [1, 4, 8],
            "n_episodes_rollout": [-1, 100, 200],
            "policy_kwargs": [
                {"net_arch": [x, y], "activation_fn": fn} 
                for x in np.arange(32, 257, 32) 
                for y in np.arange(32, 513, 32) 
                for fn in ["ReLU", "Tanh"]
            ],
        }

    def get_ddpg_params(self):
        self.models_params["ddpg"] = {
            "policy": ["MlpPolicy", "CnnPolicy"],
            # "env": ["path/to/your/custom/env"],  # Replace with the actual path to your custom environment
            "learning_rate": [(1e-5, 1e-2), "log-uniform"],
            "buffer_size": [10000, 50000, 100000],
            "learning_starts": [1000, 5000, 10000],
            "batch_size": [64, 128, 256],
            "tau": [(0.01, 0.1), "uniform"],
            "gamma": [(0.9, 0.9999), "uniform"],
            "train_freq": [100, 200, 500],
            "gradient_steps": [100, 200, 500],
            "action_noise": ["NormalActionNoise", "OrnsteinUhlenbeckActionNoise"],  # You can add more noise types as needed
            "noise_std": [(0.1, 0.5), "uniform"],
            "optimize_memory_usage": [True, False],
            "policy_kwargs": [
                {"net_arch": [x, y], "activation_fn": fn} 
                for x in np.arange(32, 257, 32) 
                for y in np.arange(32, 513, 32) 
                for fn in ["ReLU", "Tanh"]
            ],
        }

    def get_sac_params(self):
        self.models_params["sac"] = {
            "policy": ["MlpPolicy", "CnnPolicy"],
            # "env": ["path/to/your/custom/env"],  # Replace with the actual path to your custom environment
            "learning_rate": [(1e-5, 1e-2), "log-uniform"],
            "buffer_size": [10000, 50000, 100000],
            "learning_starts": [1000, 5000, 10000],
            "batch_size": [64, 128, 256],
            "tau": [(0.01, 0.1), "uniform"],
            "gamma": [(0.9, 0.9999), "uniform"],
            "train_freq": [1, 5, 10],
            "gradient_steps": [1, 5, 10],
            "ent_coef": ["auto", (0.01, 0.1), "uniform"],
            "target_entropy": ["auto", (-1.0, 1.0), "uniform"],
            "action_noise": ["NormalActionNoise", "OrnsteinUhlenbeckActionNoise"],  # You can add more noise types as needed
            "noise_std": [(0.1, 0.5), "uniform"],
            "use_sde": [True, False],
            "sde_sample_freq": [-1, 1, 5, 10],
            "optimize_memory_usage": [True, False],
            "policy_kwargs": [
                {"net_arch": [x, y], "activation_fn": fn} 
                for x in np.arange(32, 257, 32) 
                for y in np.arange(32, 513, 32) 
                for fn in ["ReLU", "Tanh"]
            ],
        }

    def get_td3_params(self):
        self.models_params["td3"] = {
            "policy": ["MlpPolicy", "CnnPolicy"],
            # "env": ["path/to/your/custom/env"],  # Replace with the actual path to your custom environment
            "learning_rate": [(1e-5, 1e-2), "log-uniform"],
            "buffer_size": [10000, 50000, 100000],
            "learning_starts": [1000, 5000, 10000],
            "batch_size": [64, 128, 256],
            "tau": [(0.01, 0.1), "uniform"],
            "gamma": [(0.9, 0.9999), "uniform"],
            "train_freq": [100, 200, 500],
            "gradient_steps": [100, 200, 500],
            "action_noise": ["NormalActionNoise", "OrnsteinUhlenbeckActionNoise"],  # You can add more noise types as needed
            "noise_std": [(0.1, 0.5), "uniform"],
            "policy_delay": [1, 2],
            "target_policy_noise": [(0.1, 0.3), "uniform"],
            "target_noise_clip": [(0.1, 0.5), "uniform"],
            "optimize_memory_usage": [True, False],
            "policy_kwargs": [
                {"net_arch": [x, y], "activation_fn": fn} 
                for x in np.arange(32, 257, 32) 
                for y in np.arange(32, 513, 32) 
                for fn in ["ReLU", "Tanh"]
            ],
        }

    def get_her_params(self):
        self.models_params["her"] = {
            "policy": ["MlpPolicy", "CnnPolicy"],
            # "env": ["path/to/your/custom/env"],  # Replace with the actual path to your custom environment
            "goal_selection_strategy": ["final", "future", "episode"],
            "online_sampling": [True, False],
            "learning_rate": [(1e-5, 1e-2), "log-uniform"],
            "buffer_size": [10000, 50000, 100000],
            "learning_starts": [1000, 5000, 10000],
            "batch_size": [64, 128, 256],
            "gamma": [(0.9, 0.9999), "uniform"],
            "train_freq": [100, 200, 500],
            "gradient_steps": [100, 200, 500],
            "n_sampled_goal": [4, 5, 10],
            "optimize_memory_usage": [True, False],
            "policy_kwargs": [
                {"net_arch": [x, y], "activation_fn": fn} 
                for x in np.arange(32, 257, 32) 
                for y in np.arange(32, 513, 32) 
                for fn in ["ReLU", "Tanh"]
            ],
        }

    def get_params(self):
        if self.settings['model_type'] == 'A2C':
            self.get_a2c_params()
        elif self.settings['model_type'] == 'PPO':
            self.get_ppo_params()
        elif self.settings['model_type'] == 'DQN':
            self.get_dqn_params()
        elif self.settings['model_type'] == 'SAC':
            self.get_sac_params()
        elif self.settings['model_type'] == 'TD3':
            self.get_td3_params()
        elif self.settings['model_type'] == 'DDPG':
            self.get_ddpg_params()
        elif self.settings['model_type'] == 'HER':
            self.get_her_params()
            
    def get_model_and_env(self, params=None):
        policy = 'MlpPolicy' if params is None or params.get('policy') is None else params['policy']

        if self.settings['use_hp_tuning'] and self.settings['optimizer_type'] == "bayesian":

            # Assuming activation_fn_mapping is defined earlier in your script
            activation_fn_mapping = {
                # Your mapping here, for example:
                "relu": "Relu Activation Function",
                "sigmoid": "Sigmoid Activation Function"
            }

            # Extract net_arch and activation_fn values from params
            net_arch_x = params.pop('net_arch_x')
            net_arch_y = params.pop('net_arch_y')
            activation_fn = activation_fn_mapping[params.pop('activation_fn')]

            # Create policy_kwargs dictionary
            policy_kwargs = {
                "net_arch": [net_arch_x, net_arch_y],
                "activation_fn": activation_fn  # Now activation_fn will have the correct string value
            }

            # Add policy_kwargs to params
            params['policy_kwargs'] = policy_kwargs
            
        if self.settings['model_type'] == 'A2C':
            action_space = spaces.MultiDiscrete([5])
            ddqn = False            
            env = TradeEnv(self.data, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=self.settings['observation_window_length'], 
                           )
            model = A2C(policy, env, verbose=2 if self.settings['debug'] else 1, **params) 
            
        elif self.settings['model_type'] == 'PPO':
            action_space = spaces.MultiDiscrete([5])
            ddqn = False
            env = TradeEnv(self.data, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=self.settings['observation_window_length'], 
                           )
            model = PPO(policy, env, verbose=2 if self.settings['debug'] else 1, **params)

        elif self.settings['model_type'] == 'DQN':
            action_space = spaces.Discrete(5)
            ddqn = False
            env = TradeEnv(self.data, action_space=action_space, ddqn=ddqn, live=False, 
                           observation_window_length=self.settings['observation_window_length'], 
                           )
            model = DQN(policy, env, verbose=2 if self.settings['debug'] else 1, **params)
            
        elif self.settings['model_type'] in ['DDPG', 'SAC', 'TD3']:
            action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
            ddqn = True
            env = TradeEnv(self.data, action_space=action_space, ddqn=ddqn, live=False, 
                        observation_window_length=self.settings['observation_window_length'], 
                        )
            if self.settings['model_type'] == 'DDPG':
                model = DDPG(policy, env, verbose=2 if self.settings['debug'] else 1, **params)
            elif self.settings['model_type'] == 'SAC':
                model = SAC(policy, env, verbose=2 if self.settings['debug'] else 1, **params)
            elif self.settings['model_type'] == 'TD3':
                model = TD3(policy, env, verbose=2 if self.settings['debug'] else 1, **params)

        else:
            raise ValueError(f"Unknown model type: {self.settings['model_type']}")

        return model, env
    

    def train_rlm_model(self, model):
        
        if self.settings['model_type'] in ['DDPG', 'SAC', 'TD3']:
            model.learn(total_timesteps=len(self.data), progress_bar=True)            
        elif self.settings['model_type'] in ['A2C', 'PPO', 'DQN']:
            model.learn(total_timesteps=len(self.data)*2, progress_bar=True)

        return model
    
    def evaluate_rlm_model(self, model, env):
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
        
        mae = mean_absolute_error(true_rewards, predicted_rewards)
        mse = mean_squared_error(true_rewards, predicted_rewards)
        mape = mean_absolute_percentage_error(true_rewards, predicted_rewards)
        r2 = r2_score(true_rewards, predicted_rewards)
        explained_variance = explained_variance_score(true_rewards, predicted_rewards)
        logger.info(f'MAE: {mae}, MSE: {mse}')
        logger.info(f'MAPE: {mape}, R2: {r2}')
        logger.info(f'Explained Variance: {explained_variance}')        

        self.results = {
            'MAE': mae,
            'MSE': mse,
            'MAPE': mape,
            'R2': r2,
            'Explained Variance': explained_variance,

        }
        
        return true_rewards, predicted_rewards, total_reward
    
    def get_shap(self, obs):
        if self.settings['model_type'] == 'DQN':
            explainer = shap.Explainer(self.model.q_net)
        elif self.settings['model_type'] in ['A2C', 'PPO']:
            explainer = shap.Explainer(self.model.policy.vf_net)
        elif self.settings['model_type'] in ['DDPG', 'TD3', 'SAC']:
            explainer = shap.Explainer(self.model.critic)

        shap_values = explainer.shap_values(obs)
        print(shap_values)
        fig = shap.summary_plot(shap_values, obs)
        plt.savefig(f'{self.models_dir}{self.settings["model_type"]}_shap.png')
        

    def visualize_predictions(self):
        pass
        

    def plot_residuals(self):
        pass

    def cross_validate_rlm(self):
        pass

    # from stable_baselines3 import PPO
    # from torch.optim import Adam
    # model = PPO(policy, env, verbose=2, optimizer_class=Adam, optimizer_kwargs=dict(lr=0.001))


    def hp_tuning(self):        
        if self.settings['optimizer_type'] == "random":
            self.hp_tuning_random()
        elif self.settings['optimizer_type'] == "optuna":
            self.hp_tuning_optuna()
        elif self.settings['optimizer_type'] == "bayesian":
            self.hp_tuning_bayesian()
        else:
            raise ValueError("Unknown optimizer type")

            
    def hp_tuning_random(self):
        # Get the parameter space for the selected model type
        param_distributions = self.models_params[self.settings['model_type'].lower()]

        # Create a ParameterSampler instance
        param_sampler = ParameterSampler(param_distributions, n_iter=100, random_state=42)

        best_score = float('inf')
        best_params = None

        for params in param_sampler:
            # Get the model and environment with the current parameters
            model, env = self.get_model_and_env(params)

            # Train the model
            model = self.train_rlm_model(model)

            # Evaluate the model
            true_rewards, predicted_rewards = self.evaluate_rlm_model(model, env)

            # Compute a score based on your evaluation criteria (e.g., mean squared error)
            score = mean_squared_error(true_rewards, predicted_rewards)

            # Update the best score and parameters if necessary
            if score < best_score:
                best_score = score
                best_params = params

        # Print the best parameters
        logger.info(f"Best Parameters: \n{best_params}")

        # Save the best parameters
        self.save_params(best_params)

        model, env = self.get_model_and_env(best_params)
        best_model = self.train_rlm_model(model)
        self.evaluate_rlm_model(best_model, env)
        logger.info(f"Best Model Results: \n{self.results}")
        self.save_model(best_model)

    def objective_optuna(self, trial):
        # Get the parameter space for the selected model type
        param_distributions = self.models_params[self.settings['model_type'].lower()]

        # Create a dictionary to store the sampled parameters
        params = {}
        for param_name, param_values in param_distributions.items():
            if isinstance(param_values, list):
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                if param_values[1] == "log-uniform":
                    params[param_name] = trial.suggest_loguniform(param_name, *param_values[0])
                elif param_values[1] == "uniform":
                    params[param_name] = trial.suggest_float(param_name, *param_values[0])
        
        # Get the model and environment with the current parameters
        model, env = self.get_model_and_env(params)

        # Train the model
        model = self.train_rlm_model(model)

        # Evaluate the model
        true_rewards, predicted_rewards, total_reward = self.evaluate_rlm_model(model, env)

        # Compute a score based on your evaluation criteria (e.g., mean squared error)
        score = mean_squared_error(true_rewards, predicted_rewards)

        return score

    def hp_tuning_optuna(self):
        study = create_study(direction='minimize')
        study.optimize(self.objective_optuna, n_trials=100)
        
        best_params = study.best_params
        best_score = study.best_value

        logger.info(f"Best Parameters: \n{best_params}")
        logger.info(f"Best Score: {best_score}")

        self.save_params(best_params)

        model, env = self.get_model_and_env(best_params)
        best_model = self.train_rlm_model(model)
        self.evaluate_rlm_model(best_model, env)
        logger.info(f"Best Model Results: \n{self.results}")
        self.save_model(best_model)

    def objective_bayesian(self, **params):
        # Convert numerical values back to their original string values
        policy_mapping = {0: 'CnnPolicy', 1: 'MlpPolicy'}
        activation_fn_mapping = {0: 'ReLU', 1: 'Tanh'}
        
        params['policy'] = policy_mapping[int(params['policy'])]
        params['policy_kwargs'] = {
            "net_arch": [int(params['net_arch_x']), int(params['net_arch_y'])],
            "activation_fn": activation_fn_mapping[int(params['activation_fn'])]
        }

        # Get the model and environment with the current parameters
        model, env = self.get_model_and_env(params)

        # Train the model
        model = self.train_rlm_model(model)

        # Evaluate the model
        true_rewards, predicted_rewards, total_reward = self.evaluate_rlm_model(model, env)

        # Compute a score based on your evaluation criteria
        mse = mean_squared_error(true_rewards, predicted_rewards)
        score = total_reward

        return score

    def hp_tuning_bayesian(self):        
        param_distributions = self.models_params[self.settings['model_type'].lower()]
        # print(param_distributions)

        # Convert parameter distributions to the format expected by BayesianOptimization
        pbounds = {}
        policy_mapping = {'CnnPolicy': 0, 'MlpPolicy': 1}
        activation_fn_mapping = {'ReLU': 0, 'Tanh': 1}
        
        for param_name, param_values in param_distributions.items():
            if param_name == 'policy':
                pbounds[param_name] = (0, 1)  # Assign numerical bounds for policy parameter
            elif param_name == 'policy_kwargs':
                pbounds['net_arch_x'] = (32, 256)  # Assign numerical bounds for net_arch x parameter
                pbounds['net_arch_y'] = (32, 512)  # Assign numerical bounds for net_arch y parameter
                pbounds['activation_fn'] = (0, 1)  # Assign numerical bounds for activation_fn parameter
            elif isinstance(param_values, tuple) and len(param_values) == 2:
                pbounds[param_name] = param_values
            elif isinstance(param_values, list) and len(param_values) == 2:
                if isinstance(param_values[0], tuple):
                    pbounds[param_name] = param_values[0]
                else:
                    pbounds[param_name] = (min(param_values), max(param_values))

        print(f'Parameter bounds: \n{pbounds}')
        
        optimizer = BayesianOptimization(
            f=self.objective_bayesian,
            pbounds=pbounds,
            random_state=1,
            allow_duplicate_points=True
        )
        optimizer.maximize(init_points=10, n_iter=90)

        best_params = optimizer.max['params']
        best_score = optimizer.max['target']

        # Convert numerical values back to their original string values
        best_params['policy'] = policy_mapping[int(best_params['policy'])]

        logger.info(f"Best Parameters: \n{best_params}")
        logger.info(f"Best Score: {best_score}")

        self.save_params(best_params)

        model, env = self.get_model_and_env(best_params)
        best_model = self.train_rlm_model(model)
        self.evaluate_rlm_model(best_model, env)
        logger.info(f"Best Model Results: \n{self.results}")
        self.save_model(best_model)



    def save_model(self, model):        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        # Save the agent
        model_filename = f'{self.models_dir}{self.settings["model_type"]}_trade_{self.settings["symbol"]}_{self.settings["binance_timeframe"]}.zip'
        model.save(model_filename)
        logger.info(f"Model saved: {model_filename}")

    def save_params(self, params):        
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        # Save the agent
        params_filename = f'{self.models_dir}best_params_for_{self.settings["model_type"]}_trade_{self.settings["symbol"]}_{self.settings["binance_timeframe"]}.json'
        with open(params_filename, 'w') as f:
            json.dump(params, f)
        logger.info(f"Params saved: {params_filename}")
        
    def run(self):
        best_params = None
        self.get_params()
        self.data = self.fetch_or_read_data()
        if self.settings['use_hp_tuning']:
            best_params = self.hp_tuning()
        else:
            best_params = self.get_existing_best_params()
            self.model, self.env = self.get_model_and_env(best_params)
            self.model = self.train_rlm_model(best_params)
            true_rewards, predicted_rewards = self.evaluate_rlm_model(self.model, self.env)
            
            if best_params is not None:
                self.save_params(best_params)
            self.save_model(self.model) 


        
if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Reinforcement learning models training and tuning.')
    parser.add_argument('--model_type', choices=['A2C', 'DDPG', 'DQN', 'PPO', 'SAC', 'TD3'], 
                        type=str, default='A2C', help='Select model type from A2C, DDPG, DQN, PPO, SAC, TD3')
    parser.add_argument('--use_hp_tuning', type=bool, default=True, help='Use hyperparameter tuning')
    parser.add_argument('--optimizer_type', choices=['random', 'bayesian', 'optuna'], 
                                                type=str, default='bayesian', help='Select optimizer type') 
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
        'optimizer_type': args.optimizer_type,
        'use_cross_validation': args.use_cross_validation,
        'observation_window_length': 48*7,
        'debug': True
        
    }
    rlm_models = RLM_Models(SETTINGS)
    rlm_models.run()