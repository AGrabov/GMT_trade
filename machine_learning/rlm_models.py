import argparse
import datetime as dt
import json
import logging
import os
import traceback

import numpy as np
import pandas as pd
import shap
from bayes_opt import BayesianOptimization
from data_feed_01 import BinanceFuturesData
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
from sklearn.model_selection import ParameterGrid, ParameterSampler
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global settings
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

csv_name = 'GMTUSDT - 30m_(since_2022-03-15).csv' #GMTUSDT - 30m_(since_2022-03-15).csv
# csv_path = f'./data_csvs/{csv_name}'
csv_path = '/root/gmt-bot/data_csvs/GMTUSDT - 30m_(since_2022-03-15).csv'

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Reinforcement Learning Model trading with the optimized "best_params".')
parser.add_argument('--model_type', choices=['DQN', 'DDPG', 'PPO', 'A2C', 'SAC', 'TD3'], 
                    type=str, default='A2C', help='Select model type')
parser.add_argument('--observation_window_length', type=int, default=48, help='Select observation window length of the model')
parser.add_argument('--param_optimization', type=bool, default=True, help='Hyper parameters optimization')
parser.add_argument('--hp_optimizer_type', choices=['optuna', 'random', 'grid', 'bayesian', 'PSO'], 
                        type=str, default='optuna', help='Select hyperparameters optimization type')
parser.add_argument('--use_fetched_data', type=bool, default=False, help='Use fetched data')

def get_data(use_fetched_data, symbol, start_date, end_date, binance_timeframe, csv_path):
    if use_fetched_data:        
        df = BinanceFuturesData.fetch_data(symbol=symbol, startdate=start_date, enddate=end_date, binance_timeframe=binance_timeframe)
        logger.info('Fetching data...')        
        
    else:        
        df = pd.read_csv(csv_path, header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])        
        df = df.dropna()        
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')        
        df = df.set_index('timestamp')
        
    return df

def handle_random_search(df, model_type, observation_window_length, n_iter=10):
    param_dist = {
        'learning_rate': uniform(1e-4, 1e-2),
        'gamma': uniform(0.85, 0.999),
        'ent_coef': uniform(1e-6, 1e-1),
        'n_steps': randint(16, 2048),
        'batch_size': randint(16, 512),
        'buffer_size': randint(1000, 10000),
        'exploration_fraction': uniform(0.1, 1.0),
        'exploration_final_eps': uniform(0.01, 0.5),
        'tau': uniform(0.001, 0.1),
        'layer_1': randint(32, 256),
        'layer_2': randint(32, 256),
        'clip_range': uniform(0.1, 0.4),
        'learning_starts': randint(0, 10000),
        'target_update_interval': randint(1, 1000),
        'gradient_steps': randint(-1, 10)
    }
    random_params = list(ParameterSampler(param_dist, n_iter=n_iter))
    best_score = float('-inf')
    best_params = None
    for params in random_params:
        score = objective_function(df, model_type, observation_window_length, params.values())
        if score > best_score:
            best_score = score
            best_params = params
    logger.info(f'Best Params: {best_params}')
    logger.info(f'Best Score: {best_score}')
    return best_params, best_score


def handle_grid_search(df, model_type, observation_window_length):
    param_grid = {
        'learning_rate': np.linspace(1e-4, 1e-2, 5),
        'gamma': np.linspace(0.85, 0.999, 5),
        'ent_coef': np.linspace(1e-6, 1e-1, 5),
        'n_steps': np.linspace(16, 2048, 5, dtype=int),
        'batch_size': np.linspace(16, 512, 5, dtype=int),
        'buffer_size': np.linspace(1000, 10000, 5, dtype=int),
        'exploration_fraction': np.linspace(0.1, 1.0, 5),
        'exploration_final_eps': np.linspace(0.01, 0.5, 5),
        'tau': np.linspace(0.001, 0.1, 5),
        'layer_1': np.linspace(32, 256, 5, dtype=int),
        'layer_2': np.linspace(32, 256, 5, dtype=int),
        'clip_range': np.linspace(0.1, 0.4, 5),
        'learning_starts': np.linspace(0, 10000, 5, dtype=int),
        'target_update_interval': np.linspace(1, 1000, 5, dtype=int),
        'gradient_steps': np.linspace(-1, 10, 5, dtype=int)
    }
    grid_params = list(ParameterGrid(param_grid))
    best_score = float('-inf')
    best_params = None
    for params in grid_params:
        score = objective_function(df, model_type, observation_window_length, params.values())
        if score > best_score:
            best_score = score
            best_params = params
    logger.info(f'Best Params: {best_params}')
    logger.info(f'Best Score: {best_score}')
    return best_params, best_score

def handle_bayesian_optimization(df, model_type, observation_window_length): 
    objective_with_args = partial(bayesian_objective, df, model_type, observation_window_length)   
    optimizer = BayesianOptimization(
        f=objective_with_args,
        pbounds = {
            'learning_rate': (1e-4, 1e-2),
            'gamma': (0.85, 0.999),
            'ent_coef': (1e-6, 1e-1),
            'n_steps': (16, 2048),
            'batch_size': (16, 512),
            'buffer_size': (1000, 10000),
            'exploration_fraction': (0.1, 1.0),
            'exploration_final_eps': (0.01, 0.5),
            'tau': (0.001, 0.1),
            'layer_1': (32, 256),
            'layer_2': (32, 256),
            'clip_range': (0.1, 0.4),
            'learning_starts': (0, 10000),
            'target_update_interval': (1, 1000),
            'gradient_steps': (-1, 10)
        },
        random_state=1,
    )
    
    # # Store the model and environment in the optimizer
    # optimizer.custom_data = {'model': model, 'env': env}
    optimizer.maximize(init_points=10, n_iter=25)
    

    # if model_type in ['DDPG', 'DQN', 'SAC']:
    #     optimizer.maximize(init_points=2, n_iter=5)
    # else:
    #     optimizer.maximize(init_points=10, n_iter=25)
    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    best_params = optimizer.max['params']
    best_score = optimizer.max['target']
    # model = optimizer.max['model']
    # env = optimizer.max['env']
    logger.info(f"Best Params:{best_params}")
    logger.info(f"Best Score:{best_score}")

    return best_params, best_score #, model, env    

def bayesian_objective(df, model_type, observation_window_length, learning_rate, gamma, ent_coef, n_steps, batch_size, buffer_size,
                           exploration_fraction, exploration_final_eps, tau, layer_1, layer_2,
                           clip_range, learning_starts, target_update_interval, gradient_steps):
        params = (learning_rate, gamma, ent_coef, n_steps, batch_size, buffer_size,
                  exploration_fraction, exploration_final_eps, tau, layer_1, layer_2,
                  clip_range, learning_starts, target_update_interval, gradient_steps)
        
        # score, model, env = objective_function(df, model_type, observation_window_length, params)        
        # return score
        return objective_function(df, model_type, observation_window_length, params)

def handle_pso(df, model_type, observation_window_length):
    lb = [1e-4, 0.85, 1e-6, 16, 16, 1000, 0.1, 0.01, 0.001, 32, 32, 0.1, 0, 1, -1]  # Lower bounds
    ub = [1e-2, 0.999, 1e-1, 2048, 512, 10000, 1.0, 0.5, 0.1, 256, 256, 0.4, 10000, 1000, 10]  # Upper bounds

    print(f'Lower bounds length: {len(lb)}')
    print(f'Upper bounds length: {len(ub)}')

    objective_func = partial(pso_objective, df, model_type, observation_window_length)
    xopt, fopt, model, env = pso(objective_func, lb, ub)
    best_params = {key: val for key, val in zip(model.get_params().keys(), xopt)}
    best_score = fopt
    logger.info(f"Best Params:{best_params}")
    logger.info(f"Best Score:{best_score}")

    return best_params, best_score, model, env

def pso_objective(df, model_type, observation_window_length, x):
    params = x
    return objective_function(df, model_type, observation_window_length, params, minimize=True, debug=False)

def handle_optuna(df, model_type, observation_window_length):
    study = create_study(direction="maximize")
    if model_type in ['DDPG', 'DQN', 'TD3', 'SAC']:                
        study.optimize(lambda trial: optuna_objective(trial, df, model_type, observation_window_length), n_trials=15)
    else:                
        study.optimize(lambda trial: optuna_objective(trial, df, model_type, observation_window_length), n_trials=50)
    best_params = study.best_params
    best_score = study.best_value

    logger.info(f"Best Params:{best_params}")
    logger.info(f"Best Score:{best_score}")

    return best_params, best_score

def optuna_objective(trial: Trial, df, model_type, observation_window_length):
    # Suggest values for hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.85, 0.999, step=0.001)
    ent_coef = trial.suggest_float("ent_coef", 1e-6, 1e-1, log=True)
    n_steps = trial.suggest_int("n_steps", 16, 2048, log=True)
    batch_size = trial.suggest_int("batch_size", 16, 512, log=True)
    buffer_size = trial.suggest_int("buffer_size", 10000, 1000000, log=True)
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 1.0)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.5)
    tau = trial.suggest_float("tau", 0.001, 0.1)
    layer_1 = trial.suggest_int("layer_1", 32, 256)
    layer_2 = trial.suggest_int("layer_2", 32, 256)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    learning_starts = trial.suggest_int("learning_starts", 0, 10000, step=100)
    target_update_interval = trial.suggest_int("target_update_interval", 1, 1000)
    gradient_steps = trial.suggest_int("gradient_steps", -1, 10)

    # Call your existing objective_function with these parameters
    return objective_function(df, model_type, observation_window_length, (learning_rate, gamma, ent_coef, n_steps, batch_size, buffer_size, 
     exploration_fraction, exploration_final_eps, tau, layer_1, layer_2, 
     clip_range, learning_starts, target_update_interval, gradient_steps))

# @jit(nopython=True)
def objective_function(df, model_type, observation_window_length, params, minimize=False, debug=False):
    (learning_rate, gamma, ent_coef, n_steps, batch_size, buffer_size, 
     exploration_fraction, exploration_final_eps, tau, layer_1, layer_2, 
     clip_range, learning_starts, target_update_interval, gradient_steps) = params
    
    policy_kwargs = dict(net_arch=[int(layer_1), int(layer_2)])  # Make sure these are integers

    n_steps = int(n_steps)
    batch_size = int(batch_size)
    buffer_size = int(buffer_size)
    gradient_steps = int(gradient_steps)
    learning_starts = int(learning_starts)
    target_update_interval = int(target_update_interval)    
    

    if model_type == 'A2C':
        action_space = spaces.MultiDiscrete([5])
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length, debug=debug)
        model = A2C('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma, 
                    ent_coef=ent_coef, n_steps=n_steps, policy_kwargs=policy_kwargs,)
        
    elif model_type == 'PPO':
        action_space = spaces.MultiDiscrete([5])
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length, debug=debug)
        model = PPO('MlpPolicy', env, verbose=2, learning_rate=learning_rate, 
                    gamma=gamma, ent_coef=ent_coef, n_steps=n_steps, 
                policy_kwargs=policy_kwargs, clip_range=clip_range, batch_size=batch_size,)
        
    elif model_type == 'DDPG':
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length, debug=debug, ddqn=True)
        model = DDPG('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma, 
                    buffer_size=buffer_size, batch_size=batch_size,
                    tau=tau, policy_kwargs=policy_kwargs, learning_starts=learning_starts, 
                    gradient_steps=gradient_steps) 
        
    elif model_type == 'DQN':
        action_space = spaces.Discrete(5)
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length, debug=debug)
        model = DQN('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma, 
                    batch_size=batch_size, buffer_size=buffer_size, tau=tau, 
                    exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, 
                    policy_kwargs=policy_kwargs, learning_starts=learning_starts, gradient_steps=gradient_steps,
                    target_update_interval=target_update_interval)  
        
    elif model_type == 'TD3':
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length, debug=debug, ddqn=True)
        model = TD3('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma,
                    buffer_size=buffer_size, batch_size=batch_size,
                    tau=tau, policy_kwargs=policy_kwargs, learning_starts=learning_starts,
                    gradient_steps=gradient_steps)
        
    elif model_type == 'SAC':
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length, debug=debug, ddqn=True)
        model = SAC('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma,
                    buffer_size=buffer_size, batch_size=batch_size,
                    tau=tau, policy_kwargs=policy_kwargs, learning_starts=learning_starts,
                    gradient_steps=gradient_steps)

    print("Buffer size:", buffer_size, type(buffer_size))
    print("Observation space shape:", env.observation_space.shape, type(env.observation_space.shape))

    if model_type in ['DDPG', 'DQN', 'SAC']:        
        model.learn(total_timesteps=len(df), progress_bar=True)        
    else:        
        model.learn(total_timesteps=len(df)*2, progress_bar=True)
        
    # Evaluate the trained model on a validation set and return the performance    
    obs = env.reset()
    total_reward = 0
    for _ in range(5000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()

    logger.info(f"Total reward: {total_reward}")    

    if minimize:
        return -total_reward #, model, env
    else:
        return total_reward #, model, env

def save_best_params(best_params, model_type):
    # Create directory
    models_directory = f'./models/RLM/{model_type}/'        
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    # Save the best parameters to a JSON file
    with open(f"./models/RLM/{model_type}/best_params_for_model_{model_type}.json", "w") as f:
        json.dump(best_params, f)

def train_model(df, model_type, observation_window_length=48):

    if model_type == 'A2C':
        action_space = spaces.MultiDiscrete([5])
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length)
        model = A2C('MlpPolicy', env, verbose=2, 
                    learning_rate=0.0018015401846872665, 
                    gamma=0.941, 
                    ent_coef=0.07470059112267045,
                    n_steps=1575,
                    policy_kwargs = dict(net_arch=[231, 205]),
                    )
        
        
    elif model_type == 'PPO':
        action_space = spaces.MultiDiscrete([5])
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length)
        model = PPO('MlpPolicy', env, verbose=2,                          
                    learning_rate=0.0014116534240643653, 
                    gamma=0.99,
                    ent_coef= 0.00024414110611047546,
                    n_steps= 1278,
                    batch_size=36,                         
                    clip_range=0.3552686778007785,
                    policy_kwargs=dict(net_arch=[103, 181]),                        
                    )
        
    elif model_type == 'DDPG':
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length, ddqn=True)
        model = DDPG('MlpPolicy', env, verbose=2, 
                    # {'batch_size': 348.5518850484875, 
                    # 'buffer_size': 4755.743221304143, 
                    # 'clip_range': 0.2676069485337255, 
                    # 'ent_coef': 0.014039553472584782, 
                    # 'exploration_final_eps': 0.1070697296515906, 
                    # 'exploration_fraction': 0.820670111807983, 
                    # 'gamma': 0.9942709747821903, 
                    # 'gradient_steps': 2.4476659597516712, 
                    # 'layer_1': 187.08026590992637, 'layer_2': 228.31117011431257, 
                    # 'learning_rate': 0.00895660596868809, 
                    # 'learning_starts': 850.4421136977791, 
                    # 'n_steps': 95.35931952921696, 
                    # 'target_update_interval': 170.66058914500434, 
                    # 'tau': 0.0879361078395119}
                        )
        
    elif model_type == 'DQN':
        action_space = spaces.Discrete(5)
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length)
        model = DQN('MlpPolicy', env, verbose=2, 
                        learning_rate=0.0007599557436260945, 
                        gamma=0.9139999999999999, 
                    #  'ent_coef': 0.0004594170353020396, 
                    #  'n_steps': 1686, 
                        batch_size=46, 
                        buffer_size=230044, 
                        exploration_fraction=0.8867039219828994, 
                        exploration_final_eps=0.21267036720628935, 
                        tau=0.0724048665314716, 
                        policy_kwargs=dict(net_arch=[33, 89]),
                    #  'clip_range': 0.22281650028737338, 
                        learning_starts=6800, 
                        target_update_interval=190, 
                        gradient_steps=25,
                        optimize_memory_usage=True             
                    )
                    # {'batch_size': 337.794394443593, 
                    # 'buffer_size': 9975.458866738589, 
                    # 'clip_range': 0.2696533217023292, 
                    # 'ent_coef': 0.00566100330273799, 
                    # 'exploration_final_eps': 0.17358097043422088, 
                    # 'exploration_fraction': 0.12342223717424922, 
                    # 'gamma': 0.8940600932255093, 
                    # 'gradient_steps': 5.605142499522665, 
                    # 'layer_1': 67.24103479077662, 'layer_2': 193.5651676765623, 
                    # 'learning_rate': 0.008967889626902117, 
                    # 'learning_starts': 7926.34198556009, 
                    # 'n_steps': 1985.1404390779905, 
                    # 'target_update_interval': 757.2946025379913, 
                    # 'tau': 0.06467849089912071}
        
    elif model_type == 'TD3':
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length, ddqn=True)
        model = TD3('MlpPolicy', env, verbose=2, 
                    batch_size = 462, 
                    buffer_size = 1363,                    
                    gamma = 0.919293480188914, 
                    gradient_steps = 1, 
                    policy_kwargs = dict(net_arch=[126, 138]),
                    learning_rate = 0.0006390149684210271,
                    learning_starts = 2180.994810763447,                    
                    tau = 0.004123313849695283
                    )        
    elif model_type == 'SAC':
        action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        env = TradeEnv(df, action_space=action_space, live=False, observation_window_length=observation_window_length, ddqn=True)
        model = SAC('MlpPolicy', env, verbose=2,
                    # buffer_size = 10000,
                    learning_rate = 0.00027319504122962784, 
                    gamma = 0.988,                    
                    batch_size = 103, 
                    buffer_size = 290595,                     
                    tau = 0.09014735733895488, 
                    policy_kwargs = dict(net_arch=[200, 143]),
                    learning_starts = 3000, 
                    target_update_interval = 248, 
                    gradient_steps = 1
                    )

    # Train the agent
    print(f"Learning {model_type} model...")  
    if model_type in ['DDPG', 'DQN', 'SAC']:            
        model.learn(total_timesteps=len(df), progress_bar=True)       
    
    else:            
        model.learn(total_timesteps=len(df)*5, progress_bar=True) 
    logger.info(f"Learning {model_type} model finished.")

    return model, env

def evaluate_model(model, env, model_type):
    # Create directory
    models_directory = f'./models/RLM/{model_type}/'        
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)
    
    # After training the model, add the following code to compute evaluation metrics:
    obs = env.reset()
    true_rewards = []
    predicted_rewards = []
    for _ in range(1000):
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

    # For SHAP values 
    if model_type == 'DQN':
        explainer = shap.Explainer(model.q_net)
    elif model_type in ['A2C', 'PPO']:
        explainer = shap.Explainer(model.policy.vf_net)
    elif model_type in ['DDPG', 'TD3', 'SAC']:
        explainer = shap.Explainer(model.critic)

    shap_values = explainer.shap_values(obs)
    shap.summary_plot(shap_values, obs)

    
    
def save_model(model, model_type):
    # Create directory
    models_directory = f'./models/RLM/{model_type}/'        
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)
    # Save the agent
    model_filename = f'{models_directory}{model_type}_trade_{symbol}_{binance_timeframe}'
    model.save(model_filename)
    logger.info(f"Model saved: {model_filename}")

def main():
    args = parser.parse_args()
    use_fetched_data = args.use_fetched_data
    param_optimization = args.param_optimization
    model_type = args.model_type
    hp_optimizer_type = args.hp_optimizer_type
    observation_window_length = args.observation_window_length

    model, env, best_params, best_score = None, None, None, None

    df = get_data(use_fetched_data, symbol, start_date, end_date, binance_timeframe, csv_path)
    if df is None:
        logger.error("Dataframe is empty. Exiting.")
        return
    # Validate command-line arguments
    if hp_optimizer_type not in ['optuna', 'random', 'grid', 'bayesian', 'PSO']:
        logger.error("Invalid optimizer type")
        return
    
    if param_optimization:
        if hp_optimizer_type == 'optuna':
            try:
                best_params, best_score = handle_optuna(df, model_type, observation_window_length)
            except Exception as e:
                logger.error(f"Error in Optuna: {e}")
                logger.error(traceback.format_exc()) 

        elif hp_optimizer_type == 'bayesian':
            try:
                # best_params, best_score, model, env = handle_bayesian_optimization(df, model_type, observation_window_length)
                best_params, best_score = handle_bayesian_optimization(df, model_type, observation_window_length)

            except Exception as e:
                logger.error(f"Error in bayesian: {e}")
                logger.error(traceback.format_exc())  
        elif hp_optimizer_type == 'PSO':
            try:
                best_params, best_score = handle_pso(df, model_type, observation_window_length)
            except Exception as e:
                logger.error(f"Error in PSO: {e}")
                logger.error(traceback.format_exc())  
        elif hp_optimizer_type == 'random':
            try:
                best_params, best_score = handle_random_search(df, model_type, observation_window_length)
            except Exception as e:
                logger.error(f"Error in random search: {e}")
                logger.error(traceback.format_exc())  
        elif hp_optimizer_type == 'grid':
            try:
                best_params, best_score = handle_grid_search(df, model_type, observation_window_length)
            except Exception as e:
                logger.error(f"Error in grid search: {e}")
                logger.error(traceback.format_exc()) 
        try:
            
            if best_params:
                save_best_params(best_params, model_type)
        except Exception as e:
            logger.error(f"Error during saving best params: {e}")
            logger.error(traceback.format_exc()) 
        
    else:
        try:
            model, env = train_model(df, model_type, observation_window_length)
            evaluate_model(model, env, model_type)
            save_model(model, model_type)
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            logger.error(traceback.format_exc())  
    
if __name__ == "__main__":
    main()
