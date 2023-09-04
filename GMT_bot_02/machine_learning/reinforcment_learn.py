
from stable_baselines3 import A2C, PPO, DDPG, DQN, TD3, SAC
from optuna import Trial, create_study
from gym_trading_env_03 import TradingEnv
from ddpg_env_03 import DDPG_Env
from dqn_env_03 import DQN_Env
import pandas as pd
import datetime as dt
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error
import os
from bayes_opt import BayesianOptimization
from pyswarm import pso
import traceback

import argparse
from data_feed_01 import BinanceFuturesData
from hp_search import HyperparameterSearch

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
parser.add_argument('--model_type', choices=['DQN', 'DDPG', 'PPO', 'A2C', 'SAC', 'TD3'], type=str, default='SAC', help='Select model type')
parser.add_argument('--param_optimization', type=bool, default=True, help='Hyper parameters optimization')
parser.add_argument('--hp_optimizer_type', choices=['optuna', 'random', 'grid', 'bayesian', 'PSO'], type=str, default='bayesian', help='Select data scaler type')
parser.add_argument('--use_fetched_data', type=bool, default=False, help='Use fetched data')

args = parser.parse_args()

use_fetched_data = args.use_fetched_data
param_optimization = args.param_optimization
model_type = args.model_type
hp_optimizer_type = args.hp_optimizer_type

def bayesian_objective(df, model_type, pbounds):
    return objective_function(df, model_type, params=pbounds, minimize=False)

def pso_objective(df, model_type):
    return objective_function(df, model_type, minimize=True)

def objective_function(df, model_type, params, minimize=False):
    # Unpack parameters
    (learning_rate, gamma, ent_coef, n_steps, batch_size, buffer_size, 
     exploration_fraction, exploration_final_eps, tau, layer_1, layer_2, 
     clip_range, learning_starts, target_update_interval, gradient_steps) = params
    
    policy_kwargs = dict(net_arch=[layer_1, layer_2])
    
    if model_type == 'A2C':
        env = TradingEnv(df, live=False, observation_window_length=48)
        model = A2C('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma, 
                    ent_coef=ent_coef, n_steps=n_steps, policy_kwargs=policy_kwargs,)
        
    elif model_type == 'PPO':
        env = TradingEnv(df, live=False, observation_window_length=48)
        model = PPO('MlpPolicy', env, verbose=2, learning_rate=learning_rate, 
                    gamma=gamma, ent_coef=ent_coef, n_steps=n_steps, 
                policy_kwargs=policy_kwargs, clip_range=clip_range, batch_size=batch_size,)
        
    elif model_type == 'DDPG':
        env = DDPG_Env(df, live=False, observation_window_length=48)
        model = DDPG('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma, 
                    buffer_size=buffer_size, batch_size=batch_size,
                    tau=tau, policy_kwargs=policy_kwargs, learning_starts=learning_starts, 
                    gradient_steps=gradient_steps, optimize_memory_usage=True) 
        
    elif model_type == 'DQN':
        env = DQN_Env(df, live=False, observation_window_length=48)
        model = DQN('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma, 
                    batch_size=batch_size, buffer_size=buffer_size, tau=tau, 
                    exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, 
                    policy_kwargs=policy_kwargs, learning_starts=learning_starts, gradient_steps=gradient_steps,
                    target_update_interval=target_update_interval, optimize_memory_usage=True)  
        
    elif model_type == 'TD3':
        env = DDPG_Env(df, live=False, observation_window_length=48)
        model = TD3('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma,
                    buffer_size=buffer_size, batch_size=batch_size,
                    tau=tau, policy_kwargs=policy_kwargs, learning_starts=learning_starts,
                    gradient_steps=gradient_steps)
        
    elif model_type == 'SAC':
        env = DDPG_Env(df, live=False, observation_window_length=48)
        model = SAC('MlpPolicy', env, verbose=2, learning_rate=learning_rate, gamma=gamma,
                    buffer_size=buffer_size, batch_size=batch_size,
                    tau=tau, policy_kwargs=policy_kwargs, learning_starts=learning_starts,
                    gradient_steps=gradient_steps)
        
    if model_type in ['DDPG', 'TD3', 'DQN', 'SAC']:
        try:
            model.learn(total_timesteps=len(df), progress_bar=True)        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            logger.error(traceback.format_exc())  
    else:
        try:
            model.learn(total_timesteps=len(df)*5, progress_bar=True)        
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            logger.error(traceback.format_exc())  
        
    # Evaluate the trained model on a validation set and return the performance    
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()

    logger.info(f"Total reward: {total_reward}")    

    if minimize:
        return -total_reward
    else:
        return total_reward

def optuna_objective(trial: Trial, df, model_type):
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
    gradient_steps = trial.suggest_int("gradient_steps", 1, 100)

    # Call your existing objective_function with these parameters
    return objective_function(df, model_type, (learning_rate, gamma, ent_coef, n_steps, batch_size, buffer_size, 
     exploration_fraction, exploration_final_eps, tau, layer_1, layer_2, 
     clip_range, learning_starts, target_update_interval, gradient_steps))


if use_fetched_data:
    # Fetch the data for the specified symbol and time range
    try:
        df = BinanceFuturesData.fetch_data(symbol=symbol, startdate=start_date, enddate=end_date, binance_timeframe=binance_timeframe)
        print('Fteching data...')
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        logger.error(traceback.format_exc())  

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
        logger.error(f"Error reading CSV file: {e}")
        logger.error(traceback.format_exc())  

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
            study = create_study(direction="maximize")
            if model_type in ['DDPG', 'DQN', 'TD3', 'SAC']:                
                study.optimize(lambda trial: optuna_objective(trial, df, model_type), n_trials=10)
            else:                
                study.optimize(lambda trial: optuna_objective(trial, df, model_type), n_trials=100)
            logger.info(study.best_params)

            # Create directory
            models_directory = f'./models/{model_type}/'        
            if not os.path.exists(models_directory):
                os.makedirs(models_directory)

            # Save the best parameters to a JSON file
            with open(f"./models/{model_type}/best_params_for_model_{model_type}.json", "w") as f:
                json.dump(study.best_params, f)

        except Exception as e:
            logger.error(f"Error during model training or saving: {e}")
            logger.error(traceback.format_exc())  

    elif args.hp_optimizer_type == 'bayesian':
        try:
            optimizer = BayesianOptimization(
                f=bayesian_objective(df, model_type,
                pbounds={
                'learning_rate' : (1e-4, 1e-2),
                'gamma': (0.85, 0.999),
                'ent_coef' : (1e-6, 1e-1),
                'n_steps' : (16, 2048),
                'batch_size' : (16, 512),
                'buffer_size' : (10000, 1000000),
                'exploration_fraction' : (0.1, 1.0),
                'exploration_final_eps' : (0.01, 0.5),
                'tau' : (0.001, 0.1),
                'layer_1' : (32, 256),
                'layer_2' : (32, 256),            
                'clip_range' : (0.1, 0.4),
                'learning_starts' : (0, 10000),
                'target_update_interval' : (1, 1000),
                'gradient_steps' : (1, 100)
                }),
                random_state=1,
            )
            if model_type in ['DDPG', 'DQN', 'TD3', 'SAC']:
                optimizer.maximize(init_points=5, n_iter=20)
            else:
                optimizer.maximize(init_points=10, n_iter=50)
        except Exception as e:
            logger.error(f"Error during Bayesian tunning: {e}")
            logger.error(traceback.format_exc())

    elif args.hp_optimizer_type == 'PSO':
        try:
            lb = [1e-4, 0.85, 1e-6, 16, 16, 10000, 0.1, 0.01, 0.001, 32, 32, 0.1, 0, 1, 1]  # Lower bounds
            ub = [1e-2, 0.999, 1e-1, 2048, 512, 1000000, 0.5, 0.5, 0.1, 256, 256, 0.4, 10000, 1000, 100]  # Upper bounds
            xopt, fopt = pso(pso_objective(df, model_type), lb, ub)
        except Exception as e:
            logger.error(f"Error during PSO tunning: {e}")
            logger.error(traceback.format_exc())
    else:
        print("Invalid optimizer type")
else:
    try:
        if model_type == 'A2C':
            env = TradingEnv(df, live=False, observation_window_length=30)
            model = A2C('MlpPolicy', env, verbose=2, 
                        learning_rate=0.0018015401846872665, 
                        gamma=0.941, 
                        ent_coef=0.07470059112267045,
                        n_steps=1575,
                        policy_kwargs = dict(net_arch=[231, 205]),
                        )           
            
        elif model_type == 'PPO':
            env = TradingEnv(df, live=False, observation_window_length=30)
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
            env = DDPG_Env(df, live=False, observation_window_length=30)
            model = DDPG('MlpPolicy', env, verbose=2,                         
                         )
            
        elif model_type == 'DQN':
            env = DQN_Env(df, live=False, observation_window_length=30)
            model = DQN('MlpPolicy', env, verbose=2, 
                         learning_rate=0.0007599557436260945, 
                         gamma=0.9139999999999999,                        
                         batch_size=46, 
                         buffer_size=230044, 
                         exploration_fraction=0.8867039219828994, 
                         exploration_final_eps=0.21267036720628935, 
                         tau=0.0724048665314716, 
                         policy_kwargs=dict(net_arch=[33, 89]),                        
                         learning_starts=6800, 
                         target_update_interval=190, 
                         gradient_steps=25,
                         optimize_memory_usage=True             
                        )
            
        elif model_type == 'TD3':
            env = DDPG_Env(df, live=False, observation_window_length=30)
            model = TD3('MlpPolicy', env, verbose=2,                         
                        )
            
        elif model_type == 'SAC':
            env = DDPG_Env(df, live=False, observation_window_length=30)
            model = SAC('MlpPolicy', env, verbose=2,  
                        )

        # Train the agent
        print(f"Learning {model_type} model...")  
        model.learn(total_timesteps=len(df)*5, progress_bar=True)
        logger.info(f"Learning {model_type} model finished.")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        logger.error(traceback.format_exc())

    try:
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

    except Exception as e:
        logger.error(f"Error during SHAP evaluation: {e}")
        logger.error(traceback.format_exc())

    try:
        # Create directory
        models_directory = f'./models/{model_type}/'        
        if not os.path.exists(models_directory):
            os.makedirs(models_directory)
        # Save the agent
        model_filename = f'{models_directory}{model_type}_trade_{symbol}_{binance_timeframe}'
        model.save(model_filename)
        logger.info(f"Model saved: {model_filename}")

    except Exception as e:
        logger.error(f"Error during model saving: {e}")
        logger.error(traceback.format_exc())  
