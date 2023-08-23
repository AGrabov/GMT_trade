import gym
from stable_baselines3 import A2C, PPO, DDPG, DQN
from stable_baselines3 import PPO
from optuna import Trial, create_study
from gym_trading_env_03 import TradingEnv
from ddpg_env_03 import DDPG_Env
from dqn_env_03 import DQN_Env
import pandas as pd
import datetime as dt
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
parser.add_argument('--model_type', choices=['DQN', 'DDPG', 'PPO', 'A2C'], type=str, default='A2C', help='Select model type')
parser.add_argument('--param_optimization', type=bool, default=True, help='Hyper parameters optimization')
parser.add_argument('--use_fetched_data', type=bool, default=False, help='Use fetched data')


args = parser.parse_args()

if args.use_fetched_data is None:
    use_fetched_data = False 
else:
    use_fetched_data = args.use_fetched_data

if args.param_optimization is None:
    param_optimization = True
else:
    param_optimization = args.param_optimization

if args.model_type is None:
    model_type = 'DQN' # 'DDPG' # 'PPO' # 'A2C'
else:
    model_type = args.model_type

# This function optimizes hyperparameters using Optuna
def objective(trial: Trial):
    global df  # Declare df as a global variable

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
    policy_kwargs = dict(net_arch=[layer_1, layer_2])
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    learning_starts = trial.suggest_int("learning_starts", 0, 10000, step=100)
    target_update_interval = trial.suggest_int("target_update_interval", 1, 1000)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 100)

    
    if model_type == 'A2C':
        env = TradingEnv(df, live=False, observation_window_length=30)
        model = A2C('MlpPolicy', env, verbose=1, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, n_steps=n_steps, policy_kwargs=policy_kwargs,)
    elif model_type == 'PPO':
        env = TradingEnv(df, live=False, observation_window_length=30)
        model = PPO('MlpPolicy', env, verbose=1, learning_rate=learning_rate, 
                    gamma=gamma, ent_coef=ent_coef, n_steps=n_steps, 
                policy_kwargs=policy_kwargs, clip_range=clip_range, batch_size=batch_size,)
    elif model_type == 'DDPG':
        env = DDPG_Env(df, live=False, observation_window_length=30)
        model = DDPG('MlpPolicy', env, verbose=1, learning_rate=learning_rate, gamma=gamma, 
                    buffer_size=buffer_size, batch_size=batch_size,
                    tau=tau, policy_kwargs=policy_kwargs, learning_starts=learning_starts, 
                    gradient_steps=gradient_steps, )
    elif model_type == 'DQN':
        env = DQN_Env(df, live=False, observation_window_length=30)
        model = DQN('MlpPolicy', env, verbose=1, learning_rate=learning_rate, gamma=gamma, 
                    batch_size=batch_size, buffer_size=buffer_size, tau=tau, 
                    exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, 
                    policy_kwargs=policy_kwargs, learning_starts=learning_starts, gradient_steps=gradient_steps,
                    target_update_interval=target_update_interval)  
       
    try:
        model.learn(total_timesteps=len(df), progress_bar=True)        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
    
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
    
    return total_reward


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
    symbol = 'GMTUSDT'
    binance_timeframe = '30m'

    # Read the CSV file using more efficient parameters
    df = pd.read_csv('GMTUSDT_30m_data.csv', header=None, names=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

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
    try:
        study = create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        logger.info(study.best_params)

        # Save the best parameters to a JSON file
        with open(f"./models/best_params_for_model_{model_type}.json", "w") as f:
            json.dump(study.best_params, f)

    except Exception as e:
        print(f"Error during model training or saving: {e}")

else:
    try:
        if model_type == 'A2C':
            env = TradingEnv(df, live=False, observation_window_length=30)
            model = A2C('MlpPolicy', env, verbose=1, 
                        learning_rate=0.006835060666822352, 
                        gamma=0.9772759055292447, 
                        ent_coef=0.0022860292100790104
                        )
        elif model_type == 'PPO':
            env = TradingEnv(df, live=False, observation_window_length=30)
            model = PPO('MlpPolicy', env, verbose=1,                          
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
            model = DDPG('MlpPolicy', env, verbose=1, 
                        #  learning_rate=learning_rate, 
                        #  gamma=gamma, 
                        #  buffer_size=buffer_size, 
                        #  batch_size=batch_size,
                        #  tau=tau, 
                        #  policy_kwargs=policy_kwargs, 
                        #  learning_starts=learning_starts, 
                        #  gradient_steps=gradient_steps, 
                         )
        elif model_type == 'DQN':
            env = DQN_Env(df, live=False, observation_window_length=30)
            model = DQN('MlpPolicy', env, verbose=1, 
                                      
                        )  
        
        

        # Train the agent  
        model.learn(total_timesteps=len(df)*10, progress_bar=True)
        logger.info(f"Learning {model_type} model finished.")        

        # Save the agent
        model.save(f"./models/{model_type}_trade_{symbol}_{binance_timeframe}")
    except Exception as e:
        print(f"Error during model training or saving: {e}")


# Trial 22 finished with value: 67104.53740322535 and parameters: 
# {'learning_rate': 0.003202748361563921, 
# 'gamma': 0.97, 'ent_coef': 0.0012194495892457902, 
# 'n_steps': 1544, 'batch_size': 52, 'buffer_size': 204540,
#  'exploration_fraction': 0.4158894343773627, 
# 'exploration_final_eps': 0.06931251549523125, 
# 'tau': 0.011338044988825732, 
# 'layer_1': 158, 'layer_2': 105, 
# 'clip_range': 0.3663510223822747, 
# 'learning_starts': 8500, 
# 'target_update_interval': 644, 
# 'gradient_steps': 88}. Best is trial 22 with value: 67104.53