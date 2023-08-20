import gym
from stable_baselines3 import A2C, PPO, DDPG, DQN
from stable_baselines3 import PPO
from optuna import Trial, create_study
# from gym_env_new_02 import TradingEnv
from gym_trading_env_03 import TradingEnv
import pandas as pd
import datetime as dt
from data_feed_01 import BinanceFuturesData
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Settings
target_coin = 'GMT'
base_currency = 'USDT' # 'BUSD' # 
binance_timeframe = '30m'
symbol = target_coin + base_currency
dataname = (f'{target_coin}/{base_currency}')
start_date = dt.datetime.strptime("2023-05-01 00:00:00", "%Y-%m-%d %H:%M:%S")
end_date = dt.datetime.strptime("2023-08-01 00:00:00", "%Y-%m-%d %H:%M:%S")
use_fteched_data = False
param_optimization = True

# This function optimizes hyperparameters using Optuna
def objective(trial: Trial):
    global df  # Declare df as a global variable

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.85, 0.9999, step=0.01)
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

    
    env = TradingEnv(df, live=False, observation_window_length=30)
    # model1 = A2C('MlpPolicy', env, verbose=1, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef, n_steps=n_steps, policy_kwargs=policy_kwargs,)
    model2 = PPO('MlpPolicy', env, verbose=1, learning_rate=learning_rate, 
                 gamma=gamma, ent_coef=ent_coef, n_steps=n_steps, 
                policy_kwargs=policy_kwargs, clip_range=clip_range)
    
    model3 = DDPG('MlpPolicy', env, verbose=1, learning_rate=learning_rate, gamma=gamma, 
                  ent_coef=ent_coef, batch_size=batch_size, buffer_size=buffer_size,
                  tau=tau, policy_kwargs=policy_kwargs, learning_starts=learning_starts, 
                  target_update_interval=target_update_interval)
    model4 = DQN('MlpPolicy', env, verbose=1, learning_rate=learning_rate, gamma=gamma, 
                 ent_coef=ent_coef, batch_size=batch_size, buffer_size=buffer_size,
             exploration_fraction=exploration_fraction, exploration_final_eps=exploration_final_eps, 
             policy_kwargs=policy_kwargs, learning_starts=learning_starts)

    
    # You might want to use a smaller number of timesteps for faster trials
    
    try:
        model2.learn(total_timesteps=len(df)*5, progress_bar=True)
        # model1.learn(total_timesteps=len(df)*5, progress_bar=True)
    except Exception as e:
        logger.error(f"Error during model training: {e}")

    
    
    # Evaluate the trained model on a validation set and return the performance
    # For simplicity, we'll use the same env for validation
    obs = env.reset()
    total_reward = 0
    for _ in range(1000):
        action, _states = model2.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            obs = env.reset()

    logger.info(f"Total reward: {total_reward}")
    
    return total_reward


if use_fteched_data:
    # Fetch the data for the specified symbol and time range
    try:
        df = BinanceFuturesData.fetch_data(symbol=symbol, startdate=start_date, enddate=end_date, binance_timeframe=binance_timeframe)
        print('Fteching data...')
    except Exception as e:
        print(f"Error fetching data: {e}")

else:
    csv_name = 'GMTUSDT_30m_data.csv' #GMTUSDT - 30m_(since_2022-03-15).csv
    csv_path = f'data_csvs/GMTUSDT_30m_data.csv'
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
    except Exception as e:
        print(f"Error during model training or saving: {e}")

else:
    try:
        # Create an instance of the custom trading environment
        env = TradingEnv(initial_data=df, live=False, debug=False)

        # Initialize agent with a smaller learning rate
        model1 = A2C('MlpPolicy', env, verbose=1, learning_rate=0.006835060666822352, gamma=0.9772759055292447, ent_coef=0.0022860292100790104)  # Adjusted learning rate
        model2 = PPO('MlpPolicy', env, verbose=1)
        model3 = DDPG('MlpPolicy', env, verbose=1)
        model4 = DQN('MlpPolicy', env, verbose=1)

        # Train agent
        model1.learn(total_timesteps=len(df)*10, progress_bar=True)
        logger.info(f"Learning A2C finished.")
        model2.learn(total_timesteps=len(df)*10, progress_bar=True)
        logger.info(f"Learning PPO finished.")
        model3.learn(total_timesteps=len(df)*10, progress_bar=True)
        logger.info(f"Learning DDPG finished.")
        model4.learn(total_timesteps=len(df)*10, progress_bar=True)
        logger.info(f"Learning DQN finished.")

        # Save the agent
        model1.save(f"./models/a2c_trade_{symbol}_{binance_timeframe}")
    except Exception as e:
        print(f"Error during model training or saving: {e}")
