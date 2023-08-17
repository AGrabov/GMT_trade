import gym
from stable_baselines3 import A2C
from optuna import Trial, create_study
# from gym_env_new_02 import TradingEnv
from gym_trading_env_03 import TradingEnv
import pandas as pd
import datetime as dt
from data_feed_01 import BinanceFuturesData

# Settings
target_coin = 'GMT'
base_currency = 'USDT' # 'BUSD' # 
symbol = target_coin + base_currency
dataname = (f'{target_coin}/{base_currency}')
start_date = dt.datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
end_date = dt.datetime.strptime("2023-08-01 00:00:00", "%Y-%m-%d %H:%M:%S")

timeframe =  'Minutes' # 'Hours' #  
compression = 30

binance_timeframe = f'{compression}m'

# def objective(trial: Trial):
#     global df  # Declare df as a global variable

#     learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
#     gamma = trial.suggest_float("gamma", 0.9, 0.9999)
#     ent_coef = trial.suggest_float("ent_coef", 1e-5, 1e-1, log=True)
    
#     env = TradingEnv(df, live=False)
#     model = A2C('MlpPolicy', env, verbose=1, learning_rate=learning_rate, gamma=gamma, ent_coef=ent_coef)
    
#     # You might want to use a smaller number of timesteps for faster trials
#     model.learn(total_timesteps=len(df)*5, progress_bar=True)
    
#     # Evaluate the trained model on a validation set and return the performance
#     # For simplicity, we'll use the same env for validation
#     obs = env.reset()
#     total_reward = 0
#     for _ in range(1000):
#         action, _states = model.predict(obs)
#         obs, reward, done, info = env.step(action)
#         total_reward += reward
#         if done:
#             obs = env.reset()
    
#     return total_reward



# Fetch the data for the specified symbol and time range
df = BinanceFuturesData.fetch_data(symbol=symbol, startdate=start_date, enddate=end_date, binance_timeframe=binance_timeframe)

# Check if the dataframe is valid and not empty
if df is None or df.empty:
    raise ValueError("Failed to fetch data or the data is empty.")

# # Create a new DataFrame without empty rows
# df = df.dropna()
    
# Print the dataframe's top few rows
print('Fteching data...')
print(df.head(3))
print(df.tail(2))
print()
print(f'Number of timesteps in the fteched data: {len(df)}')

# study = create_study(direction="maximize")
# study.optimize(objective, n_trials=100)

try:
    # Create an instance of the custom trading environment
    env = TradingEnv(initial_data=df, live=False, debug=True)

    # Initialize agent with a smaller learning rate
    model = A2C('MlpPolicy', env, verbose=1, learning_rate=0.0005, )  # Adjusted learning rate

    # Train agent
    model.learn(total_timesteps=len(df)*10, progress_bar=True)

    # Save the agent
    model.save(f"./models/a2c_trading_{symbol}_{binance_timeframe}")
except Exception as e:
    print(f"Error during model training or saving: {e}")