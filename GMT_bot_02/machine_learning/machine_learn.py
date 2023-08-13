import gym
from stable_baselines3 import A2C
from gym_env_01 import TradingEnv
import pandas as pd
import datetime as dt
from data_feed_01 import BinanceFuturesData

# Settings
target_coin = 'GMT'
base_currency = 'USDT' # 'BUSD' # 
symbol = target_coin + base_currency
dataname = (f'{target_coin}/{base_currency}')
start_date = dt.datetime.strptime("2022-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
end_date = dt.datetime.strptime("2023-07-01 00:00:00", "%Y-%m-%d %H:%M:%S")
# today = f"{dt.date.today()} 00:00:00"
# print(today)
# end_date = dt.datetime.strptime(today, "%Y-%m-%d %H:%M:%S")
timeframe =  'Minutes' # 'Hours' #  
compression = 30
binance_timeframe = f'{compression}m'

# Fetch the data for the specified symbol and time range
df = BinanceFuturesData.fetch_data(symbol=symbol, startdate=start_date, enddate=end_date, binance_timeframe=binance_timeframe)

# Print the dataframe's top few rows
print('Fteching data...')
print(df.head(3))
print()
print(f'Number of timesteps in the fteched data: {len(df)}')

# Create an instance of the custom trading environment
env = TradingEnv(df, live=False)

# Initialize agent with a smaller learning rate
model = A2C('MlpPolicy', env, verbose=1, learning_rate=0.01)  # Adjusted learning rate

# Train agent
model.learn(total_timesteps=len(df)*20, progress_bar=True)

# Save the agent
model.save(f"./models/a2c_trading_{symbol}_{binance_timeframe}")

# To use the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        obs = env.reset()