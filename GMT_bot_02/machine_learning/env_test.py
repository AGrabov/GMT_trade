# test_environment.py

from gym_env_01 import TradingEnv
import numpy as np
import datetime as dt
from data_feed_01 import BinanceFuturesData

# Settings
target_coin = 'GMT'
base_currency = 'USDT' # 'BUSD' # 
symbol = target_coin + base_currency
dataname = (f'{target_coin}/{base_currency}')
start_date = dt.datetime.strptime("2023-06-01 00:00:00", "%Y-%m-%d %H:%M:%S")
end_date = dt.datetime.strptime("2023-07-01 00:00:00", "%Y-%m-%d %H:%M:%S")
# today = f"{dt.date.today()} 00:00:00"
# print(today)
# end_date = dt.datetime.strptime(today, "%Y-%m-%d %H:%M:%S")
timeframe =  'Minutes' # 'Hours' #  
compression = 30
binance_timeframe = f'{compression}m'

def test_random_policy(env):
    obs = env.reset()
    done = False
    step_count = 0

    while not done:
        # Generate a random action based on the action space
        action = [env.action_space.sample()]
        
        obs, reward, done, info = env.step(action)
        step_count += 1

        # Log the results
        print(f"Step {step_count}: Reward: {reward}, Info: {info}")

    print("Testing finished.")       

def test_deterministic_policy(env, num_episodes=10):
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = 0  # Always hold, for example
            obs, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

if __name__ == "__main__":
    # Fetch the data for the specified symbol and time range
    df = BinanceFuturesData.fetch_data(symbol=symbol, startdate=start_date, enddate=end_date, binance_timeframe=binance_timeframe)
    env = TradingEnv(initial_data=df)  # Initialize your environment
    print("Testing Random Policy:")
    test_random_policy(env)
    print("\nTesting Deterministic Policy:")
    test_deterministic_policy(env)
