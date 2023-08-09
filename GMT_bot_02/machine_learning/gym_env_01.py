import logging
import numpy as np
import pandas as pd
import gym
from gym import Env, spaces, utils
from gym.utils import seeding
from collections import deque
import time

logger = logging.getLogger(__file__)

class TradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_data, max_buffer_size=1000, transaction_cost=0.0005) -> None:
        super(TradingEnv, self).__init__()

        # Ensure df is a DataFrame
        if not isinstance(initial_data, pd.DataFrame):
            raise ValueError("Expected df to be a pandas DataFrame")

        # Normalize prices
        self.df = initial_data / initial_data.max()
        self.transaction_cost = transaction_cost
        self.current_step = 0  # Initialize current_step

        # Define action and observation space
        num_columns = len(initial_data.columns)
        self.action_space = spaces.MultiDiscrete([3] * (num_columns // 2))
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_columns,))

        # Portfolio holds the quantity of each stock
        self.portfolio = np.zeros(num_columns // 2)

        # Initialize the data buffer with initial data
        self.data_buffer = deque(initial_data.values, maxlen=max_buffer_size)    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.array) -> tuple:
        # Wait for new data if at the end of the buffer
        while self.current_step >= len(self.data_buffer) - 1:
            time.sleep(1)  # Sleep for 1 second, or adjust as needed

        self.current_step += 1
        next_state = list(self.data_buffer)[self.current_step]  # Convert deque to list for indexing


        # Update portfolio
        current_prices = self.df.iloc[self.current_step].values[:len(self.portfolio)]
        for i, act in enumerate(action):
            if act == 1: # Buy
                self.portfolio[i] += (current_prices[i] * (1 + self.transaction_cost))
            elif act == 2: # Sell
                self.portfolio[i] -= (current_prices[i] * (1 - self.transaction_cost))

        # Calculate portfolio value
        portfolio_value = np.sum(self.portfolio * current_prices)

        # Calculate reward as portfolio value minus risk
        reward = portfolio_value - np.std(self.portfolio)

        done = self.current_step >= len(self.df) - 1

        info = {}

        # Return the current observation
        obs = self.df.iloc[self.current_step].values

        return next_state, reward, done, {}
    
    def update_current_state(self, state):
        self.current_state = state

    def reset(self):
        # Reset to the most recent data point
        self.current_step = len(self.data_buffer) - 1
        return list(self.data_buffer)[self.current_step]  # Convert deque to list for indexing
 
    def render(self, mode='human'):
        logging.info(f'Step: {self.current_step}, Portfolio Value: {np.sum(self.portfolio * self.df.iloc[self.current_step].values[:len(self.portfolio)])}')
