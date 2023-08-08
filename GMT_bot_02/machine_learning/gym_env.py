import logging
import numpy as np
import pandas as pd
import gym
from gym import Env, spaces, utils
from gym.utils import seeding

logger = logging.getLogger(__file__)

class TradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, transaction_cost=0.0005) -> None:
        super(TradingEnv, self).__init__()

        self.df = df
        self.transaction_cost = transaction_cost

        # Normalize prices
        self.df = self.df / self.df.max()

        # Define action and observation space
        # Assume df has columns for each stock's price and volume
        self.action_space = spaces.MultiDiscrete([3]*len(df.columns)//2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(df.columns),))

        # Portfolio holds the quantity of each stock
        self.portfolio = np.zeros(len(df.columns)//2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.array) -> tuple:
        self.current_step += 1

        # Update portfolio
        for i, act in enumerate(action):
            if act == 1: # Buy
                self.portfolio[i] += (1 - self.transaction_cost)
            elif act == 2: # Sell
                self.portfolio[i] -= (1 + self.transaction_cost)

        # Calculate portfolio value
        portfolio_value = np.sum(self.portfolio * self.df.iloc[self.current_step].values[:len(self.portfolio)])

        # Calculate reward as portfolio value minus risk
        # Risk can be defined in many ways, here we're just using standard deviation of portfolio values
        reward = portfolio_value - np.std(self.portfolio)

        done = self.current_step >= len(self.df) - 1

        info = {}

        # Return the current observation
        obs = self.df.iloc[self.current_step].values

        return obs, reward, done, info

    def reset(self) -> np.array:
        self.current_step = 0
        self.portfolio = np.zeros(len(self.df.columns)//2)

        # Return the initial state
        return self.df.iloc[self.current_step].values

    def render(self, mode='human'):
        logging.info(f'Step: {self.current_step}, Portfolio Value: {np.sum(self.portfolio * self.df.iloc[self.current_step].values[:len(self.portfolio)])}')

