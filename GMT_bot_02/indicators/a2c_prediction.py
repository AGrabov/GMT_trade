# indicators/a2c_prediction.py

import backtrader as bt
from stable_baselines3 import A2C
from machine_learning.gym_env_01 import TradingEnv
import pandas as pd

class A2CPredictor(bt.Indicator):
    lines = ('prediction',)

    def __init__(self):
        # Load the trained model
        self.model = A2C.load("./models/a2c_trading_30m.zip")
        
        # Convert the data to a DataFrame
        data_df = pd.DataFrame({
            'open': self.data.open.get(size=self.data.buflen()),
            'high': self.data.high.get(size=self.data.buflen()),
            'low': self.data.low.get(size=self.data.buflen()),
            'close': self.data.close.get(size=self.data.buflen()),
            'volume': self.data.volume.get(size=self.data.buflen())
        })

        print(data_df.head())
        
        # Initialize the environment with the actual data
        self.env = TradingEnv(initial_data=data_df)
        
        self.observation = self.env.reset() if len(data_df) != 0 else None

    def next(self):
        # Get the current state from the data feed
        current_state = [self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0], self.data.volume[0]]
        
        # Update the environment's current state
        self.env.update_current_state(current_state)
        
        # Predict the action using the model
        action, _ = self.model.predict(self.observation)
        
        # Update the observation by taking a step in the environment
        self.observation, _, _, _ = self.env.step(action)
        
        # Set the prediction line
        self.lines.prediction[0] = action
