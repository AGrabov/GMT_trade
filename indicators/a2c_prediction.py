# indicators/a2c_prediction.py

import backtrader as bt
from stable_baselines3 import A2C
from machine_learning.gym_trading_env_03 import TradingEnv 
# from machine_learning.env_001 import TradeEnv 
import pandas as pd

class A2CPredictor(bt.Indicator):
    lines = ('prediction',)
    params = dict(live = False,
                model_path="./models/A2C_trade_GMTUSDT_30m.zip"        
    )

    def __init__(self, data):
        # Load the trained model
        self.model = A2C.load(self.params.model_path)
        self.initialized = False
        self.data = data

    def _initialize_env(self):        
        # Convert the data to a DataFrame
        data_df = pd.DataFrame({
            'open': self.data.open.get(size=self.data.buflen(), ago=0),
            'high': self.data.high.get(size=self.data.buflen(), ago=0),
            'low': self.data.low.get(size=self.data.buflen(), ago=0),
            'close': self.data.close.get(size=self.data.buflen(), ago=0),
            'volume': self.data.volume.get(size=self.data.buflen(), ago=0)
        })

        # Check if data is available
        if len(data_df) < 10:
            return

        # Initialize the environment with the actual data
        self.env = TradingEnv(initial_data=data_df, live=self.params.live)
        # self.env = TradeEnv(initial_data=data_df, live=self.params.live)
        self.observation = self.env.reset()
        self.initialized = True

    def next(self):
        # Initialize the environment if it's the first call to next
        if not self.initialized:
            self._initialize_env()

            # If still not initialized, skip this iteration
            if not self.initialized:
                return

        # Get the current state from the data feed
        current_state = [self.data.open[0], self.data.high[0], self.data.low[0], self.data.close[0], self.data.volume[0]]
        
        # Update the environment's current state
        self.env.update_current_state(current_state)
        
        # Predict the action using the model
        action, _ = self.model.predict(self.observation)

        # Take the action in the environment
        self.observation, _, _, _ = self.env.step(action)
        
        # Set the prediction line
        self.lines.prediction[0] = action[0]

