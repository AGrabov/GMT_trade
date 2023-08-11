import logging
import numpy as np
import pandas as pd
import gym
from gym import Env, spaces, utils
from gym.utils import seeding
from collections import deque
import time
import matplotlib.pyplot as plt
import mplfinance as mpf
import talib


logger = logging.getLogger(__file__)

class TradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_data, max_buffer_size=1000, transaction_cost=0.0005) -> None:
        super(TradingEnv, self).__init__()

        # Ensure df is a DataFrame
        if not isinstance(initial_data, pd.DataFrame):
            raise ValueError("Expected df to be a pandas DataFrame")                       
        
        self.df = self._calculate_indicators(initial_data)
                
        # Normalize values
        self.df = self.df / self.df.max(axis=0)
        self.transaction_cost = transaction_cost
        self.current_step = 0  # Initialize current_step
        
        # Define action and observation space
        num_columns = len(initial_data.columns) // 2  # Assuming you have OHLC for each asset
        self.action_space = spaces.MultiDiscrete([5] * num_columns)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.df.columns),))

        print(f'Dataframe: \n{self.df.head()} \n Dataframe length: {len(self.df)}')

        # Portfolio holds the quantity of each stock
        self.portfolio = np.zeros(num_columns // 2)
        self.buy_price = np.zeros(num_columns // 2)
        self.short_price = np.zeros(num_columns // 2)

        # Initialize the data buffer with initial data
        self.data_buffer = deque(self.df.values, maxlen=max_buffer_size)

        self.prev_portfolio_value = np.sum(self.portfolio * self.df.iloc[0].values[:len(self.portfolio)])
        self.portfolio_values = [self.prev_portfolio_value]  # To store portfolio values over time for rendering

        # For tracking trades
        self.trades = []

        # For real-time plotting
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Portfolio Value Over Time')
        self.ax.set_xlabel('Time Step')
        self.ax.set_ylabel('Portfolio Value')

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators"""    

        # Calculate Heikin Ashi candlesticks
        df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
        df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)

        # Calculate Weighted Close Price
        df['WCLPRICE'] = talib.WCLPRICE(df['high'], df['low'], df['close'])

        # Calculate True Range
        df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])

        # Calculate Kaufman Adaptive Moving Average
        df['KAMA'] = talib.KAMA(df['close'], timeperiod=30)

        # Calculate MESA Adaptive Moving Average         
        mama, fama = talib.MAMA(df['close'], fastlimit=0.5, slowlimit=0.05)
        df['MAMA'] = mama
        df['FAMA'] = fama

        # Calculate Absolute Price Oscillator
        df['APO'] = talib.APO(df['close'], fastperiod=3, slowperiod=10, matype=3)

        # Calculate Percentage Price Oscillator for Haikin Ashi close
        df['PPO'] = talib.PPO(df['HA_Close'], fastperiod=12, slowperiod=26, matype=7)

        # Calculate Chande Momentum Oscillator
        df['CMO'] = talib.CMO(df['close'], timeperiod=14)        

        # Calculate Chaikin A/D Oscillator
        df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

        # Calculate Money Flow Index
        df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)

        # Calculate Normalized Average True Range
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=14)

        # Calculate Time Series Forecast for Heikin Ashi close
        df['HA_TSF'] = talib.TSF(df['HA_Close'], timeperiod=14)

        # Calculate On Balance Volume for Heikin Ashi close
        df['HA_OBV'] = talib.OBV(df['HA_Close'], df['volume'])

        # Calculate Variance
        df['VAR'] = talib.VAR(df['close'], timeperiod=5, nbdev=1)

        # Calculate Hilbert Transform
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['close']) # Instantaneous Trendline        
        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['close']) # Dominant Cycle Period        
        df['HT_DCPHASE'] = talib.HT_DCPHASE(df['close']) # Dominant Cycle Phase        
        inphase, quadrature = talib.HT_PHASOR(df['close']) # Dominant Cycle Inphase
        df['HT_INPHASE'] = inphase
        df['HT_QUADRATURE'] = quadrature        
        sine, leadsine = talib.HT_SINE(df['close']) # Dominant Cycle Sine
        df['HT_SINE'] = sine
        df['HT_LEAD_SINE'] = leadsine        
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(df['close']) # Trend Mode

        # Handle NaN values
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        return df

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.array) -> tuple:
        # Wait for new data if at the end of the buffer
        while self.current_step >= len(self.data_buffer) - 1:
            time.sleep(1)  # Sleep for 1 second, or adjust as needed

        self.current_step += 1
        next_state = self.data_buffer[self.current_step]

        # Update portfolio
        current_prices = self.df.iloc[self.current_step].values[:len(self.portfolio)]
        for i, act in enumerate(action):
            if act == 1:  # Go Long (Buy)
                transaction_value = current_prices[i]
                self.portfolio[i] -= transaction_value + (transaction_value * self.transaction_cost)
                self.buy_price[i] = current_prices[i]
                self.trades.append((self.current_step, 'buy'))
            elif act == 2:  # Close Long Position
                profit_or_loss = current_prices[i] - self.buy_price[i]
                self.portfolio[i] += profit_or_loss - (profit_or_loss * self.transaction_cost)
                self.buy_price[i] = 0
                self.trades.append((self.current_step, 'close_long'))
            elif act == 3:  # Go Short (Sell)
                transaction_value = current_prices[i]
                self.portfolio[i] += transaction_value - (transaction_value * self.transaction_cost)
                self.short_price[i] = current_prices[i]
                self.trades.append((self.current_step, 'sell'))
            elif act == 4:  # Close Short Position
                profit_or_loss = self.short_price[i] - current_prices[i]
                self.portfolio[i] -= profit_or_loss + (profit_or_loss * self.transaction_cost)
                self.short_price[i] = 0
                self.trades.append((self.current_step, 'close_short'))

        # Calculate portfolio value
        portfolio_value = np.sum(self.portfolio * current_prices)

        # Calculate reward as the difference in portfolio value from the previous step
        reward = portfolio_value - self.prev_portfolio_value
        self.prev_portfolio_value = portfolio_value
        self.portfolio_values.append(portfolio_value)

        done = self.current_step >= len(self.df) - 1

        info = {}

        # Return the current observation
        obs = self.df.iloc[self.current_step].values

        return next_state, reward, done, {}

    def reset(self):
        # Reset the current step to the beginning
        self.current_step = 0

        # Reset portfolio and prices
        self.portfolio = np.zeros(len(self.df.columns) // 2)
        self.buy_price = np.zeros(len(self.df.columns) // 2)
        self.short_price = np.zeros(len(self.df.columns) // 2)

        # Reset previous portfolio value
        self.prev_portfolio_value = np.sum(self.portfolio * self.df.iloc[0].values[:len(self.portfolio)])
        
        # Reset the list storing portfolio values over time
        self.portfolio_values = [self.prev_portfolio_value]

        # Clear trades for a new episode
        self.trades = []

        # Clear the plot for a new episode
        self.ax.clear()
        self.ax.set_title('Portfolio Value Over Time')
        self.ax.set_xlabel('Time Step')
        self.ax.set_ylabel('Portfolio Value')

        # Return the initial state
        return self.df.iloc[self.current_step].values

    def render(self, mode='human'):
        logging.info(f'Step: {self.current_step}, Portfolio Value: {np.sum(self.portfolio * self.df.iloc[self.current_step].values[:len(self.portfolio)])}')

        # Extract OHLC data up to the current step
        ohlc_data = self.df.iloc[:self.current_step+1]
        
        # Create a new figure and set of subplots
        fig, axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
        
        # Plot candlestick chart
        mpf.plot(ohlc_data, type='candle', ax=axes[0], volume=axes[1], style='charles')
        
        # Plot portfolio value on the same axis as the candlestick chart
        axes[0].plot(self.portfolio_values, color='blue', label='Portfolio Value')
        
        # Plot trades
        for trade in self.trades:
            y_value = self.portfolio_values[trade[0]]  # Get the portfolio value at the trade step
            if trade[1] == 'buy':
                axes[0].scatter(trade[0], y_value, color='green', marker='^', label='Buy')
            elif trade[1] == 'sell':
                axes[0].scatter(trade[0], y_value, color='red', marker='v', label='Sell')
            elif trade[1] == 'close_long':
                axes[0].scatter(trade[0], y_value, color='blue', marker='^', label='Close Long')
            elif trade[1] == 'close_short':
                axes[0].scatter(trade[0], y_value, color='orange', marker='v', label='Close Short')

        # To avoid duplicate labels in the legend
        handles, labels = axes[0].get_legend_handles_labels()
        unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
        axes[0].legend(*zip(*unique))

        plt.pause(0.01)  # Pause to update the plot