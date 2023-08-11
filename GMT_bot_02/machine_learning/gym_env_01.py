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
from pandas import isna
import talib


logger = logging.getLogger(__file__)

class TradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_data, max_buffer_size=1000, transaction_cost=0.0005, cash=1000.0, live=False) -> None:
        super(TradingEnv, self).__init__()

        self.live = live
        self.money = cash
        
        # Ensure df is a DataFrame
        if not isinstance(initial_data, pd.DataFrame):
            raise ValueError("Expected df to be a pandas DataFrame")                       
        
        self.df = self._calculate_indicators(initial_data)

        print(f'Dataframe and indicators: \n{self.df.head(10)} \n Dataframe length: {len(self.df)}')        
                
        # Normalize values
        self.df = self.df / self.df.max(axis=0)
        self.transaction_cost = transaction_cost
        self.current_step = 0  # Initialize current_step      
        
        self.num_columns = len(initial_data.columns) // 2
        self.action_space = spaces.MultiDiscrete([5] * self.num_columns)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.df.columns),))        
        self.buy_price = np.zeros(self.num_columns)
        self.short_price = np.zeros(self.num_columns)  

        if self.df.isna().any().any():
            print("Warning: NaN values detected in the normalized data!\n Filling NaN values with various methods.")
            # Handle NaN values
            self.df.fillna(method='ffill', inplace=True)
            self.df.fillna(method='bfill', inplace=True)
            self.df.fillna(0, inplace=True)

        # Initialize the data buffer with initial data
        self.data_buffer = deque(self.df.values, maxlen=max_buffer_size)

        # Initialize portfolio and portfolio_value
        self.portfolio = 0.0  # Represents quantity of the asset
        self.cash = self.money
        self.portfolio_value = self.cash

        # Initialize prev_portfolio_value and portfolio_values
        self.prev_portfolio_value = self.cash
        self.portfolio_values = [self.cash]
        
        # For tracking trades
        self.trades = []

        self.long_position_open = False
        self.short_position_open = False

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

        # # Calculate Weighted Close Price
        # df['WCLPRICE'] = talib.WCLPRICE(df['high'], df['low'], df['close'])

        # # Calculate True Range
        # df['TRANGE'] = talib.TRANGE(df['high'], df['low'], df['close'])

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
        df.fillna(0, inplace=True)

        return df

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.array) -> tuple:
        # If in live mode and at the end of the buffer, wait for new data
        if self.live:
            while self.current_step >= len(self.data_buffer) - 1:
                time.sleep(1)  # Sleep for 1 second, or adjust as needed

        # If not in live mode and at the end of the buffer, set done to True
        if not self.live and self.current_step >= len(self.data_buffer) - 1:
            done = True
            return self.data_buffer[self.current_step], -1000, done, {}  # Return the last state with significant negative reward

        self.current_step += 1
        

        # Update portfolio
        current_prices = self.df.iloc[self.current_step].values[:len(self.portfolio)]
        for i, act in enumerate(action):
            if not self.long_position_open and not self.short_position_open:  # No trade is currently open
                if act == 1:  # Go Long (Buy)
                    quantity = self.cash * 0.9 / current_prices[i]
                    transaction_cost = quantity * current_prices[i] * self.transaction_cost
                    if self.cash >= transaction_cost:
                        self.cash -= (quantity * current_prices[i] + transaction_cost)
                        self.portfolio[i] += quantity
                        self.buy_price[i] = current_prices[i]
                        self.trades.append((self.current_step, 'buy'))
                        self.long_position_open = True
                elif act == 3:  # Go Short (Sell)
                    quantity = self.cash * 0.9 / current_prices[i]
                    transaction_cost = quantity * current_prices[i] * self.transaction_cost
                    if self.cash >= transaction_cost:
                        self.cash -= transaction_cost
                        self.portfolio[i] -= quantity
                        self.short_price[i] = current_prices[i]
                        self.trades.append((self.current_step, 'sell'))
                        self.short_position_open = True
            elif self.long_position_open:  # There's an open long position for this asset
                if act == 2:  # Close Long Position
                    profit_or_loss = self.portfolio[i] * (current_prices[i] - self.buy_price[i])
                    self.cash += (self.portfolio[i] * current_prices[i] + profit_or_loss - profit_or_loss * self.transaction_cost)
                    self.portfolio[i] = 0
                    self.buy_price[i] = 0
                    self.trades.append((self.current_step, 'close_long'))
                    self.long_position_open = False
            elif self.short_position_open:  # There's an open short position for this asset
                if act == 4:  # Close Short Position
                    profit_or_loss = self.portfolio[i] * (self.short_price[i] - current_prices[i])
                    self.cash += (-self.portfolio[i] * current_prices[i] + profit_or_loss - profit_or_loss * self.transaction_cost)
                    self.portfolio[i] = 0
                    self.short_price[i] = 0
                    self.trades.append((self.current_step, 'close_short'))
                    self.short_position_open = False

            elif act == 0:  # Do nothing
                pass

        # Calculate portfolio value
        asset_value = self.portfolio * current_prices[0]  # Using index 0 since it's a single asset
        portfolio_value = asset_value + self.cash

        # Calculate reward as the difference in portfolio value from the previous step
        reward = float(portfolio_value - self.prev_portfolio_value)  # Ensure reward is a scalar
        self.prev_portfolio_value = portfolio_value
        self.portfolio_values.append(portfolio_value)

        # Calculate drawdown
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (peak - portfolio_value) / peak

        # Calculate SQN
        rewards_array = np.array(self.portfolio_values[1:]) - np.array(self.portfolio_values[:-1])
        sqn = np.mean(rewards_array) / np.std(rewards_array) * np.sqrt(len(rewards_array))

        # Calculate number of winning trades
        winning_trades = len([r for r in rewards_array if r > 0])

        obs = self.df.iloc[self.current_step].values

        done = self.current_step >= len(self.df) - 1

        info = {
            'drawdown': drawdown,
            'sqn': sqn,
            'winning_trades': winning_trades
        }

        return obs, reward, done, info

    
    def update_current_state(self, state):
        # Update the current state in the dataframe
        self.df.iloc[self.current_step] = state
        
        # Calculate indicators for the new data
        self.df = self._calculate_indicators(self.df)
                
        # Normalize the data
        max_values = self.df.max(axis=0)
        self.df = self.df / max_values

        # Handle NaN values
        self.df.fillna(method='ffill', inplace=True)
        self.df.fillna(method='bfill', inplace=True)
        self.df.fillna(0, inplace=True)


    def reset(self):
        # Reset the current step to the beginning
        self.current_step = 0

        # Reset prices
        self.buy_price = np.zeros(self.num_columns)
        self.short_price = np.zeros(self.num_columns)

        # Reset portfolio and portfolio_value
        self.portfolio = 0.0
        self.cash = self.money
        self.portfolio_value = self.cash

        # Reset prev_portfolio_value and portfolio_values
        self.prev_portfolio_value = self.cash
        self.portfolio_values = [self.cash]

        # Clear trades for a new episode
        self.trades = []

        # Reset the data buffer
        self.data_buffer = []

        self.long_position_open = False
        self.short_position_open = False

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