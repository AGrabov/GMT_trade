import logging
import numpy as np
import pandas as pd
import gym
from gym import Env, spaces, utils
from gym.utils import seeding
from collections import deque
from data_normalizer import CryptoDataNormalizer
import time
import matplotlib.pyplot as plt
import mplfinance as mpf
import talib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_data, max_buffer_size=1000, transaction_cost=0.0004, cash=100.0, live=False, observation_window_length=10, debug=False) -> None:
        super(TradingEnv, self).__init__()

        self.live = live
        self.money = cash
        self.observation_window_length = observation_window_length
        self.debug = debug

        # Initialize the normalizer
        self.normalizer = CryptoDataNormalizer()

        

        # # When you need to compute the actual price from normalized returns:
        # actual_price = normalizer.inverse_transform_returns(normalized_return, previous_actual_price)


        # Ensure df is a DataFrame
        if not isinstance(initial_data, pd.DataFrame):
            raise ValueError("Expected df to be a pandas DataFrame")

        self.df = self._calculate_indicators(initial_data)
        if self.debug:
            print(f'Dataframe and indicators: \n{self.df.head(5)} \n Dataframe length: {len(self.df)}')

        
        # Normalize values        
        # self.df = self.normalizer.normalize_returns(self.df)
        self.df = self.df / self.df.max(axis=0)
        self.transaction_cost = transaction_cost
        self.current_step = 0  # Initialize current_step

        self.num_columns = len(initial_data.columns) // 2
        self.action_space = spaces.MultiDiscrete([5])  # Only one action for the 'close' column
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.df.columns) * self.observation_window_length,))
        self.buy_price = np.zeros(self.num_columns)
        self.short_price = np.zeros(self.num_columns)

        # Handle NaN values
        self._handle_nan_values(self.df)

        # Store the initial data
        self.initial_data = initial_data.copy()

        # Initialize the data buffer with initial data
        self.data_buffer = deque(self.initial_data.values, maxlen=max_buffer_size)

        # Initialize portfolio and portfolio_value
        self.portfolio = np.zeros(self.num_columns)  # Represents quantity of each asset
        self.cash = self.money
        self.portfolio_value = self.cash

        # Initialize prev_portfolio_value and portfolio_values
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_values = [self.prev_portfolio_value]
        self.portfolio_difference = 0
        self.current_close = 0

        # For tracking trades
        self.trades = []

        # For tracking positions
        self.long_position_open = False
        self.short_position_open = False

        # For rewards and penalties (Rewards are positive, penalties are negative)
        self.penalty = 0
        self.reward = 0

        # Initialize a cooldown counter
        self.cooldown_counter = 0
        self.cooldown_period = 3  # Number of steps to wait after a trade

        # For real-time plotting
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Portfolio Value Over Time')
        self.ax.set_xlabel('Time Step')
        self.ax.set_ylabel('Portfolio Value')

    def _handle_nan_values(self, df=None):
        if df is None:
            df = self.df

        if df.isna().any().any() and self.debug:
            print("Warning: NaN values detected in the normalized data!\n Filling NaN values with various methods.")
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicators"""
    
        # Ensure df is not None or empty
        if df is None or df.empty:
            raise ValueError("Invalid dataframe provided for indicator calculation.")
    
        # Calculate Heikin Ashi candlesticks
        df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        df['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
        df['HA_High'] = df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
        df['HA_Low'] = df[['low', 'HA_Open', 'HA_Close']].min(axis=1)

        ha_diff = df['HA_Close'] - df['HA_Open']
        df['HA_Diff'] = talib.MA(ha_diff, timeperiod=5, matype=0)

        # Calculate Moving Average
        df['MA'] = talib.MA(df['close'], timeperiod=50, matype=0)            

        # Calculate Moving Average Convergence Divergence
        df['MACD'], df['MACDSIGNAL'], df['MACDHIST'] = talib.MACDEXT(df['close'], 
                                                                     fastperiod=12, fastmatype=2,
                                                                     slowperiod=26, slowmatype=1, 
                                                                     signalperiod=9, signalmatype=8)

        # Calculate Bollinger Bands
        df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = talib.BBANDS(df['close'], 
                                                                       timeperiod=20, nbdevup=2, 
                                                                       nbdevdn=2, matype=0)
        
        # Calculate Moving Average with variable period
        df['DEMA'] = talib.DEMA(df['close'], timeperiod=25)

        # Calculate MESA Adaptive Moving Average         
        df['MAMA'], df['FAMA'] = talib.MAMA(df['close'], fastlimit=0.5, slowlimit=0.05)
        
        # Calculate Absolute Price Oscillator
        df['APO'] = talib.APO(df['close'], fastperiod=3, slowperiod=13, matype=2)

        # Calculate Chande Momentum Oscillator
        df['CMO'] = talib.CMO(df['close'], timeperiod=14)

        # Calculate Chaikin A/D Oscillator
        df['ADOSC'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'], fastperiod=3, slowperiod=10)

        # Calculate Normalized Average True Range
        df['NATR'] = talib.NATR(df['high'], df['low'], df['close'], timeperiod=13)

        df['ST_RSI_K'], df['ST_RSI_D'] = talib.STOCHRSI(df['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=1)

        # Calculate Hilbert Transform
        df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['close']) # Instantaneous Trendline        
        df['HT_DCPERIOD'] = talib.HT_DCPERIOD(df['close']) # Dominant Cycle Period        
        df['HT_DCPHASE'] = talib.HT_DCPHASE(df['close']) # Dominant Cycle Phase        
        df['HT_INPHASE'], df['HT_QUADRATURE'] = talib.HT_PHASOR(df['close']) # Dominant Cycle Inphase        
        df['HT_SINE'], df['HT_LEAD_SINE'] = talib.HT_SINE(df['close']) # Dominant Cycle Sine        
        df['HT_TRENDMODE'] = talib.HT_TRENDMODE(df['close']) # Trend Mode

        df['CORREL'] = talib.CORREL(df['high'], df['low'], timeperiod=30)

        df['LREG'] = talib.LINEARREG(df['close'], timeperiod=14)

        df['LREG_ANGLE'] = talib.LINEARREG_ANGLE(df['close'], timeperiod=14)

        df['LREG_INT'] = talib.LINEARREG_INTERCEPT(df['close'], timeperiod=14)

        df['LREG_SLOPE'] = talib.LINEARREG_SLOPE(df['close'], timeperiod=14)

        df['STDDEV'] = talib.STDDEV(df['close'], timeperiod=5, nbdev=1)

        df['TSF'] = talib.TSF(df['close'], timeperiod=14)

        df['VAR'] = talib.VAR(df['close'], timeperiod=5, nbdev=1)

        

        self._handle_nan_values(df)

        return df

    def reset(self):
        self.current_step = 0
        self.cash = self.money
        self.portfolio = np.zeros(self.num_columns)
        self.buy_price = np.zeros(self.num_columns)
        self.short_price = np.zeros(self.num_columns)
        self.long_position_open = False
        self.short_position_open = False        
        self.penalty = 0
        self.reward = 0
        self.cooldown_counter = 0
        self.portfolio_value = self.cash
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_difference = 0
        self.current_close = 0
        self.portfolio_values = [self.prev_portfolio_value]
        self.trades = []
        self.df = self._calculate_indicators(self.initial_data)
        
        self.df = self.df / self.df.max(axis=0)
        # self.df = self.normalizer.normalize_returns(self.df)
        self._handle_nan_values(self.df)
        self.data_buffer = deque(self.initial_data.values, maxlen=self.data_buffer.maxlen)
        return self._next_observation()

    def _next_observation(self):
        if self.current_step <= self.observation_window_length:
            obs = self.df.iloc[: self.current_step].values.flatten()
        else:
            obs = self.df.iloc[self.current_step - self.observation_window_length: self.current_step].values.flatten()
        return obs

    def step(self, action):         
        self.current_step += 1        
        self._take_action(action)
        reward = self._get_reward()        
        self.prev_portfolio_value = self.portfolio_value
        if self.debug:
            logger.info(f'Step: {self.current_step}, Action: {action}, Reward: {reward:.5f}, Portfolio value: {self.portfolio_value:.5f}\n'
                        f'---------------------- ')
        done = self.current_step >= len(self.df) - 1
        if self.portfolio_value <= 0:
            done = True
        obs = self._next_observation()
        return obs, reward, done, {}

    def _take_action(self, action): 
        # done = False
        self.current_close = 0
        self.portfolio_difference = 0
        self.penalty = 0
        self.reward = 0   
             
        if self.current_step < 2:
            return
        # Ensure there are enough steps left in the dataframe for the observation window
        if self.current_step < self.observation_window_length:
            # If not enough steps, take as many as available from the start
            current_prices = self.df.iloc[:self.current_step]['close'].values
            price_diff = current_prices[-1] - current_prices[-2] if len(current_prices) > 1 else 0.01 

            self.current_close = current_prices[-1]
        else:
            current_prices = self.df.iloc[self.current_step - self.observation_window_length:self.current_step]['close'].values
            price_diff = current_prices[0] - current_prices[1]            
            self.current_close = current_prices[0]
        # price_diff *= 100
        price_diff = round(price_diff, 5)
        
        # Boundary check for current_step
        if self.current_step >= len(self.df):
            return
        
        self.portfolio_value = self.cash + self.portfolio[0] * self.current_close #self.df.iloc[self.current_step]['close']
        self.portfolio_difference = self.portfolio_value - self.prev_portfolio_value
        
        
        
        if not self.long_position_open and not self.short_position_open:
            if action == 1:  # Go Long (Buy)
                if self.current_close > 1e-10:  # Ensure current_close is not too small
                    quantity = self.cash * 0.5 / self.current_close
                else:
                    quantity = 0
                transaction_fee = quantity * self.current_close * self.transaction_cost
                trade_cost = (quantity * self.current_close + transaction_fee)
                self.cash -= trade_cost
                self.portfolio[0] += quantity
                self.buy_price[0] = self.current_close
                self.trades.append((self.current_step, 'buy'))
                self.long_position_open = True
                self.reward += 0.05
                if self.debug:
                    logger.info(f"Step {self.current_step}: Opening Long at price {self.current_close:.5f}, Cost: {trade_cost:.5f}")
            elif action == 3:  # Go Short (Sell)
                if self.current_close > 1e-10:  # Ensure current_close is not too small
                    quantity = self.cash * 0.5 / self.current_close
                else:
                    quantity = 0
                transaction_fee = quantity * self.current_close * self.transaction_cost
                self.cash += quantity * self.current_close - transaction_fee
                self.portfolio[0] -= quantity
                self.short_price[0] = self.current_close
                self.trades.append((self.current_step, 'sell'))
                self.short_position_open = True
                self.reward += 0.05
                if self.debug:
                    logger.info(f"Step {self.current_step}: Opening Short at price {self.current_close:.5f}, Cost: {quantity * self.current_close:.5f}")
            elif action == 2 or action == 4:  # Incorrect action
                self.penalty += 0.1
                if self.debug:
                    logger.info(f"Step {self.current_step}: Incorrect action, penalty: {self.penalty:.5f}")

        elif self.long_position_open:
            if action == 2:  # Close Long Position
                profit_or_loss = self.portfolio[0] * (self.current_close - self.buy_price[0])
                trade_cost = self.portfolio[0] * self.current_close
                transaction_fee = trade_cost * self.transaction_cost
                cash_update = trade_cost + profit_or_loss - transaction_fee                    
                self.cash += cash_update                
                self.portfolio[0] = 0
                self.buy_price[0] = 0
                self.trades.append((self.current_step, 'close_long'))
                self.long_position_open = False
                if profit_or_loss > 0:
                    self.reward += 0.25
                else:
                    self.penalty += 0.15
                if self.debug:
                    logger.info(f"Step {self.current_step}: Closed Long ---- Profit/loss: {profit_or_loss:.5f}")
            elif action == 3 or action == 4 or action == 1:
                self.penalty += 0.1
                if self.debug:
                    logger.info(f"Step {self.current_step}: Incorrect action, penalty: {self.penalty:.5f}")

        elif self.short_position_open:
            if action == 4:  # Close Short Position
                profit_or_loss = abs(self.portfolio[0]) * (self.short_price[0] - self.current_close)
                trade_cost = abs(self.portfolio[0]) * self.current_close
                transaction_fee = trade_cost * self.transaction_cost
                cash_update = trade_cost - profit_or_loss + transaction_fee                
                self.cash -= cash_update                
                self.portfolio[0] = 0
                self.short_price[0] = 0
                self.trades.append((self.current_step, 'close_short'))
                self.short_position_open = False
                if profit_or_loss > 0:
                    self.reward += 0.25
                else:
                    self.penalty += 0.15
                if self.debug:
                    logger.info(f"Step {self.current_step}: Closed Short ---- Profit/loss: {profit_or_loss:.5f}")
            elif action == 1 or action == 2 or action == 3:
                self.penalty += 0.1
                if self.debug:
                    logger.info(f"Step {self.current_step}: Incorrect action, penalty: {self.penalty:.5f}")
                
        elif action == 0:  # Do nothing
            if self.portfolio_difference <= 0:
                self.penalty += 0.15
            else:
                self.reward += 0.1
            if self.debug:
                logger.info(f"Step {self.current_step}: Hold")

        
        
        self.portfolio_value = self.cash + self.portfolio[0] * self.current_close #self.df.iloc[self.current_step]['close']
        self.portfolio_difference = self.portfolio_value - self.prev_portfolio_value        
        self.portfolio_values.append(self.portfolio_value)

        self.reward *= (abs(price_diff) + abs(self.portfolio_difference / self.portfolio_value) )
        self.penalty *= (abs(price_diff) + abs(self.portfolio_difference / self.portfolio_value) )

        logger.info(f"Step {self.current_step}: Cash: {self.cash:.2f}, Portfolio: {self.portfolio[0]:.5f}, Close: {self.current_close:.5f}, Portfolio Value: {self.portfolio_value:.5f}")

    # def _get_portfolio_value(self):
    #     return self.cash + self.portfolio[0] * self.df.iloc[self.current_step]['close']

    def _get_reward(self):
        reward = 0        
        reward += (self.reward - self.penalty) #* self.portfolio_value/10        
        self.portfolio_value += reward        
        if self.debug:
            logger.info(f"Step {self.current_step}: Reward: {reward:.5f}")      
        return reward
    
    def render(self, mode='human', close=False):
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

    def close(self):
        pass
