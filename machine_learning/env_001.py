import logging
import time
from collections import deque
# from machine_learning.ta_indicators import TAIndicators
from ta_indicators import TAIndicators

import gym
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import talib
from data_normalizer import CryptoDataNormalizer
from gym import Env, spaces

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradeEnv(gym.Env):
    """Custom Trading Environment that follows gym interface."""
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_data, action_space=spaces.Discrete(5), max_buffer_size=1000, transaction_cost=0.0004, cash=100.0, live=False,
                observation_window_length=48, ddqn=False, debug=False) -> None:
        super(TradeEnv, self).__init__()
        
        self.normalizer = CryptoDataNormalizer()
        self.action_space = action_space        
        self.live = live
        self.money = cash
        self.transaction_cost = transaction_cost
        self.observation_window_length = observation_window_length
        self.ddqn = ddqn
        self.debug = debug
        self.ta = TAIndicators(initial_data)
        # initial_data['close'] = initial_data['close'].astype(float)
        initial_data['close_change_pct'] = initial_data['close'].pct_change()
        self.df = TAIndicators(initial_data)._calculate_indicators()
        # self.df = self._calculate_indicators(initial_data)
        self._normalize_data()        
        
        self.current_step = 0
        self.num_columns = len(initial_data.columns) // 2
        # self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.df.columns) * self.observation_window_length,))
        self.buy_price = np.zeros(self.num_columns)
        self.short_price = np.zeros(self.num_columns)
        self.initial_data = initial_data.copy()
        self.data_buffer = deque(self.initial_data.values, maxlen=max_buffer_size)
        self.portfolio = np.zeros(self.num_columns)
        self.cash = self.money
        self.portfolio_value = self.cash
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_values = [self.prev_portfolio_value]
        self.trades = []
        self.closed_trades = []
        self.metrics = {}
        self.long_position_open = False
        self.short_position_open = False
        self.portfolio_difference = 0
        self.current_close = 0
        self.normalized_close = 0
        self.close_difference = 0
        self.penalty = 0
        self.reward = 0
        self.cooldown_counter = 0
        self.cooldown_period = 3
        self.prev_action = None
        self.no_action_counter = 0
        self.no_action_period = 10
        self.invalid_action = False
        self.invalid_action_counter = 0
        self._initialize_plotting()    

    def _normalize_data(self):
        """Normalize data values."""
        self.df = self.normalizer.normalize_returns(self.df)
        # max_values = self.df.max(axis=0)
        # self.df = self.df / max_values
        self._handle_nan_values(self.df)

    def _initialize_plotting(self):
        """Initialize real-time plotting."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_title('Portfolio Value Over Time')
        self.ax.set_xlabel('Time Step')
        self.ax.set_ylabel('Portfolio Value')

    def _handle_nan_values(self, df=None):
        """Handle NaN values in the dataframe."""
        if df is None:
            df = self.df

        # if df.isna().any().any() and self.debug:
        #     print("Warning: NaN values detected in the normalized data!\n Filling NaN values with various methods.")
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)    
    
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
        self.cooldown_period = 3
        self.prev_action = None
        self.no_action_counter = 0
        self.no_action_period = 10
        self.portfolio_value = self.cash
        self.prev_portfolio_value = self.portfolio_value
        self.portfolio_difference = 0
        self.current_close = 0
        self.normalized_close = 0
        self.close_difference = 0
        self.invalid_action = False
        self.invalid_action_counter = 0
        self.portfolio_values = [self.prev_portfolio_value]
        self.trades = []
        self.closed_trades = []
        self.metrics = {}
        self.df = TAIndicators(self.initial_data)._calculate_indicators()
        self._normalize_data()

        self._handle_nan_values(self.df)
        self._handle_nan_values(self.initial_data)
        self.data_buffer = deque(self.initial_data.values, maxlen=self.data_buffer.maxlen)
        return self._next_observation()
    
    def _next_observation(self):
        if self.current_step <= self.observation_window_length:
            step_diff = self.observation_window_length - self.current_step
            obs = self.df.iloc[: self.current_step + step_diff].values.flatten()
        else:
            obs = self.df.iloc[self.current_step - self.observation_window_length: self.current_step].values.flatten()       
        return obs

    def _live(self):
        # If in live mode and at the end of the buffer, wait for new data
        if self.live:
            while self.current_step >= len(self.data_buffer) - 1:
                time.sleep(60*30)  # Sleep for 1 second, or adjust as needed
                if self.debug:
                    logger.info(f'Waiting for new data...')
        else:
            pass
    
    def update_current_state(self, new_data):        
        # Define the columns
        columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Convert new_data to a dataframe and specify columns
        new_df = pd.DataFrame([new_data], columns=columns, index=[self.df.index[-1] + pd.Timedelta(minutes=self.timeframe)])
        
        # Append new data to the dataframe
        self.df = pd.concat([self.df, new_df], axis=0)
        
        # Calculate indicators for the updated dataframe
        self.df = TAIndicators(self.df)._calculate_indicators
        
        # Ensure the dataframe length doesn't exceed the observation window length
        if len(self.df) > self.observation_window_length:
            self.df = self.df.iloc[-self.observation_window_length:]
        
        self._normalize_data()
        
        # Handle NaN values
        self._handle_nan_values(self.df)

    def step(self, action):
        self._live()
        self.current_step += 1
        if self.ddqn:
            if 0 <= action <= 0.2:
                action = 0
            elif 0.2 < action <= 0.4:
                action = 1  
            elif 0.4 < action <= 0.6:
                action = 2
            elif 0.6 < action <= 0.8:
                action = 3
            elif 0.8 < action <= 1:
                action = 4
        
        self._take_action(action)
        reward = self._get_reward()        
        
        if self.debug:
            logger.info(f'Step: {self.current_step}, Action: {action}, Reward: {reward:.5f}, Portfolio value: {self.portfolio_value:.5f}\n'
                        f'---------------------- ')
        done = self._done()        
        obs = self._next_observation()        
        info = {}
        if self.current_step >= len(self.df) - 1:
            self.current_step = 0
        return obs, reward, done, info

    def _take_action(self, action):
        
        self.penalty = 0
        self.reward = 0        
        self.invalid_action = False
        # self.close_difference = 0

        if self.current_step < 2:
            return

        # Calculate price difference and update current_close
        self._get_current_prices()

        if self.current_step >= len(self.df):
            return        
                
        # Take action
        try:
            if action == 0:  # Hold
                self._hold_action()
            elif action == 1:  # Buy/Long
                self._open_long_action()
            elif action == 2:  # Sell/Close Long
                self._close_long_action()
            elif action == 3:  # Short
                self._open_short_action()
            elif action == 4:  # Close Short
                self._close_short_action()
            else:
                raise ValueError(f"Invalid action value: {action}")
        except Exception as e:
            logger.error(f"Error while taking action: {e}")
            raise
        
        # Update counters
        if action != 0:           
            self.no_action_counter = 0
        self.prev_action = action        
        if self.invalid_action:
            self.invalid_action_counter += 1
            self.invalid_action = False                        
        else:
            self.invalid_action_counter = 0
        
        
                
    def _efficiency_calculations(self):    
        if len(self.closed_trades) > 3:      
            metrics = self._update_metrics()            
            sqn = metrics["sqn"]
            win_ratio = metrics["win_ratio"]
            # SQN penalties and rewards 
            if 0 < sqn < 0.5:
                self.penalty += (0.5 - sqn) / 4
                if self.debug:
                    logger.info(f"Step {self.current_step}: SQN < 0.5 ({sqn:.4f}), penalty: {self.penalty:.4f}")
            elif sqn < 0:
                self.penalty += 0.2
                if self.debug:
                    logger.info(f"Step {self.current_step}: SQN < 0 ({sqn:.4f}), penalty: {self.penalty:.4f}")
            elif sqn > 1:
                self.reward += (sqn - 1) / 4
                if self.debug:
                    logger.info(f"Step {self.current_step}: SQN > 1.5 ({sqn:.4f}), reward: {self.reward:.4f}")
            
            # Win ratio penalties and rewards     
            if win_ratio < 0.5:
                self.penalty += abs(0.5 - win_ratio)
                if self.debug:
                    logger.info(f"Step {self.current_step}: Win ratio < 0.5 ({win_ratio:.4f}), penalty: {self.penalty:.4f}")
            elif win_ratio > 0.5:
                self.reward += abs(win_ratio - 0.5)
                if self.debug:
                    logger.info(f"Step {self.current_step}: Win ratio > 0.5 ({win_ratio:.4f}), reward: {self.reward:.4f}")                
            
            # Sharpe ratio penalties and rewards
            sharpe_ratio = metrics["sharpe_ratio"]
            if sharpe_ratio < 0:
                self.penalty += abs(sharpe_ratio)
                if self.debug:
                    logger.info(f"Step {self.current_step}: Sharpe ratio < 0 ({sharpe_ratio:.4f}), penalty: {self.penalty:.4f}")
            elif sharpe_ratio > 0:
                self.reward += abs(sharpe_ratio)
                if self.debug:
                    logger.info(f"Step {self.current_step}: Sharpe ratio > 0 ({sharpe_ratio:.4f}), reward: {self.reward:.4f}")
        
        
        if self.debug:
            logger.info(f"Step {self.current_step}: Cash: {self.cash:.2f}, Portfolio: {self.portfolio[0]:.2f}, Close: {self.current_close:.4f}, Portfolio Value: {self.portfolio_value:.2f}")
    
    def _get_reward(self):
        reward = 0
        self._calculate_portfolio_value()
        self._efficiency_calculations()
        if self.penalty > 1.5:
            self.penalty = 1.5
        portfolio_change_ratio = (self.portfolio_difference / self.portfolio_value) if self.portfolio_value > 0 else 0
        close_change_ratio = abs(self.close_difference / self.current_close) if self.current_close > 0 else 0
        # reward_multiplier = abs(portfolio_change_ratio - close_change_ratio) * self.portfolio_value / 5
        reward_multiplier = self.portfolio_value * self.normalized_close

        reward += (self.reward - self.penalty) * reward_multiplier
        self.portfolio_value += reward
        self.prev_portfolio_value = self.portfolio_value
        self.reward = 0
        self.penalty = 0
        if self.debug:
            logger.info(f"Step {self.current_step}: Reward: {reward:.4f}")      
        return reward

    def _calculate_portfolio_value(self):
        self.portfolio_value = self.cash + self.portfolio[0] * self.current_close
        self.portfolio_difference = self.portfolio_value - self.prev_portfolio_value
        self.portfolio_values.append(self.portfolio_value)

    def _get_current_prices(self):
        if self.current_step < self.observation_window_length:
            current_prices = self.initial_data['close'].values[:self.current_step]
        else:
            current_prices = self.initial_data['close'].values[self.current_step - self.observation_window_length:self.current_step]
        price_diff = current_prices[0] - current_prices[1] if len(current_prices) > 1 else 0.01
        price_diff = round(price_diff, 5)
        # self.current_close = self.initial_data['close'].values[self.current_step] #current_prices[0]
        self.current_close = current_prices[0]
        if self.current_close <= 0:
            self.current_close = 0.03
        # self.close_difference = self.initial_data['close_change_pct'].values[self.current_step] * self.current_close
        self.close_difference = price_diff if price_diff > 0 else 0.01
        if self.current_step < self.observation_window_length:
            normalized_prices = self.df['close'].values[:self.current_step]
        else:
            normalized_prices = self.df['close'].values[self.current_step - self.observation_window_length:self.current_step]
        self.normalized_close = normalized_prices[0]
        if self.debug:
            logger.info(f"Step {self.current_step}: Close: {self.current_close:.4f}, Close difference: {self.close_difference:.4f}")
        
        if self.current_close < 0:
            self._done()
            raise Exception

    def _incorrect_action(self):
        self.invalid_action = True
        if self.invalid_action_counter > 2:
            self.penalty += 0.15 * self.invalid_action_counter/2
            if self.debug:
                logger.info(f"Step {self.current_step}: Incorrect action {self.invalid_action_counter} times in a row, penalty: {self.penalty:.2f}")
        else:
            self.penalty += 0.15
            if self.debug:
                logger.info(f"Step {self.current_step}: Incorrect action, penalty: {self.penalty:.2f}")

    def _hold_action(self):
        self.no_action_counter += 1
        self.invalid_action = False
        if self.prev_action == 0:            
            if self.no_action_counter >= self.no_action_period:
                self.no_action_counter = 0
                self.penalty += 0.3
                if self.debug:
                    logger.info(f"Step {self.current_step}:" 
                                f"Incorrect action more than {self.no_action_period} steps," 
                                f"penalty: {self.penalty:.2f}")
            else:
                if self.debug:                            
                    logger.info(f"Step {self.current_step}:" 
                                f"Action is Hold for {self.no_action_counter} steps, No action")
        if self.long_position_open:
            if self.close_difference > 0:
                self.reward += 0.15
                if self.debug:
                    logger.info(f"Step {self.current_step}: Action is Hold, Long position, Price is growing, Reward: {self.reward:.2f}")
            elif self.close_difference < 0:
                self.penalty += 0.05
                if self.debug:
                    logger.info(f"Step {self.current_step}: Action is Hold, Long position, Price is decreasing, Penalty: {self.penalty:.2f}")
        elif self.short_position_open:
            if self.close_difference < 0:
                self.reward += 0.15
                if self.debug:
                    logger.info(f"Step {self.current_step}: Action is Hold, Short position, Price is decreasing, Reward: {self.reward:.2f}")
            elif self.close_difference > 0:
                self.penalty += 0.05
                if self.debug:
                    logger.info(f"Step {self.current_step}: Action is Hold, Short position, Price is growing, Penalty: {self.penalty:.2f}")
        else:    
            if self.debug:                            
                logger.info(f"Step {self.current_step}: Action is Hold, No action")
        
    def _open_long_action(self):
        if not self.long_position_open and not self.short_position_open:
            quantity = self.cash * 0.5 / self.current_close if self.current_close > 1e-10 else 0
            transaction_fee = quantity * self.current_close * self.transaction_cost
            trade_cost = quantity * self.current_close + transaction_fee
            self.cash -= trade_cost
            self.portfolio[0] += quantity
            self.buy_price[0] = self.current_close
            self.trades.append((self.current_step, 'buy'))
            self.long_position_open = True
            self.invalid_action = False
            self.reward += 0.05
            if self.debug:
                logger.info(f"Step {self.current_step}: Opening Long at price {self.current_close:.5f},"
                            f" Cost: {trade_cost:.5f}")        
        else:            
            self._incorrect_action()

    def _open_short_action(self):
        if not self.short_position_open and not self.long_position_open:
            quantity = self.cash * 0.5 / self.current_close if self.current_close > 1e-10 else 0
            transaction_fee = quantity * self.current_close * self.transaction_cost
            self.cash += quantity * self.current_close - transaction_fee
            self.portfolio[0] -= quantity
            self.short_price[0] = self.current_close
            self.trades.append((self.current_step, 'sell'))
            self.invalid_action = False
            self.short_position_open = True
            self.reward += 0.05
            if self.debug:
                logger.info(f"Step {self.current_step}: Opening Short at price {self.current_close:.5f}," 
                            f"Cost: {quantity * self.current_close:.5f}")        
        else:
            self.invalid_action = True
            self._incorrect_action()

    def _close_long_action(self):
        if self.long_position_open:
            profit_or_loss = self.portfolio[0] * (self.current_close - self.buy_price[0])
            trade_cost = self.portfolio[0] * self.current_close
            transaction_fee = trade_cost * self.transaction_cost
            cash_update = trade_cost + profit_or_loss - transaction_fee
            self.cash += cash_update
            self.portfolio[0] = 0
            self.buy_price[0] = 0
            
            self.invalid_action = False
            self.long_position_open = False
            if profit_or_loss > 0:
                self.reward += 0.4
            else:
                self.penalty += 0.25
            self.trades.append((self.current_step, 'close_long'))
            self.closed_trades.append(profit_or_loss)
            if self.debug:
                logger.info(f"Step {self.current_step}:" 
                            f"Closed Long ---- Profit/loss: {profit_or_loss:.5f}")
        else:
            self.invalid_action = True
            self._incorrect_action()
        pass

    def _close_short_action(self):
        if self.short_position_open:            
            profit_or_loss = abs(self.portfolio[0]) * (self.short_price[0] - self.current_close)
            trade_cost = abs(self.portfolio[0]) * self.current_close
            transaction_fee = trade_cost * self.transaction_cost
            cash_update = trade_cost - profit_or_loss + transaction_fee
            self.cash -= cash_update
            self.portfolio[0] = 0
            self.short_price[0] = 0
            self.trades.append((self.current_step, 'close_short'))
            self.closed_trades.append(profit_or_loss)
            self.invalid_action = False
            self.short_position_open = False
            if profit_or_loss > 0:
                self.reward += 0.4
            else:
                self.penalty += 0.25
            if self.debug:
                logger.info(f"Step {self.current_step}:" 
                            f"Closed Short ---- Profit/loss: {profit_or_loss:.5f}")
        else:            
            self._incorrect_action()    
        
    def _update_metrics(self):
        trades_length = len(self.closed_trades)
        if trades_length < 2:
            return 0, 0, 0, 0, 0, 0, 0
        else:
            returns_array = self.closed_trades
            total_trades = trades_length
            winning_trades = np.sum(np.array(self.closed_trades) > 0)
            losing_trades = trades_length - winning_trades

            # Calculate drawdown
            peak = np.maximum.accumulate(self.portfolio_values)
            drawdown = (peak - self.portfolio_value) / peak #if peak > 0 else 0

            # Ratio of winning trades
            win_ratio = winning_trades / total_trades 
            win_perc = win_ratio * 100

            # Calculate SQN        
            variance = np.var(returns_array)
            std_dev = np.sqrt(variance) 
            if std_dev == 0:
                sqn = 0
                sharpe_ratio = 0
            else:
                sqn = np.sqrt(len(returns_array)) * np.mean(returns_array) / std_dev 
                sharpe_ratio = np.mean(returns_array) / std_dev 
            
            if self.debug:
                logger.info(f"Step {self.current_step}: Drawdown: {drawdown[0]:.4f}, Sharpe Ratio: {round(sharpe_ratio, 5)}, SQN: {round(sqn, 5)}\n"
                        f"Total trades: {total_trades}, Win/Loss trades: {winning_trades} / {losing_trades}, Win percent: {win_perc:.0f} %")

            metrics = {
                "drawdown": drawdown,
                "sharpe_ratio": sharpe_ratio,
                "sqn": sqn,
                "win_ratio": win_ratio,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,            
                
            }

            return metrics

    def _done(self):        
        if self.portfolio_value <= 0:
            done = True
        # elif self.current_close <= 0:
        #     done = True
        else:
            done = False
        return done
    
    def render(self, mode='human', close=False):
        # Extract OHLC data up to the current step
        ohlc_data = self.initial_data.iloc[:self.current_step+1]
        
        # Create a new figure and set of subplots only if they don't exist
        if not hasattr(self, 'fig'):
            self.fig, self.axes = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(10, 6))
            self.axes[0].plot(self.portfolio_values, color='blue', label='Portfolio Value')
            self.handles, self.labels = self.axes[0].get_legend_handles_labels()
        
        # Plot candlestick chart
        mpf.plot(ohlc_data, type='candle', ax=self.axes[0], volume=self.axes[1], style='charles')
        
        # Plot trades
        for i, trade in enumerate(self.trades):
            if i >= len(self.handles):  # If the trade hasn't been plotted before
                y_value = self.portfolio_values[trade[0]]  # Get the portfolio value at the trade step
                if trade[1] == 'buy':
                    handle = self.axes[0].scatter(trade[0], y_value, color='green', marker='^', label='Buy')
                elif trade[1] == 'sell':
                    handle = self.axes[0].scatter(trade[0], y_value, color='red', marker='v', label='Sell')
                elif trade[1] == 'close_long':
                    handle = self.axes[0].scatter(trade[0], y_value, color='blue', marker='^', label='Close Long')
                elif trade[1] == 'close_short':
                    handle = self.axes[0].scatter(trade[0], y_value, color='orange', marker='v', label='Close Short')
                self.handles.append(handle)
                self.labels.append(handle.get_label())
        
        # To avoid duplicate labels in the legend
        unique = [(h, l) for i, (h, l) in enumerate(zip(self.handles, self.labels)) if l not in self.labels[:i]]
        self.axes[0].legend(*zip(*unique))
        
        plt.draw()  # Redraw the plot
        plt.pause(0.01)  # Pause to update the plot

    def close(self):
        pass