import gym
from gym import spaces

class TradingEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, 1)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, Box for continuous action
        self.action_space = spaces.Discrete(3)
        # Prices has one row for each OHLCV value
        self.observation_space = spaces.Box(low=0, high=float('inf'), shape=(5,))

    def step(self, action):
        # Execute one time step within the environment
        self.current_step += 1

        # Assume each action is buying, selling or holding
        # Execute action and get reward
        # For example, if action is 0, then you hold, etc.
        
        # You might want to update your portfolio here
        self.portfolio = ...

        # Get the current price from your dataframe
        current_price = self.df.iloc[self.current_step]['Close']

        # Calculate reward
        reward = self.portfolio - current_price

        # Check if we're done
        done = self.current_step >= len(self.df)

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return self.portfolio, reward, done, info

    def reset(self):
        # Reset the state of the environment to an initial state
        self.current_step = 0
        self.portfolio = 100 # Initial portfolio value

        return self.portfolio

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print(f'Step: {self.current_step}, Portfolio Value: {self.portfolio}')
