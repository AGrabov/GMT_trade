import gym
from stable_baselines3 import A2C
from gym_env import TradingEnv

# Create your environment
env = gym.make('TradingEnv')

# Initialize agent
model = A2C('MlpPolicy', env, verbose=1)

# Train agent
model.learn(total_timesteps=10000)

# Save the agent
model.save("a2c_trading")

# To use the trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
      obs = env.reset()
