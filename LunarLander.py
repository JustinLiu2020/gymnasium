import gym
from stable_baselines3 import PPO

env = gym.make("LunarLander-v2")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000, progress_bar=True)
model.load("ppo_lunarlander")

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
