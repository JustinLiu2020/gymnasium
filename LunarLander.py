import gymnasium as gym
from stable_baselines3 import PPO

# Create the LunarLander-v3 environment
env = gym.make("LunarLander-v3", render_mode = "human")

# Create the PPO model with a multilayer perceptron policy
model = PPO("MlpPolicy", env, verbose=1)

# Train the model for 100,000 timesteps
model.learn(total_timesteps=100_000, progress_bar=True)

# Save the model
model.save("ppo_lunarlander")

# Test the trained agent
obs, info = env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
env.close()
