import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy


env = gym.make('simple_envs:ladder-v0')

env = DummyVecEnv([lambda: env])  # wrapper for non-vectorized (simple) environments
env = VecMonitor(env)  # provides ep_len_mean and ep_rew_mean metrices

# Mlp: Multi-Layer Perceptron, Dense-Layers
model = PPO('MlpPolicy', env, verbose=1)

# train and save
model.learn(total_timesteps=10000)

evaluate_policy(model, env, n_eval_episodes=10, render=True)