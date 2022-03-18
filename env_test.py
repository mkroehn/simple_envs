# requirements
# pip install stable_baselines3
# pip install tensorboard

import os
from gym_trainsim.envs import TrainSimEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor

env = TrainSimEnv()
# env = Monitor(env) # monitoring for non-vectorized env

# vectorized environments: stacking multiple independent envs into a single env
# https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
env = DummyVecEnv([lambda: env])  # wrapper for non-vectorized (simple) environments
env = VecMonitor(env)  # provides ep_len_mean and ep_rew_mean metrices

# some paths
log_path = os.path.join('Training', 'Logs')
model_path = os.path.join('Training', 'Models', 'PPO_model')

# Mlp: Multi-Layer Perceptron, Dense-Layers
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

# train and save
model.learn(total_timesteps=1000000)
model.save(model_path)

# load
# model = PPO.load(model_path, env=env)

# evaluate model to get metrics like mean reward
evaluate_policy(model, env, n_eval_episodes=10, render=True)

# monitoring with tensorboard
# tensorboard --logdir='./Training/Logs/PPO_1'

# inference
# action = model.predict()


