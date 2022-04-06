import os

import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy
# from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

# from simple_envs.envs import LadderEnv, TrainSimEnv, FtSimEnv
# from simple_envs.envs.ftsim_env_v1 import FtSimEnvSimple
from simple_envs.envs.ftsim_env_v2 import FtSimEnvV2

config = {
    "project_name": "FTM Debugging",
    "policy_type": "MlpPolicy",
    "total_timesteps": 300000,
    "env_name": "FtSimEnvV2",
    "env_path": "./simple_envs/envs/ftsim_env_v2.py",
}


def train_environment(env: gym.Env, config):
    # vectorized environments: stacking multiple independent envs into a single env
    # https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html
    env = DummyVecEnv([lambda: env])  # wrapper for non-vectorized (simple) environments
    env = VecMonitor(env)  # provides ep_len_mean and ep_rew_mean metrices

    # some paths
    log_path = os.path.join('Training', 'Logs')
    model_path = os.path.join('Training', 'Models', 'PPO_model')

    # Mlp: Multi-Layer Perceptron, Dense-Layers
    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=log_path)

    # train and save
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_freq=1000,
            model_save_path=model_path,
        ))
    model.save(model_path)

    # load
    # model = PPO.load(model_path, env=env)

    # evaluate model to get metrics like mean reward
    evaluate_policy(model, env, n_eval_episodes=1, render=True)

    # monitoring with tensorboard
    # tensorboard --logdir='./Training/Logs/PPO_1'

    # inference
    # action = model.predict()


def training():
    wandb.login(key="ec243113b86b781113ee94276adfc3ceebc1b0ac")

    run = wandb.init(
        project=config["project_name"],
        config=config,
        sync_tensorboard=True,
        monitor_gym=False,
        save_code=True,
    )

    env = FtSimEnvV2()
    # env = Monitor(env) # monitoring for non-vectorized env
    train_environment(env, config)

    wandb.save(config["env_path"])

    run.finish()


def inference():
    model_path = os.path.join('Training', 'Models', 'PPO_model', 'model.zip')

    env = FtSimEnvV2()

    model = PPO.load(model_path, env=env)
    evaluate_policy(model, env, n_eval_episodes=1, render=True)


training()
# inference()
