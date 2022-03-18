from gym.envs.registration import register

register(
    id='ladder-v0',
    entry_point='simple_envs.envs:LadderEnv',
)
register(
    id='trainsim-v0',
    entry_point='simple_envs.envs:TrainSimEnv',
)
