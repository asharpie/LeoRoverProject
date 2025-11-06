from gymnasium.envs.registration import register

register(
    id="leoroverpybullet/MyEnv",
    entry_point="leoroverpybullet.envs.environment:MyEnv"
)

register(
    id="leoroverpybullet/MyEnv2",
    entry_point="leoroverpybullet.envs.environment2:MyEnv2"
)
