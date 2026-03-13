"""Wuji In-Hand Rotation task — gym registration."""

import gymnasium as gym

from . import agents  # noqa: F401

gym.register(
    id="Wuji-InHand-Rotation-Direct-v0",
    entry_point="tasks.wuji_inhand_rotation.wuji_inhand_rotation_env:WujiInHandRotationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "tasks.wuji_inhand_rotation.wuji_inhand_rotation_env_cfg:WujiInHandRotationEnvCfg",
        "rsl_rl_cfg_entry_point": "tasks.wuji_inhand_rotation.agents.rsl_rl_ppo_cfg:WujiInHandRotationPPORunnerCfg",
    },
)
