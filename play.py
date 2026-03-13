"""Play/debug script for Wuji In-Hand Rotation.

Usage:
    # Rollout with random actions (debug/visualize)
    ${ISAACSIM_PYTHON} play.py --task Wuji-InHand-Rotation-Direct-v0 --num_envs 16

    # Play a trained policy
    ${ISAACSIM_PYTHON} play.py --task Wuji-InHand-Rotation-Direct-v0 --num_envs 16 --checkpoint logs/wuji_inhand_rotation/model_5000.pt
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play/debug Wuji In-Hand Rotation.")
parser.add_argument("--task", type=str, default="Wuji-InHand-Rotation-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--num_steps", type=int, default=1000)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained policy checkpoint")
parser.add_argument("--zero_action", action="store_true", default=False, help="Send zero actions (static pose test)")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import tasks  # noqa: F401


def main():
    import importlib

    env_cfg_entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    module_path, class_name = env_cfg_entry.rsplit(":", 1)
    env_cfg_module = importlib.import_module(module_path)
    env_cfg = getattr(env_cfg_module, class_name)()

    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Load policy if checkpoint provided
    policy = None
    if args_cli.checkpoint and os.path.exists(args_cli.checkpoint):
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]
        agent_module_path, agent_class_name = agent_cfg_entry.rsplit(":", 1)
        agent_cfg_module = importlib.import_module(agent_module_path)
        agent_cfg = getattr(agent_cfg_module, agent_class_name)()

        wrapped_env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir="/tmp/play", device="cuda:0")
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")
        print(f"[INFO] Loaded policy from: {args_cli.checkpoint}")
    else:
        print("[INFO] No checkpoint provided. Using random actions.")

    # Rollout
    obs, _ = env.reset()
    for step in range(args_cli.num_steps):
        if policy is not None:
            actions = policy(obs["policy"])
        elif args_cli.zero_action:
            actions = torch.zeros(args_cli.num_envs, env.action_space.shape[-1], device="cuda:0")
        else:
            actions = 2.0 * torch.rand(args_cli.num_envs, env.action_space.shape[-1], device="cuda:0") - 1.0
        obs, rewards, terminated, truncated, infos = env.step(actions)


        if step % 100 == 0:
            print(f"Step {step}: mean_reward={rewards.mean().item():.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
