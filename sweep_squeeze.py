"""Sweep squeeze offset to find optimal grip strength.

Gradually increases the PD target squeeze offset from start to end,
so you can visually see the fingers close around the ball.

Usage:
    ${ISAACSIM_PYTHON} sweep_squeeze.py --start 0.0 --end 0.10 --duration 10.0
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep squeeze offset for grasp tuning.")
parser.add_argument("--task", type=str, default="Wuji-InHand-Rotation-Direct-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--start", type=float, default=0.0, help="Starting squeeze offset (rad)")
parser.add_argument("--end", type=float, default=0.10, help="Ending squeeze offset (rad)")
parser.add_argument("--duration", type=float, default=10.0, help="Sweep duration in seconds")

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
    # Long episode so we don't get resets during sweep
    env_cfg.episode_length_s = args_cli.duration + 5.0

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Access the unwrapped env to manipulate grasp_ref_pos directly
    unwrapped = env.unwrapped

    # Store the base joint positions (no squeeze)
    base_pos = unwrapped.hand.data.default_joint_pos[:, unwrapped.actuated_dof_indices].clone()

    # Calculate control frequency
    dt = env_cfg.sim.dt * env_cfg.decimation  # seconds per env step
    total_steps = int(args_cli.duration / dt)

    obs, _ = env.reset()
    print(f"[INFO] Sweeping squeeze offset from {args_cli.start:.3f} to {args_cli.end:.3f} rad")
    print(f"[INFO] Duration: {args_cli.duration}s, dt: {dt:.4f}s, total_steps: {total_steps}")
    print(f"[INFO] Kp values: joint1/2=100, joint3=60, joint4=40")
    print(f"[INFO] Grip force per joint ≈ Kp * offset")
    print()

    for step in range(total_steps + 300):  # extra steps to hold at end
        t = min(step / max(total_steps, 1), 1.0)
        # Linear interpolation of squeeze offset
        offset = args_cli.start + t * (args_cli.end - args_cli.start)

        # Update grasp_ref_pos with current offset
        unwrapped.grasp_ref_pos[:] = base_pos + offset

        # Zero actions (just hold at current grasp_ref_pos)
        actions = torch.zeros(args_cli.num_envs, env.action_space.shape[-1], device="cuda:0")
        obs, rewards, terminated, truncated, infos = env.step(actions)

        if step % 30 == 0:
            # Compute ball distance from init
            dist = torch.norm(
                unwrapped.object_pos_local - unwrapped.object_init_pos_local, dim=-1
            ).mean().item()
            sim_time = step * dt
            print(
                f"t={sim_time:5.1f}s | offset={offset:.4f} rad | "
                f"force≈{100*offset:.1f}/{60*offset:.1f}/{40*offset:.1f} Nm | "
                f"ball_dist={dist:.4f}m | reward={rewards.mean().item():.2f}"
            )

    print("\n[INFO] Sweep complete. Check the visual to find the best squeeze offset.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
