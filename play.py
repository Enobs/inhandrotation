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
        agent_dict = agent_cfg.to_dict()
        # Strip keys not accepted by RSL-RL >= 5.0
        for key in ("stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"):
            for section in ("actor", "critic"):
                agent_dict.get(section, {}).pop(key, None)
        runner = OnPolicyRunner(wrapped_env, agent_dict, log_dir="/tmp/play", device="cuda:0")
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")
        print(f"[INFO] Loaded policy from: {args_cli.checkpoint}")
    else:
        print("[INFO] No checkpoint provided. Using random actions.")

    # Rollout — use wrapped_env when running a trained policy
    if policy is not None:
        obs = wrapped_env.get_observations()
        for step in range(args_cli.num_steps):
            actions = policy(obs)
            obs, rewards, dones, infos = wrapped_env.step(actions)

            # Access underlying env for debug info
            base_env = env.unwrapped
            angvel = base_env.object_angvel[0].cpu()
            obj_pos = base_env.object_pos_local[0].cpu()
            action_norm = actions[0].norm().item()

            # Track quaternion to verify if ball actually rotates
            obj_quat = base_env.object_rot[0].cpu()

            # Per-finger action magnitudes (which fingers are the policy moving?)
            finger_action_norms = []
            act = actions[0].cpu()
            for fi in range(5):
                finger_act = act[fi * 4 : (fi + 1) * 4]
                finger_action_norms.append(finger_act.norm().item())

            # Per-finger joint positions (4 joints each)
            finger_joint_str = ""
            hand_pos = base_env.hand_dof_pos[0].cpu()
            for fi in range(5):
                joint_vals = [hand_pos[base_env.actuated_dof_indices[fi * 4 + j]].item() for j in range(4)]
                finger_joint_str += f"  F{fi+1}:[{joint_vals[0]:.2f},{joint_vals[1]:.2f},{joint_vals[2]:.2f},{joint_vals[3]:.2f}]"

            # Fingertip-to-ball distances (contact check)
            fingertip_pos = base_env.hand.data.body_pos_w[0, base_env.finger_bodies, :].cpu()  # (5, 3)
            ball_pos_w = base_env.object.data.root_pos_w[0].cpu()  # (3,)
            tip_dists = (fingertip_pos - ball_pos_w).norm(dim=-1)  # (5,)

            if step % 50 == 0:
                finger_str = "  ".join([f"F{i+1}:{f:.2f}" for i, f in enumerate(finger_action_norms)])
                dist_str = "  ".join([f"F{i+1}:{d.item():.3f}m" for i, d in enumerate(tip_dists)])
                print(
                    f"Step {step:4d}: rew={rewards[0].item():7.3f}  "
                    f"angvel_z={angvel[2].item():6.3f}  "
                    f"|angvel|={angvel.norm().item():6.3f}  "
                    f"obj_z={obj_pos[2].item():.4f}  "
                    f"|action|={action_norm:.3f}  "
                    f"done={dones[0].item()}"
                )
                print(f"         actions: {finger_str}")
                print(f"         joints:{finger_joint_str}")
                print(f"         tip_dist: {dist_str}")
                print(f"         quat=[{obj_quat[0].item():.4f},{obj_quat[1].item():.4f},{obj_quat[2].item():.4f},{obj_quat[3].item():.4f}]")
    else:
        obs, _ = env.reset()
        for step in range(args_cli.num_steps):
            if args_cli.zero_action:
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
