"""Training script for Wuji In-Hand Rotation.

Usage:
    # Train with default settings (4096 envs, headless)
    ${ISAACSIM_PYTHON} train.py --task Wuji-InHand-Rotation-Direct-v0 --num_envs 4096 --headless

    # Train with fewer envs for debugging
    ${ISAACSIM_PYTHON} train.py --task Wuji-InHand-Rotation-Direct-v0 --num_envs 64

    # Resume training from checkpoint
    ${ISAACSIM_PYTHON} train.py --task Wuji-InHand-Rotation-Direct-v0 --num_envs 4096 --headless --resume
"""

from __future__ import annotations

import argparse
import os
import sys

# Add project root to path so `tasks` package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

# ---- CLI ----
parser = argparse.ArgumentParser(description="Train Wuji In-Hand Rotation with RSL-RL PPO.")
parser.add_argument("--task", type=str, default="Wuji-InHand-Rotation-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--resume", action="store_true", default=False)
parser.add_argument("--load_run", type=str, default=None, help="Run folder to resume from")
parser.add_argument("--load_checkpoint", type=str, default=None, help="Specific checkpoint to load")
parser.add_argument("--log_dir", type=str, default=None)

# AppLauncher adds --headless, --device, --livestream, etc.
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- Imports after app launch (Isaac Sim requirement) ----
import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnvCfg

# Register our custom task
import tasks  # noqa: F401

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

# RSL-RL runner
from rsl_rl.runners import OnPolicyRunner


def main():
    # ---- Create environment ----
    env_cfg_entry = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    agent_cfg_entry = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]

    # Import config classes dynamically
    module_path, class_name = env_cfg_entry.rsplit(":", 1)
    import importlib
    env_cfg_module = importlib.import_module(module_path)
    env_cfg: DirectRLEnvCfg = getattr(env_cfg_module, class_name)()

    agent_module_path, agent_class_name = agent_cfg_entry.rsplit(":", 1)
    agent_cfg_module = importlib.import_module(agent_module_path)
    agent_cfg: RslRlOnPolicyRunnerCfg = getattr(agent_cfg_module, agent_class_name)()

    # Override with CLI args
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    # Log directory
    log_root = args_cli.log_dir or os.path.join(os.path.dirname(__file__), "logs")
    log_dir = os.path.join(log_root, agent_cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)

    # Create env
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # ---- Create runner ----
    # Build config dict and strip deprecated fields that rsl-rl >= 5.0 doesn't accept
    train_cfg = agent_cfg.to_dict()
    _DEPRECATED = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    for key in ("actor", "critic"):
        if key in train_cfg:
            for dep in _DEPRECATED:
                train_cfg[key].pop(dep, None)

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=agent_cfg.device)

    # Resume if requested
    if args_cli.resume:
        resume_path = None
        if args_cli.load_checkpoint:
            resume_path = args_cli.load_checkpoint
        elif args_cli.load_run:
            resume_dir = os.path.join(log_dir, args_cli.load_run)
            # Find latest checkpoint
            checkpoints = sorted(
                [f for f in os.listdir(resume_dir) if f.endswith(".pt")],
                key=lambda x: int(x.split("_")[-1].split(".")[0]) if x.split("_")[-1].split(".")[0].isdigit() else 0,
            )
            if checkpoints:
                resume_path = os.path.join(resume_dir, checkpoints[-1])
        if resume_path and os.path.exists(resume_path):
            print(f"[INFO] Resuming from: {resume_path}")
            runner.load(resume_path)

    # ---- Train ----
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
