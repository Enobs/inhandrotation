"""Interactive grasp pose tuning script.

Loads the Wuji hand + sphere without the full RL environment,
so you can quickly iterate on initial joint positions.

Usage:
    # View current grasp pose (static, no actions)
    python tune_grasp.py

    # With Isaac Sim GUI for visual inspection
    python tune_grasp.py --num_envs 1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tune grasp pose interactively.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--duration", type=float, default=30.0, help="How long to run (seconds)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from tasks.wuji_inhand_rotation.wuji_hand_cfg import WUJI_HAND_GRASP_CFG
from tasks.wuji_inhand_rotation.wuji_inhand_rotation_env_cfg import WujiInHandRotationEnvCfg


def main():
    # Minimal sim config
    env_cfg = WujiInHandRotationEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Setup simulation context
    sim = SimulationContext(env_cfg.sim)

    # Build scene
    scene_cfg = env_cfg.scene
    scene = InteractiveScene(scene_cfg)

    # Spawn ground
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

    # Spawn hand
    hand = Articulation(env_cfg.robot_cfg)
    scene.articulations["robot"] = hand

    # Spawn object
    obj = RigidObject(env_cfg.object_cfg)
    scene.rigid_objects["object"] = obj

    # Add light
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # Clone envs
    scene.clone_environments(copy_from_source=False)

    # Reset sim
    sim.reset()

    # Print joint info for reference
    print("\n" + "=" * 60)
    print("JOINT NAMES AND CURRENT POSITIONS:")
    print("=" * 60)
    for i, name in enumerate(hand.joint_names):
        pos = hand.data.default_joint_pos[0, i].item()
        limits = hand.root_physx_view.get_dof_limits()[0, i]
        lo, hi = limits[0].item(), limits[1].item()
        print(f"  [{i:2d}] {name:35s}  pos={pos:+.3f}  limits=[{lo:+.3f}, {hi:+.3f}]")
    print("=" * 60)

    obj_pos = env_cfg.object_cfg.init_state.pos
    print(f"\nObject initial position: {obj_pos}")
    print(f"Hand palm position:     {env_cfg.robot_cfg.init_state.pos}")
    print(f"\nRunning for {args_cli.duration}s — watch the GUI to check grasp stability.")
    print("Edit WUJI_HAND_GRASP_CFG joint_pos values and re-run to iterate.\n")

    # Write initial state
    hand.write_joint_state_to_sim(
        hand.data.default_joint_pos,
        torch.zeros_like(hand.data.default_joint_vel),
    )
    hand.set_joint_position_target(hand.data.default_joint_pos)

    # Run simulation with zero actions (static hold)
    dt = env_cfg.sim.dt
    num_steps = int(args_cli.duration / dt)

    for step in range(num_steps):
        # Keep targeting the default grasp pose (PD controller holds position)
        hand.set_joint_position_target(hand.data.default_joint_pos)

        # Step physics
        sim.step()

        # Update scene
        scene.update(dt)

        # Print object position periodically
        if step % int(1.0 / dt) == 0:  # Every 1 second
            obj_pos = obj.data.root_pos_w[0].cpu()
            obj_vel = obj.data.root_lin_vel_w[0].cpu().norm().item()
            print(f"  t={step*dt:5.1f}s  obj_pos=({obj_pos[0]:.4f}, {obj_pos[1]:.4f}, {obj_pos[2]:.4f})  vel={obj_vel:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
    simulation_app.close()
