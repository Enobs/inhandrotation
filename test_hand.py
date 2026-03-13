"""Test hand pose only (no object).

Usage:
    ${ISAACSIM_PYTHON} test_hand.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test hand pose only.")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from tasks.wuji_inhand_rotation.wuji_hand_cfg import WUJI_HAND_GRASP_CFG


# Target joint positions from WUJI_HAND_GRASP_CFG (radians)
GRASP_JOINTS = {
    "right_finger1_joint1": 0.846,
    "right_finger1_joint2": -0.0554,
    "right_finger1_joint3": 0.60476793045,
    "right_finger1_joint4": 0.13439,
    "right_finger2_joint1": 0.9058259,
    "right_finger2_joint2": 0.1867502,
    "right_finger2_joint3": 0.3263766,
    "right_finger2_joint4": 0.413643,
    "right_finger3_joint1": 1.251401,
    "right_finger3_joint2": 0.0,
    "right_finger3_joint3": 0.4066617,
    "right_finger3_joint4": 0.2897247,
    "right_finger4_joint1": 1.176352,
    "right_finger4_joint2": -0.00523599,
    "right_finger4_joint3": 0.715585,
    "right_finger4_joint4": 0.2234021,
    "right_finger5_joint1": 1.363102,
    "right_finger5_joint2": 0.010472,
    "right_finger5_joint3": 1.214749,
    "right_finger5_joint4": 0.1309,
}


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)

    # Spawn hand
    robot_cfg = WUJI_HAND_GRASP_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    hand = Articulation(robot_cfg)

    # Ground + light
    spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    sim.reset()
    hand.reset()

    # Build target position tensor from joint names
    target_pos = torch.zeros((1, hand.num_joints), device=hand.device)
    for i, name in enumerate(hand.joint_names):
        if name in GRASP_JOINTS:
            target_pos[0, i] = GRASP_JOINTS[name]

    # Print joint info
    print("\n" + "=" * 60)
    print("JOINT NAMES AND TARGET POSITIONS:")
    print("=" * 60)
    for i, name in enumerate(hand.joint_names):
        t = target_pos[0, i].item()
        d = hand.data.default_joint_pos[0, i].item()
        print(f"  [{i:2d}] {name:35s}  target={t:+.4f} rad  default={d:+.4f} rad")
    print("=" * 60 + "\n")

    # Write joint state and target
    zero_vel = torch.zeros_like(target_pos)
    hand.write_joint_state_to_sim(target_pos, zero_vel)
    hand.set_joint_position_target(target_pos)

    # Step once to apply
    sim.step()
    hand.update(sim_cfg.dt)

    # Hold pose for 10 seconds
    for step in range(1200):
        hand.write_joint_state_to_sim(target_pos, zero_vel)
        hand.set_joint_position_target(target_pos)
        sim.step()
        hand.update(sim_cfg.dt)

        if step % 120 == 0:
            cur_pos = hand.data.joint_pos[0].cpu()
            print(f"  t={step/120:.0f}s - max error: {(cur_pos - target_pos[0].cpu()).abs().max():.6f} rad")

    print("Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
