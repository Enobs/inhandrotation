"""Configuration for the Wuji dexterous hand (based on official config).

Two configurations are provided:
- WUJI_HAND_CFG: default (open hand), for general use
- WUJI_HAND_GRASP_CFG: fingers pre-curled for in-hand manipulation tasks
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.converters.urdf_converter_cfg import UrdfConverterCfg

# Path to the URDF file (resolved relative to this file)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
WUJI_HAND_URDF_PATH = os.path.join(_PROJECT_ROOT, "data_urdf", "robot", "wujihand", "urdf", "wujihand_right.urdf")

# Hand side prefix
_SIDE = "right"

# PD control gains (per-joint tuning)
_KP = {
    f"{_SIDE}_finger(1|2|3|4|5)_joint(1|2)": 100.0,
    f"{_SIDE}_finger(1|2|3|4|5)_joint3": 60.0,
    f"{_SIDE}_finger(1|2|3|4|5)_joint4": 40.0,
}
_KD = {
    f"{_SIDE}_finger.*_joint(1|2)": 1.0,
    f"{_SIDE}_finger.*_joint(3|4)": 0.5,
}

# Torque limits (Nm)
_EFFORT_LIMITS = {
    f"{_SIDE}_finger(1|2|3|4|5)_joint(1|2)": 20.0,
    f"{_SIDE}_finger(1|2|3|4|5)_joint3": 10.0,
    f"{_SIDE}_finger(1|2|3|4|5)_joint4": 5.0,
}

# Shared spawn config
_SPAWN_CFG = sim_utils.UrdfFileCfg(
    asset_path=WUJI_HAND_URDF_PATH,
    usd_dir=os.path.join(_THIS_DIR, "usd"),
    usd_file_name="wujihand",
    force_usd_conversion=True,  # Force reconversion after config changes
    fix_base=True,
    root_link_name=f"{_SIDE}_palm_link",
    link_density=1,
    collider_type="convex_hull",
    merge_fixed_joints=False,  # Keep tip links as separate bodies
    self_collision=False,  # Disable self-collision to test stability
    activate_contact_sensors=True,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="position",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=_KP,
            damping=_KD,
        ),
    ),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    ),
    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        solver_position_iteration_count=20,
        solver_velocity_iteration_count=10,
    ),
)

# Shared actuators config
_ACTUATORS = {
    "fingers": ImplicitActuatorCfg(
        joint_names_expr=[f"{_SIDE}_finger.*_joint.*"],
        effort_limit_sim=_EFFORT_LIMITS,
        stiffness=_KP,
        damping=_KD,
    ),
}

##
# Default configuration (open hand)
##

WUJI_HAND_CFG = ArticulationCfg(
    spawn=_SPAWN_CFG,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            f"{_SIDE}_finger.*_joint1": 0.08,
            f"{_SIDE}_finger.*_joint(2|3|4)": 0.0,
        },
    ),
    actuators=_ACTUATORS,
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Wuji Hand robot (open hand)."""

##
# Grasp configuration (fingers curled for in-hand manipulation)
##

WUJI_HAND_GRASP_CFG = ArticulationCfg(
    spawn=_SPAWN_CFG,
    init_state=ArticulationCfg.InitialStateCfg(
        # Palm at z=0.5, facing up (fingers point +Z), identity quaternion
        pos=(0.0, 0.0, 0.5),
        rot=(0.707, 0, -0.707, 0),
        joint_pos={
            # Thumb (finger1): curled inward to oppose other fingers
            f"{_SIDE}_finger1_joint1": 0.8464847,
            f"{_SIDE}_finger1_joint2": -0.0554,
            f"{_SIDE}_finger1_joint3": 0.90,
            f"{_SIDE}_finger1_joint4": 0.20,
            
            # Fingers 2: curled to form a cage around the sphere
            f"{_SIDE}_finger2_joint1": 0.9058259,
            f"{_SIDE}_finger2_joint2": 0.1867502,
            f"{_SIDE}_finger2_joint3": 0.25,
            f"{_SIDE}_finger2_joint4": 0.33,
            
            # Fingers 3: curled to form a cage around the sphere
            f"{_SIDE}_finger3_joint1": 1.251401,
            f"{_SIDE}_finger3_joint2": 0.0,
            f"{_SIDE}_finger3_joint3": 0.05,
            f"{_SIDE}_finger3_joint4": 0.0,
            
            # Fingers 4: curled to form a cage around the sphere
            f"{_SIDE}_finger4_joint1": 1.176352,
            f"{_SIDE}_finger4_joint2": -0.00523599,
            f"{_SIDE}_finger4_joint3": 0.30,
            f"{_SIDE}_finger4_joint4": 0.0,
            
            # Fingers 5: curled to form a cage around the sphere
            f"{_SIDE}_finger5_joint1": 1.363102,
            f"{_SIDE}_finger5_joint2": 0.010472,
            f"{_SIDE}_finger5_joint3": 1.00,
            f"{_SIDE}_finger5_joint4": 0.10,
        },
    ),
    actuators=_ACTUATORS,
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Wuji Hand robot (grasp pose for in-hand manipulation)."""

##
# Grasp target joint positions (tighter than init — PD drives toward these to grip the ball)
# These are the original GUI-tuned values where fingers contact the ball.
##
WUJI_GRASP_TARGET_JOINT_POS = {
    f"{_SIDE}_finger1_joint1": 0.8464847,
    f"{_SIDE}_finger1_joint2": -0.0554,
    f"{_SIDE}_finger1_joint3": 0.90,
    f"{_SIDE}_finger1_joint4": 0.20,
    f"{_SIDE}_finger2_joint1": 1.0058259,
    f"{_SIDE}_finger2_joint2": 0.1867502,
    f"{_SIDE}_finger2_joint3": 0.25,
    f"{_SIDE}_finger2_joint4": 0.33,
    f"{_SIDE}_finger3_joint1": 1.251401,
    f"{_SIDE}_finger3_joint2": 0.0,
    f"{_SIDE}_finger3_joint3": 0.33,
    f"{_SIDE}_finger3_joint4": 0.21,
    f"{_SIDE}_finger4_joint1": 1.176352,
    f"{_SIDE}_finger4_joint2": -0.00523599,
    f"{_SIDE}_finger4_joint3": 0.64,
    f"{_SIDE}_finger4_joint4": 0.15,
    f"{_SIDE}_finger5_joint1": 1.363102,
    f"{_SIDE}_finger5_joint2": 0.010472,
    f"{_SIDE}_finger5_joint3": 1.15,
    f"{_SIDE}_finger5_joint4": 0.20,
}
"""Grasp target positions: original tight values for PD grip force."""
