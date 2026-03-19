"""Configuration for the Wuji In-Hand Rotation task."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .wuji_hand_cfg import WUJI_HAND_GRASP_CFG

# Paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
SPHERE_URDF_PATH = os.path.join(
    _PROJECT_ROOT, "data_urdf", "object", "contactdb", "sphere_tennis",
    "coacd_decomposed_object_one_link.urdf",
)

# ---- Observation space breakdown ----
# hand joint pos:       20  (5 fingers x 4 joints)
# hand joint vel:       20
# previous actions:     20
# object pos (rel):      3
# object rot (quat):     4
# object lin vel:        3
# object ang vel:        3
# target rot axis:       3
# ---- Total:           76
_OBS_DIM = 76


@configclass
class WujiInHandRotationEnvCfg(DirectRLEnvCfg):
    """Configuration for Wuji in-hand rotation of sphere_small."""

    # ----- env -----
    decimation = 2  # env step = 2 physics steps => 60 Hz control at 120 Hz physics
    episode_length_s = 20.0
    action_space = 20  # 5 fingers x 4 joints each
    observation_space = _OBS_DIM
    state_space = 0  # 0 = symmetric; set >0 for asymmetric actor-critic later

    # ----- simulation -----
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(
            static_friction=2.0,
            dynamic_friction=2.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )

    # ----- scene -----
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=0.75,
        replicate_physics=True,
    )

    # ----- robot (grasp pose for in-hand manipulation) -----
    robot_cfg: ArticulationCfg = WUJI_HAND_GRASP_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # Actuated finger joint names (20 DOF)
    actuated_joint_names = [
        "right_finger1_joint1",
        "right_finger1_joint2",
        "right_finger1_joint3",
        "right_finger1_joint4",
        "right_finger2_joint1",
        "right_finger2_joint2",
        "right_finger2_joint3",
        "right_finger2_joint4",
        "right_finger3_joint1",
        "right_finger3_joint2",
        "right_finger3_joint3",
        "right_finger3_joint4",
        "right_finger4_joint1",
        "right_finger4_joint2",
        "right_finger4_joint3",
        "right_finger4_joint4",
        "right_finger5_joint1",
        "right_finger5_joint2",
        "right_finger5_joint3",
        "right_finger5_joint4",
    ]

    # Fingertip body names for contact proxy
    fingertip_body_names = [
        "right_finger1_tip_link",
        "right_finger2_tip_link",
        "right_finger3_tip_link",
        "right_finger4_tip_link",
        "right_finger5_tip_link",
    ]

    # ----- in-hand object (sphere_small) -----
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=SPHERE_URDF_PATH,
            fix_base=False,
            joint_drive=None,  # sphere has no joints
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=0.5,
                angular_damping=0.5,  # moderate: maintain momentum between finger pushes
                linear_damping=0.1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=100.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # Sphere placed inside the finger cage, above palm center
            # Palm is at z=0.5, fingers extend ~0.09m in +Z
            # With curled fingers, sphere center should be ~0.06-0.08m above palm
            pos=(-0.095, -0.00, 0.56),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # ----- action parameters -----
    # Action in [-1, 1] maps asymmetrically to full URDF joint range:
    #   action=0 → grasp_ref, action=+1 → upper_limit, action=-1 → lower_limit
    act_moving_average = 1  # EMA smoothing on targets (1.0 = no smoothing)

    # ----- observation parameters -----
    vel_obs_scale = 0.2  # scale joint/object velocities in obs

    # ----- target rotation -----
    # Fixed rotation axis in world frame (z-axis = palm normal, fingers direction)
    target_rotation_axis = [0.0, 0.0, -1.0]
    target_angular_velocity = 1.0  # desired rad/s around target axis

    # ----- reset parameters -----
    reset_dof_pos_noise = 0.0  # noise range for finger joints at reset (disabled for debugging)
    reset_object_pos_noise = 0.0  # noise range for object position at reset (disabled for debugging)

    # ----- reward scales (IMCopilot paper: r_rot + r_vel + r_work + r_torq + r_diff) -----
    # r_rot: rotation tracking reward (angular velocity along target axis)
    rew_rotation_scale = 5.0
    # r_vel: object linear velocity penalty (keep object still, only rotating)
    rew_vel_penalty = -0.5
    # r_work: joint work/power penalty = |torque * velocity|
    rew_work_penalty = -0.001
    # r_torq: joint torque penalty
    rew_torque_penalty = -0.0001
    # r_diff: pose deviation from grasp reference
    rew_pose_deviation_penalty = -0.002

    # ----- termination -----
    # Object drop distance threshold (from initial position)
    fall_dist = 0.15
    # Object too far from palm center laterally
    lateral_dist = 0.10
