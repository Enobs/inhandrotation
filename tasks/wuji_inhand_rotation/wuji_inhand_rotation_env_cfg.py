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

# ---- Actor observation space (proprioception + privileged object state) ----
# hand joint pos:       20  (5 fingers x 4 joints)
# hand joint vel:       20
# previous actions:     20
# object pos (rel):      3  (privileged — will be distilled out for student)
# object rot (quat):     4  (privileged)
# object lin vel:        3  (privileged)
# object ang vel:        3  (privileged)
# target rot axis:       3
# ---- Total:           76
_OBS_DIM = 76

# ---- Critic state space (actor obs + DR privileged info) ----
# actor obs:            76
# object mass:           1
# object friction:       1
# object CoM offset:     3
# PD gain scale Kp:      1  (mean across joints)
# PD gain scale Kd:      1  (mean across joints)
# ---- Total:           83
_STATE_DIM = 83


@configclass
class WujiInHandRotationEnvCfg(DirectRLEnvCfg):
    """Configuration for Wuji in-hand rotation of sphere_small."""

    # ----- env -----
    decimation = 2  # env step = 2 physics steps => 60 Hz control at 120 Hz physics
    episode_length_s = 20.0
    action_space = 20  # 5 fingers x 4 joints each
    observation_space = _OBS_DIM
    state_space = _STATE_DIM  # asymmetric actor-critic: critic gets privileged DR info

    # ----- observation history -----
    num_obs_history = 3  # number of proprioception history steps for future student distillation

    # ----- simulation -----
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.5,
            dynamic_friction=1.5,
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
    # Delta action scale (Sharpa-style): each step target += action_scale * action
    # Sharpa uses 1/24 ≈ 0.0417 with 60 Hz control
    action_scale = 1.0 / 24.0

    # ----- observation parameters -----
    vel_obs_scale = 0.2  # scale joint/object velocities in obs

    # ----- target rotation -----
    # Rotation axis in hand LOCAL frame (-X = palm normal inward direction)
    # This gets transformed to world frame based on hand orientation at reset
    target_rotation_axis_local = [-1.0, 0.0, 0.0]
    target_angular_velocity = 1.0  # desired rad/s around target axis

    # ----- reset parameters -----
    reset_dof_pos_noise = 0.05  # noise range for finger joints at reset (~2.9 degrees)
    reset_object_pos_noise = 0.005  # noise range for object position at reset (5mm)
    reset_hand_rot_noise = 3.14159  # full rotation range — grasp holds at any orientation
    # ----- domain randomization (IMCopilot: scale, mass, friction, CoM, gravity, PD gains) -----
    # DR: object mass range [min, max] in kg (default ~0.0165 from density=100)
    object_mass_range = (0.01, 0.2)  # 10g to 200g
    # DR: object scale range (uniform multiplier on mesh scale)
    # NOTE: USD-level operation, slow per-prim — disable for now, enable for final training
    object_scale_range = None  # (0.8, 1.2)
    # DR: object friction range (static & dynamic friction)
    object_friction_range = (0.5, 2.0)
    # DR: object center-of-mass offset range (m, uniform per axis)
    object_com_offset_range = (-0.01, 0.01)
    # DR: gravity magnitude range (m/s^2, nominal = 9.81)
    gravity_range = None  # disabled in favor of gravity_curriculum below
    # DR: PD gain scale range (multiplier on nominal Kp/Kd) — Sharpa: [0.5, 2.0]
    pd_gain_scale_range = (0.5, 2.0)
    # Joint observation noise stddev (Sharpa: 0.02 on normalized joint pos)
    joint_obs_noise = 0.02

    # ----- reward scales (Sharpa-style: sharpa_wave_env_cfg.py:269-276) -----
    angvel_clip_min = -0.5
    angvel_clip_max = 0.5
    rew_rotation_scale = 2.5         # r_rot: clamped angvel on target axis
    rew_linvel_penalty_scale = -0.3  # r_vel: L1 |Δpos|/dt penalty
    rew_pos_diff_scale = -0.4        # r_diff: pose deviation from default
    rew_torque_scale = -0.00001      # r_torq: sum(τ²) — our Kp>>Sharpa's, scale down
    rew_work_scale = -0.0000001      # r_work: (Σ τ·v)² — target ~-0.15 per step
    rew_object_pos_scale = 0.003     # r_pos: 1/(|pos - init_pos|+0.001) bonus

    # ----- random external forces on object (sharpa_wave_env.py:189-196) -----
    force_scale = 2.0                # external force magnitude scale
    random_force_prob_scalar = 0.25  # probability of applying force per step per env
    force_decay = 0.9                # exponential decay factor
    force_decay_interval = 0.08      # decay time interval (s)

    # ----- gravity curriculum (sharpa_wave_env.py:259-265) -----
    gravity_curriculum = True        # start near-zero g, increase as policy learns
    gravity_curriculum_init = 0.05   # m/s^2, initial gravity magnitude (Z down)
    gravity_curriculum_step = 0.1    # m/s^2 increment per success threshold (faster)
    gravity_curriculum_max = 9.81    # final gravity

    # ----- termination -----
    # Object drop distance threshold (from initial position)
    fall_dist = 0.15
    # Object too far from palm center laterally
    lateral_dist = 0.10
