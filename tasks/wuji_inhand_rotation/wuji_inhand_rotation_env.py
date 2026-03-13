"""Wuji In-Hand Rotation environment.

Single-hand in-hand rotation of sphere_small around a specified axis (default: palm Z).
IMCopilot-style atomic skill: stable grasp + continuous rotation.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, saturate

from .wuji_inhand_rotation_env_cfg import WujiInHandRotationEnvCfg


class WujiInHandRotationEnv(DirectRLEnv):
    """In-hand rotation of sphere_small using the Wuji dexterous hand.

    The policy controls 20 finger joints via delta position actions.
    The goal is to rotate the sphere continuously around a target axis
    while maintaining a stable grasp.
    """

    cfg: WujiInHandRotationEnvCfg

    def __init__(self, cfg: WujiInHandRotationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # ---- hand info ----
        self.num_hand_dofs = self.hand.num_joints

        # Actuated joint indices (finger joints only)
        self.actuated_dof_indices = []
        for joint_name in self.cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()
        self.num_actuated = len(self.actuated_dof_indices)

        # Fingertip body indices
        self.finger_bodies = []
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # Joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # ---- buffers ----
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_actuated), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.num_actuated), device=self.device)

        # Reference grasp pose (default + squeeze offset to create grip force)
        # Fingers target slightly more closed than contact → PD error = grip force
        self.grasp_ref_pos = self.hand.data.default_joint_pos[:, self.actuated_dof_indices].clone() + 0.05

        # Object initial position in env-local coordinates (for drop detection)
        # default_root_state is in world frame, so subtract env_origins
        self.object_init_pos_local = (
            self.object.data.default_root_state[:, :3] - self.scene.env_origins
        ).clone()

        # Target rotation axis (normalized, in world frame since hand is fixed)
        axis = torch.tensor(self.cfg.target_rotation_axis, dtype=torch.float32, device=self.device)
        self.target_axis = (axis / axis.norm()).unsqueeze(0).expand(self.num_envs, -1)  # (N, 3)

    # ------------------------------------------------------------------ #
    # Scene setup
    # ------------------------------------------------------------------ #

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)

        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions[:] = self.actions
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        # Absolute position control: grasp_ref + action * scale
        desired = self.grasp_ref_pos + self.actions * self.cfg.action_scale

        # Apply EMA smoothing
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * desired
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )

        # Clamp to joint limits
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # Send position targets to PD controller
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        obs = torch.cat(
            [
                # Hand joint positions (normalized to [-1, 1])
                _unscale(
                    self.hand_dof_pos[:, self.actuated_dof_indices],
                    self.hand_dof_lower_limits[:, self.actuated_dof_indices],
                    self.hand_dof_upper_limits[:, self.actuated_dof_indices],
                ),
                # Hand joint velocities (scaled)
                self.cfg.vel_obs_scale * self.hand_dof_vel[:, self.actuated_dof_indices],
                # Previous actions
                self.prev_actions,
                # Object position relative to palm (env-local)
                self.object_pos_local,
                # Object rotation (quaternion)
                self.object_rot,
                # Object linear velocity
                self.cfg.vel_obs_scale * self.object_linvel,
                # Object angular velocity
                self.cfg.vel_obs_scale * self.object_angvel,
                # Target rotation axis
                self.target_axis,
            ],
            dim=-1,
        )

        return {"policy": obs}

    # ------------------------------------------------------------------ #
    # Rewards
    # ------------------------------------------------------------------ #

    def _get_rewards(self) -> torch.Tensor:
        return _compute_rewards(
            object_angvel=self.object_angvel,
            target_axis=self.target_axis,
            target_angular_velocity=self.cfg.target_angular_velocity,
            object_pos_local=self.object_pos_local,
            object_init_pos_local=self.object_init_pos_local,
            actions=self.actions,
            hand_dof_pos_actuated=self.hand_dof_pos[:, self.actuated_dof_indices],
            grasp_ref_pos=self.grasp_ref_pos,
            hand_dof_vel_actuated=self.hand_dof_vel[:, self.actuated_dof_indices],
            fall_dist=self.cfg.fall_dist,
            # scales
            rew_rotation_scale=self.cfg.rew_rotation_scale,
            rew_non_target_rotation_penalty=self.cfg.rew_non_target_rotation_penalty,
            rew_object_drop_penalty=self.cfg.rew_object_drop_penalty,
            rew_action_penalty=self.cfg.rew_action_penalty,
            rew_pose_deviation_penalty=self.cfg.rew_pose_deviation_penalty,
            rew_joint_vel_penalty=self.cfg.rew_joint_vel_penalty,
        )

    # ------------------------------------------------------------------ #
    # Terminations
    # ------------------------------------------------------------------ #

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # Object dropped: distance from initial pos exceeds threshold
        dist_from_init = torch.norm(self.object_pos_local - self.object_init_pos_local, dim=-1)
        dropped = dist_from_init > self.cfg.fall_dist

        # Lateral drift: object moved too far horizontally from palm center
        lateral_dist = torch.norm(self.object_pos_local[:, :2] - self.object_init_pos_local[:, :2], dim=-1)
        lateral_out = lateral_dist > self.cfg.lateral_dist

        terminated = dropped | lateral_out
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ------------------------------------------------------------------ #
    # Resets
    # ------------------------------------------------------------------ #

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        num_resets = len(env_ids)

        # ---- Reset hand to reference grasp pose + noise ----
        default_dof_pos = self.hand.data.default_joint_pos[env_ids].clone()
        default_dof_vel = torch.zeros_like(self.hand.data.default_joint_vel[env_ids])

        # Add noise only to actuated joints
        noise = sample_uniform(
            -self.cfg.reset_dof_pos_noise,
            self.cfg.reset_dof_pos_noise,
            (num_resets, self.num_hand_dofs),
            device=self.device,
        )
        # Zero noise for non-actuated joints (e.g. fixed tip joints)
        for i in range(self.num_hand_dofs):
            if i not in self.actuated_dof_indices:
                noise[:, i] = 0.0
        dof_pos = default_dof_pos + noise

        # Clamp to joint limits
        dof_pos = torch.max(dof_pos, self.hand_dof_lower_limits[env_ids])
        dof_pos = torch.min(dof_pos, self.hand_dof_upper_limits[env_ids])

        self.hand.write_joint_state_to_sim(dof_pos, default_dof_vel, env_ids=env_ids)
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos

        # ---- Reset object to palm-center + noise ----
        object_default_state = self.object.data.default_root_state[env_ids].clone()
        pos_noise = sample_uniform(
            -self.cfg.reset_object_pos_noise,
            self.cfg.reset_object_pos_noise,
            (num_resets, 3),
            device=self.device,
        )
        object_default_state[:, 0:3] += pos_noise + self.scene.env_origins[env_ids]
        object_default_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        object_default_state[:, 7:] = 0.0  # zero velocity

        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # ---- Reset action buffers ----
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _compute_intermediate_values(self):
        """Pre-compute frequently used quantities."""
        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        self.object_pos_local = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w


# ====================================================================== #
# JIT-compiled helper functions
# ====================================================================== #


@torch.jit.script
def _unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Scale from [lower, upper] to [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower + 1e-8)


@torch.jit.script
def _compute_rewards(
    object_angvel: torch.Tensor,
    target_axis: torch.Tensor,
    target_angular_velocity: float,
    object_pos_local: torch.Tensor,
    object_init_pos_local: torch.Tensor,
    actions: torch.Tensor,
    hand_dof_pos_actuated: torch.Tensor,
    grasp_ref_pos: torch.Tensor,
    hand_dof_vel_actuated: torch.Tensor,
    fall_dist: float,
    rew_rotation_scale: float,
    rew_non_target_rotation_penalty: float,
    rew_object_drop_penalty: float,
    rew_action_penalty: float,
    rew_pose_deviation_penalty: float,
    rew_joint_vel_penalty: float,
) -> torch.Tensor:
    """Compute modular reward for in-hand rotation."""

    # 1. Rotation reward: angular velocity projected onto target axis
    #    Positive if rotating in the correct direction
    angvel_on_axis = torch.sum(object_angvel * target_axis, dim=-1)  # (N,)
    # Reward = how close to desired angular velocity
    rotation_reward = rew_rotation_scale * torch.clamp(angvel_on_axis / (target_angular_velocity + 1e-6), min=-1.0, max=2.0)

    # 2. Non-target rotation penalty: angular velocity off the target axis
    angvel_off_axis = object_angvel - angvel_on_axis.unsqueeze(-1) * target_axis
    non_target_penalty = rew_non_target_rotation_penalty * torch.norm(angvel_off_axis, dim=-1)

    # 3. Object drop penalty
    dist_from_init = torch.norm(object_pos_local - object_init_pos_local, dim=-1)
    drop_penalty = torch.where(
        dist_from_init > fall_dist * 0.8,
        torch.full_like(dist_from_init, rew_object_drop_penalty),
        torch.zeros_like(dist_from_init),
    )

    # 4. Action magnitude penalty (energy regularization)
    action_penalty = rew_action_penalty * torch.sum(actions ** 2, dim=-1)

    # 5. Pose deviation penalty (stay near grasp reference)
    pose_deviation = rew_pose_deviation_penalty * torch.sum(
        (hand_dof_pos_actuated - grasp_ref_pos) ** 2, dim=-1
    )

    # 6. Joint velocity penalty (smoothness)
    joint_vel_penalty = rew_joint_vel_penalty * torch.sum(hand_dof_vel_actuated ** 2, dim=-1)

    total_reward = (
        rotation_reward
        + non_target_penalty
        + drop_penalty
        + action_penalty
        + pose_deviation
        + joint_vel_penalty
    )

    return total_reward
