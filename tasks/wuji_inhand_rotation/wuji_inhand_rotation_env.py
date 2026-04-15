"""Wuji In-Hand Rotation environment.

Single-hand in-hand rotation of sphere_small around a specified axis (default: palm Z).
IMCopilot-style atomic skill: stable grasp + continuous rotation.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, saturate, quat_mul, quat_apply, quat_from_angle_axis

from .wuji_hand_cfg import WUJI_GRASP_TARGET_JOINT_POS
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
        self.raw_actions = torch.zeros((self.num_envs, self.num_actuated), device=self.device)

        # Reference grasp pose: two separate poses
        # - grasp_base_pos = init_state (open, no penetration)
        # - grasp_ref_pos  = original tight values (PD target for grip force)
        # Warmup ramps from base → ref over first N steps to avoid penetration
        self.squeeze_warmup_steps = 30  # ramp over this many env steps after reset
        self.grasp_base_pos = self.hand.data.default_joint_pos[:, self.actuated_dof_indices].clone()

        # Build grasp_ref_pos from the separate target dict
        grasp_target = self.grasp_base_pos.clone()
        for joint_name, target_val in WUJI_GRASP_TARGET_JOINT_POS.items():
            if joint_name in self.hand.joint_names:
                joint_idx = self.hand.joint_names.index(joint_name)
                if joint_idx in self.actuated_dof_indices:
                    local_idx = self.actuated_dof_indices.index(joint_idx)
                    grasp_target[:, local_idx] = target_val
        self.grasp_ref_pos = grasp_target

        # Object initial position in env-local coordinates (for drop detection)
        # default_root_state is already in local (env) frame, NOT world frame
        self.object_init_pos_local = self.object.data.default_root_state[:, :3].clone()

        # Build per-actuated-joint Kp and Kd vectors for torque estimation
        # (implicit actuator: PhysX handles PD internally, so we estimate torque = Kp*pos_err - Kd*vel)
        from .wuji_hand_cfg import _KP, _KD
        self.kp_vec = torch.zeros(self.num_actuated, device=self.device)
        self.kd_vec = torch.zeros(self.num_actuated, device=self.device)
        for local_idx, joint_idx in enumerate(self.actuated_dof_indices):
            joint_name = self.hand.joint_names[joint_idx]
            for pattern, kp_val in _KP.items():
                if re.match(pattern, joint_name):
                    self.kp_vec[local_idx] = kp_val
                    break
            for pattern, kd_val in _KD.items():
                if re.match(pattern, joint_name):
                    self.kd_vec[local_idx] = kd_val
                    break
        # Expand to (1, num_actuated) for broadcasting
        self.kp_vec = self.kp_vec.unsqueeze(0)
        self.kd_vec = self.kd_vec.unsqueeze(0)

        # Store default PD gains for DR (will be scaled per-reset)
        self.default_kp = self.hand.root_physx_view.get_dof_stiffnesses().clone()  # (N, num_dofs)
        self.default_kd = self.hand.root_physx_view.get_dof_dampings().clone()     # (N, num_dofs)

        # Store default object COM for DR offset
        self.default_object_com = self.object.root_physx_view.get_coms().clone()  # (N, 1, 7) or similar

        # ---- Privileged info buffers (for asymmetric critic) ----
        self.current_object_mass = torch.zeros(self.num_envs, 1, device=self.device)
        self.current_object_friction = torch.zeros(self.num_envs, 1, device=self.device)
        self.current_com_offset = torch.zeros(self.num_envs, 3, device=self.device)
        self.current_kp_scale = torch.ones(self.num_envs, 1, device=self.device)
        self.current_kd_scale = torch.ones(self.num_envs, 1, device=self.device)

        # ---- Proprioception history buffer (for future student distillation) ----
        # Each history step: joint_pos(20) + joint_vel(20) = 40
        self._prop_dim = self.num_actuated * 2  # pos + vel = 40
        self.obs_history = torch.zeros(
            self.num_envs, self.cfg.num_obs_history, self._prop_dim, device=self.device
        )  # (N, H, 40)

        # Base hand orientation and position (from config, before randomization)
        self.base_hand_quat = torch.tensor(
            self.cfg.robot_cfg.init_state.rot, dtype=torch.float32, device=self.device
        )  # (4,) in (w, x, y, z)
        self.base_hand_pos = torch.tensor(
            self.cfg.robot_cfg.init_state.pos, dtype=torch.float32, device=self.device
        )  # (3,)

        # Ball position in hand LOCAL frame (constant)
        # Computed from original palm-up config: hand=(0,0,0.5) rot=(0.707,0,-0.707,0), ball=(-0.095,0,0.56)
        # local X = 0.06m (above palm), local Z = 0.095m (along fingers)
        self.ball_local_offset = torch.tensor([0.06, 0.0, 0.095], dtype=torch.float32, device=self.device)

        # Per-env hand quaternion (updated at reset with randomization)
        self.hand_quat = self.base_hand_quat.unsqueeze(0).expand(self.num_envs, -1).clone()  # (N, 4)

        # Target rotation axis in hand LOCAL frame: palm normal is local X, rotation around -X
        self.target_axis_local = torch.tensor(
            self.cfg.target_rotation_axis_local, dtype=torch.float32, device=self.device
        )
        self.target_axis_local = self.target_axis_local / self.target_axis_local.norm()

        # Per-env target axis in world frame (recomputed at reset from hand orientation)
        self.target_axis = quat_apply(
            self.hand_quat, self.target_axis_local.unsqueeze(0).expand(self.num_envs, -1)
        )  # (N, 3)

        # Previous quaternion buffer for computing real angular velocity from pose changes
        # (PhysX root_ang_vel_w is unreliable for over-constrained contacts)
        self.prev_object_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.prev_object_quat[:, 0] = 1.0  # identity quaternion (w, x, y, z)

        # Sharpa-style reward buffers
        self.object_pos_prev = torch.zeros((self.num_envs, 3), device=self.device)
        self.object_default_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        # External force buffer (applied to object root, world frame)
        self.rb_forces = torch.zeros((self.num_envs, 3), device=self.device)

        # Initialize gravity curriculum (set low initial gravity)
        if self.cfg.gravity_curriculum:
            import isaaclab.sim as sim_utils_init
            import carb
            self.physics_sim_view = sim_utils_init.SimulationContext.instance().physics_sim_view
            init_g = -float(self.cfg.gravity_curriculum_init)
            self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, init_g))
            print(f"[gravity_curriculum] initial gravity set to {init_g:.3f} m/s^2")

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
        # Snapshot object pose BEFORE physics advances
        self.prev_object_quat[:] = self.object.data.root_quat_w
        self.object_pos_prev[:] = self.object.data.root_pos_w

        # Random external forces on object (Sharpa: sharpa_wave_env.py:189-196)
        if self.cfg.force_scale > 0.0:
            decay = self.cfg.force_decay ** (self.cfg.sim.dt * self.cfg.decimation / self.cfg.force_decay_interval)
            self.rb_forces *= decay
            obj_mass = self.object.root_physx_view.get_masses().reshape(self.num_envs).to(self.device)
            prob = self.cfg.random_force_prob_scalar
            apply_mask = torch.rand(self.num_envs, device=self.device) < prob
            new_forces = torch.randn(self.num_envs, 3, device=self.device) * obj_mass.unsqueeze(-1) * self.cfg.force_scale
            self.rb_forces = torch.where(apply_mask.unsqueeze(-1), new_forces, self.rb_forces)
            self.object.set_external_force_and_torque(
                forces=self.rb_forces.unsqueeze(1),
                torques=torch.zeros(self.num_envs, 1, 3, device=self.device),
            )

    def _apply_action(self) -> None:
        # Warmup: ramp from open base pose to tight grasp pose over the first N steps.
        # During warmup we OVERWRITE prev_targets to the ramped pose so the policy delta
        # is added on top of the gradually closing grasp (avoids penetration shock).
        warmup_frac = torch.clamp(
            self.episode_length_buf.float() / self.squeeze_warmup_steps, 0.0, 1.0
        ).unsqueeze(-1)  # (N, 1)
        ramped_grasp = self.grasp_base_pos + (self.grasp_ref_pos - self.grasp_base_pos) * warmup_frac
        in_warmup = (warmup_frac < 1.0)  # (N, 1)
        # During warmup: prev_targets follows the ramp; after warmup: pure delta accumulation.
        self.prev_targets[:, self.actuated_dof_indices] = torch.where(
            in_warmup,
            ramped_grasp,
            self.prev_targets[:, self.actuated_dof_indices],
        )

        # Delta action: target = prev_target + action_scale * action  (Sharpa-style)
        delta = self.cfg.action_scale * self.actions
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.prev_targets[:, self.actuated_dof_indices] + delta,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        # Update prev_targets for next step's accumulation
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # Send absolute position targets to PD controller
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices],
            joint_ids=self.actuated_dof_indices,
        )

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()

        # -- Proprioception (available on real hardware) --
        norm_joint_pos = _unscale(
            self.hand_dof_pos[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        # Sharpa-style joint observation noise (sharpa_wave_env.py: noise_scale=0.02)
        if self.cfg.joint_obs_noise > 0.0:
            norm_joint_pos = norm_joint_pos + torch.randn_like(norm_joint_pos) * self.cfg.joint_obs_noise
        scaled_joint_vel = self.cfg.vel_obs_scale * self.hand_dof_vel[:, self.actuated_dof_indices]

        # Update proprioception history: shift left, append current
        current_prop = torch.cat([norm_joint_pos, scaled_joint_vel], dim=-1)  # (N, 40)
        self.obs_history[:, :-1] = self.obs_history[:, 1:].clone()
        self.obs_history[:, -1] = current_prop

        # -- Actor observation (proprioception + privileged object state) --
        obs = torch.cat(
            [
                norm_joint_pos,                                    # 20
                scaled_joint_vel,                                  # 20
                self.prev_actions,                                 # 20
                self.object_pos_local,                             # 3  (privileged)
                self.object_rot,                                   # 4  (privileged)
                self.cfg.vel_obs_scale * self.object_linvel,       # 3  (privileged)
                self.cfg.vel_obs_scale * self.object_angvel,       # 3  (privileged)
                self.target_axis,                                  # 3
            ],
            dim=-1,
        )  # Total: 76

        # -- Critic state (actor obs + DR privileged info) --
        critic_state = torch.cat(
            [
                obs,                          # 76
                self.current_object_mass,     # 1
                self.current_object_friction, # 1
                self.current_com_offset,      # 3
                self.current_kp_scale,        # 1
                self.current_kd_scale,        # 1
            ],
            dim=-1,
        )  # Total: 83

        return {"policy": obs, "critic": critic_state}

    # ------------------------------------------------------------------ #
    # Rewards
    # ------------------------------------------------------------------ #

    def _get_rewards(self) -> torch.Tensor:
        # Estimate PD torque: τ = Kp*(target - pos) - Kd*vel
        pos_error = self.cur_targets[:, self.actuated_dof_indices] - self.hand_dof_pos[:, self.actuated_dof_indices]
        vel = self.hand_dof_vel[:, self.actuated_dof_indices]
        estimated_torque = self.kp_vec * pos_error - self.kd_vec * vel

        step_dt = self.cfg.sim.dt * self.cfg.decimation
        default_dof_pos_actuated = self.hand.data.default_joint_pos[:, self.actuated_dof_indices]
        total, components = _compute_rewards(
            object_angvel=self.object_angvel,
            object_pos=self.object.data.root_pos_w,
            object_pos_prev=self.object_pos_prev,
            object_default_pos=self.object_default_pos_w,
            target_axis=self.target_axis,
            angvel_clip_min=self.cfg.angvel_clip_min,
            angvel_clip_max=self.cfg.angvel_clip_max,
            hand_dof_pos_actuated=self.hand_dof_pos[:, self.actuated_dof_indices],
            default_dof_pos_actuated=default_dof_pos_actuated,
            hand_dof_vel_actuated=vel,
            estimated_torque=estimated_torque,
            step_dt=step_dt,
            rew_rotation_scale=self.cfg.rew_rotation_scale,
            rew_linvel_penalty_scale=self.cfg.rew_linvel_penalty_scale,
            rew_pos_diff_scale=self.cfg.rew_pos_diff_scale,
            rew_torque_scale=self.cfg.rew_torque_scale,
            rew_work_scale=self.cfg.rew_work_scale,
            rew_object_pos_scale=self.cfg.rew_object_pos_scale,
        )
        # Log reward components to extras for tensorboard
        self.extras["log"] = {
            "rew_rotation": components[0].mean(),
            "rew_linvel_penalty": components[1].mean(),
            "rew_pos_diff": components[2].mean(),
            "rew_torque_penalty": components[3].mean(),
            "rew_work_penalty": components[4].mean(),
            "rew_object_pos_bonus": components[5].mean(),
            "angvel_on_axis": components[6].mean(),
        }
        return total

    # ------------------------------------------------------------------ #
    # Terminations
    # ------------------------------------------------------------------ #

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # Gravity curriculum: increase gravity once policy can hold the ball
        if self.cfg.gravity_curriculum and hasattr(self, "physics_sim_view") and self.common_step_counter > 1000:
            # Use drop rate as success signal: if drops are rare, increase gravity
            dist = torch.norm(self.object_pos_local - self.object_init_pos_local, dim=-1)
            drop_rate = (dist > self.cfg.fall_dist).float().mean().item()
            if drop_rate < 1e-2:
                g = self.physics_sim_view.get_gravity()
                g_amp = (g[0] ** 2 + g[1] ** 2 + g[2] ** 2) ** 0.5
                if g_amp < self.cfg.gravity_curriculum_max:
                    new_g_z = -(g_amp + self.cfg.gravity_curriculum_step)
                    import carb
                    self.physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, new_g_z))
                    print(f"[gravity_curriculum] gravity → {new_g_z:.3f} m/s^2")

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

        # Set PD target to base grasp pos (NO squeeze yet — warmup ramps it in)
        grasp_targets = dof_pos.clone()
        grasp_targets[:, self.actuated_dof_indices] = self.grasp_base_pos[env_ids]
        self.hand.set_joint_position_target(grasp_targets, env_ids=env_ids)
        self.prev_targets[env_ids] = grasp_targets
        self.cur_targets[env_ids] = grasp_targets

        # ---- Randomize hand orientation ----
        if self.cfg.reset_hand_rot_noise > 0.0:
            # Random rotation: sample random axis + random angle within range
            rand_angle = sample_uniform(
                -self.cfg.reset_hand_rot_noise,
                self.cfg.reset_hand_rot_noise,
                (num_resets, 1),
                device=self.device,
            )
            rand_axis = torch.randn(num_resets, 3, device=self.device)
            rand_axis = rand_axis / rand_axis.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            # Convert to quaternion perturbation
            perturb_quat = quat_from_angle_axis(rand_angle.squeeze(-1), rand_axis)  # (N, 4)
            # Compose: new_quat = perturb * base
            new_hand_quat = quat_mul(perturb_quat, self.base_hand_quat.unsqueeze(0).expand(num_resets, -1))
        else:
            new_hand_quat = self.base_hand_quat.unsqueeze(0).expand(num_resets, -1)

        # Store per-env hand quaternion
        self.hand_quat[env_ids] = new_hand_quat

        # Write new hand root pose (position stays the same, orientation randomized)
        hand_root_pose = torch.zeros(num_resets, 7, device=self.device)
        hand_root_pose[:, :3] = self.base_hand_pos + self.scene.env_origins[env_ids]
        hand_root_pose[:, 3:7] = new_hand_quat
        self.hand.write_root_pose_to_sim(hand_root_pose, env_ids)

        # Update target rotation axis in world frame for these envs
        self.target_axis[env_ids] = quat_apply(
            new_hand_quat, self.target_axis_local.unsqueeze(0).expand(num_resets, -1)
        )

        # ---- Reset object to palm-center + noise (position relative to hand orientation) ----
        # Ball position = hand_pos + rotate(hand_quat, ball_local_offset) + noise
        ball_world_offset = quat_apply(new_hand_quat, self.ball_local_offset.unsqueeze(0).expand(num_resets, -1))
        ball_world_pos = self.base_hand_pos + ball_world_offset

        pos_noise = sample_uniform(
            -self.cfg.reset_object_pos_noise,
            self.cfg.reset_object_pos_noise,
            (num_resets, 3),
            device=self.device,
        )

        object_default_state = self.object.data.default_root_state[env_ids].clone()
        object_default_state[:, 0:3] = ball_world_pos + pos_noise + self.scene.env_origins[env_ids]
        object_default_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        object_default_state[:, 7:] = 0.0  # zero velocity

        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # ---- Update object init position for drop detection (varies per-env with hand rotation) ----
        self.object_init_pos_local[env_ids] = ball_world_pos + pos_noise

        # Sharpa-style reward buffers: snapshot world-frame default pose
        self.object_default_pos_w[env_ids] = object_default_state[:, 0:3]
        self.object_pos_prev[env_ids] = object_default_state[:, 0:3]
        # Clear external forces for reset envs
        self.rb_forces[env_ids] = 0.0

        # ---- DR: helper for env_ids on CPU ----
        env_ids_cpu = torch.tensor(env_ids, device="cpu") if not isinstance(env_ids, torch.Tensor) else env_ids.cpu()

        # ---- DR: randomize object scale (USD-level) ----
        if self.cfg.object_scale_range is not None:
            import omni.usd
            lo, hi = self.cfg.object_scale_range
            stage = omni.usd.get_context().get_stage()
            for i in range(num_resets):
                eid = env_ids[i] if isinstance(env_ids, (list, torch.Tensor)) else env_ids
                scale = sample_uniform(lo, hi, (1,), device=self.device).item()
                prim_path = f"/World/envs/env_{eid}/object"
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    from pxr import UsdGeom
                    xformable = UsdGeom.Xformable(prim)
                    scale_ops = [op for op in xformable.GetOrderedXformOps() if "scale" in op.GetOpName().lower()]
                    if scale_ops:
                        scale_ops[0].Set((scale, scale, scale))
                    else:
                        xformable.AddScaleOp().Set((scale, scale, scale))

        # ---- DR: randomize object mass ----
        if self.cfg.object_mass_range is not None:
            lo, hi = self.cfg.object_mass_range
            masses = self.object.root_physx_view.get_masses()
            rand_mass = sample_uniform(lo, hi, (num_resets, 1), device=masses.device)
            masses[env_ids_cpu] = rand_mass
            self.object.root_physx_view.set_masses(masses, env_ids_cpu)
            self.current_object_mass[env_ids] = rand_mass.to(self.device)

        # ---- DR: randomize object friction ----
        if self.cfg.object_friction_range is not None:
            lo, hi = self.cfg.object_friction_range
            # material_properties shape: (num_envs, num_shapes, 3) = [static_friction, dynamic_friction, restitution]
            mat_props = self.object.root_physx_view.get_material_properties()
            rand_friction = sample_uniform(lo, hi, (num_resets, 1, 1), device=mat_props.device)
            mat_props[env_ids_cpu, :, 0] = rand_friction.squeeze(-1)  # static friction
            mat_props[env_ids_cpu, :, 1] = rand_friction.squeeze(-1)  # dynamic friction (same value)
            self.object.root_physx_view.set_material_properties(mat_props, env_ids_cpu)
            self.current_object_friction[env_ids] = rand_friction.squeeze(-1).to(self.device)

        # ---- DR: randomize object center-of-mass offset ----
        if self.cfg.object_com_offset_range is not None:
            lo, hi = self.cfg.object_com_offset_range
            coms = self.default_object_com.clone()
            # coms may be on CPU (PhysX view); use its device
            if coms.dim() == 2:
                com_offset = sample_uniform(lo, hi, (num_resets, 3), device=coms.device)
                coms[env_ids_cpu, :3] = coms[env_ids_cpu, :3] + com_offset
            else:
                com_offset = sample_uniform(lo, hi, (num_resets, 1, 3), device=coms.device)
                coms[env_ids_cpu, :, :3] = coms[env_ids_cpu, :, :3] + com_offset
            self.object.root_physx_view.set_coms(coms, env_ids_cpu)
            self.current_com_offset[env_ids] = com_offset.reshape(num_resets, 3).to(self.device)

        # ---- DR: randomize gravity ----
        if self.cfg.gravity_range is not None:
            import isaaclab.sim as sim_utils_dr
            lo, hi = self.cfg.gravity_range
            rand_gz = -sample_uniform(lo, hi, (1,), device=self.device).item()  # negative Z
            physics_sim_view = sim_utils_dr.SimulationContext.instance().physics_sim_view
            import carb
            physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, rand_gz))

        # ---- DR: randomize PD gains (Kp/Kd) ----
        if self.cfg.pd_gain_scale_range is not None:
            lo, hi = self.cfg.pd_gain_scale_range
            # Per-env, per-dof scale factor
            stiffnesses = self.hand.root_physx_view.get_dof_stiffnesses()
            dampings = self.hand.root_physx_view.get_dof_dampings()
            kp_scale = sample_uniform(lo, hi, (num_resets, self.num_hand_dofs), device=stiffnesses.device)
            kd_scale = sample_uniform(lo, hi, (num_resets, self.num_hand_dofs), device=dampings.device)
            stiffnesses[env_ids_cpu] = self.default_kp[env_ids_cpu] * kp_scale
            dampings[env_ids_cpu] = self.default_kd[env_ids_cpu] * kd_scale
            self.hand.root_physx_view.set_dof_stiffnesses(stiffnesses, env_ids_cpu)
            self.hand.root_physx_view.set_dof_dampings(dampings, env_ids_cpu)
            # Also update torque estimation vectors for reward computation
            self.kp_vec = stiffnesses[:1, self.actuated_dof_indices].to(self.device)
            self.kd_vec = dampings[:1, self.actuated_dof_indices].to(self.device)
            # Store mean scale for privileged obs
            self.current_kp_scale[env_ids] = kp_scale.mean(dim=-1, keepdim=True).to(self.device)
            self.current_kd_scale[env_ids] = kd_scale.mean(dim=-1, keepdim=True).to(self.device)

        # ---- Reset action buffers ----
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0

        # ---- Reset observation history buffer ----
        self.obs_history[env_ids] = 0.0

        # ---- Reset previous quaternion buffer ----
        self.prev_object_quat[env_ids] = object_default_state[:, 3:7]

        # Debug: print init state on first reset for env 0
        if 0 in env_ids and not getattr(self, "_init_debug_printed", False):
            self._init_debug_printed = True
            idx = 0
            print("\n" + "=" * 70)
            print("[DEBUG] Init state for env 0:")
            print(f"  Hand quat (w,x,y,z):            {new_hand_quat[idx].cpu().tolist()}")
            print(f"  Hand init joint pos (actuated): {dof_pos[idx, self.actuated_dof_indices].cpu().tolist()}")
            print(f"  Grasp ref pos (PD target):      {self.grasp_ref_pos[idx].cpu().tolist()}")
            print(f"  Squeeze offset:                 {(self.grasp_ref_pos[idx] - dof_pos[idx, self.actuated_dof_indices]).mean().item():.4f} rad")
            obj_pos = object_default_state[idx, :3].cpu().tolist()
            env_origin = self.scene.env_origins[idx].cpu().tolist()
            local_pos = [obj_pos[i] - env_origin[i] for i in range(3)]
            print(f"  Object world pos: {obj_pos}")
            print(f"  Object local pos: {local_pos}")
            print(f"  Object init ref:  {self.object_init_pos_local[idx].cpu().tolist()}")
            print(f"  Target axis (world): {self.target_axis[idx].cpu().tolist()}")
            print("=" * 70 + "\n")

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

        # Compute REAL angular velocity from quaternion change (not PhysX solver artifact)
        # prev_object_quat is saved in _pre_physics_step (before physics advances)
        dt_env = self.cfg.sim.dt * self.cfg.decimation
        self.object_angvel = _quat_diff_to_angvel(
            self.prev_object_quat, self.object_rot, dt_env
        )



# ====================================================================== #
# JIT-compiled helper functions
# ====================================================================== #


@torch.jit.script
def _quat_diff_to_angvel(q_prev: torch.Tensor, q_curr: torch.Tensor, dt: float) -> torch.Tensor:
    """Compute angular velocity from consecutive quaternions.

    Args:
        q_prev: Previous quaternion (N, 4) in (w, x, y, z) format.
        q_curr: Current quaternion (N, 4) in (w, x, y, z) format.
        dt: Time step between the two quaternions.

    Returns:
        Angular velocity (N, 3) in world frame.
    """
    # Compute relative rotation: q_rel = q_curr * q_prev_inv
    # For unit quaternions, q_inv = q_conjugate = (w, -x, -y, -z)
    q_prev_inv = q_prev.clone()
    q_prev_inv[:, 1:] = -q_prev_inv[:, 1:]

    # Quaternion multiplication: q_rel = q_curr * q_prev_inv
    w1, x1, y1, z1 = q_curr[:, 0], q_curr[:, 1], q_curr[:, 2], q_curr[:, 3]
    w2, x2, y2, z2 = q_prev_inv[:, 0], q_prev_inv[:, 1], q_prev_inv[:, 2], q_prev_inv[:, 3]

    q_rel_w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q_rel_x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    q_rel_y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    q_rel_z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    # Ensure shortest path (q and -q represent the same rotation)
    sign = torch.sign(q_rel_w).unsqueeze(-1)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q_rel_w = q_rel_w * sign.squeeze(-1)
    q_rel_x = q_rel_x * sign.squeeze(-1)
    q_rel_y = q_rel_y * sign.squeeze(-1)
    q_rel_z = q_rel_z * sign.squeeze(-1)

    # Convert relative quaternion to axis-angle
    # angle = 2 * acos(w), axis = (x, y, z) / sin(angle/2)
    # For small angles, angular_velocity ≈ 2 * (x, y, z) / dt
    axis_vec = torch.stack([q_rel_x, q_rel_y, q_rel_z], dim=-1)  # (N, 3)
    sin_half = torch.norm(axis_vec, dim=-1, keepdim=True).clamp(min=1e-8)  # (N, 1)
    half_angle = torch.atan2(sin_half, q_rel_w.unsqueeze(-1))  # (N, 1)
    angle = 2.0 * half_angle  # (N, 1)

    # Angular velocity = angle * axis_unit / dt
    axis_unit = axis_vec / sin_half
    angvel = (angle / (dt + 1e-8)) * axis_unit  # (N, 3)

    return angvel


@torch.jit.script
def _unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Scale from [lower, upper] to [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower + 1e-8)


@torch.jit.script
def _compute_rewards(
    object_angvel: torch.Tensor,
    object_pos: torch.Tensor,
    object_pos_prev: torch.Tensor,
    object_default_pos: torch.Tensor,
    target_axis: torch.Tensor,
    angvel_clip_min: float,
    angvel_clip_max: float,
    hand_dof_pos_actuated: torch.Tensor,
    default_dof_pos_actuated: torch.Tensor,
    hand_dof_vel_actuated: torch.Tensor,
    estimated_torque: torch.Tensor,
    step_dt: float,
    rew_rotation_scale: float,
    rew_linvel_penalty_scale: float,
    rew_pos_diff_scale: float,
    rew_torque_scale: float,
    rew_work_scale: float,
    rew_object_pos_scale: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sharpa-style in-hand rotation reward (sharpa_wave_env.py:217-248)."""

    # 1. r_rot: angular velocity on target axis, CLAMPED to [-0.5, 0.5]
    angvel_on_axis = torch.sum(object_angvel * target_axis, dim=-1)
    rotate_reward = torch.clamp(angvel_on_axis, angvel_clip_min, angvel_clip_max)

    # 2. r_vel: L1 norm of object position change / dt
    object_linvel_penalty = torch.norm(object_pos - object_pos_prev, p=1, dim=-1) / step_dt

    # 3. r_diff: pose deviation from DEFAULT joint pos (not grasp_ref)
    pos_diff_penalty = torch.sum((hand_dof_pos_actuated - default_dof_pos_actuated) ** 2, dim=-1)

    # 4. r_torq: sum of squared torques
    torque_penalty = torch.sum(estimated_torque ** 2, dim=-1)

    # 5. r_work: SQUARED sum of (torque * velocity) — note: ((sum)^2), not sum(|.|)
    work_penalty = torch.sum(estimated_torque * hand_dof_vel_actuated, dim=-1) ** 2

    # 6. r_pos: bonus for staying near initial position (1/(dist+0.001))
    object_pos_diff = 1.0 / (torch.norm(object_pos - object_default_pos, dim=-1) + 0.001)

    total_reward = (
        rotate_reward * rew_rotation_scale
        + object_linvel_penalty * rew_linvel_penalty_scale
        + pos_diff_penalty * rew_pos_diff_scale
        + torque_penalty * rew_torque_scale
        + work_penalty * rew_work_scale
        + object_pos_diff * rew_object_pos_scale
    )

    components: list[torch.Tensor] = [
        rotate_reward * rew_rotation_scale,
        object_linvel_penalty * rew_linvel_penalty_scale,
        pos_diff_penalty * rew_pos_diff_scale,
        torque_penalty * rew_torque_scale,
        work_penalty * rew_work_scale,
        object_pos_diff * rew_object_pos_scale,
        angvel_on_axis,
    ]

    return total_reward, components
