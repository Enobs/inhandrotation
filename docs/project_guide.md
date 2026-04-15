# Wuji In-Hand Rotation — 项目文档

本文档覆盖 Wuji 灵巧手球体旋转任务的完整设计、配置、运行方法及调试经验。
基于 IMCopilot 论文 (Towards Human-Like Manipulation through RL-Augmented Teleoperation and MoDE-VLA)。

---

## A. 项目结构

```
inhandrotation/
├── train.py                          # 训练入口 (RSL-RL PPO)
├── play.py                           # 推理/调试入口 (可视化 rollout)
├── test_hand.py                      # 手部独立测试 (无球体)
├── tune_grasp.py                     # 交互式抓握姿态调参
├── sweep_squeeze.py                  # 抓握力度扫描工具
├── nohup.out                         # 后台训练输出日志
├── docs/
│   ├── project_guide.md              # 本文档
│   └── Towards Human-Like...VLA.pdf  # 参考论文 (IMCopilot)
├── logs/
│   └── wuji_inhand_rotation/
│       ├── model_*.pt                # 模型 checkpoint (每 1000 iter 保存)
│       └── events.out.tfevents.*     # TensorBoard 日志
├── data_urdf/
│   ├── robot/wujihand/urdf/
│   │   └── wujihand_right.urdf       # Wuji 手 URDF (20 DOF)
│   └── object/contactdb/sphere_tennis/
│       └── coacd_decomposed_object_one_link.urdf  # 网球 URDF
└── tasks/
    ├── __init__.py
    └── wuji_inhand_rotation/
        ├── __init__.py               # Gymnasium 注册: Wuji-InHand-Rotation-Direct-v0
        ├── wuji_hand_cfg.py          # 手部 ArticulationCfg (PD gains, URDF 路径)
        ├── wuji_inhand_rotation_env_cfg.py   # 环境配置
        ├── wuji_inhand_rotation_env.py       # 核心环境 (DirectRLEnv 子类)
        ├── usd/                      # URDF → USD 转换缓存
        └── agents/
            └── rsl_rl_ppo_cfg.py     # PPO 超参数 & 网络配置
```

---

## B. 环境设置与启动

### 前置依赖

- Isaac Sim (含 PhysX)
- IsaacLab (source build at `~/Documents/IsaacLab`)
- RSL-RL >= 5.0

### 环境变量

```bash
# 添加到 ~/.bashrc
export ISAACSIM_PYTHON=/home/yinan/Documents/isaacsim/_build/linux-x86_64/release/python.sh
```

**重要**: 必须使用 `$ISAACSIM_PYTHON` 而不是系统 python 或 conda python。IsaacLab 依赖 Isaac Sim 内置的 Python 环境。如果 conda 环境处于激活状态，会看到警告但通常仍能运行。

### 训练

```bash
# 标准训练 (4096 并行环境, headless)
nohup $ISAACSIM_PYTHON train.py --task Wuji-InHand-Rotation-Direct-v0 \
    --num_envs 4096 --headless &

# 小规模调试
$ISAACSIM_PYTHON train.py --task Wuji-InHand-Rotation-Direct-v0 \
    --num_envs 64 --max_iterations 1000

# 恢复训练
$ISAACSIM_PYTHON train.py --task Wuji-InHand-Rotation-Direct-v0 \
    --num_envs 4096 --headless --resume \
    --load_checkpoint logs/wuji_inhand_rotation/model_10000.pt
```

### 推理 / 调试

```bash
# 零动作测试 (验证静态抓握稳定性)
$ISAACSIM_PYTHON play.py --zero_action --num_envs 4

# 加载策略可视化
$ISAACSIM_PYTHON play.py --num_envs 16 \
    --checkpoint logs/wuji_inhand_rotation/model_20000.pt

# 指定球体质量测试
$ISAACSIM_PYTHON play.py --zero_action --object_mass 0.2 --num_envs 4
```

### 监控

```bash
tensorboard --logdir logs/wuji_inhand_rotation
```

---

## C. 任务设计

### 目标

单手灵巧操作：控制 Wuji 手的 20 个手指关节，使球体沿指定轴以目标角速度持续旋转，同时保持稳定抓握。

### Observation Space — Actor (76 维)

| 分量 | 维度 | 来源 | 部署可用? |
|------|------|------|-----------|
| Hand joint positions | 20 | 关节编码器 | 是 |
| Hand joint velocities | 20 | 关节编码器差分 | 是 (需差分) |
| Previous actions | 20 | 自身缓存 | 是 |
| Object position (local) | 3 | 仿真特权 | **否** — 需蒸馏 |
| Object rotation (quat) | 4 | 仿真特权 | **否** — 需蒸馏 |
| Object linear velocity | 3 | 仿真特权 | **否** — 需蒸馏 |
| Object angular velocity | 3 | 仿真特权 | **否** — 需蒸馏 |
| Target rotation axis | 3 | 人工指定 | 是 |

速度观测乘以 `vel_obs_scale = 0.2`。关节位置归一化到 `[-1, 1]`。

### State Space — Critic (83 维, 仅训练时)

Critic 比 Actor 额外看到 DR 的特权信息 (asymmetric actor-critic):

| 分量 | 维度 | 说明 |
|------|------|------|
| Actor obs | 76 | 全部 actor 观测 |
| Object mass | 1 | 当前 episode 随机化的质量 (kg) |
| Object friction | 1 | 当前 episode 随机化的摩擦系数 |
| CoM offset | 3 | 当前 episode 随机化的质心偏移 (m) |
| Kp scale | 1 | 当前 episode 的 PD 刚度缩放因子均值 |
| Kd scale | 1 | 当前 episode 的 PD 阻尼缩放因子均值 |

### Observation History Buffer

每步存储 `[joint_pos(20), joint_vel(20)]` = 40 维，保留最近 3 步。
当前 teacher 训练不使用 history；Phase 2 student 蒸馏时用。

### Action Space (20 维)

- 范围: `[-1, 1]`
- 映射方式: **非对称全范围**
  - `action = 0` → grasp reference position (抓握参考)
  - `action = +1` → joint upper limit
  - `action = -1` → joint lower limit
- Warmup: 前 30 步从 open pose 线性过渡到 tight grasp (避免穿透)
- EMA 平滑: `act_moving_average = 1.0` (当前未启用)

### Reward Design (对齐 IMCopilot 论文)

论文公式: `r = λ_rot·r_rot + λ_vel·r_vel + λ_work·r_work + λ_torq·r_torq + λ_diff·r_diff`

| 奖励项 | 系数 | 公式 | 说明 |
|--------|------|------|------|
| r_rot | +5.0 | `exp(-0.5 * (ω_on_axis - ω_target)²)` | **高斯型**速度追踪，精确匹配目标角速度给满分 |
| r_vel | -0.5 | `sum(linvel²)` | 惩罚线速度 (球只旋转不平移) |
| r_work | -0.001 | `sum(\|τ·v\|)` | 惩罚关节功率 |
| r_torq | -0.0001 | `sum(τ²)` | 惩罚关节力矩 |
| r_diff | -0.002 | `sum((q - q_ref)²)` | 惩罚偏离参考抓握姿态 |

其中力矩 τ 通过 PD 控制器估计: `τ = Kp*(target - pos) - Kd*vel`

**注意**: 旧版本有 r_finger (指尖距离) 和 r_smooth (关节平滑) 两个额外项，已删除以对齐论文。

### Termination

| 条件 | 阈值 |
|------|------|
| 球体掉落 (距初始位置) | 0.15 m |
| 横向偏移 | 0.10 m |
| 超时截断 | 20.0 s |

---

## D. Domain Randomization (对齐论文)

论文: "domain randomization over object scale, mass, friction, center-of-mass offset, gravity, and PD gains"

| DR 项 | 范围 | 实现方式 |
|-------|------|----------|
| Object mass | 0.01 ~ 0.2 kg | PhysX `set_masses()` |
| Object friction | 0.5 ~ 2.0 | PhysX `set_material_properties()` |
| Object CoM offset | ±1 cm per axis | PhysX `set_coms()` |
| Gravity | 9.01 ~ 10.61 m/s² | PhysX `set_gravity()` (全局) |
| PD gains | 0.7x ~ 1.3x | PhysX `set_dof_stiffnesses/dampings()` |
| Object scale | 0.8x ~ 1.2x | **关闭** (USD 操作太慢) |
| Hand orientation | ±π (全随机) | 每次 reset 随机手掌朝向 |
| Joint position noise | ±0.05 rad | 每次 reset 加噪声 |
| Object position noise | ±5 mm | 每次 reset 加噪声 |

所有 DR 参数可在 config 中设为 `None` 关闭。

**已知限制**: Gravity 是全局参数 (PhysX 限制), 所有 env 共享同一重力值。

---

## E. 训练架构 (Teacher-Student)

### Phase 1: Teacher (当前阶段)

```
Actor:  policy obs (76) → MLP [512, 256, 128] → action (20)
Critic: critic state (83) → MLP [512, 256, 128] → value (1)
              ↑ 多了 mass, friction, CoM, Kp, Kd
```

- Asymmetric actor-critic: actor 和 critic 看不同的观测
- RSL-RL PPO config 中 `obs_groups = {actor: ["policy"], critic: ["critic"]}`
- Teacher 的 actor 仍依赖仿真中的物体状态 (pos/rot/vel)

### Phase 2: Student (计划中)

```
Student obs:
  - joint pos (20) × 3 steps history = 60
  - joint vel (20) × 3 steps history = 60
  - prev actions (20)
  - target axis (3)
  Total: 143
```

蒸馏方式:
1. Teacher actor 中加 privileged encoder: 物体状态 → latent z
2. Student 用 history encoder: 3步 proprioception history → z_hat
3. 蒸馏 loss: `L = L_rl + α * ||z_hat - z.detach()||²`

### Phase 3: 部署 (计划中)

Student 输入全部来自 Wuji Hand SDK:
- 关节位置: `JointState.position` @ 1000 Hz
- 关节速度: 位置差分估算
- 上一步动作: policy 自身缓存
- 目标旋转轴: 人工指定

---

## F. 真实硬件兼容性 (Wuji Hand SDK)

### 可读取

| 数据 | API | 频率 |
|------|-----|------|
| Joint positions (20) | `JointState.position` | 1000 Hz |
| Joint effort/current (20) | `JointState.effort` | 1000 Hz |
| Temperature, voltage | `HandDiagnostics` | 10 Hz |

### 可写入

| 数据 | API |
|------|-----|
| Joint position targets (20) | `/{hand}/joint_commands` |
| Joint enable/disable | `set_enabled` service |

### 不可用

- **Joint velocity**: SDK 不直接提供，需位置差分估算
- **Fingertip contact force**: 无触觉传感器，仅有 motor current 作为粗略接触代理
- **Object state**: 需外部感知 (视觉) 或通过 teacher-student 蒸馏消除

### 控制链路

```
Policy output → joint_commands topic → [EMA 5.7Hz @ 100Hz] → [固件 10Hz LP @ 16kHz] → [PID Kp=5 Ki=0.08 Kd=150] → 电机
```

---

## G. PPO 训练配置

| 参数 | 值 |
|------|------|
| Actor 网络 | MLP [512, 256, 128], ELU |
| Critic 网络 | MLP [512, 256, 128], ELU |
| init_std | 0.3 |
| learning_rate | 3e-4 (adaptive, KL-based) |
| desired_kl | 0.016 |
| gamma | 0.99 |
| lam (GAE) | 0.95 |
| num_steps_per_env | 24 |
| num_mini_batches | 4 |
| num_learning_epochs | 5 |
| clip_param | 0.2 |
| entropy_coef | 0.001 |
| max_grad_norm | 1.0 |
| save_interval | 1000 |
| max_iterations | 100000 |
| obs normalization | True (网络层内) |

---

## H. 物理仿真参数

### Wuji Hand PD 控制器

| 关节 | Kp | Kd | Effort Limit |
|------|------|------|------|
| joint1, joint2 | 100.0 | 1.0 | 20.0 Nm |
| joint3 | 60.0 | 0.5 | 10.0 Nm |
| joint4 | 40.0 | 0.5 | 5.0 Nm |

### 仿真

| 参数 | 值 |
|------|------|
| Physics dt | 1/120 s |
| Control freq | 60 Hz (decimation=2) |
| Static friction | 1.5 |
| Dynamic friction | 1.5 |
| Restitution | 0.0 |
| Solver pos iterations | 20 |
| Solver vel iterations | 10 |
| Default num_envs | 4096 |
| Env spacing | 0.75 m |

### 球体

| 参数 | 值 |
|------|------|
| 半径 | ~3.4 cm |
| URDF 标称质量 | 57 g |
| Density (spawn) | 100 kg/m³ → ~16.5 g |
| DR 质量范围 | 10g ~ 200g |
| Angular damping | 0.5 |
| Linear damping | 0.1 |

---

## I. 已解决问题 & 调试经验

### 1. object_init_pos_local 坐标系 Bug

`default_root_state` 返回 env-local 坐标而非世界坐标。之前代码减去了 `env_origins`，导致除 env 0 外立即触发掉落终止。
修复: 直接用 `default_root_state[:, :3]`。

### 2. init_at_random_ep_len 导致 episode_length=1

RSL-RL 的 `init_at_random_ep_len=True` 第一步就截断部分环境。
修复: `runner.learn(..., init_at_random_ep_len=False)`。

### 3. RSL-RL >= 5.0 API 变更

新版移除了 `stochastic`, `init_noise_std` 等参数。
修复: train.py 中剥离废弃字段后传入 runner。

### 4. Action std 爆炸 (std=1.89)

策略找到局部最优: 随机动作也能拿 hold_bonus。
修复: 增大 action_penalty, 减小 entropy_coef。

### 5. 初始化穿透

紧握初始姿态导致球体被弹飞。
修复: 两阶段初始化 — open pose → 30 步 warmup → tight grasp。

### 6. 奖励失衡

drop_penalty (-10) 远大于正向奖励，策略只学不掉球。
修复: 降低惩罚，添加 hold_bonus 正向基线。

### 7. Reward 不对齐论文

旧 r_rot 用 `clamp(ω/ω_target, 0, 2)` — 越快越好，无法匀速旋转。
额外的 r_finger 和 r_smooth 论文里没有。
修复: r_rot 改为高斯型 `exp(-0.5*(ω-ω_target)²)`; 删除额外项。

### 8. PhysX tensor device mismatch

`root_physx_view.get_*()` 返回 CPU tensor，但 env 在 GPU 上。
修复: DR 代码中用 `env_ids_cpu` 索引 PhysX tensor，reward 中用 `.to(self.device)`。

### 9. COM tensor 维度不一致

`get_coms()` 对单 body 物体返回 `(N, 7)` 而非 `(N, 1, 7)`。
修复: 检测 `coms.dim()` 分两种路径处理。

---

## J. 与 IMCopilot 论文的差异

### 已对齐

- Reward 函数 5 项 (r_rot, r_vel, r_work, r_torq, r_diff)
- Domain randomization 6 项 (mass, friction, CoM, gravity, PD gains, scale)
- Asymmetric actor-critic (critic 看 DR 特权信息)
- PPO 训练 (IsaacLab + RSL-RL)

### 尚未实现

| 项目 | 论文描述 | 状态 |
|------|---------|------|
| Action 方式 | 增量 `q_t = q_{t-1} + λ·Δθ` | 当前用绝对位置目标 |
| Observation history | 3 步 proprioception 历史 | buffer 已建，teacher 暂不用 |
| Fingertip contact force | 加入 obs | SDK 无触觉传感器 |
| Teacher-student distillation | 蒸馏出不需要物体状态的 student | 计划中 (Phase 2) |
| Object scale DR | 0.8x ~ 1.2x | 写好但关闭 (USD 操作慢) |

---

*文档更新日期: 2026-04-02*
