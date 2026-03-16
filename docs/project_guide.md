# Wuji In-Hand Rotation — 项目文档与设置指南

本文档覆盖 Wuji 灵巧手球体旋转任务的完整设计、配置、运行方法及调试经验。

---

## A. 项目结构分析

### 文件树

```
inhandrotation/
├── train.py                          # 训练入口 (RSL-RL PPO)
├── play.py                           # 推理/调试入口 (可视化 rollout)
├── sweep_squeeze.py                  # 抓握力度扫描工具 (调参用)
├── small ball.usd                    # 球体 USD 资产
├── logs/                             # Tensorboard 日志 & 模型 checkpoint
│   └── wuji_inhand_rotation/
├── data_urdf/
│   ├── robot/wujihand/urdf/
│   │   └── wujihand_right.urdf       # Wuji 手 URDF
│   └── object/contactdb/sphere_tennis/
│       └── coacd_decomposed_object_one_link.urdf  # 球体 URDF
├── docs/
│   └── project_guide.md              # 本文档
└── tasks/
    ├── __init__.py                    # 导出所有任务 (import wuji_inhand_rotation)
    └── wuji_inhand_rotation/
        ├── __init__.py               # Gymnasium 注册: Wuji-InHand-Rotation-Direct-v0
        ├── wuji_hand_cfg.py          # 手部 ArticulationCfg (PD gains, 关节限位, URDF 路径)
        ├── wuji_inhand_rotation_env_cfg.py   # 环境配置 (obs/action空间, 奖励系数, 仿真参数)
        ├── wuji_inhand_rotation_env.py       # 核心环境逻辑 (DirectRLEnv 子类)
        ├── usd/                      # URDF→USD 转换缓存
        └── agents/
            └── rsl_rl_ppo_cfg.py     # PPO 算法 & 网络配置
```

### 关键入口点

| 用途 | 文件 | 说明 |
|------|------|------|
| 训练 | `train.py` | RSL-RL OnPolicyRunner, 支持 `--resume`, `--max_iterations` |
| 推理/调试 | `play.py` | 支持 `--checkpoint` 加载策略, `--zero_action` 静态测试 |
| 抓握调参 | `sweep_squeeze.py` | 线性扫描抓握偏移量, 观察手指闭合与球体稳定性 |
| 任务注册 | `tasks/wuji_inhand_rotation/__init__.py` | `gym.register("Wuji-InHand-Rotation-Direct-v0", ...)` |

---

## B. 任务设计 (Task Design)

### Observation Space (76 维)

| 分量 | 维度 | 说明 |
|------|------|------|
| Hand joint positions | 20 | 5 指 x 4 关节, 归一化到 [-1, 1] |
| Hand joint velocities | 20 | 乘以 `vel_obs_scale = 0.2` |
| Previous actions | 20 | 上一步动作 (用于平滑) |
| Object position (local) | 3 | 相对于 env origin 的位置 |
| Object rotation (quat) | 4 | 四元数 (w, x, y, z) |
| Object linear velocity | 3 | 乘以 `vel_obs_scale = 0.2` |
| Object angular velocity | 3 | 乘以 `vel_obs_scale = 0.2` |
| Target rotation axis | 3 | 目标旋转轴 (默认 [0, 0, 1]) |
| **总计** | **76** | |

- `vel_obs_scale = 0.2`: 所有速度观测量均乘以此系数以控制数值范围

### Action Space (20 维)

- **控制方式**: 绝对位置控制
- **公式**: `desired = current_grasp_ref + action * action_scale`
  - `current_grasp_ref` 在 warmup 期间从 `grasp_base_pos` 线性插值到 `grasp_ref_pos`
- **action_scale = 0.05 rad**: 每个动作单位对应 0.05 弧度的关节位置偏移
- **动作范围**: clamp 到 [-1, 1], 即最大偏移 +/-0.05 rad/step
- **EMA 平滑**: `act_moving_average = 1.0` (当前未启用平滑)
- 最终目标经 `saturate()` 限制在关节限位内, 送入 PD 控制器

### Reward Design (奖励设计)

| 奖励项 | 系数 | 公式 / 说明 |
|--------|------|-------------|
| `rew_hold_bonus` | **+0.5** | 每步常数生存奖励, 鼓励保持球不掉落 |
| `rew_rotation_scale` | **+5.0** | `5.0 * clamp(angvel_on_axis / target_angular_vel, -1, 2)`, 沿目标轴角速度越大奖励越高 |
| `rew_non_target_rotation_penalty` | **-0.1** | `-0.1 * ||angvel_off_axis||`, 惩罚非目标轴旋转 |
| `rew_object_drop_penalty` | **-2.0** | `-2.0 * (dist/fall_dist)^2`, 平滑二次惩罚, 球越远惩罚越大 |
| `rew_action_penalty` | **-0.05** | `-0.05 * sum(action^2)`, 能量正则化 |
| `rew_pose_deviation_penalty` | **-0.01** | `-0.01 * sum((joint_pos - grasp_ref)^2)`, 偏离参考抓握姿态惩罚 |
| `rew_joint_vel_penalty` | **-0.001** | `-0.001 * sum(joint_vel^2)`, 关节速度平滑性惩罚 |

**设计思路**: hold_bonus 保证正向基线奖励; rotation_reward 是主要驱动力; drop_penalty 使用平滑二次形式避免稀疏信号; 其余惩罚项系数较小用于正则化。

### Reset / Initialization (重置与初始化)

**两阶段系统**:

1. **init_state (开放姿态)**: `WUJI_HAND_GRASP_CFG` 中的关节位置, 手指张开, 无穿透
   - 手掌位置: `(0, 0, 0.5)`, 朝向: `rot=(0.707, 0, -0.707, 0)`
   - 球体初始位置: `(-0.095, 0.0, 0.56)` (掌心上方)

2. **grasp_target (紧握姿态)**: `WUJI_GRASP_TARGET_JOINT_POS` 中的关节位置, 手指贴合球体

3. **Warmup 过渡**: 前 **30 步** (0.5 秒) 线性从 init_state 插值到 grasp_target
   - `warmup_frac = clamp(episode_step / 30, 0, 1)`
   - `current_grasp_ref = base + (target - base) * warmup_frac`
   - 避免初始穿透导致仿真不稳定

- **噪声**: `reset_dof_pos_noise = 0.0`, `reset_object_pos_noise = 0.0` (当前禁用, 用于调试)

### Termination (终止条件)

| 条件 | 阈值 | 说明 |
|------|------|------|
| 球体掉落 | `fall_dist = 0.15m` | 球体与初始位置距离超过 15cm |
| 横向偏移 | `lateral_dist = 0.10m` | 球体水平方向偏移超过 10cm |
| 超时截断 | `episode_length_s = 10.0s` | 600 步 @ 60Hz 控制频率 |

---

## C. 关键配置参数

### PD Control (隐式执行器)

| 关节 | Kp (stiffness) | Kd (damping) | Effort Limit (Nm) |
|------|-------|-------|---------|
| finger*_joint1, joint2 | 100.0 | 1.0 | 20.0 |
| finger*_joint3 | 60.0 | 0.5 | 10.0 |
| finger*_joint4 | 40.0 | 0.5 | 5.0 |

- 驱动类型: `force` / 目标类型: `position`
- 求解器迭代: position=20, velocity=10

### Simulation (仿真参数)

| 参数 | 值 | 说明 |
|------|------|------|
| Physics dt | 1/120 s | 物理步长 |
| Decimation | 2 | 控制频率 = 60 Hz |
| Render interval | 2 | 每 2 物理步渲染一次 |
| Static friction | 2.0 | 手指-球体静摩擦 |
| Dynamic friction | 2.0 | 手指-球体动摩擦 |
| Restitution | 0.0 | 无弹性碰撞 |
| Ball density | 100 kg/m^3 | 球体密度 |
| Ball radius | 0.036 m | 网球大小 |
| Ball init pos | (-0.095, 0, 0.56) | 掌心上方 |
| Env spacing | 0.75 m | 多环境间距 |
| Default num_envs | 4096 | 默认并行环境数 |
| `self_collision` | False | 自碰撞禁用 |
| `fix_base` | True | 手掌固定 |
| `collider_type` | convex_hull | 碰撞体类型 |
| `force_usd_conversion` | True | 每次重新转换 URDF→USD |
| `merge_fixed_joints` | False | 保留 tip link 为独立刚体 |

### PPO Config (训练超参数)

| 参数 | 值 |
|------|------|
| Actor 网络 | MLP [512, 256, 128], ELU 激活 |
| Critic 网络 | MLP [512, 256, 128], ELU 激活 |
| Observation normalization | True (empirical_normalization=False, 由网络层实现) |
| init_std | 0.3 |
| entropy_coef | 0.001 |
| learning_rate | 3e-4 (adaptive schedule) |
| desired_kl | 0.016 |
| gamma | 0.99 |
| lam (GAE) | 0.95 |
| num_steps_per_env | 16 |
| num_mini_batches | 4 |
| num_learning_epochs | 5 |
| clip_param | 0.2 |
| value_loss_coef | 1.0 |
| max_grad_norm | 1.0 |
| save_interval | 2000 |
| max_iterations | 10000 (默认, 可 CLI 覆盖) |
| logger | tensorboard |

---

## D. 运行方法

### Environment Setup

```bash
# Isaac Sim Python 解释器路径
export ISAACSIM_PYTHON=/home/yinan/Documents/isaacsim/_build/linux-x86_64/release/python.sh
```

### Training (训练)

```bash
# 标准训练 (16384 并行环境, 无头模式)
$ISAACSIM_PYTHON train.py --task Wuji-InHand-Rotation-Direct-v0 \
    --num_envs 16384 --headless --max_iterations 50000

# 小规模调试训练
$ISAACSIM_PYTHON train.py --task Wuji-InHand-Rotation-Direct-v0 \
    --num_envs 64 --max_iterations 1000

# 后台训练
nohup $ISAACSIM_PYTHON train.py --task Wuji-InHand-Rotation-Direct-v0 \
    --num_envs 16384 --headless --max_iterations 50000 > nohup.out 2>&1 &

# 恢复训练
$ISAACSIM_PYTHON train.py --task Wuji-InHand-Rotation-Direct-v0 \
    --num_envs 16384 --headless --resume \
    --load_checkpoint logs/wuji_inhand_rotation/model_10000.pt
```

### Play / Debug (推理与调试)

```bash
# 零动作测试 (静态抓握, 验证初始化是否稳定)
$ISAACSIM_PYTHON play.py --task Wuji-InHand-Rotation-Direct-v0 \
    --num_envs 1 --zero_action

# 随机动作测试
$ISAACSIM_PYTHON play.py --task Wuji-InHand-Rotation-Direct-v0 --num_envs 16

# 加载训练好的策略
$ISAACSIM_PYTHON play.py --task Wuji-InHand-Rotation-Direct-v0 \
    --num_envs 16 --checkpoint logs/wuji_inhand_rotation/model_5000.pt
```

### Sweep Squeeze (抓握力度扫描)

```bash
# 线性扫描抓握偏移量 0.0 → 0.10 rad, 持续 10 秒
$ISAACSIM_PYTHON sweep_squeeze.py --start 0.0 --end 0.10 --duration 10
```

输出包含: 时间、偏移量、各关节估计抓握力 (Kp*offset)、球体偏移距离、奖励值。

### Monitoring (监控)

```bash
# Tensorboard
tensorboard --logdir logs/wuji_inhand_rotation

# GPU 状态
nvidia-smi
```

---

## E. 调试经验与已解决问题

### 1. object_init_pos_local 坐标系 Bug

**问题**: `default_root_state` 返回的是 **env-local** 坐标, 不是世界坐标。之前代码减去了 `env_origins`, 导致除 env 0 外所有环境的球体初始位置计算错误, 所有环境立即触发掉落终止。

**修复**: 直接使用 `default_root_state[:, :3]`, 不减去 `env_origins`:
```python
self.object_init_pos_local = self.object.data.default_root_state[:, :3].clone()
```

### 2. init_at_random_ep_len 导致 episode_length=1

**问题**: RSL-RL 的 `init_at_random_ep_len=True` 会在第一步就随机截断部分环境, 产生 `episode_length=1` 的异常数据。

**修复**: 在 `train.py` 中显式设置:
```python
runner.learn(num_learning_iterations=..., init_at_random_ep_len=False)
```

### 3. RSL-RL >= 5.0 API 变更

**问题**: 新版 RSL-RL 移除了 `stochastic`, `init_noise_std` 等旧参数, 传入会报错。

**修复**: 在 `train.py` 中构建 config dict 后剥离废弃字段:
```python
_DEPRECATED = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
for key in ("actor", "critic"):
    if key in train_cfg:
        for dep in _DEPRECATED:
            train_cfg[key].pop(dep, None)
```

### 4. Action std 爆炸 (std=1.89)

**问题**: 策略找到局部最优: 即使动作完全随机 (大 std), 只要球不掉就能拿到 hold_bonus。action_penalty 太小无法约束。

**修复**:
- 增大 `rew_action_penalty` (从更小值调到 -0.05)
- 减小 `entropy_coef` (调到 0.001)
- 使 action 成本显著, 策略倾向于输出小而精确的动作

### 5. 初始化穿透问题

**问题**: 如果初始关节位置就是紧握姿态, 手指与球体在第一步就穿透, 导致仿真不稳定, 球体被弹飞。

**修复**: 分离初始状态与抓握目标:
- `WUJI_HAND_GRASP_CFG` 的 init_state: 手指张开 (open pose), 无穿透
- `WUJI_GRASP_TARGET_JOINT_POS`: 紧握姿态 (tight pose), 作为 PD 目标
- **Warmup ramp**: 前 30 步 (0.5s) 线性从 open 过渡到 tight

### 6. 奖励系数失衡

**问题**: 初始 drop_penalty = -10, 远大于所有正向奖励之和。策略学到的唯一目标是不掉球, 无法学习旋转。

**修复**:
- 降低 `rew_object_drop_penalty`: -10 → **-2.0**
- 添加 `rew_hold_bonus`: **+0.5** (每步正向基线)
- 使用平滑二次惩罚代替阶跃惩罚: `penalty = -2.0 * (dist/fall_dist)^2`

---

## F. 后续扩展计划

| 方向 | 说明 | 当前状态 |
|------|------|----------|
| Asymmetric actor-critic | Critic 使用 privileged obs (接触力、精确物体状态等) | `state_space = 0` (尚未实现) |
| Teacher-student distillation | Teacher (sim privileged) → Student (real-world transferable) | 计划中 |
| Contact proxy observations | 将指尖接触力/二值接触加入 obs | `fingertip_body_names` 已定义, `activate_contact_sensors=True` |
| Reset noise | 重置时对关节/球体位置加噪声 (domain randomization) | `reset_dof_pos_noise=0.0`, `reset_object_pos_noise=0.0` (禁用) |
| Self-collision | 手指间自碰撞检测 | `self_collision=False`, `enabled_self_collisions=False` |
| Sim2Real transfer | 部署到真实 Wuji 手 | 计划中 |
| force_usd_conversion | 配置稳定后关闭以加速启动 | 当前 `True` (每次重新转换) |
| Target axis randomization | 随机旋转轴方向, 训练通用旋转策略 | 当前固定 `[0, 0, 1]` |

---

## G. 当前配置汇总表

### 环境参数

| 参数 | 值 |
|------|------|
| Task ID | `Wuji-InHand-Rotation-Direct-v0` |
| Observation dim | 76 |
| Action dim | 20 |
| State dim | 0 (symmetric) |
| Episode length | 10.0 s (600 steps) |
| Control frequency | 60 Hz |
| Physics frequency | 120 Hz |
| Decimation | 2 |
| Default num_envs | 4096 |

### 动作控制

| 参数 | 值 |
|------|------|
| action_scale | 0.05 rad |
| act_moving_average | 1.0 (无平滑) |
| Action clamp | [-1, 1] |
| squeeze_warmup_steps | 30 步 |

### 观测缩放

| 参数 | 值 |
|------|------|
| vel_obs_scale | 0.2 |
| Joint pos normalization | [-1, 1] via _unscale() |

### 目标旋转

| 参数 | 值 |
|------|------|
| target_rotation_axis | [0, 0, 1] (Z 轴) |
| target_angular_velocity | 1.0 rad/s |

### 奖励系数

| 参数 | 值 | 类型 |
|------|------|------|
| rew_hold_bonus | +0.5 | 正向 (生存) |
| rew_rotation_scale | +5.0 | 正向 (目标) |
| rew_non_target_rotation_penalty | -0.1 | 惩罚 |
| rew_object_drop_penalty | -2.0 | 惩罚 (二次) |
| rew_action_penalty | -0.05 | 惩罚 |
| rew_pose_deviation_penalty | -0.01 | 惩罚 |
| rew_joint_vel_penalty | -0.001 | 惩罚 |

### 终止条件

| 参数 | 值 |
|------|------|
| fall_dist | 0.15 m |
| lateral_dist | 0.10 m |
| episode timeout | 10.0 s |

### PD 控制器

| 关节 | Kp | Kd | Effort Limit |
|------|------|------|------|
| joint1, joint2 | 100.0 | 1.0 | 20.0 Nm |
| joint3 | 60.0 | 0.5 | 10.0 Nm |
| joint4 | 40.0 | 0.5 | 5.0 Nm |

### 物理材质

| 参数 | 值 |
|------|------|
| Static friction | 2.0 |
| Dynamic friction | 2.0 |
| Restitution | 0.0 |
| Bounce threshold velocity | 0.2 m/s |

### 球体物理

| 参数 | 值 |
|------|------|
| Density | 100 kg/m^3 |
| Gravity | enabled |
| Gyroscopic forces | enabled |
| Max depenetration velocity | 0.5 m/s |
| Sleep threshold | 0.005 |

### PPO 训练

| 参数 | 值 |
|------|------|
| Actor | [512, 256, 128] MLP, ELU |
| Critic | [512, 256, 128] MLP, ELU |
| init_std | 0.3 |
| entropy_coef | 0.001 |
| learning_rate | 3e-4 (adaptive) |
| desired_kl | 0.016 |
| gamma | 0.99 |
| lam | 0.95 |
| num_steps_per_env | 16 |
| num_mini_batches | 4 |
| num_learning_epochs | 5 |
| clip_param | 0.2 |
| max_grad_norm | 1.0 |
| save_interval | 2000 |
| max_iterations | 10000 |
| seed | 42 |

### 重置噪声

| 参数 | 值 | 状态 |
|------|------|------|
| reset_dof_pos_noise | 0.0 | 禁用 |
| reset_object_pos_noise | 0.0 | 禁用 |

---

*文档生成日期: 2026-03-16*
