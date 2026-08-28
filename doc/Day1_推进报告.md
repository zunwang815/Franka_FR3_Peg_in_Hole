# Franka FR3 Peg-in-Hole 仿真任务 — Day 1 推进报告

> 日期：2026-08-11

---

## 一、环境搭建 ✅

| 项目 | 状态 | 备注 |
|---|---|---|
| GPU 驱动修复 | 完成 | 535→580.173.02，解决 `/proc/driver` 版本冲突 |
| CUDA Toolkit | 完成 | 系统已有 12.1 |
| Miniconda + isaac_lab 环境 | 完成 | Python 3.10 |
| Isaac Sim 4.5 + Isaac Lab 2.1 | 完成 | pip install + NVIDIA PyPI |
| Fabric PhysX 版本冲突 | 完成 | symlink 106.3.2→106.5.3，patch TOML 依赖 |
| GUI 3D 可视化 | 完成 | LD_LIBRARY_PATH 修复原生库路径 |
| rsl_rl 安装 | 完成 | v2.3.3，兼容 torch 2.5.1 |

**GPU**: NVIDIA RTX 5880 Ada Generation, 48GB VRAM

---

## 二、训练历程（12 轮迭代）

| # | 控制模式 | Reward 设计 | 孔尺寸 | 轮数 | 最高 success | 关键发现 |
|---|---|---|---|---|---|---|
| 1 | IK Abs (7D) | 指数核 | 实心方块 | 5000 | 0% | approach=0，无梯度 |
| 2 | IK Abs | 线性 clamp | 实心方块 | 5000 | 0% | **孔在原点**，机器人够不到 |
| 3 | IK Rel (6D) | 线性 | 实心方块 | 500 | — | 动作空间 6D→7D 不匹配 |
| 4 | Joint Pos (9D) | 线性 | 碰撞关闭 | 5000 | ~5% | success 判定仅有 Z 坐标 |
| 5 | Joint Pos | 组合 | 碰撞关闭 | 5000 | ~3% | 对齐始终 0.2，关节奇异频繁 |
| 6 | Joint Pos | 阶段门控 | 碰撞关闭 | 1000 | 0% | 门太严，无学习信号 |
| 7 | Joint Pos | 固定高熵 | 碰撞关闭 | 3000 | 3.3% | entropy 衰减→遗忘 |
| 8 | IK Rel | 线性 | **4.65cm 假孔** | 1000 | 100% | 🚨 孔尺寸计算错误 |
| 9 | IK Rel | +接触惩罚 | **2.3cm 真孔** | 1000 | 0.5% | 首次真正成功 |
| 10 | IK Rel | +XY 终止判定 | 2.3cm | 1000 | 0.5% | 终止条件与 reward 一致 |
| 11 | IK Rel | 视觉 peg | 2.3cm | 1000 | 0% | peg 同步移除后需重训 |
| 12 | IK Rel | 完整版 | 2.3cm | 1000 | **0.5%** | ✅ **基线确立** |

---

## 三、核心 Bug 修复记录

| # | Bug | 影响 | 修复方案 |
|---|---|---|---|
| 1 | `time_out` 永远返回 0 | 训练数据恒为 0，episode 永不终止 | `env.episode_length_buf >= env.max_episode_length - 1` |
| 2 | 手写四元数旋转 bug | peg 位置计算错误，reward 信号错乱 | 改用 `isaaclab.utils.math.quat_apply` |
| 3 | 孔位置被事件**替换**而非偏移 | 孔在机器人底座旁 (0,0,0.025)，不在桌上 (0.35,0,0.025) | `root_states += rand` 替代 `root_states = rand` |
| 4 | 成功终止只检查 Z 深度 | peg 在孔外也能判定成功 | 加入 XY 孔内检查 `|peg - hole| < 0.0115` |
| 5 | 孔壁位置计算公式错误 | 孔 4.65cm（应为 2.3cm），虚假 100% 成功率 | `center = hole_center ± (0.0115 + 0.005)` |
| 6 | 指数核 `exp(-dist/0.02)` 在 1m 处≈0 | 远距离无梯度，策略无法导航 | 改用纯线性 `-dist` |
| 7 | configclass 变量被当字段 | 临时计算变量 `_hx` 等被解析为场景资产 | 硬编码数值 |
| 8 | 观测/动作维度不匹配 | visualize 用 IK Play 但训练用 Joint Pos | 统一使用同一 task ID |

---

## 四、最终技术方案

### 场景

```
Franka 底座 (0,0,0)
    │
    ├── 桌子 (0.35, 0, 0)
    │     └── 孔框架 (0.35, 0, 0.02)
    │           ├── 北墙 y=+0.0165
    │           ├── 南墙 y=-0.0165
    │           ├── 西墙 x=0.3335
    │           └── 东墙 x=0.3665
    │           孔洞: 2.3cm × 2.3cm
    │
    └── 机械臂 → panda_hand/Peg (粉色, 2cmφ, 10cm长)
```

### 控制

| 参数 | 值 |
|---|---|
| 模式 | IK Relative (6D delta pose) |
| 动作空间 | (dx, dy, dz, droll, dpitch, dyaw) |
| scale | 0.05 (最大 ±5cm/步) |

### Reward

| 分量 | 权重 | 公式 | 作用 |
|---|---|---|---|
| approach_xy | 10 | `-distance` | 引导 peg 向孔靠近 |
| alignment | 8 | `(-ee_z_z+1)/2` | 鼓励竖直下落 |
| insertion_depth | 20 | `clamp(depth/0.03, -1, 1)` | 插入进度 |
| contact_penalty | -5 | 孔外下压 → -1 | 避开孔壁 |
| success_bonus | 500 | XY 孔内 + Z≥3cm | 完全插入 |
| action_rate | -0.01 | L2 动作差分 | 平滑控制 |
| joint_vel | -0.0001 | L2 关节速度 | 抑制抖动 |

### 观测 (37 维)

关节位置(9) + 关节速度(9) + 末端位置(3) + 末端姿态(4) + peg→孔向量(3) + 孔位置(3) + 上步动作(6)

### PPO 配置

| 参数 | 值 |
|---|---|
| 网络 | [256, 256, 128] ELU |
| 学习率 | 3e-4, fixed schedule |
| Entropy | 0.05, fixed |
| Gamma/λ | 0.99 / 0.95 |
| 环境数 | 128 |

---

## 五、当前最佳结果

| 指标 | 1000 轮训练后 |
|---|---|
| approach_xy | -78.9 (距孔 ~3.3cm) |
| alignment | 4.27 / 8 |
| insertion_depth | -3.1 |
| contact_penalty | -1.89 |
| success_bonus | 0.0095 (首次非零) |
| **success rate** | **~0.5%** |

---

## 六、失败教训

1. **先验证场景几何，再调 reward**——孔位置错误（在原点而非桌面）浪费了最开始的 7 轮训练，累计约 25000 次无效迭代
2. **成功判定必须与 reward 函数一致**——Z-only 判定导致虚假 100% 成功率（第 8 轮），浪费了后续调参的信任
3. **物理 peg 同步不可靠**——`write_root_state_to_sim` 每步调用不稳定，视觉 peg 挂在 gripper 下更简单可靠
4. **指数 reward 核在远距离无梯度**——`exp(-dist/0.02)` 在默认位姿 1m 处输出 ≈0，策略完全看不到孔。纯线性 `-dist` 简单且始终有效
5. **固定熵 > 自适应熵**——adaptive schedule 在 5000 轮中将 entropy 从 9.9 衰减到 -0.56，导致策略过早收敛到局部最优
6. **IK Rel (6D) >> Joint Pos (9D)**——减少 3 个自由度，天然避免关节奇异，对齐更好
7. **1.5mm 单边间隙是硬任务**——1000 轮仅有 ~0.5% 成功率是正常的强化学习基线，需要更长的训练时间

---

## 七、下一步计划

1. **5000-10000 轮完整训练**——当前 1000 轮刚出现首次成功信号，需延长训练
2. **课程学习**——若收敛到瓶颈：大孔(3cm)→中孔(2.5cm)→目标孔(2.3cm)
3. **阶段二**——六孔阵列 (2×3, 间距 3cm)，Goal-conditioned RL
4. **实验报告**——总结训练曲线、成功率、消融实验

---

## 八、快速启动命令

```bash
# 激活环境
source scripts/setup_env.sh

# 训练
python scripts/train_phase1.py --num_envs 128 --max_iterations 5000

# 可视化
python scripts/visualize.py --checkpoint runs/phase1/<run_dir>/model_XXXX.pt --episodes 10

# TensorBoard
tensorboard --logdir runs/phase1
```
