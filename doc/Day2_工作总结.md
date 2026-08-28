# Day 2 工作总结：Franka FR3 Peg-in-Hole 仿真环境修复与训练调优

> 日期：2026-08-12

---

## 一、Day 1 遗留问题回顾

Day 1 结束时，环境存在以下关键缺陷：
- 4 面墙壁在环境重置时未随 `hole_board` 移动，导致 reward 信号与实际物理孔位置脱节
- Peg 为纯视觉物体，无碰撞物理
- 孔为正方形而非任务要求的圆形
- 1000 轮训练成功率仅 ~0.5%

---

## 二、环境修复（已完成 ✅）

### 2.1 致命 Bug 修复

| # | Bug | 修复 | 文件 |
|---|---|---|---|
| 1 | 墙壁不移动 | `reset_hole_position_uniform` 同时移动 12 个立柱 | `mdp/events.py` |
| 2 | Peg 无碰撞 | 改为 `RigidObjectCfg` + `MassPropertiesCfg`(100kg) + 高阻尼(0.99) | `peg_in_hole_env_cfg.py` |
| 3 | Peg 不跟手 | 从 `AssetBaseCfg` 改为 `RigidObjectCfg`，通过 interval event 每步同步到手指中点 | `mdp/events.py` |
| 4 | 墙壁被撞飞 | 4 面墙改为 12 个 kinematic 立柱 | `peg_in_hole_env_cfg.py` |
| 5 | reward/visual 参考系不一致 | 统一使用 `FrameTransformer` (panda_hand) + `(0,0,0.17)` 偏移 | `mdp/rewards.py`, `mdp/terminations.py`, `mdp/observations.py` |

### 2.2 场景优化

| 项目 | 最终值 | 说明 |
|---|---|---|
| 桌子位置 | x=0.525 | 多次调整确定 |
| 孔形状 | 12 立柱圆环 φ2.3cm | 替代原 4 面墙方孔 |
| 孔壁属性 | kinematic | 不可移动，保留碰撞 |
| Peg 同步 | interval event 60Hz | `sync_peg_to_ee` 追踪手指中点 |
| Peg 属性 | dynamic, 100kg, damping 0.99 | 防止接触抖动 |
| Warm-start | IK 求解，手放孔上方 12cm | `reset_robot_above_hole` |
| 约束圆柱 | 15cm 半径 + 边界惩罚 | 防止策略漫无目的探索 |

### 2.3 Reward 最终配置

| 分量 | 权重 | 说明 |
|---|---|---|
| `approach_xy` | 100 | 1.5m 线性 XY 引导 |
| `proximity_bonus` | 30 | 10cm 近距奖励 |
| `cylinder_boundary_penalty` | -10 | 10-15cm 警告区 |
| `alignment` | 3 | 竖直姿态 |
| `insertion_depth` | 30 | 无门控下降奖励 |
| `contact_penalty` | -5 | 孔外下压惩罚 |
| `success_bonus` | 500 | 成功插入 |
| `action_rate` | -0.005 | 平滑性 |
| `joint_vel` | -0.00005 | 抑制抖动 |

### 2.4 终止条件

| 条件 | 说明 |
|---|---|
| `time_out` | 240 步超时 |
| `success_insertion` | XY 孔内 + Z≥3cm |
| `peg_left_cylinder` | 离开孔 XY 15cm |

---

## 三、训练历程

### 3.1 尝试的配置

| # | 关键配置 | 熵 | 插入门控 | Warm-start | 圆柱约束 | 最高成功率 |
|---|---|---|---|---|---|---|
| 1 | 手指中点参考 | 0.01 | 无 | 无 | 无 | **25%** (iter 846, 后遗忘) |
| 2 | FrameTransformer + 高斯 XY | 0.01 | 无 | 无 | 无 | 0% |
| 3 | + 线性 XY (w=100) | 0.01 | 无 | 无 | 无 | 0% (距离太远) |
| 4 | + 门控插入 | 0.01 | 5cm | 无 | 无 | 0% (策略不靠近) |
| 5 | + Warm-start | 0.01 | 5cm | ✅ | 无 | 0% (策略走远) |
| 6 | + 圆柱约束 8cm | 0.02 | 无 | ✅ | ✅ | 0% (太紧) |
| 7 | + 圆柱约束 15cm | 0.02 | 无 | ✅ | ✅ | 0% (仍不收敛) |

### 3.2 关键发现

1. **唯一成功的配置是 #1**：高熵(0.05) + 手指中点参考 + 无门控插入。在 iter 846 达到 25% 成功率，但后续因熵过高遗忘。
2. **低熵(0.01-0.02)始终无法收敛**：探索不足以找到 1.5mm 间隙的孔。
3. **Warm-start 有效但不够**：IK 将手放到孔上方 ~12cm，但策略仍无法学会精确 XY 对准。
4. **reward signal 在远处太弱**：即使增加权重到 100，环境间距导致的距离过大使梯度不显著。

---

## 四、不足与待解决问题

### 4.1 核心瓶颈：训练未收敛

经过 7 轮 1000 轮训练的迭代，最高成功率仅 25%（且不稳定）。根本原因：

- **任务难度固有**：1.5mm 单边间隙 + 含噪观测 + 10×10cm 随机 → 纯随机探索几乎不可能找到孔
- **RL 探索-利用困境**：高熵(0.05)→探索够但遗忘；低熵(0.01)→稳定但探索不足
- **奖励函数设计复杂度**：多分量加权奖励需要在多个目标间平衡，调参空间大

### 4.2 潜在改进方向

| 方向 | 优先级 | 说明 |
|---|---|---|
| **课程学习** | 高 | 先 3cm 大孔训练到收敛，再逐步缩至 2.3cm |
| **增加训练量** | 高 | 当前 1000 轮不足，建议 3000-5000 轮 |
| **自适应熵衰减** | 中 | 从 0.05 线性衰减到 0.01，兼顾探索与收敛 |
| **Demonstration/Behavior Cloning** | 中 | 先用脚本策略生成成功轨迹，预训练策略 |
| **孔位置噪声减小** | 低 | 当前 ±5cm 可能过大，先 ±3cm 训练再逐步增大 |
| **简化 Reward** | 低 | 拆分为独立阶段：先训 XY approach，冻结后再训插入 |

### 4.3 已知小问题

- `visualize.py` 需要手动禁用 `peg_left_cylinder` 才能正常可视化
- Peg offset 可能需要微调（0.17 接近但可能不精确）
- 桌子位置 0.525 虽多次调整但仍可能与真实 FR3 安装位置有偏差

---

## 五、文件修改清单

| 文件 | 状态 | 主要改动 |
|---|---|---|
| `mdp/events.py` | ✅ | 墙壁移动、peg 同步、warm-start IK |
| `mdp/rewards.py` | ✅ | 统一 FrameTransformer 参考系、新增圆柱惩罚、回退门控 |
| `mdp/terminations.py` | ✅ | 统一参考系、新增圆柱终止 |
| `mdp/observations.py` | ✅ | 统一参考系 |
| `peg_in_hole_env_cfg.py` | ✅ | Peg 物理化、12 柱圆孔、warm-start event、圆柱约束 |
| `peg_in_hole_array_env_cfg.py` | ✅ | 同步 peg 修改 |
| `config/franka/agents/rsl_rl_ppo_cfg.py` | ✅ | 熵调节(0.01→0.02→0.05)、保存间隔 |
| `config/franka/joint_pos_env_cfg.py` | ✅ | 初始关节姿态前伸 |
| `config/franka/ik_rel_env_cfg.py` | 未改 | — |
| `scripts/visualize.py` | ✅ | 减速、禁用圆柱约束 |
| `scripts/train_phase1.py` | 未改 | — |

---

## 六、下一步建议

1. **立即**：用 `entropy=0.05` 跑 2000 轮（当前训练可能仍在运行）
2. **短期**：实现课程学习——3cm 大孔→2.5cm→2.3cm
3. **中期**：自适应熵衰减 + 更长的训练
4. **长期**：Phase 2 六孔阵列任务

---

> **总结**：Day 2 完成了仿真环境从"有致命 Bug 不可用"到"物理正确、结构完整"的转变。核心硬件（peg 跟随、圆孔碰撞、warm-start）全部就绪。训练方面，探索了 7 种配置组合，确认了高熵配置的潜力，剩余挑战是找到合适的探索-利用平衡以收敛到 >90% 成功率。
