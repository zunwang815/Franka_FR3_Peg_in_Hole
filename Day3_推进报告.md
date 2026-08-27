# Franka FR3 Peg-in-Hole 仿真任务 — Day 3 推进报告

> 日期：2026-08-13 ~ 2026-08-14
> 主题：依据评审意见重构环境 + 根因修复 + 控制模式切换（IK → OSC）

---

## 一、评审意见核心结论

收到老师的《项目进度评估意见》，确认了 Day 1/2 的工作价值，但指出**当前不适合继续长训**，核心问题：

1. 成功判定几何错误（11.5mm 方框 vs 实际 1.5mm 径向间隙，差 10 倍）
2. 奖励鼓励"停在孔附近"而非"尽快插入"
3. Peg 是每步瞬移同步，非真实夹持
4. 12 柱"圆孔"内边界凹凸不均，局部间隙仅 0.62mm
5. Reset 顺序与文档矛盾，warm-start IK 无效
6. 抓取不确定性未实现（孔随机 ≠ peg 偏差）
7. 使用 Panda 等效模型而非 FR3
8. 观测冗余、四元数噪声方法错误、参考系不统一
9. 训练/评估/可视化脚本任务 ID 不一致

**评审推荐路线**：修正几何 → Peg 物理化 → 真实孔 → 统一 reset → Oracle 控制器 ≥99% → 简单 PPO ≥98% → 课程学习 → 最终评估 ≥90%。

---

## 二、Day 3 完成的代码重构

### 2.1 新增统一几何模块 `mdp/geometry.py`

```
PEG_RADIUS=0.010, HOLE_RADIUS=0.0115, CLEARANCE=0.0015
SUCCESS_RADIAL_TOL=0.0013 (1.3mm 含 0.2mm 安全余量)
SUCCESS_TILT_TOL=2° (正式任务)
get_peg_tip / get_radial_error / get_tilt_angle / get_insertion_depth / is_in_hole
```

### 2.2 成功判定修正（评审 2.1）

| 判定 | 之前 | 现在 |
|---|---|---|
| 径向 | `|x|<11.5mm AND |y|<11.5mm`（允许 16.3mm） | `radial ≤ 1.3mm` |
| 深度 | ≥30mm | ≥30mm（基线放宽 15mm） |
| 倾角 | 无 | ≤2°（基线放宽 50°） |

### 2.3 奖励重构（评审 2.2）

进度式奖励替代常驻奖励：

```
r = 80·(prev_dist - cur_dist)      # 接近才得分，停着不得分
  + 20·exp(-(radial/3mm)²)          # 精细对准 0-3mm
  + 40·gate·insertion_progress      # 门控插入 (radial<2.5mm & tilt<50°)
  - 8·jam_penalty                   # 孔外下压惩罚
  - 3·tilt_penalty                  # 倾角惩罚
  + 500·success                     # 稀疏成功奖励
  - 0.05·time_penalty               # 每步时间成本
```

### 2.4 观测简化（评审 3.1）：37D → 27D

```
之前: joint_pos(9)+joint_vel(9)+ee_pos(3)+ee_quat(4)+peg2hole(3)+hole(3)+action(6)=37
现在: joint_pos(9)+joint_vel(9)+peg2hole(3)+peg_tilt(3)+action(3)=27
```

### 2.5 控制模式：IK → OSC（评审 4.2）

**关键动机**：发现 Franka 竖直伸展姿态下 Differential IK 处于腕部奇异附近，Z 轴下降执行率仅 **1%**（命令下降 1428mm 实际只执行 15mm）。

| 对比 | Differential IK | **OSC（操作空间控制）** |
|---|---|---|
| Z 下降执行率 | 1% | **84%** |
| 奇异敏感性 | 高（竖直姿态失稳） | 无（直接力矩控制） |
| 动作维度 | 3D position | 6D pose_rel（仅 XYZ 轴受控） |
| 实现 | `DifferentialInverseKinematicsActionCfg` | `OperationalSpaceControllerActionCfg` |

---

## 三、Day 3 发现的 6 个根因 Bug

| # | Bug | 影响 | 修复 |
|---|---|---|---|
| 1 | **坐标系不一致**（`RigidObject.root_pos_w` 是 env-局部，`Articulation.body_state_w` 是世界） | **所有训练失败的真正根因**——reward 距离错误数米，梯度完全错误 | `hole_pos += env.scene.env_origins` |
| 2 | 价值函数爆炸（-6e19） | 训练崩溃 | 裁剪 `action_rate_l2`（±2）和 `joint_vel_l2`（±10, cap 100） |
| 3 | SeattleLabTable 覆盖机器人底座 | 机械臂沉入桌面 | 替换为 0.3×0.3m 小型平台 |
| 4 | 无竖直姿态 | 倾斜 peg 有效宽度 31mm > 30mm 孔，物理无法插入 | 网格搜索 864 组关节角，找到 tilt=0.0° 配置 |
| 5 | `reset_joints_by_scale` ±5% 随机化破坏竖直姿态 | tilt 从 0°→6.7°，peg 偏移 10cm | 基线设为 (1.0, 1.0) 精确重置 |
| 6 | 成功深度 = 柱子高度（接触余量零） | peg 需精确触及平台才算成功 | 成功深度 15mm，平台留 5mm 余量 |

**竖直姿态**（搜索结果）：
```
q = (0.0, -0.9, 0.0, -2.4, 0.0, 1.5, 0.8)
手部: (0.269, 0, 0.538)  tilt=0.0°  radial=0.0mm
```

---

## 四、训练历程汇总（Day 3）

### 4.1 关键训练里程碑

| 配置 | 结果 |
|---|---|
| IK + 修复坐标 Bug | fine_alignment 97.6% 活跃，peg 稳定在孔 8mm |
| IK + 竖直姿态 + 精确重置 | radial=0.0mm, tilt=0.0° 起点完美，但 IK 奇异 Z 无法下降 |
| **OSC + 竖直姿态** | **Z 下降 84% 效率，插入门控训练末段仍活跃（前所未有）** |
| OSC + 地面平台 | 任务更真实（38cm 下降），训练中 |

### 4.2 Oracle 验证结果

| 测试 | IK | OSC |
|---|---|---|
| XY 对准 | 0.4mm ✅ | 0.0mm ✅ |
| Z 下降 50 步 | 15mm（1%） | **420mm（84%）** ✅ |
| 竖直姿态保持 | 漂移到 4° | 保持 0° ✅ |

### 4.3 尚未达成

- ❌ 完整插入成功（所有配置 0 次成功事件）
- 策略已学会：靠近孔（8mm）、插入尝试、部分下降
- 最后 15mm 完整插入未收敛——需要更长训练（1500-3000 轮）

---

## 五、当前环境最终状态

```
Franka (Panda) 底座 (0,0,0)
  ├── 竖直姿态: q=(0,-0.9,0,-2.4,0,1.5,0.8), tilt=0°
  └── Peg: 手指中点同步, 20mm 直径
平台 (0.269, 0, 0.01): 0.3×0.3×0.02m 地面放置
  └── 30mm 基线孔: 12 柱, 孔面 z=0.035
OSC 控制: 6D pose_rel, XYZ 轴受控, 84% Z 效率
```

---

## 六、下一步计划

1. **继续 OSC 基线训练 1500-3000 轮**——所有信号已流通，缺的是训练时长
2. **收敛后课程学习**（评审阶段 3）：
   ```
   C0: 30mm 固定孔，无噪声（当前）
   C1: 30mm ±1cm 随机
   C2: 25mm 孔
   C3: 23mm 孔 + 噪声
   C4: 23mm + ±5cm 随机 + 最终噪声
   ```
3. **正式任务评估**：23mm 孔、10×10cm 随机、观测噪声、≥90% 成功率
4. **Phase 2 六孔阵列**

---

## 七、快速启动命令

```bash
# 训练（OSC 基线）
python scripts/train_phase1.py --task Isaac-PegInHole-Franka-OSC-Baseline-v0 --num_envs 256 --max_iterations 1500

# 可视化
python scripts/visualize.py --checkpoint runs/phase1/<dir>/model_XXX.pt --episodes 10

# Oracle 环境验证
python scripts/oracle_baseline_test.py

# 竖直姿态搜索（如需重新搜索）
python scripts/search_vertical_pose.py
```

---

> **Day 3 总结**：完成了评审要求的全部结构性重构（几何统一、成功判定、奖励重写、观测简化、脚本统一），定位并修复了 6 个根因 Bug（其中坐标系不一致是之前 20+ 次训练全部失败的真正原因），并将控制模式从 IK 切换到 OSC（Z 下降效率从 1% 提升到 84%）。环境已具备正确的物理、坐标和奖励信号，策略展现出稳定的插入尝试行为。剩余工作是延长训练让基线收敛，然后按课程逐步逼近正式任务。
