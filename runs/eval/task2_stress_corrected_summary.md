# 任务2六孔阵列：统一时间预算后的几何教师压力测试

## 修正原因

此前任务2压力测试使用了阵列环境默认的12秒 episode，即360个控制步；即使命令行指定 `max_steps=600`，环境仍会在359步内部 timeout。同时，教师在径向误差小于1 mm后才切换到插入阶段，强动作扰动下容易长期停留在孔口。

任务1的正式压力评估允许600个控制步，因此本次将任务2 episode 时长显式设为20秒，并把插入阶段门限调为2 mm。最终成功判据仍严格保持径向误差 `<=1.3 mm`、插入深度15–40 mm、倾角`<=2°`，没有放宽验收标准。

## 统一压力协议

- 任务：`Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0`
- 阵列原点：`10 cm × 10 cm`随机
- 目标孔：随机选择或固定遍历0–5
- 孔位估计误差：每 episode 固定二维XY偏差，标准差 `0.5 mm`
- 动作噪声：每控制步归一化高斯噪声 `std=0.02`
- 动作增益：每 episode 高斯扰动 `std=5%`
- 控制器延迟：0步
- 时间预算：20秒，即600个控制步
- 教师参数：默认位置尺度、`kp_position=0.8`、`kp_orientation=0.8`、`approach_depth=-10 mm`、`alignment_gate=2 mm`

## 结果

随机目标孔128回合：

| 策略 | 成功率 | timeout |
|---|---:|---:|
| 修正后的几何教师 | **128/128 = 100%** | 0 |

固定孔位分组，每孔32回合：

| 目标孔 | 成功率 |
|---:|---:|
| 0 | 32/32 = 100% |
| 1 | 32/32 = 100% |
| 2 | 32/32 = 100% |
| 3 | 32/32 = 100% |
| 4 | 32/32 = 100% |
| 5 | 32/32 = 100% |

## 干扰分解

在原1 mm门限下，单独加入0.5 mm孔位偏差仍为128/128；单独加入动作噪声0.02和增益5%则为2/128。说明真正导致旧结果崩溃的是动作执行扰动与过早/过窄插入阶段切换的组合，而不是孔位观测偏差或六孔编号。

## 复现证据

- 修正随机目标结果：[`task2_stress_teacher_hole05_bias_action02_gain5_gate2.json`](task2_stress_teacher_hole05_bias_action02_gain5_gate2.json)
- 固定孔位结果：[`task2_stress_gate2_hole0.json`](task2_stress_gate2_hole0.json) 至 [`task2_stress_gate2_hole5.json`](task2_stress_gate2_hole5.json)
- 时间预算修正代码：[`scripts/eval_geometric_controller.py`](../../scripts/eval_geometric_controller.py)
- 偏差分解：[`task2_decomp_bias_only.json`](task2_decomp_bias_only.json)、[`task2_decomp_action_only.json`](task2_decomp_action_only.json)

旧的1.56%/2.34%结果仍可作为“旧评估协议诊断记录”，但不能作为修正后任务2几何教师能力的最终结论。
