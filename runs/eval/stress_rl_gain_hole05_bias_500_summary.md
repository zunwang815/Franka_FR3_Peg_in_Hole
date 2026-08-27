# 500 轮 PPO：0.5 mm 孔位误差压力测试最终结果

## 统一协议

- 任务：`Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0`
- seed=42，128 回合，`num_envs=64`
- 目标孔在 `10 cm × 10 cm` 工作空间内随机；抓取安装不确定性保留
- 每个 episode 采样固定二维孔位估计偏差，XY 标准差 `0.5 mm`
- 每控制步动作噪声 `std=0.02`
- 每 episode 动作增益扰动 `5%`
- 动作延迟 `0` 步；reset 归位 20 步

## 结果

| 方法 | 成功 | timeout | over-insertion | 成功率 |
|---|---:|---:|---:|---:|
| 几何教师 | 119/128 | 9 | 0 | **92.97%** |
| PPO 500 轮 `model_499` | 124/128 | 4 | 0 | **96.88%** |

PPO 相对几何教师提高 `5/128 = 3.91` 个百分点，正式实现了在满足老师孔位观测噪声要求的同时超越几何教师。

## 训练配置与产物

- 训练轮数：500
- 并行环境：1024
- rollout：64 steps/env
- learning rate：`1e-5`
- PPO epochs：1
- entropy coefficient：0
- residual penalty：100
- 初始 residual exploration：`0.03`

训练配置：[`run_config.json`](../ppo/stress/custom/20260818_164739/run_config.json)。最终模型：[`model_499.pt`](../ppo/stress/custom/20260818_164739/model_499.pt)。

教师基线：[`stress_teacher_hole05_bias_action02_gain5.json`](stress_teacher_hole05_bias_action02_gain5.json)。
