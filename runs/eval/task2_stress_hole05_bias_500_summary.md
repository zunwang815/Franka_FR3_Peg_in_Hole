# 任务2六孔阵列：按任务1协议重训后的压力评估

## 评估协议

- 任务：`Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0`
- 阵列整体位置：在 `10 cm × 10 cm` 范围内随机；目标孔从六个孔中随机选择
- 孔位估计误差：每个 episode 固定采样二维高斯 XY 偏差，标准差 `0.5 mm`
- 执行扰动：动作噪声 `std=0.02`，每个 episode 动作增益高斯扰动 `std=5%`
- 控制器延迟：`0` 步
- 评估：`seed=42`，`128` 回合，`num_envs=64`
- 六孔 reset：`reset_warmup_steps=0`；这是阵列环境的专属初始姿态设置，其他 PPO 和扰动设置与任务1一致

## 重训设置

- 算法：几何教师 residual PPO
- 训练：`num_envs=1024`，rollout `64`，`500` iterations
- 学习率：`1e-5`
- PPO epochs：`1`
- 初始策略噪声：`0.03`
- residual penalty：`100`
- entropy coefficient：`0`
- checkpoint：该历史中间模型未随公开仓库保留
- run config：[`run_config.json`](../ppo/task2_stress/custom/20260818_190619/run_config.json)

## 结果

| 策略 | 成功率 | timeout | over-insertion |
|---|---:|---:|---:|
| 六孔几何教师 | **2/128 = 1.56%** | 126 | 0 |
| 重训 PPO `model_499` | **3/128 = 2.34%** | 125 | 0 |

重训 PPO 相对几何教师提高 `0.78` 个百分点，但仍未接近任务1的 90% 目标，也没有超过重训前任务2 PPO 的 `3/128 = 2.34%`。因此，500轮训练和任务1同协议的扰动设置本身没有解决任务2在该压力协议下的失败问题。

本结果不否定任务2在原始验收协议下的 100% 结果；它说明六孔阵列在加入孔位估计误差、动作噪声和增益扰动后存在独立的泛化/控制问题，需要进一步针对阵列状态、目标孔条件化或插入阶段策略进行诊断。

## 复现命令

```bash
OMNI_KIT_ACCEPT_EULA=YES python -u scripts/eval_checkpoint_simple.py \
  --checkpoint /path/to/task2_stress_model_499.pt \
  --task Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0 \
  --geometric_teacher_residual \
  --teacher_observation_noise_mm 0 \
  --teacher_tilt_noise_deg 0 \
  --hole_xy_bias_std_mm 0.5 \
  --action_noise_std 0.02 \
  --action_gain_noise_std_pct 5 \
  --reset_warmup_steps 0 \
  --episodes 128 --num_envs 64 --seed 42 --barrier_mm 100
```
