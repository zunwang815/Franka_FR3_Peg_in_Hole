# Franka FR3 Peg-in-Hole：几何教师与 Residual PPO

本项目在 NVIDIA Isaac Sim / Isaac Lab 中实现了 Franka 机械臂圆棒插孔任务，并完成了从单孔随机插入到 `2 × 3` 六孔阵列指定孔插入的训练、评估和 GUI 可视化。

最终策略不是从零使用强化学习探索完整插入动作，而是采用：

```text
最终动作 = 冻结的几何教师动作 + PPO 学习的有界 residual action
```

这种方法先用解析几何教师验证任务的物理可达性，再由 PPO 学习对孔位估计误差、动作噪声和控制增益扰动的局部补偿。

> 当前结论仅针对 Isaac Lab 仿真，不代表真实 FR3 机械臂的实机成功率。项目中的 `panda_with_fixed_peg.usda` 使用 Isaac Lab Franka Panda 资产构建，是导入正式 FR3 USD 前的开发模型。

![任务1和任务2最终结果](artifacts/task2_visualization/final_results_summary.png)

## 1. 最终结果

| 任务与协议 | 几何教师 | Residual PPO | PPO 相对提升 |
|---|---:|---:|---:|
| 任务1基础严格协议 | 512/512，100% | 512/512，100% | 0 个百分点 |
| 任务1强扰动协议 | 119/128，92.97% | 124/128，96.88% | +3.91 个百分点 |
| 任务2基础六孔协议 | — | 512/512，100% | — |
| 任务2最终公平强扰动协议 | 110/128，85.94% | 128/128，100% | +14.06 个百分点 |

任务2最终 PPO 的 128 个成功 episode 终端指标如下：

| 指标 | 均值 | 最小值 | 最大值 |
|---|---:|---:|---:|
| 插入深度 | 16.461 mm | 15.012 mm | 26.465 mm |
| 径向误差 | 0.836 mm | 0.084 mm | 1.294 mm |
| 棒倾角 | 0.736° | 0.153° | 1.612° |

最终结果证据：

- 任务1强扰动对照：[runs/eval/stress_rl_gain_hole05_bias_500_summary.md](runs/eval/stress_rl_gain_hole05_bias_500_summary.md)
- 任务2几何教师：[runs/eval/task2_fair_teacher_action0305_gate2_128.json](runs/eval/task2_fair_teacher_action0305_gate2_128.json)
- 任务2最终 PPO：[runs/eval/task2_fair_ppo_task1ppo_action02_eval128.json](runs/eval/task2_fair_ppo_task1ppo_action02_eval128.json)
- 完整方法与结果说明：[任务1_任务2_总结报告.md](任务1_任务2_总结报告.md)

## 2. 任务定义

原始任务要求整理如下。

### 2.1 任务1：单孔随机插入

- 机械臂：Franka 7-DoF 机械臂开发模型；
- 圆棒直径：20 mm；
- 圆孔直径：23 mm；
- 圆棒长度：100 mm；
- 孔壁：36 段环形碰撞结构；
- 孔位置：在约 `10 cm × 10 cm` 可达范围内随机；
- 抓取不确定性：Peg mount X/Y 方向均允许约 ±5 mm 变化；
- 策略不能读取无噪声的绝对棒/孔位姿。

### 2.2 任务2：六孔阵列指定孔插入

- `2 × 3` 阵列，共 6 个孔；
- 相邻孔中心间距：30 mm；
- 六孔相对位置固定，阵列整体在约 `10 cm × 10 cm` 范围内随机；
- 目标孔可随机选择，也可固定指定为 `0–5`；
- 所有孔均保留碰撞几何，而不是仅显示目标孔；
- 策略通过 6 维 one-hot 观测获知目标孔编号。

### 2.3 最终成功判据

一个 episode 只有同时满足以下条件才计为成功：

- 径向误差 `<= 1.3 mm`；
- 插入深度位于 `15–40 mm`；
- 棒倾角 `<= 2°`；
- 未发生 workspace violation；
- 在 20 秒，即 600 个 30 Hz 控制步内完成。

`timeout`、`over-insertion` 和其他 termination 均单独统计，不能把任意 `terminated=True` 直接当成成功。

## 3. 方法概述

### 3.1 控制与物理频率

- 物理仿真频率：120 Hz；
- 控制 decimation：4；
- 策略控制频率：30 Hz；
- 控制器：6D Pose Operational Space Control（OSC）；
- OSC 零空间刚度：最终协议为 20；
- 求解器迭代：position 16、velocity 4；
- enhanced determinism：启用。

### 3.2 策略观测

任务1的 30 维观测包括：

- 9 维关节位置；
- 9 维关节速度；
- 3 维棒尖到目标孔的相对位移；
- 3 维棒倾斜信息；
- 6 维上一动作。

任务2在此基础上增加 6 维目标孔 one-hot，因此观测为 36 维。策略不读取隐藏的 Peg mount 关节，也不直接读取绝对孔位姿。

### 3.3 几何教师

[scripts/geometric_teacher.py](scripts/geometric_teacher.py) 根据棒尖到孔中心的相对位移、插入深度、棒倾角和姿态误差，生成有界的 6D 相对 Pose 动作。它用于：

1. 验证机器人姿态、孔壁和控制器是否物理可达；
2. 提供稳定且可解释的基础动作；
3. 作为 residual PPO 的冻结基准策略。

几何教师是解析控制器，不属于 RL。

### 3.4 Residual PPO

[scripts/residual_policy.py](scripts/residual_policy.py) 实现冻结几何教师和可训练 residual head。residual 网络为 `64-64-Tanh`，输出缩放为 `0.15`。

[scripts/residual_ppo.py](scripts/residual_ppo.py) 在标准 PPO 损失上加入 teacher-action residual penalty，限制策略偏离教师过远。最终 500 轮配置使用：

| 参数 | 最终设置 |
|---|---:|
| 并行环境 | 1024 |
| 训练轮数 | 500 |
| rollout | 64 steps/env |
| learning rate | `1e-5` |
| PPO epochs | 1 |
| 初始策略噪声 | `0.03` |
| entropy coefficient | 0 |
| residual penalty | 100 |
| 训练动作噪声 | `0.02` |
| 动作增益扰动 | 5% |

### 3.5 扰动定义

- `teacher_observation_noise_mm`：教师训练时每步相对位置均匀观测噪声；
- `hole_xy_bias_std_mm`：每个 episode 固定采样一次的 XY 孔位估计高斯偏差；
- `action_noise_std`：每个控制步施加的归一化动作高斯噪声；
- `action_gain_noise_std_pct`：每个 episode 固定的控制增益高斯扰动；
- `action_delay_steps`：最终所有主线协议均为 0。

老师的任务没有提出动作延迟要求，而且 6D OSC 对延迟较敏感，因此最终验收没有混入控制延迟。

## 4. 依赖与测试环境

项目在以下环境中完成并验证：

| 组件 | 版本/配置 |
|---|---|
| 操作系统 | Ubuntu 22.04.5 LTS |
| Python | 3.10.20 |
| NVIDIA Isaac Sim | 4.5.0 |
| Isaac Lab | 2.1.0 |
| PyTorch | 2.5.1 |
| TorchVision | 0.20.1 |
| Gymnasium | 1.3.0 |
| RSL-RL | 2.3.3 |
| NumPy | 1.26.4 |
| Matplotlib | 3.10.9 |
| TensorBoard | 2.20.0 |
| 测试 GPU | NVIDIA RTX 5880 Ada，49 GB VRAM |
| 测试驱动 | 580.173.02 |

运行要求：

- Linux 与支持 Vulkan/RTX 的 NVIDIA GPU；
- 正确安装 Isaac Sim 4.5 和 Isaac Lab 2.1；
- 可工作的 NVIDIA 驱动和图形显示环境；
- GUI 回放需要有效的 `DISPLAY`；
- 1024 个并行环境按 49 GB 显存验证。显存较小时可减小 `--num_envs`，但吞吐量和复现实验条件会改变。

本仓库没有单独的 `requirements.txt`，因为 Isaac Sim、Isaac Lab、PyTorch 和 CUDA 组件需要保持彼此兼容。建议先按照 Isaac Lab 对应版本完成基础环境安装，再核对上表中的 Python 包版本。

## 5. 环境准备

以下命令假设项目使用名为 `isaac_lab` 的 Conda 环境：

```bash
conda activate isaac_lab
cd /path/to/Franka_FR3_Peg_in_Hole
export OMNI_KIT_ACCEPT_EULA=YES
```

GUI 模式推荐直接加载项目提供的环境脚本：

```bash
cd /path/to/Franka_FR3_Peg_in_Hole
source scripts/setup_env.sh
```

`scripts/setup_env.sh` 默认假设环境安装在：

```text
$HOME/miniconda3/envs/isaac_lab
```

它还会搜索 `$HOME/.local/share/ov/data/exts/v2` 中的 Omniverse 动态库。如果 Conda 或 Omniverse 安装在其他位置，需要先修改该脚本中的对应路径。

检查关键版本：

```bash
python -c "import sys, torch, gymnasium, isaaclab; print(sys.version); print(torch.__version__); print(gymnasium.__version__)"
```

## 6. 快速验证

先用少量并行环境确认任务注册、CUDA、观测和动作空间均可用。

任务1：

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/verify_env.py \
  --task Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0 \
  --num_envs 4 --steps 10
```

任务2：

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/verify_env.py \
  --task Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0 \
  --num_envs 4 --steps 10
```

正常情况下脚本会打印观测空间、动作空间、reset 随机性，并以 `ALL CHECKS PASSED` 结束。

## 7. 项目结构

```text
Franka_FR3_Peg_in_Hole/
├── assets/
│   └── panda_with_fixed_peg.usda        # 带 Peg mount 不确定性的本地机器人资产
├── tasks/franka_peg_in_hole/
│   ├── config/franka/                    # Gym 任务注册、Franka 与 PPO 配置
│   ├── mdp/                              # 观测、奖励、事件、几何和终止条件
│   ├── osc_pose6d_env_cfg.py             # 6D OSC 基础环境
│   ├── osc_curriculum_env_cfg.py         # 单孔随机化课程与 23 mm 环境
│   └── peg_in_hole_array_env_cfg.py      # 六孔阵列基础环境
├── scripts/
│   ├── train_phase1.py                   # 最终统一训练入口
│   ├── eval_checkpoint_simple.py         # 最终 PPO 评估入口
│   ├── eval_geometric_controller.py      # 几何教师评估入口
│   ├── geometric_teacher.py              # 解析几何教师
│   ├── residual_policy.py                # residual actor-critic
│   ├── residual_ppo.py                   # residual PPO 损失
│   ├── action_perturbation.py            # 动作噪声和增益扰动 wrapper
│   ├── observation_perturbation.py       # episode 固定孔位估计偏差 wrapper
│   ├── depth_safety.py                   # 可选预测深度屏障
│   ├── visualize.py                      # 四种报告协议的采集与 GUI 回放
│   ├── visualize_task2_six_holes.py      # 任务2逐孔采集和回放
│   ├── plot_task2_visualization.py       # 生成报告图表
│   └── verify_env.py                     # 环境 smoke test
├── runs/ppo/                              # 保留的训练目录和 checkpoint
├── runs/eval/                             # 正式评估与诊断 JSON/Markdown
├── artifacts/                             # 图表、截图和可复用轨迹
├── 任务1_任务2_总结报告.md                # 最终实验报告
└── 项目文件清理说明.md                    # 保留/删除文件说明
```

## 8. 主要代码文件用法

| 文件 | 用途 | 是否直接运行 |
|---|---|---|
| `scripts/train_phase1.py` | 单孔/六孔 Pose6D residual PPO 训练 | 是 |
| `scripts/eval_checkpoint_simple.py` | 使用 `RslRlVecEnvWrapper` 评估 PPO checkpoint | 是，正式 PPO 入口 |
| `scripts/eval_geometric_controller.py` | 评估解析几何教师，可输出逐 episode JSON | 是 |
| `scripts/visualize.py` | 按具名协议先无头采集严格成功轨迹，再打开 GUI 回放 | 是 |
| `scripts/visualize_task2_six_holes.py` | 对 hole 0–5 分别采集一条成功轨迹并连续回放 | 是 |
| `scripts/plot_task2_visualization.py` | 从正式结果 JSON 重新生成总结图 | 是 |
| `scripts/record_video.py` | 兼容入口，保存协议轨迹和首末帧截图 | 是，但当前不编码 MP4 |
| `scripts/build_fixed_peg_asset.py` | 重新生成 `panda_with_fixed_peg.usda` | 可选，运行会覆盖该资产 |
| `scripts/setup_env.sh` | 激活 Conda 并补充 GUI 动态库路径 | 使用 `source` |
| `scripts/geometric_teacher.py` | 教师动作函数 | 由训练/评估导入 |
| `scripts/residual_policy.py` | residual 网络定义 | 由训练/评估导入 |
| `scripts/residual_ppo.py` | residual penalty PPO | 由训练导入 |
| `scripts/action_perturbation.py` | 动作扰动 wrapper | 由训练/评估/可视化导入 |
| `scripts/observation_perturbation.py` | 固定孔位偏差 wrapper | 由训练/评估/可视化导入 |
| `scripts/depth_safety.py` | 可选深度安全屏障 | 由训练/评估导入 |
| `scripts/compute_init_pose.py` | 早期初始姿态开发工具 | 非最终复现入口，含本机路径 |

查看任一入口的完整参数：

```bash
python scripts/train_phase1.py --help
python scripts/eval_checkpoint_simple.py --help
python scripts/eval_geometric_controller.py --help
python scripts/visualize.py --help
```

## 9. 训练

训练默认无 GUI，并将输出写入 `runs/ppo/<stage>/<timestamp>/`。每个训练目录包含：

- `run_config.json`：完整训练协议；
- `model_*.pt`：checkpoint；
- TensorBoard event；
- 训练时源代码差异记录。

### 9.1 任务1最终强扰动训练

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_phase1.py \
  --task Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0 \
  --num_envs 1024 --episode_length_s 20 \
  --num_steps_per_env 64 --max_iterations 500 --save_interval 50 \
  --seed 42 --device cuda:0 --headless \
  --learning_rate 1e-5 --num_learning_epochs 1 \
  --entropy_coef 0 --init_noise_std 0.03 \
  --nullspace_stiffness 20 \
  --geometric_teacher_residual --residual_penalty_coef 100 \
  --teacher_alignment_gate_mm 1 \
  --teacher_observation_noise_mm 0 --teacher_tilt_noise_deg 0 \
  --hole_xy_bias_std_mm 0.5 \
  --action_noise_std 0.02 --action_gain_noise_std_pct 5 \
  --reset_warmup_steps 20 \
  --log_root runs/ppo/stress
```

已保留的最终训练：[runs/ppo/stress/custom/20260818_164739/](runs/ppo/stress/custom/20260818_164739/)。

### 9.2 任务2最终强扰动训练

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_phase1.py \
  --task Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0 \
  --num_envs 1024 --episode_length_s 20 \
  --num_steps_per_env 64 --max_iterations 500 --save_interval 50 \
  --seed 42 --device cuda:0 --headless \
  --learning_rate 1e-5 --num_learning_epochs 1 \
  --entropy_coef 0 --init_noise_std 0.03 \
  --nullspace_stiffness 20 \
  --geometric_teacher_residual --residual_penalty_coef 100 \
  --teacher_alignment_gate_mm 2 \
  --teacher_observation_noise_mm 0 --teacher_tilt_noise_deg 0 \
  --hole_xy_bias_std_mm 0.5 \
  --action_noise_std 0.02 --action_gain_noise_std_pct 5 \
  --reset_warmup_steps 0
```

已保留的最终训练：[runs/ppo/custom/20260818_232714/](runs/ppo/custom/20260818_232714/)。

注意：任务2训练使用 `action_noise_std=0.02`，最终公平评估使用更强的 `action_noise_std=0.0305`。不要把两者混为同一参数。

### 9.3 快速训练链路检查

以下命令只验证训练链路，不能作为正式结果：

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/train_phase1.py \
  --task Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0 \
  --geometric_teacher_residual --residual_penalty_coef 100 \
  --smoke
```

### 9.4 TensorBoard

```bash
tensorboard --logdir runs/ppo --port 6006
```

## 10. 评估

### 10.1 重要评估约束

最终 PPO 必须使用 [scripts/eval_checkpoint_simple.py](scripts/eval_checkpoint_simple.py) 中的 `RslRlVecEnvWrapper` 路径。不要改用直接 raw-env 推理路径；该路径曾在第二个 OSC 控制步触发 Isaac 底层提前退出，所得结果不属于正式评估。

正式比较必须保持一致：

- seed、episode 数和并行环境数；
- 20 秒 / 600 步预算；
- 成功门限；
- 孔位偏差、动作噪声、增益扰动；
- teacher alignment gate；
- 动作延迟为 0。

### 10.2 任务1几何教师

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/eval_geometric_controller.py \
  --task Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0 \
  --episodes 128 --num_envs 64 --seed 42 \
  --episode_length_s 20 --max_steps 600 \
  --alignment_gate_mm 1 \
  --hole_xy_bias_std_mm 0.5 \
  --action_noise_std 0.02 --action_gain_noise_std_pct 5 \
  --action_delay_steps 0 \
  --output runs/eval/task1_teacher_recheck.json
```

### 10.3 任务1最终 PPO

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/eval_checkpoint_simple.py \
  --checkpoint runs/ppo/stress/custom/20260818_164739/model_499.pt \
  --task Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0 \
  --episodes 128 --num_envs 64 --seed 42 \
  --episode_length_s 20 --nullspace_stiffness 20 \
  --geometric_teacher_residual --teacher_alignment_gate_mm 1 \
  --teacher_observation_noise_mm 0 --teacher_tilt_noise_deg 0 \
  --hole_xy_bias_std_mm 0.5 \
  --action_noise_std 0.02 --action_gain_noise_std_pct 5 \
  --reset_warmup_steps 20
```

### 10.4 任务2最终几何教师

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/eval_geometric_controller.py \
  --task Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0 \
  --episodes 128 --num_envs 64 --seed 42 \
  --episode_length_s 20 --max_steps 600 \
  --target_hole_id -1 --alignment_gate_mm 2 \
  --hole_xy_bias_std_mm 0.5 \
  --action_noise_std 0.0305 --action_gain_noise_std_pct 5 \
  --action_delay_steps 0 \
  --output runs/eval/task2_teacher_recheck.json
```

将 `--target_hole_id -1` 改为 `0–5` 可以单独评估指定目标孔。

### 10.5 任务2最终 PPO

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/eval_checkpoint_simple.py \
  --checkpoint runs/ppo/custom/20260818_232714/model_499.pt \
  --task Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0 \
  --episodes 128 --num_envs 64 --seed 42 \
  --episode_length_s 20 --nullspace_stiffness 20 \
  --geometric_teacher_residual --teacher_alignment_gate_mm 2 \
  --teacher_observation_noise_mm 0 --teacher_tilt_noise_deg 0 \
  --hole_xy_bias_std_mm 0.5 \
  --action_noise_std 0.0305 --action_gain_noise_std_pct 5 \
  --reset_warmup_steps 0
```

`eval_checkpoint_simple.py` 将结果打印到终端，不直接写 JSON。仓库中的正式归档 JSON 同时记录了运行协议、最终统计和 checkpoint 路径。

## 11. GUI 可视化

[scripts/visualize.py](scripts/visualize.py) 使用两阶段隔离流程：

1. 在无 GUI 的批量环境中，按指定协议真实运行策略；
2. 只保存满足严格成功判据的轨迹；
3. 关闭采集进程；
4. 打开新的 Isaac Sim GUI，回放记录的关节和夹具状态。

因此 GUI 中的运动已经包含扰动造成的结果，但回放阶段不会再次施加第二遍随机扰动。

可用预设：

| 预设 | 对应实验 | 孔位偏差 | 动作噪声 | 增益扰动 |
|---|---|---:|---:|---:|
| `task1_basic` | 任务1基础 512/512 | 教师均匀观测噪声 0.5 mm/0.25° | 0 | 0 |
| `task1_stress` | 任务1 PPO 124/128 | episode 固定 XY 0.5 mm | 0.02 | 5% |
| `task2_basic` | 任务2基础 512/512 | 教师均匀观测噪声 0.5 mm/0.25° | 0 | 0 |
| `task2_fair` | 任务2 PPO 128/128 | episode 固定 XY 0.5 mm | 0.0305 | 5% |

任务1最终协议：

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/visualize.py \
  --protocol task1_stress --episodes 3 --capture_num_envs 64 \
  --show_fixture_plate \
  --screenshot_dir artifacts/task1_visualization/stress \
  --trajectory_output artifacts/task1_visualization/task1_stress.pt
```

任务2最终公平协议：

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/visualize.py \
  --protocol task2_fair --episodes 3 --capture_num_envs 64 \
  --show_fixture_plate \
  --screenshot_dir artifacts/task2_visualization/fair \
  --trajectory_output artifacts/task2_visualization/task2_fair.pt
```

任务2六个目标孔逐孔回放：

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/visualize_task2_six_holes.py \
  --protocol task2_fair --capture_num_envs 128 \
  --show_fixture_plate \
  --screenshot_dir artifacts/task2_visualization/fair_six_holes \
  --trajectory_output artifacts/task2_visualization/task2_fair_six_holes.pt
```

`--show_fixture_plate` 默认启用：任务1显示带圆孔的固定方形孔板，任务2显示六个固定孔套筒。显示几何位于独立的 `/World/ReplayVisuals` 节点，仅用于回放，不改变采集轨迹使用的碰撞物理。

其他常用参数：

- `--speed 0.5`：回放速度倍率；
- `--replay_fps 60`：GUI 回放帧率；
- `--initial_hold_seconds 3`：初始姿态停留时间；
- `--hold_seconds 5`：最终姿态停留时间；
- `--show_table`：仅在回放中添加桌面视觉资产；
- `--no-show_fixture_plate`：隐藏回放辅助孔板/孔套筒。

### 11.1 `record_video.py` 的实际行为

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/record_video.py \
  --protocol task2_fair --episodes 3 \
  --output artifacts/protocol_replay/task2_fair
```

该兼容入口调用 `visualize.py`，保存验证轨迹、协议 JSON 和首末帧截图。尽管文件名为 `record_video.py`，当前实现不编码 MP4。

## 12. 生成报告图表

```bash
python scripts/plot_task2_visualization.py \
  --output-dir artifacts/task2_visualization
```

该命令读取正式评估 JSON，生成：

- `task2_array_layout.png`：六孔布局与随机范围；
- `task2_success_summary.png`：任务2基础/教师/PPO 对照；
- `final_results_summary.png`：任务1和任务2最终总览；
- `manifest.json`：图片及其来源记录。

## 13. 已保留模型

| 用途 | 模型 | 结果 |
|---|---|---:|
| 任务1基础 PPO | `runs/ppo/hole20_reward/20260818_130108/model_49.pt` | 512/512 |
| 任务1最终强扰动 PPO | `runs/ppo/stress/custom/20260818_164739/model_499.pt` | 124/128 |
| 任务2基础 PPO | `runs/ppo/custom/20260818_141826/model_49.pt` | 512/512 |
| 任务2最终公平 PPO | `runs/ppo/custom/20260818_232714/model_499.pt` | 128/128 |

最终模型 SHA256：

```text
任务1 model_499.pt  8d7a1c66047dc433d79b8f522169b4bb1a7cd6d9706c32abae6c9f50f3334386
任务2 model_499.pt  1ff2b0cc0a3f2142fe93e20940be34270bd31d55cfddc515d129f10e3403b340
```

验证模型文件：

```bash
sha256sum \
  runs/ppo/stress/custom/20260818_164739/model_499.pt \
  runs/ppo/custom/20260818_232714/model_499.pt
```

## 14. 结果解读与历史对照

- 任务1基础协议中几何教师已经达到 100%，此时 PPO 的价值不是提高上限，而是学习扰动补偿；
- 加入 0.5 mm episode 固定孔位偏差、动作噪声和 5% 增益扰动后，任务1 PPO 从教师的 92.97% 提升到 96.88%；
- 任务2不是简单重复六次任务1，它还包含目标孔条件化、阵列整体随机和非目标孔共存碰撞；
- 任务2早期 `1.56%/2.34%` 来源于 12 秒预算、1 mm 教师门限和不一致评估路径，不是最终能力；
- 改进前任务2 PPO 在公平强扰动下约为 50.78%；改用任务1验证过的保守 PPO 更新配置后达到 100%；
- 最终 PPO 的作用是教师附近的 residual adaptation，而不是直接从零 RL。

早期错误协议仍保留为诊断证据，但不应与最终结果直接比较。保留范围见 [项目文件清理说明.md](项目文件清理说明.md)。

## 15. 常见问题

### GUI 启动但看不到窗口

确认当前会话有有效的 `DISPLAY`，然后执行：

```bash
source scripts/setup_env.sh
```

如果仍有动态库错误，检查 `scripts/setup_env.sh` 中的 Conda 和 Omniverse 路径是否与本机一致。

### 回放中看不到固定孔板或六孔

使用 `--show_fixture_plate`。当前回放器会在每帧场景同步后恢复 `/World/ReplayVisuals`，并自动兼容旧轨迹中的并行环境坐标偏移。

### 评估结果明显低于报告

优先检查：

1. checkpoint 是否正确；
2. 是否使用 `RslRlVecEnvWrapper` 的正式评估入口；
3. episode 是否为 20 秒；
4. 任务2 teacher alignment gate 是否为 2 mm；
5. 训练噪声 `0.02` 与任务2公平评估噪声 `0.0305` 是否混淆；
6. 动作延迟是否保持为 0；
7. seed、并行环境数和总 episode 数是否一致。

### 显存不足

减小 `--num_envs` 或 `--capture_num_envs`，例如从 1024/128 减为 256/32。这样可以完成调试，但不再与报告中的正式批量协议完全相同。

### 是否可以直接部署到真实 FR3

不可以直接部署。实机前至少需要：

- 导入并校准正式 FR3 模型；
- 对接真实关节、末端和力/触觉接口；
- 重新标定 Peg/flange 外参；
- 建立碰撞、速度、力和紧急停止安全层；
- 进行 sim-to-real 随机化和低速分阶段验证。

## 16. 相关文档

- [项目实施报告.md](项目实施报告.md)：面向汇报的学术论文式完整实施报告；
- [任务1_任务2_总结报告.md](任务1_任务2_总结报告.md)：最终方法与实验报告；
- [项目文件清理说明.md](项目文件清理说明.md)：保留模型、诊断结果和已删除内容；
- [Day1_推进报告.md](Day1_推进报告.md) 至 [Day6_工作记录.md](Day6_工作记录.md)：完整推进过程。

## 17. 许可证

本项目软件代码采用 [MIT License](LICENSE) 发布。项目使用的 NVIDIA Isaac Sim、Isaac Lab、PyTorch、RSL-RL 及其他第三方组件仍分别遵循其各自的许可证和使用条款。
