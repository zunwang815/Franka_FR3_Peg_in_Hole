# Franka FR3 Peg-in-Hole 仿真任务 — Day 5 工作记录

> 日期：2026-08-16  
> 主题：修复 baseline 过度插入与并行环境数依赖

## Day5 精简总结与 Day6 入口

### 最终结论

Day5 已经完成了两件关键事情：

1. 修复了成功定义、深度屏障、动作尺度和评估口径，baseline/offset5 的严格评估链路
   已稳定，offset5 仍可达到 100%。
2. 证明 hole20、±50 mm 工作空间以及最终 23 mm 孔在动力学上可达。新增的几何控制器在
   准确孔中心条件下均达到 `512/512=100%`，因此当前 PPO 失败不是机械臂不可达，而是
   随机 IK 后的关节构型、Jacobian 和策略动作分布不一致。

### 失败尝试的统一归因

此前 hole10/hole20 的大量 PPO、奖励、屏障、XY assist、IK reset 和 residual transfer
实验不再作为后续主线，统一归纳为三类：

```text
过深插入       → 已由物理成功窗口、深度屏障和门控消除
XY/姿态泛化    → 简单奖励塑形、XY 覆盖和动作 assist 无法解决
IK 后策略失配   → canonical observation、残差 adapter、KL/惩罚和冻结 critic 均不足
```

这些实验保留在第十三节作为审计档案，但明日不再继续在旧 PPO checkpoint 上缝补。

### 当前可靠基线

```text
offset5 PPO：原环境严格 100%
hole20 几何控制器：512/512 = 100%
hole50 几何控制器：512/512 = 100%
hole23mm、±50mm 几何控制器：512/512 = 100%
```

几何控制器使用准确孔中心，是 oracle 基线，不代表含噪感知条件下的最终方案；但它已经
把“可达性/动力学问题”和“感知噪声/策略泛化问题”分离开。

### Day6 计划

按以下顺序推进，不再直接从 offset5 checkpoint 迁移完整 PPO：

1. 固化几何控制器和 512 回合评估协议，增加轨迹与失败原因记录。
2. 逐级加入孔中心位置噪声、姿态噪声、末端观测噪声和动力学扰动，先找出几何控制器
   的失效边界。
3. 在 hole20 上让强化学习只学习几何控制器的 residual，保留几何控制器作为 teacher；
   先验证含噪 20 mm，再扩大到 ±50 mm。
4. 最后在 23 mm 孔、±50 mm 初始化空间上做含噪验证和最终评估。

明日的接受标准仍为：成功率、timeout、over-insertion 分开统计；任何“成功率”都必须
基于固定 seed、512 回合和真实 termination reason。

## 一、Day4 结论与今日优先级

Day4 已打通 Fixed-Joint Peg、Oracle、PPO 训练/评估与隔离回放链路，但存在两个 P0 可信度问题：

1. 成功深度 70.27 mm，超过 40 mm 物理孔深；
2. 旧策略在 128 环境成功，在 1 环境超时。

因此今日没有直接推进 offset5/hole10 课程，而是先修复成功定义、动作尺度和评估证据链。

## 二、物理成功与失败定义

修改后的 baseline 判据：

```text
radial <= 2 mm
15 mm <= insertion depth <= 40 mm
tilt <= 2 deg
depth > 42 mm -> over_insertion failure
```

40–42 mm 之间仅作为数值容差区，不计成功。评估器改为根据真实 termination reason 区分 success、over-insertion、其他失败和 timeout，不再将所有 `terminated` 误计为成功。

## 三、旧策略诊断与动作尺度

旧 `model_250.pt` 在新深度判据、旧 20 mm 平移尺度下：

```text
16 episodes: 0 success, 16 over-insertion failures
average steps: 6
max |delta z|: 19.96 mm/control-step
```

证明原 7 步成功策略依赖大动作跨过物理深度。平移动作尺度因此从 20 mm 收紧至 5 mm。收紧后旧模型不再过穿，但 16/16 超时，说明必须重新训练。

## 四、修正后 baseline 训练

Smoke 训练：64 环境、10 iterations，训练、日志、TensorBoard 和 checkpoint 链路全部通过。

正式训练：

```text
runs/ppo/baseline/20260816_154004
num_envs = 1024
iterations = 500
seed = 42
```

约 76 iteration 首次成功，约 227 iteration 训练窗口首次达到 98%，后半程多数窗口为 98–99%+，最后窗口为 99.4%。

## 五、128 环境正式评估

在每个 checkpoint 1024 episode 的严格协议下：

| Checkpoint | Success | Over/Failure | Avg steps | Max abs Z step |
|---|---:|---:|---:|---:|
| model_250.pt | 1024/1024 | 0/0 | 16 | 4.47 mm |
| model_300.pt | 1024/1024 | 0/0 | 12 | 6.43 mm |
| model_350.pt | 1024/1024 | 0/0 | 9 | 9.17 mm |
| model_400.pt | 1024/1024 | 0/0 | 6 | 13.89 mm |
| model_450.pt | 1024/1024 | 0/0 | 7 | 14.04 mm |
| model_499.pt | 1024/1024 | 0/0 | 6 | 17.71 mm |

六个模型的 Wilson CI95 均为 `[99.63%, 100%]`。同成功率下优先选择单步 Z 位移最小的模型，当前推荐：

```text
runs/ppo/baseline/20260816_154004/model_250.pt
```

该模型终止深度为 15.575–15.577 mm，未超过物理孔深。正式报告：

```text
runs/eval/baseline/20260816_day5_strict.json
```

## 六、并行环境数一致性

对推荐模型每档评估 128 episode：

| num_envs | Success | Avg steps | Over | Failure | Timeout |
|---:|---:|---:|---:|---:|---:|
| 1 | 128/128 | 16 | 0 | 0 | 0 |
| 2 | 128/128 | 16 | 0 | 0 | 0 |
| 4 | 128/128 | 16 | 0 | 0 | 0 |
| 8 | 128/128 | 16 | 0 | 0 | 0 |
| 16 | 128/128 | 16 | 0 | 0 | 0 |
| 32 | 128/128 | 16 | 0 | 0 | 0 |
| 64 | 128/128 | 16 | 0 | 0 | 0 |
| 128 | 128/128 | 16 | 0 | 0 | 0 |

报告：

```text
runs/eval_matrix/baseline/20260816_day5_model250.json
```

## 七、新增评估能力

- 输出 episode 最小/最大/终止深度；
- 输出峰值接触力和最大单步 Z 位移；
- 区分 success、timeout、over-insertion 和其他失败；
- 同成功率下优先选择更温和的轨迹；
- 新增 `scripts/eval_num_envs_matrix.py` 用于自动生成环境数一致性矩阵。

## 八、仍待解决

### 8.1 接触力审计（已完成）

在与训练/评估相同的 120 Hz、solver 16/4 物理协议下：

| 对照 | 结果 | 峰值接触力 |
|---|---|---:|
| 居中 Oracle 插入 | 成功 | 0.00 N |
| X 偏置 6 mm Oracle 下压 | 被阻挡，最大深度约 0 mm | 6.12 N |
| PPO model_250 严格成功帧 | radial=0.81 mm, depth=15.58 mm | 0.00 N |

30 mm baseline 孔与 20 mm Peg 的理论径向间隙约为 5 mm。PPO 成功时仅有 0.81 mm 径向误差，仍保留约 4.19 mm 几何间隙，因此 0 N 是合理的无接触顺畅插入，不是传感器漏检。6 mm 偏置反例同时证明孔壁碰撞和 Peg 接触传感器有效。

### 8.2 其余待办

1. 当前仍是 Panda 等效资产，不是官方 FR3；
2. offset5 已通过；hole10 已训练至 85.45% 但未达 90%，hole50/hole25/hole23 尚未重新训练；
3. 课程训练前建议先对推荐 baseline 做一次新判据下的隔离可视化回放。

## 九、当前结论

Day4 的两个 P0 风险已经完成修复和批量验证：baseline 不再过度插入，且在 1–128 环境数下表现一致。接触力对照也已证明无接触成功和有接触阻挡均符合几何与物理预期。当前可称为“物理深度受限且环境数一致的 Pose6D baseline”。

## 十、offset5 PPO 课程

### 10.1 零样本与默认 transfer 诊断

baseline `model_250.pt` 在 offset5 上的严格零样本结果：

```text
400/1024 = 39.06%
624 over-insertion failures
```

首次使用默认 PPO 更新（learning rate 3e-4、8 epochs、entropy 0.01）进行 transfer，第一训练窗口约 68%，但随后迅速退化。训练至 model_250 后严格评估为 0/1024，证明强更新造成 transfer catastrophe。该试跑已停止，不用于课程传递。

### 10.2 保守 transfer 训练

训练入口新增可记录的 PPO override，最终采用：

```text
learning_rate = 3e-5
num_learning_epochs = 2
entropy_coef = 0
num_envs = 1024
iterations = 500
source = baseline/model_250.pt
```

训练目录：

```text
runs/ppo/offset5/20260816_164356
```

保守 transfer 保留了初始 68% 能力，约 187 iteration 训练窗口达到 90%，后续逐步提升到 95% 左右，末窗口为 95.8%。

### 10.3 严格评估

| Checkpoint | Success | Over-insertion | Avg steps | Max abs Z step |
|---|---:|---:|---:|---:|
| model_250.pt | 935/1024 (91.31%) | 89 | 16.3 | 6.10 mm |
| model_300.pt | 958/1024 (93.55%) | 66 | 16.3 | 6.29 mm |
| model_350.pt | 1001/1024 (97.75%) | 23 | 14.9 | 6.72 mm |
| model_400.pt | 1009/1024 (98.54%) | 15 | 14.6 | 6.93 mm |
| model_450.pt | 1000/1024 (97.66%) | 24 | 14.0 | 7.12 mm |
| model_499.pt | 1022/1024 (99.80%) | 2 | 14.2 | 7.11 mm |

推荐 offset5 checkpoint：

```text
runs/ppo/offset5/20260816_164356/model_499.pt
```

其 Wilson CI95 为 `[99.29%, 99.95%]`，通过 90% 验收门槛。剩余 2 次失败均为过穿，最大终止深度约 43.80 mm，未被误计为成功。正式报告：

```text
runs/eval/offset5/20260816_day5_strict.json
```

## 十一、hole10 PPO 课程

### 11.1 第一轮保守 transfer

从 offset5 `model_499.pt` 迁移，使用 `learning_rate=3e-5`、`num_learning_epochs=2`、`entropy_coef=0`训练 500 iteration。训练目录：

```text
runs/ppo/hole10/20260816_171832
```

严格评估最佳为 `model_450.pt`：823/1024（80.37%），201 次失败全部为 over-insertion，未达 90% 验收线。

### 11.2 过深奖励诊断与修正

过深终止原先没有独立负奖励，策略可能因提前结束 episode 而不会被充分惩罚。新增 `over_insertion_penalty`。试验表明 `-500` 配合 `1e-4` 学习率会导致 transfer 价值函数冲击，策略在 40--108 iteration 降至 0% 成功，因此终止该无效分支。

最终采用温和终止惩罚 `-25`，并恢复 `3e-5` 学习率，从第一轮 `model_450.pt` 继续训练 300 iteration：

```text
runs/ppo/hole10/20260816_173610
```

### 11.3 第二轮严格评估

| Checkpoint | Success | Over-insertion | Avg steps | Max abs Z step |
|---|---:|---:|---:|---:|
| model_50.pt | 860/1024 (83.98%) | 164 | 20.5 | 10.10 mm |
| model_100.pt | 836/1024 (81.64%) | 188 | 20.5 | 9.91 mm |
| model_150.pt | 832/1024 (81.25%) | 192 | 19.6 | 9.97 mm |
| model_200.pt | 818/1024 (79.88%) | 206 | 19.5 | 10.11 mm |
| model_250.pt | 826/1024 (80.66%) | 198 | 18.3 | 10.41 mm |
| model_299.pt | 869/1024 (84.86%) | 155 | 17.7 | 10.42 mm |

当前 hole10 最佳 checkpoint：

```text
runs/ppo/hole10/20260816_173610/model_299.pt
```

Wilson CI95 为 `[82.54%, 86.93%]`。它比第一轮最佳提高 4.49 个百分点，over-insertion 从 201 降至 155，但仍未达 90% 验收线，因此暂不向 hole50 传递。分 checkpoint 报告位于：

```text
runs/eval/hole10/v2/
```

### 11.4 第三段继续训练与回退

从第二轮 `model_299.pt` 继续相同保守参数。训练窗口降至 74--79%，因此在 158 iteration 主动停止。严格评估为：

| Checkpoint | Success | Over-insertion |
|---|---:|---:|
| model_50.pt | 814/1024 (79.49%) | 210 |
| model_100.pt | 787/1024 (76.86%) | 237 |
| model_150.pt | 845/1024 (82.52%) | 179 |

三个点均退化，不采用；最佳模型仍回退为第二轮 `model_299.pt` (84.86%)。报告位于 `runs/eval/hole10/pass3/`。

### 11.5 `-100` 过深终止惩罚对照

从当前最佳 `model_299.pt` 出发，将 over-insertion 惩罚从 `-25` 提高到 `-100`，同时将 PPO 更新限制为 `learning_rate=1e-5`、`num_learning_epochs=1`，训练 150 iteration。

| Checkpoint | Success | Over-insertion |
|---|---:|---:|
| model_25.pt | 869/1024 (84.86%) | 155 |
| model_50.pt | 860/1024 (83.98%) | 164 |
| model_75.pt | 860/1024 (83.98%) | 164 |
| model_100.pt | 845/1024 (82.52%) | 179 |
| model_125.pt | 848/1024 (82.81%) | 176 |
| model_149.pt | 855/1024 (83.50%) | 169 |

`-100` 未降低越界：25 iteration 时与原模型持平，后续更新逐步退化。该实验不采用，配置回退至 `-25`，推荐下一步改为越界前的连续深度制动奖励，而不是继续增大稀疏终止冲击。训练目录为 `runs/ppo/hole10/20260816_175847`，评估报告位于 `runs/eval/hole10/penalty100/`。

### 11.6 从 offset5 重训：连续制动 + 终止惩罚

为避免旧 hole10 策略的局部最优，从 `offset5/model_499.pt` 重新 transfer。实验奖励为 35--42 mm 线性制动 `-20` 及越界终止 `-100`，累计训练约 500 iteration。首段目录为 `runs/ppo/hole10/20260816_181059`，续训目录为 `runs/ppo/hole10/20260816_182210`。

续训严格评估：

| Checkpoint | Success | Over-insertion | Timeout |
|---|---:|---:|---:|
| model_250.pt | 761/1024 (74.32%) | 232 | 31 |
| model_300.pt | 809/1024 (79.00%) | 160 | 55 |
| model_350.pt | 771/1024 (75.29%) | 242 | 11 |
| model_400.pt | 751/1024 (73.34%) | 261 | 12 |
| model_450.pt | 795/1024 (77.64%) | 201 | 28 |
| model_499.pt | 838/1024 (81.84%) | 150 | 36 |

连续制动将最大 Z 步长从旧最佳的 10.42 mm 降至 9.44 mm，over-insertion 从 155 降至 150，证明制动信号生效；但同时产生 36 次超时，成功率只有 81.84%。该 `-20/-100` 组合不采用，默认奖励回退至终止惩罚 `-25`，当前最佳仍为 `runs/ppo/hole10/20260816_173610/model_299.pt` (84.86%)。报告位于 `runs/eval/hole10/fresh_braking/` 和 `runs/eval/hole10/fresh_braking_cont/`。

### 11.7 弱连续制动 `-5` 对照

从 offset5 `model_499.pt` 重训 500 iteration，奖励为 35--42 mm 连续制动 `-5` 与终止惩罚 `-100`。

| Checkpoint | Success | Over-insertion | Timeout |
|---|---:|---:|---:|
| model_250.pt | 843/1024 (82.32%) | 122 | 59 |
| model_300.pt | 796/1024 (77.73%) | 186 | 42 |
| model_350.pt | 839/1024 (81.93%) | 168 | 17 |
| model_400.pt | 806/1024 (78.71%) | 218 | 0 |
| model_450.pt | 798/1024 (77.93%) | 226 | 0 |
| model_499.pt | 842/1024 (82.23%) | 177 | 5 |

弱制动可将越界最低压至 122，但对应产生 59 次超时，最高成功率 82.32% 仍低于旧最佳 84.86%。该模型不采用。训练目录为 `runs/ppo/hole10/20260816_183750`，报告位于 `runs/eval/hole10/braking5/`。

### 11.8 从 offset5 重训：仅 `-100` 终止惩罚

移除连续制动，仅保留 over-insertion 终止惩罚 `-100`，从 offset5 `model_499.pt` 重训 500 iteration。

| Checkpoint | Success | Over-insertion | Timeout | Max abs Z step |
|---|---:|---:|---:|---:|
| model_250.pt | 831/1024 (81.15%) | 180 | 13 | 8.48 mm |
| model_300.pt | 843/1024 (82.32%) | 171 | 10 | 8.51 mm |
| model_350.pt | 875/1024 (85.45%) | 142 | 7 | 8.74 mm |
| model_400.pt | 820/1024 (80.08%) | 204 | 0 | 8.85 mm |
| model_450.pt | 854/1024 (83.40%) | 170 | 0 | 9.40 mm |
| model_499.pt | 808/1024 (78.91%) | 216 | 0 | 9.47 mm |

`model_350.pt` 比旧最佳多成功 6 个 episode，over-insertion 从 155 降至 142，最大 Z 步长从 10.42 mm 降至 8.74 mm，因此成为新的 hole10 推荐 checkpoint：

```text
runs/ppo/hole10/20260816_185829/model_350.pt
```

它仍未达 90% 验收线，不向 hole50 传递。评估报告位于 `runs/eval/hole10/fresh_terminal100/`，默认终止惩罚保留为 `-100`。

### 11.9 XYZ 各向异性动作尺度试验

审计发现 Isaac Lab 的 OSC 实现可对 `position_scale` 向量广播，但派生 `configclass` 中的嵌套赋值在实例化后被恢复为 `0.005`。运行时探针首先证明历史 A/B 并未真正改变尺度，随后通过解析后协议覆盖确认实际尺度为 `(0.005, 0.005, 0.002)`。

`-100` 终止惩罚在新尺度下训练至约 70% 后于 100 iteration 左右坍塌为 0%，已停止。改用 `-25` 重训 500 iteration 的严格评估为：

| Checkpoint | Success | Over-insertion | Timeout | Max abs Z step |
|---|---:|---:|---:|---:|
| model_50.pt | 412/1024 (40.23%) | 12 | 600 | 2.72 mm |
| model_100.pt | 343/1024 (33.50%) | 605 | 76 | 2.98 mm |
| model_200.pt | 658/1024 (64.26%) | 366 | 0 | 4.31 mm |
| model_300.pt | 716/1024 (69.92%) | 308 | 0 | 4.56 mm |
| model_400.pt | 693/1024 (67.68%) | 331 | 0 | 4.83 mm |
| model_499.pt | 697/1024 (68.07%) | 327 | 0 | 4.96 mm |

2 mm Z 指令尺度初期将越界降至 12，但导致 600 次超时；后期策略增强下压后，动力学单步位移回升且越界大量恢复。该尺度不采用，运行时覆盖已撤销，恢复 5 mm 标量尺度和 `-100` 终止惩罚。训练目录为 `runs/ppo/hole10/20260816_195203`，报告位于 `runs/eval/hole10/xyz552_penalty25/`。

### 11.10 decimation=2（60 Hz 控制）试验

在保持 120 Hz 物理频率时，将 hole10 的 decimation 从 4 降至 2。旧最佳 `model_350.pt` 零训练 A/B 为 689/1024（67.29%）、332 次 over-insertion，最大 Z 步长从 8.74 mm 降至 4.82 mm。这证明控制频率覆盖真正生效，但旧策略与新动作语义不兼容。

从 offset5 重训时，`-100` 终止惩罚组合在约 70 iteration 坍塌为 0%；改为 `-25` 后仍在约 80 iteration 坍塌为 0%。两个无效分支均已停止，说明现有 PPO/奖励标定与 60 Hz 控制不兼容。decimation=2 运行时协议已撤销，恢复 decimation=4 和 `-100` 终止惩罚。零训练报告为 `runs/eval/hole10/20260816_decimation2_ab.json`。

## 十二、hole50 跳级试验

在 hole10 尚未达 90% 时，使用当前最佳 `runs/ppo/hole10/20260816_185829/model_350.pt` 实验性跳级至 hole50。hole50 保持 30 mm 物理孔径，但将孔位置随机范围从 +/-10 mm 扩大到 +/-50 mm。

零训练严格基线：

```text
58/1024 = 5.66%
631 over-insertion
335 timeout
Wilson CI95 = [4.41%, 7.25%]
```

报告为 `runs/eval/hole50/20260816_hole10_model350_zeroshot.json`。

保守 transfer 的 smoke 含少量成功，但正式 `-100` 分支在约 55 iteration 降到持续 0%；改用 `-25` 后仍在约 60 iteration 降到持续 0%。两个分支均已停止，不产生推荐 hole50 checkpoint。

结论：hole10 -> hole50 的位置范围从 20x20 mm 跳到 100x100 mm，零训练成功率仅 5.66%，成功样本不足以维持 PPO transfer。后续应加入中间课程，例如 +/-20 mm -> +/-30 mm -> +/-50 mm，而不是直接跳级。默认配置已恢复 decimation=4 和 over-insertion `-100`。

## 十三、hole20 中间课程

新增 `hole20` 和 `hole30` 任务注册，分别将孔位置随机范围扩大到 +/-20 mm 和 +/-30 mm。两个阶段已接入训练、评估和可视化脚本。

hole10 最佳 `model_350.pt` 在 hole20 上的零训练严格基线为：

```text
374/1024 = 36.52%
522 over-insertion
128 timeout
Wilson CI95 = [33.63%, 39.52%]
```

报告为 `runs/eval/hole20/20260816_hole10_model350_zeroshot.json`。该基线明显高于 hole50 的 5.66%，证明中间课程方向合理。

使用 `-100` over-insertion 终止惩罚的正式 transfer 在约 36 iteration 后降至持续 0%，已在 45 iteration 停止，目录为 `runs/ppo/hole20/20260816_203401`。将 hole20 专用终止惩罚降为 `-25` 后，500 iteration 训练稳定完成，目录为 `runs/ppo/hole20/20260816_203946`。

256 episode 筛选显示 `model_50.pt` 为 51.95%、`model_499.pt` 为 49.61%；但 1024 episode 严格复核中排序反转：

| Checkpoint | Success | Over-insertion | Timeout | Wilson CI95 |
|---|---:|---:|---:|---:|
| model_50.pt | 392/1024 (38.28%) | 392 | 240 | [35.35%, 41.30%] |
| model_499.pt | 442/1024 (43.16%) | 582 | 0 | [40.16%, 46.22%] |

`model_499.pt` 相比零训练基线提升 6.64 个百分点，是当前 hole20 候选：

```text
runs/ppo/hole20/20260816_203946/model_499.pt
```

但其失败全部是 over-insertion，且成功率仅 43.16%，尚不适合直接推进 hole30。严格报告为 `runs/eval/hole20/20260816_model_50_strict.json` 和 `runs/eval/hole20/20260816_model_499_strict.json`。hole20 保留专用 `-25` 终止惩罚，其他阶段仍继承默认 `-100`。

### 13.1 预测式深度安全屏障

在训练和评估脚本中新增可选 `--depth_safety_barrier_mm`。屏障根据当前插入深度、上一控制步的向下位移和当前 Z 指令预测下一步深度；超限时仅裁剪 Z 向下动作，保留 XY、姿态修正和向上回撤。默认关闭，不改变历史协议。共享实现位于 `scripts/depth_safety.py`。

hole20 `model_499.pt` 的 1024 episode A/B：

| Barrier | Success | Over-insertion | Timeout | Wilson CI95 |
|---|---:|---:|---:|---:|
| disabled | 442/1024 (43.16%) | 582 | 0 | [40.16%, 46.22%] |
| 37 mm | 501/1024 (48.93%) | 0 | 523 | [45.87%, 51.99%] |
| 38 mm | 520/1024 (50.78%) | 0 | 504 | [47.72%, 53.83%] |
| 39 mm | 520/1024 (50.78%) | 0 | 504 | [47.72%, 53.83%] |

38 mm 保留 2 mm 物理裕量，成功率比无屏障提高 7.62 个百分点，并将过深失败从 582 降为 0。当前推荐组合为：

```text
runs/ppo/hole20/20260816_203946/model_499.pt
--depth_safety_barrier_mm 38
```

报告为 `runs/eval/hole20/20260816_model499_barrier37.json`、`...barrier38.json` 和 `...barrier39.json`。

随后使用屏障以 `1e-5` 学习率微调，在 75 iteration 后快速退化、92 iteration 后连续归零，已于 102 iteration 停止。256 episode 筛选为 `model_25=59.77%`、`model_50=53.52%`、`model_75=16.41%`；但 `model_25` 的 1024 episode 严格复核仅为 509/1024 (49.71%)，低于未微调组合的 50.78%，因此不采用微调 checkpoint。训练目录为 `runs/ppo/hole20/20260816_211524`，严格报告为 `runs/eval/hole20/20260816_barrier38_model25_strict.json`。

### 13.2 对准门控插入

在共享安全层中增加可选对准门控及滞回：未对准时仅禁止向下 Z，仍允许 XY、姿态修正和向上回撤。首次从悬空初始位置就开启 `3 mm/3 deg` 门控时仅得 25/256 (9.77%)，证明旧策略的接近和横向修正是耦合的。

随后增加门控起始深度，允许自由接近和浅插入：

| Gate | Start depth | Episodes | Success | Over-insertion |
|---|---:|---:|---:|---:|
| 3 mm / 3 deg | 0 mm | 256 | 113/256 (44.14%) | 0 |
| 3 mm / 3 deg | 10 mm | 256 | 135/256 (52.73%) | 0 |
| 4 mm / 4 deg | 10 mm | 256 | 159/256 (62.11%) | 0 |
| 4 mm / 4 deg | 10 mm | 1024 | 505/1024 (49.32%) | 0 |

最佳小样本组合在 1024 episode 严格复核中低于单独 38 mm 屏障的 520/1024 (50.78%)，说明 256 episode 的 62.11% 是抽样高估。硬门控会将部分本可成功的接触引导轨迹截断成超时，因此实现保留为可选实验开关，但当前不推荐启用。严格报告为 `runs/eval/hole20/20260816_model499_gate4_depth10_barrier38_strict.json`。

### 13.3 软对准门控

将硬禁止改为连续缩放：10 mm 深度后，向下 Z 动作乘以 `min(1, 4mm/radial_error, 4deg/tilt_error)`，同时保留 38 mm 预测屏障。对最小缩放下限 0.10、0.25、0.50 的 256 episode 筛选均为 157/256 (61.33%)、0 过深，说明实际缩放因子未触发下限。

0.25 下限的 1024 episode 严格复核为 514/1024 (50.20%)，0 过深，Wilson CI95 [47.14%, 53.25%]。该结果仍略低于单独 38 mm 屏障的 520/1024 (50.78%)，因此软门控也不采用为默认方案。报告为 `runs/eval/hole20/20260816_model499_softgate4_depth10_barrier38_strict.json`。

结论：硬/软门控都能保持 0 过深，但没有超过单独深度屏障。剩余问题是策略在屏障前的 XY/姿态恢复能力，而不是继续调整 Z 限制。

### 13.4 hole10 最佳模型的深度屏障复核

将深度屏障应用到 hole10 最佳 `runs/ppo/hole10/20260816_185829/model_350.pt`。原始严格结果为 875/1024 (85.45%)、142 次过深、7 次超时。

| Barrier | Success | Over-insertion | Timeout | Wilson CI95 |
|---|---:|---:|---:|---:|
| disabled | 875/1024 (85.45%) | 142 | 7 | [83.16%, 87.48%] |
| 38 mm | 906/1024 (88.48%) | 0 | 118 | [86.38%, 90.29%] |
| 39 mm | 905/1024 (88.38%) | 0 | 119 | [86.27%, 90.20%] |

38 mm 屏障将 31 个原过深回合转化为成功，成功率提升 3.03 个百分点，并将过深归零；其余失败转为超时。39 mm 没有进一步改善，因此 hole10 当前推荐组合更新为：

```text
runs/ppo/hole10/20260816_185829/model_350.pt
--depth_safety_barrier_mm 38
```

实测 88.48% 尚未达 90% 验收线，但 Wilson CI95 上限已超过 90%。报告为 `runs/eval/hole10/20260816_model350_barrier38_strict.json` 和 `runs/eval/hole10/20260816_model350_barrier39_strict.json`。

#### 超时几何审计与 XY 接近辅助

评估仪表增加自动重置前的终止径向误差和倾角记录。重跑 38 mm 屏障复现 906/1024，证明仪表不改变轨迹。118 个超时回合的终止几何为：

```text
depth:  mean -34.32 mm, range [-49.81, -16.50] mm
radial: mean   6.32 mm, range [  1.92,  12.12] mm
tilt:   mean   1.38 deg, range [  0.85,   1.89] deg
radial pass 2/118, tilt pass 118/118, depth pass 0/118
```

这证明剩余失败不在屏障附近，而是部分随机孔位置上的 XY 接近失败。报告为 `runs/eval/hole10/20260816_model350_barrier38_timeout_audit.json`。

随后试验仅在深度 < -10 mm 且径向误差 > 2 mm 时，将策略 XY 动作与指向孔心的规则动作混合。0.25 和 0.50 混合在首批 256 episode 上都达到 100%；但 0.25 的 1024 episode 严格复核仅为 803/1024 (78.42%)、221 超时、0 过深。超时平均深度为 -61.13 mm，说明规则混合在后续随机重置上破坏了原策略的接近轨迹。该辅助保留为实验开关，但不采用。报告为 `runs/eval/hole10/20260816_model350_barrier38_xyassist25_strict.json`。

这也再次证明，本项目的 checkpoint 筛选不能依赖单批 256 episode，必须以完整 1024 episode 严格结果为准。

### 13.5 hole10 XY 外环加权课程

新增 `hole10_outer` 训练阶段。重置时 60% 保留原始 +/-10 mm 均匀分布，40% 强制至少一个轴落在 7--10 mm 外带；正式评估仍使用原始均匀 hole10。任务已接入训练和可视化映射。

从 `hole10/model_350.pt` 在 38 mm 屏障下以 `lr=1e-5`、单学习 epoch 训练 100 iteration，每 10 iteration 保存。smoke 的在线成功率为 89%--100%，正式训练初期约 92%，后期在更难分布上降至约 59%。训练目录为 `runs/ppo/hole10_outer/20260816_221400`。

256 episode 筛选中 `model_10` 至 `model_60` 均为 100%，但完整 1024 episode 严格复核为：

| Checkpoint | Success | Over-insertion | Timeout |
|---|---:|---:|---:|
| model_10.pt | 879/1024 (85.84%) | 0 | 145 |
| model_30.pt | 860/1024 (83.98%) | 0 | 164 |
| model_60.pt | 800/1024 (78.12%) | 0 | 224 |

三者均低于原模型 + 38 mm 屏障的 906/1024 (88.48%)。即使是极保守的 PPO 设置，10 iteration 已经出现均匀分布遗忘，因此该分支不采用。筛选和严格报告为 `runs/eval/hole10/20260816_outermix_screen.json` 和 `runs/eval/hole10/20260816_outermix_strict_candidates.json`。

结论：外环数据是必要的，但不能只靠改采样分布后继续 PPO。后续需要冻结部分策略网络，或对基线策略加入 KL/行为克隆保持约束，才能在补足边缘覆盖时避免中心能力遗忘。

### 13.6 延长回合时限 A/B

评估脚本新增可选 `--episode_length_s`，默认不改变环境的 8 s 协议。将 hole10 `model_350.pt + 38 mm barrier` 的时限从 8 s（约 240 控制步）增至 12 s（约 360 控制步）后：

| Time limit | Success | Over-insertion | Timeout |
|---|---:|---:|---:|
| 8 s | 906/1024 (88.48%) | 0 | 118 |
| 12 s | 904/1024 (88.28%) | 0 | 120 |

12 s 没有改善成功率。其 120 个超时回合的终止深度平均为 -80.51 mm，比 8 s 超时的 -34.32 mm 更高地离开孔面；平均径向误差 4.30 mm，深度合格 0/120。这说明失败策略不是缓慢接近后“差一点来不及”，而是在困难孔位上持续向上偏离。因此不继续扩大到 16 s，默认保留 8 s。报告为 `runs/eval/hole10/20260816_model350_barrier38_timeout12s_strict.json`。

### 13.7 接近阶段 Z 方向保护

为修正困难回合的向上偏离，新增可选接近 Z 保护：当深度 < -15 mm 时禁止向上 Z，并分别测试最小向下指令 0、0.1、0.5。三种设置的 1024 episode 结果全部与原始 38 mm 屏障逐项一致：906/1024 (88.48%)、0 过深、118 超时。

这说明失败轨迹中的策略 Z 输出本来已经强烈饱和向下；边缘工作空间中的实际向上偏离更可能来自六轴 OSC 耦合、零空间姿态或可操纵性下降，而不是动作符号错误。Z 保护保留为诊断开关，但不采用。报告为 `runs/eval/hole10/20260816_model350_barrier38_zguard15_strict.json`、`...down01_strict.json` 和 `...down05_strict.json`。

### 13.8 终止动力学审计与零空间刚度修正

在成功/超时判定前记录 Jacobian 最小奇异值、条件数、末端 Z 速度及每个关节的归一化限位余量。`model_350.pt + 38 mm barrier` 的 1024 episode 复现 906/1024，其中：

```text
                         success       timeout
Jacobian sigma_min       0.20745       0.22683
Jacobian condition       8.663         7.919
EE Z velocity (m/s)     -0.05887      -0.02317
min joint margin         0.11885       ~0
```

超时状态的 Jacobian 反而略好，因此可排除典型运动学奇异。分轴审计进一步定位到 `panda_joint7`：成功回合的平均位置为 0.9522 rad、限位余量 0.3357；超时回合的平均位置为 2.8973 rad，正好达到上限，归一化余量约为 0。这证明困难孔位的失败源是腕部第七关节在长轨迹中漂到限位，而不是回合时间不足。动力学报告为 `runs/eval/hole10/20260816_model350_barrier38_dynamics_audit.json`。

评估脚本新增可选 `--nullspace_stiffness`，仅覆盖 OSC 关节位置零空间刚度，不修改 checkpoint、策略观测或任务终止条件。同一 seed 下的 1024 episode A/B 为：

| Nullspace stiffness | Success | Over-insertion | Timeout | Wilson CI95 |
|---:|---:|---:|---:|---:|
| 20 (original) | 906/1024 (88.48%) | 0 | 118 | [86.38%, 90.29%] |
| 50 | 919/1024 (89.75%) | 0 | 105 | [87.74%, 91.46%] |
| 75 | 925/1024 (90.33%) | 0 | 99 | [88.37%, 91.99%] |

刚度 75 比原值多转化 19 个成功回合，首次在 hole10 完整严格评估中越过 90% 验收线，且保持 0 过深。当前推荐组合更新为：

```text
runs/ppo/hole10/20260816_185829/model_350.pt
--depth_safety_barrier_mm 38
--nullspace_stiffness 75
```

报告为 `runs/eval/hole10/20260816_model350_barrier38_nullspace50_strict.json` 和 `runs/eval/hole10/20260816_model350_barrier38_nullspace75_strict.json`。该结果表明，安全层负责消除过深，零空间约束则修正了边缘孔位上的关节构型漂移。

### 13.9 hole20 K=75 推进

训练入口新增 `--nullspace_stiffness`，使训练和评估可使用一致的 OSC 零空间配置。先对旧 hole20 `model_499.pt` 仅应用 K=75 + 38 mm 屏障，1024 episode 从 520/1024 (50.78%) 升至 525/1024 (51.27%)，说明控制器修正有效，但不足以解决 hole20 的策略覆盖问题。

从 hole10 `model_350.pt` 直接迁移的严格结果仅为 344/1024 (33.59%)；保守学习 5--9 轮后为 36.33%--36.43%，仍显著低于旧 hole20 策略，因此不采用该迁移分支。报告为 `runs/eval/hole20/20260816_hole10transfer_k75_smoke_strict.json`。

随后从旧 hole20 `model_499.pt` 做控制器适配微调：K=75、38 mm 屏障、`lr=3e-6`、单学习 epoch、熵系数 0，训练 50 iteration 并每 5 轮保存。512 episode 筛选中 `model_15.pt` 最佳，其 1024 episode 严格复核为：

```text
531/1024 = 51.86%
timeout = 493
over-insertion = 0
Wilson CI95 = [48.79%, 54.90%]
```

该结果比原 hole20 + 38 mm 屏障多 11 个成功回合，比仅将旧模型切换到 K=75 多 6 个，是当前 hole20 最佳可复现组合：

```text
runs/ppo/hole20/20260816_225900/model_15.pt
--depth_safety_barrier_mm 38
--nullspace_stiffness 75
```

严格报告为 `runs/eval/hole20/20260816_k75_adapt_model15_strict.json`。后续 checkpoint 没有继续改善，表明当前主限制已不是过深或控制器姿态漂移，而是 hole10 到 hole20 的 XY 工作空间跨度。

### 13.10 新增 hole15 中间课程

新增 `OscPose6DHoleRandom15mmEnvCfg` 及 Gym 任务 `Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom15mm-v0`，孔位 XY 均匀随机范围为 +/-15 mm，并接入训练、评估和可视化阶段映射。训练终止惩罚与 hole20 一致为 -25，运行时继续由 38 mm 预测屏障保证深度安全。

从 hole10 `model_350.pt` 以 K=75、38 mm 屏障、`lr=3e-6`、单学习 epoch、熵系数 0 进行 10 iteration smoke。训练目录为 `runs/ppo/hole15/20260816_231032`。三个节点的 1024 episode 严格评估为：

| Checkpoint | Success | Timeout | Over-insertion |
|---|---:|---:|---:|
| model_0.pt | 578/1024 (56.45%) | 446 | 0 |
| model_5.pt | 588/1024 (57.42%) | 436 | 0 |
| model_9.pt | 589/1024 (57.52%) | 435 | 0 |

hole15 迁移起点的 56.45% 显著高于 hole10 直接迁移 hole20 的 33.59%，证明 +/-15 mm 是有效的中间课程。微调 9 轮后提升 1.07 个百分点，且保持 0 过深；当前候选为 `runs/ppo/hole15/20260816_231032/model_9.pt`。报告为 `runs/eval/hole15/20260816_hole15_k75_smoke_strict.json`。

### 13.11 hole15 混合保持课程与 hole20 迁移复核

新增 `hole15_mix`：训练时 50% 从 +/-10 mm、50% 从 +/-15 mm 采样，标准 hole15 评估仍使用完整 +/-15 mm 分布。重置事件新增 `inner_probability` 和 `inner_abs` 参数，任务已接入训练与可视化映射。

从 hole15 `model_9.pt` 以 K=75、38 mm 屏障、`lr=1e-6`、单 epoch、熵系数 0 进行 10 轮 smoke。在标准 hole15 严格评估中：

| Checkpoint | Success | Over-insertion |
|---|---:|---:|
| model_0.pt | 592/1024 (57.81%) | 0 |
| model_5.pt | **593/1024 (57.91%)** | 0 |
| model_9.pt | 581/1024 (56.74%) | 0 |

`model_5.pt` 相比旧 hole15 `model_9.pt` 多 4 个成功回合，但零样本迁移到 hole20 仅为 371/1024 (36.23%)。随后以该模型作为起点进行 20 轮 hole20 低学习率适配，512 回合筛选最佳 `model_15.pt` 仅 193/512 (37.70%)，远低于现有 hole20 最佳 51.86%，因此整条分支不采用。报告为 `runs/eval/hole15/20260817_mix_standard_strict.json` 、`runs/eval/hole20/20260817_hole15mix_model5_zero_shot_strict.json` 和 `runs/eval/hole20/20260817_from15mix_screen512.json`。

结论：混合课程能轻微改善 hole15 保持性，但不足以解决 hole20 的外环定位问题。当前 hole20 最佳仍为 `runs/ppo/hole20/20260816_225900/model_15.pt`，配置为 38 mm 屏障 + K=75。

### 13.12 重建真实孔位 XY 课程

核查报告与配置后确认：旧 `offset5` 是 peg mount 安装偏移 +/-5 mm，不是孔位 XY +/-5 mm；baseline 的初始 XY 基本居中，且 hole10/15/20 均继承同一个机器人初始关节姿态。因此新增独立的 `hole_xy5`：保持 30 mm 物理孔和 +/-5 mm peg mount 偏移，另将孔中心 XY 设为 +/-5 mm。

从已通过 offset5 验收的 `model_499.pt` 进行 K=75、38 mm 屏障、`lr=3e-6`、单 epoch、熵系数 0 训练 10 轮。`hole_xy5` 严格 1024 episode 结果：

| Checkpoint | Success | Timeout | Over-insertion |
|---|---:|---:|---:|
| model_0.pt | **1001/1024 (97.75%)** | 23 | 0 |
| model_5.pt | 996/1024 (97.27%) | 28 | 0 |
| model_9.pt | 994/1024 (97.07%) | 30 | 0 |

`model_0.pt` 即保留了最佳性能，说明从安装偏移课程迁移到真实孔位 +/-5 mm 在几何与动力学上是可行的。任务配置为 `OscPose6DHoleRandom5mmEnvCfg`，已接入训练、评估和可视化。报告为 `runs/eval/hole_xy5/20260817_smoke_strict.json`。

随后从 `hole_xy5/model_0.pt` 迁移到真实 hole10（孔位 +/-10 mm），以同样参数训练 10 轮 smoke，再训练 50 轮密集保存。第一次严格评估起点为 758/1024 (74.02%)；50 轮筛选中 `model_40.pt` 为最佳，512 episode 为 83.01%，但 1024 episode 严格复核为：

```text
788/1024 = 76.95%
timeout = 236
over-insertion = 0
Wilson CI95 = [74.28%, 79.43%]
```

这一结果低于旧 hole10 最佳 88.48%，但相对新路径起点提升 2.93 个百分点，且全部失败都是超时而非过深，证明策略正在学习远距离 XY 接近，尚未达到可以向 hole20 传递的稳定水平。严格报告为 `runs/eval/hole10/20260817_xy5_refine_model40_strict.json`。

### 13.13 +/-10 mm 几何可达性网格审计

为区分“物理不可达”与“PPO 轨迹学不会”，扩展 `scripts/oracle_batch_test.py` 支持确定性 XY 网格，并增加 `mdp.events.apply_hole_offsets` 用于不经过重置随机化的物理孔位移动。

在 Pose6D hole10 任务中用 +/-10 mm、步长 5 mm 的 5x5 网格，共 125 个环境（每个网格点重复 5 次）运行高精度 120 Hz Oracle：

```text
grid points       25
environments      125
success           125/125 = 100.00%
timeout           0
over-insertion    0
initial radial    mean 9.37 mm, max 14.14 mm
```

该测试使用现有中心初始姿态，并保留了训练任务中的 +/-5 mm peg mount 随机化，因而最大初始径向误差可达 14.14 mm。Oracle 仍在全部网格点成功，证明 hole10 区域不是机械自身的硬不可达区域。之前 PPO 的 timeout 主要是长程 XY 接近、关节冗余漂移和策略覆盖不足。

后续对每个课程阶段将分两步：先在网格上搜索一个最大化最小关节限位余量和 Jacobian 可操纵性的鲁棒初始姿态，再以该姿态进行 PPO 迁移。对 +/-50 mm，如果单一鲁棒姿态无法覆盖全部网格，再考虑按象限分区的姿态库或基于孔位的 IK 条件初始化。报告使用命令为：

```text
scripts/oracle_batch_test.py \
  --task Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom10mm-v0 \
  --num_envs 125 --episode_length_s 12 \
  --grid_mm 10 --grid_step_mm 5 --grid_mount_mm 0
```

### 13.14 +/-20 mm 初始姿态搜索

将 `scripts/search_manipulable_vertical_pose.py` 改为使用与课程训练相同的 `Pose6D-Baseline` 任务，并加入目标条件 XY 网格评分。搜索会同时记录初始径向误差、全位姿 Jacobian 最小奇异值、关节限位余量和姿态倾角；另外补上 `env.reset()`，保证孔位随机化与训练/评估语义一致，并对多环境复制原点做局部坐标转换。

用 512 个环境、20 个批次、关节扰动尺度 `local_scale=0.35`，在 +/-20 mm、步长 5 mm 的 9x9 网格上进行候选搜索。全局最佳可操纵性候选为：

```text
sigma_min = 0.227922
tilt      = 1.217 deg
tip_local = (0.2283, -0.0445, 0.4149) m
q         = [-0.132547, -1.034358, -0.041407, -2.470325,
             -0.020829, 1.451517, 1.003350]
```

但该候选不是所有目标的对准姿态：81 个网格目标的最佳初始径向误差范围为 0.30--23.09 mm，均值 9.03 mm。也就是说，单纯在中心姿态附近随机搜索只能找到“条件良好”的冗余姿态，不能同时把整个 +/-20 mm 区域都放到孔上方。下一步不直接迁移该候选，而是用目标条件 IK/分区姿态库生成每个象限的可验证初始姿态，再用 Oracle 逐点验收。

作为坐标与物理控制复核，Pose6D hole10 使用 2 个环境、9 个网格点（步长 10 mm）的 Oracle 结果为 2/2 成功、超时 0、过深 0，初始径向误差 12.07--14.14 mm，说明批量复制坐标修正没有推翻此前 +/-10 mm 可达性结论。诊断日志为 `/tmp/search_pose6d_grid20.log`。

### 13.15 目标条件 6D Jacobian 姿态库

在随机搜索之外，为每个目标偏移加入加权 6D 阻尼 Jacobian 迭代：XY 平移作为主任务，peg 的竖直姿态作为角度约束，z 高度保持当前接近高度。这样避免了“XY 对准了但 peg 倾斜超过 2°”的问题。

结果如下：

| 网格 | 重启数 | 最终有效姿态 | 每目标径向误差 |
|---|---:|---:|---:|
| +/-10 mm，步长 5 mm | 81 | 70 | 0.0003--0.0370 mm，均值 0.0069 mm |
| +/-20 mm，步长 5 mm | 324 | 281 | 0.0004--0.0457 mm，均值 0.0063 mm |

±20 mm 的 81 个目标点全部有候选姿态，说明该区域在几何运动学上可以用目标条件初始化覆盖。姿态库临时保存为 `/tmp/pose_bank20_vertical.json`；搜索脚本支持 `--output` 生成持久化 JSON。

尝试将姿态库直接注入现有批量 Oracle 时，Isaac PhysX 在“写入每个环境不同关节状态后读取 body state”阶段出现阻塞；默认 Oracle 路径未修改、原有 2 环境复核仍通过。因此当前姿态库结论属于运动学验证，下一步需要用单环境逐点注入或在 reset 事件阶段写入姿态，再进行动力学验收，不能直接宣称已通过物理 Oracle。

### 13.16 deterministic grid 修正与单点动力学复核

进一步核查发现，旧的 `apply_hole_offsets` 使用 `default_root_state` 直接叠加网格偏移，reset 后可能保留随机孔位；已改为先扣除当前 reset 偏移，再写入目标偏移。修正后，Pose6D hole10 的 2 环境 deterministic grid（步长 10 mm）结果为：

```text
hole X range = -10.00..-10.00 mm
hole Y range = -10.00..0.00 mm
initial radial = 12.07..14.14 mm
success = 2/2
timeout = 0
```

这确认网格审计现在是真正的确定性孔位，而不是“随机 reset 偏移 + 网格偏移”。

随后在禁用 mount 随机化、确定性 `(-20,-20) mm` 角点上注入姿态库候选：初始径向误差为 **0.01 mm**、倾角 0.03°，说明目标条件 IK 姿态确实把 peg 放到了目标孔上方。但插入过程中 `peg_mount_joint` 从 0 漂移到约 5 mm，最终径向误差约 4.3 mm 并超时；因此当前角点的动力学失败来自 mount 关节稳定/保持，而不是初始姿态或孔位可达性。该结果提示下一步应先修复 mount 的动态保持，再进行 81 点姿态库的全量 Oracle 验收。

### 13.17 mount 漂移隔离实验

为确认漂移是否是失败的主因，在同一个 `(-20,-20) mm` 角点、同一个目标条件 IK 初始姿态上加入仅用于诊断的 `--lock_mount`：每个仿真步都将 `peg_mount_joint` 的位置和速度重写为零，其他控制与孔位保持不变。

结果为：

```text
initial radial = 0.02 mm
mount X/Y range = 0.00..0.00 mm
tilt at step 100 = 0.31 deg
success = 1/1
timeout = 0
C0 ACCEPTANCE = PASS
```

这证明目标条件初始姿态和 XY 接近轨迹本身可以完成 ±20 mm 角点插入；此前失败的必要条件是 mount 动力学漂移，而非几何不可达。`--lock_mount` 只作为定位问题的诊断开关，尚未用于训练，也没有据此盲目修改质量、刚度或阻尼参数。下一步应实现可物理解释的 mount 保持约束（或在课程任务中固定该自由度），再逐点验收 81 个 ±20 mm 目标，最后才迁移 PPO。

### 13.18 mount 驱动力上限诊断接口

为区分“刚度不足”和“驱动力饱和”，`scripts/oracle_batch_test.py` 新增了 `--mount_effort_limit`，可与已有的 `--mount_stiffness`、`--mount_damping` 组合使用。该参数只覆盖当前 Oracle 进程中的 actuator 配置，不改变任务文件和训练默认值。

本轮尝试重新启动高驱动力诊断时，主机 NVIDIA 驱动暂时不可用（`nvidia-smi` 无法连接驱动），因此没有把未完成的结果写成物理结论。恢复 GPU 后，优先测试一组同时提高刚度、阻尼和 effort 上限的单角点实验；只有在单点不漂移且 C0 通过后，才建立 81 点验收矩阵。

### 13.19 hole20 续训结果

使用 `runs/ppo/hole20/20260817_103321/model_19.pt` 续训 300 轮，生成目录为 `runs/ppo/hole20/20260817_143637/`，最终 checkpoint 为 `model_318.pt`。训练过程没有改善 hole20：

```text
success termination: 约 6.54%（第 19 轮） -> 约 0%（第 294--318 轮）
timeout termination: 约 4.54% -> 约 4.25%
mean reward: 9.25（第 19 轮） -> -8.65（第 318 轮）
```

因此 `model_318.pt` 不能作为下一阶段 hole30/50 的迁移起点。当前最有价值的对照仍是续训前的 `model_19.pt`；下一轮应先恢复 mount 稳定性或缩小课程变化，再重新训练，而不是继续延长这次训练。

### 13.20 hole20 回滚判断

对比续训前后的 checkpoint 后，旧的 `runs/ppo/hole20/20260817_103321/model_19.pt` 明显优于续训生成的 `model_318.pt`，因此将 `model_19.pt` 保留为当前 hole20 回滚基线，不采用 `model_318.pt` 向 hole30/50 迁移。

需要注意，旧模型本身也不是最终达标模型：已保存的 512 回合严格评估中成功率为 **37.70%**，失败主要包含过深插入和超时。这说明本次训练退化并不意味着 hole20 已解决，而是证明在 mount 漂移未修复前继续 PPO 会破坏已有策略。后续顺序固定为：修复/验证 mount 保持 → 用 `model_19.pt` 重新训练 → 严格评估通过后再扩展课程。

### 13.21 opt-in mount-stable 课程分支

针对漂移问题新增 `hold_fixed_peg_mount` 事件：每个控制步结束后，将两个 prismatic mount joint 重写回本 episode reset 时采样的 XY 偏移，并清零其速度，同时重新设置 position target。这样保留了每回合 +/-5 mm 的安装误差随机化，但不再允许接触冲击把安装座推到关节限位。

为避免改变历史 checkpoint 的动力学语义，该保持机制只在新任务
`Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStable-v0` 中启用，旧的 hole20 任务保持不变。`train_phase1.py` 和 `eval_phase1.py` 可分别使用阶段名 `hole20_mount_stable`。目前仅完成代码和配置检查，尚未在本机 GPU 不可用时宣称物理验收通过；恢复驱动后应先用 `model_19.pt` 做单点/小规模 smoke，再进行完整 PPO。

### 13.22 mount-stable 单角点物理验收

在新任务 `Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStable-v0` 上，不使用 Oracle 的 `--lock_mount`，仅启用课程事件本身，复核确定性 `(-20,-20) mm` 角点：

```text
initial radial mean/max = 0.02/0.02 mm
mount X range = 0.00..0.00 mm
mount Y range = 0.00..0.00 mm
success = 1/1
timeout = 0
C0 ACCEPTANCE = PASS
```

这证明新增事件本身已经阻止 mount 漂移，且没有依赖 Oracle 专用的外部锁定开关。下一步可以用旧 hole20 `model_19.pt` 在 mount-stable 分支做 smoke 和 128 环境严格评估，再决定是否进行 300 轮正式迁移训练。

### 13.23 mount-stable smoke 结果

完成了 `hole20_mount_stable` 的 64 环境、10 轮 smoke，运行目录为 `runs/ppo/hole20_mount_stable/20260817_151431/`。mount-stable 任务正常启动并产生 checkpoint，但 smoke 使用的是脚本默认 PPO 参数：学习率 `3e-4`、8 个 learning epoch、无深度安全屏障；因此策略从第 0 轮成功终止约 0.375 快速退化为 0，且过深终止上升。

该结果不能归因于 mount 漂移，也不能作为正式训练失败结论；`model_9.pt` 不作为后续起点。正式训练必须从旧 `hole20/model_19.pt` transfer，并显式使用低学习率 `1e-6`、1 个 epoch、`depth_safety_barrier_mm=38`。

### 13.24 mount-stable 旧模型评估配置复核

第一次在 mount-stable 分支评估 `model_19.pt` 时使用了评估脚本默认值：`nullspace_stiffness=20`，且未启用 `depth_safety_barrier_mm=38`。但该 checkpoint 的训练记录明确使用了 `nullspace_stiffness=75` 和 38 mm 深度屏障，因此得到的 39.26% 不能与训练策略做严格公平比较。

下一次评估必须显式补上：

```text
--nullspace_stiffness 75 --depth_safety_barrier_mm 38
```

在统一配置前，不对 `model_19.pt` 做最终淘汰，也不开始新的 PPO 训练。

### 13.25 barrier 复评后的失败归因

按训练一致的 `nullspace_stiffness=75` 和 `depth_safety_barrier_mm=38` 复评后，`model_19.pt` 结果为 **181/512 = 35.35%**。失败结构发生了清晰变化：

```text
success       = 181
timeout       = 331
over-insertion = 0
timeout depth mean = -45.06 mm
timeout radial mean = 8.43 mm
timeout tilt mean = 0.75 deg
timeout joint-7 margin mean ≈ 0
```

因此深度屏障已经消除了过深插入，但旧策略在 hole20 的远端 XY 接近阶段无法对准，最终还会把第 7 关节推到限位；当前不是 mount 漂移问题，也不是倾角问题。下一步先用评估期 XY assist 做因果验证：若成功率明显恢复，再把 assist 逻辑移入 mount-stable 训练环境，而不是盲目继续调 PPO 学习率。

### 13.26 XY assist 反事实验证

使用 `approach_xy_assist_blend=1.0`、`until_depth=-10 mm`、`radial=2 mm` 进行评估后，成功率进一步降至 **163/512 = 31.84%**。timeout 增至 349，timeout 平均倾角升至 6.43°，第 7 关节平均限位余量仅 0.006。说明直接替换 XY action 会破坏原策略的姿态/深度协调，不能把该 assist 逻辑直接移入训练。

当前结论收敛为：保留 mount-stable 和 38 mm 深度屏障；下一步改为 reset 阶段的目标条件初始姿态/IK，使 peg 在 hole20 远端目标附近以更好的冗余姿态开始接近，而不是在运行中强行覆盖策略动作。

### 13.27 target-conditioned reset 分支

新增任务 `Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableIK-v0`。它继承 mount-stable 的静态安装误差保持，并在每次 reset 后对实际孔位运行 5 次加权 6-D 阻尼 Jacobian 更新：XY 误差为主任务，姿态误差低权重约束 peg 保持竖直，关节更新限制在 soft joint limits 内。

该求解只改变 episode 初始 arm 状态，不覆盖运行时 PPO action；因此与失败的评估期 XY assist 不同。新阶段名为 `hole20_mount_stable_ik`，当前已完成语法检查，尚未在 GPU 恢复后进行物理 smoke。

### 13.28 IK reset 启动异常修正

首次运行 `hole20_mount_stable_ik` 时只生成了 `run_config.json`，没有生成 checkpoint，随后评估命令直接退出，说明异常发生在环境 reset/初始化阶段。已修正 reset IK 事件：

1. Jacobian 改为先按环境切片，再按 arm joint 切片，避免 PhysX tensor view 的混合高级索引广播错误；
2. 移除 reset 事件内部的 `scene.write_data_to_sim()`，仅保留机器人状态写入、`sim.forward()` 和 scene 状态更新，避免与 ManagerBasedEnv 的 reset 写回流程冲突。

修正后需要重新执行单环境/小规模 smoke，确认能生成 checkpoint 后再做 512 回合评估。

随后确认 IK smoke 在 64 环境下可以生成 checkpoint，但 128 环境评估在加载 `model_19.pt` 后直接退出。进一步判断为 partial reset 阶段重复调用 `sim.forward()` 的风险，而非 checkpoint 不兼容；已将 IK 事件改为首次全量 reset 做完整求解，后续 partial reset 使用缓存 Jacobian 的一次线性 XY 更新，不再调用 `sim.forward()`。该修正需要重新进行 64 环境 smoke 和 128 环境评估。

### 13.29 离线姿态库分支复评

为规避 reset 阶段的 PhysX `sim.forward()`，将 target-conditioned 初始化改为离线 25 点姿态库（孔位偏移 `-20..20 mm`、步长 `10 mm`），reset 时按最近偏移查表写入 7 个 Panda arm 关节。姿态库分支在恢复 GPU 和缺失的 editable `rsl_rl` 依赖后完成了完整 512 回合评估：

```text
task   = Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableIK-v0
config = nullspace_stiffness=75, depth_safety_barrier_mm=38, seed=42
result = 114/512 = 22.27%
timeout = 398, over-insertion = 0
timeout depth mean = -45.09 mm
timeout radial mean = 8.52 mm
timeout tilt mean = 0.73 deg
timeout joint-7 margin mean ≈ 0
report  = /tmp/hole20_ik_full_20260817b.json
```

评估过程已能稳定输出 JSON，故此前的“加载 checkpoint 后直接退出”主要是依赖路径 `/tmp/rsl_rl` 缺失造成的运行环境问题，而不是 GPU 或 checkpoint 读取失败。22.27% 低于 mount-stable 原始分支的 35.35%，因此不进行姿态库分支训练；`model_19.pt` 和 `hole20_mount_stable` 仍是当前对照基线。下一步应先检查姿态库 reset 后的实际初始位姿/观测与旧模型训练分布是否一致，再决定保留或撤销该初始化策略。

### 13.30 reset 分布诊断：姿态库改变了高度和策略输入

分别在 `hole20_mount_stable` 与 `hole20_mount_stable_ik` 上用 128 个环境、seed=42 采集 reset 后状态，结果如下：

```text
                             MountStable       OfflineBank
初始径向误差均值/最大值          15.85/29.69 mm      5.18/11.12 mm
初始插入深度均值/标准差          -50.00/0.08 mm      -80.00/20.90 mm
q7 均值/标准差                  0.8306/0           0.8159/0.0475 rad
policy obs L2 均值              3.3234             3.2453
```

姿态库确实改善了 XY 初始误差，但其搜索目标只约束“竖直、XY 对准和可达性”，没有锁定 peg tip 的 Z 高度。最终 25 个姿态的 tip 高度覆盖约 0.381--0.464 m，导致 reset 时的高度和插入深度大幅偏离旧模型训练分布；关节 1--6 也从固定初始姿态变成显著分散的分布。该分布偏移可以解释姿态库评估从 35.35% 降至 22.27%，当前不应直接基于该姿态库训练。

诊断脚本为 `scripts/diagnose_reset_distribution.py`，报告分别保存于 `/tmp/hole20_reset_distribution_base_20260817.json` 和 `/tmp/hole20_reset_distribution_ik_20260817.json`。若继续使用目标初始化，必须重新生成“固定 tip 高度（约 -50 mm 深度）+ XY 对准”的姿态库，并先做小规模复评；否则撤销姿态库，回到 `MountStable` 原始初始化分布。

### 13.31 高度锁定姿态库复评

重新生成了以旧 Pose6D 初始姿态为中心、tip 高度锁定在 `z≈0.393592 m`（约 -50 mm 深度）的 25 点姿态库。每个网格目标经过 15 次阻尼 6-D IK 精修，离线结果为：径向误差最大约 `0.0012 mm`，tip 高度约 `0.3931..0.3936 m`。新姿态库已写入 `mdp/target_pose_bank.py`。

reset 分布恢复正常后，用旧 `model_19.pt` 做同配置 128 回合对照：

```text
MountStable 原始分支       47/128 = 36.72%
HeightLocked 姿态库分支    33/128 = 25.78%
```

高度锁定修复了之前的 Z 分布偏移，但仍未解决策略不匹配：姿态库分支 timeout 的径向误差均值为 `9.20 mm`，第 7 关节仍到达限位；原始分支为 `model_19.pt` 训练时的初始关节分布，表现明显更好。因此当前结论是：姿态库需要专门训练才能发挥作用，不能直接套用旧 checkpoint。暂不进行该分支 PPO 训练，继续保留 `MountStable` 原始分支作为 hole20 评估基线。

### 13.32 高度锁定姿态库从头训练：无门控基线

考虑到旧 `model_19.pt` 与高度锁定姿态库的 reset 分布不匹配，先在
`hole20_mount_stable_ik` 上从随机策略重新训练 300 轮：1024 环境、学习率
`1e-4`、PPO 每轮 2 个 epoch、`nullspace_stiffness=75`，并保留 30 mm 预测深度屏障。
训练正常完成，输出目录为：
`runs/ppo/hole20_mount_stable_ik/20260817_164853/`。

训练奖励约由 `-23` 改善至 `-4.55`，但成功率始终为 0。独立 16 回合评估中
`model_299.pt` 为 `0/16`，失败终止深度约 `42..67 mm`，说明仅增加深度屏障仍不能让
从头策略学会 XY 对准和安全插入；该模型不接受为候选 checkpoint。

### 13.33 插入门控版从头训练

在安全屏障基础上加入对准门控：深度达到 `-10 mm` 后，只有径向误差≤2 mm 且倾角≤2°
才允许继续下降；滞回系数为 `4/3`，屏障为 30 mm。训练命令对应 1024 环境、300 轮，
输出目录为：
`runs/ppo/hole20_mount_stable_ik/20260817_170536/`。

训练完整结束，奖励稳定到约 `-4.8..-5.0`，但训练日志成功率仍为 0。独立直接循环评估结果：

```text
model_25   0/16，timeout=16，depth=-96.8 mm，radial=136.6 mm
model_100  0/16，over=16，depth=47.3 mm，radial=49.1 mm
model_200  0/16，over=16，depth=49.9 mm，radial=35.9 mm
model_275  0/16，over=16，depth=45.9 mm，radial=31.3 mm
model_299  0/32，over=32，depth=56.4 mm，radial=36.6 mm，tilt=45.0 deg
```

门控降低了早期奖励发散，但没有学出可用的对准—插入闭环；当前两次从头训练均不应
用于 hole20 成功声明。下一步应改为分阶段课程（中心/±5 mm 对准成功后再扩大到 ±20 mm），
并在训练中显式奖励“门控开启后保持对准并推进”，而不是继续延长同一配置的训练。

### 13.34 多尺度奖励塑形实验

为验证 hole20 的失败是否来自奖励稀疏，新增独立任务
`HoleRandom20mm-MountStableReward-v0`，加入 15/7/2.5 mm 三尺度径向对准进度、
30--42 mm 连续制动惩罚，并保留 30 mm 深度屏障。训练目录为：
`runs/ppo/hole20_reward/20260817_181718/`。

训练奖励由早期约 `-25` 改善到 `-2.27`，但独立评估为 `0/32`，全部 timeout；
终止深度均值约 `-397.4 mm`。策略学会了完全上撤以规避插入风险，说明单纯增加制动
惩罚会产生新的撤离局部最优。

### 13.35 加入深度接近进度后的复训

新增从初始 `-50 mm` 向 `-10 mm` 门控深度的有界接近进度，并降低制动项权重，训练目录为：
`runs/ppo/hole20_reward/20260817_182656/`。

训练最终奖励约 `-5.39`，撤离局部最优消失，但独立评估仍为 `0/32`，全部过深：
终止深度均值 `59.3 mm`，径向误差均值 `42.3 mm`，倾角均值 `24.2°`。

### 13.36 内圈混合课程实验

在上述奖励基础上新增 `HoleRandom20mm-MountStableRewardMix-v0`，其中 60% reset
采样 ±5 mm 内圈、40% 采样完整 ±20 mm，训练目录为：
`runs/ppo/hole20_reward_mix/20260817_183636/`。

训练最终奖励约 `-5.54`，但独立评估仍为 `0/32`，全部过深；终止深度均值 `59.0 mm`，
径向误差均值 `60.1 mm`。因此当前证据表明，奖励稀疏确实存在，但仅修改奖励和 reset
混合仍不足以解决 hole20；下一步应回到动作/初始姿态可达性与 XY--姿态协调的控制诊断，
不再继续盲目增加惩罚权重。

### 13.37 简化评估器口径修正与奖励混合课程复核

复查发现，`scripts/eval_checkpoint_simple.py` 原先默认启用了径向 2 mm、倾角 2° 的插入
门控，而历史正式评估并未启用该门控。因此此前 13.34--13.36 中基于该脚本的直接循环
分数不能与正式评估比较，相关低分结论作废。已将两个门控参数默认值改为 `None`，仅在
命令行显式传入时启用；深度屏障仍保留为 38 mm，以匹配历史 hole10 对照。

修正后先复核已知基线：

```text
hole10 model_350.pt，64 回合，无门控，barrier=38 mm：64/64 = 100%
```

随后按同一口径复核 hole20：

```text
offset5 model_499.pt → RewardMix：25/32 = 78.13%
offset5 model_499.pt → uniform MountStable：10/32 = 31.25%
RewardMix transfer model_25：24/32 = 75.00%
RewardMix transfer model_100：27/32 = 84.38%
RewardMix transfer model_125：28/32 = 87.50%
RewardMix transfer model_125，扩展至 64 回合：47/64 = 73.44%
RewardMix transfer model_100，扩展至 64 回合：44/64 = 68.75%
```

这说明 RewardMix 转移训练已经能把部分 hole20 表现提升到约 70% 量级，但短样本的
87.5% 存在明显方差，且继续训练会漂移（model_150 为 17/32）。从 model_125 以
`lr=1e-5`、单 epoch 再微调 100 轮，输出目录为
`runs/ppo/hole20_reward_mix/20260817_193408/`；model_25/50/75/99 的 32 回合结果分别
为 23/32、17/32、17/32、21/32，未观察到稳定增益。因此当前暂定候选为原始
`20260817_185132/model_100.pt` 或 `model_125.pt`，后续必须用 512 回合正式评估确认，
不能依据单次 32 回合结果宣称 hole20 已解决。

### 13.38 full-±20 mm 低学习率转移复训

为排除 RewardMix 内圈偏置，直接在完整均匀 ±20 mm 的 `MountStableReward-v0` 上从
offset5 `model_499.pt` 转移，使用 `lr=1e-5`、单 epoch、50 轮，输出目录为
`runs/ppo/hole20_reward/20260817_195053/`。`model_49.pt` 在 128 回合中为 38/128
（29.69%），但扩大到 512 回合后为 131/512（25.59%）；原始 offset5 在同一口径为
126/512（24.61%）。因此该短程复训只有约 1 个百分点的稳定提升，尚不足以证明奖励
塑形已经解决宽工作空间问题。

当前最可靠结论是：奖励项可以改善局部训练信号，但 PPO 转移会损伤原有策略，且
RewardMix 的内圈成功率不能代表均匀 ±20 mm 泛化。下一步应优先做“按 XY 位置分桶的
成功率/终止深度诊断”，确认失败是否集中在工作空间边缘，再针对边缘样本设计课程或
初始化，而不是继续盲目延长全空间训练。

### 13.39 XY 工作空间分桶定位

在评估器中加入目标 XY 偏移分桶，记录每个 episode reset 时的目标径向偏移，避免使用
终止后已被 reset 的新目标。均匀 ±20 mm 的 offset5 原模型（512 回合）结果为：

```text
目标径向 0--5 mm：   19/20 = 95.00%
目标径向 5--10 mm：  60/81 = 74.07%
目标径向 10--15 mm： 37/121 = 30.58%
目标径向 15--20 mm： 10/185 =  5.41%
目标径向 >=20 mm：    0/105 =  0.00%
```

所有主要失败均为 timeout，而不是 over-insertion，故 hole20 的主瓶颈已经定位为边缘
XY 对准/可达性泛化。full-±20 mm 低学习率奖励复训 `model_49.pt` 的分桶结果为
`0--5: 92.86%`、`5--10: 68.12%`、`10--15: 32.17%`、`15--20: 10.87%`、
`>=20: 0.86%`，总体仅由 24.61% 提升至 25.59%，不具备稳定统计增益。

RewardMix 原模型的 512 回合分桶为 `0--5: 97.74%`、`5--10: 79.12%`、
`10--15: 30.77%`、`15--20: 9.38%`、`>=20: 0%`；由于其 60% 样本本来就在 ±5 mm
内圈，整体 69.14% 主要反映内圈成功，不能证明奖励函数解决了边缘泛化。因此当前结论
是：奖励修改对局部进度有弱作用，但尚未有效解决真正的 ±20 mm 工作空间问题。

补充控制变量复核：此前部分新训练使用 `nullspace_stiffness=20`，而简化评估器默认是
75。将评估刚度统一为训练值 20 后，均匀 ±20 mm、512 回合结果为：

```text
offset5 model_499：       135/512 = 26.37%
full reward model_49：    143/512 = 27.93%

offset5 分桶：   0--5 90.32%，5--10 59.74%，10--15 23.13%，15--20 14.44%，>=20 4.44%
reward 分桶：    0--5 75.00%，5--10 50.00%，10--15 32.28%，15--20 20.23%，>=20 5.77%
```

统一刚度后，奖励复训在 10 mm 以上区域有一致但幅度有限的改善（总体约 +1.6 个百分点），
说明奖励塑形并非无效；但主要失败仍是 timeout，且边缘成功率仍低于 21%。后续所有训练
和评估应固定 `nullspace_stiffness=20`，避免把控制器刚度差异混入奖励结论。

### 13.40 边缘环带课程与内圈锚点实验

新增 `HoleRandom20mm-MountStableRewardEdge-v0`，在原有 ±20 mm 方形内拒绝采样，集中
训练径向 10--28 mm 的边缘目标；固定训练/评估 `nullspace_stiffness=20`。从 offset5
转移训练 100 轮，输出目录为 `runs/ppo/hole20_reward_edge/20260817_201610/`。

边缘专训的中期 `model_50` 在边缘任务 128 回合为 `25/128=19.53%`，原 offset5 为
`13/128=10.16%`；但在完整均匀 ±20 mm 的 512 回合中为 `117/512=22.85%`，低于
offset5 的 26.37%，说明纯边缘训练造成内圈能力遗忘。其边缘分桶虽然改善：
`15--20 mm=13.66%`、`>=20 mm=10.64%`，但不足以弥补内圈下降。

随后加入 20% 的 ±5 mm 内圈锚点，形成 `RewardEdgeAnchor-v0`，从 edge `model_50`
继续训练 50 轮，输出目录为 `runs/ppo/hole20_reward_edge_anchor/20260817_202459/`。
中期 `model_20` 的完整均匀评估为 `137/512=26.76%`，与 offset5 的 26.37% 基本相当，
但边缘分桶有局部改善：

```text
10--15 mm：29.91%
15--20 mm：23.40%
>=20 mm：  7.14%
```

相较 offset5 的 `23.13%/14.44%/4.44%`，边缘确实提升，但内圈下降到约 50%，总体尚
未形成稳定增益。当前结论是：边缘重采样方向正确，但需要加入旧策略 replay/蒸馏或更高
的内圈保持比例，防止课程训练遗忘原有中心能力。

### 13.41 目标条件 IK 的动力学可行性验证

为验证“根据随机孔位补偿机械臂初始姿态”是否在实际仿真动力学中可行，新增
`scripts/validate_target_ik_dynamics.py`。在 MountStableReward 环境中使用 128 个随机
±20 mm 孔位和 ±5 mm 固定 mount 偏移，按当前 tip 高度和姿态做 20 轮阻尼 6-D IK，只补偿
XY，然后写入关节状态并运行 60 个零动作物理步。

结果：

```text
IK 后初始径向误差 mean/max：0.00048 / 0.00191 mm
初始倾角 mean/max：         1.7061 / 1.7064 deg
60 步径向最大漂移 mean/max： 0.3488 / 0.9371 mm
60 步 Z 最大漂移 mean/max：   0.0223 / 0.0740 mm
60 步倾角最大漂移 mean/max：  0.0278 / 0.1098 deg
Jacobian sigma_min mean/min： 0.227905 / 0.227465
关节余量 mean/min：           0.1952 / 0.1843
```

因此目标条件 IK 在当前动力学、随机 mount 和随机 hole 下是可行的：初始对准误差远小于
1 mm，静置 60 步后仍小于 1 mm 漂移，且没有接近关节限位或 Jacobian 奇异。之前的姿态库
低分是 checkpoint/reset 分布不匹配问题，不是目标条件 IK 本身不可达。下一步可以基于
该 reset 逻辑训练，但必须保持固定 tip 高度/姿态，并逐步混入固定中心姿态，避免策略只
会依赖 IK 初始化。

### 13.42 在线目标 IK reset 训练结果

新增 `HoleRandom20mm-MountStableOnlineIK-v0`，每次 reset 按实际随机 hole 和 mount
偏移在线求解高度锁定的目标条件 IK，事件顺序为 hole 随机化、mount 随机化、在线 IK。
smoke test 已通过，正式训练配置为 K=20、1024 环境、`lr=1e-4`、PPO 两个 epoch、300
轮，输出目录：`runs/ppo/hole20_online_ik/20260817_210752/`。

从 offset5 转移的短程训练目录为 `runs/ppo/hole20_online_ik/20260817_210256/`：
`model_50` 在 128 回合为 `37/128=28.91%`，`model_99` 为 `20/128=15.63%`，出现后期
漂移。从头训练的 `model_100`、`model_200`、`model_299` 均为 `0/128`，全部
over-insertion，说明策略虽然获得了可达的初始姿态，却没有学会“对准后停止横向、再
安全推进”的动作闭环。

因此当前结论是：目标条件 IK 的几何和动力学可行性已确认，但不能直接把旧 offset5
策略或普通 PPO 从头训练套到该 reset 分布上。下一步应加入 IK teacher/残差动作或阶段化
训练：先冻结 XY、只学习安全 Z 插入，再逐步开放 6-D 动作；暂保留 transfer `model_50`
作为在线 IK 分支对照，不宣布成功。

### 13.43 不训练直接复用 offset5 策略的两阶段验证

按照“两阶段：先在线目标条件 IK，再直接执行 offset5 插入策略”的方案，使用
`runs/ppo/offset5/20260816_164356/model_499.pt` 做了对照验证。原始 offset5 环境在
512 回合中成功 `512/512=100.00%`；加入 ±20 mm hole 的在线 IK 后，即使将目标设置为
hole 中心加上随机 mount 残差，使 IK 后的 peg-hole 相对几何尽量接近 offset5，512 回合
仅成功 `134/512=26.17%`，全部失败为超时而不是过深插入。

进一步把策略输入中的关节位置/速度观测替换为 offset5 的 canonical posture，作为“只
消除本体感知分布差异”的诊断实验，结果提高到 `180/512=35.16%`，仍远低于 100%。按
目标半径分箱，残差版本的成功率从 0--5 mm 的 `96.30%` 下降到 15--20 mm 的 `13.86%`
以及 20 mm 以上的 `6.45%`；canonical proprioception 后边缘区有所改善，但仍只有
`27.59%`。

结论是：相对位置对齐是必要条件，但不是充分条件。IK 会改变机械臂的绝对关节姿态、
Jacobian 和动作到末端速度的映射；offset5 策略使用的关节位置/速度观测和原有动力学
响应因此发生分布偏移。当前不能宣称“无需训练即可把 offset5 策略迁移到 hole20”。

### 13.44 XY 动作适配器诊断

在评估器中加入了可选的 hole-centering XY action assist，参数为
`--xy_assist_blend`、`--xy_assist_until_depth_mm` 和 `--xy_assist_radial_mm`，默认关闭，
不影响原有评估。用 offset5 `model_499` 在在线 IK 残差环境做 32 回合诊断：

```text
blend=1.0, assist until depth=100 mm: 7/32=21.88%, timeout=24, over=1
canonical proprioception + blend=0.5: 4/32=12.50%, timeout=28, over=0
blend=0.5, assist only above -10 mm: 5/32=15.63%, timeout=27, over=0
```

这些小样本结果均没有超过此前不加适配器的 `26.17%`，说明简单地强行覆盖 XY
动作会破坏 offset5 策略原有的动作节奏，不能作为最终迁移方案。该适配器仅保留为
诊断开关；后续应改为受限的残差 teacher/动作适配器，并用短程 transfer 训练其参数，
而不是把 XY action 完全替换为几何控制量。

在同一环境下再测试“只门控下降、不改 XY”的安全插入策略（径向 `2 mm`、倾角 `2°`、
全程门控，32 回合），结果为 `7/32=21.88%`，`timeout=25`、`over=0`。它能够消除
过深插入，但不能解决边缘 XY 对准，因此仍不足以直接复用旧策略。当前应把门控作为
训练/评估安全层保留，而把主要精力放在带 teacher 的残差策略适配上。

### 13.45 offset5 100% 成功轨迹可视化

新增 `scripts/plot_offset5_rollout.py`，使用已验证的
`runs/ppo/offset5/20260816_164356/model_499.pt`，在原始
`Isaac-PegInHole-Franka-OSC-Pose6D-PegOffset5mm-v0` 环境、seed=42、单环境下记录径向
误差、插入深度和倾角。该次 rollout 在 15 个控制步后成功，终端指标为：

```text
radial=0.979 mm, depth=19.218 mm, tilt=0.713 deg
```

图像输出为 `/tmp/offset5_model499_rollout.png`，原始轨迹数据为
`/tmp/offset5_model499_rollout.npz`。图中的红色/绿色阈值分别对应 C0 的径向、倾角和
15--40 mm 成功深度窗口。

### 13.46 Isaac Lab GUI 回放

使用已有 `scripts/visualize.py` 启动了 offset5 `model_499.pt` 的两阶段可视化：先在
128 个并行环境中捕获严格成功轨迹，再启动独立 Isaac Lab GUI 回放。此次捕获到的轨迹
为 `12` 个控制步，终端径向 `1.31 mm`、深度 `15.92 mm`、倾角 `1.10°`，回放过程
正常完成。该 GUI 模式是已验证关节轨迹的可视化回放，不重新执行策略控制，避免界面
进程和策略进程之间的物理状态差异。

### 13.47 GUI 桌面消失问题修复

反馈发现回放时机械臂运动、时间线启动后桌面夹具消失。原因是 `visualize.py` 原先只在
`timeline.play()` 前写入一次静态刚体状态；Isaac Sim 4.5 的 Fabric/PhysX 更新可能在
后续时间线帧覆盖这些状态。现已修改 `present()`，每一帧重新写入所有静态刚体，包括
`fixture_left/right/front/back`、`hole_board` 和 36 个 `hole_wall`。修复后的短回放已
正常完成，日志确认全部对象持续参与回放。

### 13.48 GUI-only SeattleLabTable 显示

进一步确认 offset5 的物理配置在 Day3 为避免桌面碰撞，明确设置了 `scene.table=None`，
因此严格训练/评估场景不包含原始 SeattleLabTable。为满足可视化需求，`visualize.py`
新增 `--show_table` 选项：仅在 GUI 回放环境中恢复
`Props/Mounts/SeattleLabTable/table_instanceable.usd`，不改变 checkpoint 捕获阶段的
物理场景。已用该选项完成短回放，Table 资产成功加载，静态夹具也持续保留。

### 13.49 GUI 圆孔薄板持续可见

进一步确认 `hole_board` 本身只是奖励/观测使用的不可见参考标记，训练物理中的薄板由
四块 `fixture_*` 夹具拼成；因此仅恢复 SeattleLabTable 仍不能保证用户看到一块完整的
带圆孔薄板。现已在 `visualize.py` 的 GUI 回放阶段增加无碰撞 USD 可视化网格：外轮廓为
150 mm 半边长、厚度 20 mm，中心为真实圆形开口，并在每个回放帧根据四块夹具的记录位置
更新平移。该网格只存在于回放，不进入训练或评估物理。

短回放验证命令：

```bash
source scripts/setup_env.sh
python -u scripts/visualize.py \
  --checkpoint runs/ppo/offset5/20260816_164356/model_499.pt \
  --stage offset5 --episodes 1 --capture_num_envs 1 --device cuda:0 \
  --show_table --show_fixture_plate \
  --screenshot_dir /tmp/offset5_gui_plate_check
```

验证结果为 `1/1` 严格成功；初始和末端截图均显示薄板、圆孔、圆柱插棒及机械臂，输出为
`/tmp/offset5_gui_plate_check/episode_1_initial.png` 和
`/tmp/offset5_gui_plate_check/episode_1_final.png`。

### 13.50 继续验证 20 mm IK 补偿与 canonical transfer

为检验“完整 IK 改变关节观测分布”这一判断，在
`target_conditioned_arm_pose_online` 中加入了 `target_blend` 参数。它允许只消除初始
XY 误差的一部分，剩余误差交给策略；同时新增
`Isaac-PegInHole-Franka-OSC-Pose6D-HoleRandom20mm-MountStableOnlineIKCanonical-v0`，物理
上使用真实 IK 姿态，但策略的关节位置/速度观测固定为 offset5 的 canonical 接口。

使用旧 offset5 `model_499.pt`、38 mm 深度屏障、2 mm/2° 插入门控进行 128 回合诊断：

```text
IK blend=0.50, native proprioception       24/128 = 18.75%
IK blend=0.50, canonical proprioception    41/128 = 32.03%
IK blend=0.75, canonical proprioception    46/128 = 35.94%
IK blend=1.00, canonical proprioception    53/128 = 41.41%
```

这说明 canonical 接口比原始 IK 关节观测明显稳定，且完整 IK 优于部分 IK；但它仍不是
可直接部署的成功方案。

随后在 canonical-IK 环境上进行 512 环境、100 轮 transfer（`lr=1e-4`、2 epochs）时，
第一个更新后的 checkpoint 即退化到 `0/64`，证明普通 PPO 更新会破坏旧插入策略。于是
改用保护性微调（512 环境、50 轮、`lr=1e-5`、1 epoch、entropy=0），运行目录为：

```text
runs/ppo/hole20_online_ik_canonical/20260817_225744/
```

其 `model_49.pt` 在 64 回合短评估中为 `34/64=53.12%`，但正式 512 回合为
`216/512=42.19%`，全部失败为 timeout、`over=0`。因此当前保留旧 offset5 actor 作为
canonical-IK 基线，不采用该 transfer checkpoint 作为最终模型。

当前结论：目标条件 IK 的几何补偿仍然可行；真正瓶颈是 IK 后的 Jacobian/动作映射和
策略分布，而不是初始可达性。下一步应冻结旧 actor，只训练小幅 residual action/adapter，
并在中心样本保持率约 50% 的混合课程上逐步开放边缘样本，避免再次出现 PPO 一轮更新
摧毁原有插入能力。

### 13.51 冻结 teacher 的 bounded residual adapter

为避免普通 PPO transfer 直接破坏 offset5 actor，新增 `scripts/residual_policy.py` 中的
`ResidualActorCritic`：加载旧 actor 后冻结其参数和探索标准差，只训练一个零初始化的
`64-64-Tanh` residual head，动作残差缩放为 `0.15`。训练器和评估器分别通过
`--residual_adapter` 启用该策略，旧 checkpoint 在没有 residual 参数时仍可严格作为零残差
teacher 使用。

先用 3 轮 smoke/transfer 验证链路，运行目录为：

```text
runs/ppo/custom/20260817_233032/
```

在 canonical-IK、38 mm 深度屏障、2 mm/2° 门控、seed=42、512 回合正式评估中：

```text
冻结旧 teacher（零残差）：198/512 = 38.67%，timeout=314，over=0
residual model_0（3 轮训练后）：217/512 = 42.38%，timeout=295，over=0
residual model_2（3 轮训练后）：260/512 = 50.78%，timeout=252，over=0
```

其中 `model_2.pt` 是目前该分支的最好 checkpoint，但尚未达到接受阈值，不能称为成功
模型。继续用较小学习率 `5e-5`、1 epoch 训练 30 轮，运行目录为
`runs/ppo/custom/20260817_233853/`；其 `model_29.pt` 正式评估为 `186/512=36.33%`，
说明 residual head 仍可能逐步漂移，长训练会损害短期获得的收益。

因此当前应保留 `model_2.pt` 作为研究性候选，生产/后续课程仍以原始 offset5 actor 和
在线 IK canonical 基线为准。下一步重点是给 residual 增加显式范数约束或 KL/teacher
动作保持项，并采用按目标半径分阶段开放的课程，而不是继续无约束延长 PPO 训练。

### 13.52 残差输出显式有界化复核

检查发现原 residual head 虽然隐藏层使用 Tanh，但最后一层是无界线性层，因此此前的
“bounded residual”并不是真正的动作有界。现已在 `update_distribution` 和
`act_inference` 中对最终输出增加 `tanh`，再乘以 `0.15` 缩放。代码通过 py_compile，且
旧 residual checkpoint 可以正常加载。

对已有最好候选 `runs/ppo/custom/20260817_233032/model_2.pt` 重新评估，显式有界后为
`243/512=47.46%`，仍高于冻结 teacher 的 `198/512=38.67%`，但低于未有界版本的
`260/512=50.78%`。在有界版本上重新训练 10 轮，目录为
`runs/ppo/custom/20260817_234424/`，其 `model_9.pt` 为 `194/512=37.89%`。

这表明单纯输出裁剪能限制危险动作尖峰，但不能解决长期 PPO 漂移；当前最稳妥的研究
候选仍是未有界版本的早期 `model_2.pt`，而不是长训或有界长训 checkpoint。后续若继续，
应加入显式 teacher-action KL/残差平方惩罚，并保存每轮独立评估的 best checkpoint，不能
仅按训练 reward 或最后一轮选择模型。

### 13.53 residual transfer 的 weight-decay 约束尝试

为在不重写 RSL-RL PPO 更新器的情况下抑制 residual 漂移，训练器新增
`--residual_weight_decay`。启用 residual adapter 时，该参数会给优化器加入保守的 L2
衰减（当前实现同时作用于可训练 residual 和自适应 critic）；默认值为 `0`，不影响旧流程。

使用 `weight_decay=1e-3`、`lr=5e-5`、1 epoch、10 轮短训，运行目录为
`runs/ppo/custom/20260818_000702/`。其 `model_9.pt` 在相同 512 回合评估中为
`194/512=37.89%`，与有界 residual 的无衰减短训结果基本相同，暂未观察到收益。

因此 weight decay 只能作为可选稳定化旋钮，不能替代 teacher-action KL 或显式 residual
平方惩罚。当前最佳研究性结果仍是未有界 residual 的早期 `model_2.pt`（50.78%/512），
但尚未达到接受标准。

### 13.54 显式 residual-action 惩罚接入 PPO

新增 `scripts/residual_ppo.py`，在标准 PPO minibatch loss 中加入：

```text
loss = PPO_loss + residual_penalty_coef * mean(||bounded_residual_action||²)
```

由于 RSL-RL 只把算法名 `PPO` 识别为 RL 模式，训练器在启用该功能时临时将
`on_policy_runner.PPO` 替换为 `ResidualPPO`，配置入口仍保持 `class_name=PPO`。3 轮
smoke 已成功完成，checkpoint 可以正常保存。

在 512 环境、10 轮、`lr=5e-5`、1 epoch、canonical-IK、38 mm 屏障和 2 mm/2° 门控下，
分别测试惩罚系数 `10` 和 `1000`：

```text
coef=10：runs/ppo/custom/20260818_002239/model_9.pt  -> 194/512 = 37.89%
coef=1000：runs/ppo/custom/20260818_002602/model_9.pt -> 194/512 = 37.89%
```

系数 `1000` 确实使 residual head 最后一层权重范数下降，但成功率没有改善，说明当前
瓶颈不是 residual 幅度本身，而是 teacher actor 在 IK 后的动作映射失配。该惩罚实现已
保留，后续可以配合中心到边缘的分阶段课程；当前仍不接受这些 checkpoint 作为最终模型。

### 13.55 residual 惩罚的长程稳定性测试

进一步使用 `residual_penalty_coef=1000`、512 环境、`lr=5e-5`、1 epoch 训练 30 轮，
运行目录为 `runs/ppo/custom/20260818_004414/`。最终 `model_29.pt` 的正式评估结果为：

```text
180/512 = 35.16%，timeout=332，over=0
```

该结果低于 10 轮 checkpoint 的 `37.89%`，也低于早期未有界 residual 候选的 `50.78%`。
因此残差平方惩罚虽然确实减少了 residual 参数幅度，但不能阻止 critic/策略更新造成的
长期性能下降。当前不再继续单纯延长惩罚训练，下一步应改为“固定 teacher actor +
冻结 critic 或极低学习率 critic”，并配合中心样本到边缘样本的分阶段开放。

### 13.56 冻结 critic 的 residual transfer 对照

为区分 actor residual 漂移和 critic 漂移，新增 `--residual_freeze_critic`，使 transfer
阶段同时冻结 teacher actor 与旧 critic，仅更新 residual head。使用惩罚系数 `1000`、
512 环境、10 轮训练，运行目录为 `runs/ppo/custom/20260818_004809/`。

`model_9.pt` 正式评估为：

```text
195/512 = 38.09%，timeout=317，over=0
```

相比未冻结 critic 的 10 轮结果 `194/512=37.89%` 只有随机波动级别的提升，说明当前
主要问题不是 critic 单独漂移。冻结 critic 选项已保留用于后续消融，但暂不作为解决方案。

### 13.57 几何控制器验证：从 hole20 到 hole50/23mm

按照“先抛弃 PPO，用几何控制器验证可达性和插入动力学”的方案，新增
`scripts/eval_geometric_controller.py`。控制器完全不加载 checkpoint，仅使用实时 peg-tip、
孔中心和姿态误差，通过现有 6-D OSC 的 bounded relative-pose action 完成：

```text
XY 闭环对准 → 姿态保持 → 径向/倾角门控 → 低速 Z 插入 → 物理 termination 判定
```

在 `tilt_gate=2°`、512 环境、seed=42 下得到：

```text
hole20 MountStableReward：512/512 = 100.00%，timeout=0，over=0
hole50：                  512/512 = 100.00%，timeout=0，over=0
hole23mm（±50mm）：       512/512 = 100.00%，timeout=0，over=0
```

终端统计分别为：

```text
hole20：radial=0.493mm，depth=15.388mm，tilt=1.965°
hole50：radial=0.484mm，depth=15.387mm，tilt=1.967°
hole23mm：radial=0.953mm，depth=15.695mm，tilt=1.897°
```

这证明当前几何、IK/OSC 动力学和 ±50 mm 随机空间本身是可行的；此前 PPO 失败主要是
策略迁移和观测/动作分布不一致，而不是机械臂不可达。该结果是几何控制器的 oracle 基线，
因为它使用仿真中准确的孔中心；下一步应在此控制器上加入位置噪声、孔中心估计误差和动力学
扰动，再让 RL 学习 residual，而不是重新训练完整插入策略。
