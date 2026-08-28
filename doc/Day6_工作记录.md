# Franka FR3 Peg-in-Hole 仿真任务 — Day 6 工作记录

> 日期：2026-08-18  
> 主题：几何控制器评估固化、感知噪声边界与最终孔鲁棒性验证

## 一、今日目标

承接 Day5 的结论：旧 PPO 暂不继续缝补，先把几何控制器变成可审计的 teacher，并逐级
加入感知与控制扰动。所有结果继续使用固定 seed、明确 termination reason 和 512 回合
严格口径；小样本只用于筛选，不作为最终验收。

## 二、代码推进

修改 `scripts/eval_geometric_controller.py`：

1. 默认 `tilt_gate_deg` 与 Day5 实际几何控制器协议统一为 `2.0°`。
2. 新增可复现的测量噪声接口：
   - `--hole_xy_noise_std_mm`
   - `--tip_position_noise_std_mm`
   - `--tip_orientation_noise_std_deg`
3. 新增测量 EMA 滤波：`--measurement_ema_alpha`，`1.0` 表示关闭。
4. 新增控制链路扰动接口：
   - `--action_noise_std`
   - `--action_delay_steps`
   - `--action_gain_noise_std_pct`
5. 新增 `--output` JSON 报告。报告包含总成功率、timeout、over-insertion、终止深度/径向
   误差/倾角、阶段计数，以及每回合的初始径向误差、孔位偏移、步数和真实终止原因。

当前这些扰动均注入控制器接口，不改变仿真几何和 termination 判据。动作噪声、延迟和
增益扰动属于控制链路扰动，尚不能等同于真实外力/质量/摩擦 domain randomization。

## 三、512 回合结果

任务均使用 `seed=42`、`num_envs=64`，终止原因分开统计。

| 场景 | 配置 | 成功 | Timeout | Over | 结论 |
|---|---|---:|---:|---:|---|
| hole20 | 无噪声 | 512/512 | 0 | 0 | Day5 基线复现 |
| hole20 | 孔中心 0.5 mm、末端位置 0.5 mm、姿态 0.25°，无滤波 | 8/512 | 504 | 0 | 瞬时测量抖动导致门控反复切换 |
| hole20 | 同上 + EMA `alpha=0.2` | 512/512 | 0 | 0 | 传感器噪声可被滤波消化 |
| hole20 | 上行配置 + 动作噪声 0.005、增益扰动 2% | 512/512 | 0 | 0 | 当前可用弱扰动基线 |
| hole23 mm / ±50 mm | 上行配置 | 512/512 | 0 | 0 | 最终孔的首轮鲁棒基线通过 |

控制扰动边界筛选（128 回合）：

| 配置 | 成功 | Timeout | Over | 结论 |
|---|---:|---:|---:|---|
| 传感器噪声 + EMA + 动作噪声 0.02 + 1 步延迟 + 增益扰动 5% | 2/128 | 126 | 0 | 对动作时序扰动敏感 |
| 传感器噪声 + EMA + 动作噪声 0.005 + 增益扰动 2% | 128/128 | 0 | 0 | 可作为 residual 第一阶段目标 |

## 四、结果文件

报告已保存到 `runs/eval/day6_geom/`：

```text
hole20_clean_20260818.json
hole20_sensor_05_20260818.json
hole20_sensor_05_ema02_20260818_full.json
hole20_sensor_control_weak_20260818_full.json
hole20_sensor_control_20260818.json
hole23mm_sensor_control_weak_20260818_full.json
```

最终 hole23 mm 鲁棒结果的终端统计为：

```text
success       = 512/512
timeout       = 0
over          = 0
depth         = 15.000--21.626 mm
radial        = mean 0.828 mm, max 1.299 mm
tilt          = mean 1.816 deg, max 1.999 deg
```

## 五、今日结论

1. 几何控制器已经从“只打印总成功率”升级为固定 seed、逐回合可审计的评估器。
2. 0.5 mm 级孔中心/末端噪声并不会破坏控制器本身；不滤波时会把 504 个回合变成超时，
   EMA `alpha=0.2` 后 hole20 和最终 hole23 mm 均保持 512/512。
3. 当前主要敏感项是控制时序：1 步延迟叠加较强动作噪声后成功率迅速下降，后续 residual
   学习应优先处理动作预测/补偿，而不是重新学习完整插入策略。
4. 23 mm 孔、±50 mm 工作空间在弱感知/控制扰动下仍有完整通过证据，但这仍是已知孔中心
   的几何 teacher 结果，不代表最终感知部署成功。

## 六、下一步入口

下一轮按以下顺序推进：

1. 把当前几何控制器的 action 计算提取为共享 `geometric_teacher` 模块，使训练环境和
   评估器使用同一 teacher 语义。
2. 在 hole20 上冻结 teacher，只训练有界 residual head；第一阶段固定使用：
   `hole_xy=0.5 mm`、`tip=0.5 mm`、`tilt=0.25°`、EMA `0.2`、动作噪声 `0.005`、增益
   扰动 `2%`，保留 hole20 内圈样本作为锚点。
3. 每个 checkpoint 用 512 回合严格评估，并分开报告成功、timeout、over-insertion；
   residual 训练未通过前不扩展到 hole50/23 mm。
4. 老师原始任务未要求控制器延迟或动作时序扰动；后续主线固定保持
   `action_delay_steps=0`，不再把延迟作为验收条件。若后续需要扩展鲁棒性，只单独评估
   任务要求范围内的观测噪声、Peg 抓取偏差、孔位随机化和多孔阵列，不把延迟混入主线。

## 七、Geometric teacher residual 推进

继续推进后新增 `scripts/geometric_teacher.py`，将几何 teacher 的位置门控、深度目标和
姿态修正统一为共享 action 规则。评估器和训练器现在使用相同的 action 语义。

新增 `GeometricTeacherResidualActorCritic`：

```text
冻结解析 teacher
只训练 64-64-Tanh residual head
residual scale = 0.15
teacher 直接读取 30-D 相对策略观测
```

teacher 使用策略观测中的 `peg_to_hole_vec` 和 `peg_tilt`，不读取绝对世界位姿或隐藏
mount 关节；训练时启用位置 ±0.5 mm、倾角 ±0.25° 的观测噪声，始终不启用动作延迟。

Smoke 训练已通过：

```text
runs/ppo/hole20_reward/20260818_125830/
64 envs, 3 iterations, checkpoint model_0.pt 正常生成
```

保守 residual 训练已完成：

```text
runs/ppo/hole20_reward/20260818_130108/
1024 envs, 50 iterations, lr=1e-5, 1 epoch,
entropy=0, residual penalty=100, K=20, barrier=38mm
```

确定性严格评估结果：

| Checkpoint | hole20 成功 | Timeout | Over |
|---|---:|---:|---:|
| model_0.pt | 512/512 | 0 | 0 |
| model_49.pt | 512/512 | 0 | 0 |

model_49 从 hole20 迁移到最终 23 mm 孔、±50 mm 空间后仍为：

```text
512/512 = 100%
timeout = 0
over-insertion = 0
```

该结果证明当前解析 teacher + residual 入口已经覆盖 Day6 首级噪声协议；下一步可以
转向老师任务的第二部分：2×3 静止孔阵列、3 cm 孔间距和 10×10 cm 阵列整体随机化。

## 八、2×3 阵列环境审计与第一轮修复

核查发现旧阵列任务原本不能启动：`ArrayRewardsCfg` 引用了不存在的 `mdp.alignment`
等旧接口，目标 `hole_id` 也始终是全零。已完成以下修复：

1. 将阵列 reward/termination 接到统一的几何函数和 15--40 mm 成功窗口。
2. `hole_id_onehot` 改为读取每回合真实 `_target_hole_id`。
3. 新增 reset 阶段目标孔选择：训练时在 0--5 均匀采样，也支持指定固定 hole id。
4. 接入 3 cm 间距的 2×3 坐标偏移，并新增 23 mm 物理尺寸的目标 sleeve 环。
5. 修复 legacy standalone Peg 的几何读取和 finger body 查找异常。

`Isaac-PegInHoleArray-Franka-IK-Abs-v0` 当前已通过 2 环境 reset/随机动作 smoke，配置
可以加载，目标 id 和目标环 reset 链路已接通。

## 九、今日继续推进：完整六孔实体与阵列 OSC

将过渡实现补齐为真正的 2×3 物理孔阵列：

1. `PegInHoleArraySceneCfg` 现在生成 6 个独立的 36 段碰撞套筒，共 216 个物理段；孔中心
   的列/行间距固定为 30 mm，reset 时只移动阵列整体原点，目标 marker 随指定 hole id
   移到对应孔中心。
2. 阵列 observation 增加 `peg_tilt`，并新注册
   `Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0`，动作空间确认是 6 维，策略观测确认是
   `9+9+3+3+6+6=36` 维。
3. 阵列 OSC 使用已搜索的可操作初始姿态，关闭旧桌面碰撞，将套筒设置在初始 peg tip
   下方约 60 mm；增加 reset 时的 peg 同步，避免控制器读到第一步 stale peg pose。
4. `eval_geometric_controller.py` 增加固定目标孔选项，并兼容阵列没有 over-insertion
   终止项的情况；主线仍保持 `action_delay_steps=0`。

已验证：

```text
Isaac-PegInHoleArray-Franka-OSC-Pose6D-v0
2 env reset/random-step smoke: PASS
action shape: 6
policy observation shape: 36
完整六孔场景加载: PASS
```

阵列闭环的当前结果还不是验收通过：固定 hole 1、seed 42 的几何 teacher 在调整到
阵列接近阶段后，12 s 回合末径向误差约 7--12 mm，未进入 1.3 mm 成功窗口；部分目标在
更大的控制尺度下约为 6 mm，仍失败。问题集中在阵列行偏移下的 OSC 横向收敛和姿态漂移，
不是延迟噪声。下一步应先用逐目标的 IK/OSC reachability 轨迹校准初始姿态与横向增益，
再进行六孔 teacher 逐孔验收；在达到 90% 前不训练阵列 residual policy，也不把当前
结果写成“阵列任务已完成”。

## 十、分层验证结果：先固定阵列原点

新增 `eval_geometric_controller.py --array_origin_zero`，先关闭整个阵列的 ±5 cm 随机化，
只保留 2×3 阵列自身的目标孔偏移。复测前修正了阵列 marker 基准中心，使 marker、六个
物理 sleeve 和目标 id 使用同一个 `(0.254, 0.0, 0.493)` 基准。

在固定阵列原点、固定 seed、12 s 回合、无动作延迟下：

| 目标孔 | 结果 | 终端径向误差 |
|---:|---:|---:|
| 0 | timeout | 7.385 mm |
| 1 | timeout | 6.762 mm |

这一步验证了诊断思路，但也排除了“阵列整体 ±5 cm 随机化是唯一原因”：即使原点固定，
当前阵列 OSC 仍无法把 15--30 mm 的行/列目标偏移收敛到 1.3 mm。当前需要继续校准
阵列目标偏移下的横向 reachability、姿态保持和 Peg/hand 几何映射；在此之前不加入
整体随机化，也不开始阵列 RL 训练。

## 十一、固定 Peg 链修复与阵列分层验收通过

进一步对照任务 1 的 OSC 配置后定位到根因：阵列版使用普通 Panda + 独立刚体 Peg，
而任务 1 的已验收控制器使用 `assets/panda_with_fixed_peg.usda` 和 `peg_mount` 动力学。
两者虽然观测中的 Peg tip 位置相近，但 OSC 实际控制的刚体链不同，导致阵列版出现
横向残差和姿态漂移。

已将阵列 Pose6D-OSC 切换到任务 1 的固定 Peg 链，并同步修正：

1. 固定 Peg USD、Peg mount actuator 和任务 1 的可操作初始姿态。
2. 阵列孔面移动到任务 1 已验证的 fixture 高度。
3. 阵列 joint observation 显式只读取 arm + finger 的 9 个关节，保持 teacher 的
   36 维观测布局，不把 Peg mount 关节混入策略输入。

分层验收结果（无动作延迟）：

| 阶段 | 回合 | 成功 | 成功率 | 终端径向误差 |
|---|---:|---:|---:|---:|
| 固定阵列原点，hole 0--5 各 4 回合 | 24 | 24 | 100% | 0.012--0.016 mm |
| 阵列原点 ±5 cm、目标孔随机 | 128 | 128 | 100% | 0.011--0.019 mm |

随机阵列报告：`runs/eval/day6_geom/array_stage_random_origin_128.json`。
这验证了“任务 2 是任务 1 的多目标/坐标扩展”在几何教师物理层面成立。当前尚未
训练阵列 residual policy；下一步应把这套固定 Peg 阵列环境接入 PPO residual，之后
再用同样的 128/512 回合协议验证 RL checkpoint。

## 十二、阵列 residual PPO smoke 与 checkpoint 闭环

已将六孔 Pose6D 阵列接入 `GeometricTeacherResidualActorCritic`。训练入口原先在写入
阵列 run metadata 时无条件读取不存在的 `over_insertion` termination，已改为对该项
进行可选记录；checkpoint evaluator 也已兼容阵列没有 over-insertion mask 的配置。

无动作延迟、teacher 相对观测噪声 0.5 mm / 0.25° 下，64 环境、10 iteration smoke
训练通过，输出：

```text
runs/ppo/custom/20260818_141553/model_0.pt
runs/ppo/custom/20260818_141553/model_9.pt
```

`model_9.pt` 在 64 个随机阵列回合上的严格闭环结果为：

```text
success = 64/64 = 100%
timeout = 0, over-insertion = 0
```

目标阵列整体偏移的 5--10、10--15、15--20 和 ≥20 mm 分组均为 100% 成功。该结果
只说明 residual PPO 的接入链路和 smoke checkpoint 正常，尚不能替代正式训练；下一步
开始阵列正式迭代，并按 128/512 回合协议验收最终 checkpoint。

正式训练使用 1024 环境、50 iteration、学习率 `1e-5` 和 teacher-action residual penalty
`100`，输出目录为：

```text
runs/ppo/custom/20260818_141826/
```

最终模型 `model_49.pt` 的严格随机阵列验收结果：

```text
128/128 = 100%
512/512 = 100%
timeout = 0
over-insertion = 0（阵列配置未启用该终止项，按 0 统计）
```

512 回合中，目标阵列整体偏移 0--5、5--10、10--15、15--20 和 ≥20 mm 五个分组
分别为 3/3、15/15、23/23、34/34、437/437，全部成功。终端径向误差均值 0.194 mm，
最大 0.538 mm；终端倾角均值 0.612°，仍低于任务 2° 成功阈值。至此，任务 2 的
固定 Peg 物理阵列、共享 geometric teacher 和 residual PPO checkpoint 均已完成
当前仿真验收；全程 `action_delay_steps=0`。

## 十三、任务 2 可视化交付

已补齐任务 2 的报告图和 Isaac Sim 实景回放：

1. `task2_array_layout.png`：展示 2×3 六孔布局、23 mm 孔径、30 mm 行列间距、目标
   hole id 以及阵列整体在 10 cm × 10 cm 区域内的随机化范围。
2. `task2_success_summary.png`：汇总固定原点 teacher、随机阵列 teacher、PPO 初始
   checkpoint 和最终 checkpoint 的成功率；最终模型 512/512 以柱状图显示。
3. `replay/episode_1_initial.png`：最终 PPO checkpoint 的真实六孔阵列初始姿态。
4. `replay/episode_1_final.png`：同一条严格成功轨迹的插入完成姿态，Peg 已进入目标孔。
5. `six_holes/episode_1..6_final.png`：修正后的六孔连续回放，目标孔按
   `0 → 1 → 2 → 3 → 4 → 5` 依次切换；六个目标孔各捕获 1 条严格成功轨迹，回放日志
   为 `replayed 6 verified successes`。

可视化产物目录：

```text
artifacts/task2_visualization/
```

可复现命令：

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/visualize_task2_six_holes.py \
  --checkpoint runs/ppo/custom/20260818_141826/model_49.pt \
  --capture_num_envs 128 \
  --screenshot_dir artifacts/task2_visualization/six_holes
```
