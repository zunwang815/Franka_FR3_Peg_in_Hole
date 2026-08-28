# Franka FR3 Peg-in-Hole 仿真任务 — Day 4 工作总结

> 日期：2026-08-14  
> 主题：Fixed-Joint Peg 物理化、Oracle 全链路验收、PPO 训练/评估重构与可信可视化

---

## 一、今日总体结论

今天完成了项目从“控制器与环境仍在调试”到“具备可训练、可评估、可回放的单孔开发基线”的关键跨越：

1. Peg 从逐帧瞬移同步改为机器人 articulation 内的 Fixed Joint 工具；
2. 建立了真实孔壁碰撞、接触传感器与阻挡反例测试；
3. Oracle 在各级开发任务上完成批量验收，最难的 23 mm 孔任务达到 128/128；
4. PPO 训练入口、奖励、日志与评估入口完成重构；
5. baseline PPO 检查点在 128 并行环境、1024 episode 评估中达到 100%；
6. 可视化改为“headless 严格采集 + 独立 GUI 纯回放”，解决 GUI 改变物理轨迹的问题；
7. 同时暴露出两个不能忽略的可信度问题：并行环境数量依赖和过度插入。

因此，当前可以称为 **Pose6D 单孔 baseline 开发链路已打通**，但还不能称为最终任务完成，也不应直接开始 Phase 2 六孔任务。

---

## 二、环境与物理模型成果

### 2.1 Fixed-Joint Peg 资产

新增并生成：

```text
assets/panda_with_fixed_peg.usda
```

资产参数：

| 项目 | 数值 |
|---|---:|
| Peg 半径 | 10 mm |
| Peg 高度 | 100 mm |
| Peg 质量 | 0.25 kg |
| 固定安装偏移 | 177 mm |
| 连接方式 | Fixed Joint |

验证日志已经明确显示：

```text
PHYSICS MODEL: FixedJoint articulation peg
```

这意味着 Peg 不再依赖 interval event 每步瞬移到末端，机器人、Peg 和接触力进入同一个 articulation 物理链路。

> 限制：当前底层机器人资产仍是 Panda 等效模型，不是最终要求的官方 FR3 资产。

### 2.2 碰撞与接触验证

完成了两类互补测试：

- 正常对准插入：成功，接触力接近 0 N；
- 故意偏置 6 mm 的 blocked 测试：无法插入并产生真实接触力。

关键结果：

```text
TEST MODE: blocked (target offset=6.0mm)
SUCCESS: NO
MAX CONTACT FORCE: 22.92N
BLOCKED TEST: PASS
```

该反例证明孔壁碰撞不是纯视觉装饰，错误位置下压会被物理阻挡。

### 2.3 位姿与可操纵性修复

针对竖直姿态与 Jacobian 奇异问题，完成非奇异竖直候选搜索。采用候选关节姿态后：

```text
sigma_min ≈ 0.228
初始 tilt ≈ 1.7°
初始插入深度 ≈ -50 mm
```

随后加入 6D 姿态反馈，并以 settled tilt 为目标、释放圆柱 Peg 无意义的 yaw 约束，使单环境 Oracle 首次在严格条件下成功：

```text
radial <= 2 mm
depth >= 15 mm
tilt <= 2°
SUCCESS: YES
```

---

## 三、Oracle 验收成果

### 3.1 Oracle 控制与批量脚本

`oracle_baseline_test.py` 与 `oracle_batch_test.py` 经过多轮诊断和重构，主要解决：

- 3D/6D action 维度不一致；
- Peg/末端坐标偏移错误；
- hole 高度与机械臂工作空间不匹配；
- reset 后状态与真实 settled 状态不一致；
- orientation feedback 错误锁定 yaw；
- 批量脚本中 GPU 同步与张量访问造成的假“卡死”；
- mount offset 随机化写入 Fixed Joint 后不生效；
- 多环境下 mount 目标和实际状态漂移；
- GUI render 更新改变控制物理步数。

### 3.2 分阶段验收结果

已完成的代表性验收：

| 任务 | 环境数 | 结果 |
|---|---:|---:|
| Fixed-Joint baseline | 128 | 100% |
| Peg mount offset ±5 mm | 128 | 100% |
| Hole random ±10 mm | 32 | 100% |
| Hole random ±50 mm | 32 | 100% |
| 25 mm 孔 | 128 | 100% |
| 23 mm 孔 + ±50 mm 随机 + mount offset | 128 | 100% |

最难 Oracle 验收日志：

```text
Task: Isaac-PegInHole-Franka-OSC-Pose6D-Hole23mm-v0
Environments: 128
Insertion radial gate: 0.70mm
Success: 128
Timeout: 0
Success rate: 100.00%
C0 ACCEPTANCE: PASS
```

这说明：在已知目标几何信息的解析式控制器下，环境、动作接口、机械臂工作空间和碰撞结构具备完成任务的能力。

> Oracle 100% 证明“任务可解”，不等于 PPO 已经在所有课程阶段学会该任务。

---

## 四、PPO 训练链路重构

### 4.1 训练入口统一

重构 `scripts/train_phase1.py`，形成六阶段任务入口：

```text
baseline -> offset5 -> hole10 -> hole50 -> hole25 -> hole23
```

训练入口现支持：

- 统一任务 ID；
- verified physics 配置；
- checkpoint transfer / resume；
- 固定 seed；
- run metadata；
- smoke test；
- TensorBoard；
- tqdm 进度条替代逐 iteration 大段日志。

统一物理设置：

```text
physics dt = 1/120 s
control decimation = 4
solver iterations = 16 / 4
enhanced determinism = True
```

### 4.2 奖励修复

今天进一步修复了奖励中的实现问题：

1. `time_penalty` 改为返回正 1，由负权重形成真实时间惩罚；
2. `insertion_progress` 改为基于相邻状态的真实深度增量；
3. `fine_alignment` 改为 potential difference，避免策略停在孔边持续刷奖励；
4. 保留 success bonus、jam、tilt、action rate 和 joint velocity 约束。

Smoke 训练验证了训练入口、网络、环境、TensorBoard 和 checkpoint 保存链路均能正常运行。

### 4.3 baseline 正式训练结果

训练目录：

```text
runs/ppo/baseline/20260814_221904
```

评估的检查点：

```text
model_250.pt
model_400.pt
model_450.pt
model_499.pt
```

---

## 五、PPO 评估成果

### 5.1 评估入口重构

`scripts/eval_phase1.py` 现在支持：

- 多 checkpoint 依次评估；
- 128 并行环境；
- 1024 episode；
- deterministic inference；
- success / timeout 分开统计；
- Wilson 95% 置信区间；
- 平均 episode 步数；
- JSON 评估报告；
- 自动选择通过验收的最佳 checkpoint。

### 5.2 baseline 评估结果

四个检查点均得到：

```text
1024 / 1024 success
success rate = 100.00%
CI95 = [99.63%, 100.00%]
```

其中：

| Checkpoint | 成功率 | 平均步数 |
|---|---:|---:|
| model_250.pt | 100% | 7 |
| model_400.pt | 100% | 12 |
| model_450.pt | 100% | 13 |
| model_499.pt | 100% | 12 |

当前 baseline 推荐检查点：

```text
runs/ppo/baseline/20260814_221904/model_250.pt
```

评估报告：

```text
runs/eval/baseline/20260814_223136.json
```

---

## 六、可视化重构成果

### 6.1 发现的问题

最初直接在 GUI 中运行策略时出现：

- 窗口冻结，结束后突然跳到成功画面；
- 修复刷新后，GUI render 又改变物理步进，导致策略失败；
- 策略一度插入后继续拔出并摆动；
- 单环境 GUI 结果和 128 环境评估不一致。

### 6.2 最终方案：两进程隔离

`scripts/visualize.py` 现采用：

```text
轻量父进程
  ├── Headless Isaac：128 环境运行正式评估物理，截取严格成功帧
  └── GUI Isaac：只回放关节与夹具状态，不再运行策略
```

关键保证：

- 捕获使用与正式评估相同的 128 并行环境；
- 成功终止自动 reset 前缓存精确成功帧；
- GUI 与策略物理完全隔离；
- 随机孔位从源环境平移到 GUI env0；
- timeline 运行时逐帧覆盖 articulation 状态，保证 Fabric 刷新；
- 夹爪在回放中保持闭合；
- 空轨迹不会启动 GUI；
- checkpoint 目录与 `--stage` 不匹配时主动报错；
- 默认初始姿态停留 3 秒，便于观察；
- 最终成功姿态可配置保持时间。

已验证采集结果：

```text
[CAPTURE] SUCCESS 1/1: env=0 steps=7
radial=0.60mm depth=70.27mm tilt=1.74deg
[CAPTURE] Saved 1 verified trajectory/trajectories
```

当前 baseline 可视化命令：

```bash
python -u scripts/visualize.py \
  --stage baseline \
  --checkpoint runs/ppo/baseline/20260814_221904/model_250.pt \
  --episodes 1 \
  --initial_hold_seconds 3 \
  --speed 0.2 \
  --replay_fps 60 \
  --hold_seconds 15
```

---

## 七、今日暴露的不足与风险

### 7.1 严重：PPO 结果依赖并行环境数量

同一 baseline checkpoint：

- 128 环境：7 步成功；
- 1 环境：239 步超时。

这说明策略结果对 PhysX batching、GPU 求解顺序或环境规模存在依赖。当前 1024 episode 的 100% 结果在“128 并行环境评估协议”下成立，但尚不能宣称对任意仿真配置都稳健。

明天需要进行：

```text
num_envs = 1, 2, 4, 8, 16, 32, 64, 128
```

的分层一致性测试，定位成功率从何处发生突变。

### 7.2 严重：成功条件允许过度插入

成功轨迹最终深度：

```text
depth = 70.27mm
```

而 baseline 只要求：

```text
depth >= 15mm
```

当前成功条件没有最大深度限制，可能允许：

- 过度插入；
- 穿透底部或孔壁；
- 大动作一步跨过接触区域；
- 利用数值离散误差获得成功。

因此当前 100% baseline PPO 是“按现有判据成功”，还不能等同于理想物理插入。

### 7.3 baseline 初始条件较简单

当前 baseline：

- Peg tip 初始位于孔面上方约 50 mm；
- 初始 XY 基本居中；
- 100 mm 长 Peg 在画面上显得已经接近孔口。

这是课程 C0 的有意设计，不是回放缺帧，但不能代表最终随机孔位任务难度。

### 7.4 PPO 课程尚未完成

今天真正完成训练和正式评估的是：

```text
baseline
```

尚未证明 PPO 已完成：

```text
offset5 / hole10 / hole50 / hole25 / hole23
```

Oracle 在这些阶段 100% 只说明环境可解。baseline checkpoint 不能通过 `--stage hole23` 冒充最终策略。

### 7.5 机器人仍非官方 FR3

项目名称和目标是 Franka FR3，但当前 Fixed-Joint 资产基于 Panda USD。控制与研究流程可以继续验证，但最终报告必须明确这是等效开发模型，后续仍需迁移官方 FR3 资产并重新验收。

### 7.6 接触数据仍需进一步审计

blocked 测试能测到约 23 N，证明接触传感器有效；但许多正常成功批次接触力为 0 N。需要确认这是无碰撞顺畅插入，还是接触采样/孔底结构不足导致的漏检。

---

## 八、明日优先任务

### P0：先解决可信度，不立即推进长课程训练

1. 增加成功最大深度，例如限定在物理孔深和 Peg 几何允许范围内；
2. 增加底部/穿透检测与失败终止；
3. 对成功轨迹输出最小/最大深度、接触峰值与每步 Z 位移；
4. 运行 `num_envs=1~128` 一致性矩阵；
5. 检查 7 步策略是否因 5 cm/step 动作尺度过大而跳过接触求解；
6. 重新训练并评估修正后的 baseline，目标仍为 ≥98%。

### P1：可信 baseline 通过后推进 PPO 课程

按顺序训练并 transfer：

```text
baseline -> offset5 -> hole10 -> hole50 -> hole25 -> hole23
```

每一阶段都应：

- 保存独立目录；
- 使用对应 stage 评估；
- 记录 1024 episode 结果；
- 不允许 checkpoint/stage 混用；
- 通过门槛后才进入下一阶段。

### P2：最终模型迁移与 Phase 2

1. Panda 等效模型迁移至官方 FR3；
2. 重跑 Oracle、blocked、PPO 验收；
3. 单孔正式任务稳定 ≥90% 后，再开始六孔阵列环境。

---

## 九、明日建议启动检查

```bash
source scripts/setup_env.sh

# 1. 确认 baseline checkpoint 与 baseline task 一致
python -u scripts/eval_phase1.py \
  --stage baseline \
  --checkpoint runs/ppo/baseline/20260814_221904/model_250.pt \
  --episodes 1024 \
  --num_envs 128

# 2. 可视化当前已验证轨迹
python -u scripts/visualize.py \
  --stage baseline \
  --checkpoint runs/ppo/baseline/20260814_221904/model_250.pt \
  --episodes 1 \
  --initial_hold_seconds 3 \
  --speed 0.2 \
  --hold_seconds 15
```

在修改最大深度、动作尺度或碰撞条件后，旧 checkpoint 的成功率不再具有直接可比性，必须重新训练与评估。

---

## 十、今日总结

Day 4 的核心成果不是单一的“100% 成功率”，而是建立了较完整的工程证据链：

```text
物理资产 -> 接触反例 -> Oracle 可解性 -> PPO 训练 -> 批量评估 -> 独立可视化
```

Fixed-Joint Peg、23 mm Oracle 128/128、baseline PPO 1024/1024 和可信回放都是实质性进展。但可视化也帮助发现：当前策略存在并行环境依赖，并以 70.27 mm 深度过度插入。这两个问题必须在课程训练和最终结果汇报前解决。

> 当前状态：开发 baseline 已贯通，环境具备继续研究的基础；最终物理可信度、课程 PPO 和官方 FR3 迁移尚未完成。明天应优先修正验收定义，再继续训练。
