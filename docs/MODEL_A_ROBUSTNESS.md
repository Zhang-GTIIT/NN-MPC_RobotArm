# Model A 鲁棒性实验操作手册

本工作流只评估冻结的 Model A GRU MPC 与 Direct IK，不依赖 Model C 数据、C1/C2 checkpoint、Oracle 或 Model C benchmark。Model A MPC 固定使用 `threaded_asap`；Direct IK 没有后台 planner，因此固定为同步 100 Hz 基线。

四类扰动均独立可选，默认 `0` 关闭，`1` 到 `6` 由轻到重：

| 参数 | 扰动对象 | 说明 |
| --- | --- | --- |
| `--payload_level` | 真实 plant | 末端未知 payload；MPC 继续使用标称重力补偿和冻结 Model A。 |
| `--actuator_gain_level` | 真实 plant | 位置执行器 Kp/Kd 衰减，模拟执行器标定失配。 |
| `--force_pulse_level` | 真实 plant | episode 中点、世界 `+Y` 方向、持续 0.10 s 的末端外力。 |
| `--observation_noise_level` | 控制器观测 | q/dq 分开加高斯噪声；真实状态日志保持无噪。 |

等级数值为：payload = 1/2/4/6/9/12 kg；Kp = 0.90/0.80/0.70/0.60/0.45/0.30；外力 = 50/100/200/300/400/500 N；q 噪声 = 0.0002/0.0005/0.001/0.002/0.0035/0.005 rad；dq 噪声 = 0.002/0.005/0.01/0.02/0.035/0.05 rad/s。

## 0. 先理解将要运行什么

每个完整扰动条件会顺序运行两个控制器：

```text
ModelA_MPC：冻结 Model A + threaded_asap + 100 Hz 主控制线程 + 后台 CUDA CEM planner
DirectIK：   不使用 learned dynamics/CEM，直接执行 task-space IK reference 的同步基线
```

`ModelA_MPC` 会运行全部 70 个 case。Direct IK 只对 task-space reference 有定义，因此只运行 circle、figure-8、ellipse、square 共 40 个 case；它不会运行 multi-joint sine、waypoint、chirp。二者的统计比较只在共有的 40 个 task case 上进行。

每个 case 有 500 个 10 ms 控制步，即约 5 s 的真实墙钟控制时间。因此一个完整条件仅控制周期本身约需 `70 × 5 + 40 × 5 = 550 s`，再加上 CEM worker 初始化和保存图像的时间。先运行第 4 节的两个 case 冒烟测试，再启动完整 sweep。

## 1. 初始化

在仓库根目录运行：

```bash
conda activate pendulum-rl
export DEVICE=cuda
export MODEL_A=dynamics_modeling/outputs/checkpoints/gru_20260717_152930
```

`threaded_asap` 需要可用 CUDA，因为 MPC worker 在后台 GPU 上规划：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no CUDA')"
```

输出必须以 `True` 开头。若为 `False`，不要启动 threaded-asap；请先检查 conda 环境中的 PyTorch CUDA 版本、NVIDIA 驱动和 `nvidia-smi`。可选地设置一个可写的 Matplotlib cache 目录，避免首次保存图时出现 cache 警告：

```bash
export MPLCONFIGDIR=/tmp/nn_mpc_matplotlib_cache
mkdir -p "$MPLCONFIGDIR"
```

确认 Model A 文件真实存在：

```bash
ls -lh "$MODEL_A/best_model.pt" "$MODEL_A/normalizer.pt"
```

## 2. 标定逻辑 delay

该脚本用无 sleep 的 virtual run 收集 Model A 规划时间 p95；结果只用于 threaded-asap 的 anticipation delay，不是鲁棒性控制协议。

```bash
python scripts/robustness/calibrate_model_a_delay.py \
  --checkpoint "$MODEL_A/best_model.pt" \
  --normalizer "$MODEL_A/normalizer.pt" \
  --model_type gru --device "$DEVICE" --plans 500 \
  --output_path outputs/robustness/timing/model_a.json

export ROBUST_DELAY=$(python -c "import json; print(json.load(open('outputs/robustness/timing/model_a.json'))['anticipation_delay_steps'])")

cat outputs/robustness/timing/model_a.json
echo "ROBUST_DELAY=$ROBUST_DELAY"
```

该阶段使用 `virtual_asap` 仅为了快速测量 GPU 规划耗时；正式鲁棒性控制阶段仍是 `threaded_asap`。不要把此阶段生成的 rollout 当作鲁棒性结果。

## 3. 建立独立固定 benchmark

创建 70 个固定 case：multi-joint sine、waypoint、chirp、circle、figure-8、ellipse、square 各 10 个。前 30 个是 joint-space reference；后 40 个是 task-space reference。Model A 运行全部 70 个，Direct IK 仅运行 40 个 task-space case。

```bash
python scripts/robustness/generate_benchmark_references.py \
  --output_dir outputs/robustness/references \
  --delay "$ROBUST_DELAY" --seed 20260722

python scripts/robustness/build_benchmark_manifest.py \
  --reference_dir outputs/robustness/references \
  --output_path outputs/robustness/benchmark.json \
  --delay "$ROBUST_DELAY" --seed 20260722
```

manifest 不可覆盖；需要改变 reference、delay 或 CEM 配置时，请使用新的输出目录和 manifest 文件名。

建议立即检查 manifest：

```bash
python -c "import json; d=json.load(open('outputs/robustness/benchmark.json')); print(d['kind'], len(d['cases'])); print(set(c['run_args']['multirate_mode'] for c in d['cases']))"
```

正常输出应包含 `model_a_robustness`、`70` 与 `threaded_asap`。

## 4. 冒烟测试：先确认 threaded worker、保存与结果目录

先运行两个 task-space case。使用 level 0 可先验证标称流程；若这一步失败，请不要直接运行完整 70-case 条件。

```bash
python scripts/robustness/evaluate_model_a.py \
  --manifest outputs/robustness/benchmark.json \
  --checkpoint "$MODEL_A/best_model.pt" \
  --normalizer "$MODEL_A/normalizer.pt" --model_type gru --device "$DEVICE" \
  --case_ids circle_00,figure8_00 \
  --save_dir outputs/robustness/runs/nominal_smoke
```

成功后确认：

```bash
ls outputs/robustness/runs/nominal_smoke
column -s, -t < outputs/robustness/runs/nominal_smoke/model_a_robustness_summary.csv | less -S
```

其中每个 case 应有 `ModelA_MPC` 和 `DirectIK` 两个目录。`ModelA_MPC/rollout.npz` 中的 `multirate_mode` 应为 `threaded_asap`。

## 5. 运行单一扰动等级

payload level 3（4 kg）的完整运行：

```bash
python scripts/robustness/evaluate_model_a.py \
  --manifest outputs/robustness/benchmark.json \
  --checkpoint "$MODEL_A/best_model.pt" \
  --normalizer "$MODEL_A/normalizer.pt" --model_type gru --device "$DEVICE" \
  --payload_level 3 \
  --save_dir outputs/robustness/runs/payload_l3
```

替换成以下任意参数可测试其他单一类别：

```text
--actuator_gain_level 3
--force_pulse_level 3
--observation_noise_level 3
```

## 6. Sweep 与组合扰动

payload 的 1–6 sweep：

```bash
for LEVEL in 1 2 3 4 5 6; do
  python scripts/robustness/evaluate_model_a.py \
    --manifest outputs/robustness/benchmark.json \
    --checkpoint "$MODEL_A/best_model.pt" \
    --normalizer "$MODEL_A/normalizer.pt" --model_type gru --device "$DEVICE" \
    --payload_level "$LEVEL" \
    --save_dir "outputs/robustness/runs/payload_l${LEVEL}" --resume
done
```

执行器增益、外力和观测噪声 sweep 使用相同命令，只替换等级参数和输出目录。组合条件示例：

```bash
python scripts/robustness/evaluate_model_a.py \
  --manifest outputs/robustness/benchmark.json \
  --checkpoint "$MODEL_A/best_model.pt" \
  --normalizer "$MODEL_A/normalizer.pt" --model_type gru --device "$DEVICE" \
  --payload_level 3 --actuator_gain_level 3 \
  --force_pulse_level 3 --observation_noise_level 3 \
  --save_dir outputs/robustness/runs/combined_l3
```

建议主文报告单因素实验；组合扰动可作为补充材料。

## 7. 输出、恢复与结果解读

每个目录会写入：

```text
model_a_robustness_summary.csv
paired_bootstrap.json
<case>/ModelA_MPC/rollout.npz
<case>/DirectIK/rollout.npz
```

汇总包括 tracking RMSE、failure rate、planner late-drop、控制周期、q/dq 观测误差、外力后的峰值/积分误差/恢复时间、residual 与 feedback 饱和率及命令平滑性。`paired_bootstrap.json` 只使用 Model A 与 Direct IK 都存在的 40 个 task-space case；不能将它解释为 joint-space case 的 Direct IK 比较。

`--resume` 仅在 checkpoint、normalizer、reference、扰动等级、plant XML、控制配置完全一致时复用 rollout；任一项变动都会拒绝复用。

中断完整条件后，使用原命令加 `--resume` 即可继续。例如：

```bash
python scripts/robustness/evaluate_model_a.py \
  --manifest outputs/robustness/benchmark.json \
  --checkpoint "$MODEL_A/best_model.pt" \
  --normalizer "$MODEL_A/normalizer.pt" --model_type gru --device "$DEVICE" \
  --payload_level 3 \
  --save_dir outputs/robustness/runs/payload_l3 --resume
```

不要在同一个 `--save_dir` 中改变等级、checkpoint、delay 或 reference；评测器会因 fingerprint 不匹配而拒绝复用。这是为了防止不同实验被误写进同一张结果表。

重点查看的列：

| 指标 | 如何解释 |
| --- | --- |
| `failure_rate` | 越低越好；首先确认扰动下没有安全失败。 |
| `tcp_position_rmse_m` | task-space case 的 TCP 跟踪误差，越低越好。 |
| `joint_position_rmse_rad` | 全部 case 可用的关节跟踪误差。 |
| `planner_late_drop_rate` | threaded planner 过晚发布、packet 被丢弃的比例；它反映实时预算，不应归因于 payload dynamics 本身。 |
| `force_peak_tracking_error` / `force_recovery_time_s` | 只在外力条件有效，分别表示扰动后的最大误差和恢复所需时间。 |
| `observation_q_rmse_rad` / `observation_dq_rmse_rad_s` | 实际施加到控制器观测上的噪声规模，用于审计。 |
| `residual_saturation_rate` / `feedback_saturation_rate` | 接近边界的频率；高值说明控制器正在耗尽补偿余量。 |

## 8. 常见问题

### `torch.cuda.is_available()` 为 `False`

`threaded_asap` 会拒绝运行，这是预期保护。确认进入的是 `pendulum-rl`，执行 `nvidia-smi`，再检查 PyTorch 是否安装了匹配 CUDA 的版本。不要把正式 threaded 实验改为 CPU。

### `FileExistsError: Refusing to overwrite immutable robustness manifest`

已有 manifest 已被冻结。若只想继续实验，直接复用它；若确实要更换 delay/reference，则使用新目录，例如 `outputs/robustness_v2/references` 和 `outputs/robustness_v2/benchmark.json`。

### `RuntimeError: main thread is not in main loop`

这是旧版 Matplotlib/Tk 后端在退出时清理 GUI 对象的警告，不是 MPC 失败。当前代码保存图时已强制使用无 GUI 的 `Agg` 后端；更新代码后不应再出现。若本次输出目录已经有 `model_a_robustness_summary.csv`，结果通常已保存；之后使用 `--resume` 继续。

### `Resume fingerprint mismatch`

说明当前命令与该输出目录中已完成 rollout 的 checkpoint、扰动等级、reference、plant XML 或控制配置不一致。保留旧结果，换一个新的 `--save_dir`；不要删除或覆盖已有实验来绕过检查。

### threaded 运行时没有窗口或不能 `--visualize`

这是正常的。`threaded_asap` 是无 GUI 的实时控制路径，明确不支持 `--visualize`；检查 `run_summary.json`、`rollout.csv`、PNG 图和汇总 CSV，而不是等待 MuJoCo viewer。
