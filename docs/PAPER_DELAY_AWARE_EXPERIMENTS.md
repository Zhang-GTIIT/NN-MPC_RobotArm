# Delay-Aware MPC 论文实验操作手册

本文档对应 ROBIO 论文的五控制器主比较、因果消融、delay sweep、Preview IK、GRU 验证和 oracle-dynamics upper bound。所有新产物均写入：

```text
outputs/paper_delay_aware/
```

Model C 不进入本文主线。Payload、观测噪声、执行器失配和外力实验继续使用 [`MODEL_A_ROBUSTNESS.md`](MODEL_A_ROBUSTNESS.md)，不复制进本工作流。

## 1. 环境和冻结模型检查

```bash
conda activate pendulum-rl
cd ~/Data/RL_Projects/NN-MPC_RobotArm

export PAPER_OUT=outputs/paper_delay_aware
export PAPER_CKPT=outputs/checkpoints/gru_20260720_202923/best_model.pt
export PAPER_NORM=outputs/checkpoints/gru_20260720_202923/normalizer.pt
mkdir -p "$PAPER_OUT/logs"

test -f "$PAPER_CKPT"
test -f "$PAPER_NORM"
nvidia-smi
git status --short
```

论文配置显式使用 GRU、history 16、H20、128 candidates、2 CEM iterations 和每5个逻辑 tick 重规划一次。不要依赖根 CLI 的 Transformer 默认值。正式 manifest 必须在 clean worktree 上生成，并记录 commit、checkpoint/reference hash、Python、PyTorch、CUDA、MuJoCo、GPU 和 driver。

## 2. 控制协议

底层只提供五个不重复的 `--delay_protocol`：

| ID | Activation state/reference | 执行方式 | Feedback |
|---|---|---|---|
| `full` | 均对齐到 `k+D` | 当前 IK nominal + age-aligned residual | 开 |
| `naive_delayed` | 均停留在 launch time | 激活后从 age 0 重放 planner-projected absolute commands | 关 |
| `no_future_alignment` | 均停留在 launch time | 当前 nominal + residual | 开 |
| `no_reanchor` | 均对齐到 `k+D` | 重放 planner-projected absolute commands | 开 |
| `no_feedback` | 均对齐到 `k+D` | 当前 nominal + residual | 关 |

Ideal zero-delay 定义为 `full + D=0`。它保持相同的逻辑20 Hz重规划频率和 CEM budget，不代表100 Hz实时 CEM。

virtual runner 是 deterministic fixed-delay simulation：wall time 只记录为诊断，绝不决定 virtual packet 是否激活。真实 late drop 只由 `threaded_asap` 报告。

`naive_delayed` 和 `no_reanchor` 的 absolute sequence 已经在 CEM candidate rollout 中依据预测锚点做过完整运动学投影；执行时仅做 joint-limit clip，不再按真实上一拍命令做第二次速度/加速度投影。否则第二次投影本身就是 execution-time reconciliation，会让 `no_reanchor` 与 Full 退化为同一控制器。相应的实际速度/加速度违例会原样记录并进入结果表。

## 3. 生成独立 calibration reference

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  generate-calibration-reference
```

生成的 joint chirp 只用于平台延迟标定，不进入 circle、figure-eight、fast ellipse 或 rounded-square 正式测试。

## 4. 用真实 planner E2E 标定 D

标定期间不要同时运行其他 GPU 工作负载：

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  calibrate-delay \
  --samples 500 \
  --provisional-delay 10 \
  --guard-ms 5
```

标定读取 `planner_end_to_end_latency_s`：

```text
snapshot launch -> future forecast/history -> CEM -> packet construction -> publication
```

冻结公式：

```text
D_cal = ceil((P95(E2E) + 5 ms) / 10 ms)
```

查看结果：

```bash
cat "$PAPER_OUT/calibration/delay.json"
```

所有 virtual 主比较、消融和 threaded full 共用该 D；禁止为某个消融单独重新标定。

## 5. 生成正式 immutable references

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  generate-references
```

生成：

- `circle`：3圈平滑圆；
- `figure8`：3圈 Gerono figure-eight；
- `fast_ellipse`：3圈高速椭圆；
- `rounded_square`：直线与圆弧连续相切并按弧长重采样；
- `preview_calibration`：不进入正式测试的独立慢速椭圆。

旧 `square` 实现及历史 robustness references 不会被修改。所有 reference 都带 horizon、最大 sweep delay 和 preview padding，并将 SHA-256 写入 reference manifest。

## 6. 标定统一 Preview IK

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  calibrate-preview \
  --preview-values 0,1,2,3,4
```

选择规则为 calibration TCP RMSE 最小；完全相同时选择更小 preview。选出的同一个 preview 用于全部四条正式轨迹，禁止在测试轨迹上重新选择。

## 7. GRU 冻结模型验证

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  validate-model \
  --num-rollouts 20 \
  --rollout-len 200
```

输出包括：

- one-step prediction with ground-truth history；
- 1/5/10/20-step open-loop q/dq RMSE；
- 每个关节的 RMSE、NMSE、R² 和 amplitude ratio；
- divergence rate；
- 冻结 evaluation action/state rollouts 及 hash。

这些 rollout 在 checkpoint 冻结后新生成，不参与训练。不要在论文中称为 sequence-decoder teacher forcing。

## 8. 构建唯一 paper manifest

完成上述步骤并提交实现后，在 clean worktree 上执行：

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  build-manifest \
  --profile paper
```

正式 manifest 路径：

```text
outputs/paper_delay_aware/manifests/paper.json
```

文件不可覆盖。配置或 commit 改变时必须使用新的输出根目录，例如 `outputs/paper_delay_aware_rerun1/`。

## 9. Smoke test

Smoke 使用短轨迹、H3、8 candidates 和1次 CEM iteration，覆盖 D=0、五个 virtual protocol、Direct IK 和 threaded full：

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  smoke 2>&1 | tee "$PAPER_OUT/logs/smoke.log"
```

Smoke 只验证数据流和输出字段，不能写入论文结果。
如果当前进程的 PyTorch 看不到 CUDA，virtual 和 Direct IK smoke 会改用 CPU，threaded smoke 会明确记录为 `skipped_cuda_unavailable`；正式实验机必须在 CUDA 可见时重新运行，状态保存在 `smoke/environment_status.json`。

回归测试：

```bash
python -m unittest \
  mpc.test_paper_experiments \
  mpc.test_residual_mpc \
  mpc.test_asap_planner_worker \
  mpc.test_asap_timing \
  mpc.test_logging_threaded \
  mpc.model_c.test_oracle \
  mpc.test_robustness \
  mpc.test_model_a_robustness_evaluate \
  -v 2>&1 | tee "$PAPER_OUT/logs/unit_tests.log"
```

## 10. 正式 P0 主实验

### 五控制器主比较：84 cases

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite main --resume
```

矩阵：

```text
Direct IK                         4 trajectories × 1
Ideal full, D=0                  4 × 5 paired CEM seeds
Naive delayed, D=D_cal           4 × 5
Full virtual, D=D_cal            4 × 5
Full threaded, D=D_cal           4 × 5
```

### 核心消融：80个表格 case，只有60个新增 rollout

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite ablation --resume
```

`FullVirtual` 的20个 case 与 main 完全相同，通过 fingerprint cache 复用；只新增 NoFutureAlignment、NoReanchor、NoFeedback 各20次。

## 11. 正式 P1 实验

### Delay sweep：60 cases

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite delay_sweep --resume
```

```text
protocols: full, naive_delayed
trajectories: circle, fast_ellipse
D: 0, 2, 4, 6, 8
paired CEM seeds: 0, 1, 2
```

与 main 完全相同的 D=0 或 D=`D_cal` case 会自动复用。

### Preview IK

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite preview --resume
```

Preview IK 单独成表，不扩充论文五控制器主表。

## 12. P2 Oracle upper bound

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  run --suite oracle --resume
```

该组为 learned/MuJoCo-oracle × circle/fast-ellipse × 3 seeds，共12次。它是 oracle-dynamics upper bound，不是保持控制动作不变的纯模型误差消融。

## 13. 汇总与恢复运行

单独重建全部 summary：

```bash
python -m scripts.paper_experiments.workflow \
  --output-root "$PAPER_OUT" \
  summarize --suite all --bootstrap-samples 10000
```

`runs/cache/<fingerprint>/` 是 canonical rollout；`runs/indexes/` 将论文方法名称映射到 cache。`--resume` 只接受精确 fingerprint，缺文件或 hash 不一致会立即失败。

主要结果位于：

```text
summaries/main.csv
summaries/main_aggregate.csv
summaries/main_paired_bootstrap.json
summaries/latency_recovery.json
summaries/ablation.csv
summaries/ablation_aggregate.csv
summaries/ablation_paired_bootstrap.json
summaries/delay_sweep.csv
summaries/delay_sweep_aggregate.csv
summaries/preview.csv
summaries/oracle.csv
```

统计单位是 trajectory-seed case。五个 seed 只表示 CEM sampling stochasticity，不表示五个随机机器人环境。10 ms timestep 不能作为独立 bootstrap 样本。
`failure_rate` 是每个 case 是否出现过 planner failure 的二值量；瞬时 fallback 频率另存为 `planner_failure_step_rate`。

Latency recovery 定义为：

```text
(E_naive - E_full) / (E_naive - E_ideal)
```

仅当分母大于 `1e-6 m` 时报告，不截断到 `[0,1]`。

## 14. 指标定义

| 字段 | 定义 |
|---|---|
| `planner_requested_residual` | CEM normalized action 经 residual bound 后、候选运动学投影前的 residual |
| `buffered_residual` | CEM 真正评估并写入 packet 的 projected residual |
| `feedback_raw` | feedback bound 前的状态反馈 |
| `feedback_correction` | feedback bound 后的反馈 |
| `requested_correction` | 当前 tick 请求的 MPC + feedback correction |
| `executed_residual` | 实际 actuator command 相对当前 IK nominal 的 residual |
| `projection_discrepancy` | `requested_correction - executed_residual` |
| `projection_active` | discrepancy 任一关节大于 `1e-6 rad` |

virtual 的固定逻辑 D 与 threaded 的真实 snapshot-to-publication E2E 必须分开报告。只有 threaded 结果可以用于 soft-real-time、late packet、planner rate、period、jitter 和 deadline-miss 结论。
