# NN-MPC_RobotArm

面向 ABB IRB 2400 的 MuJoCo 学习动力学与 CEM-MPC 项目。动力学模型学习位置执行器的闭环状态转移；默认控制器是 **residual MPC**：以可执行的 IK nominal command 为基线，只搜索有界补偿，而不是重新生成一条无锚定的绝对命令轨迹。

## 快速开始

在仓库根目录运行：

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm
conda run -n pendulum-rl python scripts/run_cem_mpc.py --help
```

依赖与动力学数据采集、训练和开环评估说明见 [dynamics_modeling/README.md](dynamics_modeling/README.md)。

冻结 Model A 在 payload、执行器失配、外力和观测噪声下的 threaded-asap 鲁棒性评测见 [docs/MODEL_A_ROBUSTNESS.md](docs/MODEL_A_ROBUSTNESS.md)。

## 当前 MPC 方法

MuJoCo 和 learned dynamics 的动作语义始终是绝对 position-actuator target：

```text
state   = [q(6), dq(6)]
action  = q_ref(6)     # absolute actuator reference, rad
target  = delta_dq(6)
```

任务空间参考先经连续 DLS IK 生成并验证 `q_des`、`dq_des`。在时刻 `t`，默认 residual MPC 的流程是：

```text
q_des[t+1:t+H]
    -> project to executable nominal q_nom
    -> CEM samples normalized residual r / r_max in [-1, 1]
    -> q_ref = project(q_nom + r)
    -> learned dynamics rollout
    -> residual joint-space cost
    -> execute the first q_ref in MuJoCo
```

`q_nom` 已满足命令速度、加速度和关节边界约束。零 residual 是每拍必有的 direct baseline；因此 `r=0` 永远代表“执行可行 nominal”。默认 `--cem_execute lowest_cost` 会比较 baseline、CEM best sample 和最终 CEM mean，执行预测 cost 最低的候选。

`--mpc_policy legacy_acceleration` 仍可复现实验中的无锚定命令加速度动作空间，但不是默认方法。

## 推荐运行

默认配置是 CUDA 上的 **threaded ASAP residual MPC**：预测 horizon 20、6 步计划激活延迟补偿、128 candidates、2 次 CEM iteration、batch 128。主线程以 100 Hz 持续重锚定 residual 并施加状态反馈；后台 CUDA worker 在每次求解完成后使用最新 snapshot 尽快启动下一次规划，因此实际 planner update rate 由 GPU 负载决定，而不是固定 20 Hz。

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt \
  --model_type transformer \
  --reference_mode multi_joint_sine \
  --episode_len 200 \
  --horizon 20 \
  --multirate_mode threaded_asap \
  --anticipation_delay_steps 6 \
  --num_samples 128 \
  --cem_iters 2 \
  --rollout_batch_size 128 \
  --mpc_policy residual \
  --cem_execute lowest_cost \
  --save_dir outputs/mpc/joint_sine_residual
```

`threaded_asap` 需要 `--device cuda`（默认值）。主实验除 tracking 指标外，应同时报告 planner update rate、control deadline miss、late packet drop 和 Direct-IK fallback。若需要可重复的逻辑延迟消融，可显式传入 `--multirate_mode virtual_asap`；该模式不是默认部署控制器。

参数含义：

- `episode_len`：非 task 模式的执行控制步数；task 模式改用 reference 的 `execution_steps`。
- `horizon`：每拍预测与优化的未来控制步数。
- `num_samples`：每个 CEM iteration 的候选序列数量；residual 模式含 forced baseline 和 mean 候选。
- `cem_iters`：每个控制步的 CEM 更新次数。
- `rollout_batch_size`：一次模型前向计算的候选上限；可大于 `num_samples`，但不会增加候选数。
- `cem_execute`：`mean` 执行最终分布均值，`best` 执行最低 cost sample，`lowest_cost` 比较 baseline、best、mean 后执行最低者；推荐后者。

输出目录包含 `rollout.npz`、`rollout.csv`、跟踪/控制图、`run_summary.json`。结束报告中的 `recovery triggers` 是 recovery 的总触发次数；`recovery active steps` 是 nominal 回退实际执行的步数。

## 任务空间参考与 IK direct

生成并验证三圈圆形参考：

```bash
conda run -n pendulum-rl python scripts/generate_task_reference.py \
  --shape circle --repeat_count 3 \
  --save_dir outputs/references/circle_3laps

conda run -n pendulum-rl python scripts/validate_ik.py \
  --reference_file outputs/references/circle_3laps/reference.npz
```

运行 residual MPC：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --checkpoint dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt \
  --model_type transformer \
  --reference_mode task \
  --reference_file outputs/references/circle_3laps/reference.npz \
  --multirate_mode threaded_asap \
  --mpc_policy residual --cem_execute lowest_cost \
  --save_dir outputs/mpc/task_circle_residual
```

IK direct baseline 不加载 learned dynamics 或 CEM，直接发送后继 `q_des`：

```bash
conda run -n pendulum-rl python scripts/run_cem_mpc.py \
  --controller_mode ik_direct \
  --reference_mode task \
  --reference_file outputs/references/circle_3laps/reference.npz \
  --save_dir outputs/mpc/task_circle_ik_direct
```

对比两次运行的 `task_tracking_summary.json` 和 `run_summary.json`。task 模式忽略 `episode_len`，使用参考文件的 `execution_steps`。本地图形桌面可添加 `--visualize`，关闭窗口会停止并保存部分 rollout。

## 安全与 recovery

默认 residual bound 为 `[0.12, 0.10, 0.12, 0.15, 0.15, 0.20] rad`。命令速度/加速度上限是 MuJoCo 规划上限而非 ABB 硬件额定值；达到这些上限只记录诊断，不会单独触发 recovery。

Residual MPC 在以下情况下回退到 `q_nom` 并 reset CEM warm start：planner failure 立即触发；跟踪误差持续恶化或 residual 持续接近 bound 时按 `recovery_consecutive_steps` 触发。默认持续步数为 3，冷却期为 5 步。

## 目录与测试

```text
dynamics_modeling/  ABB XML、数据采集、训练与开环评估
mpc/                CEM、nominal projection、rollout、cost、recovery、IK
scripts/            闭环 MPC 与参考生成命令
docs/               当前规范、历史设计和实验材料
outputs/            生成的模型、参考和运行结果
```

快速验证 residual MPC 单元测试：

```bash
conda run -n pendulum-rl python -m unittest mpc/tests/test_residual_mpc.py -v
```

完整项目测试入口见 [dynamics_modeling/README.md](dynamics_modeling/README.md)。

## 相关文档

- [Cost function](docs/CostFunction.md)
- [MPC 伪代码](docs/mpc-pseudocode.md)
- [运行命令](docs/run_command.md)
- [Delay-Aware MPC 论文实验操作手册](docs/PAPER_DELAY_AWARE_EXPERIMENTS.md)
- [项目结构](docs/PROJECT_STRUCTURE.md)
- [完成状态（历史快照）](docs/PROJECT_COMPLETION_STATUS.md)
