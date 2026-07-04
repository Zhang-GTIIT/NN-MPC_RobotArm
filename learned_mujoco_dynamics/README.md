# Learned MuJoCo Dynamics 子模块使用说明

这个目录是整个 `MPC_RL_RobotArm` 仓库中的 learned dynamics 子模块，用于在 MuJoCo 中采集机械臂运动数据，并用 PyTorch 训练神经网络动力学模型。顶层仓库还包含 CEM-MPC 闭环控制代码；完整目录说明见仓库根目录的 `PROJECT_STRUCTURE.md`。

默认机械臂模型是项目根目录下的 `ABB_IRB2400.xml`。如果你要换成别的 MuJoCo XML/MJCF 模型，可以在命令中传入 `--model_xml path/to/robot.xml`。

## 1. 进入项目目录

本 README 第 1-16 节中的 dynamics 数据采集、训练、open-loop 评估命令都应该从 `learned_mujoco_dynamics/` 目录运行：

```bash
cd /home/xinlei/Data/RL_Projects/MPC_RL_RobotArm/learned_mujoco_dynamics
```

项目目录中应该能看到：

```text
ABB_IRB2400.xml
abb_irb2400_assets/
scripts/
learned_dynamics/
outputs/
```

其中 `ABB_IRB2400.xml` 会引用 `abb_irb2400_assets/` 里的 STL 网格文件，所以不要把 XML 单独移动到别的目录。当前 STL 只用于显示，质量和惯量由 XML 中显式 `<inertial>` 参数给出。

## 2. Python 环境

推荐使用你现有的 conda 环境 `pendulum-rl`：

```bash
conda activate pendulum-rl
```

如果不想激活环境，也可以在每条命令前加：

```bash
conda run -n pendulum-rl
```

例如：

```bash
conda run -n pendulum-rl python scripts/collect_data.py --num_episodes 1 --episode_len 3
```

如果环境里缺少依赖，再安装：

```bash
pip install -r requirements.txt
```

检查 MuJoCo 和 PyTorch 是否可用：

```bash
python -c "import mujoco, torch; print('mujoco', mujoco.__version__); print('torch', torch.__version__)"
```

## 3. 快速测试 ABB IRB 2400 模型

不传 `--model_xml` 时，脚本默认使用：

```text
ABB_IRB2400.xml
```

先跑一个很小的数据采集测试：

```bash
python scripts/collect_data.py \
  --n_joints 6 \
  --num_episodes 1 \
  --episode_len 3 \
  --num_envs 1 \
  --save_path outputs/datasets/abb_smoke.npz
```

成功时会看到类似输出：

```text
Saved dataset to outputs/datasets/abb_smoke.npz with states=(3, 12), actions=(3, 6), next_states=(3, 12)
```

这里：

- `states=(3, 12)` 表示 3 条样本，每条状态为 6 个关节角度加 6 个关节速度。
- `actions=(3, 6)` 表示每条动作是 6 维目标关节角 `q_ref`，单位为 rad。
- `next_states=(3, 12)` 表示下一时刻状态。

## 4. 可视化结构化闭环 q_ref 运动

如果当前机器支持图形界面，可以运行：

```bash
 python scripts/rollout_visualize.py \
  --n_joints 6 \
  --episode_len 10000 \
  --action_std 0.9
```

这个脚本会打开 MuJoCo viewer，先安全 reset 和 settle，然后按 hold、小阶跃、平滑随机 waypoint、正弦四段生成目标关节角 `q_ref`。可视化只使用单环境，不会启动多进程。

如果你在无显示器或远程终端环境中运行，viewer 可能无法打开。数据采集和训练不需要 viewer。

## 5. 单环境采集训练数据

先用单环境版本确认 XML、actuator、状态维度都正常：

```bash
python scripts/collect_data.py \
  --n_joints 6 \
  --num_episodes 20 \
  --episode_len 200 \
  --num_envs 1 \
  --action_std 0.5 \
  --seed 0 \
  --save_path outputs/datasets/irb2400_parallel_data.npz
```

单环境采集是最稳的调试入口。建议先让这条命令跑通，再使用多环境并行采集。

如果你要继续往已有数据集里追加新采集的数据，使用同一个 `--save_path` 并加上 `--append`：

```bash
python scripts/collect_data.py \
  --n_joints 6 \
  --num_episodes 5000 \
  --episode_len 600 \
  --num_envs 32 \
  --action_std 0.2 \
  --seed 1 \
  --save_path outputs/datasets/irb2400_parallel_data.npz \
  --append
```

这样会先读取 `outputs/datasets/irb2400_parallel_data.npz` 中已有的 `states/actions/next_states`，再把本次新采集的数据拼到后面保存回同一个文件。

如果你已经有多个 `.npz` 数据文件，也可以单独合并：

```bash
python scripts/merge_datasets.py \
  --inputs outputs/datasets/irb2400_parallel_data.npz outputs/datasets/irb2400_parallel_data_v2.npz \
  --output outputs/datasets/irb2400_parallel_data.npz
```

合并后的文件仍然包含同样的三个数组：

```text
states
actions
next_states
```

## 6. 多环境并行采集数据

单环境确认正常后，可以用多进程加速数据生成：

```bash
python scripts/collect_data.py \
  --n_joints 6 \
  --num_episodes 200 \
  --episode_len 300 \
  --num_envs 8 \
  --action_std 0.5 \
  --seed 0 \
  --save_path outputs/datasets/irb2400_parallel_data.npz
```

建议：

- `num_envs` 设为 CPU 核心数的一半或接近核心数，例如 4、8、12。
- 多环境采集只用于生成数据，不用于可视化。
- 每个 worker 会独立加载 MuJoCo model 和 data，不共享 `MjData`。

## 7. 使用其他机器人模型

如果要测试 UR5e、UR10e、Franka Panda、KUKA iiwa14 或其他 MJCF 模型，传入 `--model_xml`：

```bash
python scripts/collect_data.py \
  --model_xml path/to/robot.xml \
  --n_joints 6 \
  --num_episodes 20 \
  --episode_len 200 \
  --num_envs 1 \
  --save_path outputs/datasets/other_robot_data.npz
```

注意：

- 相对路径会按项目根目录 `learned_mujoco_dynamics/` 解析。
- XML 中 actuator 数量必须至少等于 `--n_joints`。
- 当前 `ABB_IRB2400.xml` 使用 MuJoCo `<position>` actuator，动作会写入 `data.ctrl[:n_joints]`，语义是每个关节的目标角 `q_ref`，单位为 rad。
- actuator 的目标角范围来自 XML 的 `ctrlrange`，并与 6 个 joint 的 `range` 保持一致。
- 默认 MuJoCo 环境会在每个仿真 substep 前用当前姿态计算 `qfrc_bias`，并写入 `data.qfrc_applied[:n_joints]` 做重力补偿，所以总输入近似为位置伺服力矩加重力前馈。
- 当前 `abb_irb2400_assets/*_visual.stl` 来自 ROS-Industrial ABB IRB2400 per-link visual DAE mesh，并通过 `scripts/convert_collada_to_stl.py` 转成 MuJoCo 可加载的 binary STL。来源仓库是 `https://github.com/ros-industrial/abb`。
- XML 中的 `<geom>` 默认 `mass="0"`、`contype="0"`、`conaffinity="0"`，所以 STL mesh 只用于显示，不再参与质量、惯量或碰撞计算。
- 机器人总质量按 ABB IRB 2400 规格近似设置为 380 kg；`link_5/link_6` 显式设置为 12 kg / 8 kg，避免粗糙 STL 自动惯性导致腕部 `dq4` 尖峰。
- 旧的粗糙 STL 仍保留在 `abb_irb2400_assets/` 中，但当前 XML 引用的是 `base_link_visual.stl`、`link_1_visual.stl` 到 `link_6_visual.stl` 和两个 lever visual STL。
- 旧的 velocity/motor/torque 数据集和 checkpoint 不能和当前 position-actuator XML 混用；改 actuator 或 action 采样语义后需要重新采集数据。
- `--action_std` 是归一化关节坐标中的随机目标标准差，不是 rad。每个关节先按自身 `ctrlrange` 映射到 `[-1, 1]`，采样后再映射回实际目标角 `q_ref`。
- 例如 `--action_std 0.5` 表示所有关节都使用 normalized std 0.5；也支持 6 个逗号分隔的 normalized std。
- 数据采集的是闭环位置伺服系统：`x_next = f(x, q_ref)`，不是 velocity command 数据，也不是直接 torque command 数据。
- 每个 episode 会先在安全工作空间 reset，随后用 `q_ref = q` settle 默认 50 个 control step，再开始记录训练样本。
- `q_ref` 由 hold、小阶跃、平滑随机 waypoint 和正弦轨迹混合生成，并经过一阶滤波，避免每步随机跳目标。
- 环境会在每次 `step()` 后检查真实 `qpos` 是否仍在 XML joint limit 内；越界会直接报错，避免保存坏数据。
- 如果 XML 的 actuator 数量不足，代码会直接报清晰错误。

## 8. 训练 MLP 动力学模型

MLP 使用单步输入：

```text
[q_t, dq_t, q_ref_t] -> delta_dq_t
```

训练命令：

```bash
python scripts/train_dynamics.py \
  --data_path outputs/datasets/irb2400_parallel_data.npz \
  --model_type mlp \
  --epochs 100 \
  --batch_size 1024 \
  --lr 0.001 \
  --save_dir outputs/checkpoints_mlp \
  --num_workers 4
  --pin_memory \
  --amp \
```

训练输出会保存到类似目录：

```text
outputs/checkpoints/mlp_YYYYMMDD_HHMMSS/
```

其中包含：

```text
best_model.pt
latest_model.pt
normalizer.pt
config.yaml
```

`best_model.pt` 是 validation loss 最低的 checkpoint，适合评估；`latest_model.pt` 是最后一个 epoch 的 checkpoint，适合中断后继续训练。

## 9. 训练 GRU 动力学模型

GRU 使用历史序列输入：

```text
过去 history_len 步的 [q, dq, q_ref] -> 当前 delta_dq
```

训练命令：

```bash
python scripts/train_dynamics.py \
  --data_path outputs/datasets/irb2400_parallel_data.npz \
  --model_type gru \
  --history_len 8 \
  --epochs 100 \
  --batch_size 512 \
  --lr 0.001 \
  --save_dir outputs/checkpoints
```

## 10. 训练 Transformer 动力学模型

Transformer 同样使用历史序列输入：

```bash
  conda run --no-capture-output -n pendulum-rl python scripts/train_dynamics.py \
    --data_path outputs/datasets/irb2400_parallel_data_transformer_18m.npz \
    --model_type transformer \
    --history_len 16 \
    --epochs 40 \
    --batch_size 128 \
    --lr 3e-4 \
    --num_workers 0 \
    --pin_memory \
    --amp \
    --save_dir outputs/checkpoints_transformer

```

推荐给 Transformer 单独采集带 episode 边界的数据，避免历史窗口跨越 reset：

```bash
python scripts/collect_transformer_data.py \
  --num_episodes 10000 \
  --episode_len 600 \
  --num_envs 10 \
  --action_std 0.5 \
  --seed 0 \
  --history_len 16 \
  --save_path outputs/datasets/irb2400_parallel_data_transformer.npz
```

该文件会保存：

```text
states
actions
next_states
episode_ids
q_ref
delta_q_ref
tau_actuator
tau_gravity
tau_total
action_std_normalized
settle_steps
motion_mode_ids
termination_reasons
```

其中 `actions` 和 `q_ref` 相同，都是写入 `data.ctrl[:n_joints]` 的目标关节角。`delta_q_ref` 目前不作为第一版模型输入，但会保存给后续 MPC 或参考轨迹建模使用。`action_std_normalized` 记录每条样本来自哪个归一化采样强度，`settle_steps` 记录 episode 开始前稳定了多少 control step；如果 append 到旧数据集，旧样本这两个字段会填 `-1` 表示未知。`episode_ids` 用于保证 GRU/Transformer 的历史序列只来自同一个 episode。推荐从 `history_len=16` 开始训练；当前 MuJoCo `timestep=0.002` 且 `frame_skip=5`，每条样本间隔约 `0.01s`，16 步约覆盖 `0.16s` 历史。

采集后先跑覆盖诊断：

```bash
python scripts/diagnose_dynamics_data.py \
  --data_path outputs/datasets/irb2400_parallel_data_transformer.npz \
  --coverage_dir outputs/figures/irb2400_parallel_data_transformer_diagnose
```

重点检查 `q_norm_hist.png`、`q_ref_norm_hist.png`、`q_ref_tracking_first2.png`、`tau_total_hist.png`、`action_std_normalized_hist.png`、`termination_summary.csv` 和 `settle_steps_summary.csv`。诊断脚本会优先从 `--model_xml` 对应 XML 读取真实 joint limit 后再计算 normalized joint value。

推荐训练命令：

```bash
python scripts/train_dynamics.py \
  --data_path outputs/datasets/irb2400_parallel_data_transformer.npz \
  --model_type transformer \
  --history_len 16 \
  --epochs 40 \
  --batch_size 512 \
  --lr 3e-4 \
  --num_workers 4 \
  --pin_memory \
  --amp \
  --save_dir outputs/checkpoints_transformer
```

## 11. 使用 GPU、DataLoader worker 和 AMP

如果 `pendulum-rl` 环境中的 PyTorch 能看到 GPU，可以开启更快的训练配置：

```bash
python scripts/train_dynamics.py \
  --data_path outputs/datasets/irb2400_parallel_data.npz \
  --model_type transformer \
  --history_len 8 \
  --epochs 100 \
  --batch_size 512 \
  --lr 0.001 \
  --num_workers 4 \
  --pin_memory \
  --amp \
  --save_dir outputs/checkpoints
```

说明：

- `--num_workers` 加速 DataLoader 读数据。
- `--pin_memory` 在 GPU 训练时加速 CPU 到 GPU 的数据传输。
- `--amp` 开启自动混合精度。没有 CUDA GPU 时会自动不用 AMP。

## 12. 继续训练和从旧模型初始化

如果同一次训练中断了，用 `latest_model.pt` 继续。这里 `--epochs 200` 表示最终训练到第 200 个 epoch；如果 checkpoint 已经在 epoch 80，会继续跑 epoch 81 到 200。

```bash
  python scripts/train_dynamics.py \
    --data_path outputs/datasets/irb2400_parallel_data_transformer_manual_inertia_v2.npz \
    --model_type transformer \
    --history_len 16 \
    --target_mode delta_dq \
    --control_dt 0.01 \
    --epochs 40 \
    --batch_size 4096 \
    --lr 5e-5 \
    --init_from_checkpoint outputs/checkpoints_transformer/transformer_20260603_174800/best_model.pt \
    --rollout_loss_steps 5 \
    --rollout_loss_weight 0.1 \
    --save_dir outputs/checkpoints_transformer \
    --num_workers 8 \
    --pin_memory \
    --amp 

```

如果你新采集了更多数据，推荐在采集时直接用 `--append` 追加到原来的数据集，然后从旧模型权重初始化训练。这个模式会重新 fit normalizer，并重新创建 optimizer。

```bash
python scripts/train_dynamics.py \
  --data_path outputs/datasets/irb2400_parallel_data.npz \
  --model_type transformer \
  --history_len 8 \
  --epochs 100 \
  --init_from_checkpoint outputs/checkpoints/transformer_YYYYMMDD_HHMMSS/best_model.pt
```

如果是 mlp，把 --model_type transformer 改成 --model_type mlp，删掉
  --history_len 8，checkpoint 路径改成对应的 mlp_... 目录。

  继续训练：

```bash
  python scripts/train_dynamics.py \
    --data_path outputs/datasets/irb2400_parallel_data.npz \
    --model_type mlp \
    --epochs 100 \
    --num_workers 4 \
    --resume_checkpoint outputs/checkpoints_mlp/mlp_20260530_094540/latest_model.pt
```

  从旧 MLP 权重初始化新训练：

```bash
  python scripts/train_dynamics.py \
    --data_path outputs/datasets/irb2400_parallel_data.npz \
    --model_type mlp \
    --epochs 100 \
    --init_from_checkpoint outputs/checkpoints_mlp/mlp_YYYYMMDD_HHMMSS/best_model.pt
```

不要同时使用 `--resume_checkpoint` 和 `--init_from_checkpoint`。

## 13. 评估训练好的模型

评估脚本会在 MuJoCo 中生成真实 rollout，然后让 learned dynamics 做 open-loop prediction，并保存对比图。

以 mlp 为例：

```bash
python scripts/eval_dynamics.py \
  --checkpoint outputs/checkpoints_mlp/mlp_20260530_094540/best_model.pt \
  --normalizer outputs/checkpoints_mlp/mlp_20260530_094540/normalizer.pt \
  --model_type mlp \
  --n_joints 6 \
  --rollout_len 200 \
  --num_rollouts 3 \
  --save_dir outputs/figures/mlp
  # --history_len 8 \
```

Transformer 评估示例：

```bash
python scripts2/eval_dynamics.py \
  --checkpoint outputs/checkpoints_transformer_v2/transformer_20260604_094824/best_model.pt \
  --normalizer outputs/checkpoints_transformer_v2/transformer_20260604_094824/normalizer.pt \
  --model_type transformer \
  --n_joints 6 \
  --rollout_len 200 \
  --num_rollouts 10 \
  --save_dir outputs/figures/transformer_v2 \
  --warmup_steps 50 \
  --action_std 0.3 \
  --horizons 1,5,10,20,50,100,200 

python scripts2/eval_dynamics.py \
  --checkpoint outputs/checkpoints_transformer_v2/transformer_20260606_193925/best_model.pt \
  --normalizer outputs/checkpoints_transformer_v2/transformer_20260606_193925/normalizer.pt \
  --model_type transformer \
  --n_joints 6 \
  --history_len 16 \
  --rollout_len 200 \
  --num_rollouts 10 \
  --action_std 0.95 \
  --warmup_steps 50 \
  --horizons 1,5,10,20,50,100,200 \
  --teacher_forcing \
  --save_dir outputs/figures/transformer_20260606_193925

```

如果 checkpoint 里保存了 `history_len`，评估脚本会自动使用该值；需要手动覆盖时再显式传 `--history_len`。

MLP 评估示例：

```bash
python scripts/eval_dynamics.py \
  --checkpoint outputs/checkpoints/mlp_YYYYMMDD_HHMMSS/best_model.pt \
  --normalizer outputs/checkpoints/mlp_YYYYMMDD_HHMMSS/normalizer.pt \
  --model_type mlp \
  --n_joints 6 \
  --rollout_len 200 \
  --num_rollouts 3 \
  --save_dir outputs/figures
```

评估会保存三类图：

- 每个关节角度 `q` 的真实轨迹和预测轨迹。
- 每个关节速度 `dq` 的真实轨迹和预测轨迹。
- 状态预测误差随时间变化曲线。
- `collect_truth_rollout_with_torque` 路径还会保存 `total_tau`、`actuator_tau` 和 `gravity_tau` 的力矩曲线。

## 14. 数据格式

采集结果是 `.npz` 文件，包含：

```text
states
actions
next_states
```

Transformer 专用采集文件还会额外包含：

```text
episode_ids
```

shape 约定：

```text
states.shape      = [N, 2 * n_joints]
actions.shape     = [N, n_joints]
next_states.shape = [N, 2 * n_joints]
```

当前项目中：

```text
n_joints = 6
state_dim = 12
action_dim = 6
```

状态定义：

```text
state = concat(qpos[:n_joints], qvel[:n_joints])
action = q_ref[:n_joints]
```

模型训练目标：

```text
delta_dq = dq_next - dq
```

当前推荐只预测速度增量：

```bash
python scripts/train_dynamics.py \
  --data_path outputs/datasets/irb2400_qref_data.npz \
  --model_type mlp \
  --target_mode delta_dq \
  --control_dt 0.01 \
  --loss_type huber \
  --dq_extra_weights 1,1,1,1,2,2 \
  --save_dir outputs/checkpoints_qref
```

`delta_dq` 模式输出 6 维 `dq_next - dq`，再用半隐式积分恢复完整 next state：

```text
dq_next = dq + predicted_delta_dq
q_next  = q + dq_next * control_dt
```

## 数据诊断

检查 `dq4/dq5` 尖峰、动作滞后相关性和 qacc substep 语义：

```bash
python scripts/diagnose_dynamics_data.py \
  --data_path outputs/datasets/irb2400_qref_data.npz \
  --save_csv outputs/diagnostics/qref_delta_stats.csv \
  --lag_csv outputs/diagnostics/qref_lag_corr.csv \
  --qacc_rollout_steps 10 \
  --action_std 0.5
```

注意不要用最后一个 substep 的 `qacc * control_dt` 直接判断对齐；RK4 和 `frame_skip` 会让这个近似在腕部关节上非常误导。脚本会同时输出精确 substep `qvel` 累加误差和 post-step `qacc` Euler 近似误差。

## 15. 推荐运行顺序

第一次使用建议按这个顺序：

```bash
cd /home/xinlei/Data/RL_Projects/MPC_RL_RobotArm/learned_mujoco_dynamics
conda activate pendulum-rl

python scripts/collect_data.py --num_episodes 1 --episode_len 3 --save_path outputs/datasets/abb_smoke.npz

python scripts/collect_data.py \
  --num_episodes 20 \
  --episode_len 200 \
  --num_envs 1 \
  --save_path outputs/datasets/irb2400_parallel_data.npz

python scripts/train_dynamics.py \
  --data_path outputs/datasets/irb2400_parallel_data.npz \
  --model_type mlp \
  --epochs 20 \
  --batch_size 512
```

确认 MLP 能跑通后，再采更多数据并训练 GRU 或 Transformer。

## 16. 常见问题

### 找不到 XML 文件

确认你在项目根目录运行：

```bash
pwd
```

应该输出：

```text
/home/xinlei/Data/RL_Projects/MPC_RL_RobotArm/learned_mujoco_dynamics
```

默认 XML 文件应存在：

```bash
ls ABB_IRB2400.xml
```

### 找不到 mesh 或 STL 文件

确认资产目录存在：

```bash
ls abb_irb2400_assets
```

`ABB_IRB2400.xml` 中配置了：

```xml
<compiler angle="radian" meshdir="abb_irb2400_assets"/>
```

所以 `abb_irb2400_assets/` 必须和 `ABB_IRB2400.xml` 在同一个项目目录下。

### actuator 数量不匹配

如果出现 actuator 数量不足的报错，说明 XML 中可控 actuator 数量少于 `--n_joints`。对于 ABB IRB 2400，当前 XML 有 6 个 actuator，适合：

```bash
--n_joints 6
```

### MuJoCo viewer 打不开

这通常是图形界面或远程显示问题。可以先跳过 `rollout_visualize.py`，直接运行数据采集和训练。

### matplotlib cache 目录警告

如果看到类似：

```text
Matplotlib created a temporary cache directory at /tmp/...
```

一般不影响训练或评估。需要消除警告时，可以设置：

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```

## 17. Learned CEM-MPC

MPC 代码位于 Git 仓库顶层，和 `learned_mujoco_dynamics/` 同级：

```text
MPC_RL_RobotArm/
├── learned_mujoco_dynamics/
├── mpc/
├── scripts/
└── tests/
```

训练、dynamics 数据采集、open-loop 评估仍从 `learned_mujoco_dynamics/` 目录运行；MPC 脚本从 Git 顶层运行。`learned_dynamics/` 仍只负责模型、normalizer、数据集和 learned rollout；CEM、cost、constraint、planner rollout 等控制逻辑都在顶层 `mpc/` 中。MPC 闭环运行和 A/B/C 评估的新输出默认写入顶层 `outputs/mpc/`。

命名约定很重要：

- `q_ref` / `actuator_q_ref` 表示发送给 MuJoCo position actuator 的绝对目标关节角，也是 dynamics 模型训练时 `actions` 字段的语义。
- `q_des` 表示希望真实机械臂关节角跟踪的任务参考轨迹。
- CEM 默认优化 `delta_q_ref`，但 planner 会先把它按 horizon 累积成绝对 `actuator_q_ref`，然后再喂给 learned dynamics。不要把 raw `delta_q_ref` 直接作为模型 action 输入。

闭环运行示例：

```bash
cd /home/xinlei/Data/RL_Projects/MPC_RL_RobotArm

python scripts/run_cem_mpc.py \
  --checkpoint learned_mujoco_dynamics/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt \
  --normalizer learned_mujoco_dynamics/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt \
  --model_type transformer \
  --horizon 20 \
  --num_samples 1024 \
  --cem_iters 4 \
  --rollout_batch_size 256 \
  --reference_mode multi_joint_sine \
  --save_dir outputs/mpc/transformer_20260606_154206
```

输出包括：

- `rollout.npz`：actual state、`q_des`、selected `actuator_q_ref`、selected `delta_q_ref`、planning time、best/elite cost、failure flags 等。
- `rollout.csv`：逐步标量诊断。
- `q_tracking.png`、`dq.png`、`control.png`、`tracking_error.png`、`planning_diagnostics.png`。

采集 MPC-induced 数据用于 Model C：

```bash
cd /home/xinlei/Data/RL_Projects/MPC_RL_RobotArm

python scripts/collect_mpc_data.py \
  --checkpoint learned_mujoco_dynamics/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt \
  --normalizer learned_mujoco_dynamics/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt \
  --model_type transformer \
  --episode_len 200 \
  --save_path outputs/datasets/mpc_induced_data.npz
```

该数据集保持 `train_dynamics.py` 可读取的核心 schema：

```text
states
actions
next_states
episode_ids
q_ref
delta_q_ref
motion_mode_ids
tau_actuator
tau_gravity
tau_total
```

额外的 `mpc_cost`、`planning_time`、`failure_flags`、`source_policy`、`reference_mode_ids` 只作为实验元数据。

## 18. Model A/B/C 实验框架

默认 A/B/C 定义为数据来源对比：

- Model A：原始四类 motion modes（`hold`、`step`、`smooth_random`、`sine`）。
- Model B：Model A 数据加扩展 collector modes（`delta_ref_random`、`mpc_correlated_random`）。
- Model C：Model B 数据加 MPC-induced closed-loop 数据。

统一闭环评估：

```bash
cd /home/xinlei/Data/RL_Projects/MPC_RL_RobotArm

python scripts/evaluate_model_abc.py \
  --model_spec A,learned_mujoco_dynamics/outputs/checkpoints_A/best_model.pt,learned_mujoco_dynamics/outputs/checkpoints_A/normalizer.pt,transformer,learned_mujoco_dynamics/outputs/datasets/model_a.npz \
  --model_spec B,learned_mujoco_dynamics/outputs/checkpoints_B/best_model.pt,learned_mujoco_dynamics/outputs/checkpoints_B/normalizer.pt,transformer,learned_mujoco_dynamics/outputs/datasets/model_b.npz \
  --model_spec C,learned_mujoco_dynamics/outputs/checkpoints_C/best_model.pt,learned_mujoco_dynamics/outputs/checkpoints_C/normalizer.pt,transformer,learned_mujoco_dynamics/outputs/datasets/model_c.npz \
  --reference_mode multi_joint_sine \
  --save_dir outputs/mpc/model_abc
```

OOD 分析按每个模型自己的训练集统计，同时可选使用 Dataset A 作为公共 baseline：

```bash
cd /home/xinlei/Data/RL_Projects/MPC_RL_RobotArm

python scripts/analyze_ood_mpc.py \
  --pair A,outputs/mpc/model_abc/A/rollout.npz,learned_mujoco_dynamics/outputs/datasets/model_a.npz \
  --pair B,outputs/mpc/model_abc/B/rollout.npz,learned_mujoco_dynamics/outputs/datasets/model_b.npz \
  --pair C,outputs/mpc/model_abc/C/rollout.npz,learned_mujoco_dynamics/outputs/datasets/model_c.npz \
  --baseline_dataset learned_mujoco_dynamics/outputs/datasets/model_a.npz \
  --save_csv outputs/mpc/model_abc/ood_summary.csv
```
