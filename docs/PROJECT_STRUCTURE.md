# NN-MPC_RobotArm Project Structure

This repository is organized as one top-level robotics learning and control project. The learned dynamics code remains in `dynamics_modeling/`; the CEM-MPC control pipeline lives at the repository root.

## Top-Level Layout

```text
NN-MPC_RobotArm/
├── docs/                      # Project documentation and experiment notes
├── dynamics_modeling/   # MuJoCo robot model, dynamics learning code, dynamics scripts
├── mpc/                       # CEM-MPC controller, costs, constraints, planner rollout
├── scripts/                   # Top-level MPC experiment entrypoints
├── tests/                     # Top-level MPC pipeline tests
├── outputs/                   # Top-level MPC experiment outputs, created on demand and ignored by git
└── .gitignore
```

Use the repository root for MPC experiments:

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm
python scripts/run_cem_mpc.py ...
```

Run dynamics data collection, training, and open-loop learned-dynamics evaluation from the repository root as well:

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm
python dynamics_modeling/scripts/train_dynamics.py ...
```

## Dynamics Submodule

`dynamics_modeling/` contains the ABB IRB 2400 MuJoCo model, mesh assets, learned dynamics models, data collectors, training scripts, diagnostics, and legacy dynamics experiment outputs.

Important directories:

- `ABB_IRB2400.xml`: default MuJoCo model.
- `abb_irb2400_assets/`: visual STL meshes used by the XML.
- `neural_dynamics/`: current dynamics package used by the top-level MPC pipeline.
- `scripts/`: current dynamics data collection, training, evaluation, plotting, and diagnostics.
- `outputs/`: dynamics checkpoints, datasets, figures, and diagnostics; ignored by git.

The core dataset schema is:

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
```

For current position-control data, `actions` and `q_ref` both mean absolute actuator target joint angles.

## `neural_dynamics`

`neural_dynamics` is the only dynamics-modeling package used by the top-level MPC code. It supports MLP, GRU, and Transformer dynamics models, rollout loss in `scripts/train_dynamics.py`, and the learned rollout utilities used by `mpc/planner_rollout.py`.

## MPC Pipeline

Top-level MPC code is split into reusable control logic and runnable scripts.

`mpc/` contains:

- `cem_controller.py`: CEM optimizer and fallback behavior.
- `planner_rollout.py`: maps normalized residual candidates around a projected nominal to executable absolute actuator `q_ref`, rolls out learned dynamics, and computes costs.
- `cost_functions.py`: residual anchor, black-box servo proxy, residual smoothness, tracking, and joint/velocity-limit costs.
- `constraints.py`: nominal projection plus command velocity, acceleration, and joint-limit handling.
- `recovery.py`: sustained-failure recovery decisions; command-limit saturation remains diagnostic only.
- `reference.py`: reference trajectory generation and finite-difference desired velocity.
- `kinematics_utils.py`: isolated MuJoCo TCP FK, Jacobian, singular-value, and orientation-error helpers.
- `ik_solver.py`: bounded DLS pose IK and continuous warm-started trajectory IK.
- `task_space_reference.py`: circle, ellipse, figure-8, and square TCP pose trajectories.
- `reference_pipeline.py`: offline task-reference assembly, IK validation, and `ReferenceBundle` persistence.
- `logging.py`: saves `rollout.npz`, `rollout.csv`, diagnostic figures, and run summaries including CEM selection and recovery counts.
- `utils.py`: shared helper utilities such as history tensor construction.

Top-level `scripts/` contains:

- `run_cem_mpc.py`: runs one closed-loop learned CEM-MPC rollout in MuJoCo.
- `generate_task_reference.py`: generates and validates a task-space reference file and diagnostic plots.
- `validate_ik.py`: independently revalidates a saved task-space reference.
- `collect_mpc_data.py`: converts MPC rollouts into training datasets for Model C style experiments.
- `evaluate_model_abc.py`: evaluates A/B/C checkpoints under common MPC settings.
- `analyze_ood_mpc.py`: compares MPC rollout state/action queries against training dataset distributions.

New MPC outputs are written under top-level `outputs/`, for example:

```text
outputs/mpc/cem_run/
outputs/mpc/model_abc/
outputs/datasets/mpc_induced_data.npz
outputs/references/circle_3laps/reference.npz
```

Dynamics datasets, checkpoints, figures, and diagnostics are written explicitly under `dynamics_modeling/outputs/...`; this keeps them separate from top-level MPC outputs.

This project map is maintained at `docs/PROJECT_STRUCTURE.md`.

## Path Rules

The repository now uses these path conventions:

- Run both dynamics and MPC scripts from the repository root. Invoke dynamics entrypoints as `dynamics_modeling/scripts/<script>.py`.
- New top-level MPC outputs use `outputs/...`.
- Existing dynamics artifacts use `dynamics_modeling/outputs/...`.
- Dynamics `--model_xml` paths are resolved by their scripts relative to `dynamics_modeling/`; dataset and output paths in root-level commands should explicitly start with `dynamics_modeling/outputs/...`.
- A bare model XML path such as `ABB_IRB2400.xml` resolves to `dynamics_modeling/ABB_IRB2400.xml` in MPC scripts.
- `--reference_mode task` requires a prevalidated `--reference_file`; it uses that file's `execution_steps` instead of `--episode_len`.

Example closed-loop MPC run:

```bash
cd /home/xinlei/Data/RL_Projects/NN-MPC_RobotArm

python scripts/run_cem_mpc.py \
  --checkpoint dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/best_model.pt \
  --normalizer dynamics_modeling/outputs/checkpoints_transformer/transformer_20260606_154206/normalizer.pt \
  --model_type transformer \
  --horizon 10 \
  --num_samples 128 \
  --cem_iters 3 \
  --rollout_batch_size 128 \
  --mpc_policy residual \
  --cem_execute lowest_cost \
  --reference_mode multi_joint_sine \
  --save_dir outputs/mpc/transformer_20260606_154206
```

## Current Continuation Point

The codebase has already moved beyond pure learned dynamics. The current natural continuation point is to run and compare closed-loop learned CEM-MPC experiments:

1. Confirm which checkpoints and datasets exist under `dynamics_modeling/outputs/`.
2. Run a small `scripts/run_cem_mpc.py` smoke rollout into `outputs/mpc/...`.
3. If the rollout works, collect MPC-induced data with `scripts/collect_mpc_data.py`.
4. Train or compare Model A/B/C variants.
5. Analyze OOD behavior with `scripts/analyze_ood_mpc.py`.

If closed-loop performance is poor, return to dynamics training and diagnostics under `dynamics_modeling/`.
