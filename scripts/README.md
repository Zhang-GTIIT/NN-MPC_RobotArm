# Script layout

Stable user-facing entry points remain directly under `scripts/`:

| Entry point | Purpose |
|---|---|
| `run_cem_mpc.py` | Main MPC and Direct-IK runner |
| `run_cem_budget_sweep.py` | CEM compute-budget sweep |
| `generate_task_reference.py` | Task-space reference generation |
| `validate_ik.py` | IK validation |
| `collect_mpc_data.py` | General MPC data collection |
| `analyze_ood_mpc.py` | General OOD analysis |

Specialized workflows are grouped by domain:

| Directory | Purpose |
|---|---|
| `robustness/` | Model-A, Direct-IK, and delay-aware robustness evaluation |
| `model_c/` | Model-C collection, dataset, benchmark, and evaluation workflows |
| `paper_experiments/` | Reproducible paper experiment matrix and summaries |
| `experiment_utils/` | Shared hashing, manifest, bootstrap, environment, and resume helpers |
| `experiments/planner_projection/` | Planner-projection calibration and ablations |
| `experiments/comparisons/` | Cross-controller result comparisons |

Experiment outputs belong under `outputs/`; scripts should not write generated
files into the source tree.

