# MPC package layout

The package keeps runtime modules at the top level so existing `mpc.*` imports
remain stable. Files are grouped by responsibility below.

| Area | Modules |
|---|---|
| CEM planning | `cem_controller.py`, `planner_rollout.py`, `cost_functions.py` |
| Physical constraints | `constraints.py`, `recovery.py` |
| Delay-aware control | `delay_aware.py`, `delay_protocol.py`, `delay_aware_runner.py` |
| Threaded ASAP | `asap_runner.py`, `asap_planner_worker.py`, `asap_shared.py`, `asap_timing.py`, `asap_types.py` |
| References and IK | `reference.py`, `reference_pipeline.py`, `task_space_reference.py`, `ik_solver.py`, `kinematics_utils.py` |
| Diagnostics | `logging.py`, `replay_diagnostics.py`, `counterfactual.py`, `robustness.py` |
| Model-C components | `model_c/` |
| Tests | `tests/`, with Model-C tests in `tests/model_c/` |

Run the MPC test suite from the repository root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 conda run -n pendulum-rl pytest -q mpc/tests
```

