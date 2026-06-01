# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

CA-AC-MPC is a CUDA-accelerated Actor-Critic Model Predictive Control framework for agile quadrotor gate racing (ICUAS 2026 paper, arXiv:2605.29155). A neural network (the actor) predicts cost parameters for a differentiable MPC layer; PPO (RL) trains it end-to-end. The headline contribution is **CA-DiffMPC**, a fused C++/CUDA iLQR solver (~10x faster inference vs. the PyTorch backend).

## Commands

Training (always via a YAML config):
```bash
python train_acmpc.py --config configs/train_acmpc_fixed_map.yaml
```

Override any config key from the CLI (dot-paths, repeatable):
```bash
python train_acmpc.py --config configs/train_acmpc_fixed_map.yaml \
  --override ppo.total_timesteps=1000000 --override mpc_horizon=5 --override vec_env.n_envs=16
```

Evaluation (generates plots/videos under `--log-dir`):
```bash
python utils/evaluate_acmpc2.py \
  --model-path runs/acmpc_h_2/zurich/model.zip --log-dir runs/acmpc_h_2/zurich \
  --track-config envs/tracks/zurich.yaml --policy-type acmpc_diffmpc --mpc-backend fast
```

There is no test suite, linter, or build step. The CUDA extension is JIT-compiled by `torch.utils.cpp_extension.load` on first use (requires `nvcc` and a correct `CUDA_HOME`); build artifacts land in `differentialMPCPerformance/build/`.

## Architecture

**Vendored dependencies live in the repo and are wired up via `sys.path` insertions** (see the top of `train_acmpc.py` and `utils/train_support.py`) — there is no installed package. Key vendored trees under `acmpc_public-master/`:
- `stable-baselines3-acmpc-acmpc/` — a forked SB3 whose `env.reset()` returns `(obs, state)` instead of just `obs`.
- `mpc.pytorch/` — the original Amos et al. differentiable MPC (`pytorch` backend).
- `diff_mpc_drones/drone.py` — quadrotor dynamics (`DroneDx`).
- `training_modules/` — the three policy classes.

**Three MPC backends**, selected by `mpc_backend` and resolved against `policy_type` in `resolve_policy_class` (`utils/train_support.py`):
- `fast` → CUDA iLQR in `differentialMPCPerformance/` (`drone_ilqr.py` loads `cpp/` kernels; `mpc_compat.py` exposes a `FastMPC` with an mpc.pytorch-like API). Requires CUDA.
- `diffmpc` → pure-PyTorch iLQR in `DifferentialMPC/` (`controller.py`, `cost.py`). `ILQRSolve` is a custom `torch.autograd.Function`.
- `pytorch` → legacy vendored `mpc.pytorch`.
With `mpc_backend: auto`, the choice depends on `policy_type` and CUDA availability. `fast` and `diffmpc` are interchangeable for the `acmpc_diffmpc` policy (same `MlpMpcPolicyDiffMPC` class); `pytorch` uses `MlpMpcPolicy`.

**Three policy types** (`policy_type`): `acmpc_diffmpc` (actor predicts MPC cost), `acmpc_mlp` (MPC via mpc.pytorch), `mlp_only` (no MPC baseline).

**Config flows partly through environment variables.** `train_acmpc.py` writes `ACMPC_T` (horizon), `ACMPC_MPC_MAX_ITER`, and `ACMPC_MPC_BACKEND` into `os.environ` before constructing the policy — the vendored policy modules read these at import/init time. Changing horizon/backend means re-running the entrypoint, not just editing in place.

**Observation layout is split.** The env emits a single flat vector: the first `mpc_state_dim` (=10) entries are the raw MPC state `[p, q, v]`; the remainder is the "paper observation" tail `[v, R, o_track]`. `AcmpcPolicyObsExtractor` (`utils/acmpc_obs_extractor.py`) slices off the tail so the cost network never sees the raw state prefix. The MPC block consumes the prefix via the forked SB3's `(obs, state)` reset contract (`ResetWithRawStateWrapper`). `obs_mode: paper` and `n_future_gates` in the env config determine the tail's size — keep `mpc_state_dim` consistent with them.

**Environment**: `envs/gate_racing_env.py` (`GateRacingEnv`), a Gym 0.21 env. Tracks are YAML files in `envs/tracks/` (gates as `gate_N: [x, y, z, yaw]`). Pass a fixed track with `env.track_config_path` / `--track-config`; otherwise `env.track` selects a preset (`envs/track_presets.py`).

**Run management** (`utils/train_support.py`): with `logging.auto_track_naming: true`, the track tag is appended to `log_dir`/`checkpoint_dir`/`save_path`. Checkpoints carry a *training signature* (policy_type, mpc_backend, horizon, track); resume auto-loads `best_model.zip` only if the signature matches. VecNormalize stats are saved alongside checkpoints as `<name>.vecnorm.pkl`.

## Conventions

- Add new training presets as `configs/*.yaml`; mirror the structure of `train_acmpc_fixed_map.yaml` (sections: `env`, `vec_env`, `ppo`, `logging`, `checkpoint`, `resume`, `eval`, `post_eval`).
- New tracks: add a YAML to `envs/tracks/` and pass it via `--track-config`.
- When touching the CUDA path, the `.cu`/`.cuh`/wrapper sources are in `differentialMPCPerformance/cpp/`; delete `differentialMPCPerformance/build/` to force a clean recompile.
