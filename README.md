# Arrive and Survive: Scaling Safe Goal-Conditioned Policy Learning from One-Bit Failure Signals

Official JAX/Brax implementation of **Safe-CRL**, a survival-aware extension of
Scaling-CRL for goal-conditioned reinforcement learning with one-bit failure
signals.

Safe-CRL preserves the scalable contrastive goal-reaching objective while
adding two survival corrections: replay-observed survival-mass weighting for
critic InfoNCE and a learned goal-independent survival model used by the actor.

## Methods

The trainer exposes exactly two critic modes:

- `scaling_crl`: the reachability-only Scaling-CRL baseline.
- `scaling_crl_survive`: Safe-CRL, with mass-weighted InfoNCE, a learned
  survival model `Z(s,a)`, and the actor `-log Z` correction.

Both modes use the same concatenated residual MLP actor and two-tower
negative-L2 contrastive critic. Safe-CRL is the default in the supplied paper
configurations.

## Environment configuration

The experiments use the following core environment:

```yaml
python: "3.11"
jax-cuda13: "0.10.0"
brax: "0.14.2"
mujoco: "3.8.0"
mujoco-mjx: "3.8.0"
```

Full experiments are intended for Linux with CUDA 13-compatible NVIDIA
drivers. Windows execution is CPU-only.

```bash
conda create -n safe-crl python=3.11 -y
conda activate safe-crl
python -m pip install \
  jax-cuda13==0.10.0 \
  brax==0.14.2 \
  mujoco==3.8.0 \
  mujoco-mjx==3.8.0
```

## Coding-agent context

We provide a compact [project handoff](project_info/) and a
[`give-command` skill](.agents/skills/give-command/SKILL.md) for coding agents.
Ask Codex or DeepSeek-V4-Flash to read both before working with the repository;
they describe the environments, observations, Safe-CRL implementation,
configurations, caveats, and verified training workflow.

For example:

> Read `project_info/README.md` and all linked project notes, then read
> `.agents/skills/give-command/SKILL.md`. Use the current code and per-environment
> YAMLs as the source of truth for subsequent work.

## Training examples

Each benchmark task has a complete configuration under
`source-code/scalable_safe_rl/configs/`.

Single Humanoid Goal run with seed 0:

```bash
cd source-code
python -m scalable_safe_rl.train \
  --config scalable_safe_rl/configs/config_humanoid_goal.yaml \
  --seed 0
```

Five Humanoid Goal runs with seeds 0, 5, 10, 15, and 20 distributed across
three GPUs:

```bash
cd source-code
GPUS="0 1 2" \
SEEDS="0 5 10 15 20" \
RUN_NAME=humanoid_goal_safe_crl \
ENV_ID=humanoid_goal_headless \
CONFIG=scalable_safe_rl/configs/config_humanoid_goal.yaml \
./run_seeds_dynamic.sh
```
