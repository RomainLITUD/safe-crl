# Seed Runs And Policy Visualization

Multi-seed CRL runs use this output layout:

```text
results/<env_id>/<method>/<run_name>/
  commands.txt
  runs.tsv
  metrics_long.tsv
  metrics_final.tsv
  policies.tsv
  policy_visualization_commands.txt
  <seed>/
    seed_<seed>.log
    config.json
    config.yaml
    metrics.jsonl
    metrics_long.tsv
    metrics_final.tsv
    final_policy/
      actor_params.pkl
      args.pkl
      policy.pkl
```

An existing run name is automatically suffixed with `_1`, `_2`, and so on.

## Multi-Seed Training

Use the complete environment-specific YAMLs in `scalable_safe_rl/configs/` for
paper runs. The launcher requires `ENV_ID` to match the selected YAML's
`env_id`, because it forwards that value to training and uses it to organize
outputs.

```bash
RUN_NAME=ant_big_maze_safe_crl \
GPUS="0 1 2" \
SEEDS="0 5 10 15 20" \
ENV_ID=crl_ant_big_maze \
CONFIG=scalable_safe_rl/configs/config_ant_big_maze.yaml \
./run_seeds_dynamic.sh
```

`RESULTS_ROOT` defaults to `results`. `RUN_ID` remains a legacy alias for
`RUN_NAME`; `RUN_DIR` and `LOG_DIR` remain explicit path overrides.
The launcher passes an exact per-seed output directory to the trainer, so
logs, resolved configs, metrics, and policies cannot be split across roots.

`runs.tsv` uses the shared columns:

```text
run_id, seed, gpu, env_id, method, status, exit_code,
log_file, run_dir, started_at, finished_at
```

Each seed has its own `metrics_long.tsv` and `metrics_final.tsv`; aggregate
files with byte-compatible headers live in the parent run directory.

## Background Execution

The launcher starts seed jobs as concurrent background child processes, but
the launcher itself remains attached to the terminal and waits for every seed.
Closing that terminal may terminate the launcher and active jobs.

For a persistent run:

```bash
nohup env SEEDS="0 5 10 15 20" GPUS="0 1 2" RUN_NAME=ant_big_maze_safe_crl \
  ENV_ID=crl_ant_big_maze \
  CONFIG=scalable_safe_rl/configs/config_ant_big_maze.yaml \
  ./run_seeds_dynamic.sh > my_run_launcher.log 2>&1 &
```

Monitor the launcher with `tail -f my_run_launcher.log`. A `tmux` or
`screen` session is preferable when interactive monitoring is needed.

## Policy Files

Canonical policy paths are indexed in `policies.tsv`. Runs also receive
ready-to-run commands in `policy_visualization_commands.txt`:

```bash
python -m scalable_safe_rl.visualize_policy \
  --policy-dir results/<env_id>/<method>/<run_name>/<seed>/final_policy \
  --output results/<env_id>/<method>/<run_name>/<seed>/final_policy/eval_rollout.html
```
