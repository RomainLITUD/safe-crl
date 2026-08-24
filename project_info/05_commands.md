# Common Commands

Run these commands from `source-code/` in the public repository.

Dry-run init:

```bash
python -m scalable_safe_rl.train \
  --config scalable_safe_rl/configs/config_point_goal.yaml \
  --dry-run \
  --num-envs 2 \
  --num-eval-envs 2 \
  --max-replay-size 1000 \
  --min-replay-size 1000 \
  --batch-size 2 \
  --num-sgd-batches-per-training-step 1
```

Run one Safe-CRL paper experiment (one seed):

```bash
python -m scalable_safe_rl.train \
  --config scalable_safe_rl/configs/config_ant_big_maze.yaml \
  --seed 0
```

The YAML supplies the environment and evaluation IDs, Safe-CRL critic mode,
training horizon, network depths, replay settings, evaluation settings, and
artifact controls. Replace the config filename to select another one of the 12
paper environments listed in `project_info/04_networks_config.md`.

Run the matched Scaling-CRL baseline with the same environment configuration:

```bash
python -m scalable_safe_rl.train \
  --config scalable_safe_rl/configs/config_ant_big_maze.yaml \
  --critic-loss-type scaling_crl \
  --seed 0
```

Run the five main-paper seeds across multiple GPUs:

```bash
RUN_NAME=ant_big_maze_safe_crl \
GPUS="0 1 2" \
SEEDS="0 5 10 15 20" \
ENV_ID=crl_ant_big_maze \
CONFIG=scalable_safe_rl/configs/config_ant_big_maze.yaml \
./run_seeds_dynamic.sh
```

For the launcher, `CONFIG` selects the YAML while `ENV_ID` must repeat that
YAML's `env_id`; the launcher uses `ENV_ID` for both the CLI override and output
directory organization. Add `--critic-loss-type scaling_crl` to the launcher
command for the matched baseline.

Two-method seed-run outputs:

- `${RUN_DIR}/<seed>/seed_<seed>.log`
- `${RUN_DIR}/<seed>/config.json` and `config.yaml`
- `${RUN_DIR}/<seed>/metrics.jsonl`
- `${RUN_DIR}/<seed>/final_policy/policy.pkl`
- `${RUN_DIR}/metrics_long.tsv`
- `${RUN_DIR}/metrics_final.tsv`
- `${RUN_DIR}/runs.tsv`
- `${RUN_DIR}/policies.tsv`
- `${RUN_DIR}/policy_visualization_commands.txt`

The launcher parallelizes seed jobs internally but remains attached to the
terminal. For persistent `nohup` and monitoring commands, see
`project_info/09_seed_runs_plotting_visualization.md`.

Visualize saved policy:

```bash
python -m scalable_safe_rl.visualize_policy \
  --policy-dir results/<env_id>/<method>/<run_name>/<seed>/final_policy \
  --output results/<env_id>/<method>/<run_name>/<seed>/final_policy/eval_rollout.html
```

The visualizer maps `_headless` env IDs to renderable env IDs automatically.

CPU note:

- This machine may not have CUDA-enabled JAX.
- Avoid long training tests locally; use dry runs and tiny smoke tests only.
