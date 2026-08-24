# Project Info Index

This folder is a compact handoff for future coding threads.

The 12 paper environments each have a complete standalone training YAML under
`source-code/scalable_safe_rl/configs/`. The configuration inventory is in
`04_networks_config.md`, and verified single-seed and multi-GPU examples using
those YAMLs are in `05_commands.md`.

Read in this order:

1. `01_overview.md` - project goal and repo layout.
2. `02_envs_observations.md` - SafeNav envs, observations, costs, layouts.
3. `03_training_algorithm.md` - replay relabeling and critic/actor losses.
4. `04_networks_config.md` - shared residual network structure and important config knobs.
5. `05_commands.md` - common training, saved-policy visualization, and smoke-test commands.
6. `06_status_caveats.md` - implementation status and important caveats.
7. `09_seed_runs_plotting_visualization.md` - multi-seed output and saved-policy workflow.
