---
name: give-command
description: Inspect this repository and its current project_info documentation to generate verified, copy-ready Python or shell commands for experiment training, dry runs, smoke tests, multi-seed or multi-GPU launches, persistence, policy visualization, metric plotting, and monitoring. Use when the user asks what command to run for a method, environment, configuration, seed set, GPU assignment, saved policy, experiment utility, or related workflow in this repository.
---

# Give Command

Generate commands from the repository's current state. Print commands only; never execute an experiment or modify its configuration.

## Resolve The Request

1. Identify the requested operation, method, environment, config, overrides, seeds, GPUs, and run name.
2. Inspect only the relevant current files. Start from `project_info/README.md`, then route to the applicable `project_info/` command or subsystem document. Executable code lives under `source-code/`. For paper training commands, read `project_info/04_networks_config.md`, `project_info/05_commands.md`, and the selected YAML under `source-code/scalable_safe_rl/configs/`. For multi-seed commands, also read `project_info/09_seed_runs_plotting_visualization.md` and `source-code/run_seeds_dynamic.sh`. When seed policy or paper experiment classification matters, inspect the current experiment-organization document referenced by the project docs.
3. Confirm syntax against the current entrypoint, parser, YAML config, or launcher script before composing the command.
4. Use this precedence when sources disagree:
   1. Explicit user requirements.
   2. Executable entrypoint or launcher behavior.
   3. Current YAML configuration.
   4. Current `project_info/` documentation.
   5. Package README examples.
5. Exclude `backup_versions/`, caches, `results/`, `runs/`, and previous run artifacts unless the user explicitly asks to reproduce, resume, visualize, plot, or inspect one.

Use `rg` or `rg --files` for discovery. Verify method names, environment IDs, config paths, dashed CLI flag names, launcher environment variables, seed policy, and automatic evaluation-environment mappings. Do not assume examples remain current.

## Paper Benchmark Configuration Routing

- The 12 Safe-CRL paper environments each have one complete standalone YAML under `source-code/scalable_safe_rl/configs/`. Resolve the paper display name to its current filename using `project_info/04_networks_config.md`, then verify the YAML's `env_id`, `eval_env_id`, `critic_loss_type`, horizon, and network depths before composing the command.
- For a single-seed Safe-CRL paper run, select that environment YAML with `--config` and pass only the requested seed or genuine overrides. Do not redundantly restate its environment IDs, method, depths, or horizon.
- For the matched Scaling-CRL baseline, reuse the same environment YAML and override only `--critic-loss-type scaling_crl` plus any user-requested run settings. This keeps environment and throughput controls matched.
- Use `source-code/scalable_safe_rl/config.yaml` for generic, exploratory, or non-paper commands, not as the default for one of the 12 paper environments.
- `source-code/run_seeds_dynamic.sh` does not derive its `ENV_ID` from `CONFIG`: it always forwards `ENV_ID` as a training CLI override and uses it for output organization. In commands that first enter `source-code/`, set `CONFIG` to `scalable_safe_rl/configs/<selected-file>.yaml` and set `ENV_ID` to exactly that YAML's `env_id`; never rely on the launcher's Point Goal default for another environment.
- Let the launcher infer `METHOD` from `critic_loss_type` unless the user explicitly needs a different output alias. Use the current documented main-result or ablation seed set rather than the launcher's generic default seeds.

## Select The Command Form

- From the public repository root, begin executable commands with `cd source-code`; paths documented in command examples are relative to that directory.
- Use `python -m scalable_safe_rl.train` for one configuration and one seed, including dry runs and smoke tests.
- Use `./run_seeds_dynamic.sh` for multi-seed or multi-GPU training. Populate its documented environment variables, including the matched `CONFIG` and `ENV_ID`, and append only supported CLI overrides.
- Use persistence, visualization, plotting, or monitoring utilities only when the user requests that operation.
- For deterministic saved-policy rollout HTML, use the current `scalable_safe_rl.visualize_policy` module after verifying its CLI.
- Inherit unspecified values from the selected current config. Do not restate defaults as CLI flags unless needed for clarity or correctness.
- Apply the current documented seed policy when the user identifies the experiment as a main result or ablation. If the experiment class is genuinely ambiguous and changes the seed count, ask one concise question.

If the requested method, environment, flag, launcher, or combination is unsupported, say what failed validation and provide verified nearby choices. Never invent a command.

## Output

For a supported request, return exactly one copy-ready fenced `bash` block. After it, add at most one short sentence naming inherited defaults or a necessary assumption. Do not add setup instructions, alternatives, or an explanation unless the user asks. For an unsupported request, return only the concise validation failure and verified nearby choices, with no command block.

Never run the command, launch a job, edit a config, create a result directory, or test by starting training. Read-only syntax checks such as inspecting `--help` are allowed when they cannot launch an experiment.
