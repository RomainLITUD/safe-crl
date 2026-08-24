# Project Overview

Goal: benchmark scalable goal-conditioned safe RL using Scaling-CRL and its
survival-aware Safe-CRL extension in JAX/Brax SafeNav-style environments.

Main packages:

- `source-code/safenav_jax/`: the Point/Car, locomotion-goal, and pitfall environments used by the benchmark.
- `source-code/scaling_envs/`: the Ant and Humanoid maze environments used by the benchmark.
- `source-code/scalable_safe_rl/`: replay buffer, training loop, CRL losses, actor/critic networks, and saved-policy visualization.
- `source-code/scalable_safe_rl/configs/`: one complete, standalone training YAML for each
  of the 12 Safe-CRL paper environments. Use these files for paper experiments
  instead of rebuilding environment overrides from the generic root config.

The only supported `critic_loss_type` values are:

- `scaling_crl`: reachability-only Scaling-CRL.
- `scaling_crl_survive`: Safe-CRL, including mass-weighted InfoNCE and learned
  goal-independent survival. Older experiment names may call this mode
  MassCRL. This is the shipped default.

Core algorithm idea:

- Both methods use Scaling-CRL's two-tower critic:
  `score = -||SA(...) - G(g)||_2`.
- `scaling_crl_survive` uses the same score matrix but weights each anchor's
  InfoNCE row by its replay-observed survival mass. It also adds a
  goal-independent survival model `Z(s,a)` used by the actor through `-log Z`.
- The mass-weighted critic reduction is unnormalized: it divides by the number
  of valid anchors, not by the sum of survival-mass weights.
- Use replay relabeling: future achieved positions from the same trajectory become hindsight goals.
- Use hindsight goals from the correct achieved robot XY or XYZ quantity for
  each benchmark environment.

Current practical recommendation:

- Use `critic_loss_type: scaling_crl_survive` (Safe-CRL) for the safety-aware method.
- Use `critic_loss_type: scaling_crl` as the reachability-only benchmark.
- Keep the current MLP architecture: both methods use relabeled future goals for actor training and no budget input in the actor or critic.
- During rollout/evaluation, actors still receive the real task goal from the environment.
  During actor training, CRL modes use same-trajectory relabeled goals from the replay sampler.
- Treat `scaling_crl` as the matched baseline for every Safe-CRL comparison.
- Object/wall/hazard lidar is available where the env exposes it.
  `ignore_layout_obs=true` removes every declared layout channel from policy,
  SA-encoder, and Z-encoder inputs.

The public benchmark contains exactly these environment groups:

- Goal: Point Goal, Car Goal, Ant Goal, and Humanoid Goal.
- Maze: Ant H-Maze, Ant Cross Maze, Ant Big Maze, and Humanoid U-Maze.
- Pitfall: Ant Big Pitfall, Ant Hardest Pitfall, Humanoid Loop Pitfall, and
  Humanoid Big Pitfall.
- Ant H-Maze uses `crl_ant_h_maze(_eval)` and Ant Cross Maze uses
  `crl_ant_cross_maze(_eval)`; the old `u4`/`u7` layout IDs are inactive.
- The supplied Point/Car configs use global lidar, state, and goal coordinates
  (`ego_view: false`).
- Training normally uses `_headless`; visualization uses the corresponding renderable env without `_headless`.
- Each supplied YAML declares its training and evaluation IDs explicitly.
