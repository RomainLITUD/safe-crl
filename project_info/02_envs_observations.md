# Envs And Observations

Env code lives mainly in `source-code/safenav_jax/envs/` and
`source-code/scaling_envs/`;
their YAML defaults live in the corresponding config directories.

## Supported Groups

- Goal navigation: `point_goal(_headless)`, `car_goal(_headless)`,
  `ant_goal(_headless)`, and `humanoid_goal(_headless)`.
- Pitfalls: `ant_goal_grid(_headless)` and `humanoid_goal_grid(_headless)` with
  the named Big, Hardest, or Loop pitfall layout selected by the benchmark YAML.
- Mazes: `crl_ant_h_maze(_eval)`, `crl_ant_cross_maze(_eval)`,
  `crl_ant_big_maze(_eval)`, and `crl_humanoid_u_maze(_eval)`.
- The active Ant H-Maze and Cross-Maze IDs are `crl_ant_h_maze(_eval)` and
  `crl_ant_cross_maze(_eval)`. The former `crl_ant_u4_maze(_eval)` and
  `crl_ant_u7_maze(_eval)` names are no longer supported by active code.

Renderable IDs are for visualization and `_headless` SafeNav IDs are preferred
for training. `scaling_envs` IDs intentionally have no headless variants. With
empty `eval_env_id`, training and epoch evaluation use the same environment ID
and layout. Set an explicit eval ID to request a distinct eval layout.

## Training Observation Convention

- Trainer wrappers expose `state_context + task_goal` to the actor and relabeling code.
- `obs_dim` is the state-context length; `raw_goal_dim` is the appended goal length.
- Replay stores raw rollout observations plus extras such as `cost`, `task_goal`, `achieved_goal`, `agent_yaw`, and `relabel_anchor_xy`.
- Relabeling always uses future achieved quantities from the same trajectory, not future task goals.

## Goal Frames And Relabeling

- The supplied Point/Car Goal configs set `ego_view=false`. Robot state uses
  `q + qd` beginning with global robot XY; lidar, task goals, and replay goals
  remain in global coordinates, and robot XY remains in policy/SA input.
- Ant tasks relabel future robot XY. Humanoid tasks use the environment's
  achieved-goal contract, including XYZ where configured by the locomotion
  environment.

## Lidar And Layout

- Point/car goal lidar is type-separated into hazards, obstacles, and gremlins;
  it rotates with the robot only when `ego_view=true`.
- Goal-grid lidar is hazard-only by default because gremlins are removed.
- `ignore_layout_obs=true` removes all declared lidar channels from actor and
  encoder state inputs. It does not generally stop the underlying environment
  from computing its raw lidar observation.

## Costs And Rewards

- Goal navigation: sparse reward on reaching the goal; binary cost from unsafe objects.
- Goal grid: reward is robot root entering the goal area; cost is robot root entering ghost hazard cells.

## Goal-Grid Layout Contract

- Ant and Humanoid Goal-Grid share named symbolic layouts defined in
  `goal_grid_maze_base.py`: `R` is the unique reset cell, `G` is a goal cell,
  `1` is a fixed ghost-hazard cell, and `0` is traversable but not a goal.
- Robot XY is always reset exactly to `R`; `fixed_agent_on_reset` and object
  relocation flags do not change the symbolic layout.
- Goals are sampled only from `G` cells. Goal respawn is controlled by
  `goal_respawn_on_success`; the current scalable config sets it to `false`.
- `grid_layout_name` selects training layout and `eval_grid_layout_name` selects
  eval layout only when the environment is constructed with evaluation mode.
- Active named pitfall pairs are `big_pitfall(_eval)`,
  `hardest_pitfall(_eval)`, and `loop_pitfall(_eval)`. The trainer exposes both
  selectors directly so the benchmark YAMLs can distinguish the four pitfall
  tasks without editing environment source.

## Cost-Free `scaling_envs`

- These environments are accepted only by `scaling_crl` and
  `scaling_crl_survive` and expose no safety cost beyond stable zero metrics.
- CRL maze layouts contain one `R`, reset there with the configured
  Ant/Humanoid noise, include wall lidar, and never respawn the episode goal.
- Cost-free Ant goals use XY achieved/task goals; Humanoid goal and maze tasks
  use XYZ. Rollout/evaluation receives the task goal and replay relabels future
  achieved robot positions.
