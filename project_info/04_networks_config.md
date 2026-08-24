# Networks And Configuration

Both methods use the same Scaling-CRL actor and two-tower critic:

- `Actor`: state and goal to action mean/log-standard-deviation.
- `SA_encoder`: state/action embedding.
- `G_encoder`: goal embedding.
- `Z_encoder`: state/action survival logit, created only for
  `scaling_crl_survive`.

The critic score is negative L2 distance between the state-action and goal
embeddings. `SA_encoder` always concatenates the full selected state with the
action and processes them with the existing residual MLP trunk. Lidar layout
features, when enabled, remain ordinary entries in that state vector rather
than using a separate convolutional branch.

There is no encoder selector or configurable skip-connection field. Network
depths of at least four use the existing fixed four-layer residual blocks;
this structure is identical for the shared Actor, SA, and G paths in both
methods. Safe-CRL applies the same trunk structure to Z at its configured depth.

Important configuration:

- `critic_loss_type`: `scaling_crl` or `scaling_crl_survive`.
- Actor/critic depths, widths, and embedding dimension.
- `z_encoder_depth`: survival-network depth; `-1` inherits critic depth.
- `gamma`: future-goal bias, Safe-CRL geometric survival-label distribution,
  and the replay-observed mass `1 - gamma ** L_t`.
- `logsumexp_penalty_coeff`: Scaling-CRL critic regularization.
- Entropy and layout-observation settings.

Both modes receive the same default batch size and shared-network depth
settings unless explicitly overridden. `scaling_crl_survive` is the shipped
default.

## Paper Benchmark Configurations

Each main-benchmark environment has a complete standalone YAML in
`source-code/scalable_safe_rl/configs/`. These files contain all shared trainer settings as
well as the correct environment/evaluation IDs, training horizon, network
depths, goal-respawn behavior, layout-observation choice, object-boundary
choice, and pitfall layout selectors.

| Paper environment | Configuration |
|---|---|
| Point Goal | `config_point_goal.yaml` |
| Car Goal | `config_car_goal.yaml` |
| Ant Goal | `config_ant_goal.yaml` |
| Humanoid Goal | `config_humanoid_goal.yaml` |
| Ant H-Maze | `config_ant_h_maze.yaml` |
| Ant Cross Maze | `config_ant_cross_maze.yaml` |
| Ant Big Maze | `config_ant_big_maze.yaml` |
| Humanoid U-Maze | `config_humanoid_u_maze.yaml` |
| Ant Big Pitfall | `config_ant_big_pitfall.yaml` |
| Ant Hardest Pitfall | `config_ant_hardest_pitfall.yaml` |
| Humanoid Loop Pitfall | `config_humanoid_loop_pitfall.yaml` |
| Humanoid Big Pitfall | `config_humanoid_big_pitfall.yaml` |

All filenames in this table are relative to
`source-code/scalable_safe_rl/configs/`.
