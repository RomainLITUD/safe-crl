# SafeNav JAX Package Tutorial

This package-level note keeps the most commonly used SafeNav commands close to
the scripts. The fuller environment tutorial is in `env_mds/tutorial.md`.

## Observation Summary

Default observation sizes:

| Env IDs | Observation size | Robot/context size | Goal tail |
|---|---:|---:|---|
| `point_goal(_headless)` | 54 | 52 | local XY |
| `car_goal(_headless)` | 72 | 70 | local XY |
| `ant_goal(_headless)` | 131 | 129 | local XY |
| `humanoid_goal(_headless)` | 342 | 340 | local XY |

Point/Car use agent-frame pseudo-lidar for hazards, obstacles, and current
gremlins. Ant/Humanoid use local-frame object positions. Global agent XY is
removed from the default observation for all four robots.

## Configs

Default env constructor parameters are in:

```text
safenav_jax/configs/<env_id>.yaml
```

Scripts load these YAML files automatically and accept:

```bash
--config-dir /path/to/configs
```

Python construction from YAML:

```python
from safenav_jax.envs import make

env = make("humanoid_goal_headless", use_config=True)
env_custom = make("humanoid_goal_headless", use_config=True, num_hazards=4)
```

Useful reset/layout flags:

```bash
--relocate-objects-on-reset
--no-relocate-objects-on-reset
--fixed-object-layout-seed 0
```

## Fast Visualization Rollouts

Renderable HTML visualization with random actions:

```bash
python safenav_jax/visualize_short_rollouts.py \
  --env-id ant_goal \
  --steps 1000 \
  --action-mode random \
  --jit-rollout \
  --frame-stride 10 \
  --max-render-frames 120 \
  --height 720 \
  --output-dir safenav_jax/visualizations/one_episode
```

Headless SVG visualization with random actions:

```bash
python safenav_jax/visualize_headless_short_rollouts.py \
  --env-id ant_goal_headless \
  --steps 1000 \
  --action-mode random \
  --jit-rollout \
  --frame-stride 10 \
  --max-render-frames 120 \
  --output-dir safenav_jax/visualizations/one_episode
```

Useful rollout options:

- `--action-mode zero|random`
- `--jit-rollout`
- `--frame-stride N`
- `--max-render-frames N`
- `--no-auto-reset-on-done`

GPU acceleration helps rollout generation when JAX is using a GPU. HTML and SVG
serialization still run on Python/CPU, so downsample long episodes. You normally
do not need to force `JAX_PLATFORMS=cuda`; JAX will use CUDA if the installed
build and runtime support it.

If Chrome shows the Brax figure as a narrow strip, regenerate with a larger
`--height`, such as `--height 720` or `--height 900`.

## Visualization Commands

All headless initial states:

```bash
python safenav_jax/visualize_headless_initial_states.py
```

One headless rollout:

```bash
python safenav_jax/visualize_headless_short_rollouts.py \
  --env-id ant_goal_headless \
  --steps 32 \
  --action-mode random
```

All renderable initial states:

```bash
python safenav_jax/visualize_initial_states.py
```

One renderable rollout:

```bash
python safenav_jax/visualize_short_rollouts.py \
  --env-id ant_goal \
  --steps 32 \
  --action-mode random
```

## Speed Benchmarks

Headless benchmark:

```bash
python safenav_jax/benchmark_headless_speed.py \
  --env-id ant_goal_headless \
  --num-envs 1024 \
  --rollout-steps 512 \
  --warmup-rollouts 2 \
  --measure-rollouts 5 \
  --action-mode random
```

Renderable benchmark:

```bash
python safenav_jax/benchmark_renderable_speed.py \
  --env-id ant_goal \
  --num-envs 512 \
  --rollout-steps 256 \
  --warmup-rollouts 2 \
  --measure-rollouts 5 \
  --action-mode random
```

Run heavy benchmark settings on a CUDA-JAX GPU server.
