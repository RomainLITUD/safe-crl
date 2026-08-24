"""Benchmark batched stepping speed for renderable safe-navigation envs."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax

from safenav_jax.benchmark_headless_speed import _print_result
from safenav_jax.benchmark_headless_speed import benchmark_env
from safenav_jax.visualize_initial_states import DEFAULT_ENV_IDS
from safenav_jax.visualization_rollout import resolve_visual_env_id


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark JAX-vmapped renderable SafeNav env stepping speed. "
            "Renderable envs mirror headless task logic and keep MJCF visual bodies synced."
        )
    )
    parser.add_argument(
        "--env-id",
        action="append",
        dest="env_ids",
        help="Env id to benchmark. Repeat to benchmark multiple envs. Defaults to all renderable envs.",
    )
    parser.add_argument("--num-envs", type=int, default=256, help="Number of parallel envs.")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Jitted scan length per timing sample.")
    parser.add_argument("--warmup-rollouts", type=int, default=1, help="Untimed rollouts after compilation.")
    parser.add_argument("--measure-rollouts", type=int, default=5, help="Timed rollout samples.")
    parser.add_argument(
        "--action-mode",
        choices=("zero", "random"),
        default="zero",
        help="Use constant zero actions or random uniform actions in [-1, 1].",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base PRNG seed.")
    parser.add_argument("--config-dir", default=None, help="Directory containing env_id.yaml config files.")
    parser.add_argument(
        "--backend",
        default=None,
        help=(
            "Optional backend override. Ant/Humanoid renderable geometry costs require mjx, "
            "so leave this unset or use mjx when benchmarking all envs."
        ),
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=None,
        help="Optional n_frames override passed to each env constructor.",
    )
    parser.add_argument(
        "--compact-robot-obs",
        action="store_true",
        help="Use full_robot_observation=False for renderable Ant/Humanoid envs.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1.")
    if args.rollout_steps < 1:
        raise ValueError("--rollout-steps must be at least 1.")
    if args.measure_rollouts < 1:
        raise ValueError("--measure-rollouts must be at least 1.")

    env_ids = tuple(args.env_ids) if args.env_ids else DEFAULT_ENV_IDS
    print(f"JAX devices: {jax.devices()}", flush=True)
    print(f"JAX default backend: {jax.default_backend()}", flush=True)

    all_sps = []
    for offset, env_id in enumerate(env_ids):
        env_id = resolve_visual_env_id(env_id, headless=False)
        env_kwargs: dict[str, Any] = {"use_config": True}
        if args.config_dir is not None:
            env_kwargs["config_dir"] = args.config_dir
        if args.backend is not None:
            env_kwargs["backend"] = args.backend
        if args.n_frames is not None:
            env_kwargs["n_frames"] = args.n_frames
        if args.compact_robot_obs and env_id in ("ant_goal", "humanoid_goal"):
            env_kwargs["full_robot_observation"] = False

        result = benchmark_env(
            env_id=env_id,
            num_envs=args.num_envs,
            rollout_steps=args.rollout_steps,
            warmup_rollouts=args.warmup_rollouts,
            measure_rollouts=args.measure_rollouts,
            action_mode=args.action_mode,
            seed=args.seed + offset,
            env_kwargs=env_kwargs,
        )
        all_sps.append(float(result["mean_steps_per_second"]))
        _print_result(result)

    if len(all_sps) > 1:
        print(f"mean_sps_across_envs={statistics.fmean(all_sps):.0f}", flush=True)


if __name__ == "__main__":
    main()
