"""Benchmark batched stepping speed for headless safe-navigation envs."""

from __future__ import annotations

import argparse
from pathlib import Path
import statistics
import sys
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp

from safenav_jax.envs import make
from safenav_jax.visualize_headless_initial_states import DEFAULT_ENV_IDS
from safenav_jax.visualization_rollout import resolve_visual_env_id


def _block_until_ready(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _make_batched_rollout(env, num_envs: int, rollout_steps: int, action_mode: str):
    batched_reset = jax.jit(jax.vmap(env.reset))
    batched_step = jax.vmap(env.step)

    def rollout(state, key):
        def scan_step(carry, _):
            state, key = carry
            if action_mode == "random":
                key, action_key = jax.random.split(key)
                action = jax.random.uniform(
                    action_key,
                    (num_envs, env.action_size),
                    minval=-1.0,
                    maxval=1.0,
                )
            else:
                action = jp.zeros((num_envs, env.action_size))

            next_state = batched_step(state, action)
            step_checksum = (
                jp.sum(next_state.reward)
                + jp.sum(next_state.done)
                + jp.sum(next_state.metrics["cost"])
                + 1e-6 * jp.sum(next_state.obs)
            )
            return (next_state, key), step_checksum

        (state, key), checksums = jax.lax.scan(scan_step, (state, key), None, length=rollout_steps)
        return state, key, jp.sum(checksums)

    return batched_reset, jax.jit(rollout)


def benchmark_env(
    env_id: str,
    num_envs: int,
    rollout_steps: int,
    warmup_rollouts: int,
    measure_rollouts: int,
    action_mode: str,
    seed: int,
    env_kwargs: dict[str, Any],
) -> dict[str, float | str | int]:
    env = make(env_id, **env_kwargs)
    reset_fn, rollout_fn = _make_batched_rollout(env, num_envs, rollout_steps, action_mode)

    reset_key, rollout_key = jax.random.split(jax.random.PRNGKey(seed))
    reset_keys = jax.random.split(reset_key, num_envs)

    compile_t0 = time.perf_counter()
    state = reset_fn(reset_keys)
    state, rollout_key, checksum = rollout_fn(state, rollout_key)
    _block_until_ready((state, checksum))
    compile_seconds = time.perf_counter() - compile_t0

    for _ in range(warmup_rollouts):
        state, rollout_key, checksum = rollout_fn(state, rollout_key)
        _block_until_ready((state, checksum))

    samples = []
    for _ in range(measure_rollouts):
        t0 = time.perf_counter()
        state, rollout_key, checksum = rollout_fn(state, rollout_key)
        _block_until_ready((state, checksum))
        elapsed = time.perf_counter() - t0
        samples.append(elapsed)

    env_steps = num_envs * rollout_steps
    steps_per_second = [env_steps / elapsed for elapsed in samples]
    return {
        "env_id": env_id,
        "backend": env.backend,
        "num_envs": num_envs,
        "rollout_steps": rollout_steps,
        "action_mode": action_mode,
        "obs_size": int(env.state_dim),
        "action_size": int(env.action_size),
        "compile_seconds": compile_seconds,
        "mean_rollout_seconds": statistics.fmean(samples),
        "std_rollout_seconds": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "mean_steps_per_second": statistics.fmean(steps_per_second),
        "max_steps_per_second": max(steps_per_second),
        "checksum": float(checksum),
    }


def _print_result(result: dict[str, float | str | int]) -> None:
    print(
        " | ".join(
            [
                f"env={result['env_id']}",
                f"backend={result['backend']}",
                f"num_envs={result['num_envs']}",
                f"rollout_steps={result['rollout_steps']}",
                f"action={result['action_mode']}",
                f"obs={result['obs_size']}",
                f"act={result['action_size']}",
                f"compile={result['compile_seconds']:.3f}s",
                f"rollout={result['mean_rollout_seconds']:.6f}s",
                f"sps={result['mean_steps_per_second']:.2f}",
                f"best_sps={result['max_steps_per_second']:.2f}",
                f"checksum={result['checksum']:.3f}",
            ]
        ),
        flush=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark JAX-vmapped headless SafeNav env stepping speed. "
            "The reported SPS is num_envs * rollout_steps / wall_time after JIT compilation."
        )
    )
    parser.add_argument(
        "--env-id",
        action="append",
        dest="env_ids",
        help="Env id to benchmark. Repeat to benchmark multiple envs. Defaults to all headless envs.",
    )
    parser.add_argument("--num-envs", type=int, default=512, help="Number of parallel envs.")
    parser.add_argument("--rollout-steps", type=int, default=256, help="Jitted scan length per timing sample.")
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
            "Optional backend override. Ant/Humanoid headless geometry costs currently require mjx, "
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
        help="Use full_robot_observation=False for Ant/Humanoid headless envs.",
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
        env_id = resolve_visual_env_id(env_id, headless=True)
        env_kwargs: dict[str, Any] = {"use_config": True}
        if args.config_dir is not None:
            env_kwargs["config_dir"] = args.config_dir
        if args.backend is not None:
            env_kwargs["backend"] = args.backend
        if args.n_frames is not None:
            env_kwargs["n_frames"] = args.n_frames
        if args.compact_robot_obs and env_id in ("ant_goal_headless", "humanoid_goal_headless"):
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
