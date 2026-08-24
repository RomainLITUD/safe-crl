"""Brax HTML visualizations for renderable safe-navigation envs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp
from brax.io import html
from brax.io import json as brax_json

from safenav_jax.envs import make
from safenav_jax.visualization_rollout import force_square_html
from safenav_jax.visualization_rollout import resolve_visual_env_id
from safenav_jax.visualization_rollout import sync_mocap_pipeline_state_for_render


DEFAULT_ENV_IDS = (
    "point_goal",
    "car_goal",
    "point_push",
    "ant_goal_grid",
    "humanoid_goal_grid",
)


def assert_visual_state_matches_info(env, state) -> None:
    """Checks that renderable MJCF visual joints mirror canonical task state."""
    if hasattr(env, "_visual_q_start") and hasattr(env, "_num_hazards"):
        layout_count = env._num_hazards
        if layout_count:
            layout_q = state.pipeline_state.q[
                env._visual_q_start : env._visual_q_start + 2 * layout_count
            ].reshape((layout_count, 2))
            if not bool(jp.allclose(layout_q, state.info["hazards_xy"])):
                raise ValueError("Renderable hazard q positions do not match state.info['hazards_xy'].")
        return
    if not hasattr(env, "_visual_target_q_idx"):
        return

    goal_q = state.pipeline_state.q[env._visual_target_q_idx : env._visual_target_q_idx + 2]
    if not bool(jp.allclose(goal_q, state.info["goal_xy"])):
        raise ValueError("Renderable target q does not match state.info['goal_xy'].")

    layout_count = env._num_hazards + env._num_obstacles + env._num_gremlins
    layout_q = state.pipeline_state.q[
        env._visual_layout_q_idx : env._visual_layout_q_idx + 2 * layout_count
    ].reshape((layout_count, 2))
    hazards_xy = layout_q[: env._num_hazards]
    obstacles_xy = layout_q[env._num_hazards : env._num_hazards + env._num_obstacles]
    gremlins_xy = layout_q[env._num_hazards + env._num_obstacles :]

    if not bool(jp.allclose(hazards_xy, state.info["hazards_xy"])):
        raise ValueError("Renderable hazard q positions do not match state.info['hazards_xy'].")
    if not bool(jp.allclose(obstacles_xy, state.info["obstacles_xy"])):
        raise ValueError("Renderable obstacle q positions do not match state.info['obstacles_xy'].")
    if not bool(jp.allclose(gremlins_xy, state.info["gremlins_xy"])):
        raise ValueError("Renderable gremlin q positions do not match state.info['gremlins_xy'].")


def _patch_brax_json_render_bug() -> None:
    """Applies the local workaround for the known brax.io.json `jp` bug."""
    if not hasattr(brax_json, "jp"):
        brax_json.jp = jp


def save_initial_state_visualizations(
    output_dir: str | Path,
    seed: int = 0,
    env_ids: tuple[str, ...] = DEFAULT_ENV_IDS,
    height: int = 480,
    square_size: int = 0,
    check_visual_sync: bool = True,
    config_dir: str | Path | None = None,
    relocate_objects_on_reset: bool | None = None,
    fixed_object_layout_seed: int | None = None,
) -> list[Path]:
    """Saves one Brax HTML visualization per renderable environment initial state."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _patch_brax_json_render_bug()

    written_files: list[Path] = []
    for offset, env_id in enumerate(env_ids):
        render_env_id = resolve_visual_env_id(env_id, headless=False)
        env_kwargs = {}
        if relocate_objects_on_reset is not None:
            env_kwargs["relocate_objects_on_reset"] = relocate_objects_on_reset
        if fixed_object_layout_seed is not None:
            env_kwargs["fixed_object_layout_seed"] = fixed_object_layout_seed
        env = make(
            render_env_id,
            use_config=True,
            config_dir=str(config_dir) if config_dir is not None else None,
            **env_kwargs,
        )
        rng = jax.random.PRNGKey(seed + offset)
        state = env.reset(rng)
        if check_visual_sync:
            assert_visual_state_matches_info(env, state)
        render_height = square_size if square_size > 0 else height
        render_state = sync_mocap_pipeline_state_for_render(env, state.pipeline_state)
        html_string = html.render(
            env.sys,
            [render_state],
            height=render_height,
            colab=False,
        )
        if square_size > 0:
            html_string = force_square_html(html_string, square_size)
        target = output_path / f"{render_env_id}_initial_state.html"
        target.write_text(html_string, encoding="utf-8")
        written_files.append(target)

    return written_files


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Save Brax HTML visualizations of renderable safe-navigation initial states.",
    )
    parser.add_argument(
        "--output-dir",
        default="safenav_jax/visualizations/initial_states",
        help="Directory where the HTML files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used for env.reset().",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="HTML canvas height passed to brax.io.html.render.",
    )
    parser.add_argument(
        "--square-size",
        type=int,
        default=0,
        help="If positive, force a square HTML/canvas viewport of this size in pixels.",
    )
    parser.add_argument("--config-dir", default=None, help="Directory containing env_id.yaml config files.")
    parser.add_argument(
        "--relocate-objects-on-reset",
        dest="relocate_objects_on_reset",
        action="store_true",
        default=None,
        help="Override config so hazards, obstacles, and gremlins are resampled on reset.",
    )
    parser.add_argument(
        "--no-relocate-objects-on-reset",
        dest="relocate_objects_on_reset",
        action="store_false",
        help="Override config so hazards, obstacles, and gremlins use a fixed reset layout.",
    )
    parser.add_argument(
        "--fixed-object-layout-seed",
        type=int,
        default=None,
        help="Seed for the fixed object layout when object relocation is disabled.",
    )
    parser.add_argument(
        "--env-id",
        action="append",
        dest="env_ids",
        help="Optional env id to visualize. Repeat to render multiple envs. Defaults to all finalized env groups.",
    )
    parser.add_argument(
        "--no-check-visual-sync",
        action="store_true",
        help="Skip validation that MJCF visual joints match state.info task arrays before rendering.",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args()

    env_ids = tuple(args.env_ids) if args.env_ids else DEFAULT_ENV_IDS
    written_files = save_initial_state_visualizations(
        output_dir=args.output_dir,
        seed=args.seed,
        env_ids=env_ids,
        height=args.height,
        square_size=args.square_size,
        check_visual_sync=not args.no_check_visual_sync,
        config_dir=args.config_dir,
        relocate_objects_on_reset=args.relocate_objects_on_reset,
        fixed_object_layout_seed=args.fixed_object_layout_seed,
    )
    for path in written_files:
        print(path)


if __name__ == "__main__":
    main()
