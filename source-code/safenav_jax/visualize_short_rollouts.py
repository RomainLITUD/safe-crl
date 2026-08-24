"""Brax HTML visualizations for short renderable safe-navigation rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp
from brax.io import html

from safenav_jax.envs import make
from safenav_jax.visualize_initial_states import DEFAULT_ENV_IDS
from safenav_jax.visualize_initial_states import assert_visual_state_matches_info
from safenav_jax.visualize_initial_states import _patch_brax_json_render_bug
from safenav_jax.visualization_rollout import force_square_html
from safenav_jax.visualization_rollout import jitted_rollout_states
from safenav_jax.visualization_rollout import python_rollout_states
from safenav_jax.visualization_rollout import resolve_visual_env_id
from safenav_jax.visualization_rollout import selected_frame_steps


def save_short_rollout_visualizations(
    output_dir: str | Path,
    seed: int = 0,
    env_ids: tuple[str, ...] = DEFAULT_ENV_IDS,
    rollout_steps: int = 12,
    height: int = 480,
    square_size: int = 0,
    check_visual_sync: bool = True,
    config_dir: str | Path | None = None,
    jit_rollout: bool = False,
    frame_stride: int = 1,
    max_render_frames: int = 0,
    auto_reset_on_done: bool = True,
    action_mode: str = "zero",
    relocate_objects_on_reset: bool | None = None,
    fixed_object_layout_seed: int | None = None,
) -> list[Path]:
    """Saves one short rollout visualization per renderable environment."""
    frame_steps = selected_frame_steps(rollout_steps, frame_stride, max_render_frames)

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
        reset_rng, rollout_rng = jax.random.split(jax.random.PRNGKey(seed + offset))
        state = env.reset(reset_rng)
        if check_visual_sync:
            assert_visual_state_matches_info(env, state)
        action = jp.zeros(env.action_size)
        if jit_rollout:
            states = jitted_rollout_states(
                env,
                state,
                action,
                frame_steps,
                auto_reset_on_done,
                action_mode,
                rollout_rng,
            )
        else:
            states = python_rollout_states(
                env,
                state,
                action,
                frame_steps,
                auto_reset_on_done,
                action_mode,
                rollout_rng,
            )
        pipeline_states = []
        for state in states:
            if check_visual_sync:
                assert_visual_state_matches_info(env, state)
            pipeline_states.append(state.pipeline_state)

        render_height = square_size if square_size > 0 else height
        html_string = html.render(
            env.sys,
            pipeline_states,
            height=render_height,
            colab=False,
        )
        if square_size > 0:
            html_string = force_square_html(html_string, square_size)
        target = output_path / f"{render_env_id}_short_rollout.html"
        target.write_text(html_string, encoding="utf-8")
        written_files.append(target)

    return written_files


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Save Brax HTML visualizations of short renderable safe-navigation rollouts.",
    )
    parser.add_argument(
        "--output-dir",
        default="safenav_jax/visualizations/short_rollouts",
        help="Directory where the HTML files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used for env.reset().",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=12,
        help="Number of rollout steps to include after the initial frame.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("zero", "random"),
        default="zero",
        help="Action source for visualization rollouts.",
    )
    parser.add_argument(
        "--jit-rollout",
        action="store_true",
        help="Generate rollout states with jax.jit and jax.lax.scan/fori_loop.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Render every Nth rollout frame. The final frame is always included.",
    )
    parser.add_argument(
        "--max-render-frames",
        type=int,
        default=0,
        help="Cap rendered frames after stride selection. Use 0 for no cap.",
    )
    parser.add_argument(
        "--no-auto-reset-on-done",
        action="store_true",
        help="Keep stepping terminal states instead of resetting after done/unhealthy.",
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
        help="Optional env id to visualize. Repeat to render multiple envs. Defaults to all current goal envs.",
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
    written_files = save_short_rollout_visualizations(
        output_dir=args.output_dir,
        seed=args.seed,
        env_ids=env_ids,
        rollout_steps=args.steps,
        height=args.height,
        square_size=args.square_size,
        check_visual_sync=not args.no_check_visual_sync,
        config_dir=args.config_dir,
        jit_rollout=args.jit_rollout,
        frame_stride=args.frame_stride,
        max_render_frames=args.max_render_frames,
        auto_reset_on_done=not args.no_auto_reset_on_done,
        action_mode=args.action_mode,
        relocate_objects_on_reset=args.relocate_objects_on_reset,
        fixed_object_layout_seed=args.fixed_object_layout_seed,
    )
    for path in written_files:
        print(path)


if __name__ == "__main__":
    main()
