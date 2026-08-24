"""Top-down short-rollout visualizations for headless safe navigation envs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp

from safenav_jax.envs import make
from safenav_jax.visualize_headless_initial_states import DEFAULT_ENV_IDS
from safenav_jax.visualize_headless_initial_states import _maze_bound
from safenav_jax.visualize_headless_initial_states import _project
from safenav_jax.visualize_headless_initial_states import _svg_agent
from safenav_jax.visualize_headless_initial_states import _svg_circle
from safenav_jax.visualize_headless_initial_states import _svg_square
from safenav_jax.visualization_rollout import jitted_rollout_states
from safenav_jax.visualization_rollout import python_rollout_states
from safenav_jax.visualization_rollout import resolve_visual_env_id
from safenav_jax.visualization_rollout import selected_frame_steps


def _svg_polyline(points, bound: float, canvas_size: int, pad: int, style: str) -> str:
    projected = [_project(point, bound, canvas_size, pad) for point in points]
    serialized = " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
    return f'<polyline points="{serialized}" style="{style}" />'


def _render_svg(
    env_id: str,
    env,
    states,
    action_mode: str = "zero",
    canvas_size: int = 720,
    pad: int = 48,
) -> str:
    if hasattr(env, "inner_wall_centers") and hasattr(env, "_maze_size_scaling"):
        return _render_maze_svg(
            env_id,
            env,
            states,
            action_mode=action_mode,
            canvas_size=canvas_size,
            pad=pad,
        )

    bound = float(env._playground_size)
    first = states[0]
    last = states[-1]
    goal_xy = last.info["goal_xy"]
    hazards_xy = first.info["hazards_xy"]
    obstacles_xy = first.info["obstacles_xy"]
    gremlin_centers_xy = first.info["gremlin_centers_xy"]
    gremlin_paths = [
        [state.info["gremlins_xy"][i] for state in states]
        for i in range(int(env._num_gremlins))
    ]
    agent_path = [env._success_xy(state.pipeline_state) for state in states]

    x0, y0 = _project((-bound, bound), bound, canvas_size, pad)
    x1, y1 = _project((bound, -bound), bound, canvas_size, pad)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" viewBox="0 0 {canvas_size} {canvas_size}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{pad}" y="28" font-family="sans-serif" font-size="18" fill="black">{env_id}</text>',
        f'<text x="{pad + 260}" y="28" font-family="sans-serif" font-size="13" fill="black">{action_mode.title()}-action rollout, top-down</text>',
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" fill="none" stroke="black" stroke-width="2" />',
    ]

    for hazard in hazards_xy:
        parts.append(_svg_circle(hazard, env._hazard_radius, bound, canvas_size, pad, "fill:rgb(31,119,180);fill-opacity:0.20;stroke:rgb(31,119,180);stroke-opacity:0.65"))
    for obstacle in obstacles_xy:
        parts.append(_svg_circle(obstacle, env._obstacle_radius, bound, canvas_size, pad, "fill:rgb(128,82,35);fill-opacity:0.80;stroke:rgb(80,50,20);stroke-opacity:0.8"))
    for center in gremlin_centers_xy:
        parts.append(_svg_circle(center, env._gremlin_travel + env._gremlin_radius, bound, canvas_size, pad, "fill:none;stroke:purple;stroke-width:1.2;stroke-opacity:0.5"))
    for path in gremlin_paths:
        parts.append(_svg_polyline(path, bound, canvas_size, pad, "fill:none;stroke:purple;stroke-width:1.4;stroke-opacity:0.65"))
        parts.append(_svg_circle(path[-1], env._gremlin_radius, bound, canvas_size, pad, "fill:purple;fill-opacity:0.85;stroke:purple"))

    parts.append(_svg_circle(goal_xy, env._goal_radius, bound, canvas_size, pad, "fill:rgb(44,160,44);fill-opacity:0.35;stroke:rgb(44,160,44);stroke-width:2"))
    parts.append(_svg_polyline(agent_path, bound, canvas_size, pad, "fill:none;stroke:rgb(214,39,40);stroke-width:2.5;stroke-opacity:0.85"))
    parts.append(_svg_agent(agent_path[-1], bound, canvas_size, pad))
    parts.append("</svg>")
    return "\n".join(parts)


def _render_maze_svg(
    env_id: str,
    env,
    states,
    action_mode: str = "zero",
    canvas_size: int = 720,
    pad: int = 48,
) -> str:
    bound = max(_maze_bound(env, state) for state in states)
    scale = float(env._maze_size_scaling)
    first = states[0]
    last = states[-1]
    goal_xy = jp.asarray([last.metrics["goal_x"], last.metrics["goal_y"]])
    agent_path = [state.pipeline_state.x.pos[0, :2] for state in states]

    x0, y0 = _project((-bound, bound), bound, canvas_size, pad)
    x1, y1 = _project((bound, -bound), bound, canvas_size, pad)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" viewBox="0 0 {canvas_size} {canvas_size}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{pad}" y="28" font-family="sans-serif" font-size="18" fill="black">{env_id}</text>',
        f'<text x="{pad + 300}" y="28" font-family="sans-serif" font-size="13" fill="black">{action_mode.title()}-action rollout; red cells = ghost cost cells</text>',
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" fill="none" stroke="black" stroke-width="2" />',
    ]
    if hasattr(env, "possible_goals"):
        for goal in jp.asarray(env.possible_goals).reshape((-1, 2)):
            parts.append(_svg_circle(goal, 0.06 * scale, bound, canvas_size, pad, "fill:rgb(44,160,44);fill-opacity:0.18;stroke:none"))
    if hasattr(env, "inner_wall_centers"):
        for wall in jp.asarray(env.inner_wall_centers).reshape((-1, 2)):
            parts.append(_svg_square(wall, scale, bound, canvas_size, pad, "fill:rgb(214,39,40);fill-opacity:0.25;stroke:rgb(160,0,0);stroke-opacity:0.7"))

    parts.append(_svg_circle(goal_xy, env._goal_radius, bound, canvas_size, pad, "fill:rgb(44,160,44);fill-opacity:0.45;stroke:rgb(44,160,44);stroke-width:2"))
    parts.append(_svg_polyline(agent_path, bound, canvas_size, pad, "fill:none;stroke:rgb(214,39,40);stroke-width:2.5;stroke-opacity:0.85"))
    parts.append(_svg_agent(agent_path[-1], bound, canvas_size, pad))
    parts.append("</svg>")
    return "\n".join(parts)


def save_headless_short_rollout_visualizations(
    output_dir: str | Path,
    seed: int = 0,
    env_ids: tuple[str, ...] = DEFAULT_ENV_IDS,
    rollout_steps: int = 32,
    config_dir: str | Path | None = None,
    jit_rollout: bool = False,
    frame_stride: int = 1,
    max_render_frames: int = 0,
    auto_reset_on_done: bool = True,
    action_mode: str = "zero",
    relocate_objects_on_reset: bool | None = None,
    fixed_object_layout_seed: int | None = None,
) -> list[Path]:
    """Saves top-down SVG visualizations of short headless rollouts."""
    frame_steps = selected_frame_steps(rollout_steps, frame_stride, max_render_frames)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_files: list[Path] = []
    for offset, env_id in enumerate(env_ids):
        headless_env_id = resolve_visual_env_id(env_id, headless=True)
        env_kwargs = {}
        if relocate_objects_on_reset is not None:
            env_kwargs["relocate_objects_on_reset"] = relocate_objects_on_reset
        if fixed_object_layout_seed is not None:
            env_kwargs["fixed_object_layout_seed"] = fixed_object_layout_seed
        env = make(
            headless_env_id,
            use_config=True,
            config_dir=str(config_dir) if config_dir is not None else None,
            **env_kwargs,
        )
        reset_rng, rollout_rng = jax.random.split(jax.random.PRNGKey(seed + offset))
        state = env.reset(reset_rng)
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

        target = output_path / f"{headless_env_id}_short_rollout.svg"
        target.write_text(_render_svg(headless_env_id, env, states, action_mode=action_mode), encoding="utf-8")
        written_files.append(target)

    return written_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save top-down SVG visualizations of short headless safe-navigation rollouts.",
    )
    parser.add_argument(
        "--output-dir",
        default="safenav_jax/visualizations/headless_short_rollouts",
        help="Directory where SVG files will be written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed used for env.reset().")
    parser.add_argument("--steps", type=int, default=32, help="Number of rollout steps after reset.")
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
        help="Optional env id to visualize. Repeat to render multiple envs.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    env_ids = tuple(args.env_ids) if args.env_ids else DEFAULT_ENV_IDS
    for path in save_headless_short_rollout_visualizations(
        args.output_dir,
        args.seed,
        env_ids,
        args.steps,
        args.config_dir,
        args.jit_rollout,
        args.frame_stride,
        args.max_render_frames,
        not args.no_auto_reset_on_done,
        args.action_mode,
        args.relocate_objects_on_reset,
        args.fixed_object_layout_seed,
    ):
        print(path)


if __name__ == "__main__":
    main()
