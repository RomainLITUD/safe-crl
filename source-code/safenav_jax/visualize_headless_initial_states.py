"""Top-down visualizations for headless safe navigation envs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import jax
from jax import numpy as jp

from safenav_jax.envs import make
from safenav_jax.visualization_rollout import resolve_visual_env_id


DEFAULT_ENV_IDS = (
    "ant_goal_headless",
    "humanoid_goal_headless",
    "point_goal_headless",
    "car_goal_headless",
)


def _project(xy, bound: float, canvas_size: int, pad: int) -> tuple[float, float]:
    scale = (canvas_size - 2 * pad) / (2 * bound)
    x = pad + (float(xy[0]) + bound) * scale
    y = canvas_size - (pad + (float(xy[1]) + bound) * scale)
    return x, y


def _svg_circle(xy, radius: float, bound: float, canvas_size: int, pad: int, style: str) -> str:
    x, y = _project(xy, bound, canvas_size, pad)
    scale = (canvas_size - 2 * pad) / (2 * bound)
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius * scale:.2f}" style="{style}" />'


def _svg_agent(xy, bound: float, canvas_size: int, pad: int) -> str:
    x, y = _project(xy, bound, canvas_size, pad)
    size = 13.0
    return "\n".join(
        [
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size + 5:.2f}" style="fill:white;stroke:rgb(180,0,0);stroke-width:3" />',
            f'<path d="M {x:.2f} {y - size:.2f} L {x + size:.2f} {y + size:.2f} L {x - size:.2f} {y + size:.2f} Z" style="fill:rgb(214,39,40);stroke:rgb(120,0,0);stroke-width:2" />',
        ]
    )


def _svg_square(center_xy, side: float, bound: float, canvas_size: int, pad: int, style: str) -> str:
    half = side / 2.0
    x0, y0 = _project((float(center_xy[0]) - half, float(center_xy[1]) + half), bound, canvas_size, pad)
    x1, y1 = _project((float(center_xy[0]) + half, float(center_xy[1]) - half), bound, canvas_size, pad)
    return f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" style="{style}" />'


def _maze_bound(env, state) -> float:
    half = float(env._maze_size_scaling) / 2.0
    points = [state.pipeline_state.x.pos[0, :2]]
    if hasattr(env, "possible_goals"):
        points.append(jp.asarray(env.possible_goals).reshape((-1, 2)))
    if hasattr(env, "inner_wall_centers") and env.inner_wall_centers.shape[0] > 0:
        points.append(jp.asarray(env.inner_wall_centers).reshape((-1, 2)))
    stacked = jp.concatenate([jp.asarray(point).reshape((-1, 2)) for point in points], axis=0)
    return float(jp.max(jp.abs(stacked))) + half + 0.5


def _render_maze_svg(env_id: str, env, state, canvas_size: int = 720, pad: int = 48) -> str:
    bound = _maze_bound(env, state)
    scale = float(env._maze_size_scaling)
    agent_xy = state.pipeline_state.x.pos[0, :2]
    goal_xy = jp.asarray([state.metrics["goal_x"], state.metrics["goal_y"]])

    x0, y0 = _project((-bound, bound), bound, canvas_size, pad)
    x1, y1 = _project((bound, -bound), bound, canvas_size, pad)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" viewBox="0 0 {canvas_size} {canvas_size}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{pad}" y="28" font-family="sans-serif" font-size="18" fill="black">{env_id}</text>',
        f'<text x="{pad + 300}" y="28" font-family="sans-serif" font-size="13" fill="black">Inner red cells = ghost cost cells</text>',
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1 - x0:.2f}" height="{y1 - y0:.2f}" fill="none" stroke="black" stroke-width="2" />',
    ]
    if hasattr(env, "possible_goals"):
        for goal in jp.asarray(env.possible_goals).reshape((-1, 2)):
            parts.append(_svg_circle(goal, 0.06 * scale, bound, canvas_size, pad, "fill:rgb(44,160,44);fill-opacity:0.22;stroke:none"))
    if hasattr(env, "inner_wall_centers"):
        for wall in jp.asarray(env.inner_wall_centers).reshape((-1, 2)):
            parts.append(_svg_square(wall, scale, bound, canvas_size, pad, "fill:rgb(214,39,40);fill-opacity:0.25;stroke:rgb(160,0,0);stroke-opacity:0.7"))
    parts.append(_svg_circle(goal_xy, env._goal_radius, bound, canvas_size, pad, "fill:rgb(44,160,44);fill-opacity:0.45;stroke:rgb(44,160,44);stroke-width:2"))
    parts.append(_svg_agent(agent_xy, bound, canvas_size, pad))
    parts.append("</svg>")
    return "\n".join(parts)


def _render_svg(env_id: str, env, state, canvas_size: int = 720, pad: int = 48) -> str:
    if hasattr(env, "inner_wall_centers") and hasattr(env, "_maze_size_scaling"):
        return _render_maze_svg(env_id, env, state, canvas_size=canvas_size, pad=pad)

    bound = float(env._playground_size)
    agent_xy = state.pipeline_state.x.pos[0, :2]
    goal_xy = state.info["goal_xy"]
    hazards_xy = state.info["hazards_xy"]
    obstacles_xy = state.info["obstacles_xy"]
    gremlin_centers_xy = state.info.get("gremlin_centers_xy", state.info["gremlins_xy"])
    gremlins_xy = state.info["gremlins_xy"]

    x0, y0 = _project((-bound, bound), bound, canvas_size, pad)
    x1, y1 = _project((bound, -bound), bound, canvas_size, pad)
    width = x1 - x0
    height = y1 - y0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_size}" height="{canvas_size}" viewBox="0 0 {canvas_size} {canvas_size}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{pad}" y="28" font-family="sans-serif" font-size="18" fill="black">{env_id}</text>',
        f'<text x="{pad + 260}" y="28" font-family="sans-serif" font-size="13" fill="black">Agent = red triangle, Goal = green circle</text>',
        f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{width:.2f}" height="{height:.2f}" fill="none" stroke="black" stroke-width="2" />',
    ]

    for hazard in hazards_xy:
        parts.append(_svg_circle(hazard, env._hazard_radius, bound, canvas_size, pad, "fill:rgb(31,119,180);fill-opacity:0.25;stroke:rgb(31,119,180);stroke-opacity:0.7"))
    for obstacle in obstacles_xy:
        parts.append(_svg_circle(obstacle, env._obstacle_radius, bound, canvas_size, pad, "fill:rgb(128,82,35);fill-opacity:0.85;stroke:rgb(80,50,20);stroke-opacity:0.8"))
    for center, gremlin in zip(gremlin_centers_xy, gremlins_xy):
        parts.append(_svg_circle(center, env._gremlin_travel + env._gremlin_radius, bound, canvas_size, pad, "fill:none;stroke:purple;stroke-width:1.5;stroke-opacity:0.75"))
        parts.append(_svg_circle(center, 0.025, bound, canvas_size, pad, "fill:purple;fill-opacity:0.65;stroke:none"))
        parts.append(_svg_circle(gremlin, env._gremlin_radius, bound, canvas_size, pad, "fill:purple;fill-opacity:0.85;stroke:purple"))

    parts.append(_svg_circle(goal_xy, env._goal_radius, bound, canvas_size, pad, "fill:rgb(44,160,44);fill-opacity:0.35;stroke:rgb(44,160,44);stroke-width:2"))
    parts.append(_svg_agent(agent_xy, bound, canvas_size, pad))
    parts.append("</svg>")
    return "\n".join(parts)


def save_headless_initial_state_visualizations(
    output_dir: str | Path,
    seed: int = 0,
    env_ids: tuple[str, ...] = DEFAULT_ENV_IDS,
    config_dir: str | Path | None = None,
    relocate_objects_on_reset: bool | None = None,
    fixed_object_layout_seed: int | None = None,
) -> list[Path]:
    """Saves top-down SVG visualizations of headless env initial states."""
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
        state = env.reset(jax.random.PRNGKey(seed + offset))
        target = output_path / f"{headless_env_id}_initial_state.svg"
        target.write_text(_render_svg(headless_env_id, env, state), encoding="utf-8")
        written_files.append(target)

    return written_files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Save top-down SVG visualizations of headless safe navigation env initial states.",
    )
    parser.add_argument(
        "--output-dir",
        default="safenav_jax/visualizations/headless_initial_states",
        help="Directory where SVG files will be written.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed used for env.reset().")
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
    for path in save_headless_initial_state_visualizations(
        args.output_dir,
        args.seed,
        env_ids,
        args.config_dir,
        args.relocate_objects_on_reset,
        args.fixed_object_layout_seed,
    ):
        print(path)


if __name__ == "__main__":
    main()
