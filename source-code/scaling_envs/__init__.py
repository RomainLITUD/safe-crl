"""Cost-free renderable environments for Scaling-CRL methods only."""

from __future__ import annotations

from scaling_envs.ant_goal import AntGoal
from scaling_envs.ant_maze import CrlAntMaze, SUPPORTED_MAZE_LAYOUTS as ANT_MAZE_LAYOUTS
from scaling_envs.env_config import load_env_config
from scaling_envs.humanoid_goal import HumanoidGoal
from scaling_envs.humanoid_maze import (
    CrlHumanoidMaze,
    CrlHumanoidMazeNoWall,
    SUPPORTED_MAZE_LAYOUTS as HUMANOID_MAZE_LAYOUTS,
)


_ENV_REGISTRY = {
    "scaling_ant_goal": AntGoal,
    "scaling_humanoid_goal": HumanoidGoal,
    "crl_ant_maze": CrlAntMaze,
    "crl_humanoid_maze": CrlHumanoidMaze,
    "crl_humanoid_maze_nowall": CrlHumanoidMazeNoWall,
}


def _parse_maze_env_id(env_id: str):
    if env_id.endswith("_headless"):
        return None
    if env_id == "crl_ant_maze":
        return CrlAntMaze, "crl_ant_maze", "u_maze"
    if env_id.startswith("crl_ant_"):
        layout_name = env_id[len("crl_ant_") :]
        if layout_name in ANT_MAZE_LAYOUTS:
            return CrlAntMaze, "crl_ant_maze", layout_name
    if env_id == "crl_humanoid_maze":
        return CrlHumanoidMaze, "crl_humanoid_maze", "u_maze"
    if env_id == "crl_humanoid_maze_nowall":
        return CrlHumanoidMazeNoWall, "crl_humanoid_maze_nowall", "u_maze"
    if env_id.startswith("crl_humanoid_") and env_id.endswith("_nowall"):
        layout_name = env_id[len("crl_humanoid_") : -len("_nowall")]
        if layout_name in HUMANOID_MAZE_LAYOUTS:
            return CrlHumanoidMazeNoWall, "crl_humanoid_maze_nowall", layout_name
    if env_id.startswith("crl_humanoid_"):
        layout_name = env_id[len("crl_humanoid_") :]
        if layout_name in HUMANOID_MAZE_LAYOUTS:
            return CrlHumanoidMaze, "crl_humanoid_maze", layout_name
    return None


def is_maze_env_id(env_id: str) -> bool:
    return _parse_maze_env_id(env_id) is not None


def is_supported_env_id(env_id: str) -> bool:
    return env_id in _ENV_REGISTRY or is_maze_env_id(env_id)


def config_env_id_for(env_id: str) -> str:
    parsed = _parse_maze_env_id(env_id)
    return parsed[1] if parsed is not None else env_id


def available_env_ids() -> tuple[str, ...]:
    return tuple(sorted(_ENV_REGISTRY))


def eval_env_id(env_id: str) -> str:
    parsed = _parse_maze_env_id(env_id)
    if parsed is None:
        return env_id
    env_cls, _, layout_name = parsed
    supported = ANT_MAZE_LAYOUTS if env_cls is CrlAntMaze else HUMANOID_MAZE_LAYOUTS
    eval_layout = (
        layout_name
        if layout_name == "hardest_maze" or "eval" in layout_name
        else f"{layout_name}_eval"
    )
    if eval_layout not in supported:
        raise ValueError(
            f"Maze env_id {env_id!r} has no corresponding eval layout {eval_layout!r}."
        )
    prefix = "crl_ant" if env_cls is CrlAntMaze else "crl_humanoid"
    suffix = "_nowall" if env_cls is CrlHumanoidMazeNoWall else ""
    return f"{prefix}_{eval_layout}{suffix}"


def make(
    env_id: str,
    *,
    use_config: bool = False,
    config_dir: str | None = None,
    config_path: str | None = None,
    **kwargs,
):
    maze = _parse_maze_env_id(env_id)
    if maze is None:
        try:
            env_cls = _ENV_REGISTRY[env_id]
            config_env_id = env_id
            maze_layout_name = None
        except KeyError as exc:
            raise ValueError(
                f"Unknown scaling environment {env_id!r}. "
                f"Available base environments: {', '.join(available_env_ids())}."
            ) from exc
    else:
        env_cls, config_env_id, maze_layout_name = maze

    env_kwargs = {}
    if use_config or config_dir is not None or config_path is not None:
        env_kwargs.update(
            load_env_config(
                config_env_id,
                config_dir=config_dir,
                config_path=config_path,
            )
        )
    env_kwargs.update(kwargs)
    if maze_layout_name is not None:
        env_kwargs["maze_layout_name"] = maze_layout_name
    return env_cls(**env_kwargs)


__all__ = [
    "AntGoal",
    "CrlAntMaze",
    "CrlHumanoidMaze",
    "CrlHumanoidMazeNoWall",
    "HumanoidGoal",
    "available_env_ids",
    "config_env_id_for",
    "eval_env_id",
    "is_maze_env_id",
    "is_supported_env_id",
    "make",
]
