"""Safe navigation environment factory."""

from __future__ import annotations

from safenav_jax.envs.ant_goal import AntGoal
from safenav_jax.envs.ant_goal_grid import AntGoalGrid
from safenav_jax.envs.ant_goal_grid_headless import AntGoalGridHeadless
from safenav_jax.envs.ant_goal_headless import AntGoalHeadless
from safenav_jax.envs.car_goal import CarGoal
from safenav_jax.envs.car_goal_headless import CarGoalHeadless
from safenav_jax.envs.humanoid_goal import HumanoidGoal
from safenav_jax.envs.humanoid_goal_grid import HumanoidGoalGrid
from safenav_jax.envs.humanoid_goal_grid_headless import HumanoidGoalGridHeadless
from safenav_jax.envs.humanoid_goal_headless import HumanoidGoalHeadless
from safenav_jax.envs.point_goal import PointGoal
from safenav_jax.envs.point_goal_headless import PointGoalHeadless
from safenav_jax.envs.point_push import PointPush
from safenav_jax.envs.point_push_headless import PointPushHeadless
from safenav_jax.env_config import load_env_config


_ENV_REGISTRY = {
    "ant_goal": AntGoal,
    "ant_goal_grid": AntGoalGrid,
    "ant_goal_grid_headless": AntGoalGridHeadless,
    "ant_goal_headless": AntGoalHeadless,
    "car_goal": CarGoal,
    "car_goal_headless": CarGoalHeadless,
    "humanoid_goal": HumanoidGoal,
    "humanoid_goal_grid": HumanoidGoalGrid,
    "humanoid_goal_grid_headless": HumanoidGoalGridHeadless,
    "humanoid_goal_headless": HumanoidGoalHeadless,
    "point_goal": PointGoal,
    "point_goal_headless": PointGoalHeadless,
    "point_push": PointPush,
    "point_push_headless": PointPushHeadless,
}


def is_supported_env_id(env_id: str) -> bool:
    return env_id in _ENV_REGISTRY


def config_env_id_for(env_id: str) -> str:
    return env_id


def available_env_ids() -> tuple[str, ...]:
    return tuple(sorted(_ENV_REGISTRY))


def make(
    env_id: str,
    *,
    use_config: bool = False,
    config_dir: str | None = None,
    config_path: str | None = None,
    **kwargs,
):
    """Creates a safe navigation environment by id."""
    try:
        env_cls = _ENV_REGISTRY[env_id]
    except KeyError as exc:
        available = ", ".join(sorted(_ENV_REGISTRY))
        raise ValueError(
            f"Unknown env_id {env_id!r}. Available envs: {available}."
        ) from exc

    if use_config or config_dir is not None or config_path is not None:
        env_kwargs = load_env_config(env_id, config_dir=config_dir, config_path=config_path)
        env_kwargs.update(kwargs)
    else:
        env_kwargs = kwargs
    return env_cls(**env_kwargs)


__all__ = [
    "AntGoal",
    "AntGoalGrid",
    "AntGoalGridHeadless",
    "AntGoalHeadless",
    "CarGoal",
    "CarGoalHeadless",
    "HumanoidGoal",
    "HumanoidGoalGrid",
    "HumanoidGoalGridHeadless",
    "HumanoidGoalHeadless",
    "PointGoal",
    "PointGoalHeadless",
    "PointPush",
    "PointPushHeadless",
    "available_env_ids",
    "config_env_id_for",
    "is_supported_env_id",
    "make",
]
