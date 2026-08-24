"""Renderable ant grid-world goal environment."""

from __future__ import annotations

from safenav_jax.envs.ant_goal_grid_headless import AntGoalGridHeadless
from safenav_jax.envs.goal_grid_maze_base import AntGoalGridBase


class AntGoalGrid(AntGoalGridBase):
    """Renderable ant goal-grid env built with scaling-maze-style XML."""

    def __init__(self, **kwargs):
        AntGoalGridHeadless._discard_legacy_kwargs(kwargs)
        kwargs.setdefault("grid_cell_size", 4.0)
        kwargs.setdefault("grid_layout_name", "default")
        kwargs.setdefault("eval_grid_layout_name", "default_eval")
        kwargs.setdefault("evaluation_mode", False)
        kwargs.setdefault("min_goal_cell_distance", 2.0)
        kwargs.setdefault("hazard_radius", 0.75)
        kwargs.setdefault("goal_radius", 0.5)
        kwargs.setdefault("hazard_height", 1.0)
        kwargs.setdefault("backend", "mjx")
        kwargs.setdefault("n_frames", 10)
        kwargs.setdefault("episode_length", 1000)
        kwargs.setdefault("reset_noise_scale", 0.1)
        kwargs.setdefault("healthy_z_range", (0.2, 1.0))
        kwargs.setdefault("robot_cost_margin", 0.10)
        kwargs.setdefault("object_boundary", True)
        kwargs.setdefault("terminate_on_cost", False)
        kwargs.setdefault("cost_limit_max", 25.0)
        kwargs.setdefault("relocate_objects_on_reset", True)
        kwargs.setdefault("fixed_object_layout_seed", 0)
        kwargs.setdefault("different_object_layout_per_env", False)
        kwargs.setdefault("goal_respawn_on_success", True)
        kwargs.setdefault("fixed_goal_on_reset", False)
        kwargs.setdefault("fixed_agent_on_reset", False)
        kwargs.setdefault("layout_lidar_num_bins", 16)
        kwargs.setdefault("layout_lidar_max_dist", None)
        kwargs.setdefault("full_robot_observation", True)
        super().__init__(render_hazards=True, **kwargs)
