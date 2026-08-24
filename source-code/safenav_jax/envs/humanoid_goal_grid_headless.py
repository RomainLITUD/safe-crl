"""Headless humanoid grid-world goal environment."""

from __future__ import annotations

from safenav_jax.envs.goal_grid_maze_base import HumanoidGoalGridBase


class HumanoidGoalGridHeadless(HumanoidGoalGridBase):
    """Headless humanoid goal-grid env built with scaling-maze-style XML."""

    def __init__(
        self,
        grid_cell_size: float = 2.0,
        grid_layout_name: str = "default",
        eval_grid_layout_name: str = "default_eval",
        evaluation_mode: bool = False,
        min_goal_cell_distance: float = 2.0,
        hazard_radius: float = 0.35,
        goal_radius: float = 0.5,
        hazard_height: float = 2.0,
        backend: str = "mjx",
        n_frames: int | None = 10,
        episode_length: int = 1000,
        reset_noise_scale: float = 0.0,
        healthy_z_range: tuple[float, float] = (1.0, 2.0),
        robot_cost_margin: float = 0.05,
        object_boundary: bool = True,
        terminate_on_cost: bool = False,
        cost_limit_max: float = 25.0,
        relocate_objects_on_reset: bool = True,
        fixed_object_layout_seed: int = 0,
        different_object_layout_per_env: bool = False,
        goal_respawn_on_success: bool = True,
        fixed_goal_on_reset: bool = False,
        fixed_agent_on_reset: bool = False,
        layout_lidar_num_bins: int = 16,
        layout_lidar_max_dist: float | None = None,
        full_robot_observation: bool = True,
        humanoid_use_spring_gear: bool = False,
        **kwargs,
    ):
        self._discard_legacy_kwargs(kwargs)
        super().__init__(
            render_hazards=False,
            grid_cell_size=grid_cell_size,
            grid_layout_name=grid_layout_name,
            eval_grid_layout_name=eval_grid_layout_name,
            evaluation_mode=evaluation_mode,
            min_goal_cell_distance=min_goal_cell_distance,
            hazard_radius=hazard_radius,
            goal_radius=goal_radius,
            hazard_height=hazard_height,
            backend=backend,
            n_frames=n_frames,
            episode_length=episode_length,
            reset_noise_scale=reset_noise_scale,
            healthy_z_range=healthy_z_range,
            robot_cost_margin=robot_cost_margin,
            object_boundary=object_boundary,
            terminate_on_cost=terminate_on_cost,
            cost_limit_max=cost_limit_max,
            relocate_objects_on_reset=relocate_objects_on_reset,
            fixed_object_layout_seed=fixed_object_layout_seed,
            different_object_layout_per_env=different_object_layout_per_env,
            goal_respawn_on_success=goal_respawn_on_success,
            fixed_goal_on_reset=fixed_goal_on_reset,
            fixed_agent_on_reset=fixed_agent_on_reset,
            layout_lidar_num_bins=layout_lidar_num_bins,
            layout_lidar_max_dist=layout_lidar_max_dist,
            full_robot_observation=full_robot_observation,
            humanoid_use_spring_gear=humanoid_use_spring_gear,
        )

    @staticmethod
    def _discard_legacy_kwargs(kwargs: dict) -> None:
        for key in (
            "num_obstacles",
            "num_gremlins",
            "obstacle_radius",
            "gremlin_radius",
            "gremlin_travel",
            "gremlin_speed",
            "obstacle_keepout",
            "gremlin_keepout",
            "goal_z",
            "include_object_layout_obs",
            "include_object_type_obs",
            "layout_lidar_ego_frame",
            "layout_margin",
            "wall_height",
            "wall_thickness",
            "grid_num_cells",
            "num_hazards",
        ):
            kwargs.pop(key, None)
