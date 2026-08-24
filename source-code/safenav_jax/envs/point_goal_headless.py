"""Headless point goal environment."""

from __future__ import annotations

from pathlib import Path

from brax.io import mjcf

from safenav_jax.envs.headless_goal_base import HeadlessGoalEnv


XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "xmls" / "point.xml"


class PointGoalHeadless(HeadlessGoalEnv):
    """Point goal with task objects represented by arrays."""

    def __init__(
        self,
        backend: str = "mjx",
        n_frames: int = 10,
        episode_length: int = 1000,
        reset_noise_scale: float = 0.0,
        agent_spawn_bound: float | None = None,
        playground_size: float = 2.0,
        goal_wall_margin: float = 0.305,
        min_goal_dist: float = 1.5,
        max_goal_dist: float = 3.0,
        goal_radius: float = 0.3,
        num_hazards: int = 10,
        num_obstacles: int = 10,
        num_gremlins: int = 6,
        hazard_radius: float = 0.2,
        obstacle_radius: float = 0.1,
        obstacle_height: float = 0.1,
        gremlin_radius: float = 0.1,
        gremlin_height: float = 0.1,
        gremlin_travel: float = 0.35,
        gremlin_speed: float = 1.0,
        layout_margin: float = 0.0,
        agent_keepout: float = 0.4,
        goal_keepout: float = 0.305,
        hazard_keepout: float = 0.18,
        obstacle_keepout: float = 0.15,
        gremlin_keepout: float = 0.4,
        include_object_layout_obs: bool = True,
        uniform_initial_goal_sampling: bool = True,
        first_valid_layout_candidate: bool = True,
        ego_view: bool = True,
        parking_mode: bool = False,
        parking_yaw_tolerance_degrees: float = 20.0,
        **kwargs,
    ):
        sys = mjcf.load(XML_PATH)
        super().__init__(
            sys=sys,
            backend=backend,
            n_frames=n_frames,
            episode_length=episode_length,
            reset_noise_scale=reset_noise_scale,
            min_goal_dist=min_goal_dist,
            max_goal_dist=max_goal_dist,
            goal_radius=goal_radius,
            playground_size=playground_size,
            goal_wall_margin=goal_wall_margin,
            num_hazards=num_hazards,
            num_obstacles=num_obstacles,
            num_gremlins=num_gremlins,
            hazard_radius=hazard_radius,
            obstacle_radius=obstacle_radius,
            obstacle_height=obstacle_height,
            gremlin_radius=gremlin_radius,
            gremlin_height=gremlin_height,
            gremlin_travel=gremlin_travel,
            gremlin_speed=gremlin_speed,
            layout_margin=layout_margin,
            agent_keepout=agent_keepout,
            goal_keepout=goal_keepout,
            hazard_keepout=hazard_keepout,
            obstacle_keepout=obstacle_keepout,
            gremlin_keepout=gremlin_keepout,
            random_agent=True,
            agent_spawn_bound=agent_spawn_bound,
            success_link_name="agent",
            include_object_layout_obs=include_object_layout_obs,
            uniform_initial_goal_sampling=uniform_initial_goal_sampling,
            first_valid_layout_candidate=first_valid_layout_candidate,
            robot_sensor_data_start=0 if ego_view else None,
            robot_sensor_data_dim=12,
            layout_lidar_ego_frame=ego_view,
            robot_yaw_q_idx=2,
            parking_mode=parking_mode,
            parking_yaw_tolerance_degrees=parking_yaw_tolerance_degrees,
            initial_goal_require_path_objects=True,
            terminate_on_goal_exit_after_success=True,
            **kwargs,
        )
