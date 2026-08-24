"""Shared implementation for headless goal environments."""

from __future__ import annotations

from brax import actuator
from brax import base
from brax.envs.base import PipelineEnv, State
import jax
from jax import numpy as jp


class HeadlessGoalEnv(PipelineEnv):
    """Goal navigation with ghost task objects stored as JAX arrays."""

    def __init__(
        self,
        sys: base.System,
        backend: str,
        n_frames: int,
        episode_length: int,
        reset_noise_scale: float,
        min_goal_dist: float,
        max_goal_dist: float,
        goal_radius: float,
        playground_size: float,
        goal_wall_margin: float,
        num_hazards: int,
        num_obstacles: int,
        num_gremlins: int,
        hazard_radius: float,
        obstacle_radius: float,
        gremlin_radius: float,
        gremlin_travel: float,
        gremlin_speed: float,
        layout_margin: float,
        random_agent: bool,
        playground_center_xy: tuple[float, float] | None = None,
        agent_keepout: float | None = None,
        goal_keepout: float | None = None,
        hazard_keepout: float | None = None,
        obstacle_keepout: float | None = None,
        gremlin_keepout: float | None = None,
        layout_candidate_count: int = 256,
        obstacle_height: float = 0.25,
        gremlin_height: float = 0.25,
        agent_spawn_bound: float | None = None,
        agent_wall_margin: float | None = None,
        random_yaw: bool = False,
        use_geom_cost: bool = False,
        use_3d_object_cost: bool = False,
        robot_cost_margin: float = 0.0,
        success_link_name: str | None = None,
        healthy_z_range: tuple[float, float] | None = None,
        include_actuator_forces: bool = False,
        relocate_objects_on_reset: bool = True,
        fixed_object_layout_seed: int = 0,
        different_object_layout_per_env: bool = False,
        layout_pool_size: int = 0,
        goal_respawn_on_success: bool = True,
        respawn_goal_require_path_objects: bool = False,
        respawn_goal_min_path_objects: int = 1,
        respawn_goal_path_band_scale: float = 1.0,
        initial_goal_path_objects_mode: bool = False,
        initial_goal_path_objects_probability: float = 0.5,
        initial_goal_require_path_objects: bool = False,
        terminate_on_goal_exit_after_success: bool = False,
        fixed_agent_on_reset: bool = False,
        fixed_agent_xy: tuple[float, float] | None = None,
        fixed_agent_orientation_on_reset: bool = False,
        fixed_goal_on_reset: bool = False,
        include_object_layout_obs: bool = True,
        include_object_type_obs: bool = True,
        layout_lidar_num_bins: int = 16,
        layout_lidar_max_dist: float | None = None,
        goal_z: float | None = None,
        uniform_initial_goal_sampling: bool = False,
        first_valid_layout_candidate: bool = False,
        terminate_on_cost: bool = False,
        cost_limit_max: float = 25.0,
        object_boundary: bool = False,
        robot_sensor_data_start: int | None = None,
        robot_sensor_data_dim: int = 12,
        layout_lidar_ego_frame: bool = False,
        robot_yaw_q_idx: int | None = None,
        robot_quat_q_start: int | None = None,
        parking_mode: bool = False,
        parking_yaw_tolerance_degrees: float = 20.0,
        **kwargs,
    ):
        if use_geom_cost and backend != "mjx":
            raise ValueError("Geometry-based headless costs require backend='mjx'.")
        if robot_cost_margin < 0.0:
            raise ValueError("robot_cost_margin must be non-negative.")
        if parking_yaw_tolerance_degrees < 0.0:
            raise ValueError("parking_yaw_tolerance_degrees must be non-negative.")
        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)
        self._episode_length = episode_length
        self._reset_noise_scale = reset_noise_scale
        self._min_goal_dist = min_goal_dist
        self._max_goal_dist = max_goal_dist
        self._goal_radius = goal_radius
        self._playground_size = playground_size
        self._playground_center_xy = (
            jp.zeros((2,), dtype=jp.float32)
            if playground_center_xy is None
            else jp.asarray(playground_center_xy, dtype=jp.float32)
        )
        self._goal_wall_margin = goal_wall_margin
        self._num_hazards = num_hazards
        self._num_obstacles = num_obstacles
        self._num_gremlins = num_gremlins
        self._hazard_radius = hazard_radius
        self._obstacle_radius = obstacle_radius
        self._gremlin_radius = gremlin_radius
        self._gremlin_travel = gremlin_travel
        self._gremlin_speed = gremlin_speed
        self._obstacle_height = obstacle_height
        self._gremlin_height = gremlin_height
        self._layout_margin = layout_margin
        self._agent_keepout = goal_radius if agent_keepout is None else agent_keepout
        self._goal_keepout = goal_radius if goal_keepout is None else goal_keepout
        self._hazard_keepout = hazard_radius if hazard_keepout is None else hazard_keepout
        self._obstacle_keepout = obstacle_radius if obstacle_keepout is None else obstacle_keepout
        default_gremlin_keepout = gremlin_travel + gremlin_radius
        self._gremlin_keepout = default_gremlin_keepout if gremlin_keepout is None else gremlin_keepout
        self._layout_candidate_count = layout_candidate_count
        self._random_agent = random_agent
        self._agent_spawn_bound = agent_spawn_bound
        self._agent_wall_margin = agent_wall_margin
        self._random_yaw = random_yaw
        self._use_geom_cost = use_geom_cost
        self._use_3d_object_cost = use_3d_object_cost
        self._robot_cost_margin = robot_cost_margin
        self._success_link_idx = self._link_index(success_link_name) if success_link_name is not None else 0
        self._healthy_z_range = healthy_z_range
        self._include_actuator_forces = include_actuator_forces
        self._relocate_objects_on_reset = relocate_objects_on_reset
        self._fixed_object_layout_seed = fixed_object_layout_seed
        self._different_object_layout_per_env = different_object_layout_per_env
        self._layout_pool_size = layout_pool_size
        self._goal_respawn_on_success = goal_respawn_on_success
        if respawn_goal_min_path_objects < 0:
            raise ValueError("respawn_goal_min_path_objects must be non-negative.")
        if respawn_goal_path_band_scale < 0.0:
            raise ValueError("respawn_goal_path_band_scale must be non-negative.")
        if not 0.0 <= initial_goal_path_objects_probability <= 1.0:
            raise ValueError("initial_goal_path_objects_probability must be in [0, 1].")
        self._respawn_goal_require_path_objects = respawn_goal_require_path_objects
        self._respawn_goal_min_path_objects = respawn_goal_min_path_objects
        self._respawn_goal_path_band_scale = respawn_goal_path_band_scale
        self._initial_goal_path_objects_mode = initial_goal_path_objects_mode
        self._initial_goal_path_objects_probability = initial_goal_path_objects_probability
        self._initial_goal_require_path_objects = initial_goal_require_path_objects
        self._terminate_on_goal_exit_after_success = terminate_on_goal_exit_after_success
        self._fixed_agent_on_reset = fixed_agent_on_reset
        self._fixed_agent_xy = None if fixed_agent_xy is None else jp.asarray(fixed_agent_xy, dtype=jp.float32)
        self._fixed_agent_orientation_on_reset = fixed_agent_orientation_on_reset
        self._fixed_goal_on_reset = fixed_goal_on_reset
        self._include_object_layout_obs = include_object_layout_obs
        self._include_object_type_obs = include_object_type_obs
        self._layout_lidar_num_bins = layout_lidar_num_bins
        if self._layout_lidar_num_bins < 1:
            raise ValueError("layout_lidar_num_bins must be at least 1.")
        default_lidar_max_dist = 2.0 * float(2.0**0.5) * playground_size
        self._layout_lidar_max_dist = (
            default_lidar_max_dist if layout_lidar_max_dist is None else layout_lidar_max_dist
        )
        self._parking_mode = bool(parking_mode)
        self._parking_yaw_tolerance = jp.asarray(
            parking_yaw_tolerance_degrees * jp.pi / 180.0,
            dtype=jp.float32,
        )
        self._goal_z = goal_z
        self._goal_dim = 3 if self._parking_mode or goal_z is not None else 2
        self._uniform_initial_goal_sampling = uniform_initial_goal_sampling
        self._first_valid_layout_candidate = first_valid_layout_candidate
        self._terminate_on_cost = terminate_on_cost
        self._cost_limit_max = cost_limit_max
        self._object_boundary = object_boundary
        self._robot_sensor_data_start = None if self._parking_mode else robot_sensor_data_start
        self._robot_sensor_data_dim = robot_sensor_data_dim
        self._layout_lidar_ego_frame = False if self._parking_mode else layout_lidar_ego_frame
        self._robot_yaw_q_idx = robot_yaw_q_idx
        self._robot_quat_q_start = robot_quat_q_start
        self._robot_geom_mask = jp.asarray(self.sys.geom_bodyid != 0)
        self._geom_type = jp.asarray(self.sys.geom_type)
        self._geom_size = jp.asarray(self.sys.geom_size)
        self._geom_rbound = jp.asarray(self.sys.geom_rbound)
        self._robot_sphere_geom_mask = self._robot_geom_mask & (self._geom_type == 2)
        self._robot_capsule_geom_mask = self._robot_geom_mask & (self._geom_type == 3)
        self._geom_radius = self._geom_size[:, 0]
        self._geom_capsule_half_length = self._geom_size[:, 1]

        self._robot_obs_dim = self._robot_obs_size()
        self._layout_is_lidar_obs = include_object_layout_obs
        self._layout_lidar_num_channels = 2 if num_obstacles == 0 else 3
        self._layout_token_dim = 0
        self._layout_num_tokens = 0
        object_layout_dim = (
            self._layout_lidar_num_channels * self._layout_lidar_num_bins
            if include_object_layout_obs
            else 0
        )
        if not include_object_layout_obs:
            self._layout_obs_dim = self._goal_dim
            goal_start = self._robot_obs_dim
        else:
            self._layout_obs_dim = self._goal_dim + object_layout_dim
            goal_start = self._robot_obs_dim + object_layout_dim
        self.state_dim = self._robot_obs_dim + self._layout_obs_dim
        self.goal_indices = jp.arange(goal_start, goal_start + self._goal_dim)
        self.raw_goal_dim = self._goal_dim
        self.relabel_goal_dim = self._goal_dim
        self.scaling_crl_goal_indices = jp.arange(self._goal_dim)

    def reset(self, rng: jax.Array) -> State:
        reset_rng = rng
        if self._parking_mode:
            (
                rng,
                q_rng,
                qd_rng,
                agent_rng,
                yaw_rng,
                goal_rng,
                goal_yaw_rng,
                hazard_rng,
                obstacle_rng,
                gremlin_rng,
                respawn_rng,
            ) = jax.random.split(rng, 11)
        else:
            (
                rng,
                q_rng,
                qd_rng,
                agent_rng,
                yaw_rng,
                goal_rng,
                hazard_rng,
                obstacle_rng,
                gremlin_rng,
                respawn_rng,
            ) = jax.random.split(rng, 10)
            goal_yaw_rng = goal_rng
        q = self.sys.init_q
        qd = jp.zeros(self.sys.qd_size())
        layout_id = self._layout_id_from_rng(reset_rng)

        if self._reset_noise_scale:
            low, hi = -self._reset_noise_scale, self._reset_noise_scale
            q = q + jax.random.uniform(q_rng, (self.sys.q_size(),), minval=low, maxval=hi)
            qd = hi * jax.random.normal(qd_rng, (self.sys.qd_size(),))
        if self._fixed_agent_orientation_on_reset:
            q = q.at[2:7].set(self.sys.init_q[2:7])

        agent_xy, goal_xy, hazards_xy, obstacles_xy, gremlin_centers_xy = self._sample_reset_layout(
            agent_rng,
            goal_rng,
            hazard_rng,
            obstacle_rng,
            gremlin_rng,
            layout_id,
        )
        goal_yaw = self._sample_goal_yaw(goal_yaw_rng, fixed=self._fixed_goal_on_reset)
        q = q.at[:2].set(agent_xy)
        if self._random_yaw:
            yaw = jax.random.uniform(yaw_rng, (), minval=-jp.pi, maxval=jp.pi)
            q = q.at[3:7].set(jp.array([jp.cos(yaw / 2.0), 0.0, 0.0, jp.sin(yaw / 2.0)]))

        pipeline_state = self.pipeline_init(q, qd)
        gremlin_time = getattr(pipeline_state, "time", jp.asarray(0.0, dtype=jp.float32))
        gremlins_xy = self._gremlin_positions(gremlin_centers_xy, gremlin_time)
        obs = self._get_obs(
            pipeline_state,
            jp.zeros(self.sys.act_size()),
            goal_xy,
            goal_yaw,
            hazards_xy,
            obstacles_xy,
            gremlins_xy,
            gremlin_centers_xy,
        )
        zero = jp.array(0.0)
        metrics = self._metrics(
            zero,
            zero,
            zero,
            zero,
            zero,
            self._achieved_goal(pipeline_state),
            self._goal_position(goal_xy, goal_yaw),
            self._agent_yaw(pipeline_state),
        )
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                "steps": jp.array(0),
                "seed": jp.array(0, dtype=jp.int32),
                "rng": respawn_rng,
                "goal_xy": goal_xy,
                "goal_yaw": goal_yaw,
                "hazards_xy": hazards_xy,
                "obstacles_xy": obstacles_xy,
                "gremlin_centers_xy": gremlin_centers_xy,
                "gremlins_xy": gremlins_xy,
                "episode_cost": zero,
                "layout_id": layout_id,
                **(
                    {"goal_reached": jp.array(False)}
                    if self._terminate_on_goal_exit_after_success
                    else {}
                ),
            },
        )

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        success_goal = self._achieved_goal(pipeline_state)
        goal_xy = state.info["goal_xy"]
        goal_yaw = state.info.get("goal_yaw", jp.asarray(0.0, dtype=goal_xy.dtype))
        goal_position = self._goal_position(goal_xy, goal_yaw)
        dist, yaw_error, success = self._goal_success(success_goal, goal_position)
        reward = success
        goal_reached, goal_exit = self._goal_retention_transition(
            state.info.get("goal_reached", jp.array(False)),
            success,
        )
        gremlin_time = getattr(pipeline_state, "time", jp.asarray(0.0, dtype=jp.float32))
        gremlins_xy = self._gremlin_positions(state.info["gremlin_centers_xy"], gremlin_time)
        cost_hazards, cost_obstacles, cost_gremlins = self._costs(
            pipeline_state,
            state.info["hazards_xy"],
            state.info["obstacles_xy"],
            gremlins_xy,
        )
        object_boundary_cost = self._object_boundary_cost(
            pipeline_state,
            state.info["hazards_xy"],
            state.info["obstacles_xy"],
            gremlins_xy,
        )

        if self._parking_mode:
            rng, respawn_rng, respawn_yaw_rng = jax.random.split(state.info["rng"], 3)
        else:
            rng, respawn_rng = jax.random.split(state.info["rng"])
            respawn_yaw_rng = respawn_rng
        if self._goal_respawn_on_success:
            respawn_goal_xy = self._sample_respawn_goal(
                respawn_rng,
                success_goal[:2],
                state.info["hazards_xy"],
                state.info["obstacles_xy"],
                state.info["gremlin_centers_xy"],
            )
            respawn_goal_yaw = self._sample_goal_yaw(respawn_yaw_rng)
            next_goal_xy = jp.where(success.astype(bool), respawn_goal_xy, goal_xy)
            next_goal_yaw = jp.where(success.astype(bool), respawn_goal_yaw, goal_yaw)
            next_goal_reached = jp.array(False)
        else:
            next_goal_xy = goal_xy
            next_goal_yaw = goal_yaw
            next_goal_reached = goal_reached

        wrapped_by_training = "truncation" in state.info
        steps = state.info["steps"] + 1
        done = jp.array(False) if wrapped_by_training else steps >= self._episode_length
        if self._healthy_z_range is not None:
            min_z, max_z = self._healthy_z_range
            z = pipeline_state.x.pos[0, 2]
            done = jp.logical_or(done, jp.logical_or(z < min_z, z > max_z))
        cost = jp.maximum(cost_hazards, jp.maximum(cost_obstacles, cost_gremlins))
        episode_cost = state.info["episode_cost"] + cost
        if self._object_boundary:
            done = jp.logical_or(done, object_boundary_cost.astype(bool))
        if self._terminate_on_cost:
            done = jp.logical_or(done, episode_cost >= self._cost_limit_max)
        done = jp.logical_or(done, goal_exit)
        done = done.astype(float)

        obs = self._get_obs(
            pipeline_state,
            action,
            next_goal_xy,
            next_goal_yaw,
            state.info["hazards_xy"],
            state.info["obstacles_xy"],
            gremlins_xy,
            state.info["gremlin_centers_xy"],
        )
        metrics = self._metrics(
            success,
            dist,
            cost_hazards,
            cost_obstacles,
            cost_gremlins,
            success_goal,
            self._goal_position(next_goal_xy, next_goal_yaw),
            self._agent_yaw(pipeline_state),
            yaw_error,
        )
        seed = state.info["seed"] + jp.where(
            state.info["steps"],
            jp.zeros_like(state.info["seed"]),
            jp.ones_like(state.info["seed"]),
        )
        next_steps = state.info["steps"] if wrapped_by_training else steps
        info = dict(state.info)
        info.update(
            {
                "steps": next_steps,
                "seed": seed,
                "rng": rng,
                "goal_xy": next_goal_xy,
                "goal_yaw": next_goal_yaw,
                "hazards_xy": state.info["hazards_xy"],
                "obstacles_xy": state.info["obstacles_xy"],
                "gremlin_centers_xy": state.info["gremlin_centers_xy"],
                "gremlins_xy": gremlins_xy,
                "episode_cost": episode_cost,
                "layout_id": state.info["layout_id"],
            }
        )
        if self._terminate_on_goal_exit_after_success:
            info["goal_reached"] = next_goal_reached
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _sample_reset_layout(
        self,
        agent_rng: jax.Array,
        goal_rng: jax.Array,
        hazard_rng: jax.Array,
        obstacle_rng: jax.Array,
        gremlin_rng: jax.Array,
        layout_id: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        fixed_objects = not self._relocate_objects_on_reset
        if fixed_objects:
            if self._layout_pool_size > 0:
                layout_rng = self._layout_rng(layout_id)
            else:
                layout_rng = hazard_rng if self._different_object_layout_per_env else None
            hazards_xy, obstacles_xy, gremlin_centers_xy = self._sample_fixed_reset_objects(layout_rng)
            placed_xy, placed_keepout = self._object_centers_and_keepout(
                hazards_xy,
                obstacles_xy,
                gremlin_centers_xy,
            )
        else:
            hazards_xy = jp.zeros((self._num_hazards, 2))
            obstacles_xy = jp.zeros((self._num_obstacles, 2))
            gremlin_centers_xy = jp.zeros((self._num_gremlins, 2))
            placed_xy, placed_keepout = None, None

        if self._relocate_objects_on_reset:
            if self._fixed_agent_on_reset:
                agent_xy = self._sample_agent_xy(self._fixed_agent_rng())
            elif self._fixed_goal_on_reset:
                goal_xy = self._sample_initial_goal(self._fixed_goal_rng(), jp.zeros((2,)))
                goal_placed_xy, goal_placed_keepout = self._append_placed_circle(
                    None,
                    None,
                    goal_xy,
                    self._goal_keepout,
                )
                agent_xy = self._sample_agent_xy(agent_rng, goal_placed_xy, goal_placed_keepout)
            else:
                agent_xy = self._sample_agent_xy(agent_rng)

            if self._fixed_goal_on_reset:
                goal_anchor_xy = agent_xy if self._fixed_agent_on_reset else jp.zeros((2,))
                goal_xy = self._sample_initial_goal(self._fixed_goal_rng(), goal_anchor_xy)
            else:
                goal_xy = self._sample_initial_goal(goal_rng, agent_xy)
        else:
            if self._fixed_agent_on_reset:
                agent_xy = self._sample_agent_xy(self._fixed_agent_rng(), placed_xy, placed_keepout)
            elif self._fixed_goal_on_reset:
                goal_xy = self._sample_initial_goal(self._fixed_goal_rng(), jp.zeros((2,)), placed_xy, placed_keepout)
                goal_placed_xy, goal_placed_keepout = self._append_placed_circle(
                    placed_xy,
                    placed_keepout,
                    goal_xy,
                    self._goal_keepout,
                )
                agent_xy = self._sample_agent_xy(agent_rng, goal_placed_xy, goal_placed_keepout)
            else:
                agent_xy = self._sample_agent_xy(agent_rng, placed_xy, placed_keepout)

            if self._fixed_goal_on_reset:
                if self._fixed_agent_on_reset:
                    goal_xy = self._sample_initial_goal(self._fixed_goal_rng(), agent_xy, placed_xy, placed_keepout)
            else:
                goal_xy = self._sample_initial_goal(goal_rng, agent_xy, placed_xy, placed_keepout)

        if self._relocate_objects_on_reset:
            hazards_xy, obstacles_xy, gremlin_centers_xy = self._sample_reset_objects(
                hazard_rng,
                obstacle_rng,
                gremlin_rng,
                agent_xy,
                goal_xy,
            )
        if (
            (
                self._initial_goal_path_objects_mode
                or self._initial_goal_require_path_objects
            )
            and self._respawn_goal_require_path_objects
            and not self._fixed_goal_on_reset
        ):
            activation_rng, path_goal_rng = jax.random.split(goal_rng)
            path_goal_xy = self._sample_respawn_goal(
                path_goal_rng,
                agent_xy,
                hazards_xy,
                obstacles_xy,
                gremlin_centers_xy,
                require_path_objects=True,
            )
            path_filter_active = jp.asarray(
                True
                if self._initial_goal_require_path_objects
                else jax.random.bernoulli(
                    activation_rng,
                    p=self._initial_goal_path_objects_probability,
                )
            )
            goal_xy = jp.where(path_filter_active, path_goal_xy, goal_xy)
        return agent_xy, goal_xy, hazards_xy, obstacles_xy, gremlin_centers_xy

    def _goal_retention_transition(
        self,
        previously_reached: jax.Array,
        success: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        success_now = success.astype(bool)
        previously_reached = previously_reached.astype(bool)
        reached = jp.logical_or(previously_reached, success_now)
        if (
            self._terminate_on_goal_exit_after_success
            and not self._goal_respawn_on_success
        ):
            exited = jp.logical_and(previously_reached, jp.logical_not(success_now))
        else:
            exited = jp.array(False)
        return reached, exited

    def _layout_id_from_rng(self, rng: jax.Array) -> jax.Array:
        if self._layout_pool_size <= 0:
            return jp.array(-1, dtype=jp.int32)
        return jp.mod(rng[1].astype(jp.int32), jp.asarray(self._layout_pool_size, dtype=jp.int32))

    def _layout_rng(self, layout_id: jax.Array) -> jax.Array:
        base_rng = jax.random.PRNGKey(self._fixed_object_layout_seed)
        return jax.random.fold_in(base_rng, layout_id)

    def _fixed_agent_rng(self) -> jax.Array:
        return jax.random.PRNGKey(self._fixed_object_layout_seed + 1009)

    def _fixed_goal_rng(self) -> jax.Array:
        return jax.random.PRNGKey(self._fixed_object_layout_seed + 2003)

    def _append_placed_circle(
        self,
        placed_xy: jax.Array | None,
        placed_keepout: jax.Array | None,
        circle_xy: jax.Array,
        circle_keepout: float,
    ) -> tuple[jax.Array, jax.Array]:
        circle_xy = circle_xy[None, :]
        circle_keepout = jp.asarray([circle_keepout])
        if placed_xy is None or placed_keepout is None:
            return circle_xy, circle_keepout
        return (
            jp.concatenate([placed_xy, circle_xy], axis=0),
            jp.concatenate([placed_keepout, circle_keepout], axis=0),
        )

    def _sample_agent_xy(
        self,
        rng: jax.Array,
        placed_xy: jax.Array | None = None,
        placed_keepout: jax.Array | None = None,
    ) -> jax.Array:
        if not self._random_agent:
            if self._fixed_agent_xy is not None:
                return self._fixed_agent_xy
            return jp.array([0.0, 0.0])
        if self._agent_spawn_bound is not None:
            bound = self._agent_spawn_bound
            lo = self._playground_center_xy - bound
            hi = self._playground_center_xy + bound
        elif self._agent_wall_margin is None:
            keepout = self._agent_keepout
            lo, hi = self._playground_bounds(keepout)
        else:
            keepout = max(self._agent_wall_margin, self._agent_keepout)
            lo, hi = self._playground_bounds(keepout)
        if placed_xy is None or placed_keepout is None:
            return jax.random.uniform(rng, (2,), minval=lo, maxval=hi)
        candidates = jax.random.uniform(
            rng,
            (self._layout_candidate_count, 2),
            minval=lo,
            maxval=hi,
        )
        keepout_score = self._candidate_keepout_score(
            candidates,
            placed_xy,
            placed_keepout,
            self._agent_keepout,
        )
        return self._select_layout_candidate(candidates, keepout_score)

    def _sample_reset_objects(
        self,
        hazard_rng: jax.Array,
        obstacle_rng: jax.Array,
        gremlin_rng: jax.Array,
        agent_xy: jax.Array,
        goal_xy: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        gremlin_centers_xy = self._sample_gremlins(gremlin_rng, agent_xy, goal_xy)
        hazards_xy = self._sample_hazards(hazard_rng, agent_xy, goal_xy, gremlin_centers_xy)
        obstacles_xy = self._sample_obstacles(
            obstacle_rng,
            agent_xy,
            hazards_xy,
            gremlin_centers_xy,
            goal_xy,
        )
        return self._canonicalize_layout_order(hazards_xy, obstacles_xy, gremlin_centers_xy)

    def _sample_fixed_reset_objects(self, layout_rng: jax.Array | None = None) -> tuple[jax.Array, jax.Array, jax.Array]:
        if layout_rng is None:
            fixed_rng = jax.random.PRNGKey(self._fixed_object_layout_seed)
        else:
            fixed_rng = jax.random.fold_in(layout_rng, self._fixed_object_layout_seed)
        fixed_goal_rng, gremlin_rng, hazard_rng, obstacle_rng = jax.random.split(fixed_rng, 4)
        agent_xy = self._sample_agent_xy(self._fixed_agent_rng())
        goal_xy = self._sample_initial_goal(fixed_goal_rng, agent_xy)
        return self._sample_reset_objects(hazard_rng, obstacle_rng, gremlin_rng, agent_xy, goal_xy)

    def _canonical_xy_order(self, xy: jax.Array) -> jax.Array:
        return jp.lexsort((xy[:, 1], xy[:, 0]))

    def _canonicalize_layout_order(
        self,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        hazards_xy = hazards_xy[self._canonical_xy_order(hazards_xy)]
        obstacles_xy = obstacles_xy[self._canonical_xy_order(obstacles_xy)]
        gremlin_centers_xy = gremlin_centers_xy[self._canonical_xy_order(gremlin_centers_xy)]
        return hazards_xy, obstacles_xy, gremlin_centers_xy

    def _canonicalize_gremlins_by_center(
        self,
        gremlins_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        order = self._canonical_xy_order(gremlin_centers_xy)
        return gremlins_xy[order], gremlin_centers_xy[order]

    def _object_centers_and_keepout(
        self,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        centers_xy = jp.concatenate([hazards_xy, obstacles_xy, gremlin_centers_xy], axis=0)
        keepout = jp.concatenate(
            [
                jp.full((self._num_hazards,), self._hazard_keepout),
                jp.full((self._num_obstacles,), self._obstacle_keepout),
                jp.full((self._num_gremlins,), self._gremlin_keepout),
            ],
            axis=0,
        )
        return centers_xy, keepout

    def _sample_initial_goal(
        self,
        rng: jax.Array,
        agent_xy: jax.Array,
        placed_xy: jax.Array | None = None,
        placed_keepout: jax.Array | None = None,
    ) -> jax.Array:
        lo, hi = self._playground_bounds(self._goal_keepout)
        if self._uniform_initial_goal_sampling:
            candidates = jax.random.uniform(
                rng,
                (self._layout_candidate_count, 2),
                minval=lo,
                maxval=hi,
            )
            score = self._candidate_clearance(
                candidates,
                agent_xy[None, :],
                self._agent_keepout + self._goal_keepout + self._layout_margin,
            )
            if placed_xy is not None and placed_keepout is not None:
                object_score = self._candidate_keepout_score(
                    candidates,
                    placed_xy,
                    placed_keepout,
                    self._goal_keepout,
                )
                score = jp.minimum(score, object_score)
            return self._select_layout_candidate(candidates, score)

        rng_angle, rng_dist = jax.random.split(rng)
        angles = jax.random.uniform(
            rng_angle,
            (self._layout_candidate_count,),
            minval=0.0,
            maxval=2.0 * jp.pi,
        )
        dists = jax.random.uniform(
            rng_dist,
            (self._layout_candidate_count,),
            minval=self._min_goal_dist,
            maxval=self._max_goal_dist,
        )
        directions = jp.stack([jp.cos(angles), jp.sin(angles)], axis=-1)
        candidates = agent_xy + dists[:, None] * directions
        boundary_score = self._playground_boundary_score(candidates, self._goal_keepout)
        agent_score = self._candidate_clearance(
            candidates,
            agent_xy[None, :],
            self._agent_keepout + self._goal_keepout + self._layout_margin,
        )
        score = jp.minimum(boundary_score, agent_score)
        if placed_xy is not None and placed_keepout is not None:
            object_score = self._candidate_keepout_score(
                candidates,
                placed_xy,
                placed_keepout,
                self._goal_keepout,
            )
            score = jp.minimum(score, object_score)
        return self._select_layout_candidate(candidates, score)

    def _playground_bounds(self, keepout: float | jax.Array) -> tuple[jax.Array, jax.Array]:
        half = jp.maximum(jp.asarray(self._playground_size - keepout), 0.0)
        return self._playground_center_xy - half, self._playground_center_xy + half

    def _playground_boundary_score(self, candidates: jax.Array, keepout: float | jax.Array) -> jax.Array:
        lo, hi = self._playground_bounds(keepout)
        margin = jp.minimum(candidates - lo, hi - candidates)
        return jp.min(margin, axis=-1)

    def _candidate_clearance(
        self,
        candidates: jax.Array,
        centers_xy: jax.Array,
        min_dist: float | jax.Array,
    ) -> jax.Array:
        if centers_xy.shape[0] == 0:
            return jp.full(candidates.shape[:-1], jp.inf)
        dists = jp.linalg.norm(candidates[..., None, :] - centers_xy, axis=-1)
        return jp.min(dists - min_dist, axis=-1)

    def _select_layout_candidate(
        self,
        candidates: jax.Array,
        score: jax.Array,
        hard_score: jax.Array | None = None,
    ) -> jax.Array:
        if hard_score is None:
            hard_score = score
        valid = score >= 0.0
        hard_valid = hard_score >= 0.0
        has_valid = jp.any(valid, axis=-1)
        has_hard_valid = jp.any(hard_valid, axis=-1)
        if self._first_valid_layout_candidate:
            valid_idx = jp.argmax(valid.astype(jp.int32), axis=-1)
            hard_valid_idx = jp.argmax(hard_valid.astype(jp.int32), axis=-1)
        else:
            valid_score = jp.where(valid, score, -jp.inf)
            hard_valid_score = jp.where(hard_valid, score, -jp.inf)
            valid_idx = jp.argmax(valid_score, axis=-1)
            hard_valid_idx = jp.argmax(hard_valid_score, axis=-1)
        fallback_idx = jp.argmax(hard_score, axis=-1)
        idx = jp.where(has_valid, valid_idx, jp.where(has_hard_valid, hard_valid_idx, fallback_idx))
        selected = jp.take_along_axis(candidates, idx[..., None, None], axis=-2)
        return selected[..., 0, :]

    def _candidate_keepout_score(
        self,
        candidates: jax.Array,
        centers_xy: jax.Array,
        centers_keepout: jax.Array,
        keepout: float,
        active: jax.Array | None = None,
    ) -> jax.Array:
        if centers_xy.shape[0] == 0:
            return jp.full(candidates.shape[:-1], jp.inf)
        dists = jp.linalg.norm(candidates[..., None, :] - centers_xy, axis=-1)
        min_dists = keepout + centers_keepout + self._layout_margin
        score = dists - min_dists
        if active is not None:
            score = jp.where(active, score, jp.inf)
        return jp.min(score, axis=-1)

    def _sample_keepout_objects(
        self,
        rng: jax.Array,
        num_objects: int,
        keepout: float,
        placed_xy: jax.Array,
        placed_keepout: jax.Array,
    ) -> jax.Array:
        lo, hi = self._playground_bounds(keepout)
        candidates = jax.random.uniform(
            rng,
            (num_objects, self._layout_candidate_count, 2),
            minval=lo,
            maxval=hi,
        )
        objects = jp.zeros((num_objects, 2))
        placed_active = jp.ones((placed_xy.shape[0],), dtype=bool)

        def place_one(i: int, current: jax.Array) -> jax.Array:
            all_xy = jp.concatenate([placed_xy, current], axis=0)
            all_keepout = jp.concatenate([placed_keepout, jp.full((num_objects,), keepout)])
            active = jp.concatenate([placed_active, jp.arange(num_objects) < i], axis=0)
            score = self._candidate_keepout_score(candidates[i], all_xy, all_keepout, keepout, active)
            return current.at[i].set(self._select_layout_candidate(candidates[i], score))

        for i in range(num_objects):
            objects = place_one(i, objects)
        return objects

    def _sample_hazards(
        self,
        rng: jax.Array,
        agent_xy: jax.Array,
        goal_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
    ) -> jax.Array:
        placed_xy = jp.concatenate([agent_xy[None, :], goal_xy[None, :], gremlin_centers_xy], axis=0)
        placed_keepout = jp.concatenate(
            [
                jp.array([self._agent_keepout, self._goal_keepout]),
                jp.full((self._num_gremlins,), self._gremlin_keepout),
            ],
            axis=0,
        )
        return self._sample_keepout_objects(rng, self._num_hazards, self._hazard_keepout, placed_xy, placed_keepout)

    def _sample_obstacles(
        self,
        rng: jax.Array,
        agent_xy: jax.Array,
        hazards_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
        goal_xy: jax.Array,
    ) -> jax.Array:
        placed_xy = jp.concatenate([agent_xy[None, :], goal_xy[None, :], gremlin_centers_xy, hazards_xy], axis=0)
        placed_keepout = jp.concatenate(
            [
                jp.array([self._agent_keepout, self._goal_keepout]),
                jp.full((self._num_gremlins,), self._gremlin_keepout),
                jp.full((self._num_hazards,), self._hazard_keepout),
            ],
            axis=0,
        )
        return self._sample_keepout_objects(
            rng,
            self._num_obstacles,
            self._obstacle_keepout,
            placed_xy,
            placed_keepout,
        )

    def _sample_gremlins(
        self,
        rng: jax.Array,
        agent_xy: jax.Array,
        goal_xy: jax.Array,
    ) -> jax.Array:
        placed_xy = jp.stack([agent_xy, goal_xy])
        placed_keepout = jp.array([self._agent_keepout, self._goal_keepout])
        return self._sample_keepout_objects(
            rng,
            self._num_gremlins,
            self._gremlin_keepout,
            placed_xy,
            placed_keepout,
        )

    def _gremlin_positions(self, centers_xy: jax.Array, time: jax.Array) -> jax.Array:
        phase = time * self._gremlin_speed
        offset = jp.array([jp.sin(phase), jp.cos(phase)]) * self._gremlin_travel
        return centers_xy + offset[None, :]

    def _max_goal_distance_in_playground(self, agent_xy: jax.Array, keepout: float) -> jax.Array:
        lo, hi = self._playground_bounds(keepout)
        corners = jp.stack(
            [
                jp.array([lo[0], lo[1]], dtype=agent_xy.dtype),
                jp.array([lo[0], hi[1]], dtype=agent_xy.dtype),
                jp.array([hi[0], lo[1]], dtype=agent_xy.dtype),
                jp.array([hi[0], hi[1]], dtype=agent_xy.dtype),
            ],
            axis=0,
        )
        return jp.max(jp.linalg.norm(corners - agent_xy[None, :], axis=-1))

    def _respawn_path_object_count(
        self,
        candidates: jax.Array,
        agent_xy: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
    ) -> jax.Array:
        objects_xy = jp.concatenate([hazards_xy, obstacles_xy, gremlin_centers_xy], axis=0)
        effective_radii = jp.concatenate(
            [
                jp.full((self._num_hazards,), self._hazard_radius),
                jp.full((self._num_obstacles,), self._obstacle_radius),
                jp.full((self._num_gremlins,), self._gremlin_radius + self._gremlin_travel),
            ],
            axis=0,
        )
        segment = candidates - agent_xy[None, :]
        segment_len2 = jp.maximum(jp.sum(segment * segment, axis=-1), 1e-8)
        rel_objects = objects_xy[None, :, :] - agent_xy[None, None, :]
        projection = jp.sum(rel_objects * segment[:, None, :], axis=-1) / segment_len2[:, None]
        closest = agent_xy[None, None, :] + projection[:, :, None] * segment[:, None, :]
        perp_dist = jp.linalg.norm(objects_xy[None, :, :] - closest, axis=-1)
        band = effective_radii[None, :] * self._respawn_goal_path_band_scale
        on_open_segment = (projection > 0.0) & (projection < 1.0)
        near_path = on_open_segment & (perp_dist <= band)
        return jp.sum(near_path.astype(jp.int32), axis=-1)

    def _sample_respawn_goal(
        self,
        rng: jax.Array,
        agent_xy: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
        require_path_objects: bool | None = None,
    ) -> jax.Array:
        lo, hi = self._playground_bounds(self._goal_keepout)
        candidates = jax.random.uniform(
            rng,
            (self._layout_candidate_count, 2),
            minval=lo,
            maxval=hi,
        )
        placed_xy = jp.concatenate([agent_xy[None, :], hazards_xy, obstacles_xy, gremlin_centers_xy], axis=0)
        placed_keepout = jp.concatenate(
            [
                jp.array([self._agent_keepout]),
                jp.full((self._num_hazards,), self._hazard_keepout),
                jp.full((self._num_obstacles,), self._obstacle_keepout),
                jp.full((self._num_gremlins,), self._gremlin_keepout),
            ],
            axis=0,
        )
        boundary_score = self._playground_boundary_score(candidates, self._goal_keepout)
        keepout_score = self._candidate_keepout_score(
            candidates,
            placed_xy,
            placed_keepout,
            self._goal_keepout,
        )
        hard_score = jp.minimum(boundary_score, keepout_score)
        max_feasible_dist = self._max_goal_distance_in_playground(agent_xy, self._goal_keepout)
        effective_min_goal_dist = jp.minimum(
            jp.asarray(self._min_goal_dist, dtype=candidates.dtype),
            max_feasible_dist,
        )
        distance_score = jp.linalg.norm(candidates - agent_xy[None, :], axis=-1) - effective_min_goal_dist
        preferred_score = jp.minimum(hard_score, distance_score)
        selected = self._select_layout_candidate(candidates, preferred_score, hard_score)
        if require_path_objects is None:
            require_path_objects = (
                self._respawn_goal_require_path_objects
                and not self._initial_goal_path_objects_mode
            )
        if require_path_objects:
            path_count = self._respawn_path_object_count(
                candidates,
                agent_xy,
                hazards_xy,
                obstacles_xy,
                gremlin_centers_xy,
            )
            min_count = jp.asarray(self._respawn_goal_min_path_objects, dtype=path_count.dtype)
            qualified = (preferred_score >= 0.0) & (path_count >= min_count)
            ranking = path_count.astype(candidates.dtype) * 1000.0 + jp.maximum(distance_score, 0.0)
            qualified_score = jp.where(qualified, ranking, -jp.inf)
            qualified_idx = jp.argmax(qualified_score, axis=0)
            qualified_selected = candidates[qualified_idx]
            selected = jp.where(jp.any(qualified), qualified_selected, selected)
        return selected

    def _get_obs(
        self,
        pipeline_state: base.State,
        action: jax.Array,
        goal_xy: jax.Array,
        goal_yaw: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlins_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
    ) -> jax.Array:
        return jp.concatenate(
            [
                self._robot_obs(pipeline_state, action),
                self._layout_obs(
                    pipeline_state,
                    goal_xy,
                    goal_yaw,
                    hazards_xy,
                    obstacles_xy,
                    gremlins_xy,
                    gremlin_centers_xy,
                ),
            ]
        )

    def _robot_obs_size(self) -> int:
        if self._parking_mode and self._robot_quat_q_start is not None:
            return self.sys.q_size() + self.sys.qd_size() + 1
        if self._robot_sensor_data_start is not None:
            return 2 + self._robot_sensor_data_dim
        q_size = self.sys.q_size()
        size = q_size + self.sys.qd_size()
        if self._include_actuator_forces:
            size += self.sys.act_size()
        return size

    def _robot_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        if self._parking_mode and self._robot_yaw_q_idx is not None:
            return jp.concatenate(
                [
                    pipeline_state.q[:2],
                    self._agent_yaw(pipeline_state)[None],
                    pipeline_state.q[3:],
                    pipeline_state.qd,
                ]
            )
        if self._parking_mode and self._robot_quat_q_start is not None:
            return jp.concatenate(
                [
                    pipeline_state.q[:2],
                    self._agent_yaw(pipeline_state)[None],
                    pipeline_state.q[2:],
                    pipeline_state.qd,
                ]
            )
        if self._robot_sensor_data_start is not None:
            sensor_obs = jax.lax.dynamic_slice(
                pipeline_state.sensordata,
                (self._robot_sensor_data_start,),
                (self._robot_sensor_data_dim,),
            )
            return jp.concatenate([self._agent_xy(pipeline_state), sensor_obs])
        obs_parts = [pipeline_state.q, pipeline_state.qd]
        if self._include_actuator_forces:
            obs_parts.append(actuator.to_tau(self.sys, action, pipeline_state.q, pipeline_state.qd))
        return jp.concatenate(obs_parts)

    def _layout_obs(
        self,
        pipeline_state: base.State,
        goal_xy: jax.Array,
        goal_yaw: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlins_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
    ) -> jax.Array:
        if not self._include_object_layout_obs:
            return self._goal_obs(pipeline_state, goal_xy, goal_yaw)
        del gremlin_centers_xy
        agent_xy = self._agent_xy(pipeline_state)
        lidar_parts = [self._global_lidar(hazards_xy, agent_xy, self._hazard_radius, pipeline_state)]
        if self._num_obstacles > 0:
            lidar_parts.append(self._global_lidar(obstacles_xy, agent_xy, self._obstacle_radius, pipeline_state))
        lidar_parts.append(self._global_lidar(gremlins_xy, agent_xy, self._gremlin_radius, pipeline_state))
        lidar_parts.append(self._goal_obs(pipeline_state, goal_xy, goal_yaw))
        return jp.concatenate(lidar_parts)

    def _global_lidar(
        self,
        xy: jax.Array,
        agent_xy: jax.Array,
        radius: float,
        pipeline_state: base.State,
    ) -> jax.Array:
        """Agent-centered lidar, optionally binned in the robot ego frame."""
        del radius
        if xy.shape[0] == 0:
            return jp.zeros((self._layout_lidar_num_bins,), dtype=agent_xy.dtype)
        rel_xy = xy - agent_xy[None, :]
        dist = jp.linalg.norm(rel_xy, axis=-1)
        max_dist = jp.asarray(self._layout_lidar_max_dist, dtype=xy.dtype)
        signal = jp.clip(1.0 - dist / jp.maximum(max_dist, 1e-6), 0.0, 1.0)
        rel_xy = self._lidar_frame_xy(rel_xy, pipeline_state)
        angle = jp.mod(jp.arctan2(rel_xy[:, 1], rel_xy[:, 0]), 2.0 * jp.pi)
        bin_idx = jp.floor(angle / (2.0 * jp.pi) * self._layout_lidar_num_bins).astype(jp.int32)
        bin_idx = jp.clip(bin_idx, 0, self._layout_lidar_num_bins - 1)
        bin_mask = jax.nn.one_hot(bin_idx, self._layout_lidar_num_bins, dtype=xy.dtype)
        return jp.max(bin_mask * signal[:, None], axis=0)

    def _lidar_frame_xy(self, rel_xy: jax.Array, pipeline_state: base.State) -> jax.Array:
        if not self._layout_lidar_ego_frame:
            return rel_xy
        yaw = self._agent_yaw(pipeline_state)
        cos_yaw = jp.cos(yaw)
        sin_yaw = jp.sin(yaw)
        local_x = cos_yaw * rel_xy[:, 0] + sin_yaw * rel_xy[:, 1]
        local_y = -sin_yaw * rel_xy[:, 0] + cos_yaw * rel_xy[:, 1]
        return jp.stack([local_x, local_y], axis=-1)

    def _agent_yaw(self, pipeline_state: base.State) -> jax.Array:
        if self._robot_yaw_q_idx is not None:
            yaw = pipeline_state.q[self._robot_yaw_q_idx]
        elif self._robot_quat_q_start is not None:
            quat = pipeline_state.q[self._robot_quat_q_start : self._robot_quat_q_start + 4]
            w, x, y, z = quat
            yaw = jp.arctan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
        else:
            yaw = jp.asarray(0.0, dtype=pipeline_state.q.dtype)
        return jp.arctan2(jp.sin(yaw), jp.cos(yaw))

    def _object_layout_entries(self, xy: jax.Array, type_id: int) -> jax.Array:
        if not self._include_object_type_obs:
            return xy.reshape((-1,))
        type_one_hot = jp.eye(3, dtype=xy.dtype)[type_id]
        type_obs = jp.broadcast_to(type_one_hot, (xy.shape[0], 3))
        return jp.concatenate([xy, type_obs], axis=-1).reshape((-1,))

    def _gremlin_layout_entries(self, gremlins_xy: jax.Array, gremlin_centers_xy: jax.Array) -> jax.Array:
        gremlin_motion = jp.concatenate([gremlins_xy, gremlin_centers_xy], axis=-1)
        if not self._include_object_type_obs:
            return gremlin_motion.reshape((-1,))
        type_one_hot = jp.array([0.0, 0.0, 1.0], dtype=gremlins_xy.dtype)
        type_obs = jp.broadcast_to(type_one_hot, (gremlins_xy.shape[0], 3))
        return jp.concatenate([gremlin_motion, type_obs], axis=-1).reshape((-1,))

    def _goal_obs(
        self,
        pipeline_state: base.State,
        goal_xy: jax.Array,
        goal_yaw: jax.Array,
    ) -> jax.Array:
        del pipeline_state
        return self._goal_position(goal_xy, goal_yaw)

    def _goal_position(self, goal_xy: jax.Array, goal_yaw: jax.Array | None = None) -> jax.Array:
        if self._parking_mode:
            if goal_yaw is None:
                goal_yaw = jp.asarray(0.0, dtype=goal_xy.dtype)
            return jp.concatenate([goal_xy, jp.asarray(goal_yaw, dtype=goal_xy.dtype)[None]])
        if self._goal_dim == 2:
            return goal_xy
        goal_z = jp.asarray(self._goal_z, dtype=goal_xy.dtype)
        return jp.concatenate([goal_xy, goal_z[None]], axis=-1)

    def _agent_xy(self, pipeline_state: base.State) -> jax.Array:
        return pipeline_state.x.pos[0, :2]

    def _success_xy(self, pipeline_state: base.State) -> jax.Array:
        return pipeline_state.x.pos[self._success_link_idx, :2]

    def _achieved_goal(self, pipeline_state: base.State) -> jax.Array:
        if self._parking_mode:
            return jp.concatenate(
                [self._success_xy(pipeline_state), self._agent_yaw(pipeline_state)[None]]
            )
        if self._goal_dim == 2:
            return self._success_xy(pipeline_state)
        return pipeline_state.x.pos[self._success_link_idx, :3]

    def _success_goal(self, pipeline_state: base.State) -> jax.Array:
        return self._achieved_goal(pipeline_state)

    def _sample_goal_yaw(self, rng: jax.Array, *, fixed: bool = False) -> jax.Array:
        if not self._parking_mode:
            return jp.asarray(0.0, dtype=jp.float32)
        if fixed:
            rng = jax.random.fold_in(self._fixed_goal_rng(), 1)
        return jax.random.uniform(rng, (), minval=-jp.pi, maxval=jp.pi)

    def _goal_success(
        self,
        achieved_goal: jax.Array,
        goal_position: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        dist = jp.linalg.norm(achieved_goal[:2] - goal_position[:2])
        if not self._parking_mode:
            full_dist = jp.linalg.norm(achieved_goal - goal_position)
            return full_dist, jp.asarray(0.0, dtype=full_dist.dtype), (
                full_dist <= self._goal_radius
            ).astype(float)
        yaw_delta = achieved_goal[2] - goal_position[2]
        yaw_error = jp.abs(jp.arctan2(jp.sin(yaw_delta), jp.cos(yaw_delta)))
        success = jp.logical_and(
            dist <= self._goal_radius,
            yaw_error <= self._parking_yaw_tolerance,
        )
        return dist, yaw_error, success.astype(float)

    def _costs(
        self,
        pipeline_state: base.State,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlins_xy: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self._use_geom_cost:
            geom_pos, seg_a, seg_b = self._robot_geom_segments(pipeline_state)
            cost_hazards = self._geom_hazard_cost(geom_pos, seg_a, seg_b, hazards_xy)
            cost_obstacles = self._geom_sphere_object_cost(
                geom_pos,
                seg_a,
                seg_b,
                obstacles_xy,
                self._obstacle_radius,
                self._obstacle_height,
            )
            cost_gremlins = self._geom_sphere_object_cost(
                geom_pos,
                seg_a,
                seg_b,
                gremlins_xy,
                self._gremlin_radius,
                self._gremlin_height,
            )
        else:
            agent_xy = self._agent_xy(pipeline_state)
            cost_hazards = self._center_disk_cost(agent_xy, hazards_xy, self._hazard_radius)
            if self._use_3d_object_cost:
                agent_xyz = pipeline_state.x.pos[0, :3]
                cost_obstacles = self._center_sphere_object_cost(
                    agent_xyz,
                    obstacles_xy,
                    self._obstacle_radius,
                    self._obstacle_height,
                )
                cost_gremlins = self._center_sphere_object_cost(
                    agent_xyz,
                    gremlins_xy,
                    self._gremlin_radius,
                    self._gremlin_height,
                )
            else:
                cost_obstacles = self._center_disk_cost(agent_xy, obstacles_xy, self._obstacle_radius)
                cost_gremlins = self._center_disk_cost(agent_xy, gremlins_xy, self._gremlin_radius)
        return cost_hazards, cost_obstacles, cost_gremlins

    def _object_boundary_cost(
        self,
        pipeline_state: base.State,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlins_xy: jax.Array,
    ) -> jax.Array:
        agent_xy = self._agent_xy(pipeline_state)
        boundary_hazards = self._center_disk_cost(agent_xy, hazards_xy, self._hazard_radius, margin=0.0)
        if self._use_3d_object_cost:
            agent_xyz = pipeline_state.x.pos[0, :3]
            boundary_obstacles = self._center_sphere_object_cost(
                agent_xyz,
                obstacles_xy,
                self._obstacle_radius,
                self._obstacle_height,
                margin=0.0,
            )
            boundary_gremlins = self._center_sphere_object_cost(
                agent_xyz,
                gremlins_xy,
                self._gremlin_radius,
                self._gremlin_height,
                margin=0.0,
            )
        else:
            boundary_obstacles = self._center_disk_cost(agent_xy, obstacles_xy, self._obstacle_radius, margin=0.0)
            boundary_gremlins = self._center_disk_cost(agent_xy, gremlins_xy, self._gremlin_radius, margin=0.0)
        return jp.maximum(boundary_hazards, jp.maximum(boundary_obstacles, boundary_gremlins))

    def _center_disk_cost(
        self,
        agent_xy: jax.Array,
        centers_xy: jax.Array,
        radius: float,
        margin: float | None = None,
    ) -> jax.Array:
        delta = centers_xy - agent_xy[None, :]
        dist_sq = jp.sum(delta * delta, axis=-1)
        effective_radius = radius + (self._robot_cost_margin if margin is None else margin)
        return jp.any(dist_sq <= effective_radius * effective_radius).astype(float)

    def _center_sphere_object_cost(
        self,
        agent_xyz: jax.Array,
        centers_xy: jax.Array,
        radius: float,
        height: float,
        margin: float | None = None,
    ) -> jax.Array:
        centers_z = jp.full((centers_xy.shape[0], 1), height, dtype=centers_xy.dtype)
        centers_xyz = jp.concatenate([centers_xy, centers_z], axis=-1)
        delta = centers_xyz - agent_xyz[None, :]
        dist_sq = jp.sum(delta * delta, axis=-1)
        effective_radius = radius + (self._robot_cost_margin if margin is None else margin)
        return jp.any(dist_sq <= effective_radius * effective_radius).astype(float)

    def _robot_geom_segments(self, pipeline_state: base.State) -> tuple[jax.Array, jax.Array, jax.Array]:
        geom_pos = pipeline_state.geom_xpos
        geom_xmat = pipeline_state.geom_xmat.reshape((-1, 3, 3))
        capsule_axis = geom_xmat[:, :, 2]
        seg_a = geom_pos - capsule_axis * self._geom_capsule_half_length[:, None]
        seg_b = geom_pos + capsule_axis * self._geom_capsule_half_length[:, None]
        return geom_pos, seg_a, seg_b

    def _geom_hazard_cost(
        self,
        geom_pos: jax.Array,
        seg_a: jax.Array,
        seg_b: jax.Array,
        hazards_xy: jax.Array,
    ) -> jax.Array:
        sphere_delta = hazards_xy[None, :, :] - geom_pos[:, None, :2]
        sphere_dist_sq = jp.sum(sphere_delta * sphere_delta, axis=-1)
        sphere_radius_sq = (self._geom_radius + self._hazard_radius) ** 2
        sphere_overlap = sphere_dist_sq <= sphere_radius_sq[:, None]

        capsule_dist_sq = self._point_segment_distance_sq(
            hazards_xy[None, :, :],
            seg_a[:, None, :2],
            seg_b[:, None, :2],
        )
        capsule_radius_sq = (self._geom_radius + self._hazard_radius) ** 2
        capsule_overlap = capsule_dist_sq <= capsule_radius_sq[:, None]

        overlap = (
            (sphere_overlap & self._robot_sphere_geom_mask[:, None])
            | (capsule_overlap & self._robot_capsule_geom_mask[:, None])
        )
        return jp.any(overlap).astype(float)

    def _geom_sphere_object_cost(
        self,
        geom_pos: jax.Array,
        seg_a: jax.Array,
        seg_b: jax.Array,
        centers_xy: jax.Array,
        object_radius: float,
        object_height: float,
    ) -> jax.Array:
        centers_z = jp.full((centers_xy.shape[0], 1), object_height)
        centers_xyz = jp.concatenate([centers_xy, centers_z], axis=-1)

        sphere_delta = centers_xyz[None, :, :] - geom_pos[:, None, :]
        sphere_dist_sq = jp.sum(sphere_delta * sphere_delta, axis=-1)
        sphere_radius_sq = (self._geom_radius + object_radius) ** 2
        sphere_overlap = sphere_dist_sq <= sphere_radius_sq[:, None]

        capsule_dist_sq = self._point_segment_distance_sq(
            centers_xyz[None, :, :],
            seg_a[:, None, :],
            seg_b[:, None, :],
        )
        capsule_radius_sq = (self._geom_radius + object_radius) ** 2
        capsule_overlap = capsule_dist_sq <= capsule_radius_sq[:, None]

        overlap = (
            (sphere_overlap & self._robot_sphere_geom_mask[:, None])
            | (capsule_overlap & self._robot_capsule_geom_mask[:, None])
        )
        return jp.any(overlap).astype(float)

    def _point_segment_distance_sq(self, points: jax.Array, start: jax.Array, end: jax.Array) -> jax.Array:
        segment = end - start
        segment_len_sq = jp.sum(segment * segment, axis=-1)
        point_offset = points - start
        t = jp.sum(point_offset * segment, axis=-1) / jp.maximum(segment_len_sq, 1e-12)
        t = jp.clip(t, 0.0, 1.0)
        closest = start + t[..., None] * segment
        delta = points - closest
        return jp.sum(delta * delta, axis=-1)

    def _link_index(self, link_name: str) -> int:
        try:
            return self.sys.link_names.index(link_name)
        except ValueError as exc:
            raise ValueError(f"Link {link_name!r} is not present in system links: {self.sys.link_names}") from exc

    def _metrics(
        self,
        success: jax.Array,
        dist: jax.Array,
        cost_hazards: jax.Array,
        cost_obstacles: jax.Array,
        cost_gremlins: jax.Array,
        agent_goal: jax.Array,
        goal_position: jax.Array,
        agent_yaw: jax.Array | None = None,
        yaw_error: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        if agent_yaw is None:
            agent_yaw = jp.asarray(0.0, dtype=success.dtype)
        if yaw_error is None:
            yaw_error = jp.asarray(0.0, dtype=success.dtype)
        cost = jp.maximum(cost_hazards, jp.maximum(cost_obstacles, cost_gremlins))
        metrics = {
            "reward": success,
            "success": success,
            "cost": cost,
            "cost_hazards": cost_hazards,
            "cost_obstacles": cost_obstacles,
            "cost_gremlins": cost_gremlins,
            "dist": dist,
            "x_position": agent_goal[0],
            "y_position": agent_goal[1],
            "agent_yaw": agent_yaw,
            "goal_x": goal_position[0],
            "goal_y": goal_position[1],
            "yaw_error": yaw_error,
        }
        if self._parking_mode:
            metrics["goal_yaw"] = goal_position[2]
        elif self._goal_dim == 3:
            metrics.update(
                {
                    "z_position": agent_goal[2],
                    "goal_z": goal_position[2],
                }
            )
        return metrics
