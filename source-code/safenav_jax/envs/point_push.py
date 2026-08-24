"""Point push environment with ghost hazards and a physical push cube."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from brax import base
from brax.envs.base import State
from brax.io import mjcf
import jax
from jax import numpy as jp

from safenav_jax.envs.headless_goal_base import HeadlessGoalEnv


XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "xmls" / "point.xml"


def _add_xy_slide_body(
    worldbody: ET.Element,
    name: str,
    z_pos: float,
    geom_type: str,
    geom_size: str,
    rgba: str,
) -> None:
    body = ET.SubElement(worldbody, "body", name=name, pos=f"0 0 {z_pos}")
    ET.SubElement(
        body,
        "joint",
        name=f"{name}_x",
        type="slide",
        axis="1 0 0",
        limited="true",
        range="-10 10",
        damping="0",
        stiffness="0",
        armature="0",
    )
    ET.SubElement(
        body,
        "joint",
        name=f"{name}_y",
        type="slide",
        axis="0 1 0",
        limited="true",
        range="-10 10",
        damping="0",
        stiffness="0",
        armature="0",
    )
    ET.SubElement(
        body,
        "geom",
        name=name,
        type=geom_type,
        size=geom_size,
        contype="0",
        conaffinity="0",
        rgba=rgba,
        mass="0.001",
    )


def _xml_with_push_box(
    *,
    render_layout: bool,
    cube_half_size: float,
    cube_density: float,
    goal_radius: float,
    num_hazards: int,
    hazard_radius: float,
) -> str:
    tree = ET.parse(XML_PATH)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError("point.xml is missing a worldbody.")

    cube = ET.SubElement(worldbody, "body", name="push_box", pos=f"0 0 {cube_half_size}")
    ET.SubElement(cube, "joint", name="push_box_free", type="free", damping="0", armature="0")
    ET.SubElement(
        cube,
        "geom",
        name="push_box",
        type="box",
        size=f"{cube_half_size} {cube_half_size} {cube_half_size}",
        density=f"{cube_density}",
        friction="1 0.01 0.01",
        rgba="1 0.7 0 1",
    )

    if render_layout:
        target_z = goal_radius / 2.0 + 0.01
        _add_xy_slide_body(
            worldbody,
            "target",
            target_z,
            "cylinder",
            f"{goal_radius} {goal_radius / 2.0}",
            "0 1 0 0.25",
        )
        for i in range(num_hazards):
            _add_xy_slide_body(
                worldbody,
                f"hazard{i}",
                0.02,
                "cylinder",
                f"{hazard_radius} 0.01",
                "0 0 1 0.25",
            )
    return ET.tostring(tree.getroot(), encoding="unicode")


class PointPushBase(HeadlessGoalEnv):
    """Shared point-push task logic.

    Observation layout:
      cube_global_xy + robot_global_xy + 12D robot sensors
      + hazard global-lidar + goal_global_xy.
    """

    def __init__(
        self,
        *,
        render_layout: bool,
        backend: str = "mjx",
        n_frames: int = 10,
        episode_length: int = 1000,
        reset_noise_scale: float = 0.0,
        agent_spawn_bound: float | None = None,
        playground_size: float = 2.0,
        goal_wall_margin: float = 0.4,
        min_goal_dist: float = 1.0,
        max_goal_dist: float = 3.0,
        goal_radius: float = 0.3,
        num_hazards: int = 7,
        hazard_radius: float = 0.3,
        cube_half_size: float = 0.2,
        cube_density: float = 0.001,
        layout_margin: float = 0.0,
        agent_keepout: float = 0.4,
        goal_keepout: float = 0.4,
        hazard_keepout: float = 0.4,
        cube_keepout: float = 0.2,
        max_agent_cube_reset_dist: float = 1.0,
        include_object_layout_obs: bool = True,
        uniform_initial_goal_sampling: bool = True,
        first_valid_layout_candidate: bool = True,
        random_yaw: bool = False,
        ego_view: bool = False,
        **kwargs,
    ):
        del ego_view  # Point push always uses global lidar and global goal coordinates.
        kwargs.pop("include_object_type_obs", None)
        min_agent_cube_reset_dist = agent_keepout + cube_keepout + layout_margin
        if max_agent_cube_reset_dist <= 0.0:
            raise ValueError("max_agent_cube_reset_dist must be positive.")
        if max_agent_cube_reset_dist < min_agent_cube_reset_dist:
            raise ValueError(
                "max_agent_cube_reset_dist must be at least "
                "agent_keepout + cube_keepout + layout_margin "
                f"({min_agent_cube_reset_dist:g}), got {max_agent_cube_reset_dist:g}."
            )
        sys = mjcf.loads(
            _xml_with_push_box(
                render_layout=render_layout,
                cube_half_size=cube_half_size,
                cube_density=cube_density,
                goal_radius=goal_radius,
                num_hazards=num_hazards,
                hazard_radius=hazard_radius,
            )
        )
        self._render_layout = render_layout
        self._cube_half_size = cube_half_size
        self._cube_density = cube_density
        self._cube_keepout = cube_keepout
        self._min_agent_cube_reset_dist = min_agent_cube_reset_dist
        self._max_agent_cube_reset_dist = max_agent_cube_reset_dist
        self._point_robot_q_size = 3
        self._point_robot_qd_size = 3
        self._cube_q_idx = self._point_robot_q_size
        self._cube_qd_idx = self._point_robot_qd_size
        self._visual_target_q_idx = self._cube_q_idx + 7
        self._visual_layout_q_idx = self._visual_target_q_idx + 2
        self._visual_target_qd_idx = self._cube_qd_idx + 6
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
            num_obstacles=0,
            num_gremlins=0,
            hazard_radius=hazard_radius,
            obstacle_radius=0.0,
            obstacle_height=0.0,
            gremlin_radius=0.0,
            gremlin_height=0.0,
            gremlin_travel=0.0,
            gremlin_speed=0.0,
            layout_margin=layout_margin,
            random_agent=True,
            agent_spawn_bound=agent_spawn_bound,
            agent_keepout=agent_keepout,
            goal_keepout=goal_keepout,
            hazard_keepout=hazard_keepout,
            include_object_layout_obs=include_object_layout_obs,
            include_object_type_obs=False,
            uniform_initial_goal_sampling=uniform_initial_goal_sampling,
            first_valid_layout_candidate=first_valid_layout_candidate,
            random_yaw=random_yaw,
            success_link_name="push_box",
            robot_sensor_data_start=0,
            robot_sensor_data_dim=12,
            layout_lidar_ego_frame=False,
            robot_yaw_q_idx=2,
            **kwargs,
        )
        self._cube_link_idx = self._link_index("push_box")
        self._robot_obs_dim = 2 + 2 + self._robot_sensor_data_dim
        self._layout_lidar_num_channels = 1
        object_layout_dim = (
            self._layout_lidar_num_channels * self._layout_lidar_num_bins
            if include_object_layout_obs
            else 0
        )
        self._layout_obs_dim = self._goal_dim + object_layout_dim
        goal_start = self._robot_obs_dim + object_layout_dim
        self.state_dim = self._robot_obs_dim + self._layout_obs_dim
        self.goal_indices = jp.arange(goal_start, goal_start + self._goal_dim)
        self.scaling_crl_goal_indices = jp.arange(2)
        self.raw_goal_dim = 2
        self.relabel_goal_dim = 2

    def reset(self, rng: jax.Array) -> State:
        reset_rng = rng
        (
            rng,
            q_rng,
            qd_rng,
            agent_rng,
            yaw_rng,
            cube_rng,
            goal_rng,
            hazard_rng,
            respawn_rng,
        ) = jax.random.split(rng, 9)
        q = self.sys.init_q
        qd = jp.zeros(self.sys.qd_size())
        layout_id = self._layout_id_from_rng(reset_rng)

        if self._reset_noise_scale:
            low, hi = -self._reset_noise_scale, self._reset_noise_scale
            robot_q = q[: self._point_robot_q_size] + jax.random.uniform(
                q_rng,
                (self._point_robot_q_size,),
                minval=low,
                maxval=hi,
            )
            robot_qd = hi * jax.random.normal(qd_rng, (self._point_robot_qd_size,))
            q = q.at[: self._point_robot_q_size].set(robot_q)
            qd = qd.at[: self._point_robot_qd_size].set(robot_qd)

        agent_xy, cube_xy, goal_xy, hazards_xy = self._sample_push_reset_layout(
            reset_rng,
            agent_rng,
            cube_rng,
            goal_rng,
            hazard_rng,
            layout_id,
        )
        q = q.at[:2].set(agent_xy)
        if self._random_yaw:
            yaw = jax.random.uniform(yaw_rng, (), minval=-jp.pi, maxval=jp.pi)
            q = q.at[2].set(yaw)
        q = self._set_cube_q(q, cube_xy)
        if self._render_layout:
            q = self._sync_visual_q(q, goal_xy, hazards_xy)
            qd = self._zero_visual_qd(qd)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()), goal_xy, hazards_xy)
        zero = jp.array(0.0)
        cube_goal = self._cube_xy(pipeline_state)
        dist = jp.linalg.norm(cube_goal - goal_xy)
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=zero,
            done=zero,
            metrics=self._metrics(
                zero,
                dist,
                zero,
                cube_goal,
                self._agent_xy(pipeline_state),
                goal_xy,
                self._agent_yaw(pipeline_state),
            ),
            info={
                "steps": jp.array(0),
                "seed": jp.array(0, dtype=jp.int32),
                "rng": respawn_rng,
                "goal_xy": goal_xy,
                "hazards_xy": hazards_xy,
                "episode_cost": zero,
                "layout_id": layout_id,
            },
        )

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        cube_xy = self._cube_xy(pipeline_state)
        goal_xy = state.info["goal_xy"]
        dist = jp.linalg.norm(cube_xy - goal_xy)
        success = (dist <= self._goal_radius).astype(float)
        reward = success
        cost_hazards = self._center_disk_cost(self._agent_xy(pipeline_state), state.info["hazards_xy"], self._hazard_radius)
        object_boundary_cost = self._center_disk_cost(
            self._agent_xy(pipeline_state),
            state.info["hazards_xy"],
            self._hazard_radius,
            margin=0.0,
        )

        rng, respawn_rng = jax.random.split(state.info["rng"])
        if self._goal_respawn_on_success:
            respawn_goal_xy = self._sample_respawn_goal(
                respawn_rng,
                cube_xy,
                state.info["hazards_xy"],
                jp.zeros((0, 2), dtype=cube_xy.dtype),
                jp.zeros((0, 2), dtype=cube_xy.dtype),
            )
            next_goal_xy = jp.where(success.astype(bool), respawn_goal_xy, goal_xy)
        else:
            next_goal_xy = goal_xy

        wrapped_by_training = "truncation" in state.info
        steps = state.info["steps"] + 1
        done = jp.array(False) if wrapped_by_training else steps >= self._episode_length
        episode_cost = state.info["episode_cost"] + cost_hazards
        if self._object_boundary:
            done = jp.logical_or(done, object_boundary_cost.astype(bool))
        if self._terminate_on_cost:
            done = jp.logical_or(done, episode_cost >= self._cost_limit_max)
        done = done.astype(float)

        if self._render_layout:
            q = self._sync_visual_q(pipeline_state.q, next_goal_xy, state.info["hazards_xy"])
            qd = self._zero_visual_qd(pipeline_state.qd)
            synced_pipeline_state = self.pipeline_init(q, qd)
            pipeline_state = synced_pipeline_state.replace(time=pipeline_state.time)

        obs = self._get_obs(pipeline_state, action, next_goal_xy, state.info["hazards_xy"])
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
                "hazards_xy": state.info["hazards_xy"],
                "episode_cost": episode_cost,
                "layout_id": state.info["layout_id"],
            }
        )
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=self._metrics(
                success,
                dist,
                cost_hazards,
                cube_xy,
                self._agent_xy(pipeline_state),
                next_goal_xy,
                self._agent_yaw(pipeline_state),
            ),
            info=info,
        )

    def _sample_push_reset_layout(
        self,
        reset_rng: jax.Array,
        agent_rng: jax.Array,
        cube_rng: jax.Array,
        goal_rng: jax.Array,
        hazard_rng: jax.Array,
        layout_id: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if self._relocate_objects_on_reset:
            if self._fixed_agent_on_reset:
                agent_xy = self._sample_agent_xy(self._fixed_agent_rng())
            else:
                agent_xy = self._sample_agent_xy(agent_rng)
            hazards_xy = self._sample_push_hazards(hazard_rng, agent_xy)
        else:
            fixed_agent_xy = self._sample_agent_xy(self._fixed_agent_rng())
            hazards_xy = self._sample_push_hazards(self._fixed_layout_rng(reset_rng, layout_id), fixed_agent_xy)
            hazard_keepout = jp.full((self._num_hazards,), self._hazard_keepout)
            if self._fixed_agent_on_reset:
                agent_xy = self._sample_agent_xy(self._fixed_agent_rng(), hazards_xy, hazard_keepout)
            else:
                agent_xy = self._sample_agent_xy(agent_rng, hazards_xy, hazard_keepout)
        cube_xy = self._sample_cube_xy(cube_rng, agent_xy, hazards_xy)
        if self._fixed_goal_on_reset:
            goal_xy = self._sample_push_goal(self._fixed_goal_rng(), cube_xy, agent_xy, hazards_xy)
        else:
            goal_xy = self._sample_push_goal(goal_rng, cube_xy, agent_xy, hazards_xy)
        return agent_xy, cube_xy, goal_xy, hazards_xy

    def _fixed_layout_rng(self, reset_rng: jax.Array, layout_id: jax.Array) -> jax.Array:
        if self._layout_pool_size > 0:
            return self._layout_rng(layout_id)
        base_rng = jax.random.PRNGKey(self._fixed_object_layout_seed)
        if self._different_object_layout_per_env:
            return jax.random.fold_in(base_rng, reset_rng[1].astype(jp.int32))
        return base_rng

    def _sample_push_hazards(self, rng: jax.Array, agent_xy: jax.Array) -> jax.Array:
        placed_xy = agent_xy[None, :]
        placed_keepout = jp.array([self._agent_keepout])
        return self._sample_keepout_objects(
            rng,
            self._num_hazards,
            self._hazard_keepout,
            placed_xy,
            placed_keepout,
        )

    def _sample_cube_xy(self, rng: jax.Array, agent_xy: jax.Array, hazards_xy: jax.Array) -> jax.Array:
        angle_rng, radius_rng = jax.random.split(rng)
        angles = jax.random.uniform(
            angle_rng,
            (self._layout_candidate_count,),
            minval=0.0,
            maxval=2.0 * jp.pi,
        )
        radius_sq = jax.random.uniform(
            radius_rng,
            (self._layout_candidate_count,),
            minval=self._min_agent_cube_reset_dist**2,
            maxval=self._max_agent_cube_reset_dist**2,
        )
        radii = jp.sqrt(radius_sq)
        directions = jp.stack([jp.cos(angles), jp.sin(angles)], axis=-1)
        candidates = agent_xy[None, :] + radii[:, None] * directions

        placed_xy = jp.concatenate([agent_xy[None, :], hazards_xy], axis=0)
        placed_keepout = jp.concatenate(
            [jp.array([self._agent_keepout]), jp.full((self._num_hazards,), self._hazard_keepout)],
            axis=0,
        )
        boundary_score = self._playground_boundary_score(candidates, self._cube_keepout)
        keepout_score = self._candidate_keepout_score(
            candidates, placed_xy, placed_keepout, self._cube_keepout
        )
        score = jp.minimum(boundary_score, keepout_score)
        return self._select_layout_candidate(candidates, score)

    def _sample_push_goal(
        self,
        rng: jax.Array,
        cube_xy: jax.Array,
        agent_xy: jax.Array,
        hazards_xy: jax.Array,
    ) -> jax.Array:
        lo, hi = self._playground_bounds(self._goal_keepout)
        candidates = jax.random.uniform(
            rng,
            (self._layout_candidate_count, 2),
            minval=lo,
            maxval=hi,
        )
        placed_xy = jp.concatenate([agent_xy[None, :], cube_xy[None, :], hazards_xy], axis=0)
        placed_keepout = jp.concatenate(
            [
                jp.array([self._agent_keepout, self._cube_keepout]),
                jp.full((self._num_hazards,), self._hazard_keepout),
            ],
            axis=0,
        )
        boundary_score = self._playground_boundary_score(candidates, self._goal_keepout)
        keepout_score = self._candidate_keepout_score(candidates, placed_xy, placed_keepout, self._goal_keepout)
        distance_score = jp.linalg.norm(candidates - cube_xy[None, :], axis=-1) - self._min_goal_dist
        hard_score = jp.minimum(boundary_score, keepout_score)
        preferred_score = jp.minimum(hard_score, distance_score)
        selected = self._select_layout_candidate(candidates, preferred_score, hard_score)
        if self._respawn_goal_require_path_objects:
            path_count = self._respawn_path_object_count(
                candidates,
                cube_xy,
                hazards_xy,
                jp.zeros((0, 2), dtype=cube_xy.dtype),
                jp.zeros((0, 2), dtype=cube_xy.dtype),
            )
            min_count = jp.asarray(self._respawn_goal_min_path_objects, dtype=path_count.dtype)
            qualified = (preferred_score >= 0.0) & (path_count >= min_count)
            ranking = path_count.astype(candidates.dtype) * 1000.0 + jp.maximum(distance_score, 0.0)
            qualified_idx = jp.argmax(jp.where(qualified, ranking, -jp.inf), axis=0)
            selected = jp.where(jp.any(qualified), candidates[qualified_idx], selected)
        return selected

    def _set_cube_q(self, q: jax.Array, cube_xy: jax.Array) -> jax.Array:
        cube_q = jp.array(
            [cube_xy[0], cube_xy[1], self._cube_half_size, 1.0, 0.0, 0.0, 0.0],
            dtype=q.dtype,
        )
        return q.at[self._cube_q_idx : self._cube_q_idx + 7].set(cube_q)

    def _sync_visual_q(self, q: jax.Array, goal_xy: jax.Array, hazards_xy: jax.Array) -> jax.Array:
        q = q.at[self._visual_target_q_idx : self._visual_target_q_idx + 2].set(goal_xy)
        flat_layout = hazards_xy.reshape((-1,))
        return q.at[self._visual_layout_q_idx : self._visual_layout_q_idx + flat_layout.shape[0]].set(flat_layout)

    def _zero_visual_qd(self, qd: jax.Array) -> jax.Array:
        return qd.at[self._visual_target_qd_idx :].set(0.0)

    def _get_obs(
        self,
        pipeline_state: base.State,
        action: jax.Array,
        goal_xy: jax.Array,
        hazards_xy: jax.Array,
    ) -> jax.Array:
        del action
        cube_xy = self._cube_xy(pipeline_state)
        robot_obs = self._robot_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        if self._include_object_layout_obs:
            agent_xy = self._agent_xy(pipeline_state)
            layout_obs = self._global_lidar(hazards_xy, agent_xy, self._hazard_radius, pipeline_state)
            return jp.concatenate([cube_xy, robot_obs, layout_obs, goal_xy])
        return jp.concatenate([cube_xy, robot_obs, goal_xy])

    def _robot_obs_size(self) -> int:
        return 2 + self._robot_sensor_data_dim

    def _robot_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        del action
        sensor_obs = jax.lax.dynamic_slice(
            pipeline_state.sensordata,
            (self._robot_sensor_data_start,),
            (self._robot_sensor_data_dim,),
        )
        return jp.concatenate([self._agent_xy(pipeline_state), sensor_obs])

    def _cube_xy(self, pipeline_state: base.State) -> jax.Array:
        return pipeline_state.x.pos[self._cube_link_idx, :2]

    def _metrics(
        self,
        success: jax.Array,
        dist: jax.Array,
        cost_hazards: jax.Array,
        cube_xy: jax.Array,
        agent_xy: jax.Array,
        goal_xy: jax.Array,
        agent_yaw: jax.Array,
    ) -> dict[str, jax.Array]:
        return {
            "reward": success,
            "success": success,
            "cost": cost_hazards,
            "cost_hazards": cost_hazards,
            "dist": dist,
            "x_position": cube_xy[0],
            "y_position": cube_xy[1],
            "agent_x": agent_xy[0],
            "agent_y": agent_xy[1],
            "agent_yaw": agent_yaw,
            "goal_x": goal_xy[0],
            "goal_y": goal_xy[1],
        }



class PointPush(PointPushBase):
    """Renderable point push environment."""

    def __init__(self, **kwargs):
        super().__init__(render_layout=True, **kwargs)
