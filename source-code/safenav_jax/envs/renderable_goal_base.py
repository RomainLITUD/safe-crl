"""Renderable goal environments aligned to the headless task logic."""

from __future__ import annotations

from brax import base
from brax.envs.base import State
from jax import numpy as jp
import jax

from safenav_jax.envs.headless_goal_base import HeadlessGoalEnv


class RenderableGoalEnv(HeadlessGoalEnv):
    """Headless task logic with MJCF visual bodies synced from `state.info`."""

    def __init__(
        self,
        *,
        robot_q_size: int,
        robot_qd_size: int,
        visual_target_q_idx: int,
        visual_layout_q_idx: int,
        visual_target_qd_idx: int,
        visual_target_link_name: str = "target",
        **kwargs,
    ):
        self._robot_q_size = robot_q_size
        self._robot_qd_size = robot_qd_size
        self._visual_target_q_idx = visual_target_q_idx
        self._visual_layout_q_idx = visual_layout_q_idx
        self._visual_target_qd_idx = visual_target_qd_idx
        super().__init__(**kwargs)
        self._visual_target_link_idx = self._link_index(visual_target_link_name)
        visual_start_bodyid = self._visual_target_link_idx + 1
        self._robot_geom_mask = (jp.asarray(self.sys.geom_bodyid) != 0) & (
            jp.asarray(self.sys.geom_bodyid) < visual_start_bodyid
        )
        self._robot_sphere_geom_mask = self._robot_geom_mask & (self._geom_type == 2)
        self._robot_capsule_geom_mask = self._robot_geom_mask & (self._geom_type == 3)

    def reset(self, rng: jax.Array):
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
            robot_q = q[: self._robot_q_size] + jax.random.uniform(
                q_rng,
                (self._robot_q_size,),
                minval=low,
                maxval=hi,
            )
            robot_qd = hi * jax.random.normal(qd_rng, (self._robot_qd_size,))
            q = q.at[: self._robot_q_size].set(robot_q)
            qd = qd.at[: self._robot_qd_size].set(robot_qd)
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

        initial_pipeline_state = self.pipeline_init(q, qd)
        gremlin_time = getattr(initial_pipeline_state, "time", jp.asarray(0.0, dtype=jp.float32))
        gremlins_xy = self._gremlin_positions(gremlin_centers_xy, gremlin_time)
        q = self._sync_visual_q(q, goal_xy, goal_yaw, hazards_xy, obstacles_xy, gremlins_xy)
        qd = self._zero_visual_qd(qd)
        pipeline_state = self.pipeline_init(q, qd)
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
        gremlins_xy = self._gremlin_positions(
            state.info["gremlin_centers_xy"],
            getattr(pipeline_state, "time", jp.asarray(0.0, dtype=jp.float32)),
        )
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

        q = self._sync_visual_q(
            pipeline_state.q,
            next_goal_xy,
            next_goal_yaw,
            state.info["hazards_xy"],
            state.info["obstacles_xy"],
            gremlins_xy,
        )
        qd = self._zero_visual_qd(pipeline_state.qd)
        synced_pipeline_state = self.pipeline_init(q, qd)
        if hasattr(pipeline_state, "time"):
            pipeline_state = synced_pipeline_state.replace(time=pipeline_state.time)
        else:
            pipeline_state = synced_pipeline_state
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

    def _robot_obs_size(self) -> int:
        if self._parking_mode and self._robot_quat_q_start is not None:
            return self._robot_q_size + self._robot_qd_size + 1
        if self._robot_sensor_data_start is not None:
            return super()._robot_obs_size()
        q_size = self._robot_q_size
        return q_size + self._robot_qd_size

    def _robot_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        if self._parking_mode and self._robot_yaw_q_idx is not None:
            q = pipeline_state.q[: self._robot_q_size]
            qd = pipeline_state.qd[: self._robot_qd_size]
            return jp.concatenate([q[:2], self._agent_yaw(pipeline_state)[None], q[3:], qd])
        if self._parking_mode and self._robot_quat_q_start is not None:
            q = pipeline_state.q[: self._robot_q_size]
            qd = pipeline_state.qd[: self._robot_qd_size]
            return jp.concatenate([q[:2], self._agent_yaw(pipeline_state)[None], q[2:], qd])
        if self._robot_sensor_data_start is not None:
            return super()._robot_obs(pipeline_state, action)
        q = pipeline_state.q[: self._robot_q_size]
        qd = pipeline_state.qd[: self._robot_qd_size]
        return jp.concatenate(
            [
                q,
                qd,
            ]
        )

    def _sync_visual_q(
        self,
        q: jax.Array,
        goal_xy: jax.Array,
        goal_yaw: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlins_xy: jax.Array,
    ) -> jax.Array:
        layout_xy = jp.concatenate([hazards_xy, obstacles_xy, gremlins_xy], axis=0)
        flat_layout = layout_xy.reshape((-1,))
        q = q.at[self._visual_target_q_idx : self._visual_target_q_idx + 2].set(goal_xy)
        if self._parking_mode:
            q = q.at[self._visual_target_q_idx + 2].set(goal_yaw)
        return q.at[self._visual_layout_q_idx : self._visual_layout_q_idx + flat_layout.shape[0]].set(flat_layout)

    def _zero_visual_qd(self, qd: jax.Array) -> jax.Array:
        return qd.at[self._visual_target_qd_idx :].set(0.0)
