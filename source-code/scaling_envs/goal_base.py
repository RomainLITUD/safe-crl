"""Shared cost-free goal-reaching environment logic."""

from __future__ import annotations

from abc import ABC, abstractmethod

from brax import base
from brax.envs.base import PipelineEnv, State
import jax
from jax import numpy as jp


class ScalingGoalEnv(PipelineEnv, ABC):
    """Renderable fixed-start, fixed-goal-per-episode locomotion task."""

    def __init__(
        self,
        *,
        sys: base.System,
        backend: str,
        n_frames: int,
        episode_length: int,
        reset_noise_scale: float,
        playground_size: float,
        min_goal_dist: float,
        eval_min_goal_dist: float,
        evaluation_mode: bool,
        goal_radius: float,
        healthy_z_range: tuple[float, float] | None,
        robot_q_size: int,
        robot_qd_size: int,
        target_q_idx: int,
        target_qd_idx: int,
        **kwargs,
    ):
        if playground_size <= 0.0:
            raise ValueError("playground_size must be positive.")
        if min_goal_dist < 0.0 or eval_min_goal_dist < 0.0:
            raise ValueError("Goal-distance constraints must be non-negative.")
        active_min_dist = eval_min_goal_dist if evaluation_mode else min_goal_dist
        if active_min_dist > (2.0**0.5) * playground_size:
            raise ValueError(
                "The minimum goal distance is unreachable within the configured playground."
            )
        if goal_radius <= 0.0:
            raise ValueError("goal_radius must be positive.")

        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)
        self._episode_length = episode_length
        self._reset_noise_scale = reset_noise_scale
        self._playground_size = playground_size
        self._min_goal_dist = active_min_dist
        self._training_min_goal_dist = min_goal_dist
        self._eval_min_goal_dist = eval_min_goal_dist
        self._evaluation_mode = evaluation_mode
        self._goal_radius = goal_radius
        self._healthy_z_range = healthy_z_range
        self._robot_q_size = robot_q_size
        self._robot_qd_size = robot_qd_size
        self._target_q_idx = target_q_idx
        self._target_qd_idx = target_qd_idx

        self._robot_obs_dim = self._robot_obs_size()
        self.state_dim = self._robot_obs_dim
        self.goal_indices = jp.arange(self._robot_obs_dim, self._robot_obs_dim + 2)
        self.scaling_crl_goal_indices = jp.array([0, 1], dtype=jp.int32)
        self.raw_goal_dim = 2
        self.relabel_goal_dim = 2
        self.layout_obs_dim = 0
        self.layout_start_idx = self._robot_obs_dim
        self.layout_end_idx = self._robot_obs_dim
        self._layout_obs_dim = 0
        self._layout_is_lidar_obs = False

    def reset(self, rng: jax.Array) -> State:
        rng, q_rng, qd_rng, goal_rng = jax.random.split(rng, 4)
        q = self.sys.init_q
        qd = jp.zeros(self.sys.qd_size())
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

        q = q.at[:2].set(0.0)
        q = q.at[2:7].set(self.sys.init_q[2:7])
        goal_xy = self._sample_goal(goal_rng)
        q = q.at[self._target_q_idx : self._target_q_idx + 2].set(goal_xy)
        qd = qd.at[self._target_qd_idx :].set(0.0)
        pipeline_state = self.pipeline_init(q, qd)
        zero = jp.asarray(0.0)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()), goal_xy)
        metrics = self._metrics(zero, self._distance(pipeline_state, goal_xy), pipeline_state, goal_xy)
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                "steps": jp.asarray(0, dtype=jp.int32),
                "seed": jp.asarray(0, dtype=jp.int32),
                "goal_xy": goal_xy,
                "episode_cost": zero,
            },
        )

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        goal_xy = state.info["goal_xy"]
        dist = self._distance(pipeline_state, goal_xy)
        success = (dist <= self._goal_radius).astype(jp.float32)

        wrapped_by_training = "truncation" in state.info
        steps = state.info["steps"] + 1
        done = jp.asarray(False) if wrapped_by_training else steps >= self._episode_length
        if self._healthy_z_range is not None:
            min_z, max_z = self._healthy_z_range
            root_z = pipeline_state.x.pos[0, 2]
            done = jp.logical_or(done, jp.logical_or(root_z < min_z, root_z > max_z))

        obs = self._get_obs(pipeline_state, action, goal_xy)
        metrics = self._metrics(success, dist, pipeline_state, goal_xy)
        seed = state.info["seed"] + jp.where(
            state.info["steps"],
            jp.zeros_like(state.info["seed"]),
            jp.ones_like(state.info["seed"]),
        )
        info = dict(state.info)
        info.update(
            {
                "steps": state.info["steps"] if wrapped_by_training else steps,
                "seed": seed,
                "goal_xy": goal_xy,
                "episode_cost": jp.asarray(0.0),
            }
        )
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=success,
            done=done.astype(jp.float32),
            metrics=metrics,
            info=info,
        )

    def _sample_goal(self, rng: jax.Array) -> jax.Array:
        limit = jp.asarray(self._playground_size, dtype=jp.float32)
        min_dist = jp.asarray(self._min_goal_dist, dtype=jp.float32)

        def sample(key: jax.Array) -> tuple[jax.Array, jax.Array]:
            key, sample_key = jax.random.split(key)
            goal = jax.random.uniform(sample_key, (2,), minval=-limit, maxval=limit)
            return key, goal

        rng, goal_xy = sample(rng)

        def cond(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            _, candidate = carry
            return jp.linalg.norm(candidate) < min_dist

        def body(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            key, _ = carry
            return sample(key)

        _, goal_xy = jax.lax.while_loop(cond, body, (rng, goal_xy))
        return goal_xy

    def _distance(self, pipeline_state: base.State, goal_xy: jax.Array) -> jax.Array:
        return jp.linalg.norm(pipeline_state.x.pos[0, :2] - goal_xy)

    def _get_obs(
        self,
        pipeline_state: base.State,
        action: jax.Array,
        goal_xy: jax.Array,
    ) -> jax.Array:
        return jp.concatenate([self._robot_obs(pipeline_state, action), goal_xy])

    def _metrics(
        self,
        success: jax.Array,
        dist: jax.Array,
        pipeline_state: base.State,
        goal_xy: jax.Array,
    ) -> dict[str, jax.Array]:
        zero = jp.zeros_like(success)
        root_xy = pipeline_state.x.pos[0, :2]
        return {
            "reward": success,
            "success": success,
            "cost": zero,
            "dist": dist,
            "x_position": root_xy[0],
            "y_position": root_xy[1],
            "goal_x": goal_xy[0],
            "goal_y": goal_xy[1],
        }

    @abstractmethod
    def _robot_obs_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def _robot_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        raise NotImplementedError
