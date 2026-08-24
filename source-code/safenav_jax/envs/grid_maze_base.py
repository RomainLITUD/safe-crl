"""Shared fixed-layout grid maze support for locomotion tasks."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import xml.etree.ElementTree as ET

from brax import base
from brax.envs.base import PipelineEnv, State
import jax
from jax import numpy as jp


RESET = "r"
GOAL = "g"
MAZE_HEIGHT = 0.5
HUMANOID_TARGET_Z = 1.25

U_MAZE = (
    (1, 1, 1, 1, 1),
    (1, RESET, GOAL, GOAL, 1),
    (1, 1, 1, GOAL, 1),
    (1, GOAL, GOAL, GOAL, 1),
    (1, 1, 1, 1, 1),
)

BIG_MAZE = (
    (1, 1, 1, 1, 1, 1, 1, 1),
    (1, RESET, GOAL, 1, 1, GOAL, GOAL, 1),
    (1, GOAL, GOAL, 1, GOAL, GOAL, GOAL, 1),
    (1, 1, GOAL, GOAL, GOAL, 1, 1, 1),
    (1, GOAL, GOAL, 1, GOAL, GOAL, GOAL, 1),
    (1, GOAL, 1, GOAL, GOAL, 1, GOAL, 1),
    (1, GOAL, GOAL, GOAL, 1, GOAL, GOAL, 1),
    (1, 1, 1, 1, 1, 1, 1, 1),
)

HARDEST_MAZE = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, RESET, GOAL, GOAL, GOAL, 1, GOAL, GOAL, GOAL, GOAL, GOAL, 1),
    (1, GOAL, 1, 1, GOAL, 1, GOAL, 1, GOAL, 1, GOAL, 1),
    (1, GOAL, GOAL, GOAL, GOAL, GOAL, GOAL, 1, GOAL, GOAL, GOAL, 1),
    (1, GOAL, 1, 1, 1, 1, GOAL, 1, 1, 1, GOAL, 1),
    (1, GOAL, GOAL, 1, GOAL, 1, GOAL, GOAL, GOAL, GOAL, GOAL, 1),
    (1, 1, GOAL, 1, GOAL, 1, GOAL, 1, GOAL, 1, 1, 1),
    (1, GOAL, GOAL, 1, GOAL, GOAL, GOAL, 1, GOAL, GOAL, GOAL, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
)

CRL_LAYOUTS = {
    "u_maze": U_MAZE,
    "u_maze_eval": U_MAZE,
    "big_maze": BIG_MAZE,
    "big_maze_eval": BIG_MAZE,
    "hardest_maze": HARDEST_MAZE,
}


@dataclass(frozen=True)
class GridLayout:
    rows: int
    cols: int
    scale: float
    start_cell: tuple[int, int]
    goal_cell: tuple[int, int]
    wall_cells: tuple[tuple[int, int], ...]
    box_cells: tuple[tuple[int, int], ...]

    @property
    def start_xy(self) -> tuple[float, float]:
        return _cell_xy(self.start_cell, self.scale)

    @property
    def goal_xy(self) -> tuple[float, float]:
        return _cell_xy(self.goal_cell, self.scale)

    @property
    def boxes_xy(self) -> tuple[tuple[float, float], ...]:
        return tuple(_cell_xy(cell, self.scale) for cell in self.box_cells)


def _cell_xy(cell: tuple[int, int], scale: float) -> tuple[float, float]:
    row, col = cell
    return row * scale, col * scale


def _coerce_cell(cell) -> tuple[int, int]:
    if len(cell) != 2:
        raise ValueError(f"Grid cell must contain two indices, got {cell!r}.")
    return int(cell[0]), int(cell[1])


def _outer_wall_cells(rows: int, cols: int) -> tuple[tuple[int, int], ...]:
    cells: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            if row in (0, rows - 1) or col in (0, cols - 1):
                cells.append((row, col))
    return tuple(cells)


def _default_box_cells(
    rows: int,
    cols: int,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    num_boxes: int,
) -> tuple[tuple[int, int], ...]:
    if num_boxes < 0:
        raise ValueError("num_boxes must be non-negative.")
    route = set()
    row, col = start_cell
    goal_row, goal_col = goal_cell
    col_step = 1 if goal_col >= col else -1
    for c in range(col, goal_col + col_step, col_step):
        route.add((row, c))
    row_step = 1 if goal_row >= row else -1
    for r in range(row, goal_row + row_step, row_step):
        route.add((r, goal_col))

    candidates = [
        (r, c)
        for r in range(1, rows - 1)
        for c in range(1, cols - 1)
        if (r, c) not in route and (r, c) not in (start_cell, goal_cell)
    ]
    if num_boxes > len(candidates):
        raise ValueError(
            f"num_boxes={num_boxes} is too large for a {rows}x{cols} grid; "
            f"at most {len(candidates)} default cells remain off the reserved route."
        )
    return tuple(candidates[:num_boxes])


def _find_layout_cell(layout, marker: str) -> tuple[int, int] | None:
    for row, values in enumerate(layout):
        for col, value in enumerate(values):
            if value == marker:
                return row, col
    return None


def _validate_connected(
    rows: int,
    cols: int,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
    blocked_cells: set[tuple[int, int]],
) -> None:
    if start_cell in blocked_cells:
        raise ValueError(f"start_cell={start_cell} is blocked.")
    if goal_cell in blocked_cells:
        raise ValueError(f"goal_cell={goal_cell} is blocked.")
    seen = {start_cell}
    queue: deque[tuple[int, int]] = deque([start_cell])
    while queue:
        row, col = queue.popleft()
        if (row, col) == goal_cell:
            return
        for nxt in ((row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)):
            nr, nc = nxt
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if nxt in blocked_cells or nxt in seen:
                continue
            seen.add(nxt)
            queue.append(nxt)
    raise ValueError(
        f"No connected free route from start_cell={start_cell} to goal_cell={goal_cell}."
    )


def make_grid_layout(
    *,
    grid_rows: int,
    grid_cols: int,
    grid_scale: float,
    num_boxes: int,
    box_cells=None,
    start_cell=None,
    goal_cell=None,
    crl_like: bool = False,
    maze_layout_name: str = "u_maze",
) -> GridLayout:
    """Builds a fixed grid layout and validates start-goal connectivity."""
    if crl_like:
        try:
            layout = CRL_LAYOUTS[maze_layout_name]
        except KeyError as exc:
            raise ValueError(f"Unknown crl_like maze_layout_name={maze_layout_name!r}.") from exc
        rows, cols = len(layout), len(layout[0])
        start = _find_layout_cell(layout, RESET)
        if start is None:
            raise ValueError(f"crl_like layout {maze_layout_name!r} has no reset cell.")
        bottom_right = (rows - 2, cols - 2)
        goal = bottom_right if layout[bottom_right[0]][bottom_right[1]] != 1 else None
        if goal is None:
            goals = [
                (r, c)
                for r, values in enumerate(layout)
                for c, value in enumerate(values)
                if value == GOAL
            ]
            if not goals:
                raise ValueError(f"crl_like layout {maze_layout_name!r} has no goal cells.")
            goal = goals[-1]
        wall_cells = _outer_wall_cells(rows, cols)
        wall_set = set(wall_cells)
        interior_boxes = tuple(
            (r, c)
            for r, values in enumerate(layout)
            for c, value in enumerate(values)
            if value == 1 and (r, c) not in wall_set
        )
    else:
        rows, cols = int(grid_rows), int(grid_cols)
        if rows < 4 or cols < 4:
            raise ValueError("grid_rows and grid_cols must both be at least 4.")
        start = _coerce_cell(start_cell) if start_cell is not None else (1, 1)
        goal = _coerce_cell(goal_cell) if goal_cell is not None else (rows - 2, cols - 2)
        wall_cells = _outer_wall_cells(rows, cols)
        interior_boxes = (
            tuple(_coerce_cell(cell) for cell in box_cells)
            if box_cells is not None
            else _default_box_cells(rows, cols, start, goal, int(num_boxes))
        )

    for cell in (start, goal, *interior_boxes):
        row, col = cell
        if row <= 0 or row >= rows - 1 or col <= 0 or col >= cols - 1:
            raise ValueError(f"Cell {cell} must be inside the outer wall border.")

    wall_set = set(wall_cells)
    box_set = set(interior_boxes)
    if len(box_set) != len(interior_boxes):
        raise ValueError("box_cells must not contain duplicates.")
    if wall_set & box_set:
        raise ValueError("box_cells cannot overlap outer wall cells.")
    _validate_connected(rows, cols, start, goal, wall_set | box_set)
    return GridLayout(
        rows=rows,
        cols=cols,
        scale=float(grid_scale),
        start_cell=start,
        goal_cell=goal,
        wall_cells=tuple(wall_cells),
        box_cells=tuple(interior_boxes),
    )


def add_grid_geoms_to_xml(
    root: ET.Element,
    layout: GridLayout,
    *,
    wall_height: float,
    box_height: float,
    target_joint_range: float | None = None,
) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF is missing a worldbody.")

    floor = worldbody.find("./geom[@name='floor']")
    if floor is not None:
        half_extent = max(layout.rows, layout.cols) * layout.scale
        floor.set("size", f"{half_extent:.6f} {half_extent:.6f} 0.125")

    half = 0.5 * layout.scale
    for idx, (row, col) in enumerate(layout.wall_cells):
        x, y = _cell_xy((row, col), layout.scale)
        z = 0.5 * wall_height
        ET.SubElement(
            worldbody,
            "geom",
            name=f"grid_outer_wall_{idx}",
            type="box",
            pos=f"{x:.6f} {y:.6f} {z:.6f}",
            size=f"{half:.6f} {half:.6f} {z:.6f}",
            contype="1",
            conaffinity="1",
            rgba="0.35 0.35 0.35 1",
        )

    for idx, (row, col) in enumerate(layout.box_cells):
        x, y = _cell_xy((row, col), layout.scale)
        z = 0.5 * box_height
        ET.SubElement(
            worldbody,
            "geom",
            name=f"grid_cost_box_{idx}",
            type="box",
            pos=f"{x:.6f} {y:.6f} {z:.6f}",
            size=f"{half:.6f} {half:.6f} {z:.6f}",
            contype="0",
            conaffinity="0",
            rgba="0.85 0.25 0.18 0.65",
        )

    if target_joint_range is not None:
        for joint_name in ("target_x", "target_y"):
            joint = root.find(f".//joint[@name='{joint_name}']")
            if joint is not None:
                joint.set("range", f"{-target_joint_range:.6f} {target_joint_range:.6f}")


class GridMazeEnv(PipelineEnv):
    """Base class for fixed grid maze locomotion environments."""

    def __init__(
        self,
        *,
        sys: base.System,
        backend: str,
        n_frames: int,
        episode_length: int,
        reset_noise_scale: float,
        goal_radius: float,
        healthy_z_range: tuple[float, float],
        terminate_when_unhealthy: bool,
        healthy_reward: float,
        ctrl_cost_weight: float,
        layout: GridLayout,
        goal_dim: int,
        robot_obs_dim: int,
        success_link_name: str,
        target_link_name: str = "target",
        fallback_link_radius: float = 0.25,
        include_box_layout_obs: bool = False,
        terminate_on_cost: bool = False,
        cost_limit_max: float = 25.0,
        **kwargs,
    ):
        super().__init__(sys=sys, backend=backend, n_frames=n_frames, **kwargs)
        self._episode_length = episode_length
        self._reset_noise_scale = reset_noise_scale
        self._goal_radius = goal_radius
        self._healthy_z_range = healthy_z_range
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_reward = healthy_reward
        self._ctrl_cost_weight = ctrl_cost_weight
        self._layout = layout
        self._start_xy = jp.asarray(layout.start_xy)
        self._goal_xy = jp.asarray(layout.goal_xy)
        self._boxes_xy = jp.asarray(layout.boxes_xy).reshape((-1, 2))
        self._box_half_size = jp.array([0.5 * layout.scale, 0.5 * layout.scale])
        self._success_link_idx = self._link_index(success_link_name)
        self._target_link_idx = self._link_index(target_link_name)
        self._fallback_link_radius = fallback_link_radius
        self._include_box_layout_obs = include_box_layout_obs
        self._terminate_on_cost = terminate_on_cost
        self._cost_limit_max = cost_limit_max
        self._robot_link_mask = jp.asarray(
            [idx != self._target_link_idx for idx in range(len(self.sys.link_names))]
        )
        target_body_id = self._target_link_idx + 1
        self._robot_geom_mask = jp.asarray(
            (self.sys.geom_bodyid != 0) & (self.sys.geom_bodyid != target_body_id)
        )
        self._geom_type = jp.asarray(self.sys.geom_type)
        self._geom_size = jp.asarray(self.sys.geom_size)
        self._robot_sphere_geom_mask = self._robot_geom_mask & (self._geom_type == 2)
        self._robot_capsule_geom_mask = self._robot_geom_mask & (self._geom_type == 3)
        self._geom_radius = self._geom_size[:, 0]
        self._geom_capsule_half_length = self._geom_size[:, 1]

        self._robot_obs_dim = robot_obs_dim
        self._goal_dim = goal_dim
        self._box_layout_obs_dim = 2 * self._boxes_xy.shape[0] if include_box_layout_obs else 0
        goal_start = robot_obs_dim + self._box_layout_obs_dim
        self.state_dim = goal_start + goal_dim
        self.goal_indices = jp.arange(goal_start, goal_start + goal_dim)
        self.completion_goal_indices = jp.arange(goal_dim)

    def reset(self, rng: jax.Array) -> State:
        rng, q_rng, qd_rng = jax.random.split(rng, 3)
        q = self.sys.init_q
        qd = jp.zeros(self.sys.qd_size())
        if self._reset_noise_scale:
            low, hi = -self._reset_noise_scale, self._reset_noise_scale
            q = q + jax.random.uniform(q_rng, (self.sys.q_size(),), minval=low, maxval=hi)
            qd = hi * jax.random.normal(qd_rng, (self.sys.qd_size(),))
        q = q.at[:2].set(self._start_xy)
        q = q.at[-2:].set(self._goal_xy)
        qd = qd.at[-2:].set(0.0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()))
        zero = jp.zeros(())
        metrics = self._metrics(
            reward=zero,
            healthy_reward=zero,
            ctrl_cost=zero,
            forward_reward=zero,
            dist=zero,
            success=zero,
            cost_boxes=zero,
            agent_goal=jp.zeros((self._goal_dim,)),
        )
        info = {
            "seed": zero,
            "goal_xy": self._goal_xy,
            "boxes_xy": self._boxes_xy,
            "episode_cost": zero,
            "agent_yaw": zero,
        }
        state = State(pipeline_state, obs, zero, zero, metrics)
        state.info.update(info)
        return state

    def step(self, state: State, action: jax.Array) -> State:
        action_for_physics = self._action_for_physics(action)
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action_for_physics)

        seed = state.info["seed"]
        if "steps" in state.info:
            seed = seed + jp.where(state.info["steps"], 0.0, 1.0)

        obs = self._get_obs(pipeline_state, action_for_physics)
        agent_goal = self._achieved_goal(pipeline_state)
        goal = self._goal_for_distance()
        dist = jp.linalg.norm(agent_goal - goal)
        success = (dist <= self._goal_radius).astype(float)
        is_healthy = self._is_healthy(pipeline_state)
        healthy_reward = self._healthy_reward_value(is_healthy)
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action_for_physics))
        forward_reward = self._forward_reward(pipeline_state0, pipeline_state)
        reward = -dist + healthy_reward - ctrl_cost
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else jp.zeros(())
        cost_boxes = self._box_cost(pipeline_state)
        episode_cost = state.info["episode_cost"] + cost_boxes
        if self._terminate_on_cost:
            done = jp.logical_or(done.astype(bool), episode_cost >= self._cost_limit_max).astype(float)

        state.metrics.update(
            self._metrics(
                reward=reward,
                healthy_reward=healthy_reward,
                ctrl_cost=ctrl_cost,
                forward_reward=forward_reward,
                dist=dist,
                success=success,
                cost_boxes=cost_boxes,
                agent_goal=agent_goal,
            )
        )
        state.info.update(seed=seed, goal_xy=self._goal_xy, boxes_xy=self._boxes_xy, episode_cost=episode_cost)
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
        )

    def _get_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        goal_xy = pipeline_state.x.pos[self._target_link_idx, :2]
        if self._goal_dim == 3:
            goal = jp.concatenate([goal_xy, jp.array([HUMANOID_TARGET_Z])])
        else:
            goal = goal_xy
        obs_parts = [self._robot_obs(pipeline_state, action)]
        if self._include_box_layout_obs:
            obs_parts.append(self._box_layout_obs())
        obs_parts.append(goal)
        return jp.concatenate(obs_parts)

    def _box_layout_obs(self) -> jax.Array:
        return self._boxes_xy.reshape((-1,))

    def _robot_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        raise NotImplementedError

    def _achieved_goal(self, pipeline_state: base.State) -> jax.Array:
        raise NotImplementedError

    def _action_for_physics(self, action: jax.Array) -> jax.Array:
        return action

    def _forward_reward(self, pipeline_state0: base.State, pipeline_state: base.State) -> jax.Array:
        del pipeline_state0, pipeline_state
        return jp.zeros(())

    def _goal_for_distance(self) -> jax.Array:
        if self._goal_dim == 3:
            return jp.concatenate([self._goal_xy, jp.array([HUMANOID_TARGET_Z])])
        return self._goal_xy

    def _is_healthy(self, pipeline_state: base.State) -> jax.Array:
        min_z, max_z = self._healthy_z_range
        z = pipeline_state.x.pos[self._success_link_idx, 2]
        is_healthy = jp.where(z < min_z, 0.0, 1.0)
        return jp.where(z > max_z, 0.0, is_healthy)

    def _healthy_reward_value(self, is_healthy: jax.Array) -> jax.Array:
        if self._terminate_when_unhealthy:
            return jp.asarray(self._healthy_reward)
        return self._healthy_reward * is_healthy

    def _metrics(
        self,
        *,
        reward: jax.Array,
        healthy_reward: jax.Array,
        ctrl_cost: jax.Array,
        forward_reward: jax.Array,
        dist: jax.Array,
        success: jax.Array,
        cost_boxes: jax.Array,
        agent_goal: jax.Array,
    ) -> dict[str, jax.Array]:
        metrics = {
            "reward": reward,
            "success": success,
            "success_easy": (dist < 2.0).astype(float),
            "cost": cost_boxes,
            "cost_boxes": cost_boxes,
            "dist": dist,
            "x_position": agent_goal[0],
            "y_position": agent_goal[1],
            "distance_from_origin": jp.linalg.norm(agent_goal),
            "x_velocity": forward_reward,
            "y_velocity": jp.zeros_like(forward_reward),
            "goal_x": self._goal_xy[0],
            "goal_y": self._goal_xy[1],
            "forward_reward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
        }
        if self._goal_dim == 3:
            metrics["z_position"] = agent_goal[2]
            metrics["goal_z"] = jp.asarray(HUMANOID_TARGET_Z)
        return metrics

    def _box_cost(self, pipeline_state: base.State) -> jax.Array:
        if self._boxes_xy.shape[0] == 0:
            return jp.zeros(())
        if not hasattr(pipeline_state, "geom_xpos"):
            return self._link_box_cost(pipeline_state)
        geom_pos, seg_a, seg_b = self._robot_geom_segments(pipeline_state)
        sphere_overlap = self._sphere_box_overlap(geom_pos[:, :2], self._boxes_xy)
        capsule_overlap = self._segment_box_overlap(seg_a[:, :2], seg_b[:, :2], self._boxes_xy)
        overlap = (
            (sphere_overlap & self._robot_sphere_geom_mask[:, None])
            | (capsule_overlap & self._robot_capsule_geom_mask[:, None])
        )
        return jp.any(overlap).astype(float)

    def _link_box_cost(self, pipeline_state: base.State) -> jax.Array:
        points_xy = pipeline_state.x.pos[:, :2]
        box_min = self._boxes_xy[None, :, :] - self._box_half_size
        box_max = self._boxes_xy[None, :, :] + self._box_half_size
        closest = jp.minimum(jp.maximum(points_xy[:, None, :], box_min), box_max)
        delta = points_xy[:, None, :] - closest
        dist_sq = jp.sum(delta * delta, axis=-1)
        overlap = dist_sq <= self._fallback_link_radius**2
        return jp.any(overlap & self._robot_link_mask[:, None]).astype(float)

    def _robot_geom_segments(self, pipeline_state: base.State) -> tuple[jax.Array, jax.Array, jax.Array]:
        geom_pos = pipeline_state.geom_xpos
        geom_xmat = pipeline_state.geom_xmat.reshape((-1, 3, 3))
        capsule_axis = geom_xmat[:, :, 2]
        seg_a = geom_pos - capsule_axis * self._geom_capsule_half_length[:, None]
        seg_b = geom_pos + capsule_axis * self._geom_capsule_half_length[:, None]
        return geom_pos, seg_a, seg_b

    def _sphere_box_overlap(self, points_xy: jax.Array, boxes_xy: jax.Array) -> jax.Array:
        box_min = boxes_xy[None, :, :] - self._box_half_size
        box_max = boxes_xy[None, :, :] + self._box_half_size
        closest = jp.minimum(jp.maximum(points_xy[:, None, :], box_min), box_max)
        delta = points_xy[:, None, :] - closest
        dist_sq = jp.sum(delta * delta, axis=-1)
        return dist_sq <= (self._geom_radius[:, None] ** 2)

    def _segment_box_overlap(
        self,
        seg_a_xy: jax.Array,
        seg_b_xy: jax.Array,
        boxes_xy: jax.Array,
    ) -> jax.Array:
        start = seg_a_xy[:, None, :]
        end = seg_b_xy[:, None, :]
        delta = end - start
        half = self._box_half_size[None, None, :] + self._geom_radius[:, None, None]
        box_min = boxes_xy[None, :, :] - half
        box_max = boxes_xy[None, :, :] + half
        abs_delta = jp.abs(delta)
        inside_parallel = (start >= box_min) & (start <= box_max)
        inv_delta = jp.where(abs_delta > 1e-8, 1.0 / delta, 0.0)
        t1 = (box_min - start) * inv_delta
        t2 = (box_max - start) * inv_delta
        t_low = jp.minimum(t1, t2)
        t_high = jp.maximum(t1, t2)
        t_low = jp.where(abs_delta > 1e-8, t_low, jp.where(inside_parallel, -jp.inf, jp.inf))
        t_high = jp.where(abs_delta > 1e-8, t_high, jp.where(inside_parallel, jp.inf, -jp.inf))
        t_enter = jp.max(t_low, axis=-1)
        t_exit = jp.min(t_high, axis=-1)
        return (t_exit >= jp.maximum(t_enter, 0.0)) & (t_enter <= 1.0)

    def _link_index(self, link_name: str) -> int:
        try:
            return self.sys.link_names.index(link_name)
        except ValueError as exc:
            raise ValueError(f"Link {link_name!r} is not present in system links: {self.sys.link_names}") from exc
