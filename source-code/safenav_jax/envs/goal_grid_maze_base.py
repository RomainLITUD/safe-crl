"""Scaling-maze-style ant/humanoid goal-grid environments."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco

from safenav_jax.envs._humanoid_actuation import apply_spring_humanoid_gear_for_mjx


_HAZARD_RGBA = "0 0 1 0.25"
RESET = R = "r"
GOAL = G = "g"

BIG_PITFALL_LAYOUT = (
    (G, 1, G, G, G, G, G),
    (G, 1, G, 1, 1, 1, 1),
    (G, 1, G, G, G, G, G),
    (G, G, 1, R, 1, G, G),
    (G, G, G, G, G, 1, G),
    (1, 1, 1, 1, G, 1, G),
    (G, G, G, G, G, 1, G),
)
BIG_PITFALL_EVAL_LAYOUT = BIG_PITFALL_LAYOUT

LOOP_PITFALL_LAYOUT = (
    (G, G, G, G, G),
    (G, 1, R, 1, G),
    (G, 1, G, 1, G),
    (G, 1, G, 1, G),
    (G, G, G, G, G),
)
LOOP_PITFALL_EVAL_LAYOUT = LOOP_PITFALL_LAYOUT

HARDEST_PITFALL_LAYOUT = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, R, G, G, G, 1, G, G, G, G, G, 1),
    (1, G, 1, 1, G, 1, G, 1, G, 1, G, 1),
    (1, G, G, G, G, G, G, 1, G, G, G, 1),
    (1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1),
    (1, G, G, 1, G, 1, G, G, G, G, G, 1),
    (1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1),
    (1, G, G, 1, G, G, G, 1, G, G, G, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
)  # Ant Hardest Pitfall

HARDEST_PITFALL_EVAL_LAYOUT = (
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    (1, R, G, G, G, 1, G, G, G, G, G, 1),
    (1, G, 1, 1, G, 1, G, 1, G, 1, G, 1),
    (1, G, G, G, G, G, G, 1, G, G, G, 1),
    (1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1),
    (1, G, G, 1, G, 1, G, G, G, G, G, 1),
    (1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1),
    (1, G, G, 1, G, G, G, 1, G, G, G, 1),
    (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
)


# DEFAULT_LAYOUT = (
#     (R, 0, G, 0, G, 0),
#     (0, 1, 0, 0, 1, G),
#     (G, 0, 1, G, 0, 0),
#     (0, 0, G, 1, 0, G),
#     (G, 1, 0, 0, 1, 0),
#     (0, G, 1, G, 0, 1),
# )

# DEFAULT_EVAL_LAYOUT = (
#     (R, 0, 0, 0, G, 0),
#     (0, 1, 0, 0, 1, G),
#     (0, 0, 1, 0, 0, 0),
#     (0, 0, 0, 1, 0, G),
#     (G, 1, 0, 0, 1, 0),
#     (0, G, 1, G, 0, 1),
# )

# DEFAULT_LAYOUT = (
#     (R, 0, G, 0, G, 0),
#     (0, 1, 0, 0, 1, G),
#     (G, 0, 1, G, 0, 0),
#     (0, 0, G, 1, 0, G),
#     (G, 1, 0, 0, 1, 0),
#     (0, G, 1, G, 0, 1),
# )

# DEFAULT_EVAL_LAYOUT = (
#     (R, 0, 0, 0, G, 0),
#     (0, 1, 0, 0, 1, G),
#     (0, 0, 1, 0, 0, 0),
#     (0, 0, 0, 1, 0, G),
#     (G, 1, 0, 0, 1, 0),
#     (0, G, 1, G, 0, 1),
# )

_GOAL_GRID_LAYOUTS = {
    # Keep the legacy default mapped to the previously active layout.
    "default": HARDEST_PITFALL_LAYOUT,
    "default_eval": HARDEST_PITFALL_EVAL_LAYOUT,
    "big_pitfall": BIG_PITFALL_LAYOUT,
    "big_pitfall_eval": BIG_PITFALL_EVAL_LAYOUT,
    "loop_pitfall": LOOP_PITFALL_LAYOUT,
    "loop_pitfall_eval": LOOP_PITFALL_EVAL_LAYOUT,
    "hardest_pitfall": HARDEST_PITFALL_LAYOUT,
    "hardest_pitfall_eval": HARDEST_PITFALL_EVAL_LAYOUT,
}


def _parse_goal_grid_layout(
    layout_name: str,
) -> tuple[
    tuple[tuple[object, ...], ...],
    tuple[int, int],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    try:
        layout = _GOAL_GRID_LAYOUTS[layout_name]
    except KeyError as exc:
        supported = ", ".join(sorted(_GOAL_GRID_LAYOUTS))
        raise ValueError(
            f"Unknown goal-grid layout {layout_name!r}; supported layouts: {supported}."
        ) from exc

    if not layout or not layout[0]:
        raise ValueError("Goal-grid layout must be non-empty.")
    num_cols = len(layout[0])
    if any(len(row) != num_cols for row in layout):
        raise ValueError("Goal-grid layout must be rectangular.")

    valid_symbols = {0, 1, RESET, GOAL}
    invalid = sorted(
        {cell for row in layout for cell in row if cell not in valid_symbols},
        key=str,
    )
    if invalid:
        raise ValueError(f"Goal-grid layout contains invalid symbols: {invalid!r}.")

    reset_cells = tuple(
        (row, col)
        for row, values in enumerate(layout)
        for col, cell in enumerate(values)
        if cell == RESET
    )
    if len(reset_cells) != 1:
        raise ValueError(
            f"Goal-grid layout must contain exactly one R cell; found {len(reset_cells)}."
        )
    goal_cells = tuple(
        (row, col)
        for row, values in enumerate(layout)
        for col, cell in enumerate(values)
        if cell == GOAL
    )
    if not goal_cells:
        raise ValueError("Goal-grid layout must contain at least one G cell.")
    hazard_cells = tuple(
        (row, col)
        for row, values in enumerate(layout)
        for col, cell in enumerate(values)
        if cell == 1
    )

    num_rows = len(layout)
    reset_cell = reset_cells[0]
    reachable = {reset_cell}
    frontier = [reset_cell]
    while frontier:
        row, col = frontier.pop()
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if not (0 <= next_row < num_rows and 0 <= next_col < num_cols):
                continue
            cell = (next_row, next_col)
            if cell in reachable or layout[next_row][next_col] == 1:
                continue
            reachable.add(cell)
            frontier.append(cell)
    disconnected_goals = tuple(cell for cell in goal_cells if cell not in reachable)
    if disconnected_goals:
        raise ValueError(
            "Every G cell must be four-connected to R through non-hazard cells; "
            f"disconnected goals: {disconnected_goals!r}."
        )

    return layout, reset_cell, goal_cells, hazard_cells


def _asset_xml_path(agent: str) -> Path:
    filename = {
        "ant": "scaling_ant_maze.xml",
        "humanoid": "scaling_humanoid_maze.xml",
    }[agent]
    return Path(__file__).resolve().parents[1] / "assets" / "xmls" / filename


def _set_ant_init_qpos(tree: ET.ElementTree, render_task_objects: bool) -> None:
    numeric = tree.find(".//numeric[@name='init_qpos']")
    if numeric is None:
        return
    values = numeric.get("data", "").split()
    # The source stores robot qpos without root x/y and includes two trailing
    # target coordinates.  Headless goal-grid models remove that target body.
    if len(values) == 15:
        values = ["0", "0", *values]
    if not render_task_objects and len(values) == 17:
        values = values[:-2]
    numeric.set("data", " ".join(values))


def _remove_target_body(worldbody: ET.Element) -> None:
    for body in list(worldbody):
        if body.tag == "body" and body.get("name") == "target":
            worldbody.remove(body)
            return


def _add_hazard_visuals(
    worldbody: ET.Element,
    num_hazards: int,
    hazard_radius: float,
    hazard_height: float,
) -> None:
    # Pitfalls render as low ground pads rather than tall pillars: clamp the
    # visual height so each cylinder sits flat on the floor.
    visual_height = min(float(hazard_height), 0.2)
    for i in range(num_hazards):
        body = ET.SubElement(
            worldbody,
            "body",
            name=f"hazard_{i}",
            pos=f"0 0 {0.5 * visual_height:.8f}",
        )
        ET.SubElement(
            body,
            "joint",
            name=f"hazard_{i}_x",
            type="slide",
            axis="1 0 0",
            pos="0 0 0",
            range="-100 100",
            limited="true",
            damping="0",
            stiffness="0",
            armature="0",
        )
        ET.SubElement(
            body,
            "joint",
            name=f"hazard_{i}_y",
            type="slide",
            axis="0 1 0",
            pos="0 0 0",
            range="-100 100",
            limited="true",
            damping="0",
            stiffness="0",
            armature="0",
        )
        ET.SubElement(
            body,
            "geom",
            name=f"hazard_{i}",
            type="cylinder",
            pos="0 0 0",
            size=f"{hazard_radius:.8f} {0.5 * visual_height:.8f}",
            rgba=_HAZARD_RGBA,
            contype="0",
            conaffinity="0",
            mass="0.001",
        )


def _load_goal_grid_xml(
    agent: str,
    render_task_objects: bool,
    num_hazards: int,
    hazard_radius: float,
    hazard_height: float,
) -> bytes:
    tree = ET.parse(_asset_xml_path(agent))
    if agent == "ant":
        _set_ant_init_qpos(tree, render_task_objects)
    worldbody = tree.find(".//worldbody")
    if worldbody is None:
        raise ValueError("Goal-grid XML has no worldbody.")
    if render_task_objects:
        target_geom = worldbody.find("./body[@name='target']/geom")
        if target_geom is None:
            raise ValueError("Renderable goal-grid XML has no target geom.")
        target_geom.set("contype", "0")
        target_geom.set("conaffinity", "0")
        _add_hazard_visuals(worldbody, num_hazards, hazard_radius, hazard_height)
        if agent == "ant":
            numeric = tree.find(".//numeric[@name='init_qpos']")
            if numeric is not None:
                data = numeric.get("data", "")
                numeric.set("data", data + " " + " ".join(["0"] * (2 * num_hazards)))
    else:
        _remove_target_body(worldbody)
    return ET.tostring(tree.getroot())


class GoalGridMazeBase(PipelineEnv):
    """Goal-grid task built directly on scaling-maze robot XMLs."""

    def __init__(
        self,
        agent: str,
        render_hazards: bool,
        grid_cell_size: float,
        grid_layout_name: str,
        eval_grid_layout_name: str,
        evaluation_mode: bool,
        min_goal_cell_distance: float,
        hazard_radius: float | None,
        goal_radius: float,
        hazard_height: float,
        backend: str,
        n_frames: int | None,
        episode_length: int,
        reset_noise_scale: float,
        healthy_z_range: tuple[float, float],
        robot_cost_margin: float,
        object_boundary: bool,
        terminate_on_cost: bool,
        cost_limit_max: float,
        relocate_objects_on_reset: bool,
        fixed_object_layout_seed: int,
        different_object_layout_per_env: bool,
        goal_respawn_on_success: bool,
        fixed_goal_on_reset: bool,
        fixed_agent_on_reset: bool,
        layout_lidar_num_bins: int,
        layout_lidar_max_dist: float | None,
        full_robot_observation: bool,
        humanoid_use_spring_gear: bool = False,
        **kwargs,
    ):
        if grid_cell_size <= 0.0:
            raise ValueError("grid_cell_size must be positive.")
        if layout_lidar_num_bins < 1:
            raise ValueError("layout_lidar_num_bins must be at least 1.")
        if robot_cost_margin < 0.0:
            raise ValueError("robot_cost_margin must be non-negative.")
        if min_goal_cell_distance < 0.0:
            raise ValueError("min_goal_cell_distance must be non-negative.")
        del kwargs

        self._agent = agent
        self._render_hazards = render_hazards
        _parse_goal_grid_layout(grid_layout_name)
        _parse_goal_grid_layout(eval_grid_layout_name)
        self._train_grid_layout_name = grid_layout_name
        self._eval_grid_layout_name = eval_grid_layout_name
        self._evaluation_mode = bool(evaluation_mode)
        selected_layout_name = (
            eval_grid_layout_name if self._evaluation_mode else grid_layout_name
        )
        layout, reset_cell, goal_cells, hazard_cells = _parse_goal_grid_layout(
            selected_layout_name
        )
        self._grid_layout_name = selected_layout_name
        self._grid_layout = layout
        self._grid_cell_size = float(grid_cell_size)
        self._grid_num_rows = len(layout)
        self._grid_num_cols = len(layout[0])
        self._grid_num_total_cells = self._grid_num_rows * self._grid_num_cols
        self._grid_playground_extent = jp.asarray(
            [self._grid_num_rows, self._grid_num_cols],
            dtype=jp.float32,
        ) * float(grid_cell_size)
        self._grid_playground_lower = jp.full(
            (2,),
            -0.5 * float(grid_cell_size),
            dtype=jp.float32,
        )
        self._grid_playground_upper = (
            self._grid_playground_lower + self._grid_playground_extent
        )
        self._grid_reset_cell = jp.asarray(reset_cell, dtype=jp.int32)
        self._grid_goal_cells = jp.asarray(goal_cells, dtype=jp.int32)
        self._grid_hazard_cells = jp.asarray(hazard_cells, dtype=jp.int32).reshape((-1, 2))
        self._grid_goal_mask = jp.asarray(
            [cell == GOAL for row in layout for cell in row],
            dtype=bool,
        )
        self._grid_min_goal_cell_distance = float(min_goal_cell_distance)
        self._num_hazards = len(hazard_cells)
        if not any(
            math.hypot(
                goal_row - reset_cell[0],
                goal_col - reset_cell[1],
            )
            >= self._grid_min_goal_cell_distance
            for goal_row, goal_col in goal_cells
        ):
            raise ValueError(
                "Goal-grid layout has no G cell satisfying min_goal_cell_distance="
                f"{self._grid_min_goal_cell_distance} from R."
            )
        self._num_obstacles = 0
        self._num_gremlins = 0
        self._hazard_radius = float(0.25 * grid_cell_size if hazard_radius is None else hazard_radius)
        self._goal_radius = float(goal_radius)
        self._hazard_height = float(hazard_height)
        self._episode_length = int(episode_length)
        self._reset_noise_scale = float(reset_noise_scale)
        self._healthy_z_range = tuple(healthy_z_range)
        self._robot_cost_margin = float(robot_cost_margin)
        self._object_boundary = bool(object_boundary)
        self._terminate_on_cost = bool(terminate_on_cost)
        self._cost_limit_max = float(cost_limit_max)
        del relocate_objects_on_reset, different_object_layout_per_env, fixed_agent_on_reset
        self._relocate_objects_on_reset = False
        self._fixed_object_layout_seed = int(fixed_object_layout_seed)
        self._different_object_layout_per_env = False
        self._goal_respawn_on_success = bool(goal_respawn_on_success)
        self._fixed_goal_on_reset = bool(fixed_goal_on_reset)
        self._fixed_agent_on_reset = True
        self._layout_lidar_num_bins = int(layout_lidar_num_bins)
        default_lidar_max_dist = math.hypot(
            self._grid_num_rows * grid_cell_size,
            self._grid_num_cols * grid_cell_size,
        )
        self._layout_lidar_max_dist = (
            float(default_lidar_max_dist)
            if layout_lidar_max_dist is None
            else float(layout_lidar_max_dist)
        )
        self._full_robot_observation = bool(full_robot_observation)

        xml_string = _load_goal_grid_xml(
            agent=agent,
            render_task_objects=render_hazards,
            num_hazards=self._num_hazards,
            hazard_radius=self._hazard_radius,
            hazard_height=self._hazard_height,
        )
        sys = mjcf.loads(xml_string)

        if n_frames is None:
            n_frames = 10
        if backend in ("spring", "positional"):
            timestep = 0.005 if agent == "ant" else 0.0015
            sys = sys.tree_replace({"opt.timestep": timestep})
            n_frames = 10 if n_frames is None else n_frames
        if backend == "mjx":
            timestep = 0.005 if agent == "ant" else 0.0015
            sys = sys.tree_replace(
                {
                    "opt.timestep": timestep,
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        if agent == "humanoid":
            sys = apply_spring_humanoid_gear_for_mjx(
                sys,
                backend=backend,
                enabled=humanoid_use_spring_gear,
            )
        if backend == "positional":
            gear = 200 * jp.ones_like(sys.actuator.gear)
            if agent == "humanoid":
                gear = jp.array([350.0] * 11 + [100.0] * 6)
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        super().__init__(sys=sys, backend=backend, n_frames=n_frames)

        self._humanoid_use_spring_gear = bool(
            agent == "humanoid"
            and backend == "mjx"
            and humanoid_use_spring_gear
        )
        self._humanoid_actuator_gear_mode = (
            "spring" if self._humanoid_use_spring_gear else "xml"
        )

        self._target_q_start = 15 if agent == "ant" else 24
        self._target_qd_start = 14 if agent == "ant" else 23
        if self._render_hazards:
            self._target_link_idx = self.sys.link_names.index("target")
            self._robot_link_count = self._target_link_idx
        else:
            self._target_link_idx = None
            self._robot_link_count = len(self.sys.link_names)
        self._hazard_q_start = self._target_q_start + 2
        self._hazard_qd_start = self._target_qd_start + 2
        self._goal_dim = 3 if agent == "humanoid" else 2
        self.raw_goal_dim = self._goal_dim
        self.relabel_goal_dim = self._goal_dim
        self._robot_obs_dim = self._robot_obs_size()
        self._layout_lidar_num_channels = 1
        self.layout_obs_dim = self._layout_lidar_num_bins
        self.layout_start_idx = self._robot_obs_dim
        self.layout_end_idx = self.layout_start_idx + self.layout_obs_dim
        self.goal_start_idx = self.layout_end_idx
        self.goal_end_idx = self.goal_start_idx + self.raw_goal_dim
        self.state_dim = self.goal_end_idx
        self.goal_indices = jp.arange(self.goal_start_idx, self.goal_end_idx)
        self.scaling_crl_goal_indices = jp.arange(0, self.raw_goal_dim)

    def reset(self, rng: jax.Array) -> State:
        rng, q_rng, qd_rng, goal_rng, respawn_rng = jax.random.split(rng, 5)
        q = self.sys.init_q
        qd = jp.zeros(self.sys.qd_size())
        if self._reset_noise_scale:
            low, hi = -self._reset_noise_scale, self._reset_noise_scale
            q = q + jax.random.uniform(q_rng, (self.sys.q_size(),), minval=low, maxval=hi)
            qd = hi * jax.random.normal(qd_rng, (self.sys.qd_size(),))
        q = self._fix_agent_orientation(q)

        hazards_xy = self._grid_hazard_cells.astype(jp.float32) * self._grid_cell_size
        agent_xy = self._grid_reset_cell.astype(jp.float32) * self._grid_cell_size
        goal_rng = jax.random.PRNGKey(self._fixed_object_layout_seed + 17) if self._fixed_goal_on_reset else goal_rng
        goal_xy = self._sample_grid_goal(goal_rng, agent_xy, hazards_xy)
        q = q.at[:2].set(agent_xy)
        if self._render_hazards:
            q = q.at[self._target_q_start : self._target_q_start + 2].set(goal_xy)
            qd = qd.at[self._target_qd_start : self._target_qd_start + 2].set(0.0)
        q, qd = self._set_hazard_q(q, qd, hazards_xy)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state, jp.zeros(self.sys.act_size()), goal_xy, hazards_xy)
        zero = jp.asarray(0.0, dtype=obs.dtype)
        metrics = self._metrics(
            zero,
            zero,
            zero,
            self._achieved_goal(pipeline_state),
            self._goal_position(goal_xy),
        )
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=zero,
            done=zero,
            metrics=metrics,
            info={
                "steps": jp.array(0, dtype=jp.int32),
                "seed": jp.array(0, dtype=jp.int32),
                "rng": respawn_rng,
                "goal_xy": goal_xy,
                "hazards_xy": hazards_xy,
                "obstacles_xy": jp.zeros((0, 2), dtype=obs.dtype),
                "gremlin_centers_xy": jp.zeros((0, 2), dtype=obs.dtype),
                "gremlins_xy": jp.zeros((0, 2), dtype=obs.dtype),
                "episode_cost": zero,
                "layout_id": jp.array(0, dtype=jp.int32),
            },
        )

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state = self.pipeline_step(state.pipeline_state, action)
        success_goal = self._achieved_goal(pipeline_state)
        success_xy = success_goal[:2]
        goal_xy = state.info["goal_xy"]
        goal_position = self._goal_position(goal_xy)
        dist = jp.linalg.norm(success_goal - goal_position)
        success = (dist <= self._goal_radius).astype(float)
        reward = success
        cost_hazards = self._hazard_cost(success_xy, state.info["hazards_xy"], self._robot_cost_margin)
        boundary_cost = self._hazard_cost(success_xy, state.info["hazards_xy"], 0.0)
        cost = cost_hazards
        episode_cost = state.info["episode_cost"] + cost

        rng, respawn_rng = jax.random.split(state.info["rng"])
        if self._goal_respawn_on_success:
            respawn_goal_xy = self._sample_grid_goal(respawn_rng, success_xy, state.info["hazards_xy"])
            next_goal_xy = jp.where(success.astype(bool), respawn_goal_xy, goal_xy)
        else:
            next_goal_xy = goal_xy
        pipeline_state = self._sync_target_pipeline_state(pipeline_state, next_goal_xy)

        wrapped_by_training = "truncation" in state.info
        steps = state.info["steps"] + 1
        done = jp.array(False) if wrapped_by_training else steps >= self._episode_length
        min_z, max_z = self._healthy_z_range
        z = pipeline_state.x.pos[0, 2]
        done = jp.logical_or(done, jp.logical_or(z < min_z, z > max_z))
        if self._object_boundary:
            done = jp.logical_or(done, boundary_cost.astype(bool))
        if self._terminate_on_cost:
            done = jp.logical_or(done, episode_cost >= self._cost_limit_max)
        done = done.astype(float)

        obs = self._get_obs(pipeline_state, action, next_goal_xy, state.info["hazards_xy"])
        metrics = self._metrics(
            success,
            dist,
            cost_hazards,
            success_goal,
            self._goal_position(next_goal_xy),
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
                "hazards_xy": state.info["hazards_xy"],
                "obstacles_xy": state.info["obstacles_xy"],
                "gremlin_centers_xy": state.info["gremlin_centers_xy"],
                "gremlins_xy": state.info["gremlins_xy"],
                "episode_cost": episode_cost,
                "layout_id": state.info["layout_id"],
            }
        )
        return state.replace(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )

    def _robot_obs_size(self) -> int:
        if self._agent == "ant":
            return self._target_q_start + self._target_qd_start
        if not self._full_robot_observation:
            return self._target_q_start + self._target_qd_start
        return (
            self._target_q_start
            + self._target_qd_start
            + 10 * self._robot_link_count
            + 6 * self._robot_link_count
        )

    def _robot_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        del action
        q = pipeline_state.q[: self._target_q_start]
        qd = pipeline_state.qd[: self._target_qd_start]
        if self._agent == "ant" or not self._full_robot_observation:
            return jp.concatenate([q, qd])
        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )
        xd_i = (
            base.Transform.create(pos=x_i.pos - pipeline_state.x.pos[: self._robot_link_count])
            .vmap()
            .do(jax.tree_util.tree_map(lambda x: x[: self._robot_link_count], pipeline_state.xd))
        )
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])
        return jp.concatenate([q, qd, com_inertia.ravel(), com_velocity.ravel()])

    def _com(self, pipeline_state: base.State) -> tuple[jax.Array, base.Inertia, jax.Array, base.Transform]:
        inertia = jax.tree_util.tree_map(lambda x: x[: self._robot_link_count], self.sys.link.inertia)
        if self.backend in ("spring", "positional"):
            inertia = inertia.replace(
                i=jax.vmap(jp.diag)(
                    jax.vmap(jp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jp.sum(inertia.mass)
        robot_x = jax.tree_util.tree_map(lambda x: x[: self._robot_link_count], pipeline_state.x)
        x_i = robot_x.vmap().do(inertia.transform)
        com = jp.sum(jax.vmap(jp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        return com, inertia, mass_sum, x_i

    def _get_obs(
        self,
        pipeline_state: base.State,
        action: jax.Array,
        goal_xy: jax.Array,
        hazards_xy: jax.Array,
    ) -> jax.Array:
        agent_xy = self._agent_xy(pipeline_state)
        return jp.concatenate(
            [
                self._robot_obs(pipeline_state, action),
                self._hazard_lidar(hazards_xy, agent_xy),
                self._goal_position(goal_xy).astype(agent_xy.dtype),
            ]
        )

    def _agent_xy(self, pipeline_state: base.State) -> jax.Array:
        return pipeline_state.x.pos[0, :2]

    def _achieved_goal(self, pipeline_state: base.State) -> jax.Array:
        if self._agent == "humanoid":
            return pipeline_state.x.pos[0, :3]
        return self._agent_xy(pipeline_state)

    def _goal_position(self, goal_xy: jax.Array) -> jax.Array:
        if self._agent == "humanoid":
            return jp.concatenate(
                [
                    goal_xy,
                    jp.asarray([self._target_z()], dtype=goal_xy.dtype),
                ]
            )
        return goal_xy

    def _fix_agent_orientation(self, q: jax.Array) -> jax.Array:
        return q.at[2:7].set(self.sys.init_q[2:7])

    def _grid_cells(self) -> jax.Array:
        rows = jp.arange(self._grid_num_rows, dtype=jp.int32)
        cols = jp.arange(self._grid_num_cols, dtype=jp.int32)
        xx, yy = jp.meshgrid(rows, cols, indexing="ij")
        return jp.stack([xx.reshape((-1,)), yy.reshape((-1,))], axis=-1)

    def _grid_centers(self) -> jax.Array:
        return self._grid_cells().astype(jp.float32) * jp.asarray(self._grid_cell_size, dtype=jp.float32)

    def _xy_to_grid_cell(self, xy: jax.Array) -> jax.Array:
        cell = jp.floor((xy + 0.5 * self._grid_cell_size) / self._grid_cell_size).astype(jp.int32)
        upper = jp.asarray(
            [self._grid_num_rows - 1, self._grid_num_cols - 1],
            dtype=jp.int32,
        )
        return jp.clip(cell, 0, upper)

    def _grid_cell_index(self, cell: jax.Array) -> jax.Array:
        return cell[0] * self._grid_num_cols + cell[1]

    def _valid_goal_mask(
        self,
        agent_cell_index: jax.Array,
        min_goal_cell_distance: float | jax.Array | None = None,
    ) -> jax.Array:
        cells = self._grid_cells()
        agent_cell = cells[agent_cell_index]
        distance = jp.linalg.norm(
            cells.astype(jp.float32) - agent_cell[None, :].astype(jp.float32),
            axis=-1,
        )
        min_distance = jp.asarray(
            self._grid_min_goal_cell_distance
            if min_goal_cell_distance is None
            else min_goal_cell_distance,
            dtype=distance.dtype,
        )
        not_agent = jp.arange(self._grid_num_total_cells) != agent_cell_index
        return self._grid_goal_mask & not_agent & (distance >= min_distance)

    def _sample_grid_goal(self, rng: jax.Array, agent_xy: jax.Array, hazards_xy: jax.Array) -> jax.Array:
        del hazards_xy
        centers = self._grid_centers()
        agent_cell_index = self._grid_cell_index(self._xy_to_grid_cell(agent_xy))
        valid = self._valid_goal_mask(agent_cell_index)
        priorities = jax.random.uniform(rng, (self._grid_num_total_cells,))
        scores = jp.where(valid, priorities, -jp.inf)
        valid_idx = jp.argmax(scores)
        has_valid = jp.any(valid)
        alternate_goal = self._valid_goal_mask(
            agent_cell_index,
            min_goal_cell_distance=0.0,
        )
        goal_fallback = jp.where(
            jp.any(alternate_goal),
            alternate_goal,
            self._grid_goal_mask,
        )
        agent_cell = self._grid_cells()[agent_cell_index]
        candidate_distance = jp.linalg.norm(
            self._grid_cells().astype(centers.dtype) - agent_cell[None, :].astype(centers.dtype),
            axis=-1,
        )
        fallback_scores = jp.where(goal_fallback, candidate_distance, -jp.inf)
        fallback_idx = jp.argmax(fallback_scores)
        idx = jp.where(has_valid, valid_idx, fallback_idx)
        return centers[idx]

    def _hazard_lidar(self, hazards_xy: jax.Array, agent_xy: jax.Array) -> jax.Array:
        if hazards_xy.shape[0] == 0:
            return jp.zeros((self._layout_lidar_num_bins,), dtype=agent_xy.dtype)
        rel_xy = hazards_xy.astype(agent_xy.dtype) - agent_xy[None, :]
        dist = jp.linalg.norm(rel_xy, axis=-1)
        max_dist = jp.asarray(self._layout_lidar_max_dist, dtype=agent_xy.dtype)
        signal = jp.clip(1.0 - dist / jp.maximum(max_dist, 1e-6), 0.0, 1.0)
        angle = jp.mod(jp.arctan2(rel_xy[:, 1], rel_xy[:, 0]), 2.0 * jp.pi)
        bin_idx = jp.floor(angle / (2.0 * jp.pi) * self._layout_lidar_num_bins).astype(jp.int32)
        bin_idx = jp.clip(bin_idx, 0, self._layout_lidar_num_bins - 1)
        bin_mask = jax.nn.one_hot(bin_idx, self._layout_lidar_num_bins, dtype=agent_xy.dtype)
        return jp.max(bin_mask * signal[:, None], axis=0)

    def _hazard_cost(self, agent_xy: jax.Array, hazards_xy: jax.Array, margin: float) -> jax.Array:
        if hazards_xy.shape[0] == 0:
            return jp.asarray(0.0, dtype=agent_xy.dtype)
        dist = jp.linalg.norm(hazards_xy.astype(agent_xy.dtype) - agent_xy[None, :], axis=-1)
        return jp.any(dist <= (self._hazard_radius + margin)).astype(agent_xy.dtype)

    def _set_hazard_q(self, q: jax.Array, qd: jax.Array, hazards_xy: jax.Array) -> tuple[jax.Array, jax.Array]:
        if not self._render_hazards or self._num_hazards == 0:
            return q, qd
        hazard_q = hazards_xy.reshape((-1,))
        q = q.at[self._hazard_q_start : self._hazard_q_start + hazard_q.shape[0]].set(hazard_q)
        qd = qd.at[self._hazard_qd_start : self._hazard_qd_start + hazard_q.shape[0]].set(0.0)
        return q, qd

    def _sync_target_pipeline_state(self, pipeline_state: base.State, goal_xy: jax.Array) -> base.State:
        if not self._render_hazards:
            return pipeline_state
        q = pipeline_state.q.at[self._target_q_start : self._target_q_start + 2].set(goal_xy)
        qd = pipeline_state.qd.at[self._target_qd_start : self._target_qd_start + 2].set(0.0)
        goal_pos = jp.array([goal_xy[0], goal_xy[1], self._target_z()], dtype=pipeline_state.x.pos.dtype)
        x = pipeline_state.x.replace(pos=pipeline_state.x.pos.at[self._target_link_idx].set(goal_pos))
        return pipeline_state.replace(q=q, qd=qd, x=x)

    def _target_z(self) -> float:
        return 0.01 if self._agent == "ant" else 1.25

    def _metrics(
        self,
        success: jax.Array,
        dist: jax.Array,
        cost_hazards: jax.Array,
        achieved_goal: jax.Array,
        goal_position: jax.Array,
    ) -> dict[str, jax.Array]:
        zero = jp.asarray(0.0, dtype=achieved_goal.dtype)
        metrics = {
            "reward": success,
            "success": success,
            "cost": cost_hazards,
            "cost_hazards": cost_hazards,
            "cost_obstacles": zero,
            "cost_gremlins": zero,
            "dist": dist,
            "x_position": achieved_goal[0],
            "y_position": achieved_goal[1],
            "agent_yaw": zero,
            "goal_x": goal_position[0],
            "goal_y": goal_position[1],
        }
        if self._agent == "humanoid":
            metrics.update(
                {
                    "z_position": achieved_goal[2],
                    "goal_z": goal_position[2],
                }
            )
        return metrics


class AntGoalGridBase(GoalGridMazeBase):
    def __init__(self, render_hazards: bool, **kwargs):
        kwargs.setdefault("backend", "mjx")
        kwargs.setdefault("n_frames", 10)
        kwargs.setdefault("episode_length", 1000)
        kwargs.setdefault("reset_noise_scale", 0.1)
        kwargs.setdefault("healthy_z_range", (0.2, 1.0))
        kwargs.setdefault("robot_cost_margin", 0.10)
        super().__init__(agent="ant", render_hazards=render_hazards, **kwargs)


class HumanoidGoalGridBase(GoalGridMazeBase):
    def __init__(self, render_hazards: bool, **kwargs):
        kwargs.setdefault("backend", "mjx")
        kwargs.setdefault("n_frames", 10)
        kwargs.setdefault("episode_length", 1000)
        kwargs.setdefault("reset_noise_scale", 0.0)
        kwargs.setdefault("healthy_z_range", (1.0, 2.0))
        kwargs.setdefault("robot_cost_margin", 0.05)
        super().__init__(agent="humanoid", render_hazards=render_hazards, **kwargs)
