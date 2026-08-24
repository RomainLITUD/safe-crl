import os
from pathlib import Path
from typing import Tuple

from brax import base
from brax import math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco
import xml.etree.ElementTree as ET

# This is based on original Ant environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/ant.py
# Maze creation dapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py

RESET = R = 'r'
GOAL = G = 'g'


U_MAZE = [[1, 1, 1, 1, 1],
          [1, R, G, G, 1],
          [1, 1, 1, G, 1],
          [1, G, G, G, 1],
          [1, 1, 1, 1, 1]]

U_MAZE_EVAL = [[1, 1, 1, 1, 1],
               [1, R, 0, 0, 1],
               [1, 1, 1, 0, 1],
               [1, G, G, G, 1],
               [1, 1, 1, 1, 1]]

U_MAZE_SINGLE_EVAL = [[1, 1, 1, 1, 1],
               [1, R, 0, 0, 1],
               [1, 1, 1, 0, 1],
               [1, G, 0, 0, 1],
               [1, 1, 1, 1, 1]]

U_MAZE_EVAL_1f2f3f4f5f = [[1, 1, 1, 1, 1],
               [1, R, G, G, 1],
               [1, 1, 1, G, 1],
               [1, 0, G, G, 1],
               [1, 1, 1, 1, 1]]

U_MAZE_EVAL_1f2f3f4f = [[1, 1, 1, 1, 1],
               [1, R, G, G, 1],
               [1, 1, 1, G, 1],
               [1, 0, 0, G, 1],
               [1, 1, 1, 1, 1]]

U_MAZE_EVAL_1f2f3f = [[1, 1, 1, 1, 1],
               [1, R, G, G, 1],
               [1, 1, 1, G, 1],
               [1, 0, 0, 0, 1],
               [1, 1, 1, 1, 1]]

U_MAZE_EVAL_5f6f = [[1, 1, 1, 1, 1],
               [1, R, 0, 0, 1],
               [1, 1, 1, 0, 1],
               [1, G, G, 0, 1],
               [1, 1, 1, 1, 1]]


U2_MAZE = [[1, 1, 1, 1, 1, 1],
           [1, R, G, G, G, 1],
           [1, 1, 1, 1, G, 1],
           [1, G, G, G, G, 1],
           [1, 1, 1, 1, 1, 1]]

U2_MAZE_EVAL = [[1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 1],
                [1, 1, 1, 1, 0, 1],
                [1, G, G, G, G, 1],
                [1, 1, 1, 1, 1, 1]]

U3_MAZE = [[1, 1, 1, 1, 1, 1, 1],
           [1, R, G, G, G, G, 1],
           [1, 1, 1, 1, 1, G, 1],
           [1, G, G, G, G, G, 1],
           [1, 1, 1, 1, 1, 1, 1]]

U3_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 0, 1],
                [1, G, G, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1]]

U3_MAZE_SINGLE_EVAL = [[1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 0, 1],
                [1, G, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1]]

H_MAZE = [[1, 1, 1, 1, 1],
           [1, G, 1, G, 1],
           [1, G, 1, G, 1],
           [1, G, R, G, 1],
           [1, G, 1, G, 1],
           [1, G, 1, G, 1],
           [1, 1, 1, 1, 1]]

H_MAZE_EVAL = [[1, 1, 1, 1, 1],
                [1, G, 1, G, 1],
                [1, G, 1, G, 1],
                [1, 0, R, 0, 1],
                [1, G, 1, G, 1],
                [1, G, 1, G, 1],
                [1, 1, 1, 1, 1]]


U5_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
           [1, G, G, G, G, G, G, 1],
           [1, R, 1, 1, 1, 1, G, 1],
           [1, 1, 1, 1, 1, 1, G, 1],
           [1, G, 1, 1, 1, 1, G, 1],
           [1, G, G, G, G, G, G, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]]

U5_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, R, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 1],
                [1, G, 1, 1, 1, 1, G, 1],
                [1, G, G, G, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]

U5_MAZE_SINGLE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, R, 1, 1, 1, 1, 0, 1],
                [1, 1, 1, 1, 1, 1, 0, 1],
                [1, G, 1, 1, 1, 1, 0, 1],
                [1, 0, 0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]

U6_MAZE = [[1, 1, 1, 1, 1, 1, 1],
           [1, G, G, G, G, G, 1],
           [1, R, 1, 1, 1, G, 1],
           [1, 1, 1, 1, 1, G, 1],
           [1, G, 1, 1, 1, G, 1],
           [1, G, G, G, G, G, 1],
           [1, 1, 1, 1, 1, 1, 1]]

U6_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1],
           [1, 0, 0, 0, 0, 0, 1],
           [1, R, 1, 1, 1, 0, 1],
           [1, 1, 1, 1, 1, 0, 1],
           [1, G, 1, 1, 1, G, 1],
           [1, G, G, G, G, G, 1],
           [1, 1, 1, 1, 1, 1, 1]]

CROSS_MAZE = [[1, 1, 1, 1, 1, 1, 1],
           [1, G, G, G, 1, G, 1],
           [1, 1, 1, G, 1, G, 1],
           [1, G, G, R, G, G, 1],
           [1, 1, 1, G, 1, 1, 1],
           [1, G, G, G, G, G, 1],
           [1, 1, 1, 1, 1, 1, 1]]

CROSS_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1],
                [1, G, G, G, 1, G, 1],
                [1, 1, 1, 0, 1, G, 1],
                [1, G, 0, R, 0, G, 1],
                [1, 1, 1, 0, 1, 1, 1],
                [1, G, G, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1]]

# U5_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
#                 [1, 0, 0, 0, 0, 0, 0, 1],
#                 [1, R, 1, 1, 1, 1, 0, 1],
#                 [1, 1, 1, 1, 1, 1, 0, 1],
#                 [1, G, 1, 1, 1, 1, 0, 1],
#                 [1, G, 0, 0, 0, G, G, 1],
#                 [1, 1, 1, 1, 1, 1, 1, 1]]


BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, G, 1, 1, G, G, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, 1, G, G, G, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, G, 1, 1, G, G, 1],
                [1, G, G, 1, G, G, G, 1],
                [1, 1, G, G, G, 1, 1, 1],
                [1, G, G, 1, G, G, G, 1],
                [1, G, 1, G, G, 1, G, 1],
                [1, G, G, G, 1, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]

HARDEST_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, G, G, G, 1, G, G, G, G, G, 1],
                [1, G, 1, 1, G, 1, G, 1, G, 1, G, 1],
                [1, G, G, G, G, G, G, 1, G, G, G, 1],
                [1, G, 1, 1, 1, 1, G, 1, 1, 1, G, 1],
                [1, G, G, 1, G, 1, G, G, G, G, G, 1],
                [1, 1, G, 1, G, 1, G, 1, G, 1, 1, 1],
                [1, G, G, 1, G, G, G, 1, G, G, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

MAZE_HEIGHT = 0.5

def find_robot(structure, size_scaling):
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                # print(f"reset position: {i * size_scaling}, {j * size_scaling}")
                return i * size_scaling, j * size_scaling
            
            
def find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append([i * size_scaling, j * size_scaling])
                # print(f"possible goal: {goals[-1]}")

    return jp.array(goals)

def _is_boundary_cell(i, j, maze_layout):
    return i == 0 or j == 0 or i == len(maze_layout) - 1 or j == len(maze_layout[0]) - 1


# Create a xml with maze and a list of possible goal positions
SUPPORTED_MAZE_LAYOUTS = frozenset(
    {
        "u_maze",
        "u_maze_eval",
        "u_maze_single_eval",
        "u_maze_eval_1f2f3f4f5f",
        "u_maze_eval_1f2f3f4f",
        "u_maze_eval_1f2f3f",
        "u_maze_eval_5f6f",
        "u2_maze",
        "u2_maze_eval",
        "u3_maze",
        "u3_maze_eval",
        "u3_maze_single_eval",
        "h_maze",
        "h_maze_eval",
        "u5_maze",
        "u5_maze_eval",
        "u5_maze_single_eval",
        "u6_maze",
        "u6_maze_eval",
        "cross_maze",
        "cross_maze_eval",
        "big_maze",
        "big_maze_eval",
        "hardest_maze",
    }
)


def make_maze(
    maze_layout_name,
    maze_size_scaling,
    thin_outer_walls=False,
):
    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "u_maze_eval":
        maze_layout = U_MAZE_EVAL
    elif maze_layout_name == "u_maze_single_eval":
        maze_layout = U_MAZE_SINGLE_EVAL
    elif maze_layout_name == "u_maze_eval_1f2f3f4f5f":
        maze_layout = U_MAZE_EVAL_1f2f3f4f5f
    elif maze_layout_name == "u_maze_eval_1f2f3f4f":
        maze_layout = U_MAZE_EVAL_1f2f3f4f
    elif maze_layout_name == "u_maze_eval_1f2f3f":
        maze_layout = U_MAZE_EVAL_1f2f3f
    elif maze_layout_name == "u_maze_eval_5f6f":
        maze_layout = U_MAZE_EVAL_5f6f
    elif maze_layout_name == "u2_maze":
        maze_layout = U2_MAZE
    elif maze_layout_name == "u2_maze_eval":
        maze_layout = U2_MAZE_EVAL
    elif maze_layout_name == "u3_maze":
        maze_layout = U3_MAZE
    elif maze_layout_name == "u3_maze_eval":
        maze_layout = U3_MAZE_EVAL
    elif maze_layout_name == "u3_maze_single_eval":
        maze_layout = U3_MAZE_SINGLE_EVAL
    elif maze_layout_name == "h_maze":
        maze_layout = H_MAZE
    elif maze_layout_name == "h_maze_eval":
        maze_layout = H_MAZE_EVAL
    elif maze_layout_name == "u5_maze":
        maze_layout = U5_MAZE
    elif maze_layout_name == "u5_maze_eval":
        maze_layout = U5_MAZE_EVAL
    elif maze_layout_name == "u6_maze":
        maze_layout = U6_MAZE
    elif maze_layout_name == "u6_maze_eval":
        maze_layout = U6_MAZE_EVAL
    elif maze_layout_name == "cross_maze":
        maze_layout = CROSS_MAZE
    elif maze_layout_name == "cross_maze_eval":
        maze_layout = CROSS_MAZE_EVAL
    elif maze_layout_name == "u5_maze_single_eval":
        maze_layout = U5_MAZE_SINGLE_EVAL

    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    elif maze_layout_name == "big_maze_eval":
        maze_layout = BIG_MAZE_EVAL
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")
    
    xml_path = Path(__file__).resolve().parent / "assets" / "xmls" / "scaling_ant_maze.xml"

    robot_x, robot_y = find_robot(maze_layout, maze_size_scaling)
    possible_goals = find_goals(maze_layout, maze_size_scaling)
    wall_centers = []

    tree = ET.parse(xml_path)
    worldbody = tree.find(".//worldbody")

    for i in range(len(maze_layout)):
        for j in range(len(maze_layout[0])):
            struct = maze_layout[i][j]
            if struct == 1:
                wall_centers.append([i * maze_size_scaling, j * maze_size_scaling])
                geom_height = MAZE_HEIGHT * maze_size_scaling
                half = 0.5 * maze_size_scaling
                thin = 0.25 * maze_size_scaling
                cx = i * maze_size_scaling
                cy = j * maze_size_scaling
                sx, sy = half, half
                if thin_outer_walls and _is_boundary_cell(i, j, maze_layout):
                    nrows, ncols = len(maze_layout), len(maze_layout[0])
                    if i == 0 or i == nrows - 1:
                        sx = thin
                        cx += half - thin if i == 0 else -(half - thin)
                    if j == 0 or j == ncols - 1:
                        sy = thin
                        cy += half - thin if j == 0 else -(half - thin)
                ET.SubElement(
                    worldbody, "geom",
                    name="block_%d_%d" % (i, j),
                    pos="%f %f %f" % (cx, cy, geom_height / 2.0),
                    size="%f %f %f" % (sx, sy, geom_height / 2.0),
                    type="box",
                    material="",
                    contype="1",
                    conaffinity="1",
                    rgba="0.7 0.5 0.3 1.0",
                )

    
    torso = tree.find(".//numeric[@name='init_qpos']")
    data = torso.get("data")
    torso.set("data", f"{robot_x} {robot_y} " + data) 

    tree = tree.getroot()
    xml_string = ET.tostring(tree)
    
    wall_centers = jp.asarray(wall_centers, dtype=jp.float32)
    if wall_centers.ndim == 1:
        wall_centers = jp.zeros((0, 2), dtype=jp.float32)
    return xml_string, possible_goals, wall_centers

class CrlAntMaze(PipelineEnv):
    def __init__(
        self,
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=False,
        backend="spring",
        maze_layout_name="u_maze",
        maze_size_scaling=4.0,
        goal_radius=0.5,
        evaluation_mode=False,
        layout_lidar_num_bins=16,
        layout_lidar_max_dist=12.0,
        thin_outer_walls=False,
        **kwargs,
    ):
        del evaluation_mode
        if layout_lidar_num_bins < 1:
            raise ValueError("layout_lidar_num_bins must be at least 1.")
        if layout_lidar_max_dist <= 0.0:
            raise ValueError("layout_lidar_max_dist must be positive.")
        xml_string, possible_goals, wall_centers = make_maze(
            maze_layout_name,
            maze_size_scaling,
            thin_outer_walls=thin_outer_walls,
        )

        sys = mjcf.loads(xml_string)
        self.maze_layout_name = maze_layout_name
        self.possible_goals = possible_goals
        self.wall_centers = wall_centers
        self._maze_size_scaling = maze_size_scaling
        self._goal_radius = goal_radius
        self._layout_lidar_num_bins = int(layout_lidar_num_bins)
        self._layout_lidar_max_dist = float(layout_lidar_max_dist)

        n_frames = 10

        if backend in ["spring", "positional"]:
            sys = sys.tree_replace({"opt.timestep": 0.005})
            n_frames = 10

        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.timestep": 0.005,
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )

        if backend == "positional":
            # TODO: does the same actuator strength work as in spring
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )

        kwargs["n_frames"] = kwargs.get("n_frames", n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._use_contact_forces = use_contact_forces
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._robot_obs_dim = 29
        self.layout_obs_dim = self._layout_lidar_num_bins
        self.layout_start_idx = self._robot_obs_dim
        self.layout_end_idx = self.layout_start_idx + self.layout_obs_dim
        self.state_dim = self.layout_end_idx
        self.raw_goal_dim = 2
        self.relabel_goal_dim = 2
        self.goal_start_idx = self.layout_end_idx
        self.goal_end_idx = self.goal_start_idx + self.raw_goal_dim
        self.goal_indices = jp.arange(self.goal_start_idx, self.goal_end_idx)
        self.scaling_crl_goal_indices = jp.arange(0, self.raw_goal_dim)

        if self._use_contact_forces:
            raise NotImplementedError("use_contact_forces not implemented.")

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""

        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))

        # set the target q, qd
        _, target = self._random_target(rng)
        q = q.at[-2:].set(target)
        qd = qd.at[-2:].set(0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            "reward_forward": zero,
            "reward_survive": zero,
            "reward_ctrl": zero,
            "reward_contact": zero,
            "x_position": zero,
            "y_position": zero,
            "distance_from_origin": zero,
            "x_velocity": zero,
            "y_velocity": zero,
            "forward_reward": zero,
            "dist": zero,
            "success": zero,
            "success_easy": zero,
            "cost": zero,
            "episode_cost": zero,
            "goal_x": target[0],
            "goal_y": target[1],
        }
        info = {
            "seed": 0,
            "rng": rng,
            "episode_cost": zero,
            "goal_reached": zero,
        }
        state = State(pipeline_state, obs, reward, done, metrics)
        state.info.update(info)
        return state

    # Todo rename seed to traj_id
    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        rng = state.info["rng"] if "rng" in state.info else jax.random.PRNGKey(0)
        rng, _ = jax.random.split(rng)
        info = {"seed": seed, "rng": rng}

        velocity = (pipeline_state.x.pos[0] - pipeline_state0.x.pos[0]) / self.dt
        forward_reward = velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        contact_cost = 0.0

        obs = self._get_obs(pipeline_state)
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        dist = jp.linalg.norm(obs[:2] - obs[-2:])
        success = jp.array(dist < self._goal_radius, dtype=float)
        success_easy = jp.array(dist < 2., dtype=float)
        zero = jp.asarray(0.0, dtype=obs.dtype)
        newly_reached = success * (1.0 - state.info["goal_reached"])
        goal_reached = jp.maximum(state.info["goal_reached"], success)
        reward = newly_reached
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            x_position=pipeline_state.x.pos[0, 0],
            y_position=pipeline_state.x.pos[0, 1],
            distance_from_origin=math.safe_norm(pipeline_state.x.pos[0]),
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            forward_reward=forward_reward,
            dist=dist,
            success=success,
            success_easy=success_easy,
            cost=zero,
            episode_cost=zero,
            goal_x=obs[-2],
            goal_y=obs[-1],
        )
        state.info.update(info)
        state.info.update(
            episode_cost=zero,
            goal_reached=goal_reached,
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body position and velocities."""
        root_xy = pipeline_state.q[:2]
        qpos = pipeline_state.q[:-2]
        qvel = pipeline_state.qd[:-2]

        target_pos = pipeline_state.x.pos[-1][:2]

        if self._exclude_current_positions_from_observation:
            qpos = qpos[2:]

        robot_obs = jp.concatenate([qpos] + [qvel])
        wall_lidar = self._wall_lidar(root_xy)
        return jp.concatenate([robot_obs, wall_lidar, target_pos])

    def _wall_lidar(self, xy: jax.Array) -> jax.Array:
        if self.wall_centers.shape[0] == 0:
            return jp.zeros((self._layout_lidar_num_bins,), dtype=xy.dtype)
        dtype = xy.dtype
        angles = (
            jp.arange(self._layout_lidar_num_bins, dtype=dtype)
            * (2.0 * jp.pi / self._layout_lidar_num_bins)
        )
        dirs = jp.stack([jp.cos(angles), jp.sin(angles)], axis=-1)
        centers = self.wall_centers.astype(dtype)
        half = jp.asarray(0.5 * self._maze_size_scaling, dtype=dtype)
        bounds_min = centers - half
        bounds_max = centers + half
        eps_dirs = jp.where(dirs >= 0.0, 1e-6, -1e-6)
        inv_dirs = 1.0 / jp.where(jp.abs(dirs) < 1e-6, eps_dirs, dirs)
        t1 = (bounds_min[None, :, :] - xy[None, None, :]) * inv_dirs[:, None, :]
        t2 = (bounds_max[None, :, :] - xy[None, None, :]) * inv_dirs[:, None, :]
        tmin = jp.max(jp.minimum(t1, t2), axis=-1)
        tmax = jp.min(jp.maximum(t1, t2), axis=-1)
        hit = (tmax >= jp.maximum(tmin, 0.0)) & (tmin <= self._layout_lidar_max_dist)
        hit_dist = jp.where(hit, jp.maximum(tmin, 0.0), jp.inf)
        nearest = jp.min(hit_dist, axis=-1)
        signal = 1.0 - nearest / jp.asarray(self._layout_lidar_max_dist, dtype=dtype)
        return jp.where(jp.isfinite(nearest), jp.clip(signal, 0.0, 1.0), 0.0)

    def _random_target(self, rng: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return rng, jp.array(self.possible_goals[idx])[0]
