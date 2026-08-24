from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jnp
import mujoco
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from safenav_jax.envs._humanoid_actuation import apply_spring_humanoid_gear_for_mjx

# This is based on original Humanoid environment from Brax
# https://github.com/google/brax/blob/main/brax/envs/humanoid.py

# This is chosen to be very close to the z coordinate of the humanoid torso, when it is standing straight
TARGET_Z_COORD = 1.25

# Maze creation adapted from: https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/locomotion/maze_env.py
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

BIG_MAZE = [[1, 1, 1, 1, 1, 1, 1, 1],
            [1, R, G, 1, 1, G, G, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, 1, G, G, G, 1, 1, 1],
            [1, G, G, 1, G, G, G, 1],
            [1, G, 1, G, G, 1, G, 1],
            [1, G, G, G, 1, G, G, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]]

BIG_MAZE_EVAL = [[1, 1, 1, 1, 1, 1, 1, 1],
                 [1, R, 0, 1, 1, G, G, 1],
                 [1, 0, 0, 1, 0, G, G, 1],
                 [1, 1, 0, 0, 0, 1, 1, 1],
                 [1, 0, 0, 1, 0, 0, 0, 1],
                 [1, 0, 1, G, 0, 1, G, 1],
                 [1, 0, G, G, 1, G, G, 1],
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


def find_starts(structure, size_scaling):
    starts = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == RESET:
                starts.append([i * size_scaling, j * size_scaling])

    return jnp.array(starts)
            
def find_goals(structure, size_scaling):
    goals = []
    for i in range(len(structure)):
        for j in range(len(structure[0])):
            if structure[i][j] == GOAL:
                goals.append([i * size_scaling, j * size_scaling])

    return jnp.array(goals)

def _is_boundary_cell(i, j, maze_layout):
    return i == 0 or j == 0 or i == len(maze_layout) - 1 or j == len(maze_layout[0]) - 1


# Create a xml with maze and a list of possible goal positions
SUPPORTED_MAZE_LAYOUTS = frozenset(
    {
        "u_maze",
        "u_maze_eval",
        "big_maze",
        "big_maze_eval",
        "hardest_maze",
    }
)


def make_maze(
    maze_layout_name,
    maze_size_scaling,
    walls_collide=True,
    thin_outer_walls=False,
):
    if maze_layout_name == "u_maze":
        maze_layout = U_MAZE
    elif maze_layout_name == "u_maze_eval":
        maze_layout = U_MAZE_EVAL
    elif maze_layout_name == "big_maze":
        maze_layout = BIG_MAZE
    elif maze_layout_name == "big_maze_eval":
        maze_layout = BIG_MAZE_EVAL
    elif maze_layout_name == "hardest_maze":
        maze_layout = HARDEST_MAZE
    else:
        raise ValueError(f"Unknown maze layout: {maze_layout_name}")
    
    xml_path = Path(__file__).resolve().parent / "assets" / "xmls" / "scaling_humanoid_maze.xml"

    possible_starts = find_starts(maze_layout, maze_size_scaling)
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
                    contype="1" if walls_collide else "0",
                    conaffinity="1" if walls_collide else "0",
                    rgba="0.7 0.5 0.3 1.0",
                )

    tree = tree.getroot()
    xml_string = ET.tostring(tree)
    
    wall_centers = jnp.asarray(wall_centers, dtype=jnp.float32)
    if wall_centers.ndim == 1:
        wall_centers = jnp.zeros((0, 2), dtype=jnp.float32)
    return xml_string, possible_starts, possible_goals, wall_centers

class CrlHumanoidMaze(PipelineEnv):
    def __init__(
        self,
        forward_reward_weight=1.25,
        ctrl_cost_weight=0.1,
        healthy_reward=5.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(1.0, 2.0),
        reset_noise_scale=0.0,
        exclude_current_positions_from_observation=False,
        backend='mjx',
        maze_layout_name="u_maze",
        maze_size_scaling=2.0, # Was 4.0 for antmaze -- just trying to make it tractable
        goal_radius=0.5,
        evaluation_mode=False,
        layout_lidar_num_bins=16,
        layout_lidar_max_dist=6.0,
        walls_collide=True,
        thin_outer_walls=False,
        humanoid_use_spring_gear=False,
        **kwargs,
    ):
        del evaluation_mode
        if layout_lidar_num_bins < 1:
            raise ValueError("layout_lidar_num_bins must be at least 1.")
        if layout_lidar_max_dist <= 0.0:
            raise ValueError("layout_lidar_max_dist must be positive.")
        xml_string, possible_starts, possible_goals, wall_centers = make_maze(
            maze_layout_name,
            maze_size_scaling,
            walls_collide=walls_collide,
            thin_outer_walls=thin_outer_walls,
        )
        sys = mjcf.loads(xml_string)
        self.maze_layout_name = maze_layout_name
        self.possible_starts = possible_starts
        self.possible_goals = possible_goals
        self.wall_centers = wall_centers
        self._maze_size_scaling = maze_size_scaling
        self._goal_radius = goal_radius
        self._layout_lidar_num_bins = int(layout_lidar_num_bins)
        self._layout_lidar_max_dist = float(layout_lidar_max_dist)

        n_frames = 10

        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.0015})
            n_frames = 10
            gear = jnp.array([
              350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
              350.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])  # pyformat: disable
            sys = sys.replace(actuator=sys.actuator.replace(gear=gear))

        if backend == 'mjx':
            sys = sys.tree_replace({
                'opt.timestep': 0.0015,
                'opt.solver': mujoco.mjtSolver.mjSOL_NEWTON,
                'opt.disableflags': mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                'opt.iterations': 1,
                'opt.ls_iterations': 4,
            })
        sys = apply_spring_humanoid_gear_for_mjx(
            sys,
            backend=backend,
            enabled=humanoid_use_spring_gear,
        )

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)

        super().__init__(sys=sys, backend=backend, **kwargs)
        self._humanoid_use_spring_gear = bool(
            backend == "mjx" and humanoid_use_spring_gear
        )
        self._humanoid_actuator_gear_mode = (
            "spring"
            if backend in ("spring", "positional") or self._humanoid_use_spring_gear
            else "xml"
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        self._target_ind = self.sys.link_names.index('target')
        self._robot_q_size = self.sys.q_size() - 2
        self._robot_qd_size = self.sys.qd_size() - 2
        position_size = self._robot_q_size - (
            2 if self._exclude_current_positions_from_observation else 0
        )

        self._robot_obs_dim = (
            position_size
            + self._robot_qd_size
            + 10 * self._target_ind
            + 6 * self._target_ind
        )
        self.layout_obs_dim = self._layout_lidar_num_bins
        self.layout_start_idx = self._robot_obs_dim
        self.layout_end_idx = self.layout_start_idx + self.layout_obs_dim
        self.state_dim = self.layout_end_idx
        self.raw_goal_dim = 3
        self.relabel_goal_dim = 3
        self.goal_start_idx = self.state_dim
        self.goal_end_idx = self.goal_start_idx + self.raw_goal_dim
        self.goal_indices = jnp.arange(self.goal_start_idx, self.goal_end_idx)
        self.scaling_crl_goal_indices = jnp.arange(0, self.raw_goal_dim)

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        qpos = self.sys.init_q + jax.random.uniform(rng1, [self.sys.q_size()], minval=low, maxval=hi)
        qvel = jax.random.uniform(rng2, [self.sys.qd_size()], minval=low, maxval=hi)

        # Set the start and target qpos and qvel
        start = self._random_start(rng3)
        qpos = qpos.at[:2].set(start)
        
        target = self._random_target(rng)
        qpos = qpos.at[-2:].set(target)
        qvel = qvel.at[-2:].set(0)       

        pipeline_state = self.pipeline_init(qpos, qvel)
        obs = self._get_obs(pipeline_state, jnp.zeros(self.sys.act_size()))
        
        reward, done, zero = jnp.zeros(3)
        root_position = pipeline_state.q[:3]
        metrics = {
            'forward_reward': zero,
            'reward_linvel': zero,
            'reward_quadctrl': zero,
            'reward_alive': zero,
            'x_position': root_position[0],
            'y_position': root_position[1],
            'z_position': root_position[2],
            'distance_from_origin': zero,
            'dist': zero,
            'x_velocity': zero,
            'y_velocity': zero,
            "success": zero,
            "success_easy": zero,
            "cost": zero,
            "episode_cost": zero,
            "goal_x": target[0],
            "goal_y": target[1],
            "goal_z": jnp.asarray(TARGET_Z_COORD),
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

    def step(self, state: State, action: jax.Array) -> State:
        """Runs one timestep of the environment's dynamics."""

        if "steps" in state.info.keys():
            seed = state.info["seed"] + jnp.where(state.info["steps"], 0, 1)
        else:
            seed = state.info["seed"]
        rng = state.info["rng"] if "rng" in state.info else jax.random.PRNGKey(0)
        rng, _ = jax.random.split(rng)
        info = {"seed": seed, "rng": rng}

        # Scale action from [-1,1] to actuator limits
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state0 = state.pipeline_state
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        com_before, *_ = self._com(pipeline_state0)
        com_after, *_ = self._com(pipeline_state)
        velocity = (com_after - com_before) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        min_z, max_z = self._healthy_z_range
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] < min_z, 0.0, 1.0)
        is_healthy = jnp.where(pipeline_state.x.pos[0, 2] > max_z, 0.0, is_healthy)
        if self._terminate_when_unhealthy:
            healthy_reward = self._healthy_reward
        else:
            healthy_reward = self._healthy_reward * is_healthy

        ctrl_cost = self._ctrl_cost_weight * jnp.sum(jnp.square(action))

        obs = self._get_obs(pipeline_state, action)
        distance_to_target = jnp.linalg.norm(obs[:3] - obs[-3:])

        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
        success = jnp.array(distance_to_target < self._goal_radius, dtype=float)
        success_easy = jnp.array(distance_to_target < 2., dtype=float)
        zero = jnp.asarray(0.0, dtype=obs.dtype)
        root_position = pipeline_state.q[:3]
        newly_reached = success * (1.0 - state.info["goal_reached"])
        goal_reached = jnp.maximum(state.info["goal_reached"], success)
        reward = newly_reached
        state.metrics.update(
            forward_reward=forward_reward,
            reward_linvel=forward_reward,
            reward_quadctrl=-ctrl_cost,
            reward_alive=healthy_reward,
            x_position=root_position[0],
            y_position=root_position[1],
            z_position=root_position[2],
            distance_from_origin=jnp.linalg.norm(root_position),
            dist=distance_to_target,
            x_velocity=velocity[0],
            y_velocity=velocity[1],
            success=success,
            success_easy=success_easy,
            cost=zero,
            episode_cost=zero,
            goal_x=obs[-3],
            goal_y=obs[-2],
            goal_z=obs[-1],
        )
        state.info.update(info)
        state.info.update(
            episode_cost=zero,
            goal_reached=goal_reached,
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(
        self, pipeline_state: base.State, action: jax.Array
    ) -> jax.Array:
        """Observes humanoid body position, velocities, and angles."""
        del action
        root_xy = pipeline_state.q[:2]
        position = pipeline_state.q[: self._robot_q_size]
        velocity = pipeline_state.qd[: self._robot_qd_size]

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        com, inertia, mass_sum, x_i = self._robot_com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jnp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )

        robot_x = jax.tree_util.tree_map(lambda x: x[: self._target_ind], pipeline_state.x)
        robot_xd = jax.tree_util.tree_map(lambda x: x[: self._target_ind], pipeline_state.xd)
        xd_i = (
            base.Transform.create(pos=x_i.pos - robot_x.pos)
            .vmap()
            .do(robot_xd)
        )
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_ang = xd_i.ang
        com_velocity = jnp.hstack([com_vel, com_ang])

        robot_obs = jnp.concatenate([
            position,
            velocity,
            com_inertia.ravel(),
            com_velocity.ravel(),
        ])
        target_pos = pipeline_state.x.pos[-1][:3]
        wall_lidar = self._wall_lidar(root_xy)
        # external_contact_forces are excluded
        return jnp.concatenate([robot_obs, wall_lidar, target_pos])

    def _robot_com(self, pipeline_state: base.State):
        inertia = jax.tree_util.tree_map(
            lambda x: x[: self._target_ind], self.sys.link.inertia
        )
        if self.backend in ['spring', 'positional']:
            inertia = inertia.replace(
                i=jax.vmap(jnp.diag)(
                    jax.vmap(jnp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jnp.sum(inertia.mass)
        robot_x = jax.tree_util.tree_map(lambda x: x[: self._target_ind], pipeline_state.x)
        x_i = robot_x.vmap().do(inertia.transform)
        com = (
            jnp.sum(jax.vmap(jnp.multiply)(inertia.mass, x_i.pos), axis=0)
            / mass_sum
        )
        return com, inertia, mass_sum, x_i

    def _wall_lidar(self, xy: jax.Array) -> jax.Array:
        if self.wall_centers.shape[0] == 0:
            return jnp.zeros((self._layout_lidar_num_bins,), dtype=xy.dtype)
        dtype = xy.dtype
        angles = (
            jnp.arange(self._layout_lidar_num_bins, dtype=dtype)
            * (2.0 * jnp.pi / self._layout_lidar_num_bins)
        )
        dirs = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        centers = self.wall_centers.astype(dtype)
        half = jnp.asarray(0.5 * self._maze_size_scaling, dtype=dtype)
        bounds_min = centers - half
        bounds_max = centers + half
        eps_dirs = jnp.where(dirs >= 0.0, 1e-6, -1e-6)
        inv_dirs = 1.0 / jnp.where(jnp.abs(dirs) < 1e-6, eps_dirs, dirs)
        t1 = (bounds_min[None, :, :] - xy[None, None, :]) * inv_dirs[:, None, :]
        t2 = (bounds_max[None, :, :] - xy[None, None, :]) * inv_dirs[:, None, :]
        tmin = jnp.max(jnp.minimum(t1, t2), axis=-1)
        tmax = jnp.min(jnp.maximum(t1, t2), axis=-1)
        hit = (tmax >= jnp.maximum(tmin, 0.0)) & (tmin <= self._layout_lidar_max_dist)
        hit_dist = jnp.where(hit, jnp.maximum(tmin, 0.0), jnp.inf)
        nearest = jnp.min(hit_dist, axis=-1)
        signal = 1.0 - nearest / jnp.asarray(self._layout_lidar_max_dist, dtype=dtype)
        return jnp.where(jnp.isfinite(nearest), jnp.clip(signal, 0.0, 1.0), 0.0)

    def _com(self, pipeline_state: base.State) -> jax.Array:
        inertia = self.sys.link.inertia
        if self.backend in ['spring', 'positional']:
            inertia = inertia.replace(
                i=jax.vmap(jnp.diag)(
                    jax.vmap(jnp.diagonal)(inertia.i)
                    ** (1 - self.sys.spring_inertia_scale)
                ),
                mass=inertia.mass ** (1 - self.sys.spring_mass_scale),
            )
        mass_sum = jnp.sum(inertia.mass)
        x_i = pipeline_state.x.vmap().do(inertia.transform)
        com = (
            jnp.sum(jax.vmap(jnp.multiply)(inertia.mass, x_i.pos), axis=0) / mass_sum
        )
        return com, inertia, mass_sum, x_i  # pytype: disable=bad-return-type  # jax-ndarray
    
    def _random_target(self, rng: jax.Array) -> jax.Array:
        """Returns a random target location chosen from possibilities specified in the maze layout."""
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_goals))
        return jnp.array(self.possible_goals[idx])[0]

    def _random_start(self, rng: jax.Array) -> jax.Array:
        idx = jax.random.randint(rng, (1,), 0, len(self.possible_starts))
        return jnp.array(self.possible_starts[idx])[0]


class CrlHumanoidMazeNoWall(CrlHumanoidMaze):
    """Humanoid maze with visible ghost walls and terminal wall entry."""

    def __init__(self, backend="spring", **kwargs):
        super().__init__(backend=backend, walls_collide=False, **kwargs)

    def _robot_inside_wall(self, robot_xy: jax.Array) -> jax.Array:
        if self.wall_centers.shape[0] == 0:
            return jnp.asarray(False)
        half = jnp.asarray(0.5 * self._maze_size_scaling, dtype=robot_xy.dtype)
        centers = self.wall_centers.astype(robot_xy.dtype)
        inside_each = jnp.all(
            jnp.abs(robot_xy[None, :] - centers) <= half,
            axis=-1,
        )
        return jnp.any(inside_each)

    def step(self, state: State, action: jax.Array) -> State:
        next_state = super().step(state, action)
        wall_entry = self._robot_inside_wall(next_state.pipeline_state.q[:2])
        done = jnp.maximum(next_state.done, wall_entry.astype(next_state.done.dtype))
        return next_state.replace(done=done)
