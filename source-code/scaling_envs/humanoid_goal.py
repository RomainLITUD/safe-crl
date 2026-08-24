"""Cost-free renderable Humanoid goal task for Scaling-CRL methods."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from brax import base
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco

from scaling_envs.goal_base import ScalingGoalEnv
from safenav_jax.envs._humanoid_actuation import apply_spring_humanoid_gear_for_mjx


XML_PATH = Path(__file__).resolve().parent / "assets" / "xmls" / "humanoid.xml"
TARGET_Z_COORD = 1.25


def _xml_with_goal_radius(goal_radius: float) -> str:
    tree = ET.parse(XML_PATH)
    target = tree.getroot().find("./worldbody/body[@name='target']/geom[@name='target']")
    if target is None:
        raise ValueError("humanoid.xml is missing the target geom.")
    target.set("size", str(goal_radius))
    return ET.tostring(tree.getroot(), encoding="unicode")


class HumanoidGoal(ScalingGoalEnv):
    """Humanoid starts at the origin and navigates to one fixed global XY goal."""

    def __init__(
        self,
        backend: str = "mjx",
        n_frames: int = 10,
        episode_length: int = 1000,
        reset_noise_scale: float = 0.0,
        playground_size: float = 6.0,
        min_goal_dist: float = 1.0,
        max_goal_dist: float = 5.0,
        eval_min_goal_dist: float = 2.0,
        eval_max_goal_dist: float = 5.0,
        evaluation_mode: bool = False,
        goal_radius: float = 0.55,
        healthy_z_range: tuple[float, float] = (1.0, 2.0),
        full_robot_observation: bool = True,
        humanoid_use_spring_gear: bool = False,
        **kwargs,
    ):
        active_min_goal_dist = eval_min_goal_dist if evaluation_mode else min_goal_dist
        active_max_goal_dist = eval_max_goal_dist if evaluation_mode else max_goal_dist
        if max_goal_dist < min_goal_dist:
            raise ValueError("max_goal_dist must be at least min_goal_dist.")
        if eval_max_goal_dist < eval_min_goal_dist:
            raise ValueError("eval_max_goal_dist must be at least eval_min_goal_dist.")
        self._radial_min_goal_dist = float(active_min_goal_dist)
        self._radial_max_goal_dist = float(active_max_goal_dist)

        sys = mjcf.loads(_xml_with_goal_radius(goal_radius))
        if backend == "mjx":
            sys = sys.tree_replace(
                {
                    "opt.timestep": 0.0015,
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        sys = apply_spring_humanoid_gear_for_mjx(
            sys,
            backend=backend,
            enabled=humanoid_use_spring_gear,
        )
        self._humanoid_use_spring_gear = bool(
            backend == "mjx" and humanoid_use_spring_gear
        )
        self._humanoid_actuator_gear_mode = (
            "spring" if self._humanoid_use_spring_gear else "xml"
        )
        self._full_robot_observation = full_robot_observation
        self._robot_link_count = sys.link_names.index("target")
        super().__init__(
            sys=sys,
            backend=backend,
            n_frames=n_frames,
            episode_length=episode_length,
            reset_noise_scale=reset_noise_scale,
            playground_size=playground_size,
            min_goal_dist=min_goal_dist,
            eval_min_goal_dist=eval_min_goal_dist,
            evaluation_mode=evaluation_mode,
            goal_radius=goal_radius,
            healthy_z_range=healthy_z_range,
            robot_q_size=24,
            robot_qd_size=23,
            target_q_idx=24,
            target_qd_idx=23,
            **kwargs,
        )

        self.raw_goal_dim = 3
        self.relabel_goal_dim = 3
        self.goal_indices = jp.arange(self._robot_obs_dim, self._robot_obs_dim + 3)
        self.scaling_crl_goal_indices = jp.arange(3, dtype=jp.int32)

    def _sample_goal(self, rng: jax.Array) -> jax.Array:
        """Samples radius and angle exactly like Scaling-CRL's Humanoid."""
        _, radius_rng, angle_rng = jax.random.split(rng, 3)
        radius = jax.random.uniform(
            radius_rng,
            minval=self._radial_min_goal_dist,
            maxval=self._radial_max_goal_dist,
        )
        angle = 2.0 * jp.pi * jax.random.uniform(angle_rng)
        return radius * jp.stack([jp.cos(angle), jp.sin(angle)])

    def _goal_xyz(self, goal_xy: jax.Array) -> jax.Array:
        return jp.concatenate(
            [goal_xy, jp.asarray([TARGET_Z_COORD], dtype=goal_xy.dtype)]
        )

    def _distance(self, pipeline_state: base.State, goal_xy: jax.Array) -> jax.Array:
        return jp.linalg.norm(pipeline_state.x.pos[0, :3] - self._goal_xyz(goal_xy))

    def _get_obs(
        self,
        pipeline_state: base.State,
        action: jax.Array,
        goal_xy: jax.Array,
    ) -> jax.Array:
        return jp.concatenate(
            [self._robot_obs(pipeline_state, action), self._goal_xyz(goal_xy)]
        )

    def _metrics(
        self,
        success: jax.Array,
        dist: jax.Array,
        pipeline_state: base.State,
        goal_xy: jax.Array,
    ) -> dict[str, jax.Array]:
        metrics = super()._metrics(success, dist, pipeline_state, goal_xy)
        metrics.update(
            z_position=pipeline_state.x.pos[0, 2],
            goal_z=jp.asarray(TARGET_Z_COORD, dtype=goal_xy.dtype),
        )
        return metrics

    def _robot_obs_size(self) -> int:
        if not self._full_robot_observation:
            return self._robot_q_size + self._robot_qd_size
        num_links = int(self._robot_link_count)
        return (
            self._robot_q_size
            + self._robot_qd_size
            + 10 * num_links
            + 6 * num_links
        )

    def _robot_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        del action
        position = pipeline_state.q[: self._robot_q_size]
        velocity = pipeline_state.qd[: self._robot_qd_size]
        if not self._full_robot_observation:
            return jp.concatenate([position, velocity])

        com, inertia, mass_sum, x_i = self._com(pipeline_state)
        cinr = x_i.replace(pos=x_i.pos - com).vmap().do(inertia)
        com_inertia = jp.hstack(
            [cinr.i.reshape((cinr.i.shape[0], -1)), inertia.mass[:, None]]
        )
        robot_x = jax.tree_util.tree_map(lambda x: x[: self._robot_link_count], pipeline_state.x)
        robot_xd = jax.tree_util.tree_map(lambda x: x[: self._robot_link_count], pipeline_state.xd)
        xd_i = (
            base.Transform.create(pos=x_i.pos - robot_x.pos)
            .vmap()
            .do(robot_xd)
        )
        com_vel = inertia.mass[:, None] * xd_i.vel / mass_sum
        com_velocity = jp.hstack([com_vel, xd_i.ang])
        return jp.concatenate(
            [
                position,
                velocity,
                com_inertia.ravel(),
                com_velocity.ravel(),
            ]
        )

    def _com(
        self,
        pipeline_state: base.State,
    ) -> tuple[jax.Array, base.Inertia, jax.Array, base.Transform]:
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
