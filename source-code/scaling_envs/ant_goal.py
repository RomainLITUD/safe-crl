"""Cost-free renderable Ant goal task for Scaling-CRL methods."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from brax import base
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco

from scaling_envs.goal_base import ScalingGoalEnv


XML_PATH = Path(__file__).resolve().parent / "assets" / "xmls" / "ant.xml"


def _xml_with_goal_radius(goal_radius: float) -> str:
    tree = ET.parse(XML_PATH)
    target = tree.getroot().find("./worldbody/body[@name='target']/geom[@name='target']")
    if target is None:
        raise ValueError("ant.xml is missing the target geom.")
    target.set("size", str(goal_radius))
    return ET.tostring(tree.getroot(), encoding="unicode")


class AntGoal(ScalingGoalEnv):
    """Ant starts at the origin and navigates to one fixed global XY goal."""

    def __init__(
        self,
        backend: str = "mjx",
        n_frames: int = 10,
        episode_length: int = 1000,
        reset_noise_scale: float = 0.1,
        playground_size: float = 10.0,
        min_goal_dist: float = 10.0,
        max_goal_dist: float = 10.0,
        eval_min_goal_dist: float = 10.0,
        eval_max_goal_dist: float = 10.0,
        evaluation_mode: bool = False,
        goal_radius: float = 0.9,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
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
                    "opt.timestep": 0.005,
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
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
            robot_q_size=15,
            robot_qd_size=14,
            target_q_idx=15,
            target_qd_idx=14,
            **kwargs,
        )

    def _sample_goal(self, rng: jax.Array) -> jax.Array:
        """Samples a radial goal, matching Scaling-CRL's fixed-radius default."""
        _, radius_rng, angle_rng = jax.random.split(rng, 3)
        radius = jax.random.uniform(
            radius_rng,
            minval=self._radial_min_goal_dist,
            maxval=self._radial_max_goal_dist,
        )
        angle = 2.0 * jp.pi * jax.random.uniform(angle_rng)
        return radius * jp.stack([jp.cos(angle), jp.sin(angle)])

    def _robot_obs_size(self) -> int:
        return self._robot_q_size + self._robot_qd_size

    def _robot_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
        del action
        return jp.concatenate(
            [
                pipeline_state.q[: self._robot_q_size],
                pipeline_state.qd[: self._robot_qd_size],
            ]
        )
