"""Humanoid goal environment."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from brax import base
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco

from safenav_jax.envs._continuous_goal_lidar import TwoChannelGlobalLidarMixin
from safenav_jax.envs._humanoid_actuation import apply_spring_humanoid_gear_for_mjx
from safenav_jax.envs.renderable_goal_base import RenderableGoalEnv


XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "xmls" / "humanoid.xml"


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
        range="-20 20",
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
        range="-20 20",
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


def _xml_with_layout(
    goal_radius: float,
    num_hazards: int,
    num_gremlins: int,
    hazard_radius: float,
    hazard_height: float,
    gremlin_radius: float,
    gremlin_height: float,
) -> str:
    tree = ET.parse(XML_PATH)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError("humanoid.xml is missing a worldbody.")
    target_geom = worldbody.find("./body[@name='target']/geom[@name='target']")
    if target_geom is not None:
        target_geom.set("size", str(goal_radius))

    for i in range(num_hazards):
        _add_xy_slide_body(
            worldbody,
            f"hazard{i}",
            hazard_height / 2.0,
            "cylinder",
            f"{hazard_radius} {hazard_height / 2.0}",
            "0 0 1 0.25",
        )
    for i in range(num_gremlins):
        _add_xy_slide_body(
            worldbody,
            f"gremlin{i}",
            gremlin_height,
            "sphere",
            f"{gremlin_radius}",
            "0.5 0 1 1",
        )
    return ET.tostring(tree.getroot(), encoding="unicode")


class HumanoidGoal(TwoChannelGlobalLidarMixin, RenderableGoalEnv):
    """Humanoid goal environment with a virtual bounded task layout."""

    def __init__(
        self,
        backend: str = "mjx",
        n_frames: int = 10,
        episode_length: int = 1000,
        reset_noise_scale: float = 0.0,
        min_goal_dist: float = 2.0,
        max_goal_dist: float = 7.0,
        goal_radius: float = 0.55,
        healthy_z_range: tuple[float, float] = (1.0, 2.0),
        playground_size: float = 6.0,
        agent_wall_margin: float = 1.0,
        goal_wall_margin: float = 1.0,
        num_hazards: int = 12,
        num_gremlins: int = 3,
        hazard_radius: float = 0.35,
        hazard_height: float = 0.5,
        gremlin_radius: float = 0.3,
        gremlin_height: float = 1.25,
        gremlin_travel: float = 1.0,
        gremlin_speed: float = 0.1,
        layout_margin: float = 0.0,
        agent_keepout: float = 1.0,
        goal_keepout: float = 1.0,
        hazard_keepout: float = 1.0,
        gremlin_keepout: float = 1.3,
        full_robot_observation: bool = True,
        include_object_layout_obs: bool = True,
        uniform_initial_goal_sampling: bool = True,
        first_valid_layout_candidate: bool = True,
        fixed_agent_on_reset: bool = False,
        evaluation_mode: bool = False,
        humanoid_use_spring_gear: bool = False,
        **kwargs,
    ):
        sys = mjcf.loads(
            _xml_with_layout(
                goal_radius,
                num_hazards,
                num_gremlins,
                hazard_radius,
                hazard_height,
                gremlin_radius,
                gremlin_height,
            )
        )
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
            min_goal_dist=min_goal_dist,
            max_goal_dist=max_goal_dist,
            goal_radius=goal_radius,
            goal_z=None,
            playground_size=playground_size,
            playground_center_xy=(0.0, 0.0),
            goal_wall_margin=goal_wall_margin,
            num_hazards=num_hazards,
            num_obstacles=0,
            num_gremlins=num_gremlins,
            hazard_radius=hazard_radius,
            obstacle_radius=0.0,
            obstacle_height=0.0,
            gremlin_radius=gremlin_radius,
            gremlin_height=gremlin_height,
            gremlin_travel=gremlin_travel,
            gremlin_speed=gremlin_speed,
            layout_margin=layout_margin,
            agent_keepout=agent_keepout,
            goal_keepout=goal_keepout,
            hazard_keepout=hazard_keepout,
            obstacle_keepout=0.0,
            gremlin_keepout=gremlin_keepout,
            random_agent=not fixed_agent_on_reset,
            fixed_agent_on_reset=fixed_agent_on_reset,
            fixed_agent_xy=(0.0, 0.0),
            fixed_agent_orientation_on_reset=True,
            agent_wall_margin=agent_wall_margin,
            use_3d_object_cost=False,
            success_link_name="torso",
            healthy_z_range=healthy_z_range,
            include_actuator_forces=True,
            robot_q_size=24,
            robot_qd_size=23,
            visual_target_q_idx=24,
            visual_layout_q_idx=26,
            visual_target_qd_idx=23,
            include_object_layout_obs=include_object_layout_obs,
            uniform_initial_goal_sampling=uniform_initial_goal_sampling,
            first_valid_layout_candidate=first_valid_layout_candidate,
            initial_goal_path_objects_mode=True,
            initial_goal_path_objects_probability=1.0 if evaluation_mode else 0.5,
            layout_lidar_ego_frame=False,
            **kwargs,
        )
    def _robot_obs_size(self) -> int:
        q_size = self._robot_q_size
        if not self._full_robot_observation:
            return q_size + self._robot_qd_size
        num_links = int(self._robot_link_count)
        return q_size + self._robot_qd_size + 10 * num_links + 6 * num_links + self.sys.act_size()

    def _robot_obs(self, pipeline_state: base.State, action: jax.Array) -> jax.Array:
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
        com_ang = xd_i.ang
        com_velocity = jp.hstack([com_vel, com_ang])

        return jp.concatenate(
            [
                position,
                velocity,
                com_inertia.ravel(),
                com_velocity.ravel(),
                getattr(
                    pipeline_state,
                    "actuator_force",
                    jp.zeros((self.sys.act_size(),), dtype=position.dtype),
                ),
            ]
        )

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
