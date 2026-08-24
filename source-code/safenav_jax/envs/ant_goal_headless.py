"""Headless ant goal environment."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from brax.io import mjcf
import mujoco

from safenav_jax.envs._continuous_goal_lidar import TwoChannelGlobalLidarMixin
from safenav_jax.envs.headless_goal_base import HeadlessGoalEnv


XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "xmls" / "ant.xml"


def _xml_without_target() -> str:
    tree = ET.parse(XML_PATH)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("ant.xml is missing a worldbody.")
    target = worldbody.find("./body[@name='target']")
    if target is not None:
        worldbody.remove(target)

    init_qpos = root.find(".//numeric[@name='init_qpos']")
    if init_qpos is not None:
        values = init_qpos.get("data", "").split()
        init_qpos.set("data", " ".join(values[:-2]))
    return ET.tostring(root, encoding="unicode")


class AntGoalHeadless(TwoChannelGlobalLidarMixin, HeadlessGoalEnv):
    """Ant goal with task objects represented by arrays."""

    def __init__(
        self,
        backend: str = "mjx",
        n_frames: int = 10,
        episode_length: int = 1000,
        reset_noise_scale: float = 0.1,
        min_goal_dist: float = 4.0,
        max_goal_dist: float = 7.0,
        goal_radius: float = 0.9,
        healthy_z_range: tuple[float, float] = (0.2, 1.0),
        playground_size: float = 12.0,
        agent_wall_margin: float = 2.0,
        goal_wall_margin: float = 2.0,
        num_hazards: int = 12,
        num_gremlins: int = 3,
        hazard_radius: float = 0.75,
        gremlin_radius: float = 0.6,
        gremlin_height: float = 0.01,
        gremlin_travel: float = 2.0,
        gremlin_speed: float = 0.1,
        layout_margin: float = 0.0,
        agent_keepout: float = 2.0,
        goal_keepout: float = 2.0,
        hazard_keepout: float = 2.0,
        gremlin_keepout: float = 2.6,
        full_robot_observation: bool = True,
        include_object_layout_obs: bool = True,
        uniform_initial_goal_sampling: bool = True,
        first_valid_layout_candidate: bool = True,
        fixed_agent_on_reset: bool = False,
        evaluation_mode: bool = False,
        **kwargs,
    ):
        self._full_robot_observation = full_robot_observation
        sys = mjcf.loads(_xml_without_target())
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
            gremlin_radius=gremlin_radius,
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
            obstacle_height=0.0,
            gremlin_height=gremlin_height,
            agent_wall_margin=agent_wall_margin,
            use_3d_object_cost=False,
            success_link_name="torso",
            healthy_z_range=healthy_z_range,
            include_object_layout_obs=include_object_layout_obs,
            uniform_initial_goal_sampling=uniform_initial_goal_sampling,
            first_valid_layout_candidate=first_valid_layout_candidate,
            initial_goal_path_objects_mode=True,
            initial_goal_path_objects_probability=1.0 if evaluation_mode else 0.5,
            layout_lidar_ego_frame=False,
            **kwargs,
        )
