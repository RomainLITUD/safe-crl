"""Ant goal environment."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from brax.io import mjcf
import mujoco

from safenav_jax.envs._continuous_goal_lidar import TwoChannelGlobalLidarMixin
from safenav_jax.envs.renderable_goal_base import RenderableGoalEnv


XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "xmls" / "ant.xml"


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
        raise ValueError("ant.xml is missing a worldbody.")
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
    init_qpos = tree.getroot().find(".//numeric[@name='init_qpos']")
    if init_qpos is not None:
        extra_q = 2 * (num_hazards + num_gremlins)
        init_qpos.set("data", init_qpos.get("data", "") + " " + " ".join(["0.0"] * extra_q))
    return ET.tostring(tree.getroot(), encoding="unicode")


class AntGoal(TwoChannelGlobalLidarMixin, RenderableGoalEnv):
    """Ant goal environment with a virtual bounded task layout."""

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
        hazard_height: float = 0.5,
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
                    "opt.timestep": 0.005,
                    "opt.solver": mujoco.mjtSolver.mjSOL_NEWTON,
                    "opt.disableflags": mujoco.mjtDisableBit.mjDSBL_EULERDAMP,
                    "opt.iterations": 1,
                    "opt.ls_iterations": 4,
                }
            )
        self._full_robot_observation = full_robot_observation
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
            robot_q_size=15,
            robot_qd_size=14,
            visual_target_q_idx=15,
            visual_layout_q_idx=17,
            visual_target_qd_idx=14,
            include_object_layout_obs=include_object_layout_obs,
            uniform_initial_goal_sampling=uniform_initial_goal_sampling,
            first_valid_layout_candidate=first_valid_layout_candidate,
            initial_goal_path_objects_mode=True,
            initial_goal_path_objects_probability=1.0 if evaluation_mode else 0.5,
            layout_lidar_ego_frame=False,
            **kwargs,
        )
