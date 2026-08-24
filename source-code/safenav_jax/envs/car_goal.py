"""Car goal environment."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from brax.io import mjcf

from safenav_jax.envs.renderable_goal_base import RenderableGoalEnv


XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "xmls" / "car.xml"


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
        range="-10 10",
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
        range="-10 10",
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
    num_obstacles: int,
    num_gremlins: int,
    hazard_radius: float,
    obstacle_radius: float,
    obstacle_height: float,
    gremlin_radius: float,
    gremlin_height: float,
    parking_mode: bool,
) -> str:
    tree = ET.parse(XML_PATH)
    worldbody = tree.getroot().find("worldbody")
    if worldbody is None:
        raise ValueError("car.xml is missing a worldbody.")

    target_z = goal_radius / 2.0 + 0.01
    target = ET.SubElement(worldbody, "body", name="target", pos=f"0 0 {target_z}")
    ET.SubElement(
        target,
        "joint",
        name="target_x",
        type="slide",
        axis="1 0 0",
        limited="true",
        range="-10 10",
        damping="0",
        stiffness="0",
        armature="0",
    )
    ET.SubElement(
        target,
        "joint",
        name="target_y",
        type="slide",
        axis="0 1 0",
        limited="true",
        range="-10 10",
        damping="0",
        stiffness="0",
        armature="0",
    )
    if parking_mode:
        ET.SubElement(
            target,
            "joint",
            name="target_yaw",
            type="hinge",
            axis="0 0 1",
            limited="false",
            damping="0",
            stiffness="0",
            armature="0",
        )
    ET.SubElement(
        target,
        "geom",
        name="target",
        type="cylinder",
        size=f"{goal_radius} {goal_radius / 2.0}",
        contype="0",
        conaffinity="0",
        rgba="0 1 0 0.25",
        mass="0.001",
    )
    if parking_mode:
        ET.SubElement(
            target,
            "geom",
            name="target_heading",
            type="box",
            pos=f"{0.55 * goal_radius} 0 0",
            size=f"{0.45 * goal_radius} {0.15 * goal_radius} {goal_radius / 2.0}",
            contype="0",
            conaffinity="0",
            rgba="0 0.7 0 0.65",
            mass="0.001",
        )
    for i in range(num_hazards):
        _add_xy_slide_body(
            worldbody,
            f"hazard{i}",
            0.02,
            "cylinder",
            f"{hazard_radius} 0.01",
            "0 0 1 0.25",
        )
    for i in range(num_obstacles):
        _add_xy_slide_body(
            worldbody,
            f"obstacle{i}",
            obstacle_height,
            "sphere",
            f"{obstacle_radius}",
            "0.6 0.45 0.3 1",
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


class CarGoal(RenderableGoalEnv):
    """A minimal car environment with one randomly sampled goal."""

    def __init__(
        self,
        backend: str = "mjx",
        n_frames: int = 10,
        episode_length: int = 1000,
        reset_noise_scale: float = 0.0,
        agent_spawn_bound: float | None = None,
        playground_size: float = 2.0,
        goal_wall_margin: float = 0.305,
        min_goal_dist: float = 1.5,
        max_goal_dist: float = 3.0,
        goal_radius: float = 0.3,
        num_hazards: int = 10,
        num_obstacles: int = 10,
        num_gremlins: int = 6,
        hazard_radius: float = 0.2,
        obstacle_radius: float = 0.1,
        obstacle_height: float = 0.1,
        gremlin_radius: float = 0.1,
        gremlin_height: float = 0.1,
        gremlin_travel: float = 0.35,
        gremlin_speed: float = 1.0,
        layout_margin: float = 0.0,
        agent_keepout: float = 0.4,
        goal_keepout: float = 0.305,
        hazard_keepout: float = 0.18,
        obstacle_keepout: float = 0.15,
        gremlin_keepout: float = 0.4,
        include_object_layout_obs: bool = True,
        uniform_initial_goal_sampling: bool = True,
        first_valid_layout_candidate: bool = True,
        ego_view: bool = True,
        parking_mode: bool = False,
        parking_yaw_tolerance_degrees: float = 20.0,
        **kwargs,
    ):
        sys = mjcf.loads(
            _xml_with_layout(
                goal_radius,
                num_hazards,
                num_obstacles,
                num_gremlins,
                hazard_radius,
                obstacle_radius,
                obstacle_height,
                gremlin_radius,
                gremlin_height,
                parking_mode,
            )
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
            playground_size=playground_size,
            goal_wall_margin=goal_wall_margin,
            num_hazards=num_hazards,
            num_obstacles=num_obstacles,
            num_gremlins=num_gremlins,
            hazard_radius=hazard_radius,
            obstacle_radius=obstacle_radius,
            obstacle_height=obstacle_height,
            gremlin_radius=gremlin_radius,
            gremlin_height=gremlin_height,
            gremlin_travel=gremlin_travel,
            gremlin_speed=gremlin_speed,
            layout_margin=layout_margin,
            agent_keepout=agent_keepout,
            goal_keepout=goal_keepout,
            hazard_keepout=hazard_keepout,
            obstacle_keepout=obstacle_keepout,
            gremlin_keepout=gremlin_keepout,
            random_agent=True,
            agent_spawn_bound=agent_spawn_bound,
            random_yaw=True,
            success_link_name="agent",
            include_object_layout_obs=include_object_layout_obs,
            uniform_initial_goal_sampling=uniform_initial_goal_sampling,
            first_valid_layout_candidate=first_valid_layout_candidate,
            robot_sensor_data_start=7 if ego_view else None,
            robot_sensor_data_dim=12,
            layout_lidar_ego_frame=ego_view,
            robot_quat_q_start=3,
            parking_mode=parking_mode,
            parking_yaw_tolerance_degrees=parking_yaw_tolerance_degrees,
            robot_q_size=13,
            robot_qd_size=11,
            visual_target_q_idx=13,
            visual_layout_q_idx=16 if parking_mode else 15,
            visual_target_qd_idx=11,
            initial_goal_require_path_objects=True,
            terminate_on_goal_exit_after_success=True,
            **kwargs,
        )
