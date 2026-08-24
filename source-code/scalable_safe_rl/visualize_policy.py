"""Deterministic policy rollout visualization for saved scalable_safe_rl actors."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from brax.io import html
from brax.io import json as brax_json

from safenav_jax.env_config import load_env_config
from safenav_jax.envs import config_env_id_for, make as make_safenav_env
from safenav_jax.visualization_rollout import sync_mocap_pipeline_state_for_render

try:
    from .train import (
        Actor,
        Args,
        SUPPORTED_CRITIC_LOSS_TYPES,
        EgoGoalObservationWrapper,
        PointPushCombinedGoalObservationWrapper,
        ScalingCrlObservationWrapper,
        _optional_bool_arg,
        is_safenav_env_id,
        is_scaling_env_id,
        is_ant_humanoid_goal_env_id,
        is_humanoid_env_id,
        is_point_car_goal_env_id,
        is_point_push_env_id,
        safenav_env_id_hint,
        load_params,
    )
except ImportError:
    from train import (
        Actor,
        Args,
        SUPPORTED_CRITIC_LOSS_TYPES,
        EgoGoalObservationWrapper,
        PointPushCombinedGoalObservationWrapper,
        ScalingCrlObservationWrapper,
        _optional_bool_arg,
        is_safenav_env_id,
        is_scaling_env_id,
        is_ant_humanoid_goal_env_id,
        is_humanoid_env_id,
        is_point_car_goal_env_id,
        is_point_push_env_id,
        safenav_env_id_hint,
        load_params,
    )


class _CompatUnpickler(pickle.Unpickler):
    """Unpickles old args.pkl files saved from python -m entrypoints."""

    def find_class(self, module: str, name: str):
        if module == "__main__" and name == "Args":
            return Args
        return super().find_class(module, name)


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return _CompatUnpickler(f).load()


def _args_from_saved(raw_args: Any) -> Args:
    if isinstance(raw_args, Args):
        args_dict = dict(vars(raw_args))
    elif isinstance(raw_args, dict):
        args_dict = dict(raw_args)
    else:
        args_dict = dict(vars(raw_args))

    args = Args()
    for name, value in args_dict.items():
        if hasattr(args, name):
            setattr(args, name, value)
    if args.critic_loss_type not in SUPPORTED_CRITIC_LOSS_TYPES:
        raise ValueError(
            f"Unsupported saved critic_loss_type={args.critic_loss_type!r}. "
            f"Expected one of {SUPPORTED_CRITIC_LOSS_TYPES!r}."
        )
    return args


def _load_policy(policy_dir: str | Path) -> tuple[Any, Args, Path]:
    """Loads actor params and training args from a policy or run directory."""
    root = Path(policy_dir)
    if not root.exists():
        raise FileNotFoundError(f"Policy path does not exist: {root}")

    if (root / "policy.pkl").exists():
        bundle = load_params(str(root / "policy.pkl"))
        return bundle["actor_params"], _args_from_saved(bundle["args"]), root

    if (root / "actor_params.pkl").exists() and (root / "args.pkl").exists():
        actor_params = load_params(str(root / "actor_params.pkl"))
        args = _args_from_saved(_load_pickle(root / "args.pkl"))
        return actor_params, args, root

    final_policy_dir = root / "final_policy"
    if (final_policy_dir / "policy.pkl").exists():
        bundle = load_params(str(final_policy_dir / "policy.pkl"))
        return bundle["actor_params"], _args_from_saved(bundle["args"]), final_policy_dir

    if (root / "final.pkl").exists() and (root / "args.pkl").exists():
        checkpoint = load_params(str(root / "final.pkl"))
        if not isinstance(checkpoint, tuple) or len(checkpoint) not in (3, 4):
            raise ValueError("Legacy final.pkl must contain a 3- or 4-item checkpoint tuple.")
        actor_params = checkpoint[1]
        args = _args_from_saved(_load_pickle(root / "args.pkl"))
        return actor_params, args, root

    raise FileNotFoundError(
        "Could not find a saved policy. Expected one of: policy.pkl, "
        "actor_params.pkl + args.pkl, final_policy/policy.pkl, or final.pkl + args.pkl."
    )


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if value == "":
        return None
    return _optional_bool_arg(value)


def _arg(saved_args: Args, name: str, default: Any = None) -> Any:
    return getattr(saved_args, name, default)


def _saved_env_params(saved_args: Args, env_id: str) -> dict[str, Any] | None:
    eval_env_id = _arg(saved_args, "eval_env_id", "")
    eval_params = _arg(saved_args, "eval_env_params", None)
    eval_render_env_id = eval_env_id
    if eval_render_env_id.endswith("_headless"):
        eval_render_env_id = eval_render_env_id[: -len("_headless")]
    if eval_params and (not eval_env_id or env_id == eval_env_id or env_id == eval_render_env_id):
        return dict(eval_params)
    env_params = _arg(saved_args, "env_params", None)
    if env_params:
        return dict(env_params)
    return None


def _renderable_env_id(saved_args: Args, override: str) -> str:
    env_id = override or _arg(saved_args, "eval_env_id", "") or saved_args.env_id
    if is_scaling_env_id(env_id):
        return env_id
    if env_id.endswith("_headless"):
        env_id = env_id[: -len("_headless")]
    if not is_safenav_env_id(env_id):
        raise ValueError(f"Unsupported env_id={env_id!r}. Expected one of {safenav_env_id_hint()}.")
    if env_id.endswith("_headless"):
        raise ValueError(f"Visualization needs a renderable env_id, got {env_id!r}.")
    return env_id


def _make_env(saved_args: Args, env_id: str):
    saved_params = _saved_env_params(saved_args, env_id)
    if is_scaling_env_id(env_id):
        import scaling_envs
        from scaling_envs.env_config import load_env_config as load_scaling_env_config

        if saved_params is not None:
            env_params = dict(saved_params)
        else:
            env_params = {}
            if _arg(saved_args, "use_env_config", True):
                env_params.update(
                    load_scaling_env_config(
                        scaling_envs.config_env_id_for(env_id),
                        config_dir=_arg(saved_args, "env_config_dir", "") or None,
                        config_path=_arg(saved_args, "env_config_path", "") or None,
                    )
                )
        if scaling_envs.is_maze_env_id(env_id):
            env_params.setdefault(
                "layout_lidar_num_bins",
                _arg(saved_args, "layout_lidar_num_bins", 16),
            )
        if is_humanoid_env_id(env_id):
            env_params.setdefault(
                "humanoid_use_spring_gear",
                _arg(saved_args, "humanoid_use_spring_gear", False),
            )
        env_params.setdefault("evaluation_mode", True)
        env = scaling_envs.make(env_id, use_config=False, **env_params)
        env = ScalingCrlObservationWrapper(env, env.goal_indices)

    else:
        if saved_params is not None:
            env_params = dict(saved_params)
        else:
            env_config_dir = _arg(saved_args, "env_config_dir", "") or None
            env_config_path = _arg(saved_args, "env_config_path", "") or None
            env_kwargs: dict[str, Any] = {
                "relocate_objects_on_reset": _arg(saved_args, "relocate_objects_on_reset", True),
                "fixed_object_layout_seed": _arg(saved_args, "fixed_object_layout_seed", 0),
                "different_object_layout_per_env": _arg(saved_args, "different_object_layout_per_env", False),
                "goal_respawn_on_success": _arg(saved_args, "goal_respawn_on_success", True),
                "terminate_on_cost": _arg(saved_args, "eval_terminate_on_cost", False),
                "cost_limit_max": _arg(saved_args, "cost_limit_max", 25.0),
            }
            if is_humanoid_env_id(env_id):
                env_kwargs["humanoid_use_spring_gear"] = _arg(
                    saved_args,
                    "humanoid_use_spring_gear",
                    False,
                )
            if (
                ("goal" in env_id and "maze" not in env_id)
                or is_point_push_env_id(env_id)
            ):
                env_kwargs["object_boundary"] = _arg(saved_args, "object_boundary", False)
            if ("goal" in env_id and "grid" not in env_id) or is_point_push_env_id(env_id):
                env_kwargs.update(
                    {
                        "respawn_goal_require_path_objects": _arg(saved_args, "respawn_goal_require_path_objects", False),
                        "respawn_goal_min_path_objects": _arg(saved_args, "respawn_goal_min_path_objects", 1),
                        "respawn_goal_path_band_scale": _arg(saved_args, "respawn_goal_path_band_scale", 1.0),
                    }
                )
            if "maze" not in env_id:
                env_kwargs["robot_cost_margin"] = _arg(saved_args, "robot_cost_margin", 0.0)
            if "maze" not in env_id:
                env_kwargs.update(
                    {
                        "fixed_agent_on_reset": _arg(saved_args, "fixed_agent_on_reset", False),
                        "fixed_goal_on_reset": _arg(saved_args, "fixed_goal_on_reset", False),
                    }
                )

            include_object_layout_obs = _optional_bool(_arg(saved_args, "include_object_layout_obs", ""))
            include_object_type_obs = _optional_bool(_arg(saved_args, "include_object_type_obs", ""))
            include_box_layout_obs = _optional_bool(_arg(saved_args, "include_box_layout_obs", ""))

            if include_object_layout_obs is not None and "maze" not in env_id:
                env_kwargs["include_object_layout_obs"] = include_object_layout_obs
            if include_object_type_obs is not None and "maze" not in env_id:
                env_kwargs["include_object_type_obs"] = include_object_type_obs
            if "maze" not in env_id:
                env_kwargs["layout_lidar_num_bins"] = _arg(saved_args, "layout_lidar_num_bins", 16)
            if "goal_grid" in env_id:
                grid_layout_name = _arg(saved_args, "grid_layout_name", "")
                eval_grid_layout_name = _arg(saved_args, "eval_grid_layout_name", "")
                if grid_layout_name:
                    env_kwargs["grid_layout_name"] = grid_layout_name
                if eval_grid_layout_name:
                    env_kwargs["eval_grid_layout_name"] = eval_grid_layout_name
            if is_point_car_goal_env_id(env_id):
                parking_mode = _arg(saved_args, "parking_mode", False)
                env_kwargs.update(
                    {
                        "ego_view": False if parking_mode else _arg(saved_args, "ego_view", True),
                        "parking_mode": parking_mode,
                        "parking_yaw_tolerance_degrees": _arg(
                            saved_args, "parking_yaw_tolerance_degrees", 20.0
                        ),
                    }
                )
            elif is_point_push_env_id(env_id):
                env_kwargs["ego_view"] = False
            if include_box_layout_obs is not None and "maze" in env_id and ("ant" in env_id or "humanoid" in env_id):
                env_kwargs["include_box_layout_obs"] = include_box_layout_obs

            env_params: dict[str, Any] = {}
            if _arg(saved_args, "use_env_config", True) or env_config_dir is not None or env_config_path is not None:
                env_params.update(
                    load_env_config(
                        config_env_id_for(_arg(saved_args, "env_id", env_id)),
                        config_dir=env_config_dir,
                        config_path=env_config_path,
                    )
                )
            env_params.update(env_kwargs)

        if "goal_grid" in env_id:
            env_params.setdefault("evaluation_mode", True)
        elif is_ant_humanoid_goal_env_id(env_id):
            env_params.setdefault("evaluation_mode", True)
        if is_point_push_env_id(env_id):
            env_params["ego_view"] = False
        if is_humanoid_env_id(env_id):
            env_params.setdefault(
                "humanoid_use_spring_gear",
                _arg(saved_args, "humanoid_use_spring_gear", False),
            )
        env = make_safenav_env(
            env_id,
            use_config=False,
            **env_params,
        )
        if is_point_push_env_id(env_id):
            if _arg(saved_args, "point_push_combined_goal", True):
                env = PointPushCombinedGoalObservationWrapper(env, env.goal_indices)
            else:
                env = ScalingCrlObservationWrapper(env, env.goal_indices)
        elif (
            is_point_car_goal_env_id(env_id)
            and _arg(saved_args, "ego_view", True)
            and not _arg(saved_args, "parking_mode", False)
        ):
            env = EgoGoalObservationWrapper(env, env.goal_indices, goal_lidar=_arg(saved_args, "goal_lidar", False))
        else:
            env = ScalingCrlObservationWrapper(env, env.goal_indices)

    if int(env.obs_dim) != int(saved_args.obs_dim) or int(env.observation_size) != int(
        saved_args.obs_dim + saved_args.raw_goal_dim
    ):
        raise ValueError(
            "Rebuilt env observation shape does not match the saved policy metadata: "
            f"saved obs_dim={saved_args.obs_dim}, saved raw_goal_dim={saved_args.raw_goal_dim}, "
            f"{env_id} obs_dim={env.obs_dim}, observation_size={env.observation_size}. "
            "Check saved env_params/env_config or pass the matching --env-id."
        )
    return env

def _infer_actor_input_size(actor_params: Any) -> int | None:
    params = actor_params.get("params", actor_params) if hasattr(actor_params, "get") else actor_params
    try:
        return int(params["Dense_0"]["kernel"].shape[0])
    except (KeyError, TypeError, AttributeError):
        return None


def _state_without_layout(obs: jnp.ndarray, saved_args: Args) -> jnp.ndarray:
    if not saved_args.ignore_layout_obs:
        return obs
    return jnp.concatenate(
        [
            obs[..., : saved_args.layout_start_idx],
            obs[..., saved_args.layout_end_idx : saved_args.obs_dim],
        ],
        axis=-1,
    )


def _actor_observation_from_state(state, saved_args: Args) -> jnp.ndarray:
    if not saved_args.ignore_layout_obs:
        return state.obs
    state_context = state.obs[..., : saved_args.obs_dim]
    goal = state.obs[
        ..., saved_args.obs_dim : saved_args.obs_dim + saved_args.raw_goal_dim
    ]
    return jnp.concatenate(
        [_state_without_layout(state_context, saved_args), goal],
        axis=-1,
    )


def _select_render_frames(states: list[Any], frame_stride: int, max_render_frames: int) -> list[Any]:
    stride = max(frame_stride, 1)
    selected = states[::stride]
    if states and selected[-1] is not states[-1]:
        selected.append(states[-1])
    if max_render_frames > 0 and len(selected) > max_render_frames:
        indices = np.linspace(0, len(selected) - 1, max_render_frames, dtype=np.int32)
        selected = [selected[int(i)] for i in indices]
    return selected


def _force_square_html(html_string: str, size: int) -> str:
    square_css = f"""
<style>
html, body {{
  margin: 0;
  padding: 0;
  width: {size}px;
  height: {size}px;
  overflow: hidden;
  background: #ffffff;
}}
body > div:first-child {{
  width: {size}px !important;
  height: {size}px !important;
}}
canvas {{
  width: {size}px !important;
  height: {size}px !important;
  display: block;
}}
</style>
"""
    viewer_init = "var viewer = new Viewer(domElement, system);"
    fixed_camera = """
      viewer.camera.follow = false;
      viewer.camera.followDistance = 20;
      const arenaBounds = {{
        minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity,
      }};
      const expandArenaBounds = (x, y, halfX, halfY) => {{
        arenaBounds.minX = Math.min(arenaBounds.minX, x - halfX);
        arenaBounds.maxX = Math.max(arenaBounds.maxX, x + halfX);
        arenaBounds.minY = Math.min(arenaBounds.minY, y - halfY);
        arenaBounds.maxY = Math.max(arenaBounds.maxY, y + halfY);
      }};
      const firstFramePositions = system.states?.x?.[0]?.pos ?? [];
      Object.entries(system.geoms ?? {{}}).forEach(([groupName, colliders]) => {{
        const isLinkedArenaGeom =
            /hazard|wall|obstacle|boundary|maze|pit|platform/i.test(groupName);
        colliders.forEach((collider) => {{
          if (collider.name === 'Plane') return;
          const size = collider.size ?? [0, 0];
          if (groupName === 'world') {{
            const pos = collider.pos ?? [0, 0];
            expandArenaBounds(pos[0], pos[1], size[0], size[1]);
          }} else if (isLinkedArenaGeom &&
                     firstFramePositions[collider.link_idx]) {{
            const linkPos = firstFramePositions[collider.link_idx];
            const localPos = collider.pos ?? [0, 0];
            expandArenaBounds(
                linkPos[0] + localPos[0], linkPos[1] + localPos[1],
                size[0], size[1]);
          }}
        }});
      }});
      const arenaCenterX = Number.isFinite(arenaBounds.minX) ?
          (arenaBounds.minX + arenaBounds.maxX) / 2 : 0;
      const arenaCenterY = Number.isFinite(arenaBounds.minY) ?
          (arenaBounds.minY + arenaBounds.maxY) / 2 : 0;
      viewer.camera.position.set(arenaCenterX + 10, arenaCenterY + 16, 4);
      viewer.controls.target.set(arenaCenterX, arenaCenterY, 0);
      viewer.controlTargetPos.copy(viewer.controls.target);
      viewer.controls.update();
      viewer.setDirty();
      viewer.gui.controllersRecursive?.().forEach((controller) => {
        if (controller.property === 'follow' ||
            controller.property === 'followDistance') {
          controller.updateDisplay();
        }
      });"""
    if viewer_init in html_string and "viewer.camera.follow = false;" not in html_string:
        html_string = html_string.replace(
            viewer_init, f"{viewer_init}{fixed_camera}", 1
        )
    if "</head>" in html_string:
        return html_string.replace("</head>", f"{square_css}</head>", 1)
    return square_css + html_string


def _patch_brax_json_render_bug() -> None:
    if not hasattr(brax_json, "jp"):
        brax_json.jp = jnp


def run_rollout(
    actor_params: Any,
    saved_args: Args,
    env_id: str,
    seed: int,
    steps: int,
    frame_stride: int,
    max_render_frames: int,
    square_size: int,
) -> str:
    env = _make_env(saved_args, env_id)
    saved_state_input_size = (
        saved_args.obs_dim - saved_args.layout_obs_dim
        if saved_args.ignore_layout_obs
        else saved_args.obs_dim
    )
    expected_actor_input_size = saved_state_input_size + saved_args.raw_goal_dim
    saved_actor_input_size = _infer_actor_input_size(actor_params)
    if saved_actor_input_size is not None and saved_actor_input_size != expected_actor_input_size:
        raise ValueError(
            "Saved actor params and saved args disagree on actor input size: "
            f"actor first layer expects {saved_actor_input_size}, "
            f"saved args imply {expected_actor_input_size} "
            f"(obs_dim={saved_args.obs_dim}, raw_goal_dim={saved_args.raw_goal_dim}). "
            "Use the final_policy/policy.pkl from the matching training run."
        )
    actor = Actor(
        action_size=env.action_size,
        network_width=saved_args.actor_network_width,
        network_depth=saved_args.actor_depth,
        use_relu=saved_args.use_relu,
    )
    @jax.jit
    def policy_step(env_state):
        actor_obs = _actor_observation_from_state(env_state, saved_args)
        means, _ = actor.apply(actor_params, actor_obs)
        actions = nn.tanh(means)
        return env.step(env_state, actions)

    state = jax.jit(env.reset)(jax.random.PRNGKey(seed))
    pipeline_states = [sync_mocap_pipeline_state_for_render(env, state.pipeline_state)]
    for _ in range(steps):
        state = policy_step(state)
        pipeline_states.append(sync_mocap_pipeline_state_for_render(env, state.pipeline_state))
        if bool(np.asarray(state.done)):
            break

    _patch_brax_json_render_bug()
    render_states = _select_render_frames(pipeline_states, frame_stride, max_render_frames)
    html_string = html.render(env.sys, render_states, height=square_size, colab=False)
    return _force_square_html(html_string, square_size)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load a saved scalable_safe_rl actor and save one deterministic renderable rollout as HTML.",
    )
    parser.add_argument("--policy-dir", required=True, help="Path to final_policy/ or a run directory.")
    parser.add_argument("--output", default="", help="Output HTML path. Defaults to <policy-dir>/eval_rollout.html.")
    parser.add_argument("--env-id", default="", help="Renderable env override. Defaults to saved env_id without _headless.")
    parser.add_argument("--seed", type=int, default=-1, help="Reset seed. Use -1 for the saved training seed.")
    parser.add_argument("--steps", type=int, default=-1, help="Episode steps. Use -1 for saved episode_length.")
    parser.add_argument("--square-size", type=int, default=720, help="Square HTML/canvas size in pixels.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Render every Nth rollout frame.")
    parser.add_argument(
        "--max-render-frames",
        type=int,
        default=300,
        help="Cap rendered frames after striding. Use 0 to render all frames.",
    )
    return parser


def main() -> None:
    cli_args = build_arg_parser().parse_args()
    actor_params, saved_args, resolved_policy_dir = _load_policy(cli_args.policy_dir)
    env_id = _renderable_env_id(saved_args, cli_args.env_id)
    seed = saved_args.seed if cli_args.seed < 0 else cli_args.seed
    steps = saved_args.episode_length if cli_args.steps < 0 else cli_args.steps
    html_string = run_rollout(
        actor_params=actor_params,
        saved_args=saved_args,
        env_id=env_id,
        seed=seed,
        steps=steps,
        frame_stride=cli_args.frame_stride,
        max_render_frames=cli_args.max_render_frames,
        square_size=cli_args.square_size,
    )

    output = Path(cli_args.output) if cli_args.output else resolved_policy_dir / "eval_rollout.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_string, encoding="utf-8")
    print(f"Saved deterministic rollout for {env_id} to {output}", flush=True)


if __name__ == "__main__":
    main()
