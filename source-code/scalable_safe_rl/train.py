import os
import time
import pickle
import random
import argparse
import sys
import json
import math

try:
    import yaml
except ImportError:
    yaml = None


def _early_arg_value(argv, names):
    for index, arg in enumerate(argv):
        for name in names:
            if arg == name and index + 1 < len(argv):
                return argv[index + 1]
            if arg.startswith(f"{name}="):
                return arg.split("=", 1)[1]
    return None


def _early_yaml_value(path, name):
    if not path or yaml is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return None
    if not isinstance(data, dict):
        return None
    return data.get(name)


def _early_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _append_xla_flag(flag):
    current = os.environ.get("XLA_FLAGS", "")
    flags = current.split()
    if flag not in flags:
        flags.append(flag)
        os.environ["XLA_FLAGS"] = " ".join(flags).strip()


def _configure_deterministic_runtime_from_argv():
    config_path = _early_arg_value(sys.argv, ("--config",))
    config_value = _early_yaml_value(config_path, "deterministic_runtime")
    cli_value = _early_arg_value(sys.argv, ("--deterministic-runtime", "--deterministic_runtime"))
    enabled = _early_bool(cli_value if cli_value is not None else config_value, default=True)
    if not enabled:
        return
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _append_xla_flag("--xla_gpu_deterministic_ops")


def _configure_visible_gpus_from_argv():
    config_path = _early_arg_value(sys.argv, ("--config",))
    config_gpu_device = _early_yaml_value(config_path, "gpu_device")
    cli_gpu_device = _early_arg_value(sys.argv, ("--gpu-device", "--gpu_device"))
    gpu_device = cli_gpu_device if cli_gpu_device is not None else config_gpu_device
    if gpu_device is None:
        return
    gpu_device = str(gpu_device).strip()
    if gpu_device == "":
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device


_configure_deterministic_runtime_from_argv()
_configure_visible_gpus_from_argv()

import jax
import flax
import optax
import numpy as np
import flax.linen as nn
import jax.numpy as jnp

from pathlib import Path
from brax import envs
from brax.envs import base
from brax.envs.wrappers import training as brax_training
from etils import epath
from dataclasses import dataclass, fields
from typing import NamedTuple, Any
from flax.training.train_state import TrainState
from flax.linen.initializers import variance_scaling
from brax.io import html
from safenav_jax.visualization_rollout import sync_mocap_pipeline_state_for_render
from experiment_artifacts import (
    append_metrics_json_line,
    resolved_config_dict,
    unique_run_dir,
    write_resolved_config,
)

try:
    import tyro
except ImportError:
    tyro = None

try:
    import wandb
    import wandb_osh
    from wandb_osh.hooks import TriggerWandbSyncHook
except ImportError:
    wandb = None
    wandb_osh = None
    TriggerWandbSyncHook = None

try:
    from .evaluator import CrlEvaluator
    from .buffer import TrajectoryUniformSamplingQueue
except ImportError:
    from evaluator import CrlEvaluator
    from buffer import TrajectoryUniformSamplingQueue


SAFENAV_ENV_IDS = frozenset(
    {
        "point_goal",
        "point_goal_headless",
        "car_goal",
        "car_goal_headless",
        "point_push",
        "point_push_headless",
        "ant_goal",
        "ant_goal_grid",
        "ant_goal_grid_headless",
        "ant_goal_headless",
        "humanoid_goal",
        "humanoid_goal_grid",
        "humanoid_goal_grid_headless",
        "humanoid_goal_headless",
    }
)

SCALING_ENV_IDS = frozenset(
    {
        "crl_ant_maze",
        "crl_humanoid_maze",
        "crl_humanoid_maze_nowall",
        "scaling_ant_goal",
        "scaling_humanoid_goal",
    }
)


def is_safenav_env_id(env_id: str) -> bool:
    try:
        from safenav_jax.envs import is_supported_env_id

        return is_supported_env_id(env_id)
    except Exception:
        return env_id in SAFENAV_ENV_IDS


def is_scaling_env_id(env_id: str) -> bool:
    try:
        import scaling_envs

        return scaling_envs.is_supported_env_id(env_id)
    except Exception:
        return env_id in SCALING_ENV_IDS


def is_point_car_goal_env_id(env_id: str) -> bool:
    return env_id in {
        "point_goal",
        "point_goal_headless",
        "car_goal",
        "car_goal_headless",
    }


def is_point_push_env_id(env_id: str) -> bool:
    return env_id in {"point_push", "point_push_headless"}


def is_ant_humanoid_goal_env_id(env_id: str) -> bool:
    return env_id in {
        "ant_goal",
        "ant_goal_headless",
        "humanoid_goal",
        "humanoid_goal_headless",
    }


def is_humanoid_env_id(env_id: str) -> bool:
    return "humanoid" in env_id and (
        is_safenav_env_id(env_id) or is_scaling_env_id(env_id)
    )


def safenav_env_id_hint() -> str:
    return (
        f"{sorted(SAFENAV_ENV_IDS)} and scaling-only IDs "
        f"{sorted(SCALING_ENV_IDS)}."
    )


def _resolve_eval_env_request(env_id: str, eval_env_id: str) -> tuple[str, bool]:
    """Returns the literal eval ID and whether eval-only env semantics apply."""
    explicit_eval_env_id = eval_env_id.strip()
    if explicit_eval_env_id:
        return explicit_eval_env_id, True
    return env_id, False


SCALING_CRL_LOSS_TYPE = "scaling_crl"
SCALING_CRL_SURVIVE_LOSS_TYPE = "scaling_crl_survive"
SUPPORTED_CRITIC_LOSS_TYPES = (
    SCALING_CRL_LOSS_TYPE,
    SCALING_CRL_SURVIVE_LOSS_TYPE,
)
ENTROPY_MODE_LEARNED = "learned"
ENTROPY_MODE_FIXED = "fixed"
ENTROPY_MODE_NONE = "none"


def _reduce_infonce_rows(per_anchor_loss, future_valid, survival_mass=None):
    """Averages valid InfoNCE rows, optionally using unnormalized mass weights."""
    valid_weight = future_valid.astype(per_anchor_loss.dtype)
    weighted_loss = per_anchor_loss
    if survival_mass is not None:
        weighted_loss = weighted_loss * survival_mass.astype(per_anchor_loss.dtype)
    return jnp.sum(weighted_loss * valid_weight) / jnp.maximum(
        jnp.sum(valid_weight),
        1.0,
    )


class EgoGoalObservationWrapper(base.Wrapper):
    """Drops global XY from point/car goal state and converts task goals to ego coordinates."""

    def __init__(self, env: base.Env, goal_indices: jnp.ndarray, goal_lidar: bool = False):
        super().__init__(env)
        goal_indices = np.asarray(goal_indices, dtype=np.int32)
        observation_indices = np.arange(env.observation_size, dtype=np.int32)
        non_goal_indices = observation_indices[~np.isin(observation_indices, goal_indices)]
        robot_obs_dim = int(getattr(env, "_robot_obs_dim", goal_indices[0]))
        if robot_obs_dim < 2:
            raise ValueError("EgoGoalObservationWrapper requires robot XY at the beginning of the observation.")
        if len(goal_indices) != 2:
            raise ValueError("EgoGoalObservationWrapper requires a 2D goal.")
        self._context_indices = jnp.asarray(non_goal_indices[non_goal_indices >= 2])
        self._goal_indices = jnp.asarray(goal_indices)
        self.robot_obs_dim = robot_obs_dim - 2
        self.layout_start_idx = self.robot_obs_dim
        self.layout_obs_dim = int(np.sum(non_goal_indices >= robot_obs_dim))
        self.layout_end_idx = self.layout_start_idx + self.layout_obs_dim
        self.obs_dim = int(self._context_indices.shape[0])
        self.goal_lidar = bool(goal_lidar)
        self.relabel_goal_dim = 2
        self.goal_lidar_num_bins = int(getattr(env, "_layout_lidar_num_bins", 16))
        self.goal_lidar_max_dist = float(getattr(env, "_layout_lidar_max_dist", 1.0))
        self.raw_goal_dim = self.goal_lidar_num_bins if self.goal_lidar else 2
        self.goal_start_idx = self.obs_dim
        self.goal_end_idx = self.obs_dim + self.raw_goal_dim
        self.goal_indices = jnp.arange(self.goal_start_idx, self.goal_end_idx)
        self.uses_ego_goal_relabel = True
        self.uses_goal_lidar = self.goal_lidar

    @property
    def observation_size(self) -> int:
        return self.goal_end_idx

    def _agent_yaw(self, state: base.State) -> jnp.ndarray:
        if "agent_yaw" in state.metrics:
            return state.metrics["agent_yaw"]
        return jnp.zeros(state.obs.shape[:-1], dtype=state.obs.dtype)

    def _ego_xy(self, raw_obs: jnp.ndarray, global_xy: jnp.ndarray, yaw: jnp.ndarray) -> jnp.ndarray:
        rel_xy = global_xy - raw_obs[..., :2]
        cos_yaw = jnp.cos(yaw)
        sin_yaw = jnp.sin(yaw)
        local_x = cos_yaw * rel_xy[..., 0] + sin_yaw * rel_xy[..., 1]
        local_y = -sin_yaw * rel_xy[..., 0] + cos_yaw * rel_xy[..., 1]
        return jnp.stack([local_x, local_y], axis=-1)

    def _ego_xy_to_lidar(self, ego_xy: jnp.ndarray) -> jnp.ndarray:
        dist = jnp.linalg.norm(ego_xy, axis=-1)
        max_dist = jnp.asarray(self.goal_lidar_max_dist, dtype=ego_xy.dtype)
        signal = jnp.clip(1.0 - dist / jnp.maximum(max_dist, 1e-8), 0.0, 1.0)
        angle = jnp.mod(jnp.arctan2(ego_xy[..., 1], ego_xy[..., 0]), 2.0 * jnp.pi)
        bin_idx = jnp.floor(angle / (2.0 * jnp.pi) * self.goal_lidar_num_bins).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, self.goal_lidar_num_bins - 1)
        return jax.nn.one_hot(bin_idx, self.goal_lidar_num_bins, dtype=ego_xy.dtype) * signal[..., None]

    def _project_state(self, state: base.State) -> jnp.ndarray:
        state_obs = jnp.take(state.obs, self._context_indices, axis=-1)
        goal_global = jnp.take(state.obs, self._goal_indices, axis=-1)
        goal_ego = self._ego_xy(state.obs, goal_global, self._agent_yaw(state))
        if self.goal_lidar:
            goal_ego = self._ego_xy_to_lidar(goal_ego)
        return jnp.concatenate([state_obs, goal_ego], axis=-1)

    def reset(self, rng: jax.Array) -> base.State:
        state = self.env.reset(rng)
        return state.replace(obs=self._project_state(state))

    def step(self, state: base.State, action: jax.Array) -> base.State:
        state = self.env.step(state, action)
        return state.replace(obs=self._project_state(state))


class PointPushCombinedGoalObservationWrapper(base.Wrapper):
    """Uses task/cube goals for acting and future cube/robot goals for relabeling."""

    def __init__(self, env: base.Env, goal_indices: jnp.ndarray):
        super().__init__(env)
        goal_indices = np.asarray(goal_indices, dtype=np.int32)
        if len(goal_indices) != 2:
            raise ValueError("Point push requires a 2D task goal.")
        observation_indices = np.arange(env.observation_size, dtype=np.int32)
        non_goal_indices = observation_indices[~np.isin(observation_indices, goal_indices)]
        robot_obs_dim = int(getattr(env, "_robot_obs_dim", goal_indices[0]))
        if robot_obs_dim < 4:
            raise ValueError("Point push requires leading cube XY and robot XY in its state.")

        self._state_indices = jnp.asarray(non_goal_indices)
        self._task_goal_indices = jnp.asarray(goal_indices)
        self.robot_obs_dim = robot_obs_dim
        self.layout_start_idx = robot_obs_dim
        self.layout_obs_dim = int(np.sum(non_goal_indices >= robot_obs_dim))
        self.layout_end_idx = self.layout_start_idx + self.layout_obs_dim
        self.obs_dim = int(non_goal_indices.shape[0])
        self.raw_goal_dim = 4
        self.relabel_goal_dim = 4
        self.goal_start_idx = 0
        self.goal_end_idx = 4
        self.task_goal_start_idx = self.obs_dim
        self.task_goal_end_idx = self.obs_dim + self.raw_goal_dim
        self.goal_indices = jnp.arange(self.task_goal_start_idx, self.task_goal_end_idx)
        self.uses_ego_goal_relabel = False
        self.uses_goal_lidar = False
        self.goal_lidar_max_dist = 0.0

    @property
    def observation_size(self) -> int:
        return self.task_goal_end_idx

    def _project_state(self, state: base.State) -> base.State:
        state_obs = jnp.take(state.obs, self._state_indices, axis=-1)
        task_goal_xy = jnp.take(state.obs, self._task_goal_indices, axis=-1)
        current_cube_xy = state.obs[..., :2]
        policy_goal = jnp.concatenate([task_goal_xy, current_cube_xy], axis=-1)
        return state.replace(obs=jnp.concatenate([state_obs, policy_goal], axis=-1))

    def reset(self, rng: jax.Array) -> base.State:
        return self._project_state(self.env.reset(rng))

    def step(self, state: base.State, action: jax.Array) -> base.State:
        return self._project_state(self.env.step(state, action))


class ScalingCrlObservationWrapper(base.Wrapper):
    """Projects SafeNav observations to CRL order: state context then task goal.

    If the underlying env includes object/box layout observations, they remain in
    the state context after the robot state. The future-goal relabeler still uses
    the leading achieved-position coordinates, matching scaling-crl conventions.
    """

    def __init__(self, env: base.Env, goal_indices: jnp.ndarray):
        super().__init__(env)
        goal_indices = np.asarray(goal_indices, dtype=np.int32)
        observation_indices = np.arange(env.observation_size, dtype=np.int32)
        non_goal_indices = observation_indices[~np.isin(observation_indices, goal_indices)]
        robot_obs_dim = int(getattr(env, "_robot_obs_dim", goal_indices[0]))
        env_name = env.__class__.__name__.lower()
        raw_goal_dim = len(goal_indices)
        if "humanoid" in env_name:
            expected_state_dim = 268
        elif "ant" in env_name:
            expected_state_dim = 29
        else:
            expected_state_dim = robot_obs_dim
        self._source_state_dim = robot_obs_dim
        self._expected_state_dim = expected_state_dim
        self._robot_indices = jnp.arange(min(robot_obs_dim, expected_state_dim))
        self._layout_indices = jnp.asarray(non_goal_indices[non_goal_indices >= robot_obs_dim])
        self._goal_indices = jnp.asarray(goal_indices)
        self.robot_obs_dim = expected_state_dim
        self.layout_start_idx = expected_state_dim
        self.layout_obs_dim = int(self._layout_indices.shape[0])
        self.layout_end_idx = expected_state_dim + self.layout_obs_dim
        self.obs_dim = self.layout_end_idx
        self.raw_goal_dim = raw_goal_dim
        achieved_goal_indices = np.asarray(
            getattr(env, "scaling_crl_goal_indices", np.arange(raw_goal_dim)),
            dtype=np.int32,
        )
        self.goal_start_idx = int(achieved_goal_indices[0])
        self.goal_end_idx = int(achieved_goal_indices[-1]) + 1
        self.task_goal_start_idx = self.obs_dim
        self.task_goal_end_idx = self.obs_dim + raw_goal_dim
        self.goal_indices = jnp.arange(self.task_goal_start_idx, self.task_goal_end_idx)

    @property
    def observation_size(self) -> int:
        return self.task_goal_end_idx

    def _project_state(self, state: base.State) -> jnp.ndarray:
        state_obs = jnp.take(state.obs, self._robot_indices, axis=-1)
        if self._expected_state_dim > self._source_state_dim:
            pad_shape = state_obs.shape[:-1] + (self._expected_state_dim - self._source_state_dim,)
            state_obs = jnp.concatenate([state_obs, jnp.zeros(pad_shape, dtype=state_obs.dtype)], axis=-1)
        layout_obs = jnp.take(state.obs, self._layout_indices, axis=-1)
        state_obs = jnp.concatenate([state_obs, layout_obs], axis=-1)
        goal_obs = jnp.take(state.obs, self._goal_indices, axis=-1)
        return jnp.concatenate([state_obs, goal_obs], axis=-1)

    def reset(self, rng: jax.Array) -> base.State:
        state = self.env.reset(rng)
        return state.replace(obs=self._project_state(state))

    def step(self, state: base.State, action: jax.Array) -> base.State:
        state = self.env.step(state, action)
        return state.replace(obs=self._project_state(state))


class SafeNavAutoResetWrapper(base.Wrapper):
    """Auto-reset wrapper that also resets SafeNav task info used by env.step."""

    _TASK_INFO_KEYS = (
        "goal_xy",
        "goal_yaw",
        "goal_positions",
        "goal_index",
        "hazard_positions",
        "respawn_rng",
        "hazards_xy",
        "obstacles_xy",
        "gremlin_centers_xy",
        "gremlins_xy",
        "episode_cost",
        "cost_streak",
        "goal_reached",
    )

    def __init__(
        self,
        env: base.Env,
    ):
        super().__init__(env)

    def _ensure_episode_seed(self, state: base.State) -> base.State:
        if "seed" in state.info:
            return state
        state.info["seed"] = jnp.zeros_like(state.done, dtype=jnp.int32)
        state.info["_synthetic_episode_seed"] = jnp.ones_like(state.done, dtype=bool)
        return state

    def reset(self, rng: jax.Array) -> base.State:
        state = self.env.reset(rng)
        state = self._ensure_episode_seed(state)
        state.info["rng"] = state.info.get("rng", rng)
        state.info["truncation"] = jnp.zeros_like(state.done)
        state.info["raw_next_observation"] = state.obs
        return state

    def step(self, state: base.State, action: jax.Array) -> base.State:
        if "steps" in state.info:
            steps = jnp.where(state.done, jnp.zeros_like(state.info["steps"]), state.info["steps"])
            state.info.update(steps=steps)
        previous_steps = state.info.get("steps", jnp.zeros_like(state.done))
        previous_seed = state.info.get("seed", jnp.zeros_like(state.done, dtype=jnp.int32))
        state.info["rng"] = state.info.get("rng", jax.random.PRNGKey(0))
        state.info["truncation"] = jnp.zeros_like(state.done)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)
        if "_synthetic_episode_seed" in state.info:
            state.info["seed"] = previous_seed + jnp.where(
                previous_steps,
                jnp.zeros_like(previous_seed),
                jnp.ones_like(previous_seed),
            )
        state.info["rng"] = state.info.get("rng", jax.random.PRNGKey(0))
        state.info["truncation"] = state.info.get("truncation", jnp.zeros_like(state.done))
        state.info["raw_next_observation"] = state.obs

        def where_done(reset_value, current_value):
            done = state.done
            if done.shape and done.shape[0] != current_value.shape[0]:
                return current_value
            if done.shape:
                done = jnp.reshape(done, [current_value.shape[0]] + [1] * (len(current_value.shape) - 1))
            return jnp.where(done, reset_value, current_value)

        def reset_done_envs(done_state):
            reset_state = self.env.reset(done_state.info["rng"])
            reset_state = self._ensure_episode_seed(reset_state)
            reset_state.info["rng"] = reset_state.info.get("rng", done_state.info["rng"])
            reset_state.info["truncation"] = jnp.zeros_like(reset_state.done)
            reset_state.info["raw_next_observation"] = reset_state.obs
            pipeline_state = jax.tree_util.tree_map(where_done, reset_state.pipeline_state, done_state.pipeline_state)
            obs = jax.tree_util.tree_map(where_done, reset_state.obs, done_state.obs)
            info = dict(done_state.info)
            reset_keys = set(self._TASK_INFO_KEYS) | {"rng"}
            for key in reset_keys:
                if key in done_state.info and key in reset_state.info:
                    info[key] = jax.tree_util.tree_map(where_done, reset_state.info[key], done_state.info[key])
            return done_state.replace(pipeline_state=pipeline_state, obs=obs, info=info)

        return jax.lax.cond(
            jnp.any(state.done.astype(bool)),
            reset_done_envs,
            lambda done_state: done_state,
            state,
        )


def wrap_safenav_training_env(
    env: base.Env,
    episode_length: int,
) -> base.Wrapper:
    env = brax_training.VmapWrapper(env)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat=1)
    return SafeNavAutoResetWrapper(env)

@dataclass
class Args:
    exp_name: str = "train"
    seed: int = 1000
    gpu_device: str = ""
    deterministic_runtime: bool = True
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "scalable_safe_rl"
    wandb_entity: str = ""
    wandb_mode: str = 'offline'
    wandb_dir: str = '.'
    wandb_group: str = '.'
    results_root: str = "results"
    run_name: str = ""
    seed_output_dir: str = ""
    capture_vis: bool = False
    vis_length: int = 1000
    checkpoint: bool = True
    save_final_policy: bool = True
    checkpoint_every_epochs: int = 10
    print_every_epochs: int = 1
    print_args: bool = False
    dry_run: bool = False

    #environment specific arguments
    env_id: str = "point_goal_headless" # SafeNav env id, e.g. "ant_goal_headless"
    episode_length: int = 1000
    env_config_dir: str = ""
    env_config_path: str = ""
    env_params: Any = None
    eval_env_params: Any = None
    relocate_objects_on_reset: bool = True
    fixed_object_layout_seed: int = 0
    layout_seed_from_seed: bool = False
    different_object_layout_per_env: bool = False
    goal_respawn_on_success: bool = True
    respawn_goal_require_path_objects: bool = False
    respawn_goal_min_path_objects: int = 1
    respawn_goal_path_band_scale: float = 1.0
    terminate_on_cost: bool = False
    eval_terminate_on_cost: bool = False
    cost_limit_max: float = 25.0
    robot_cost_margin: float = 0.0
    object_boundary: bool = False
    fixed_agent_on_reset: bool = False
    fixed_goal_on_reset: bool = False
    include_object_layout_obs: str = ""  # optional bool override for goal envs
    include_object_type_obs: str = ""  # optional bool override for typed object layout entries in goal envs
    include_box_layout_obs: str = ""  # optional bool override for ant/humanoid maze envs
    layout_lidar_num_bins: int = 16
    grid_layout_name: str = ""
    eval_grid_layout_name: str = ""
    ego_view: bool = True
    parking_mode: bool = False
    parking_yaw_tolerance_degrees: float = 20.0
    humanoid_use_spring_gear: bool = False
    # to be filled in runtime
    obs_dim: int = 0
    goal_start_idx: int = 0
    goal_end_idx: int = 0
    raw_goal_dim: int = 0
    layout_start_idx: int = 0
    layout_end_idx: int = 0
    layout_obs_dim: int = 0
    layout_requested_num_hazards: int = 0
    layout_num_hazards: int = 0
    layout_num_obstacles: int = 0
    layout_num_gremlins: int = 0
    layout_num_boxes: int = 0
    layout_has_type_obs: bool = False
    ego_goal_relabel: bool = False
    goal_lidar: bool = False
    goal_lidar_max_dist: float = 0.0
    relabel_goal_dim: int = 0
    point_push_combined_goal: bool = True

    # Algorithm specific arguments
    total_env_steps: int = 100000000 # 50000000
    num_epochs: int = 100 # 50
    num_envs: int = 512
    eval_env_id: str = ""
    use_env_config: bool = True
    num_eval_envs: int = 128
    eval_every_epochs: int = 1
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    max_grad_norm: float = 10.0
    skip_nonfinite_updates: bool = True
    actor_mean_clip: float = 10.0
    batch_size: int = 256
    gamma: float = 0.99
    logsumexp_penalty_coeff: float = 0.1
    critic_loss_type: str = SCALING_CRL_SURVIVE_LOSS_TYPE
    ignore_layout_obs: bool = False

    max_replay_size: int = 10000
    min_replay_size: int = 1000
    
    unroll_length: int  = 62
    critic_network_width: int = 256
    actor_network_width: int = 256
    embedding_dim: int = 64
    actor_depth: int = 4
    critic_depth: int = 4
    z_encoder_depth: int = -1  # -1 inherits the resolved critic_depth
    
    num_episodes_per_env: int = 1 #recommended to keep at 1
    training_steps_multiplier: int = 1 #recommended to keep at 1
    use_all_batches: int = 0 # recommended to keep at 0
    num_sgd_batches_per_training_step: int = 800
    
    eval_actor: int = 0 # recommended to keep at 0
    # if 0, use deterministic actor for evaluation
    # if 1, use stochastic actor for evaluation
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    expl_actor: int = 1 # recommended to keep at 1
    # if 0, use deterministic actor for exploration/collecting data
    # if 1, use stochastic actor for exploration/collecting data
    # if 2, sample two actions and take the one with the higher Q value
    # if K >= 2, sample K actions and take the one with the highest Q value
    
    entropy_param: float = 0.5
    entropy_mode: str = ENTROPY_MODE_LEARNED
    fixed_alpha: float = 0.05
    disable_entropy: int = 0
    use_relu: int = 0
    num_render: int = 1
    save_buffer: int = 0
    
    # to be filled in runtime
    env_steps_per_actor_step : int = 0
    """number of env steps per actor step (computed in runtime)"""
    num_prefill_env_steps : int = 0
    """number of env steps to fill the buffer before starting training (computed in runtime)"""
    num_prefill_actor_steps : int = 0
    """number of actor steps to fill the buffer before starting training (computed in runtime)"""
    num_training_steps_per_epoch : int = 0
    """the number of training steps per epoch(computed in runtime)"""


RUNTIME_ENV_METADATA_FIELDS = (
    "obs_dim",
    "goal_start_idx",
    "goal_end_idx",
    "raw_goal_dim",
    "layout_start_idx",
    "layout_end_idx",
    "layout_obs_dim",
    "layout_requested_num_hazards",
    "layout_num_hazards",
    "layout_num_obstacles",
    "layout_num_gremlins",
    "layout_num_boxes",
    "layout_has_type_obs",
    "ego_goal_relabel",
    "goal_lidar",
    "goal_lidar_max_dist",
    "relabel_goal_dim",
)


def _snapshot_runtime_env_metadata(args: Args) -> dict[str, Any]:
    return {name: getattr(args, name) for name in RUNTIME_ENV_METADATA_FIELDS}


def _restore_runtime_env_metadata(args: Args, metadata: dict[str, Any]) -> None:
    for name in RUNTIME_ENV_METADATA_FIELDS:
        setattr(args, name, metadata[name])

lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
bias_init = nn.initializers.zeros
def residual_block(x, width, normalize, activation):
    identity = x
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    x = x + identity
    return x

def depth_trunk(x, width, network_depth, normalize, activation):
    x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
    x = normalize(x)
    x = activation(x)
    if network_depth < 4:
        identity = x
        for _ in range(network_depth):
            x = nn.Dense(width, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
            x = normalize(x)
            x = activation(x)
        return x + identity if network_depth > 0 else x
    for _ in range(network_depth // 4):
        x = residual_block(x, width, normalize, activation)
    return x

class SA_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    use_relu: int = 0
    output_dim: int = 64
    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish
            
        x = jnp.concatenate([s, a], axis=-1)
        x = depth_trunk(x, self.network_width, self.network_depth, normalize, activation)
        #Final layer
        x = nn.Dense(self.output_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x


class Z_encoder(nn.Module):
    """State-action encoder for a scalar log-normalizer."""

    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    use_relu: int = 0

    @nn.compact
    def __call__(self, s: jnp.ndarray, a: jnp.ndarray):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x

        activation = nn.relu if self.use_relu else nn.swish
        x = jnp.concatenate([s, a], axis=-1)
        x = depth_trunk(
            x,
            self.network_width,
            self.network_depth,
            normalize,
            activation,
        )
        return nn.Dense(1, kernel_init=lecun_unfirom, bias_init=bias_init)(x)


class G_encoder(nn.Module):
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    use_relu: int = 0
    output_dim: int = 64
    @nn.compact
    def __call__(self, g: jnp.ndarray):

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros

        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
        
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish
        
        x = g
        x = depth_trunk(x, self.network_width, self.network_depth, normalize, activation)
        #Final layer
        x = nn.Dense(self.output_dim, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        return x
  
class Actor(nn.Module):
    action_size: int
    norm_type = "layer_norm"
    network_width: int = 1024
    network_depth: int = 4
    use_relu: int = 0
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    @nn.compact
    def __call__(self, x):
        if self.norm_type == "layer_norm":
            normalize = lambda x: nn.LayerNorm()(x)
        else:
            normalize = lambda x: x
            
        if self.use_relu:
            activation = nn.relu
        else:
            activation = nn.swish

        lecun_unfirom = variance_scaling(1/3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
    
        x = depth_trunk(x, self.network_width, self.network_depth, normalize, activation)
        #Final layer
        mean = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        log_std = nn.Dense(self.action_size, kernel_init=lecun_unfirom, bias_init=bias_init)(x)
        
        log_std = nn.tanh(log_std)
        log_std = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner"""
    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    actor_state: TrainState
    critic_state: TrainState
    z_state: Any
    alpha_state: TrainState
    target_actor_params: Any = None
    target_critic_params: Any = None

class Transition(NamedTuple):
    """Container for a transition"""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    extras: jnp.ndarray = ()

def _str_to_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")

def _optional_bool_arg(value: str) -> bool | None:
    if value == "":
        return None
    return _str_to_bool(value)

def _coerce_config_value(name: str, value: Any, target_type: Any) -> Any:
    if target_type is bool:
        return _str_to_bool(value)
    if target_type in (int, float, str):
        return target_type(value)
    return value

def _load_yaml_config(path: str) -> dict[str, Any]:
    if yaml is None:
        raise ImportError("Reading --config requires PyYAML. Install pyyaml or pass hyperparameters via CLI.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path!r} must contain a YAML mapping.")
    valid_fields = {field.name: field for field in fields(Args)}
    unknown = sorted(set(data) - set(valid_fields))
    if unknown:
        raise ValueError(f"Unknown config keys in {path!r}: {unknown}")
    return {
        name: _coerce_config_value(name, value, valid_fields[name].type)
        for name, value in data.items()
    }

def _parse_args() -> tuple[Args, set[str]]:
    """Parses Args with optional YAML defaults and CLI overrides.

    Usage:
      python -m scalable_safe_rl.train --config scalable_safe_rl/config.yaml --seed 1

    If tyro is installed, it is used for the final CLI parsing.  Otherwise this
    lightweight argparse path supports the same `--field value` style for all
    dataclass fields.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default=None, help="YAML file with Args fields as keys.")
    pre_args, remaining = pre_parser.parse_known_args()

    defaults = Args()
    config_overrides = set()
    if pre_args.config is not None:
        config_values = _load_yaml_config(pre_args.config)
        config_overrides = set(config_values)
        for name, value in config_values.items():
            setattr(defaults, name, value)

    if tyro is not None:
        try:
            return tyro.cli(Args, default=defaults, args=remaining), config_overrides
        except TypeError:
            return tyro.cli(Args, default=defaults), config_overrides

    parser = argparse.ArgumentParser(parents=[pre_parser])
    for field in fields(Args):
        default = getattr(defaults, field.name)
        arg_name = f"--{field.name.replace('_', '-')}"
        alias_name = f"--{field.name}"
        kwargs = {"default": default, "dest": field.name}
        if field.type is bool:
            kwargs.update({"type": _str_to_bool, "nargs": "?", "const": True})
        else:
            kwargs["type"] = field.type if field.type in (int, float, str) else type(default)
        parser.add_argument(arg_name, alias_name, **kwargs)
    parsed_args = parser.parse_args(remaining, namespace=defaults)
    if hasattr(parsed_args, "config"):
        delattr(parsed_args, "config")
    return parsed_args, config_overrides

def _cli_override_names(argv: list[str] | None = None) -> set[str]:
    argv = sys.argv[1:] if argv is None else argv
    names = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0].replace("-", "_")
        if name != "config":
            names.add(name)
    return names

def _explicit_override_names(config_overrides: set[str], cli_overrides: set[str]) -> set[str]:
    return set(config_overrides) | set(cli_overrides)

def _apply_mode_defaults(args: Args, explicit_overrides: set[str]) -> None:
    if args.disable_entropy and "entropy_mode" not in explicit_overrides:
        args.entropy_mode = ENTROPY_MODE_NONE
    if args.critic_loss_type in SUPPORTED_CRITIC_LOSS_TYPES:
        scaling_crl_defaults = {
            "batch_size": 512,
            "actor_depth": 8,
            "critic_depth": 8,
        }
        for name, value in scaling_crl_defaults.items():
            if name not in explicit_overrides:
                setattr(args, name, value)
    if args.z_encoder_depth < -1:
        raise ValueError("z_encoder_depth must be at least -1.")
    if args.z_encoder_depth == -1:
        args.z_encoder_depth = args.critic_depth

def _validate_args(args: Args) -> None:
    if args.critic_loss_type not in SUPPORTED_CRITIC_LOSS_TYPES:
        raise ValueError(
            f"Unknown critic_loss_type={args.critic_loss_type!r}. "
            f"Expected one of {SUPPORTED_CRITIC_LOSS_TYPES!r}."
        )
    if not (0.0 < args.gamma <= 1.0):
        raise ValueError("gamma must satisfy 0 < gamma <= 1.")
    if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE and args.gamma >= 1.0:
        raise ValueError("scaling_crl_survive requires gamma < 1 for geometric horizon sampling.")
    allowed_entropy_modes = (
        ENTROPY_MODE_LEARNED,
        ENTROPY_MODE_FIXED,
        ENTROPY_MODE_NONE,
    )
    if args.entropy_mode not in allowed_entropy_modes:
        raise ValueError(
            f"Unknown entropy_mode={args.entropy_mode!r}. "
            f"Expected one of {allowed_entropy_modes!r}."
        )
    if args.fixed_alpha < 0.0:
        raise ValueError("fixed_alpha must be non-negative.")
    if args.actor_depth < 0:
        raise ValueError("actor_depth must be non-negative.")
    if args.critic_depth < 0:
        raise ValueError("critic_depth must be non-negative.")
    if args.z_encoder_depth < 0:
        raise ValueError("z_encoder_depth must be non-negative after default resolution.")
    if args.embedding_dim < 1:
        raise ValueError("embedding_dim must be at least 1.")
    if args.episode_length < 2:
        raise ValueError("episode_length must be at least 2.")
    if args.cost_limit_max <= 0.0:
        raise ValueError("cost_limit_max must be positive.")
    if args.robot_cost_margin < 0.0:
        raise ValueError("robot_cost_margin must be non-negative.")
    if args.unroll_length < 1:
        raise ValueError("unroll_length must be at least 1.")
    if args.num_envs < 1 or args.num_eval_envs < 1:
        raise ValueError("num_envs and num_eval_envs must be at least 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if args.num_episodes_per_env < 1:
        raise ValueError("num_episodes_per_env must be at least 1.")
    if args.max_grad_norm < 0.0:
        raise ValueError("max_grad_norm must be non-negative. Use 0 to disable gradient clipping.")
    if args.actor_mean_clip < 0.0:
        raise ValueError("actor_mean_clip must be non-negative. Use 0 to disable actor mean clipping.")
    if args.min_replay_size < args.episode_length:
        raise ValueError(
            "min_replay_size must be at least episode_length because the replay "
            "sampler draws full episode-length sequences."
        )
    if args.max_replay_size < args.episode_length:
        raise ValueError("max_replay_size must be at least episode_length.")
    if args.max_replay_size < args.min_replay_size:
        raise ValueError("max_replay_size must be at least min_replay_size.")
    if args.eval_every_epochs < 0:
        raise ValueError("eval_every_epochs must be non-negative. Use 0 to disable evaluation.")
    if args.checkpoint_every_epochs < 1:
        raise ValueError("checkpoint_every_epochs must be at least 1.")
    if args.print_every_epochs < 1:
        raise ValueError("print_every_epochs must be at least 1.")
    if args.capture_vis and args.num_render < 1:
        raise ValueError("num_render must be at least 1 when capture_vis is enabled.")
    if args.env_config_path and args.eval_env_id and args.eval_env_id != args.env_id:
        raise ValueError(
            "env_config_path points to one concrete env config. Use env_config_dir "
            "instead when eval_env_id differs from env_id."
        )
    max_samples_per_training_step = args.num_episodes_per_env * args.num_envs * (args.episode_length - 1)
    if max_samples_per_training_step < args.batch_size:
        raise ValueError(
            "batch_size is larger than the number of relabelled samples available "
            "per training step. Need "
            "num_episodes_per_env * num_envs * (episode_length - 1) >= batch_size."
        )

def load_params(path: str):
    with epath.Path(path).open('rb') as fin:
        buf = fin.read()
    return pickle.loads(buf)

def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open('wb') as fout:
        fout.write(pickle.dumps(params))

def save_final_policy(policy_dir: str | Path, args: Args, actor_params: Any) -> None:
    """Saves the actor-only policy and enough metadata to reload it."""
    policy_dir = Path(policy_dir)
    policy_dir.mkdir(parents=True, exist_ok=True)
    args_dict = resolved_config_dict(args)
    save_params(str(policy_dir / "actor_params.pkl"), actor_params)
    save_params(str(policy_dir / "args.pkl"), args_dict)
    save_params(
        str(policy_dir / "policy.pkl"),
        {
            "actor_params": actor_params,
            "args": args_dict,
        },
    )

def _metric_to_float(value: Any) -> float | None:
    try:
        array = np.asarray(value)
        if array.size != 1:
            return None
        return float(array)
    except (TypeError, ValueError):
        return None

def _format_metric_value(value: float, precision: int = 4) -> str:
    abs_value = abs(value)
    if value == 0.0:
        return "0"
    if abs_value >= 1e5 or abs_value < 1e-3:
        return f"{value:.{precision}e}"
    return f"{value:.{precision}g}"

PROGRESS_METRIC_LABELS = {
    "training/sps": "sps",
    "training/actor_loss": "actor_loss",
    "training/critic_loss": "critic_loss",
    "training/main_crl_loss": "main_crl",
    "training/z_bce_loss": "z_bce",
    "training/actor_survival_penalty": "actor_survival",
    "training/survival_valid_fraction": "z_valid",
    "training/survival_label_mean": "z_target",
    "training/survival_horizon_mean": "z_horizon",
    "training/predicted_z_mean": "pred_z",
    "training/predicted_log_z_mean": "pred_log_z",
    "training/sample_entropy": "entropy",
    "training/log_alpha": "log_alpha",
    "training/num_sgd_batches_per_training_step": "sgd_batches",
    "training/mean_inbatch_negatives": "inbatch_negs",
    "training/future_vs_negative_logit_gap": "pos_neg_gap",
    "eval/episode_reward": "reward",
    "eval/episode_cost": "cost",
    "eval/episode_success": "success",
    "eval/episode_success_easy": "success_easy",
    "eval/episode_success_hard": "success_hard",
    "eval/episode_success_any": "success_any",
    "eval/episode_dist": "dist",
    "eval/avg_episode_length": "length",
}

PROGRESS_METRIC_CATEGORIES = {
    "training/sps": "train",
    "training/actor_loss": "train",
    "training/critic_loss": "train",
    "training/main_crl_loss": "train",
    "training/z_bce_loss": "critic",
    "training/actor_survival_penalty": "actor",
    "training/survival_valid_fraction": "critic",
    "training/survival_label_mean": "critic",
    "training/survival_horizon_mean": "critic",
    "training/predicted_z_mean": "critic",
    "training/predicted_log_z_mean": "critic",
    "training/sample_entropy": "train",
    "training/log_alpha": "train",
    "training/num_sgd_batches_per_training_step": "train",
    "training/mean_inbatch_negatives": "critic",
    "training/future_vs_negative_logit_gap": "logits",
    "eval/episode_reward": "eval",
    "eval/episode_cost": "eval",
    "eval/episode_success": "eval",
    "eval/episode_success_easy": "eval",
    "eval/episode_success_hard": "eval",
    "eval/episode_success_any": "eval",
    "eval/episode_dist": "eval",
    "eval/avg_episode_length": "eval",
}


def _format_metric(metrics: dict[str, Any], name: str, precision: int = 4) -> tuple[str, str] | None:
    if name not in metrics:
        return None
    value = _metric_to_float(metrics[name])
    if value is None:
        return None
    return PROGRESS_METRIC_LABELS.get(name, name.split("/")[-1]), _format_metric_value(value, precision)

def _metric_json_value(value: Any) -> float | str | None:
    value = _metric_to_float(value)
    if value is None:
        return None
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"

def _metrics_json_line(ne: int, args: Args, metrics: dict[str, Any], epoch_time: float, hours_passed: float) -> str:
    env_steps = int(_metric_to_float(metrics.get("training/envsteps")) or 0)
    metric_rows = []
    for key in sorted(metrics):
        value = _metric_json_value(metrics[key])
        if value is None:
            continue
        metric_rows.append(
            {
                "key": key,
                "category": PROGRESS_METRIC_CATEGORIES.get(key, key.split("/", 1)[0]),
                "metric": PROGRESS_METRIC_LABELS.get(key, key.split("/")[-1]),
                "value": value,
            }
        )
    payload = {
        "epoch": ne + 1,
        "total_epochs": args.num_epochs,
        "env_steps": env_steps,
        "epoch_seconds": float(epoch_time),
        "elapsed_hours": float(hours_passed),
        "metrics": metric_rows,
    }
    return "METRICS_JSON\t" + json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)

def _format_metric_rows(
    metrics: dict[str, Any],
    names: list[str],
    label: str,
    per_row: int = 2,
    name_width: int = 32,
    value_width: int = 12,
) -> list[str]:
    items = [_format_metric(metrics, name) for name in names]
    items = [item for item in items if item is not None]
    if not items:
        return []

    rows = []
    label_cell = f"  {label:<7}| "
    blank_label_cell = f"  {'':<7}| "
    for start in range(0, len(items), per_row):
        cells = []
        for metric_name, value in items[start : start + per_row]:
            cells.append(f"{metric_name:<{name_width}} {value:>{value_width}}")
        prefix = label_cell if start == 0 else blank_label_cell
        rows.append(prefix + " | ".join(cells))
    return rows

def _format_progress(ne: int, args: Args, metrics: dict[str, Any], epoch_time: float, hours_passed: float) -> str:
    env_steps = int(_metric_to_float(metrics.get("training/envsteps")) or 0)
    header = (
        f"Epoch {ne + 1:>{len(str(args.num_epochs))}}/{args.num_epochs}  "
        f"env_steps {env_steps:>14,}  "
        f"epoch {epoch_time:>8.2f}s  "
        f"elapsed {hours_passed:>8.3f}h"
    )
    lines = ["", header]

    train_metrics = [
        "training/sps",
        "training/actor_loss",
        "training/critic_loss",
        "training/main_crl_loss",
        "training/z_bce_loss",
        "training/actor_survival_penalty",
        "training/survival_valid_fraction",
        "training/predicted_z_mean",
        "training/log_alpha",
        "training/sample_entropy",
        "training/num_sgd_batches_per_training_step",
    ]
    lines.extend(_format_metric_rows(metrics, train_metrics, "train"))

    lines.extend(
        _format_metric_rows(
            metrics,
            [
                "training/mean_inbatch_negatives",
                "training/future_vs_negative_logit_gap",
            ],
            "critic",
        )
    )

    eval_metrics = [
        "eval/episode_reward",
        "eval/episode_cost",
        "eval/episode_success",
        "eval/episode_success_easy",
        "eval/episode_success_hard",
        "eval/episode_success_any",
        "eval/episode_dist",
        "eval/avg_episode_length",
    ]
    lines.extend(
        _format_metric_rows(
            metrics,
            eval_metrics,
            "eval",
        )
    )

    return "\n".join(lines)

if __name__ == "__main__":

    cli_overrides = _cli_override_names()
    args, config_overrides = _parse_args()
    if args.layout_seed_from_seed:
        args.fixed_object_layout_seed = int(args.seed)
    explicit_overrides = _explicit_override_names(config_overrides, cli_overrides)
    _apply_mode_defaults(args, explicit_overrides)
    
    if args.print_args:
        print("Arguments:", flush=True)
        for arg, value in vars(args).items():
            print(f"{arg}: {value}", flush=True)
        print("\n", flush=True)
    else:
        print(
            "Run: "
            f"env={args.env_id}, seed={args.seed}, loss={args.critic_loss_type}, "
            f"num_envs={args.num_envs}, batch_size={args.batch_size}, "
            f"actor_width={args.actor_network_width}, critic_width={args.critic_network_width}",
            flush=True,
        )

    _validate_args(args)
    if args.dry_run:
        args.track = False
        args.checkpoint = False
        args.save_final_policy = False
        args.capture_vis = False
        args.save_buffer = 0

    args.env_steps_per_actor_step = args.num_envs * args.unroll_length
    print(f"env_steps_per_actor_step: {args.env_steps_per_actor_step}", flush=True)

    args.num_prefill_env_steps = args.min_replay_size * args.num_envs
    print(f"num_prefill_env_steps: {args.num_prefill_env_steps}", flush=True)

    args.num_prefill_actor_steps = int(np.ceil(args.min_replay_size / args.unroll_length))
    print(f"num_prefill_actor_steps: {args.num_prefill_actor_steps}", flush=True)

    args.num_training_steps_per_epoch = (args.total_env_steps - args.num_prefill_env_steps) // (args.num_epochs * args.env_steps_per_actor_step)
    args.num_training_steps_per_epoch = int(max(args.num_training_steps_per_epoch, 1))
    print(f"num_training_steps_per_epoch: {args.num_training_steps_per_epoch}", flush=True)
    
    run_name = f"{args.env_id}{'_' + args.eval_env_id if args.eval_env_id else ''}_{args.batch_size}_{args.total_env_steps}_nenvs:{args.num_envs}_criticwidth:{args.critic_network_width}_actorwidth:{args.actor_network_width}_criticdepth:{args.critic_depth}_actordepth:{args.actor_depth}_{args.seed}"
    print(f"run_name: {run_name}", flush=True)
    
    if args.track:
        if wandb is None or wandb_osh is None or TriggerWandbSyncHook is None:
            raise ImportError("Tracking requires wandb and wandb_osh. Disable tracking with --track False or install the project dependencies.")

        if args.wandb_group ==  '.':
            args.wandb_group = None
            
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group=args.wandb_group,
            dir=args.wandb_dir,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        if args.wandb_mode == 'offline':
            wandb_osh.set_log_level("ERROR")
            trigger_sync = TriggerWandbSyncHook()
        
    save_path = None
    if (
        args.seed_output_dir
        or args.results_root
        or args.checkpoint
        or args.save_final_policy
        or args.capture_vis
        or args.save_buffer
    ):
        from datetime import datetime
        if args.seed_output_dir:
            save_path = Path(args.seed_output_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        elif args.results_root:
            result_run_name = args.run_name or f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            save_path = unique_run_dir(
                Path(args.results_root)
                / args.env_id
                / args.critic_loss_type
                / result_run_name
                / str(args.seed)
            )
        else:
            short_run_name = f"runs/{args.env_id}_{args.seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            save_path = unique_run_dir(Path(args.wandb_dir) / Path(short_run_name))

    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, buffer_key, env_key, eval_env_key, actor_key, sa_key, g_key, _ = jax.random.split(key, 8)

    def make_env(
        env_id=args.env_id,
        eval_mode: bool = False,
        record_as_eval: bool | None = None,
    ):
        print(f"making env with env_id: {env_id}", flush=True)
        if record_as_eval is None:
            record_as_eval = eval_mode
        scaling_env_id = is_scaling_env_id(env_id)
        if not is_safenav_env_id(env_id) and not scaling_env_id:
            raise ValueError(
                f"Unsupported env_id={env_id!r}. scalable_safe_rl currently supports "
                f"SafeNav and scaling_envs: {safenav_env_id_hint()}."
            )

        if scaling_env_id:
            import scaling_envs
            from scaling_envs.env_config import load_env_config as load_scaling_env_config

            env_config_dir = args.env_config_dir or None
            env_config_path = args.env_config_path or None
            env_params: dict[str, Any] = {}
            if args.use_env_config or env_config_dir is not None or env_config_path is not None:
                env_params.update(
                    load_scaling_env_config(
                        scaling_envs.config_env_id_for(env_id),
                        config_dir=env_config_dir,
                        config_path=env_config_path,
                    )
                )
            if scaling_envs.is_maze_env_id(env_id):
                env_params["layout_lidar_num_bins"] = args.layout_lidar_num_bins
            if is_humanoid_env_id(env_id):
                env_params["humanoid_use_spring_gear"] = (
                    args.humanoid_use_spring_gear
                )
            env_params["evaluation_mode"] = eval_mode
            if record_as_eval:
                args.eval_env_params = dict(env_params)
            else:
                args.env_params = dict(env_params)
            env = scaling_envs.make(env_id, use_config=False, **env_params)
            args.layout_requested_num_hazards = 0
            args.layout_num_hazards = 0
            args.layout_num_obstacles = 0
            args.layout_num_gremlins = 0
            args.layout_num_boxes = 0
            args.layout_has_type_obs = False
            env = ScalingCrlObservationWrapper(env, env.goal_indices)
            args.obs_dim = env.obs_dim
            args.goal_start_idx = env.goal_start_idx
            args.goal_end_idx = env.goal_end_idx
            args.raw_goal_dim = int(env.raw_goal_dim)
            args.layout_start_idx = int(env.layout_start_idx)
            args.layout_end_idx = int(env.layout_end_idx)
            args.layout_obs_dim = int(env.layout_obs_dim)
            args.ego_goal_relabel = False
            args.goal_lidar_max_dist = 0.0
            args.relabel_goal_dim = int(env.relabel_goal_dim)
            return env

        from safenav_jax.env_config import load_env_config
        from safenav_jax.envs import config_env_id_for, make as make_safenav_env

        env_config_dir = args.env_config_dir or None
        env_config_path = args.env_config_path or None
        env_kwargs = {
            "relocate_objects_on_reset": args.relocate_objects_on_reset,
            "fixed_object_layout_seed": args.fixed_object_layout_seed,
            "different_object_layout_per_env": args.different_object_layout_per_env,
            "goal_respawn_on_success": args.goal_respawn_on_success,
            "terminate_on_cost": args.eval_terminate_on_cost if eval_mode else args.terminate_on_cost,
            "cost_limit_max": args.cost_limit_max,
        }
        if is_humanoid_env_id(env_id):
            env_kwargs["humanoid_use_spring_gear"] = (
                args.humanoid_use_spring_gear
            )
        if (
            ("goal" in env_id and "maze" not in env_id)
            or is_point_push_env_id(env_id)
        ):
            env_kwargs["object_boundary"] = args.object_boundary
        if ("goal" in env_id and "grid" not in env_id) or is_point_push_env_id(env_id):
            env_kwargs.update(
                {
                    "respawn_goal_require_path_objects": args.respawn_goal_require_path_objects,
                    "respawn_goal_min_path_objects": args.respawn_goal_min_path_objects,
                    "respawn_goal_path_band_scale": args.respawn_goal_path_band_scale,
                }
            )
        if "maze" not in env_id:
            env_kwargs["robot_cost_margin"] = args.robot_cost_margin
        if "maze" not in env_id:
            env_kwargs.update(
                {
                    "fixed_agent_on_reset": args.fixed_agent_on_reset,
                    "fixed_goal_on_reset": args.fixed_goal_on_reset,
                }
            )
        if "goal_grid" in env_id and not eval_mode:
            env_kwargs["min_goal_cell_distance"] = 0.0
        if "goal_grid" in env_id:
            if args.grid_layout_name:
                env_kwargs["grid_layout_name"] = args.grid_layout_name
            if args.eval_grid_layout_name:
                env_kwargs["eval_grid_layout_name"] = args.eval_grid_layout_name
        if is_ant_humanoid_goal_env_id(env_id) and not eval_mode:
            env_kwargs["min_goal_dist"] = 0.0
        if is_ant_humanoid_goal_env_id(env_id) or "goal_grid" in env_id:
            env_kwargs["evaluation_mode"] = eval_mode
        include_object_layout_obs = _optional_bool_arg(args.include_object_layout_obs)
        include_object_type_obs = _optional_bool_arg(args.include_object_type_obs)
        include_box_layout_obs = _optional_bool_arg(args.include_box_layout_obs)
        if include_object_layout_obs is not None and "maze" not in env_id:
            env_kwargs["include_object_layout_obs"] = include_object_layout_obs
        if include_object_type_obs is not None and "maze" not in env_id:
            env_kwargs["include_object_type_obs"] = include_object_type_obs
        if "maze" not in env_id:
            env_kwargs["layout_lidar_num_bins"] = args.layout_lidar_num_bins
        if is_point_car_goal_env_id(env_id):
            if args.parking_mode:
                args.goal_lidar = False
            env_kwargs.update(
                {
                    "ego_view": False if args.parking_mode else args.ego_view,
                    "parking_mode": args.parking_mode,
                    "parking_yaw_tolerance_degrees": args.parking_yaw_tolerance_degrees,
                }
            )
        elif is_point_push_env_id(env_id):
            env_kwargs["ego_view"] = False
        if include_box_layout_obs is not None and "maze" in env_id and (
            "ant" in env_id or "humanoid" in env_id
        ):
            env_kwargs["include_box_layout_obs"] = include_box_layout_obs

        env_params: dict[str, Any] = {}
        if args.use_env_config or env_config_dir is not None or env_config_path is not None:
            env_params.update(
                load_env_config(
                    config_env_id_for(env_id),
                    config_dir=env_config_dir,
                    config_path=env_config_path,
                )
            )
        env_params.update(env_kwargs)
        if record_as_eval:
            args.eval_env_params = dict(env_params)
        else:
            args.env_params = dict(env_params)
        env = make_safenav_env(
            env_id,
            use_config=False,
            **env_params,
        )
        args.layout_requested_num_hazards = int(
            getattr(env, "requested_num_hazards", getattr(env, "_num_hazards", 0))
        )
        args.layout_num_hazards = int(getattr(env, "_num_hazards", 0))
        args.layout_num_obstacles = int(getattr(env, "_num_obstacles", 0))
        args.layout_num_gremlins = int(getattr(env, "_num_gremlins", 0))
        args.layout_num_boxes = int(getattr(getattr(env, "_boxes_xy", np.zeros((0, 2))), "shape", (0,))[0])
        args.layout_has_type_obs = bool(getattr(env, "_include_object_type_obs", False))
        if is_point_push_env_id(env_id):
            args.goal_lidar = False
            if args.point_push_combined_goal:
                env = PointPushCombinedGoalObservationWrapper(env, env.goal_indices)
            else:
                env = ScalingCrlObservationWrapper(env, env.goal_indices)
        elif is_point_car_goal_env_id(env_id) and args.ego_view and not args.parking_mode:
            env = EgoGoalObservationWrapper(env, env.goal_indices, goal_lidar=args.goal_lidar)
        else:
            env = ScalingCrlObservationWrapper(env, env.goal_indices)
        args.obs_dim = env.obs_dim
        args.goal_start_idx = env.goal_start_idx
        args.goal_end_idx = env.goal_end_idx
        args.raw_goal_dim = int(getattr(env, "raw_goal_dim", args.goal_end_idx - args.goal_start_idx))
        args.layout_start_idx = int(getattr(env, "layout_start_idx", args.obs_dim))
        args.layout_end_idx = int(getattr(env, "layout_end_idx", args.obs_dim))
        args.layout_obs_dim = int(getattr(env, "layout_obs_dim", args.layout_end_idx - args.layout_start_idx))
        args.ego_goal_relabel = bool(getattr(env, "uses_ego_goal_relabel", False))
        args.goal_lidar_max_dist = float(getattr(env, "goal_lidar_max_dist", 0.0))
        args.relabel_goal_dim = int(getattr(env, "relabel_goal_dim", args.raw_goal_dim))
        return env
        
    env = make_env()
    env = wrap_safenav_training_env(
        env,
        episode_length=args.episode_length,
    )

    obs_size = env.observation_size
    action_size = env.action_size
    if args.raw_goal_dim == 0:
        args.raw_goal_dim = args.goal_end_idx - args.goal_start_idx
    if args.relabel_goal_dim == 0:
        args.relabel_goal_dim = 2 if args.ego_goal_relabel else args.raw_goal_dim
    state_input_size = args.obs_dim - args.layout_obs_dim if args.ignore_layout_obs else args.obs_dim
    actor_state_input_size = state_input_size
    actor_input_size = actor_state_input_size + args.raw_goal_dim
    critic_state_input_size = state_input_size
    critic_goal_input_size = args.raw_goal_dim
    train_env_metadata = _snapshot_runtime_env_metadata(args)
    env_keys = jax.random.split(env_key, args.num_envs)
    env_state = jax.jit(env.reset)(env_keys)
    env.step = jax.jit(env.step)
    
    print(f"obs_size: {obs_size}, action_size: {action_size}", flush=True)
    
    
    resolved_eval_env_id, resolved_eval_mode = _resolve_eval_env_request(
        args.env_id,
        args.eval_env_id,
    )
        
    # make eval env
    eval_env = None
    if args.eval_every_epochs != 0:
        eval_env = make_env(
            resolved_eval_env_id,
            eval_mode=resolved_eval_mode,
            record_as_eval=True,
        )
        eval_env = wrap_safenav_training_env(
            eval_env,
            episode_length=args.episode_length,
        )
        eval_env_keys = jax.random.split(eval_env_key, args.num_eval_envs)
        eval_env_state = jax.jit(eval_env.reset)(eval_env_keys)
        eval_env.step = jax.jit(eval_env.step)
        if eval_env.observation_size != obs_size or eval_env.action_size != action_size:
            raise ValueError(
                "eval_env_id must expose the same observation and action sizes as env_id. "
                f"train=({obs_size}, {action_size}), "
                f"eval=({eval_env.observation_size}, {eval_env.action_size}), "
                f"resolved_eval_env_id={resolved_eval_env_id!r}"
            )
    _restore_runtime_env_metadata(args, train_env_metadata)
    metrics_jsonl_path = None
    if save_path is not None:
        write_resolved_config(save_path, args)
        metrics_jsonl_path = Path(save_path) / "metrics.jsonl"
        metrics_jsonl_path.unlink(missing_ok=True)

    def make_adam_optimizer(learning_rate: float) -> optax.GradientTransformation:
        transforms = []
        if args.max_grad_norm > 0.0:
            transforms.append(optax.clip_by_global_norm(args.max_grad_norm))
        transforms.append(optax.adam(learning_rate=learning_rate))
        tx = optax.chain(*transforms)
        if args.skip_nonfinite_updates:
            tx = optax.apply_if_finite(tx, max_consecutive_errors=100)
        return tx

    # Network setup: Scaling-CRL-style MLP actor/critic towers.
    actor = Actor(
        action_size=action_size,
        network_width=args.actor_network_width,
        network_depth=args.actor_depth,
        use_relu=args.use_relu,
    )
    actor_params = actor.init(actor_key, np.ones([1, actor_input_size]))
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor_params,
        tx=make_adam_optimizer(args.actor_lr),
    )

    sa_encoder = SA_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        use_relu=args.use_relu,
        output_dim=args.embedding_dim,
    )
    g_encoder = G_encoder(
        network_width=args.critic_network_width,
        network_depth=args.critic_depth,
        use_relu=args.use_relu,
        output_dim=args.embedding_dim,
    )
    sa_encoder_params = sa_encoder.init(sa_key, np.ones([1, critic_state_input_size]), np.ones([1, action_size]))
    g_encoder_params = g_encoder.init(g_key, np.ones([1, critic_goal_input_size]))
    critic_params = {
        "sa_encoder": sa_encoder_params,
        "g_encoder": g_encoder_params,
    }
    critic_state = TrainState.create(
        apply_fn=None,
        params=critic_params,
        tx=make_adam_optimizer(args.critic_lr),
    )

    z_encoder = None
    z_state = None
    if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE:
        z_encoder = Z_encoder(
            network_width=args.critic_network_width,
            network_depth=args.z_encoder_depth,
            use_relu=args.use_relu,
        )
        z_key = jax.random.fold_in(sa_key, 0x5A17)
        z_params = z_encoder.init(
            z_key,
            np.ones([1, critic_state_input_size]),
            np.ones([1, action_size]),
        )
        z_state = TrainState.create(
            apply_fn=z_encoder.apply,
            params=z_params,
            tx=make_adam_optimizer(args.critic_lr),
        )

    # Entropy coefficient
    target_entropy = -args.entropy_param * action_size # action_size = 8 for ant, 17 for humanoid, etc
    log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
    alpha_state = TrainState.create(
        apply_fn=None,
        params={"log_alpha": log_alpha},
        tx=make_adam_optimizer(args.alpha_lr),
    )

    def critic_scaling_crl_score(sa_repr, g_repr):
        def neg_l2_from_squared(squared_dist):
            return -jnp.sqrt(jnp.maximum(squared_dist, 0.0) + 1e-12)

        if (
            sa_repr.ndim == 3
            and g_repr.ndim == 3
            and sa_repr.shape[1] == 1
            and g_repr.shape[0] == 1
        ):
            sa = sa_repr[:, 0, :]
            g = g_repr[0, :, :]
            squared_dist = (
                jnp.sum(sa**2, axis=-1)[:, None]
                + jnp.sum(g**2, axis=-1)[None, :]
                - 2.0 * jnp.matmul(sa, g.T)
            )
            return neg_l2_from_squared(squared_dist)

        return neg_l2_from_squared(jnp.sum((sa_repr - g_repr) ** 2, axis=-1))
    
    # Trainstate
    training_state = TrainingState(
        env_steps=jnp.zeros(()),
        gradient_steps=jnp.zeros(()),
        actor_state=actor_state,
        critic_state=critic_state,
        z_state=z_state,
        alpha_state=alpha_state,
        target_actor_params=None,
        target_critic_params=None,
    )

    def training_checkpoint_params(state):
        params = (
            state.alpha_state.params,
            state.actor_state.params,
            state.critic_state.params,
        )
        if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE:
            return params + (state.z_state.params,)
        return params

    #Replay Buffer
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_state_extras = {
        "seed": 0.0,
        "agent_yaw": 0.0,
        "task_goal": jnp.zeros((args.raw_goal_dim,)),
        "task_goal_xy": jnp.zeros((args.relabel_goal_dim,)),
        "achieved_goal": jnp.zeros((args.relabel_goal_dim,)),
        "relabel_anchor_xy": jnp.zeros((2,)),
    }
    if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE:
        dummy_state_extras["truncation"] = 0.0
    dummy_transition = Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=0.0,
        discount=0.0,
        extras={"state_extras": dummy_state_extras},
    )

    def jit_wrap(buffer):
        buffer.insert_internal = jax.jit(buffer.insert_internal)
        buffer.sample_internal = jax.jit(buffer.sample_internal)
        return buffer
    
    replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=args.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=args.batch_size,
                num_envs=args.num_envs,
                episode_length=args.episode_length,
            )
        )
    buffer_state = jax.jit(replay_buffer.init)(buffer_key)

    if args.dry_run:
        print("Dry run complete: initialized envs, networks, and replay buffer.", flush=True)
        print(
            "Shapes: "
            f"obs_size={obs_size}, actor_input_size={actor_input_size}, "
            f"critic_state_dim={args.obs_dim}, goal_dim={args.raw_goal_dim}, "
            f"critic_state_input_size={critic_state_input_size}, "
            f"layout_dim={args.layout_obs_dim}, "
            f"ignore_layout_obs={args.ignore_layout_obs}, "
            f"critic_goal_input_size={critic_goal_input_size}, "
            f"action_size={action_size}",
            flush=True,
        )
        raise SystemExit(0)

    def transition_goal_from_obs(state):
        return state.obs[..., args.goal_start_idx : args.goal_end_idx]

    def state_without_layout(obs):
        if not args.ignore_layout_obs:
            return obs
        return jnp.concatenate(
            [obs[..., : args.layout_start_idx], obs[..., args.layout_end_idx : args.obs_dim]],
            axis=-1,
        )

    def apply_actor_network(actor_params, state, goal):
        actor_state = state_without_layout(state)
        return actor.apply(actor_params, jnp.concatenate([actor_state, goal], axis=-1))

    def achieved_goal_from_metrics(state):
        if is_point_push_env_id(args.env_id):
            cube_xy = jnp.stack(
                [state.metrics["x_position"], state.metrics["y_position"]],
                axis=-1,
            )
            if not args.point_push_combined_goal:
                return cube_xy
            agent_xy = jnp.stack(
                [state.metrics["agent_x"], state.metrics["agent_y"]],
                axis=-1,
            )
            return jnp.concatenate([cube_xy, agent_xy], axis=-1)
        if is_point_car_goal_env_id(args.env_id) and args.parking_mode:
            return jnp.stack(
                [
                    state.metrics["x_position"],
                    state.metrics["y_position"],
                    state.metrics["agent_yaw"],
                ],
                axis=-1,
            )
        if args.ego_goal_relabel and "x_position" in state.metrics and "y_position" in state.metrics:
            return jnp.stack([state.metrics["x_position"], state.metrics["y_position"]], axis=-1)
        if args.raw_goal_dim == 1 and "x_position" in state.metrics:
            return state.metrics["x_position"][..., None]
        if args.raw_goal_dim == 2 and "x_position" in state.metrics and "y_position" in state.metrics:
            return jnp.stack([state.metrics["x_position"], state.metrics["y_position"]], axis=-1)
        if (
            args.raw_goal_dim == 3
            and "x_position" in state.metrics
            and "y_position" in state.metrics
            and "z_position" in state.metrics
        ):
            return jnp.stack(
                [
                    state.metrics["x_position"],
                    state.metrics["y_position"],
                    state.metrics["z_position"],
                ],
                axis=-1,
            )
        return state.obs[..., args.goal_start_idx : args.goal_end_idx]

    def agent_yaw_from_metrics(state):
        if "agent_yaw" in state.metrics:
            return state.metrics["agent_yaw"]
        return jnp.zeros(state.obs.shape[:-1], dtype=state.obs.dtype)

    def relabel_anchor_xy_from_metrics(state):
        if args.ego_goal_relabel and "x_position" in state.metrics and "y_position" in state.metrics:
            return jnp.stack([state.metrics["x_position"], state.metrics["y_position"]], axis=-1)
        return jnp.zeros(state.obs.shape[:-1] + (2,), dtype=state.obs.dtype)

    def task_goal_xy_from_state(state):
        if is_point_push_env_id(args.env_id):
            if "goal_xy" in state.info:
                task_goal_xy = state.info["goal_xy"]
            else:
                task_goal_xy = jnp.stack(
                    [state.metrics["goal_x"], state.metrics["goal_y"]],
                    axis=-1,
                )
            if not args.point_push_combined_goal:
                return task_goal_xy
            return jnp.concatenate([task_goal_xy, jnp.zeros_like(task_goal_xy)], axis=-1)
        if args.relabel_goal_dim != 2:
            return task_goal_from_state(state)
        if "goal_xy" in state.info:
            return state.info["goal_xy"]
        if "goal_x" in state.metrics and "goal_y" in state.metrics:
            return jnp.stack([state.metrics["goal_x"], state.metrics["goal_y"]], axis=-1)
        if args.relabel_goal_dim == 2 and args.raw_goal_dim == 2:
            return transition_goal_from_obs(state)
        return jnp.zeros(state.obs.shape[:-1] + (args.relabel_goal_dim,), dtype=state.obs.dtype)

    def task_goal_from_state(state):
        return state.obs[..., args.obs_dim : args.obs_dim + args.raw_goal_dim]

    def build_state_extras(env_state, next_state, extra_fields):
        done = next_state.done
        state_extras = {}
        for field in extra_fields:
            if field in next_state.info:
                state_extras[field] = next_state.info[field]
            elif field == "truncation":
                state_extras[field] = jnp.zeros_like(done)
            elif field == "seed":
                state_extras[field] = jnp.zeros_like(done, dtype=jnp.int32)
            else:
                raise KeyError(field)
        state_extras.update(
            {
                "agent_yaw": agent_yaw_from_metrics(env_state),
                "task_goal": task_goal_from_state(env_state),
                "task_goal_xy": task_goal_xy_from_state(env_state),
                "achieved_goal": achieved_goal_from_metrics(env_state),
                "relabel_anchor_xy": relabel_anchor_xy_from_metrics(env_state),
            }
        )
        return state_extras

    def deterministic_actor_step(training_state, env, env_state, extra_fields):
        state = env_state.obs[..., :args.obs_dim]
        goal = task_goal_from_state(env_state)
        means, _ = apply_actor_network(
            training_state.actor_state.params,
            state,
            goal,
        )
        actions = nn.tanh( means )

        nstate = env.step(env_state, actions)
        state_extras = build_state_extras(env_state, nstate, extra_fields)
        
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
    
    def actor_step(training_state, env, env_state, key, extra_fields):
        state = env_state.obs[..., :args.obs_dim]
        goal = task_goal_from_state(env_state)
        means, log_stds = apply_actor_network(
            training_state.actor_state.params,
            state,
            goal,
        )
        stds = jnp.exp(log_stds)
        actions = nn.tanh( means + stds * jax.random.normal(key, shape=means.shape, dtype=means.dtype) )

        nstate = env.step(env_state, actions)
        state_extras = build_state_extras(env_state, nstate, extra_fields)
        
        return nstate, Transition(
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
        
    def multi_sample_actor_step(training_state, env, env_state, key, K, extra_fields):
        # Get K sets of actions from the actor
        keys = jax.random.split(key, K)
        state = env_state.obs[:, :args.obs_dim]
        goal = task_goal_from_state(env_state)
        means, log_stds = apply_actor_network(
            training_state.actor_state.params,
            state,
            goal,
        )
        stds = jnp.exp(log_stds)
        
        actions = jnp.stack([
            nn.tanh(means + stds * jax.random.normal(k, shape=means.shape, dtype=means.dtype))
            for k in keys
        ])
        
        state_input = state_without_layout(state)
        sa_reprs = jax.vmap(
            lambda a: sa_encoder.apply(
                training_state.critic_state.params["sa_encoder"],
                state_input,
                a,
            )
        )(actions)
        g_repr = g_encoder.apply(
            training_state.critic_state.params["g_encoder"],
            goal
        )

        q_values = critic_scaling_crl_score(sa_reprs, g_repr[None, :, :])
        
        best_action_idx = jnp.argmax(q_values, axis=0)
        best_actions = jnp.take_along_axis(
            actions,
            best_action_idx[None, :, None],
            axis=0
        )[0]
        
        # Step environment with best actions
        nstate = env.step(env_state, best_actions)
        state_extras = build_state_extras(env_state, nstate, extra_fields)
        
        return nstate, Transition(
            observation=env_state.obs,
            action=best_actions,
            reward=nstate.reward,
            discount=1-nstate.done,
            extras={"state_extras": state_extras},
        )
    
    

    @jax.jit
    def get_experience(training_state, env_state, buffer_state, key):
        rollout_extra_fields = (
            ("seed", "truncation")
            if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE
            else ("seed",)
        )

        @jax.jit
        def f(carry, unused_t): #conducts a single actor step in environment
            env_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            _, action_key = jax.random.split(current_key)
            if args.expl_actor == 1:
                env_state, transition = actor_step(
                    training_state,
                    env,
                    env_state,
                    action_key,
                    extra_fields=rollout_extra_fields,
                )
            elif args.expl_actor == 0:
                env_state, transition = deterministic_actor_step(
                    training_state,
                    env,
                    env_state,
                    extra_fields=rollout_extra_fields,
                )
            else:
                env_state, transition = multi_sample_actor_step(
                    training_state,
                    env,
                    env_state,
                    action_key,
                    args.expl_actor,
                    extra_fields=rollout_extra_fields,
                )
            return (env_state, next_key), transition

        (env_state, _), data = jax.lax.scan(f, (env_state, key), (), length=args.unroll_length)

        buffer_state = replay_buffer.insert(buffer_state, data)
        return env_state, buffer_state

    def prefill_replay_buffer(training_state, env_state, buffer_state, key):
        @jax.jit
        def f(carry, unused):
            del unused
            training_state, env_state, buffer_state, key = carry
            key, new_key = jax.random.split(key)
            env_state, buffer_state = get_experience(
                training_state,
                env_state,
                buffer_state,
                key,
            
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + args.env_steps_per_actor_step,
            )
            return (training_state, env_state, buffer_state, new_key), ()

        return jax.lax.scan(f, (training_state, env_state, buffer_state, key), (), length=args.num_prefill_actor_steps)[0]

    @jax.jit
    def update_actor_and_alpha(transitions, training_state, key):
        transitions = jax.tree_util.tree_map(
            lambda x: x[: args.batch_size] if x.ndim > 0 else x,
            transitions,
        )

        def weighted_mean(values, weights):
            weights = weights.astype(values.dtype)
            return jnp.sum(values * weights) / jnp.maximum(jnp.sum(weights), 1.0)

        def actor_loss(actor_params, critic_params, z_params, log_alpha, transitions, key):
            obs = transitions.observation
            state = obs[:, : args.obs_dim]
            _, action_key, _ = jax.random.split(key, 3)

            goal = obs[:, args.obs_dim : args.obs_dim + args.raw_goal_dim]
            actor_weight = transitions.extras["future_valid"].astype(jnp.float32)
            means, log_stds = apply_actor_network(
                actor_params,
                state,
                goal,
            )
            if args.actor_mean_clip > 0.0:
                means = jnp.clip(means, -args.actor_mean_clip, args.actor_mean_clip)
            stds = jnp.exp(log_stds)
            noise = jax.random.normal(action_key, shape=means.shape, dtype=means.dtype)
            x_ts = means + stds * noise
            action = nn.tanh(x_ts)
            log_prob = -0.5 * (
                jnp.square(noise) + 2.0 * log_stds + jnp.log(2.0 * jnp.pi)
            )
            log_prob -= jnp.log((1 - jnp.square(action)) + 1e-6)
            log_prob = log_prob.sum(-1)

            sa_encoder_params = critic_params["sa_encoder"]
            g_encoder_params = critic_params["g_encoder"]
            sa_repr = sa_encoder.apply(
                sa_encoder_params,
                state_without_layout(state),
                action,
            )
            g_repr = g_encoder.apply(g_encoder_params, goal)
            qf_pi = critic_scaling_crl_score(sa_repr, g_repr)

            if args.entropy_mode == ENTROPY_MODE_NONE:
                loss = weighted_mean(-qf_pi, actor_weight)
            elif args.entropy_mode == ENTROPY_MODE_FIXED:
                fixed_alpha = jnp.asarray(args.fixed_alpha, dtype=qf_pi.dtype)
                loss = weighted_mean(fixed_alpha * log_prob - qf_pi, actor_weight)
            else:
                loss = weighted_mean(
                    jnp.exp(log_alpha) * log_prob - qf_pi,
                    actor_weight,
                )

            survival_penalty = jnp.zeros((), dtype=loss.dtype)
            predicted_z = jnp.zeros((), dtype=loss.dtype)
            predicted_log_z = jnp.zeros((), dtype=loss.dtype)
            if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE:
                z_logits = z_encoder.apply(
                    z_params,
                    state_without_layout(state),
                    action,
                )[..., 0]
                log_z = jax.nn.log_sigmoid(z_logits)
                predicted_log_z = weighted_mean(log_z, actor_weight)
                predicted_z = weighted_mean(jax.nn.sigmoid(z_logits), actor_weight)
                survival_penalty = -predicted_log_z
                loss = loss + survival_penalty

            return loss, (
                log_prob,
                actor_weight,
                survival_penalty,
                predicted_z,
                predicted_log_z,
            )

        def alpha_loss(alpha_params, log_prob, actor_weight):
            alpha = jnp.exp(alpha_params["log_alpha"])
            actor_weight = actor_weight.astype(log_prob.dtype)
            per_example = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
            return jnp.sum(per_example * actor_weight) / jnp.maximum(
                jnp.sum(actor_weight),
                1.0,
            )

        z_params = (
            training_state.z_state.params
            if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE
            else None
        )
        (
            actorloss,
            (
                log_prob,
                actor_weight,
                actor_survival_penalty,
                actor_predicted_z,
                actor_predicted_log_z,
            ),
        ), actor_grad = jax.value_and_grad(actor_loss, has_aux=True)(
            training_state.actor_state.params,
            training_state.critic_state.params,
            z_params,
            training_state.alpha_state.params["log_alpha"],
            transitions,
            key,
        )
        new_actor_state = training_state.actor_state.apply_gradients(grads=actor_grad)

        if args.entropy_mode == ENTROPY_MODE_LEARNED:
            alphaloss, alpha_grad = jax.value_and_grad(alpha_loss)(
                training_state.alpha_state.params,
                log_prob,
                actor_weight,
            )
            new_alpha_state = training_state.alpha_state.apply_gradients(
                grads=alpha_grad
            )
        else:
            alphaloss = jnp.zeros((), dtype=log_prob.dtype)
            new_alpha_state = training_state.alpha_state

        training_state = training_state.replace(
            actor_state=new_actor_state,
            alpha_state=new_alpha_state,
        )
        metrics = {
            "sample_entropy": -log_prob,
            "actor_loss": actorloss,
            "alph_aloss": alphaloss,
            "log_alpha": training_state.alpha_state.params["log_alpha"],
            "actor_valid_fraction": jnp.mean(actor_weight),
        }
        if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE:
            metrics.update(
                actor_survival_penalty=actor_survival_penalty,
                actor_predicted_z_mean=actor_predicted_z,
                actor_predicted_log_z_mean=actor_predicted_log_z,
            )
        return training_state, metrics

    @jax.jit
    def update_critic(transitions, training_state, key):
        transitions = jax.tree_util.tree_map(
            lambda x: x[: args.batch_size] if x.ndim > 0 else x,
            transitions,
        )

        def critic_loss(critic_params, transitions, key):
            # Preserve the historical key split even though the retained loss
            # does not use its subkeys.
            jax.random.split(key)

            sa_encoder_params = critic_params["sa_encoder"]
            g_encoder_params = critic_params["g_encoder"]
            obs = transitions.observation[:, : args.obs_dim]
            action = transitions.action
            sa_repr = sa_encoder.apply(
                sa_encoder_params,
                state_without_layout(obs),
                action,
            )
            goal = transitions.observation[
                :, args.obs_dim : args.obs_dim + args.raw_goal_dim
            ]
            goal_repr = g_encoder.apply(g_encoder_params, goal)
            logits = critic_scaling_crl_score(
                sa_repr[:, None, :],
                goal_repr[None, :, :],
            )

            future_valid = transitions.extras["future_valid"].astype(bool)
            valid_logits_mask = future_valid[:, None] & future_valid[None, :]
            logits = jnp.where(valid_logits_mask, logits, -1e9)
            pos_logits = jnp.diag(logits)
            per_anchor_loss = -(
                pos_logits - jax.nn.logsumexp(logits, axis=1)
            )

            def masked_mean(values, mask):
                mask_f = mask.astype(values.dtype)
                return jnp.sum(values * mask_f) / jnp.maximum(
                    jnp.sum(mask_f),
                    1.0,
                )

            survival_mass = (
                transitions.extras["survival_mass"]
                if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE
                else None
            )
            main_crl_loss = _reduce_infonce_rows(
                per_anchor_loss,
                future_valid,
                survival_mass,
            )
            logsumexp = jax.nn.logsumexp(logits + 1e-6, axis=1)
            loss = main_crl_loss + args.logsumexp_penalty_coeff * masked_mean(
                logsumexp**2,
                future_valid,
            )

            batch_size = obs.shape[0]
            batch_indices = jnp.arange(batch_size)
            positive_mask = (
                jax.nn.one_hot(batch_indices, batch_size, dtype=bool)
                & valid_logits_mask
            )
            negative_mask = valid_logits_mask & (
                batch_indices[:, None] != batch_indices[None, :]
            )
            logits_pos = masked_mean(pos_logits, future_valid)
            logits_neg = masked_mean(logits, negative_mask)
            mean_inbatch_negatives = masked_mean(
                jnp.sum(negative_mask.astype(jnp.float32), axis=1),
                future_valid,
            )
            predicted_idx = jnp.argmax(logits, axis=1)
            categorical_accuracy = masked_mean(
                jnp.take_along_axis(
                    positive_mask,
                    predicted_idx[:, None],
                    axis=1,
                )[:, 0].astype(jnp.float32),
                future_valid,
            )
            return loss, (
                main_crl_loss,
                logsumexp,
                categorical_accuracy,
                logits_pos,
                logits_neg,
                mean_inbatch_negatives,
            )

        (
            loss,
            (
                main_crl_loss,
                logsumexp,
                categorical_accuracy,
                logits_pos,
                logits_neg,
                mean_inbatch_negatives,
            ),
        ), grad = jax.value_and_grad(critic_loss, has_aux=True)(
            training_state.critic_state.params,
            transitions,
            key,
        )
        new_critic_state = training_state.critic_state.apply_gradients(grads=grad)
        training_state = training_state.replace(critic_state=new_critic_state)

        z_bce_loss = jnp.zeros((), dtype=loss.dtype)
        survival_valid_fraction = jnp.zeros((), dtype=loss.dtype)
        survival_label_mean = jnp.zeros((), dtype=loss.dtype)
        survival_horizon_mean = jnp.zeros((), dtype=loss.dtype)
        predicted_z_mean = jnp.zeros((), dtype=loss.dtype)
        predicted_log_z_mean = jnp.zeros((), dtype=loss.dtype)
        if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE:
            def z_loss_fn(z_params):
                z_state_input = state_without_layout(
                    transitions.observation[:, : args.obs_dim]
                )
                z_logits = z_encoder.apply(
                    z_params,
                    z_state_input,
                    transitions.action,
                )[..., 0]
                survival_label = transitions.extras["survival_label"].astype(
                    z_logits.dtype
                )
                survival_valid = transitions.extras["survival_valid"].astype(bool)
                valid_float = survival_valid.astype(z_logits.dtype)
                denominator = jnp.maximum(jnp.sum(valid_float), 1.0)
                per_example_loss = optax.sigmoid_binary_cross_entropy(
                    z_logits,
                    survival_label,
                )
                z_loss = jnp.sum(per_example_loss * valid_float) / denominator
                target_mean = (
                    jnp.sum(survival_label * valid_float) / denominator
                )
                horizon_mean = jnp.sum(
                    transitions.extras["survival_horizon"].astype(z_logits.dtype)
                    * valid_float
                ) / denominator
                predicted_z = jax.nn.sigmoid(z_logits)
                predicted_log_z = jax.nn.log_sigmoid(z_logits)
                predicted_z_mean = (
                    jnp.sum(predicted_z * valid_float) / denominator
                )
                predicted_log_z_mean = (
                    jnp.sum(predicted_log_z * valid_float) / denominator
                )
                return z_loss, (
                    jnp.mean(valid_float),
                    target_mean,
                    horizon_mean,
                    predicted_z_mean,
                    predicted_log_z_mean,
                )

            (
                z_bce_loss,
                (
                    survival_valid_fraction,
                    survival_label_mean,
                    survival_horizon_mean,
                    predicted_z_mean,
                    predicted_log_z_mean,
                ),
            ), z_grad = jax.value_and_grad(z_loss_fn, has_aux=True)(
                training_state.z_state.params
            )
            new_z_state = training_state.z_state.apply_gradients(grads=z_grad)
            training_state = training_state.replace(z_state=new_z_state)

        metrics = {
            "categorical_accuracy": categorical_accuracy,
            "logits_pos": logits_pos,
            "logits_neg": logits_neg,
            "mean_logit_future": logits_pos,
            "mean_logit_negative": logits_neg,
            "mean_logit_inbatch_negative": logits_neg,
            "future_vs_negative_logit_gap": logits_pos - logits_neg,
            "logsumexp": jnp.mean(logsumexp),
            "critic_loss": loss,
            "main_crl_loss": main_crl_loss,
            "mean_inbatch_negatives": mean_inbatch_negatives,
            "num_sgd_batches_per_training_step": jnp.asarray(
                args.num_sgd_batches_per_training_step,
                dtype=loss.dtype,
            ),
        }
        if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE:
            metrics.update(
                z_bce_loss=z_bce_loss,
                survival_valid_fraction=survival_valid_fraction,
                survival_label_mean=survival_label_mean,
                survival_horizon_mean=survival_horizon_mean,
                predicted_z_mean=predicted_z_mean,
                predicted_log_z_mean=predicted_log_z_mean,
            )
        return training_state, metrics

    @jax.jit
    def sgd_step(carry, transitions):
        training_state, key = carry
        key, critic_key, actor_key = jax.random.split(key, 3)

        # Preserve the established Scaling-CRL update order.
        training_state, actor_metrics = update_actor_and_alpha(
            transitions,
            training_state,
            actor_key,
        )
        training_state, critic_metrics = update_critic(
            transitions,
            training_state,
            critic_key,
        )
        training_state = training_state.replace(
            gradient_steps=training_state.gradient_steps + 1
        )

        metrics = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        return (training_state, key), metrics

    @jax.jit
    def training_step(training_state, env_state, buffer_state, key, t):
        del t
        (
            experience_key1,
            experience_key2,
            sampling_key,
            training_key,
            sgd_batches_key,
        ) = jax.random.split(key, 5)

        env_state, buffer_state = get_experience(
            training_state,
            env_state,
            buffer_state,
            experience_key1,
        )
        training_state = training_state.replace(
            env_steps=training_state.env_steps + args.env_steps_per_actor_step,
        )

        transitions_list = []
        for _ in range(args.num_episodes_per_env):
            buffer_state, new_transitions = replay_buffer.sample(buffer_state)
            transitions_list.append(new_transitions)
        transitions = jax.tree_util.tree_map(
            lambda *arrays: jnp.concatenate(arrays, axis=0),
            *transitions_list,
        )

        batch_keys = jax.random.split(
            sampling_key,
            transitions.observation.shape[0],
        )
        flatten_fn = (
            TrajectoryUniformSamplingQueue.flatten_crl_survive_fn
            if args.critic_loss_type == SCALING_CRL_SURVIVE_LOSS_TYPE
            else TrajectoryUniformSamplingQueue.flatten_crl_fn
        )
        transitions = jax.vmap(flatten_fn, in_axes=(None, 0, 0))(
            (
                args.gamma,
                args.obs_dim,
                args.goal_start_idx,
                args.goal_end_idx,
                args.ego_goal_relabel,
                args.goal_lidar,
                args.layout_lidar_num_bins,
                args.goal_lidar_max_dist,
            ),
            transitions,
            batch_keys,
        )

        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"),
            transitions,
        )
        permutation = jax.random.permutation(
            experience_key2,
            len(transitions.observation),
        )
        transitions = jax.tree_util.tree_map(
            lambda x: x[permutation],
            transitions,
        )

        num_full_batches = len(transitions.observation) // args.batch_size
        transitions = jax.tree_util.tree_map(
            lambda x: x[: num_full_batches * args.batch_size],
            transitions,
        )
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(
                x,
                (-1, args.batch_size) + x.shape[1:],
            ),
            transitions,
        )
        if args.use_all_batches == 0:
            num_total_batches = transitions.observation.shape[0]
            selected_indices = jax.random.permutation(
                sgd_batches_key,
                num_total_batches,
            )[: args.num_sgd_batches_per_training_step]
            transitions = jax.tree_util.tree_map(
                lambda x: x[selected_indices],
                transitions,
            )

        (training_state, _), metrics = jax.lax.scan(
            sgd_step,
            (training_state, training_key),
            transitions,
        )
        return (training_state, env_state, buffer_state), metrics

    @jax.jit
    def training_epoch(
        training_state,
        env_state,
        buffer_state,
        key,
    ):  
        @jax.jit
        def f(carry, t):
            ts, es, bs, k = carry
            k, train_key = jax.random.split(k, 2)
            (ts, es, bs,), metrics = training_step(ts, es, bs, train_key, t)
            return (ts, es, bs, k), metrics

        (training_state, env_state, buffer_state, key), metrics = jax.lax.scan(f, (training_state, env_state, buffer_state, key), jnp.arange(args.num_training_steps_per_epoch * args.training_steps_multiplier))

        
        metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
        return training_state, env_state, buffer_state, metrics

    key, prefill_key = jax.random.split(key, 2)

    training_state, env_state, buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, prefill_key
    )
    

    evaluator = None
    if args.eval_every_epochs == 0:
        print("Evaluation disabled because eval_every_epochs=0.", flush=True)
    elif args.eval_actor == 0:
        '''Setting up evaluator'''
        evaluator = CrlEvaluator(
            lambda training_state, env, env_state, key, extra_fields: deterministic_actor_step(
                training_state,
                env,
                env_state,
                extra_fields,
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
        
    elif args.eval_actor == 1:
        key, _ = jax.random.split(key)
        evaluator = CrlEvaluator(
            actor_step,
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    
    elif args.eval_actor > 1:
        key, _ = jax.random.split(key)
        evaluator = CrlEvaluator(
            # Replace deterministic_actor_step with a partial function of multi_sample_actor_step
            lambda training_state, env, env_state, action_key, extra_fields: multi_sample_actor_step(
                training_state, 
                env, 
                env_state, 
                action_key, 
                args.eval_actor,
                extra_fields,
            ),
            eval_env,
            num_eval_envs=args.num_eval_envs,
            episode_length=args.episode_length,
            key=eval_env_key,
        )
    

    training_walltime = 0
    print('starting training....', flush=True)
    start_time = time.time() 
    for ne in range(args.num_epochs):
        
        t = time.time()

        key, epoch_key = jax.random.split(key)
        training_state, env_state, buffer_state, metrics = training_epoch(training_state, env_state, buffer_state, epoch_key)
        
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time

        sps = (args.env_steps_per_actor_step * args.num_training_steps_per_epoch) / epoch_training_time
        metrics = {
            "training/sps": sps,
            "training/walltime": training_walltime,
            "training/envsteps": training_state.env_steps.item(),
            **{f"training/{name}": value for name, value in metrics.items()},
        }
        should_eval = (
            evaluator is not None
            and (
                ne == 0
                or ne == args.num_epochs - 1
                or (ne + 1) % args.eval_every_epochs == 0
            )
        )
        if should_eval:
            metrics = evaluator.run_evaluation(training_state, metrics)

        hours_passed = (time.time() - start_time) / 3600
        should_print = (
            ne == 0
            or ne == args.num_epochs - 1
            or (ne + 1) % args.print_every_epochs == 0
        )
        if should_print:
            metrics_line = _metrics_json_line(
                ne,
                args,
                metrics,
                epoch_training_time,
                hours_passed,
            )
            if metrics_jsonl_path is not None:
                append_metrics_json_line(metrics_jsonl_path, metrics_line)
            print(metrics_line, flush=True)
            print(_format_progress(ne, args, metrics, epoch_training_time, hours_passed), flush=True)

        if args.checkpoint:
            if ne < 5 or ne >= args.num_epochs - 5 or (ne + 1) % args.checkpoint_every_epochs == 0:
                # Save current policy and critic params.
                params = training_checkpoint_params(training_state)
                path = f"{save_path}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)
        
        if args.track:
            wandb.log(metrics, step=ne)

            if args.wandb_mode == 'offline':
                trigger_sync()

    
    if args.checkpoint:
        # Save current policy and critic params.
        params = training_checkpoint_params(training_state)
        path = f"{save_path}/final.pkl"
        save_params(path, params)
        
    if args.save_final_policy:
        final_policy_dir = Path(save_path) / "final_policy"
        save_final_policy(final_policy_dir, args, training_state.actor_state.params)
        print(f"Saved final policy to {final_policy_dir}", flush=True)
        
    # After training is complete, render the final policy
    if args.capture_vis:
        def render_policy(training_state, save_path):
            """Renders the policy and saves it as an HTML file."""
            rollout_states = []
            for i in range(args.num_render):
                render_env = make_env(
                    resolved_eval_env_id,
                    eval_mode=resolved_eval_mode,
                    record_as_eval=True,
                )

                @jax.jit
                def policy_step(render_env_state, actor_params):
                    state = render_env_state.obs[..., :args.obs_dim]
                    goal = task_goal_from_state(render_env_state)
                    means, _ = apply_actor_network(
                        actor_params,
                        state,
                        goal,
                    )
                    actions = nn.tanh(means)
                    next_state = render_env.step(render_env_state, actions)
                    return next_state, render_env_state
                
                rng = jax.random.PRNGKey(args.seed + i + 1)
                render_env_state = jax.jit(render_env.reset)(rng)
                
                for _ in range(args.vis_length):
                    render_env_state, current_state = policy_step(
                        render_env_state,
                        training_state.actor_state.params,
                    )
                    rollout_states.append(sync_mocap_pipeline_state_for_render(render_env, current_state.pipeline_state))
            
            # Render and save
            html_string = html.render(render_env.sys, rollout_states)
            render_path = f"{save_path}/vis.html"
            with open(render_path, "w", encoding="utf-8") as f:
                f.write(html_string)
            if args.track:
                wandb.log({"vis": wandb.Html(html_string)})
            
        print("Rendering final policy...", flush=True)
        try:
            render_policy(training_state, save_path)
        except Exception as e:
            print(f"Error rendering final policy: {e}", flush=True)
        
    #After training is complete, save the Args
    if args.checkpoint:
        with open(f"{save_path}/args.pkl", 'wb') as f:
            pickle.dump(args, f)
        print(f"Saved args to {save_path}/args.pkl", flush=True)
        
    #After training is complete, save the replay buffer (if save_buffer is 1, this takes a lot of memory)
    if args.save_buffer:
        print("Saving final buffer_state and buffer data (everything needed to recreate replay_buffer)...", flush=True)
        try:
            buffer_path = f"{save_path}/final_buffer.pkl"
            buffer_data = {
                'buffer_state': buffer_state,
                'max_replay_size': args.max_replay_size,
                'batch_size': args.batch_size,
                'num_envs': args.num_envs,
                'episode_length': args.episode_length,
            }
            with open(buffer_path, 'wb') as f:
                pickle.dump(buffer_data, f)
            print(f"Saved replay_buffer to {buffer_path}", flush=True)
        except Exception as e:
            print(f"Error saving final replay buffer: {e}", flush=True)
