"""Task-local lidar assembly for continuous ant/humanoid goal environments."""

from __future__ import annotations

import jax
from jax import numpy as jp


class TwoChannelGlobalLidarMixin:
    """Build hazard and gremlin lidar channels in one world-frame operation."""

    def _layout_obs(
        self,
        pipeline_state,
        goal_xy,
        hazards_xy,
        obstacles_xy,
        gremlins_xy,
        gremlin_centers_xy,
    ):
        del obstacles_xy, gremlin_centers_xy
        if not self._include_object_layout_obs:
            return self._goal_obs(pipeline_state, goal_xy)

        max_objects = max(1, self._num_hazards, self._num_gremlins)
        hazard_pad = max_objects - self._num_hazards
        gremlin_pad = max_objects - self._num_gremlins
        object_xy = jp.stack(
            [
                jp.pad(hazards_xy, ((0, hazard_pad), (0, 0))),
                jp.pad(gremlins_xy, ((0, gremlin_pad), (0, 0))),
            ],
            axis=0,
        )
        valid = jp.arange(max_objects)[None, :] < jp.asarray(
            [[self._num_hazards], [self._num_gremlins]],
            dtype=jp.int32,
        )

        agent_xy = self._agent_xy(pipeline_state)
        relative_xy = object_xy - agent_xy[None, None, :]
        distance = jp.linalg.norm(relative_xy, axis=-1)
        max_dist = jp.asarray(self._layout_lidar_max_dist, dtype=object_xy.dtype)
        signal = jp.clip(1.0 - distance / jp.maximum(max_dist, 1e-6), 0.0, 1.0)
        signal = jp.where(valid, signal, 0.0)

        angle = jp.mod(jp.arctan2(relative_xy[..., 1], relative_xy[..., 0]), 2.0 * jp.pi)
        bin_idx = jp.floor(angle / (2.0 * jp.pi) * self._layout_lidar_num_bins).astype(jp.int32)
        bin_idx = jp.clip(bin_idx, 0, self._layout_lidar_num_bins - 1)
        bin_mask = jax.nn.one_hot(bin_idx, self._layout_lidar_num_bins, dtype=object_xy.dtype)
        lidar = jp.max(bin_mask * signal[..., None], axis=1)
        return jp.concatenate([lidar.reshape((-1,)), self._goal_obs(pipeline_state, goal_xy)])
