"""Headless point push environment."""

from __future__ import annotations

from safenav_jax.envs.point_push import PointPushBase


class PointPushHeadless(PointPushBase):
    """Point push with task objects represented by arrays."""

    def __init__(self, **kwargs):
        super().__init__(render_layout=False, **kwargs)
