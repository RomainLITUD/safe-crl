"""Shared Humanoid actuator configuration."""

from __future__ import annotations

from brax import base
from jax import numpy as jp


SPRING_HUMANOID_ACTUATOR_GEAR = (350.0,) * 11 + (100.0,) * 6


def apply_spring_humanoid_gear_for_mjx(
    sys: base.System,
    *,
    backend: str,
    enabled: bool,
) -> base.System:
    """Optionally applies Brax's Spring Humanoid gears to an MJX system."""
    if backend != "mjx" or not enabled:
        return sys

    expected = len(SPRING_HUMANOID_ACTUATOR_GEAR)
    actual = int(sys.act_size())
    if actual != expected:
        raise ValueError(
            "Spring Humanoid gear override requires "
            f"{expected} actuators, but the loaded system has {actual}."
        )

    gear = jp.asarray(
        SPRING_HUMANOID_ACTUATOR_GEAR,
        dtype=sys.actuator.gear.dtype,
    )
    if gear.shape != sys.actuator.gear.shape:
        raise ValueError(
            "Spring Humanoid gear override has shape "
            f"{gear.shape}, but the loaded actuator gear has shape "
            f"{sys.actuator.gear.shape}."
        )
    return sys.replace(actuator=sys.actuator.replace(gear=gear))


__all__ = [
    "SPRING_HUMANOID_ACTUATOR_GEAR",
    "apply_spring_humanoid_gear_for_mjx",
]
