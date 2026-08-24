"""Geometry-aware hazard placement for grid mazes.

This module is independent from the Brax environments.  It samples static
ghost-hazard centers and conservatively verifies that the reset point remains
connected to every goal point in root-configuration space.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


_FOUR_CONNECTED = np.asarray(
    [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8
)


@dataclass(frozen=True)
class HazardLayout:
    """A sampled, passability-preserving hazard layout."""

    hazards_xy: np.ndarray
    hazard_cells: np.ndarray
    num_hazards: int


@dataclass(frozen=True)
class _Raster:
    x: np.ndarray
    y: np.ndarray
    resolution: float
    discretization_margin: float
    blocked_by_walls: np.ndarray
    reset_index: tuple[int, int]
    goal_indices: tuple[tuple[int, int], ...]


def _is_marker(value: object, marker: str) -> bool:
    return isinstance(value, str) and value.lower() == marker


def _is_wall(value: object) -> bool:
    return not isinstance(value, str) and value == 1


def _validate_layout(
    maze_layout: Sequence[Sequence[object]],
) -> tuple[tuple[tuple[object, ...], ...], tuple[int, int], tuple[tuple[int, int], ...]]:
    if len(maze_layout) == 0:
        raise ValueError("maze_layout must contain at least one row.")

    rows = tuple(tuple(row) for row in maze_layout)
    num_columns = len(rows[0])
    if num_columns == 0 or any(len(row) != num_columns for row in rows):
        raise ValueError("maze_layout must be a non-empty rectangular grid.")

    resets = []
    goals = []
    for row_index, row in enumerate(rows):
        for column_index, value in enumerate(row):
            if _is_marker(value, "r"):
                resets.append((row_index, column_index))
            elif _is_marker(value, "g"):
                goals.append((row_index, column_index))

    if len(resets) != 1:
        raise ValueError(
            "maze_layout must contain exactly one reset cell ('r' or 'R'); "
            f"found {len(resets)}."
        )
    if not goals:
        raise ValueError("maze_layout must contain at least one goal cell ('g' or 'G').")

    return rows, resets[0], tuple(goals)


def _coordinate_index(coordinates: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(coordinates, value, side="left"))
    if index == len(coordinates):
        return len(coordinates) - 1
    if index > 0 and abs(coordinates[index - 1] - value) <= abs(
        coordinates[index] - value
    ):
        return index - 1
    return index


def _paint_expanded_box(
    blocked: np.ndarray,
    raster_x: np.ndarray,
    raster_y: np.ndarray,
    center_x: float,
    center_y: float,
    half_size: float,
    padding: float,
) -> None:
    x_start = int(
        np.searchsorted(raster_x, center_x - half_size - padding, side="left")
    )
    x_stop = int(
        np.searchsorted(raster_x, center_x + half_size + padding, side="right")
    )
    y_start = int(
        np.searchsorted(raster_y, center_y - half_size - padding, side="left")
    )
    y_stop = int(
        np.searchsorted(raster_y, center_y + half_size + padding, side="right")
    )
    if x_start >= x_stop or y_start >= y_stop:
        return

    local_x = raster_x[x_start:x_stop, None]
    local_y = raster_y[None, y_start:y_stop]
    distance_x = np.maximum(np.abs(local_x - center_x) - half_size, 0.0)
    distance_y = np.maximum(np.abs(local_y - center_y) - half_size, 0.0)
    local_blocked = distance_x * distance_x + distance_y * distance_y <= padding**2
    blocked[x_start:x_stop, y_start:y_stop] |= local_blocked


def _paint_disk(
    blocked: np.ndarray,
    raster_x: np.ndarray,
    raster_y: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float,
) -> None:
    x_start = int(np.searchsorted(raster_x, center_x - radius, side="left"))
    x_stop = int(np.searchsorted(raster_x, center_x + radius, side="right"))
    y_start = int(np.searchsorted(raster_y, center_y - radius, side="left"))
    y_stop = int(np.searchsorted(raster_y, center_y + radius, side="right"))
    if x_start >= x_stop or y_start >= y_stop:
        return

    delta_x = raster_x[x_start:x_stop, None] - center_x
    delta_y = raster_y[None, y_start:y_stop] - center_y
    local_blocked = delta_x * delta_x + delta_y * delta_y <= radius**2
    blocked[x_start:x_stop, y_start:y_stop] |= local_blocked


def _is_passable(blocked: np.ndarray, raster: _Raster) -> bool:
    # SciPy is needed only when constructing a safe-maze hazard layout.  Keep
    # it lazy so importing unrelated SafeNav environments does not require it.
    from scipy import ndimage

    reset_x, reset_y = raster.reset_index
    if blocked[reset_x, reset_y]:
        return False

    labels, _ = ndimage.label(~blocked, structure=_FOUR_CONNECTED)
    reset_label = labels[reset_x, reset_y]
    if reset_label == 0:
        return False
    return all(labels[goal_index] == reset_label for goal_index in raster.goal_indices)


def _build_raster(
    maze_layout: tuple[tuple[object, ...], ...],
    reset_cell: tuple[int, int],
    goal_cells: tuple[tuple[int, int], ...],
    *,
    cell_size: float,
    robot_clearance: float,
    raster_samples_per_cell: int,
) -> _Raster:
    resolution = cell_size / raster_samples_per_cell
    discretization_margin = math.sqrt(2.0) * resolution / 2.0
    half_size = cell_size / 2.0
    num_rows = len(maze_layout)
    num_columns = len(maze_layout[0])

    raster_x = -half_size + (
        np.arange(num_rows * raster_samples_per_cell) + 0.5
    ) * resolution
    raster_y = -half_size + (
        np.arange(num_columns * raster_samples_per_cell) + 0.5
    ) * resolution
    blocked = np.zeros((len(raster_x), len(raster_y)), dtype=bool)
    wall_padding = robot_clearance + discretization_margin

    for row_index, row in enumerate(maze_layout):
        for column_index, value in enumerate(row):
            if _is_wall(value):
                _paint_expanded_box(
                    blocked,
                    raster_x,
                    raster_y,
                    row_index * cell_size,
                    column_index * cell_size,
                    half_size,
                    wall_padding,
                )

    def cell_center_index(cell: tuple[int, int]) -> tuple[int, int]:
        return (
            _coordinate_index(raster_x, cell[0] * cell_size),
            _coordinate_index(raster_y, cell[1] * cell_size),
        )

    return _Raster(
        x=raster_x,
        y=raster_y,
        resolution=resolution,
        discretization_margin=discretization_margin,
        blocked_by_walls=blocked,
        reset_index=cell_center_index(reset_cell),
        goal_indices=tuple(cell_center_index(cell) for cell in goal_cells),
    )


def _empty_layout() -> HazardLayout:
    return HazardLayout(
        hazards_xy=np.empty((0, 2), dtype=np.float32),
        hazard_cells=np.empty((0, 2), dtype=np.int32),
        num_hazards=0,
    )


def sample_passable_hazards(
    maze_layout: Sequence[Sequence[object]],
    *,
    cell_size: float,
    robot_clearance: float,
    hazard_radius: float,
    robot_cost_margin: float,
    hazard_fraction: float,
    rng: np.random.Generator,
    candidate_samples_per_axis: int = 21,
    raster_samples_per_cell: int = 40,
    max_layout_attempts: int = 16,
) -> HazardLayout:
    """Samples up to a requested number of hazards without disconnecting goals.

    Maze cell centers follow the Scaling-CRL convention ``(row * cell_size,
    column * cell_size)``.  A hazard blocks root positions within
    ``hazard_radius + robot_cost_margin``.  Wall and hazard masks receive an
    additional half-raster-diagonal margin, making the raster test
    conservative with respect to discretization.

    Args:
        maze_layout: Rectangular grid using ``1`` for walls, one ``r`` reset,
            and one or more ``g`` goals. Other values are traversable.
        cell_size: Width and height of each maze cell in world units.
        robot_clearance: Root-space clearance required around wall boxes.
        hazard_radius: Radius of each ghost hazard.
        robot_cost_margin: Additional root-space hazard margin.
        hazard_fraction: Fraction of goal cells used to determine the requested
            maximum count: ``floor(hazard_fraction * number_of_goals)``.
        rng: Explicit NumPy random generator controlling all sampling.
        candidate_samples_per_axis: Number of strictly interior candidate
            center coordinates per cell axis.
        raster_samples_per_cell: Connectivity raster resolution per cell axis.
        max_layout_attempts: Number of randomized greedy layout attempts. If
            none reaches the requested count, the best passable layout found
            within this budget is returned.

    Returns:
        The first ``HazardLayout`` reaching the requested count, or the layout
        with the most hazards found within ``max_layout_attempts``.

    Raises:
        ValueError: If inputs are invalid or the maze is not initially
            passable before hazards are added.
    """
    rows, reset_cell, goal_cells = _validate_layout(maze_layout)

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be an instance of numpy.random.Generator.")
    if not math.isfinite(cell_size) or cell_size <= 0.0:
        raise ValueError("cell_size must be finite and positive.")
    if not math.isfinite(robot_clearance) or robot_clearance < 0.0:
        raise ValueError("robot_clearance must be finite and non-negative.")
    if not math.isfinite(hazard_radius) or hazard_radius <= 0.0:
        raise ValueError("hazard_radius must be finite and positive.")
    if not math.isfinite(robot_cost_margin) or robot_cost_margin < 0.0:
        raise ValueError("robot_cost_margin must be finite and non-negative.")
    if not math.isfinite(hazard_fraction) or not 0.0 <= hazard_fraction <= 1.0:
        raise ValueError("hazard_fraction must be finite and in [0, 1].")
    if (
        not isinstance(candidate_samples_per_axis, int)
        or candidate_samples_per_axis <= 0
    ):
        raise ValueError("candidate_samples_per_axis must be a positive integer.")
    if (
        not isinstance(raster_samples_per_cell, int)
        or raster_samples_per_cell <= 0
    ):
        raise ValueError("raster_samples_per_cell must be a positive integer.")
    if not isinstance(max_layout_attempts, int) or max_layout_attempts <= 0:
        raise ValueError("max_layout_attempts must be a positive integer.")

    target_count = math.floor(hazard_fraction * len(goal_cells))
    candidate_cells = np.asarray(
        [
            (row_index, column_index)
            for row_index, row in enumerate(rows)
            for column_index, value in enumerate(row)
            if not _is_wall(value) and (row_index, column_index) != reset_cell
        ],
        dtype=np.int32,
    ).reshape((-1, 2))

    if target_count > len(candidate_cells):
        raise ValueError(
            f"Requested {target_count} hazards, but only {len(candidate_cells)} "
            "distinct non-wall, non-reset cells are available."
        )

    raster = _build_raster(
        rows,
        reset_cell,
        goal_cells,
        cell_size=cell_size,
        robot_clearance=robot_clearance,
        raster_samples_per_cell=raster_samples_per_cell,
    )
    if not _is_passable(raster.blocked_by_walls, raster):
        raise ValueError(
            "The reset cell is not connected to every goal cell after applying "
            "the requested robot wall clearance."
        )
    if target_count == 0:
        return _empty_layout()

    half_size = cell_size / 2.0
    axis_offsets = np.linspace(
        -half_size,
        half_size,
        candidate_samples_per_axis + 2,
        dtype=np.float64,
    )[1:-1]
    candidate_offsets = np.stack(
        np.meshgrid(axis_offsets, axis_offsets, indexing="ij"), axis=-1
    ).reshape((-1, 2))
    effective_hazard_radius = (
        hazard_radius + robot_cost_margin + raster.discretization_margin
    )
    best_xy: list[np.ndarray] = []
    best_cells: list[np.ndarray] = []

    for _ in range(max_layout_attempts):
        blocked = raster.blocked_by_walls.copy()
        accepted_xy = []
        accepted_cells = []

        for candidate_index in rng.permutation(len(candidate_cells)):
            cell = candidate_cells[candidate_index]
            offset = candidate_offsets[rng.integers(len(candidate_offsets))]
            center = cell.astype(np.float64) * cell_size + offset

            trial_blocked = blocked.copy()
            _paint_disk(
                trial_blocked,
                raster.x,
                raster.y,
                float(center[0]),
                float(center[1]),
                effective_hazard_radius,
            )
            if not _is_passable(trial_blocked, raster):
                continue

            blocked = trial_blocked
            accepted_xy.append(center)
            accepted_cells.append(cell.copy())
            if len(accepted_xy) == target_count:
                return HazardLayout(
                    hazards_xy=np.asarray(accepted_xy, dtype=np.float32),
                    hazard_cells=np.asarray(accepted_cells, dtype=np.int32),
                    num_hazards=target_count,
                )

        if len(accepted_xy) > len(best_xy):
            best_xy = accepted_xy
            best_cells = accepted_cells

    return HazardLayout(
        hazards_xy=np.asarray(best_xy, dtype=np.float32).reshape((-1, 2)),
        hazard_cells=np.asarray(best_cells, dtype=np.int32).reshape((-1, 2)),
        num_hazards=len(best_xy),
    )


__all__ = ["HazardLayout", "sample_passable_hazards"]
