"""Grid-cell placement logic for simplified ant/humanoid goal envs."""

from __future__ import annotations

import jax
from jax import numpy as jp


class GoalGridMixin:
    """Mixin that samples goal-layout objects on fixed square grid cells."""

    def _configure_goal_grid(
        self,
        grid_cell_size: float,
        grid_num_cells: int,
        min_goal_cell_distance: float = 0.0,
    ) -> None:
        if grid_cell_size <= 0.0:
            raise ValueError("grid_cell_size must be positive.")
        if grid_num_cells < 2:
            raise ValueError("grid_num_cells must be at least 2.")
        if min_goal_cell_distance < 0.0:
            raise ValueError("min_goal_cell_distance must be non-negative.")
        self._grid_cell_size = float(grid_cell_size)
        self._grid_num_cells = int(grid_num_cells)
        self._grid_num_total_cells = self._grid_num_cells * self._grid_num_cells
        self._grid_min_goal_cell_distance = float(min_goal_cell_distance)

    def _grid_cells(self) -> jax.Array:
        idx = jp.arange(self._grid_num_cells, dtype=jp.int32)
        xx, yy = jp.meshgrid(idx, idx, indexing="ij")
        return jp.stack([xx.reshape((-1,)), yy.reshape((-1,))], axis=-1)

    def _grid_centers(self) -> jax.Array:
        return self._grid_cells().astype(jp.float32) * jp.asarray(self._grid_cell_size, dtype=jp.float32)

    def _grid_inactive_xy(self) -> jax.Array:
        far = jp.asarray(1000.0 * self._grid_cell_size * self._grid_num_cells, dtype=jp.float32)
        return jp.array([far, far], dtype=jp.float32)

    def _xy_to_grid_cell(self, xy: jax.Array) -> jax.Array:
        cell = jp.floor((xy + 0.5 * self._grid_cell_size) / self._grid_cell_size).astype(jp.int32)
        return jp.clip(cell, 0, self._grid_num_cells - 1)

    def _grid_cell_mask(self, cell: jax.Array, radius_cells: int = 0) -> jax.Array:
        cells = self._grid_cells()
        cheb_dist = jp.max(jp.abs(cells - cell[None, :]), axis=-1)
        return cheb_dist <= radius_cells

    def _grid_inside_mask(self, cells: jax.Array) -> jax.Array:
        return jp.all((cells >= 0) & (cells < self._grid_num_cells), axis=-1)

    def _sample_reset_layout(
        self,
        agent_rng: jax.Array,
        goal_rng: jax.Array,
        hazard_rng: jax.Array,
        obstacle_rng: jax.Array,
        gremlin_rng: jax.Array,
        layout_id: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        del obstacle_rng, gremlin_rng
        if self._relocate_objects_on_reset:
            object_rng = hazard_rng
        elif self._layout_pool_size > 0:
            object_rng = self._layout_rng(layout_id)
        else:
            object_rng = hazard_rng if self._different_object_layout_per_env else None
        hazards_xy, obstacles_xy, gremlin_centers_xy = self._sample_fixed_reset_objects(
            object_rng,
            reserve_fixed_agent_cell=self._fixed_agent_on_reset,
        )
        if self._fixed_agent_on_reset:
            agent_xy = self._sample_agent_xy(self._fixed_agent_rng())
        else:
            agent_xy = self._sample_grid_agent(agent_rng, hazards_xy, obstacles_xy)
        goal_rng = self._fixed_goal_rng() if self._fixed_goal_on_reset else goal_rng
        goal_xy = self._sample_grid_goal(goal_rng, agent_xy, hazards_xy, obstacles_xy)
        return agent_xy, goal_xy, hazards_xy, obstacles_xy, gremlin_centers_xy

    def _sample_fixed_reset_objects(
        self,
        layout_rng: jax.Array | None = None,
        reserve_fixed_agent_cell: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        rng = jax.random.PRNGKey(self._fixed_object_layout_seed) if layout_rng is None else layout_rng
        return self._sample_grid_objects(rng, reserve_fixed_agent_cell=reserve_fixed_agent_cell)

    def _sample_initial_goal(
        self,
        rng: jax.Array,
        agent_xy: jax.Array,
        placed_xy: jax.Array | None = None,
        placed_keepout: jax.Array | None = None,
    ) -> jax.Array:
        del placed_xy, placed_keepout
        hazards_xy = jp.zeros((self._num_hazards, 2), dtype=jp.float32)
        obstacles_xy = jp.zeros((self._num_obstacles, 2), dtype=jp.float32)
        return self._sample_grid_goal(rng, agent_xy, hazards_xy, obstacles_xy)

    def _sample_respawn_goal(
        self,
        rng: jax.Array,
        agent_xy: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
    ) -> jax.Array:
        del gremlin_centers_xy
        return self._sample_grid_goal(rng, agent_xy, hazards_xy, obstacles_xy)

    def _sample_grid_objects(
        self,
        rng: jax.Array,
        reserve_fixed_agent_cell: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        cells = self._grid_cells()
        centers = self._grid_centers()
        origin_occupied = self._grid_cell_mask(jp.array([0, 0], dtype=jp.int32))
        occupied = jp.where(
            reserve_fixed_agent_cell,
            origin_occupied,
            jp.zeros((self._grid_num_total_cells,), dtype=bool),
        )

        hazards_xy, _, _ = self._sample_grid_statics(
            rng,
            self._num_hazards,
            cells,
            centers,
            occupied,
            jp.zeros((self._grid_num_total_cells,), dtype=bool),
        )
        obstacles_xy = jp.zeros((0, 2), dtype=centers.dtype)
        gremlin_centers_xy = jp.zeros((0, 2), dtype=centers.dtype)
        return self._canonicalize_layout_order(hazards_xy, obstacles_xy, gremlin_centers_xy)

    def _sample_grid_agent(
        self,
        rng: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
    ) -> jax.Array:
        centers = self._grid_centers()
        occupied = self._grid_occupied_from_layout(
            self._grid_inactive_xy(),
            hazards_xy,
            obstacles_xy,
            include_agent=False,
        )
        valid = ~occupied
        priorities = jax.random.uniform(rng, (self._grid_num_total_cells,))
        scores = jp.where(valid, priorities, -jp.inf)
        idx = jp.argmax(scores)
        has_valid = jp.any(valid)
        return centers[jp.where(has_valid, idx, 0)]

    def _sample_grid_statics(
        self,
        rng: jax.Array,
        num_objects: int,
        cells: jax.Array,
        centers: jax.Array,
        occupied: jax.Array,
        static_forbidden: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if num_objects == 0:
            return jp.zeros((0, 2), dtype=centers.dtype), occupied, static_forbidden
        priorities = jax.random.uniform(rng, (num_objects, self._grid_num_total_cells))
        objects = jp.zeros((num_objects, 2), dtype=centers.dtype)

        for i in range(num_objects):
            valid = (~occupied) & (~static_forbidden)
            scores = jp.where(valid, priorities[i], -jp.inf)
            idx = jp.argmax(scores)
            has_valid = jp.any(valid)
            cell = cells[idx]
            same_cell = self._grid_cell_mask(cell)
            objects = objects.at[i].set(jp.where(has_valid, centers[idx], self._grid_inactive_xy()))
            occupied = occupied | (same_cell & has_valid)
        return objects, occupied, static_forbidden

    def _sample_grid_goal(
        self,
        rng: jax.Array,
        agent_xy: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
    ) -> jax.Array:
        centers = self._grid_centers()
        occupied = self._grid_occupied_from_layout(agent_xy, hazards_xy, obstacles_xy)
        priorities = jax.random.uniform(rng, (self._grid_num_total_cells,))
        min_dist = jp.asarray(
            self._grid_min_goal_cell_distance * self._grid_cell_size,
            dtype=centers.dtype,
        )
        far_enough = jp.linalg.norm(centers - agent_xy[None, :], axis=-1) >= min_dist
        valid = (~occupied) & far_enough
        scores = jp.where(valid, priorities, -jp.inf)
        valid_idx = jp.argmax(scores)
        has_valid = jp.any(valid)
        fallback_dist = jp.linalg.norm(centers - agent_xy[None, :], axis=-1)
        fallback_idx = jp.argmax(fallback_dist)
        idx = jp.where(has_valid, valid_idx, fallback_idx)
        return centers[idx]

    def _grid_occupied_from_layout(
        self,
        agent_xy: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        include_agent: bool = True,
    ) -> jax.Array:
        cells = self._grid_cells()
        occupied = jp.where(
            include_agent,
            self._grid_cell_mask(self._xy_to_grid_cell(agent_xy)),
            jp.zeros((self._grid_num_total_cells,), dtype=bool),
        )

        def object_cell_mask(xy):
            cell_float = (xy + 0.5 * self._grid_cell_size) / self._grid_cell_size
            cell = jp.floor(cell_float).astype(jp.int32)
            valid = jp.all((cell >= 0) & (cell < self._grid_num_cells))
            return self._grid_cell_mask(jp.clip(cell, 0, self._grid_num_cells - 1)) & valid

        for xy_set in (hazards_xy, obstacles_xy):
            if xy_set.shape[0] > 0:
                occupied = occupied | jp.any(jax.vmap(object_cell_mask)(xy_set), axis=0)

        return occupied

    def _gremlin_positions(self, centers_xy: jax.Array, time: jax.Array) -> jax.Array:
        del time
        return jp.zeros((0, 2), dtype=centers_xy.dtype)

    def _layout_obs(
        self,
        pipeline_state,
        goal_xy: jax.Array,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlins_xy: jax.Array,
        gremlin_centers_xy: jax.Array,
    ) -> jax.Array:
        del obstacles_xy, gremlins_xy, gremlin_centers_xy
        agent_xy = self._agent_xy(pipeline_state)
        hazard_lidar = self._global_lidar(hazards_xy, agent_xy, self._hazard_radius, pipeline_state)
        return jp.concatenate([hazard_lidar, self._goal_obs(pipeline_state, goal_xy)])

    def _costs(
        self,
        pipeline_state,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlins_xy: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del obstacles_xy, gremlins_xy
        zero = jp.asarray(0.0, dtype=hazards_xy.dtype)
        agent_xy = self._agent_xy(pipeline_state)
        cost_hazards = self._center_disk_cost(agent_xy, hazards_xy, self._hazard_radius)
        return cost_hazards, zero, zero

    def _object_boundary_cost(
        self,
        pipeline_state,
        hazards_xy: jax.Array,
        obstacles_xy: jax.Array,
        gremlins_xy: jax.Array,
    ) -> jax.Array:
        del obstacles_xy, gremlins_xy
        agent_xy = self._agent_xy(pipeline_state)
        return self._center_disk_cost(agent_xy, hazards_xy, self._hazard_radius, margin=0.0)
