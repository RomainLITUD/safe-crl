import jax
import flax
import functools
import jax.numpy as jnp

from jax import flatten_util
from brax.training.types import PRNGKey

@flax.struct.dataclass
class ReplayBufferState:
  """Contains data related to a replay buffer."""

  data: jnp.ndarray
  insert_position: jnp.ndarray
  sample_position: jnp.ndarray
  key: PRNGKey

class TrajectoryUniformSamplingQueue():
    """
    Base class for limited-size FIFO reply buffers.

    Implements an `insert()` method which behaves like a limited-size queue.
    I.e. it adds samples to the end of the queue and, if necessary, removes the
    oldest samples form the queue in order to keep the maximum size within the
    specified limit.

    Derived classes must implement the `sample()` method.
    """
    def __init__(
        self,
        max_replay_size: int,
        dummy_data_sample,
        sample_batch_size: int,
        num_envs: int,
        episode_length: int,
    ):

        self._flatten_fn = jax.vmap(jax.vmap(lambda x: flatten_util.ravel_pytree(x)[0]))
        dummy_flatten, self._unflatten_fn = flatten_util.ravel_pytree(dummy_data_sample)
        self._unflatten_fn = jax.vmap(jax.vmap(self._unflatten_fn))
        data_size = len(dummy_flatten)
        print(f"data_size: {data_size}", flush=True)

        self._data_shape = (max_replay_size, num_envs, data_size)
        self._data_dtype = dummy_flatten.dtype
        self._sample_batch_size = sample_batch_size
        self._size = 0
        self.num_envs = num_envs
        self.episode_length = episode_length

    def init(self, key):
        return ReplayBufferState(
            data=jnp.zeros(self._data_shape, self._data_dtype),
            sample_position=jnp.zeros((), jnp.int32),
            insert_position=jnp.zeros((), jnp.int32),
            key=key,
        )

    def insert(self, buffer_state, samples):
        """Insert data into the replay buffer."""
        self.check_can_insert(buffer_state, samples, 1)
        return self.insert_internal(buffer_state, samples)
    
    def check_can_insert(self, buffer_state, samples, shards):
        """Checks whether insert operation can be performed."""
        assert isinstance(shards, int), "This method should not be JITed."
        insert_size = jax.tree_util.tree_flatten(samples)[0][0].shape[0] // shards
        if self._data_shape[0] < insert_size:
            raise ValueError(
                "Trying to insert a batch of samples larger than the maximum replay"
                f" size. num_samples: {insert_size}, max replay size"
                f" {self._data_shape[0]}"
            )
        self._size = min(self._data_shape[0], self._size + insert_size)

    def check_can_sample(self, buffer_state, shards):
        """Checks whether sampling can be performed. Do not JIT this method."""
        if self._size < self.episode_length:
            raise ValueError(
                "Not enough samples in replay buffer to sample a full sequence. "
                f"size: {self._size}, episode_length: {self.episode_length}"
            )

    def insert_internal(
        self, buffer_state, samples
    ):
        """Insert data in the replay buffer.

        Args:
          buffer_state: Buffer state
          samples: Sample to insert with a leading batch size.

        Returns:
          New buffer state.
        """
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"buffer_state.data.shape ({buffer_state.data.shape}) "
                f"doesn't match the expected value ({self._data_shape})"
            )

        update = self._flatten_fn(samples) #Updates has shape (unroll_len, num_envs, self._data_shape[-1])
        data = buffer_state.data #shape = (max_replay_size, num_envs, data_size)

        # If needed, roll the buffer to make sure there's enough space to fit
        # `update` after the current position.
        position = buffer_state.insert_position
        roll = jnp.minimum(0, len(data) - position - len(update))
        data = jax.lax.cond(roll, lambda: jnp.roll(data, roll, axis=0), lambda: data)
        position = position + roll

        # Update the buffer and the control numbers.
        data = jax.lax.dynamic_update_slice_in_dim(data, update, position, axis=0)
        position = (position + len(update)) % (len(data) + 1)    # so whenever roll happens, position becomes len(data), else it is increased by len(update), what is the use of doing % (len(data) + 1)??
        sample_position = jnp.maximum(0, buffer_state.sample_position + roll) #what is the use of this line? sample_position always remains 0 as roll can never be positive

        return buffer_state.replace(
            data=data,
            insert_position=position,
            sample_position=sample_position,
        )

    def sample(self, buffer_state):
        """Sample a batch of data."""
        # Do not use the Python-side `_size` here: inserts run inside JIT, so
        # host bookkeeping can be stale even when `buffer_state` has enough
        # samples. Training validates and prefills enough rows before sampling.
        return self.sample_internal(buffer_state)

    def sample_internal(self, buffer_state):
        if buffer_state.data.shape != self._data_shape:
            raise ValueError(
                f"Data shape expected by the replay buffer ({self._data_shape}) does "
                f"not match the shape of the buffer state ({buffer_state.data.shape})"
            )
        key, sample_key, shuffle_key = jax.random.split(buffer_state.key, 3)
        # Note: this is the number of envs to sample but it can be modified if there is OOM
        shape = self.num_envs

        # Sampling envs idxs
        envs_idxs = jax.random.choice(sample_key, jnp.arange(self.num_envs), shape=(shape,), replace=False)

        @functools.partial(jax.jit, static_argnames=("rows", "cols"))
        def create_matrix(rows, cols, min_val, max_val, rng_key):
            rng_key, subkey = jax.random.split(rng_key)
            start_values = jax.random.randint(subkey, shape=(rows,), minval=min_val, maxval=max_val)
            row_indices = jnp.arange(cols)
            matrix = start_values[:, jnp.newaxis] + row_indices
            return matrix

        @jax.jit
        def create_batch(arr_2d, indices):
            return jnp.take(arr_2d, indices, axis=0, mode="wrap")

        create_batch_vmaped = jax.vmap(create_batch, in_axes=(1, 0))

        matrix = create_matrix(
            shape,
            self.episode_length,
            buffer_state.sample_position,
            buffer_state.insert_position - self.episode_length + 1,
            sample_key,
        )

        '''
        The function create_batch will be called for every envs_idxs of buffer_state.data and every row of matrix.
        Because every row of matrix has consecutive indices of self.episode_length, for every
        envs_idx of envs_idxs, we will sample a random self.episode_length length sequence from 
        buffer_state.data[:, envs_idx, :]. But I don't think the code ensures that this sequence 
        won't be across episodes?

        flatten_crl_fn takes care of this
        '''
        batch = create_batch_vmaped(buffer_state.data[:, envs_idxs, :], matrix)
        transitions = self._unflatten_fn(batch)
        extras = dict(transitions.extras)
        state_extras = dict(extras["state_extras"])
        state_extras["env_index"] = jnp.broadcast_to(
            envs_idxs[:, None],
            state_extras["seed"].shape,
        ).astype(jnp.int32)
        extras["state_extras"] = state_extras
        transitions = transitions._replace(extras=extras)
        return buffer_state.replace(key=key), transitions

    @staticmethod
    def _goal_segment_ids(seed_seq, task_goal_xy_seq, atol=1e-6):
        """Labels contiguous action segments with one unchanged task goal."""
        goal_finite = jnp.all(jnp.isfinite(task_goal_xy_seq), axis=-1)
        same_goal = jnp.all(
            jnp.abs(task_goal_xy_seq[..., 1:, :] - task_goal_xy_seq[..., :-1, :])
            <= jnp.asarray(atol, dtype=task_goal_xy_seq.dtype),
            axis=-1,
        )
        same_goal = same_goal & goal_finite[..., 1:] & goal_finite[..., :-1]
        same_episode = seed_seq[..., 1:] == seed_seq[..., :-1]
        boundary = ~(same_goal & same_episode)
        boundary = jnp.concatenate(
            [jnp.zeros_like(seed_seq[..., :1], dtype=bool), boundary],
            axis=-1,
        )
        return jnp.cumsum(boundary.astype(jnp.int32), axis=-1)

    @staticmethod
    def _goal_segment_future_mask(seed_seq, task_goal_xy_seq, anchor_indices=None):
        """Returns futures reached before any action uses a respawned goal.

        A goal change visible in state j happens after the action at j - 1, so
        state j remains a valid endpoint for the preceding goal segment.
        """
        segment_ids = TrajectoryUniformSamplingQueue._goal_segment_ids(
            seed_seq,
            task_goal_xy_seq,
        )
        endpoint_action_segment = jnp.concatenate(
            [segment_ids[..., :1], segment_ids[..., :-1]],
            axis=-1,
        )
        seq_len = seed_seq.shape[-1]
        arrangement = jnp.arange(seq_len)
        if anchor_indices is None:
            if seed_seq.ndim != 1:
                raise ValueError("anchor_indices are required for batched goal sequences.")
            return (
                (arrangement[None, :] > arrangement[:, None])
                & (seed_seq[None, :] == seed_seq[:, None])
                & (endpoint_action_segment[None, :] == segment_ids[:, None])
            )

        anchor_seed = jnp.take_along_axis(seed_seq, anchor_indices[:, None], axis=1)
        anchor_segment = jnp.take_along_axis(
            segment_ids,
            anchor_indices[:, None],
            axis=1,
        )
        return (
            (arrangement[None, :] > anchor_indices[:, None])
            & (seed_seq == anchor_seed)
            & (endpoint_action_segment == anchor_segment)
        )

    @staticmethod
    def _global_xy_to_anchor_ego(global_xy, anchor_xy, anchor_yaw):
        if global_xy.shape[-1] == 4:
            xy_shape = global_xy.shape[:-1] + (2, 2)
            global_xy_pairs = jnp.reshape(global_xy, xy_shape)
            anchor_xy_pairs = anchor_xy[..., None, :]
            if global_xy.ndim == 3:
                anchor_xy_pairs = anchor_xy[:, None, None, :]
            rel_xy = global_xy_pairs - anchor_xy_pairs
            cos_yaw = jnp.cos(anchor_yaw)
            sin_yaw = jnp.sin(anchor_yaw)
            if global_xy.ndim == 3:
                cos_yaw = cos_yaw[:, None, None]
                sin_yaw = sin_yaw[:, None, None]
            else:
                cos_yaw = cos_yaw[..., None]
                sin_yaw = sin_yaw[..., None]
            local_x = cos_yaw * rel_xy[..., 0] + sin_yaw * rel_xy[..., 1]
            local_y = -sin_yaw * rel_xy[..., 0] + cos_yaw * rel_xy[..., 1]
            return jnp.reshape(jnp.stack([local_x, local_y], axis=-1), global_xy.shape)
        rel_xy = global_xy - anchor_xy[..., None, :] if global_xy.ndim == 3 else global_xy - anchor_xy
        cos_yaw = jnp.cos(anchor_yaw)
        sin_yaw = jnp.sin(anchor_yaw)
        if global_xy.ndim == 3:
            cos_yaw = cos_yaw[:, None]
            sin_yaw = sin_yaw[:, None]
        local_x = cos_yaw * rel_xy[..., 0] + sin_yaw * rel_xy[..., 1]
        local_y = -sin_yaw * rel_xy[..., 0] + cos_yaw * rel_xy[..., 1]
        return jnp.stack([local_x, local_y], axis=-1)

    @staticmethod
    def _ego_xy_to_lidar(ego_xy, num_bins, max_dist):
        dist = jnp.linalg.norm(ego_xy, axis=-1)
        max_dist = jnp.asarray(max_dist, dtype=ego_xy.dtype)
        signal = jnp.clip(1.0 - dist / jnp.maximum(max_dist, 1e-8), 0.0, 1.0)
        angle = jnp.mod(jnp.arctan2(ego_xy[..., 1], ego_xy[..., 0]), 2.0 * jnp.pi)
        bin_idx = jnp.floor(angle / (2.0 * jnp.pi) * num_bins).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, num_bins - 1)
        return jax.nn.one_hot(bin_idx, num_bins, dtype=ego_xy.dtype) * signal[..., None]

    @staticmethod
    def _project_relabel_goal(global_xy, anchor_xy, anchor_yaw, use_ego_goal_relabel, use_goal_lidar, num_bins, max_dist):
        if use_ego_goal_relabel:
            if global_xy.shape[-1] == 1:
                return global_xy - anchor_xy[..., :1]
            goal = TrajectoryUniformSamplingQueue._global_xy_to_anchor_ego(global_xy, anchor_xy, anchor_yaw)
            if use_goal_lidar:
                goal = TrajectoryUniformSamplingQueue._ego_xy_to_lidar(goal, num_bins, max_dist)
            return goal
        return global_xy

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config"))
    def flatten_crl_fn(buffer_config, transition, sample_key):

        (
            gamma,
            obs_dim,
            goal_start_idx,
            goal_end_idx,
            use_ego_goal_relabel,
            use_goal_lidar,
            goal_lidar_num_bins,
            goal_lidar_max_dist,
        ) = buffer_config

        # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
        seq_len = transition.observation.shape[0]
        arrangement = jnp.arange(seq_len)
        is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32) # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
        discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)        
        probs = is_future_mask * discount  
        # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
        #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
        #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
        #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
        #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
        #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
        # assuming seq_len = 5
        # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))
        
        seed_seq = transition.extras["state_extras"]["seed"]
        task_goal_xy_seq = transition.extras["state_extras"]["task_goal_xy"]
        candidate_mask = TrajectoryUniformSamplingQueue._goal_segment_future_mask(
            seed_seq,
            task_goal_xy_seq,
        )
        # Match original Scaling-CRL relabeling: every anchor except the final
        # sampled transition is kept. If no same-episode candidate exists, the
        # small diagonal mass below samples the anchor itself as a fallback.
        future_valid = jnp.ones((seq_len,), dtype=bool)
        probs = jnp.where(candidate_mask, probs, 0.0) + jnp.eye(seq_len) * 1e-5
        #ith row of probs will be non zero only for time indices that 
        # 1) are greater than i
        # 2) have the same seed as the ith time index

        real_goal_index = jax.random.categorical(sample_key, jnp.log(probs))
        goal_index = real_goal_index[:-1]
        anchor_indices = arrangement[:-1]
        relabel_horizon = jnp.maximum(goal_index - anchor_indices, 0)
        future_valid_rows = future_valid[:-1]

        sampled_future_state = jnp.take(transition.observation, goal_index, axis=0) #the last goal_index cannot be considered as there is no future.
        future_action = jnp.take(transition.action, goal_index, axis=0)
        if use_ego_goal_relabel:
            achieved_goal_seq = transition.extras["state_extras"]["achieved_goal"]
            relabel_anchor_xy_seq = transition.extras["state_extras"]["relabel_anchor_xy"]
            agent_yaw_seq = transition.extras["state_extras"]["agent_yaw"]
            future_goal_global = jnp.take(achieved_goal_seq, goal_index, axis=0)
            anchor_relabel_xy = relabel_anchor_xy_seq[:-1]
            anchor_yaw = agent_yaw_seq[:-1]
            goal = TrajectoryUniformSamplingQueue._project_relabel_goal(
                future_goal_global,
                anchor_relabel_xy,
                anchor_yaw,
                use_ego_goal_relabel,
                use_goal_lidar,
                goal_lidar_num_bins,
                goal_lidar_max_dist,
            )
        else:
            goal = sampled_future_state[:, goal_start_idx : goal_end_idx]
        future_state = sampled_future_state[:, : obs_dim]
        state = transition.observation[:-1, : obs_dim] #all states are considered
        # BASICALLY HERE, for each state in the 1000 time-steps, we are creating a new observation by 
        # appending the goal to the state (where the goal is extracted from the future state, which
        # is sampled with geometric of gamma of the same trajectory)

        state_extras = {
            "seed": jnp.squeeze(transition.extras["state_extras"]["seed"][:-1]),
            "agent_yaw": jnp.squeeze(transition.extras["state_extras"]["agent_yaw"][:-1]),
            "task_goal": jnp.squeeze(transition.extras["state_extras"]["task_goal"][:-1]),
            "task_goal_xy": jnp.squeeze(transition.extras["state_extras"]["task_goal_xy"][:-1]),
            "achieved_goal": jnp.squeeze(transition.extras["state_extras"]["achieved_goal"][:-1]),
            "relabel_anchor_xy": jnp.squeeze(transition.extras["state_extras"]["relabel_anchor_xy"][:-1]),
        }
        extras = {
            "policy_extras": {},
            "state_extras": state_extras,
            "state": state,
            "future_state": future_state,
            "future_action": future_action,
            "relabel_horizon": relabel_horizon,
            "future_valid": future_valid_rows,
        }
        new_obs = jnp.concatenate([state, goal], axis=1)
        return transition._replace(
            observation=jnp.squeeze(new_obs),   #this has shape (num_envs, episode_length-1, obs_size)
            action=jnp.squeeze(transition.action[:-1]),
            reward=jnp.squeeze(transition.reward[:-1]),
            discount=jnp.squeeze(transition.discount[:-1]),
            extras=extras,
        )

    @staticmethod
    def _sample_survival_targets(gamma, transition, sample_key):
        """Samples geometric horizons and labels observed episode survival."""
        discount = jnp.asarray(transition.discount)
        truncation = jnp.asarray(
            transition.extras["state_extras"]["truncation"]
        ).astype(bool)
        seq_len = discount.shape[0]
        anchor_indices = jnp.arange(seq_len - 1, dtype=jnp.int32)

        gamma = jnp.asarray(gamma, dtype=jnp.float32)
        uniform = jax.random.uniform(
            sample_key,
            shape=(seq_len - 1,),
            minval=0.0,
            maxval=1.0,
            dtype=jnp.float32,
        )
        raw_horizon = jnp.floor(jnp.log1p(-uniform) / jnp.log(gamma)) + 1.0
        horizon = jnp.minimum(
            raw_horizon,
            jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=raw_horizon.dtype),
        ).astype(jnp.int32)

        done = discount <= 0.0
        transition_indices = jnp.arange(seq_len, dtype=jnp.int32)
        boundary_indices = jnp.where(done, transition_indices, seq_len)
        next_boundary = jax.lax.associative_scan(
            jnp.minimum,
            boundary_indices,
            reverse=True,
        )[:-1]
        clipped_boundary = jnp.minimum(next_boundary, seq_len - 1)
        boundary_distance = next_boundary - anchor_indices + 1
        boundary_within_horizon = (
            (next_boundary < seq_len) & (boundary_distance <= horizon)
        )
        boundary_is_truncation = truncation[clipped_boundary]

        # A known unsafe boundary determines label zero even when H extends
        # past the retained suffix.  Otherwise H must be fully observable.
        horizon_observed = horizon <= (seq_len - anchor_indices)
        survival_valid = jnp.where(
            boundary_within_horizon,
            ~boundary_is_truncation,
            horizon_observed,
        )
        survival_label = (~boundary_within_horizon).astype(jnp.float32)
        return horizon, survival_label, survival_valid

    @staticmethod
    def _observed_survival_mass(gamma, transition):
        """Returns each anchor's mass from the first observed episode boundary."""
        discount = jnp.asarray(transition.discount)
        truncation = jnp.asarray(
            transition.extras["state_extras"]["truncation"]
        ).astype(bool)
        seq_len = discount.shape[0]
        anchor_indices = jnp.arange(seq_len - 1, dtype=jnp.int32)

        done = discount <= 0.0
        transition_indices = jnp.arange(seq_len, dtype=jnp.int32)
        boundary_indices = jnp.where(done, transition_indices, seq_len)
        next_boundary = jax.lax.associative_scan(
            jnp.minimum,
            boundary_indices,
            reverse=True,
        )[:-1]

        boundary_future_length = jnp.maximum(next_boundary - anchor_indices, 0)
        boundary_observed = next_boundary < seq_len
        clipped_boundary = jnp.minimum(next_boundary, seq_len - 1)
        boundary_is_truncation = truncation[clipped_boundary]
        boundary_at_anchor = boundary_observed & (next_boundary == anchor_indices)
        unsafe_boundary_ahead = boundary_observed & ~boundary_is_truncation
        gamma = jnp.asarray(gamma, dtype=jnp.float32)
        unsafe_survival_mass = 1.0 - jnp.power(
            gamma,
            boundary_future_length.astype(gamma.dtype),
        )
        survival_mass = jnp.where(
            boundary_at_anchor,
            jnp.zeros_like(unsafe_survival_mass),
            jnp.where(
                unsafe_boundary_ahead,
                unsafe_survival_mass,
                jnp.ones_like(unsafe_survival_mass),
            ),
        )
        return survival_mass

    @staticmethod
    @functools.partial(jax.jit, static_argnames=("buffer_config",))
    def flatten_crl_survive_fn(buffer_config, transition, sample_key):
        """Adds observed survival mass and Monte Carlo labels to CRL samples."""
        flattened = TrajectoryUniformSamplingQueue.flatten_crl_fn(
            buffer_config,
            transition,
            sample_key,
        )
        survival_key = jax.random.fold_in(sample_key, 0x5A17)
        survival_horizon, survival_label, survival_valid = (
            TrajectoryUniformSamplingQueue._sample_survival_targets(
                buffer_config[0], transition, survival_key
            )
        )
        survival_mass = TrajectoryUniformSamplingQueue._observed_survival_mass(
            buffer_config[0], transition
        )
        extras = dict(flattened.extras)
        extras.update(
            survival_horizon=survival_horizon,
            survival_label=survival_label,
            survival_valid=survival_valid,
            survival_mass=survival_mass,
        )
        return flattened._replace(extras=extras)

    def size(self, buffer_state: ReplayBufferState) -> int:
        return (
            buffer_state.insert_position - buffer_state.sample_position
        )
