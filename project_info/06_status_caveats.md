# Current Status And Caveats

## Current Status

- The only supported critic losses are `scaling_crl` and
  `scaling_crl_survive`.
- `scaling_crl_survive` is the default Safe-CRL mode. It applies
  replay-observed survival-mass weighting to critic InfoNCE, learns a
  goal-independent Z model, and adds the actor log-Z correction.
- Actor training uses same-episode hindsight goals; rollout and evaluation use
  real task goals.
- Saved-policy HTML visualization supports both retained methods.
- CRL runs use canonical config, metric, policy-bundle, and multi-seed formats.
- Deterministic evaluation is the default. Stochastic and best-of-K evaluation
  receive a fresh reproducible action key at every step.

## Important Caveats

- Safe-CRL weights each complete InfoNCE row by `1 - gamma ** L_t` and divides
  by the valid-anchor count. Never self-normalize by the sum of these weights.
- Failure and administrative truncation both end the finite observed support
  used to compute `L_t`; truncation is still not classified as failure.
- Preserve the B x B negative-L2 objective and actor-before-critic update order.
- Survival labels distinguish true termination from truncation and mask
  horizons extending beyond observable data.
- The updated paper's Monte Carlo binary-label Z objective matches the code.
  The remaining code-paper mismatch is zero-mass replay handling: the code
  zero-weights the main InfoNCE row but retains its diagonal fallback, negative
  column, log-sum-exp contribution, and actor sample instead of masking the
  anchor before positive-goal sampling.
- Task-goal changes do not reset Z horizons.
- Empty `eval_env_id` means the training ID/layout/semantics.
- `track: false` avoids optional tracking dependency errors.
- Recheck saved-policy visualization after changing observations or actor inputs.
- Keep score construction, negative sampling, relabeling, shared networks, and
  the log-sum-exp regularizer matched. Safe-CRL differs through the observed
  critic-row mass, survival labels, Z optimization, and actor survival penalty.
- A Windows CPU smoke run has exercised one complete Safe-CRL replay, critic,
  Z, actor, and entropy update after adding mass-weighted InfoNCE.
