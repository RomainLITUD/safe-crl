# Training Algorithm

The trainer accepts exactly `scaling_crl` and `scaling_crl_survive`. Future-goal
relabeling, score construction, entropy handling, optimizer ordering, and
policy architecture are shared between them. Safe-CRL changes the reduction of
the critic's InfoNCE rows as described below.

## Replay Relabeling

Rollouts store observations, actions, episode identity, task goals, achieved
goals, and any ego-frame projection metadata needed by the environment.
`scaling_crl_survive` additionally records truncation flags. Its replay
flattener derives an observed future length `L_t` for every anchor. The length
ends at the first stored failure or administrative truncation boundary, or at
the sampled-window boundary when no earlier boundary is available. It then
stores the realized discounted survival mass
`survival_mass = 1 - gamma ** L_t`. A boundary can therefore produce
`L_t = 0` and zero mass. Goal respawning does not end this survival horizon.

For every anchor except the final sampled transition, replay selects one later
achieved goal from the same episode and unchanged task-goal segment. Candidate
probability is proportional to `gamma ** horizon`. If no later candidate is
available, the existing small diagonal fallback selects the anchor itself.
Ego-view and goal-lidar environments project the achieved goal relative to the
anchor exactly as during collection.

## Shared Scaling-CRL Objective

Both supported modes use separate state-action and goal towers with
`score(s, a, g) = -||SA(s, a) - G(g)||_2`.

A batch forms a B x B score matrix. The diagonal relabeled future is the positive
and other valid rows provide in-batch negatives. `scaling_crl` averages the
valid per-anchor InfoNCE rows uniformly.

For `scaling_crl_survive`, critic row `i` is weighted by its observed replay
mass:

```text
main_crl_loss = sum_i(valid_i * survival_mass_i * infonce_row_i)
                / max(sum_i(valid_i), 1)
```

The weights are deliberately not normalized by their own sum. Thus, a
zero-mass row contributes zero while remaining part of the minibatch-size
denominator. The configured squared log-sum-exp penalty remains the same
unweighted regularizer used by Scaling-CRL.

Current zero-mass edge case: replay still marks every retained anchor as
`future_valid` and uses the diagonal self-goal fallback when no later future is
available. Multiplication by zero removes that row from the main MW-InfoNCE
term, but it is not literally masked before positive sampling. Its fallback
goal can remain in the in-batch negative pool, the row can affect the
unweighted log-sum-exp regularizer, and the actor can train on the fallback.
This differs from Appendix A.5 of `SafeCRL.pdf`, which states that `L_t = 0`
rows are masked before positive-goal sampling.

The actor trains on the same relabeled goal as the critic positive. It maximizes
the critic score with learned, fixed, or disabled entropy regularization. Actor
and entropy updates run before the critic update.

## Survival Extension

`scaling_crl_survive` also adds a goal-independent survival logit whose sigmoid
is `Z(s,a)`.

- Replay samples a geometric survival horizon using `gamma`.
- True termination within that horizon has target zero.
- Survival through the horizon has target one.
- Truncation boundaries and unobserved suffixes are masked.
- Z is optimized separately with sigmoid binary cross entropy.
- The actor adds `-log Z(s,a)` with unit weight to its ordinary Scaling-CRL
  loss. This is the only change to the actor objective.

The sampled binary survival label used for Z optimization is distinct from the
deterministic `survival_mass` attached to the same replay anchor. The former
trains the Z-encoder as a one-sample Monte Carlo estimator; the latter weights
the critic's complete InfoNCE row. This matches the updated Section 3.3 and
Appendix A.4 of `SafeCRL.pdf`.

`scaling_crl` does not create or update Z. Full checkpoints contain
alpha/actor/critic parameters and append Z parameters only for
`scaling_crl_survive`. Final policy bundles remain actor-only with resolved
configuration metadata.
