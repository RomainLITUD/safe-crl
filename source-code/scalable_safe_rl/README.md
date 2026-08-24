# scalable_safe_rl

JAX/Brax training code for the two methods used by this project:

- `scaling_crl`: goal-conditioned Scaling-CRL with a negative-L2 contrastive critic.
- `scaling_crl_survive` (MassCRL): the same reachability objective plus a
  goal-independent survival model and actor survival penalty.

`scaling_crl_survive` is the default in the shipped YAML files. Both methods
use the same concatenated residual MLP actor and critic architecture.

Run from the repository root:

```bash
python -m scalable_safe_rl.train --config scalable_safe_rl/config.yaml
```

Select the reachability-only method with:

```bash
python -m scalable_safe_rl.train \
  --config scalable_safe_rl/config.yaml \
  --critic-loss-type scaling_crl
```

Windows verification is CPU-only. Full experiments are intended for the
configured Linux GPU environment. The repository's focused checks avoid
training, rendering, and expensive JAX/Brax compilation.
