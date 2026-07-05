# logp examples

## Where to start

| I want to... | Run |
|---|---|
| Compare divergence families on simple distributions | `divergence_landscape` |
| Compare f-divergence, OT, and MMD rankings | `distance_families` |
| Estimate MI for correlated Gaussian variables | `ksg_mutual_information` |
| See KSG behavior in a 10D joint space | `ksg_multivariate` |
| Rank synthetic features by mutual information | `feature_selection` |
| Flag categorical distribution drift | `text_similarity` |

## Example dependencies

| Dependency | Example | Check |
|---|---|---|
| `wass`, `rkhs` | `distance_families` | f-divergence, optimal-transport, and MMD rankings on the same distributions |

```sh
cargo run --example divergence_landscape
cargo run --example distance_families
cargo run --example ksg_mutual_information
cargo run --example ksg_multivariate
cargo run --example feature_selection
cargo run --example text_similarity
```

## What to inspect

- `divergence_landscape` shows why KL can grow while JS stays bounded by `ln(2)`.
- `distance_families` compares rankings from f-divergences, optimal transport, and kernel MMD on the same discrete distributions.
- `ksg_mutual_information` compares KSG estimates against the closed-form Gaussian MI curve.
- `ksg_multivariate` shows sample-size sensitivity in a 5D plus 5D setting where histogram estimators would be sparse.
- `feature_selection` ranks linear, nonlinear, and noise features by estimated MI with the target.
- `text_similarity` is a categorical drift example despite the historical filename: it simulates reference and incoming batch distributions and flags drift with JS/KL.
