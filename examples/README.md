# logp examples

Examples for the `logp` information theory crate.

## Running

```sh
cargo run -p logp --example <name>
```

## Examples

| Example | Description |
|---|---|
| `divergence_landscape` | Sweeps p = [t, 1-t] against q = [0.5, 0.5] and prints a table of KL, JS, Hellinger, Bhattacharyya, Renyi, and Tsallis divergences. Shows how each family responds to the same distributional shift. |
| `ksg_mutual_information` | Bivariate Gaussian MI estimation with the KSG estimator. Sweeps correlation rho from 0 to 0.99 and compares estimates against the closed-form MI = -0.5 * ln(1 - rho^2). |
| `ksg_multivariate` | KSG MI estimation in high dimensions (5D + 5D = 10D joint). Shows convergence as sample size increases from 200 to 5000. Demonstrates that KSG works where histogram-based estimators would need 10^10 bins. |
| `feature_selection` | Feature selection via KSG mutual information. Generates a synthetic dataset (8 features, 1 target) with linear, nonlinear, and noise features. Ranks features by MI to identify the informative subset. |
| `text_similarity` | Distribution shift detection for categorical features. Simulates a reference distribution and incoming batches with increasing drift. Uses JS and KL divergence to flag anomalous batches. |
