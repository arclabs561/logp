# logp

[![crates.io](https://img.shields.io/crates/v/logp.svg)](https://crates.io/crates/logp)
[![Documentation](https://docs.rs/logp/badge.svg)](https://docs.rs/logp)
[![CI](https://github.com/arclabs561/logp/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/logp/actions/workflows/ci.yml)

Information theory primitives.

```toml
[dependencies]
logp = "0.2.0"
```

Discrete distributions: Shannon/Renyi/Tsallis entropy, KL/JS/Hellinger/Bhattacharyya divergence, chi-squared divergence, total variation, f-divergences. Continuous distributions: KSG mutual information estimator (type I and II). All validated by property-based tests (KL non-negativity, Pinsker's inequality, JS boundedness, sqrt(JS) triangle inequality).

## Examples

**Divergence landscape**. Sweep all divergences over a family of binary distributions to see how they behave as distributions diverge:

```bash
cargo run --example divergence_landscape
```

```text
t            KL       JS     Hell    Bhatt   Renyi0.5   Tsallis2
----------------------------------------------------------------------
0.050   0.49463  0.14234  0.39075  0.16568    0.33136    0.81000
0.250   0.13081  0.03382  0.18459  0.03467    0.06934    0.25000
0.500   0.00000  0.00000  0.00000 -0.00000   -0.00000    0.00000
0.750   0.13081  0.03382  0.18459  0.03467    0.06934    0.25000
0.950   0.49463  0.14234  0.39075  0.16568    0.33136    0.81000

Observations:
  - KL diverges as p -> delta; JS stays bounded by ln(2) ~ 0.693
  - Hellinger saturates at 1.0; Bhattacharyya diverges
```

**Document similarity via bag-of-words**. JS divergence between word frequency distributions as a simple text similarity measure:

```bash
cargo run --example text_similarity
```

```text
Pair               JS  KL(a|b)  KL(b|a)
----------------------------------------
doc_a-doc_b    0.2310   7.0780   7.0780
doc_a-doc_c    0.6931  21.4651  21.0799

Note: JS is symmetric and bounded [0, ln2]; KL is asymmetric and unbounded.
```

**KSG mutual information estimator**. Estimate mutual information of correlated Gaussians using the Kraskov-Stogbauer-Grassberger method, then compare against the analytical value:

```bash
cargo run --example ksg_mutual_information
```

**High-dimensional mutual information**. KSG estimation in 10 dimensions (two correlated 5D blocks). Histogram-based MI is infeasible at this dimensionality (10^10 bins), but KSG converges cleanly. Compares estimates against the closed-form Gaussian MI:

```bash
cargo run --example ksg_multivariate
```

**Feature selection**. Rank 8 candidate features (3 linear, 1 quadratic, 4 noise) by their KSG mutual information with a target variable. The nonlinear feature (quadratic) is correctly ranked alongside the linear ones -- something correlation-based methods miss:

```bash
cargo run --example feature_selection
```

## What it provides

**Entropies**: Shannon (nats/bits), Renyi, Tsallis, cross-entropy, conditional entropy, mutual information (discrete + KSG continuous estimator), normalized MI.

**Divergences**: KL, Jensen-Shannon (equal and weighted), Hellinger, Bhattacharyya, total variation, chi-squared, Renyi, Tsallis, Amari alpha-family, Csiszar f-divergences, Bregman divergences (SquaredL2 and NegEntropy generators).

**Gaussian KL**: Closed-form KL between diagonal Gaussians.

## Usage

```toml
[dependencies]
logp = "0.2.0"
```

```rust
use logp::{entropy_nats, kl_divergence, jensen_shannon_divergence};

let p = [0.25, 0.75];
let q = [0.5, 0.5];

let h = entropy_nats(&p, 1e-9).unwrap();          // Shannon entropy
let kl = kl_divergence(&p, &q, 1e-9).unwrap();    // KL(p || q)
let js = jensen_shannon_divergence(&p, &q, 1e-9).unwrap(); // JS divergence
```

The `tol` parameter controls how strictly inputs are validated as probability distributions. Use `1e-9` for normalized inputs; use `1e-6` if inputs may have minor floating-point drift.

## Tests

```bash
cargo test -p logp
```

133 tests (99 unit + 34 doc-tests) covering all public API functions, including property-based tests for KL non-negativity, Pinsker's inequality and tightness, JS boundedness, sqrt(JS) and Hellinger and total variation triangle inequality, Renyi divergence and entropy monotonicity in alpha, Renyi/Tsallis alpha=1 Shannon limit, Amari alpha-KL correspondence, Csiszar f-divergence with KL/Hellinger/chi-squared generators, Bhattacharyya-Renyi consistency, Bhattacharyya precision for near-identical distributions, Bregman non-negativity, entropy concavity, cross-entropy decomposition, conditional entropy chain rule, chi-squared/KL upper bound, total Bregman normalization, NegEntropy Bregman/KL equivalence, digamma precision at DLMF reference values, PMI edge cases and impossible-input errors, near-boundary numerical robustness, data processing inequality for discrete MI, f-divergence monotonicity under coarse-graining, streaming log-sum-exp, weighted JS entropy bounds, KSG estimator accuracy against Gaussian ground truth, and KSG ties handling.

## License

MIT OR Apache-2.0
