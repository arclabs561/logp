# logp

[![crates.io](https://img.shields.io/crates/v/logp.svg)](https://crates.io/crates/logp)
[![Documentation](https://docs.rs/logp/badge.svg)](https://docs.rs/logp)
[![CI](https://github.com/arclabs561/logp/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/logp/actions/workflows/ci.yml)

Information theory primitives.

```toml
[dependencies]
logp = "0.2.3"
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

**Categorical drift detection**. JS and KL divergence compare an incoming categorical batch against a reference distribution:

```bash
cargo run --example text_similarity
```

```text
batch                JS  KL(b|r)    KL(r|b)     alert?
------------------------------------------------------
no_drift        0.00000  0.00000    0.00000         no
mild_drift      0.00307  0.01310    0.01164         no
moderate        0.01889  0.08575    0.06951        YES
severe_drift    0.06655  0.31664    0.24658        YES
broken_etl      0.24331  1.04635    1.15254        YES

JS is symmetric and bounded; KL is asymmetric.
```

See [`examples/README.md`](examples/README.md) for the full example map: `ksg_mutual_information`, `ksg_multivariate`, `feature_selection`, `distance_families`, and the drift-detection example above.

## Usage

```toml
[dependencies]
logp = "0.2.3"
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

All entropies and divergences are in nats unless the function name says bits (`entropy_bits`). Invalid input (empty slices, negative or NaN entries, unnormalized distributions, `q_i = 0` where `p_i > 0`) returns an `Err`; nothing panics. The KSG estimator can return small negative values on finite samples; clamp with `.max(0.0)` if you need a nonnegative MI.

## Tests

```bash
cargo test -p logp
```

Unit, doc, and property-based tests cover the public API.

Main invariants include non-negativity, boundedness, triangle inequalities,
alpha-limit behavior, entropy identities, data-processing inequalities, and
KSG accuracy against Gaussian ground truth.

Edge-case tests cover empty input, zero probabilities, unnormalized input,
exact error variants, near-boundary precision, and KSG ties.

## License

MIT OR Apache-2.0
