# logp

Information theory primitives: entropies and divergences.

## Problem

You have two probability distributions and need to quantify how different they are. KL divergence is asymmetric and unbounded; JS divergence is symmetric and bounded by ln(2). Hellinger distance satisfies the triangle inequality. Each divergence has different properties, and choosing the wrong one gives misleading results.

This library provides all of them with a consistent interface, validated by property-based tests (KL non-negativity, Pinsker's inequality, JS boundedness, sqrt(JS) triangle inequality, and more).

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

## What it provides

**Entropies**: Shannon (nats/bits), cross-entropy, mutual information (discrete + KSG continuous estimator).

**Divergences**: KL, Jensen-Shannon, Hellinger, Bhattacharyya, Renyi, Tsallis, Amari alpha-family, Csiszar f-divergences, Bregman divergences.

**Gaussian KL**: Closed-form KL between diagonal Gaussians.

## Usage

```toml
[dependencies]
logp = "0.1.0"
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

76 tests (52 unit + 24 doc-tests) covering all public API functions, including property-based tests for KL non-negativity, Pinsker's inequality, JS boundedness, sqrt(JS) triangle inequality, Hellinger triangle inequality, Renyi limit to KL, Amari alpha-KL correspondence, Csiszar f-divergence with KL generator, total Bregman normalization, digamma recurrence/domain, PMI edge cases, and KSG estimator.

## License

MIT OR Apache-2.0
