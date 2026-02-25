# logp

Information theory primitives: entropies and divergences.

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

17 tests including property-based tests for KL non-negativity, Pinsker's inequality, JS boundedness, sqrt(JS) triangle inequality, and KSG estimator behavior.

## License

MIT OR Apache-2.0
