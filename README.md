# logp

Information theory primitives: entropies and divergences.

Dual-licensed under MIT or Apache-2.0.

[crates.io](https://crates.io/crates/logp) | [docs.rs](https://docs.rs/logp)

```rust
use logp::{entropy_nats, kl_divergence, jensen_shannon_divergence};

let p = [0.1, 0.9];
let q = [0.9, 0.1];

// Shannon entropy in nats
let h = entropy_nats(&p, 1e-9).unwrap();

// Relative entropy (KL)
let kl = kl_divergence(&p, &q, 1e-9).unwrap();

// Symmetric, bounded Jensen-Shannon
let js = jensen_shannon_divergence(&p, &q, 1e-9).unwrap();
```

## Taxonomy of Divergences

| Family | Generator | Key Property |
|---|---|---|
| **f-divergences** | Convex $f(t)$ with $f(1)=0$ | Monotone under Markov morphisms (coarse-graining) |
| **Bregman** | Convex $F(x)$ | Dually flat geometry; generalized Pythagorean theorem |
| **Jensen-Shannon** | $f$-div + metric | Symmetric, bounded $[0, \ln 2]$, $\sqrt{JS}$ is a metric |
| **Alpha** | $\rho_\alpha = \int p^\alpha q^{1-\alpha}$ | Encodes Rényi, Tsallis, Bhattacharyya, Hellinger |

## Connections

- [`rkhs`](../rkhs): MMD and KL both measure distribution "distance"
- [`wass`](../wass): Wasserstein vs entropy-based divergences
- [`fynch`](../fynch): Temperature scaling affects entropy calibration
