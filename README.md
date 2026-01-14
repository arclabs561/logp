# surp

Information theory: entropy, KL divergence, f-divergences, mutual information.

(surp: from surprisal, the unit of information)

Dual-licensed under MIT or Apache-2.0.

```rust
use surp::{entropy, kl_divergence, js_divergence, mutual_information};

let p = [0.25, 0.25, 0.25, 0.25];  // uniform
let q = [0.5, 0.25, 0.125, 0.125]; // skewed

let h = entropy(&p);               // 2.0 bits
let kl = kl_divergence(&p, &q);    // asymmetric
let js = js_divergence(&p, &q);    // symmetric, bounded [0,1]
```

## Modules

| Module | Purpose |
|--------|---------|
| Root | Core: entropy, KL, JS, MI, fingerprints |
| `fdiv` | f-divergences: Hellinger, χ², Rényi, TV |
| `unseen` | Valiant-Valiant sublinear estimators |
| `zipf` | Power-law tail fitting |
| `entropy_calibration` | LLM calibration metrics (Cao, Valiant, Liang 2025) |

## Functions

| Function | Measures | Formula |
|----------|----------|---------|
| `entropy(p)` | Uncertainty | H(p) = -Σ p(x) log p(x) |
| `cross_entropy(p, q)` | Expected bits | H(p,q) = -Σ p(x) log q(x) |
| `kl_divergence(p, q)` | Asymmetric distance | KL(p‖q) = Σ p(x) log(p(x)/q(x)) |
| `js_divergence(p, q)` | Symmetric distance | JS = ½KL(p‖m) + ½KL(q‖m) |
| `mutual_information(joint, r, c)` | Shared info | I(X;Y) = H(X) + H(Y) - H(X,Y) |
| `conditional_entropy(joint, r, c)` | Remaining uncertainty | H(Y\|X) = H(X,Y) - H(X) |
| `normalized_mutual_information` | Scaled MI | NMI = 2I(X;Y) / (H(X) + H(Y)) |
| `fingerprint(samples)` | Histogram of histogram | F_i = count of elements seen i times |
| `good_turing_unseen_mass` | Unseen probability | P(unseen) ≈ F₁/n |

## f-Divergences

Generalized divergence measures parameterized by convex functions.

```rust
use surp::fdiv::{renyi_divergence, hellinger_distance, chi_squared_divergence};

let p = [0.25, 0.25, 0.25, 0.25];
let q = [0.5, 0.25, 0.125, 0.125];

let renyi = renyi_divergence(&p, &q, 2.0);  // alpha=2
let hellinger = hellinger_distance(&p, &q);  // sqrt of Hellinger distance
let chi2 = chi_squared_divergence(&p, &q);   // chi-squared
```

## Sublinear Estimation

The Valiant-Valiant estimator (JACM 2017) achieves O(k/log k) sample complexity
for entropy and support size estimation over k-element distributions. This is
optimal up to constant factors.

```rust
use surp::unseen::{entropy_unseen, support_unseen, recover_histogram};

// Sample from a distribution with 10,000 elements, but only take 2,000 samples
let samples: Vec<u32> = (0..2000).map(|i| i % 500).collect();

// Empirical entropy would be biased (too low).
// Unseen estimator reconstructs the "invisible" portion via Linear Programming.
let h = entropy_unseen(&samples);
let s = support_unseen(&samples);
```

Reference: Valiant & Valiant. "Estimating the Unseen." https://doi.org/10.1145/3125643

## Entropy Calibration

Metrics for evaluating whether an LLM's generation entropy matches its log loss
on reference text. Based on Cao, Valiant, Liang (2025).

```rust
use surp::entropy_calibration::{entropy_calibration_bits, mean_nll_bits_from_ln};

// Per-token log-probabilities (natural log) from model
let gen_logprobs = vec![(0.3_f64).ln(); 100];   // generated tokens
let ref_logprobs = vec![(0.35_f64).ln(); 100];  // reference tokens

let stats = entropy_calibration_bits(&gen_logprobs, &ref_logprobs).unwrap();
// stats.entce_bits_per_token > 0 means entropy exceeds log loss.
// This is consistent with self-conditioning mismatch / error accumulation, but not sufficient
// as a standalone diagnosis.
```

## Zipf Fitting

Fit power-law exponent α from token counts. Heavy tails (α ≈ 1) imply slow
improvement of calibration with scale in the paper’s simplified model:
singleton mass scales as m^(1/α - 1) (α > 1).

```rust
use surp::zipf::{zipf_fit_from_counts, singleton_mass_scaling_exponent};

// Synthetic counts with α ≈ 1.2
let counts: Vec<usize> = (1..=1000)
    .map(|r| (1e6 / (r as f64).powf(1.2)) as usize)
    .collect();

if let Some(fit) = zipf_fit_from_counts(&counts, 5, 5000).unwrap() {
    println!("α = {:.2}, R² = {:.3}", fit.alpha, fit.r2);
    
    // Scaling exponent from Proposition 3.1 (Cao et al.)
    let scaling = singleton_mass_scaling_exponent(fit.alpha);
    println!("Singleton mass scales as m^{:.2}", scaling);
}
```

## Connections

- [`fynch`](../fynch): Temperature scaling affects entropy calibration
- [`rkhs`](../rkhs): KL/JS for discrete; MMD for continuous distributions
- [`wass`](../wass): Wasserstein vs entropy-based divergences
- [`stratify`](../stratify): NMI for clustering evaluation
