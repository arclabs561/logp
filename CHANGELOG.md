# Changelog

All notable changes to this crate are documented here.
Format: [Keep a Changelog](https://keepachangelog.com).

## [Unreleased]

## [0.2.5] - 2026-11-04

### Fixed

- `bhattacharyya_coeff` stays at or below 1. Its sum could round one step
  above 1 (for example BC(p, p) = 1.0000000000000002 for some p), outside the
  documented [0, 1] range, which made `acos(BC)` NaN for callers.
- `bhattacharyya_distance` and `renyi_divergence` no longer return a negative
  value or `-0.0` produced by rounding; they return `+0.0` there. The README
  divergence table printed `-0.00000` for both at t = 0.5.

### Changed

- `jensen_shannon_divergence`, `kl_divergence`, `cross_entropy_nats`,
  `entropy_nats`, `hellinger_squared`, `bhattacharyya_coeff` and
  `validate_simplex` (from 16 entries) check their inputs inside the compute
  loop instead of in separate passes; Jensen-Shannon no longer allocates the
  midpoint or validates p and q three times. Values and errors are unchanged
  (bit-identical in 60,000 random cases per function). Measured on an M3 Max:
  Jensen-Shannon about 2x faster from n = 10, Hellinger 2-2.5x and KL and
  cross-entropy about 1.5x from n = 100. `bregman_divergence` uses a stack
  buffer for gradients up to 32 entries (17.4 to 9.2 ns at n = 2).

## [0.2.4] - 2026-11-01

### Added

- Optional `serde` feature deriving `Serialize`/`Deserialize` on
  `distprop::Gaussian`.

### Fixed

- `mutual_information_ksg` with `KsgVariant::Alg2` now follows Kraskov et al.
  Eq. 9: each marginal count uses that marginal's own half-width (the largest
  marginal distance among the k nearest joint neighbors) and excludes the
  sample itself. Previously it used the shared joint radius and counted the
  sample, biasing estimates low (about -0.35 nats on independent Gaussians).
  Alg2 results change.
- `mutual_information` and `normalized_mutual_information` return
  `Error::Domain` when `n_x * n_y` overflows `usize` instead of panicking.
- `distprop::activations::softplus` now remains finite for large finite inputs.
- `log_sum_exp_iter` no longer returns NaN when the first value is `-inf`; it
  now matches `log_sum_exp`.

### Changed

- Clarify KSG algorithm 2's separate marginal half-widths and inclusive counts,
  self-exclusion, and the continuous-sample assumption. Tied or quantized
  coordinates are not jittered or corrected; this documents the estimator's
  assumptions and adds no tie handling.

## [0.2.3] - 2026-06-10

### Fixed

- `cross_entropy_nats` now returns `Error::LengthMismatch` when `p` and `q` differ in length; previously it zip-truncated and returned a silently underestimated value when `p` was longer than `q`.
- `renyi_divergence` and `tsallis_divergence` now reject `alpha <= 0` and non-finite alpha with `Error::InvalidAlpha`, matching their documented domain; previously `alpha = 0` returned 0 for all inputs via the `0^0 = 1` continuity convention.

### Added

- Edge-case regression tests (`tests/edge_cases.rs`) pinning empty-input,
  zero-probability, unnormalized-input, and exact error-variant behavior for
  the entropy / KL / cross-entropy / KSG functions.

### Changed

- Documented crate-wide conventions (nats as the unit, `tol` semantics,
  the `0 ln 0 = 0` convention, no-panic policy) in the crate docs, plus
  per-function contracts: KSG negative-estimate caveat and error list,
  `entropy_unchecked` behavior on invalid input, `pmi` non-validation,
  `propagate_unscented` kappa domain, and the diagonal-Gaussian KL error
  contract. No behavior changes.

## [0.2.2] - 2026-04-28

### Changed

- Bumped `rkhs` dependency to `0.2`.

### Fixed

- Fixed `clippy::excessive_precision` on the Euler-Mascheroni constant
  (truncated literal to 16 digits, the f64-representable precision).
- Fixed `clippy::let_and_return` in the KSG `next_uniform` closure.

These clippy fixes unblocked CI which had been red since 2026-04-15 under
clippy 1.94+.

## [0.2.1] - 2026-04-06

### Changed

- Updated `distprop` module.

## [0.2.0] - 2026-03-20

### Added

- Distribution propagation module (Petersen distprop).

## [0.1.4] - 2026-03-12

Initial public release on crates.io.
