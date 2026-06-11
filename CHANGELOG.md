# Changelog

All notable changes to this crate are documented here.
Format: [Keep a Changelog](https://keepachangelog.com).

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
