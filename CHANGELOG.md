# Changelog

All notable changes to this crate are documented here.
Format: [Keep a Changelog](https://keepachangelog.com).

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
