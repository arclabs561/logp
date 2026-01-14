//! # Zipf / power-law tails
//!
//! A lot of discrete data (text tokens, code tokens, entity IDs) has heavy tails.
//! A simple model is a Zipf law:
//!
//! ```text
//! f(r) ≈ C / r^α
//!
//! where:
//!   r = rank (1 = most frequent)
//!   f(r) = frequency (or probability mass)
//!   α = power-law exponent ("tail heaviness")
//! ```
//!
//! This matters because heavy tails create persistent “rare events”. In the entropy calibration
//! paper (Cao, Valiant, Liang 2025), the tail exponent is linked to how quickly generation
//! instability improves with scale: α close to 1 implies very slow improvement.
//!
//! ## Scaling Law (Proposition 3.1)
//!
//! For a distribution with power-law exponent α, the probability of generating a "singleton"
//! (token seen exactly once during training) scales with training set size m as:
//!
//! ```text
//! E[singleton mass] ∼ m^(1/α - 1)
//! ```
//!
//! Important nuances:
//!
//! - The proposition is derived in a **simplified setting** (unigram-like rare-event model).
//! - For an *infinite* Zipf distribution, normalizability requires **α > 1**. In practice,
//!   people fit α on a finite rank range (e.g. top 5k unigrams), and those fitted values can be
//!   < 1 without implying a true infinite-vocabulary Zipf law.
//!
//! The entropy-calibration paper reports fitted α values that are **near 1 for text** and **larger
//! for code**, which is consistent with the observed “slow scaling” for text and “faster scaling”
//! for code — but this is evidence, not a theorem about all corpora or tokenizations.
//!
//! ## What this module does
//!
//! We provide a simple *diagnostic* fit of α by linear regression in log-log space:
//!
//! ```text
//! log f(r) ≈ log C - α log r
//! ```
//!
//! This is not a gold-standard estimator (power laws are subtle), but it is often good enough to:
//! - compare corpora (text vs code),
//! - sanity-check tail heaviness, and
//! - monitor drift over time.
//!
//! # Examples
//!
//! ```rust
//! use surp::zipf::zipf_fit_from_counts;
//!
//! // Synthetic Zipf-like counts: count[r] ~ 1 / (r+1)^1.5
//! let alpha_true = 1.5;
//! let scale = 1_000_000.0;
//! let counts: Vec<usize> = (1..=5000)
//!     .map(|r| (scale / (r as f64).powf(alpha_true)).round() as usize)
//!     .collect();
//!
//! let fit = zipf_fit_from_counts(&counts, 5, 5000).unwrap().unwrap();
//! assert!((fit.alpha - alpha_true).abs() < 0.1);
//! ```

use thiserror::Error;

/// Errors returned by Zipf fitting helpers.
#[derive(Debug, Error)]
pub enum ZipfError {
    /// Not enough usable points to fit a line.
    #[error("not enough data points to fit (need >= 2, got {0})")]
    NotEnoughPoints(usize),
    /// Internal consistency error (mismatched lengths).
    #[error("internal length mismatch")]
    LengthMismatch,
    /// Encountered a non-finite intermediate (should not happen for valid positive counts).
    #[error("non-finite value during fit")]
    NonFinite,
}

/// Result of a log-log linear Zipf fit.
#[derive(Clone, Debug, PartialEq)]
pub struct ZipfFit {
    /// Estimated exponent α in f(r) ~ C / r^α.
    pub alpha: f64,
    /// Estimated log(C) in the same log base used internally (natural log).
    pub ln_c: f64,
    /// Coefficient of determination for the linear fit in log-log space.
    pub r2: f64,
    /// Number of points used in the fit.
    pub n_points: usize,
}

/// Fit a Zipf law exponent from raw token counts.
///
/// - `counts[i]` is the count for token i (order does not matter).
/// - `min_count` filters out very rare tokens (default-ish: 5).
/// - `max_rank` caps how deep into the tail to fit (to avoid extreme sparsity artifacts).
///
/// Returns `None` if there are fewer than 2 usable ranks.
pub fn zipf_fit_from_counts(
    counts: &[usize],
    min_count: usize,
    max_rank: usize,
) -> Result<Option<ZipfFit>, ZipfError> {
    // Extract frequencies, sort descending, filter.
    let mut freqs: Vec<usize> = counts.iter().copied().filter(|&c| c >= min_count).collect();
    if freqs.len() < 2 {
        return Ok(None);
    }
    freqs.sort_unstable_by(|a, b| b.cmp(a));

    let k = freqs.len().min(max_rank);
    if k < 2 {
        return Ok(None);
    }

    // x = ln(rank), y = ln(freq)
    let mut xs = Vec::with_capacity(k);
    let mut ys = Vec::with_capacity(k);
    for (i, &f) in freqs.iter().take(k).enumerate() {
        let rank = (i + 1) as f64;
        let ff = f as f64;
        // rank >= 1, f >= min_count >= 1 => logs finite
        xs.push(rank.ln());
        ys.push(ff.ln());
    }

    let (slope, intercept, r2) = linear_fit(&xs, &ys)?;

    // y ≈ intercept + slope x, and slope should be negative; α = -slope.
    Ok(Some(ZipfFit {
        alpha: -slope,
        ln_c: intercept,
        r2,
        n_points: k,
    }))
}

fn linear_fit(xs: &[f64], ys: &[f64]) -> Result<(f64, f64, f64), ZipfError> {
    if xs.len() != ys.len() {
        return Err(ZipfError::LengthMismatch);
    }
    let n = xs.len();
    if n < 2 {
        return Err(ZipfError::NotEnoughPoints(n));
    }

    let n_f = n as f64;
    let mean_x = xs.iter().sum::<f64>() / n_f;
    let mean_y = ys.iter().sum::<f64>() / n_f;

    let mut sxx = 0.0;
    let mut sxy = 0.0;
    let mut syy = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }

    if !sxx.is_finite() || !sxy.is_finite() || !syy.is_finite() {
        return Err(ZipfError::NonFinite);
    }
    if sxx == 0.0 {
        return Err(ZipfError::NotEnoughPoints(n));
    }

    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;

    // R^2 = 1 - SSE/SST
    let mut sse = 0.0;
    for (&x, &y) in xs.iter().zip(ys.iter()) {
        let y_hat = intercept + slope * x;
        let err = y - y_hat;
        sse += err * err;
    }
    let r2 = if syy > 0.0 { 1.0 - (sse / syy) } else { 1.0 };

    if !slope.is_finite() || !intercept.is_finite() || !r2.is_finite() {
        return Err(ZipfError::NonFinite);
    }

    Ok((slope, intercept, r2.clamp(0.0, 1.0)))
}

/// Compute the singleton mass scaling exponent from Proposition 3.1.
///
/// For a power-law distribution with exponent α, the expected singleton mass
/// (probability of generating a token seen exactly once during training) scales as:
///
/// ```text
/// E[K_{m,1} / m] ∼ m^(1/α - 1)
/// ```
///
/// where m is the training set size and K_{m,1} is the count of singletons.
///
/// # Returns
///
/// The scaling exponent (1/α - 1). Negative values mean improvement with scale.
///
/// # Example
///
/// ```rust
/// use surp::zipf::singleton_mass_scaling_exponent;
///
/// // Zipf's law (α = 1): no improvement
/// assert!((singleton_mass_scaling_exponent(1.0) - 0.0).abs() < 1e-10);
///
/// // Code-like distribution (α = 1.5): moderate improvement
/// let exp = singleton_mass_scaling_exponent(1.5);
/// assert!((exp - (-1.0/3.0)).abs() < 1e-10);
/// ```
pub fn singleton_mass_scaling_exponent(alpha: f64) -> f64 {
    1.0 / alpha - 1.0
}

/// Estimate how much larger the training set needs to be to halve singleton mass.
///
/// Given current singleton mass scaling as m^β where β = 1/α - 1:
/// - To reduce singleton mass by factor k, need m' = m × k^(1/β)
///
/// # Returns
///
/// The factor by which training data must increase to halve singleton mass.
/// Returns `f64::INFINITY` if α ≤ 1 (no improvement possible with scale alone).
///
/// # Example
///
/// ```rust
/// use surp::zipf::data_factor_to_halve_singletons;
///
/// // For α = 1.5 (code): need ~8x more data to halve singletons
/// let factor = data_factor_to_halve_singletons(1.5);
/// assert!((factor - 8.0).abs() < 0.1);
/// ```
pub fn data_factor_to_halve_singletons(alpha: f64) -> f64 {
    let beta = singleton_mass_scaling_exponent(alpha);
    if beta >= 0.0 {
        return f64::INFINITY;
    }
    // m' / m = 2^(1/|β|)
    2.0_f64.powf(-1.0 / beta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_scaling_at_zipf() {
        // α = 1 → exponent = 0
        assert!((singleton_mass_scaling_exponent(1.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn singleton_scaling_at_code() {
        // α = 1.5 → exponent = -1/3
        let exp = singleton_mass_scaling_exponent(1.5);
        assert!((exp - (-1.0 / 3.0)).abs() < 1e-10);
    }

    #[test]
    fn data_factor_at_code() {
        // α = 1.5 → need 2^3 = 8x data to halve singletons
        let factor = data_factor_to_halve_singletons(1.5);
        assert!((factor - 8.0).abs() < 0.1);
    }

    #[test]
    fn data_factor_at_zipf_is_infinite() {
        assert!(data_factor_to_halve_singletons(1.0).is_infinite());
        assert!(data_factor_to_halve_singletons(0.9).is_infinite());
    }

    #[test]
    fn zipf_fit_recovers_alpha_reasonably() {
        let alpha_true = 1.5;
        let scale = 1_000_000.0;
        let counts: Vec<usize> = (1..=5000)
            .map(|r| (scale / (r as f64).powf(alpha_true)).round() as usize)
            .collect();

        let fit = zipf_fit_from_counts(&counts, 5, 5000)
            .unwrap()
            .expect("should have enough points");

        assert!((fit.alpha - alpha_true).abs() < 0.1, "fit={fit:?}");
        assert!(fit.r2 > 0.99, "fit={fit:?}");
    }

    #[test]
    fn zipf_fit_none_if_too_few_points() {
        let counts = vec![1usize, 1, 1, 1];
        let fit = zipf_fit_from_counts(&counts, 5, 100).unwrap();
        assert!(fit.is_none());
    }
}
