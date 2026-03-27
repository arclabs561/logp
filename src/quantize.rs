//! Distributional quantization via the Lloyd-Max algorithm.
//!
//! Given a continuous distribution family (Gaussian, logistic, Cauchy) and a
//! number of quantization levels, compute the optimal codebook -- the set of
//! reconstruction points and decision boundaries that minimize expected
//! mean-squared quantization error.
//!
//! The algorithm is the classical Lloyd-Max iteration applied to known
//! distributions (no samples needed):
//!
//! 1. Initialize boundaries uniformly across the distribution's support.
//! 2. Compute each level's representative as the conditional mean
//!    `E[X | b_i <= X < b_{i+1}]`.
//! 3. Update each interior boundary as the midpoint of adjacent representatives.
//! 4. Repeat until convergence.
//!
//! For symmetric distributions and even level counts, the codebook is symmetric
//! about zero by construction.
//!
//! # References
//!
//! - Lloyd (1982). "Least squares quantization in PCM."
//! - Max (1960). "Quantizing for minimum distortion."
//! - Petersen et al. (2022). "On Distribution-Based Quantization."

use crate::{Error, Result};
use core::f64::consts::{FRAC_1_SQRT_2, PI};

// ── Distribution primitives ─────────────────────────────────────────────────

/// Distribution family for quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeDist {
    /// Standard Gaussian N(0,1).
    Gaussian,
    /// Standard logistic distribution (mean 0, scale 1).
    Logistic,
    /// Standard Cauchy distribution (location 0, scale 1).
    /// Since the Cauchy distribution has no finite mean, the outer bins are
    /// truncated at a large but finite radius rather than extending to infinity.
    Cauchy,
}

impl QuantizeDist {
    /// Probability density function.
    fn pdf(self, x: f64) -> f64 {
        match self {
            Self::Gaussian => (-0.5 * x * x).exp() / (2.0 * PI).sqrt(),
            Self::Logistic => {
                let e = (-x).exp();
                e / (1.0 + e).powi(2)
            }
            Self::Cauchy => 1.0 / (PI * (1.0 + x * x)),
        }
    }

    /// Cumulative distribution function.
    fn cdf(self, x: f64) -> f64 {
        match self {
            Self::Gaussian => 0.5 * erfc(-x * FRAC_1_SQRT_2),
            Self::Logistic => 1.0 / (1.0 + (-x).exp()),
            Self::Cauchy => 0.5 + x.atan() / PI,
        }
    }

    /// Whether the outer bins should use finite truncation
    /// (distributions without a finite mean).
    fn needs_truncation(self) -> bool {
        matches!(self, Self::Cauchy)
    }

    /// Truncation radius for distributions without finite mean.
    fn truncation_radius(self) -> f64 {
        match self {
            Self::Cauchy => 200.0,
            _ => unreachable!(),
        }
    }

    /// Effective support radius for boundary initialization.
    fn init_radius(self) -> f64 {
        match self {
            Self::Gaussian => 4.0,
            Self::Logistic => 8.0,
            Self::Cauchy => 50.0,
        }
    }
}

// ── Codebook computation ────────────────────────────────────────────────────

/// Result of codebook computation.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Decision boundaries, length `levels - 1`.
    /// `boundaries[i]` is the threshold between level `i` and level `i+1`.
    pub boundaries: Vec<f64>,
    /// Representative (reconstruction) values, length `levels`.
    pub representatives: Vec<f64>,
    /// Number of Lloyd-Max iterations until convergence.
    pub iterations: usize,
}

/// Compute an optimal quantization codebook for a symmetric distribution
/// using the Lloyd-Max algorithm.
///
/// Returns a [`Codebook`] with `levels` reconstruction points and
/// `levels - 1` decision boundaries.
///
/// # Errors
///
/// Returns [`Error::Domain`] if `levels` is zero.
///
/// # Examples
///
/// ```
/// use logp::quantize::{optimal_codebook, QuantizeDist};
///
/// let cb = optimal_codebook(QuantizeDist::Gaussian, 4).unwrap();
/// assert_eq!(cb.representatives.len(), 4);
/// assert_eq!(cb.boundaries.len(), 3);
///
/// // Boundaries are sorted.
/// for w in cb.boundaries.windows(2) {
///     assert!(w[0] < w[1]);
/// }
/// ```
pub fn optimal_codebook(dist: QuantizeDist, levels: usize) -> Result<Codebook> {
    if levels == 0 {
        return Err(Error::Domain("levels must be positive"));
    }

    if levels == 1 {
        return Ok(Codebook {
            boundaries: vec![],
            representatives: vec![0.0],
            iterations: 0,
        });
    }

    // All supported distributions are symmetric about 0. We exploit this by
    // computing only the positive half-codebook and mirroring.
    //
    // For even levels (n = 2*h): h positive levels in (0, inf), h negative.
    //   Positive boundaries: b_0=0, b_1, ..., b_{h-1}, b_h=+inf (or trunc).
    //   We iterate on b_1..b_{h-1} (h-1 interior boundaries).
    //
    // For odd levels (n = 2*h+1): one level straddles 0, h positive, h negative.
    //   Positive boundaries: b_0 (>0), b_1, ..., b_{h-1}, b_h=+inf (or trunc).
    //   We iterate on b_0..b_{h-1} (h boundaries).

    let n = levels;
    let even = n % 2 == 0;
    let h = n / 2; // number of positive-side levels

    let truncated = dist.needs_truncation();
    let outer = if truncated {
        dist.truncation_radius()
    } else {
        f64::INFINITY
    };

    let r = dist.init_radius();

    // Number of positive-side interior boundaries (excluding 0 and outer).
    // For even n=2h: positive side has h levels with boundaries [0, b1, ..., b_{h-1}, outer].
    //   Interior: h-1 boundaries.
    // For odd n=2h+1: positive side has h levels with boundaries [b0, b1, ..., b_{h-1}, outer].
    //   Plus one center level [-b0, b0]. Interior: h boundaries (b0..b_{h-1}).
    let nb = if even { h - 1 } else { h };

    // Initialize positive interior boundaries uniformly in (0, r).
    let mut pos_b: Vec<f64> = (0..nb)
        .map(|i| r * (i as f64 + 1.0) / (nb as f64 + 1.0))
        .collect();

    let max_iter = 1000;
    let tol = 1e-8;
    let mut pos_r = vec![0.0; h]; // positive-side representatives
    let mut iters = 0;

    for iter in 0..max_iter {
        iters = iter + 1;

        // Build full positive boundary list: left_edge, interior..., outer.
        let left_edge = if even { 0.0 } else { pos_b[0] };
        let mut full_b = Vec::with_capacity(h + 1);
        full_b.push(left_edge);
        if even {
            full_b.extend_from_slice(&pos_b);
        } else {
            full_b.extend_from_slice(&pos_b[1..]);
        }
        full_b.push(outer);
        // full_b has h+1 entries: h+1 boundaries define h intervals.

        for i in 0..h {
            pos_r[i] = conditional_mean(dist, full_b[i], full_b[i + 1]);
        }

        // Update interior boundaries as midpoints of adjacent representatives.
        let mut max_delta = 0.0_f64;

        if even {
            // pos_b has h-1 entries = boundaries between h positive levels.
            // pos_b[i] = midpoint(pos_r[i], pos_r[i+1]).
            for i in 0..nb {
                let new_b = 0.5 * (pos_r[i] + pos_r[i + 1]);
                max_delta = max_delta.max((new_b - pos_b[i]).abs());
                pos_b[i] = new_b;
            }
        } else {
            // pos_b has h entries. pos_b[0] is the boundary between the center
            // level and the first positive level.
            // Center representative = conditional_mean(dist, -pos_b[0], pos_b[0]).
            // For a symmetric dist, this is 0.
            // So pos_b[0] = midpoint(0, pos_r[0]) = pos_r[0] / 2.
            let new_b0 = pos_r[0] / 2.0;
            max_delta = max_delta.max((new_b0 - pos_b[0]).abs());
            pos_b[0] = new_b0;
            for i in 1..nb {
                let new_b = 0.5 * (pos_r[i - 1] + pos_r[i]);
                max_delta = max_delta.max((new_b - pos_b[i]).abs());
                pos_b[i] = new_b;
            }
        }

        if max_delta < tol {
            break;
        }
    }

    // Assemble full codebook by mirroring.
    let mut representatives = Vec::with_capacity(n);
    let mut boundaries = Vec::with_capacity(n - 1);

    // Negative side (mirrored, reversed).
    for i in (0..h).rev() {
        representatives.push(-pos_r[i]);
    }
    // Center level for odd n.
    if !even {
        representatives.push(0.0);
    }
    // Positive side.
    for i in 0..h {
        representatives.push(pos_r[i]);
    }

    // Boundaries: mirror of positive interior, then 0 (for even), then positive.
    if even {
        for i in (0..nb).rev() {
            boundaries.push(-pos_b[i]);
        }
        boundaries.push(0.0);
        for i in 0..nb {
            boundaries.push(pos_b[i]);
        }
    } else {
        // Odd: boundaries are [-pos_b[h-1], ..., -pos_b[0], pos_b[0], ..., pos_b[h-1]].
        for i in (0..nb).rev() {
            boundaries.push(-pos_b[i]);
        }
        for i in 0..nb {
            boundaries.push(pos_b[i]);
        }
    }

    Ok(Codebook {
        boundaries,
        representatives,
        iterations: iters,
    })
}

/// Map a continuous value to the index of its quantization level.
///
/// Uses binary search on `boundaries` (length `levels - 1`).
///
/// # Examples
///
/// ```
/// use logp::quantize::{optimal_codebook, quantize, QuantizeDist};
///
/// let cb = optimal_codebook(QuantizeDist::Gaussian, 4).unwrap();
/// let idx = quantize(0.5, &cb.boundaries);
/// assert!(idx < 4);
/// ```
pub fn quantize(value: f64, boundaries: &[f64]) -> usize {
    boundaries.partition_point(|&b| b <= value)
}

/// Map a quantization level index back to its representative value.
///
/// # Panics
///
/// Panics if `level >= representatives.len()`.
pub fn dequantize(level: usize, representatives: &[f64]) -> f64 {
    representatives[level]
}

// ── Numerical integration helpers ───────────────────────────────────────────

/// Complementary error function, computed via Abramowitz & Stegun 7.1.26
/// rational approximation. Accuracy ~1.5e-7.
fn erfc(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

/// Conditional mean E[X | lo <= X < hi] for a given distribution.
///
/// Uses closed-form antiderivatives of `x * f(x)` where available.
fn conditional_mean(dist: QuantizeDist, lo: f64, hi: f64) -> f64 {
    let prob = dist.cdf(hi) - dist.cdf(lo);
    if prob < 1e-300 {
        return match (lo.is_finite(), hi.is_finite()) {
            (true, true) => 0.5 * (lo + hi),
            (false, true) => hi,
            (true, false) => lo,
            (false, false) => 0.0,
        };
    }

    // Numerator = integral_{lo}^{hi} x * f(x) dx = F1(hi) - F1(lo)
    // where F1 is the antiderivative of x*f(x), defined per distribution.
    let num = first_moment_antideriv(dist, hi) - first_moment_antideriv(dist, lo);
    num / prob
}

/// Antiderivative of `x * f(x)`, evaluated at `x`.
///
/// For each distribution, this is a function `F1(x)` such that
/// `F1'(x) = x * f(x)`, and the integral from `a` to `b` is `F1(b) - F1(a)`.
///
/// The constant of integration is chosen so `F1(-inf) = 0` where meaningful,
/// or `F1` is only used for finite arguments (Cauchy).
fn first_moment_antideriv(dist: QuantizeDist, x: f64) -> f64 {
    match dist {
        // d/dx [-phi(x)] = -phi'(x) = x * phi(x). So F1(x) = -phi(x).
        // At -inf: -phi(-inf) = 0. At +inf: -phi(+inf) = 0.
        QuantizeDist::Gaussian => {
            if !x.is_finite() {
                return 0.0;
            }
            -dist.pdf(x)
        }

        // By integration by parts:
        //   integral x*f(x) dx = x*F(x) - integral F(x) dx
        //   integral sigmoid(t) dt = softplus(t) = ln(1 + e^t)
        //
        // So: F1(x) = x*F(x) - softplus(x)  [up to a constant]
        //           = x*sigmoid(x) - ln(1 + e^x)
        //
        // Check derivative: d/dx [x*sigmoid(x) - ln(1+e^x)]
        //   = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x)) - sigmoid(x) = x*f(x). Correct.
        //
        // At x -> -inf: x*sigmoid(x) ~ x*e^x -> 0. ln(1+e^x) ~ e^x -> 0. F1 -> 0.
        // At x -> +inf: x*sigmoid(x) ~ x. ln(1+e^x) ~ x. F1 -> 0.
        QuantizeDist::Logistic => {
            if !x.is_finite() {
                return 0.0;
            }
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            // softplus(x) = ln(1 + e^x), computed stably.
            let softplus = if x > 20.0 {
                x + (-x).exp() // x + tiny
            } else if x < -20.0 {
                x.exp() // tiny
            } else {
                (1.0 + x.exp()).ln()
            };
            x * sigmoid - softplus
        }

        // integral x/(pi*(1+x^2)) dx = ln(1+x^2) / (2*pi) + C.
        // This diverges at +/- inf, so it must only be called with finite x.
        QuantizeDist::Cauchy => {
            debug_assert!(x.is_finite(), "Cauchy F1 must use finite bounds");
            (1.0 + x * x).ln() / (2.0 * PI)
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_level_counts() {
        for n in [1, 2, 3, 4, 8, 16] {
            let cb = optimal_codebook(QuantizeDist::Gaussian, n).unwrap();
            assert_eq!(cb.representatives.len(), n);
            assert_eq!(cb.boundaries.len(), n.saturating_sub(1));
        }
    }

    #[test]
    fn zero_levels_is_error() {
        assert!(optimal_codebook(QuantizeDist::Gaussian, 0).is_err());
    }

    #[test]
    fn boundaries_are_sorted() {
        for dist in [
            QuantizeDist::Gaussian,
            QuantizeDist::Logistic,
            QuantizeDist::Cauchy,
        ] {
            let cb = optimal_codebook(dist, 8).unwrap();
            for w in cb.boundaries.windows(2) {
                assert!(w[0] < w[1], "unsorted boundaries for {dist:?}: {w:?}");
            }
        }
    }

    #[test]
    fn representatives_within_boundaries() {
        for dist in [
            QuantizeDist::Gaussian,
            QuantizeDist::Logistic,
            QuantizeDist::Cauchy,
        ] {
            let cb = optimal_codebook(dist, 8).unwrap();
            let n = cb.representatives.len();
            for i in 0..n {
                let lo = if i == 0 {
                    f64::NEG_INFINITY
                } else {
                    cb.boundaries[i - 1]
                };
                let hi = if i == n - 1 {
                    f64::INFINITY
                } else {
                    cb.boundaries[i]
                };
                let r = cb.representatives[i];
                assert!(
                    r > lo && r < hi,
                    "rep {r} not in ({lo}, {hi}) for {dist:?} level {i}"
                );
            }
        }
    }

    #[test]
    fn gaussian_two_level_known_values() {
        // For N(0,1) with 2 levels, the optimal codebook is:
        //   boundary = [0], representatives = [-E[|X|], E[|X|]]
        //   E[|X|] = sqrt(2/pi) ~ 0.7978845608
        let cb = optimal_codebook(QuantizeDist::Gaussian, 2).unwrap();
        assert_eq!(cb.boundaries.len(), 1);
        assert!(
            cb.boundaries[0].abs() < 1e-6,
            "boundary should be ~0, got {}",
            cb.boundaries[0]
        );

        let expected = (2.0 / PI).sqrt();
        assert!(
            (cb.representatives[0] + expected).abs() < 1e-5,
            "low rep: expected {}, got {}",
            -expected,
            cb.representatives[0]
        );
        assert!(
            (cb.representatives[1] - expected).abs() < 1e-5,
            "high rep: expected {expected}, got {}",
            cb.representatives[1]
        );
    }

    #[test]
    fn gaussian_four_level_known_values() {
        // Known optimal 4-level Gaussian quantizer (Lloyd-Max):
        //   boundaries: [-0.9816, 0, 0.9816]
        //   representatives: [-1.510, -0.4528, 0.4528, 1.510]
        let cb = optimal_codebook(QuantizeDist::Gaussian, 4).unwrap();

        // Check symmetry: boundary[1] ~ 0.
        assert!(
            cb.boundaries[1].abs() < 1e-6,
            "middle boundary should be ~0"
        );

        // Check outer boundary.
        assert!(
            (cb.boundaries[2] - 0.9816).abs() < 0.01,
            "upper boundary: expected ~0.9816, got {}",
            cb.boundaries[2]
        );

        // Check outer representatives.
        assert!(
            (cb.representatives[3] - 1.510).abs() < 0.01,
            "high outer rep: expected ~1.510, got {}",
            cb.representatives[3]
        );

        // Check inner representatives.
        assert!(
            (cb.representatives[2] - 0.4528).abs() < 0.01,
            "high inner rep: expected ~0.4528, got {}",
            cb.representatives[2]
        );
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let cb = optimal_codebook(QuantizeDist::Gaussian, 8).unwrap();
        for (i, &r) in cb.representatives.iter().enumerate() {
            let idx = quantize(r, &cb.boundaries);
            assert_eq!(
                idx, i,
                "representative {r} should map to level {i}, got {idx}"
            );
            let recovered = dequantize(idx, &cb.representatives);
            assert!(
                (recovered - r).abs() < 1e-14,
                "dequantize(quantize(rep)) should be exact"
            );
        }
    }

    #[test]
    fn symmetric_codebook() {
        for dist in [
            QuantizeDist::Gaussian,
            QuantizeDist::Logistic,
            QuantizeDist::Cauchy,
        ] {
            let cb = optimal_codebook(dist, 8).unwrap();
            let n = cb.representatives.len();
            for i in 0..n / 2 {
                let lo = cb.representatives[i];
                let hi = cb.representatives[n - 1 - i];
                assert!(
                    (lo + hi).abs() < 1e-6,
                    "{dist:?}: reps not symmetric: [{i}]={lo}, [{}]={hi}",
                    n - 1 - i
                );
            }
            for i in 0..cb.boundaries.len() / 2 {
                let lo = cb.boundaries[i];
                let hi = cb.boundaries[cb.boundaries.len() - 1 - i];
                assert!(
                    (lo + hi).abs() < 1e-6,
                    "{dist:?}: boundaries not symmetric: {lo} vs {hi}"
                );
            }
        }
    }

    #[test]
    fn cauchy_codebook_wider_than_gaussian() {
        let gauss = optimal_codebook(QuantizeDist::Gaussian, 8).unwrap();
        let cauchy = optimal_codebook(QuantizeDist::Cauchy, 8).unwrap();
        let g_outer = gauss.representatives.last().unwrap().abs();
        let c_outer = cauchy.representatives.last().unwrap().abs();
        assert!(
            c_outer > g_outer,
            "Cauchy outer rep ({c_outer}) should exceed Gaussian ({g_outer})"
        );
    }

    #[test]
    fn logistic_two_level() {
        let cb = optimal_codebook(QuantizeDist::Logistic, 2).unwrap();
        assert!(cb.boundaries[0].abs() < 1e-6);
        assert!(cb.representatives[0] < 0.0);
        assert!(cb.representatives[1] > 0.0);
        assert!((cb.representatives[0] + cb.representatives[1]).abs() < 1e-6);
        // E[X | X > 0] for standard logistic should be around 1.39 (2*ln2).
        let val = cb.representatives[1];
        assert!(
            val > 1.0 && val < 2.0,
            "logistic E[X|X>0] should be ~1.39, got {val}"
        );
    }

    #[test]
    fn convergence_within_reasonable_iterations() {
        for dist in [
            QuantizeDist::Gaussian,
            QuantizeDist::Logistic,
            QuantizeDist::Cauchy,
        ] {
            let cb = optimal_codebook(dist, 16).unwrap();
            assert!(
                cb.iterations < 1000,
                "{dist:?}: took {} iterations for 16 levels (did not converge)",
                cb.iterations
            );
        }
    }

    #[test]
    fn erfc_accuracy() {
        let cases = [
            (0.0, 1.0),
            (1.0, 0.157299207050285),
            (2.0, 0.004677734981047),
            (-1.0, 1.842700792949715),
        ];
        for (x, expected) in cases {
            let got = erfc(x);
            assert!(
                (got - expected).abs() < 2e-7,
                "erfc({x}): expected {expected}, got {got}"
            );
        }
    }
}
