//! # f-Divergences
//!
//! Generalized divergences between probability distributions.
//!
//! ## The f-Divergence Framework
//!
//! All f-divergences have the form:
//!
//! D_f(P || Q) = Σ q(x) f(p(x)/q(x))
//!
//! where f is a convex function with f(1) = 0.
//!
//! ## Key Divergences
//!
//! | Divergence | f(t) | Properties |
//! |------------|------|------------|
//! | KL | t log t | Asymmetric, unbounded |
//! | Reverse KL | -log t | Mode-seeking |
//! | χ² | (t-1)² | Squared difference |
//! | Hellinger | (√t - 1)² | Symmetric, bounded [0,2] |
//! | Total Variation | |t-1|/2 | Metric, bounded [0,1] |
//! | α-divergence | (t^α - 1)/(α(α-1)) | Interpolates KL |
//!
//! ## Connections
//!
//! - f-divergences are convex in (P, Q) → useful for optimization
//! - Variational bounds: D_f(P||Q) ≥ sup_T E_P[T] - E_Q[f*(T)]
//! - This is the basis for f-GAN training objectives
//!
//! ## Statistical Estimation
//!
//! Hellinger distance (and its tensorization) is a standard way to reason about
//! sample complexity for hypothesis testing between i.i.d. models.
//!
//! Concretely, if \(H^2(P, Q)\) is small, then \(P^{\otimes n}\) and \(Q^{\otimes n}\)
//! stay hard to distinguish until \(n \cdot H^2(P, Q)\) is order-1.
//!
//! - Hellinger: tensorizes nicely for i.i.d. samples → [`hellinger_squared_tensorized`]
//! - Fisher: governs Cramér-Rao bound → [`fisher_distance_approx`]

use std::f64::consts::LN_2;

const EPSILON: f64 = 1e-12;

/// Rényi divergence of order α: D_α(P || Q) = (1/(α-1)) log Σ p(x)^α q(x)^(1-α)
///
/// Generalizes KL divergence. As α → 1, converges to KL.
///
/// # Special Cases
///
/// - α = 0: -log Q(support(P))
/// - α = 1/2: Related to Bhattacharyya distance
/// - α = 1: KL divergence (limit)
/// - α = 2: Related to χ² divergence
/// - α → ∞: log(max_x p(x)/q(x))
///
/// # Arguments
///
/// * `p` - Distribution P
/// * `q` - Distribution Q
/// * `alpha` - Order parameter (must not be 1)
///
/// # Example
///
/// ```rust
/// use surp::fdiv::renyi_divergence;
///
/// let p = [0.5, 0.5];
/// let q = [0.9, 0.1];
///
/// let d_half = renyi_divergence(&p, &q, 0.5);
/// let d_2 = renyi_divergence(&p, &q, 2.0);
///
/// // Higher α emphasizes tail behavior differences
/// assert!(d_2 > d_half);
/// ```
pub fn renyi_divergence(p: &[f64], q: &[f64], alpha: f64) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");
    assert!(
        (alpha - 1.0).abs() > 0.01,
        "α cannot equal 1 (use KL instead)"
    );

    let sum: f64 = p
        .iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > EPSILON && qi > EPSILON)
        .map(|(&pi, &qi)| pi.powf(alpha) * qi.powf(1.0 - alpha))
        .sum();

    if sum <= 0.0 {
        return f64::INFINITY;
    }

    sum.ln() / ((alpha - 1.0) * LN_2)
}

/// Chi-squared divergence: χ²(P || Q) = Σ (p(x) - q(x))² / q(x)
///
/// Related to Pearson's chi-squared test statistic.
/// Sensitive to differences where Q is small.
///
/// # Properties
///
/// - χ²(P || Q) ≥ 0, = 0 iff P = Q
/// - χ²(P || Q) ≥ 2 * TV(P, Q)² (Pinsker-like)
/// - Asymmetric: χ²(P || Q) ≠ χ²(Q || P)
///
/// # Example
///
/// ```rust
/// use surp::fdiv::chi_squared_divergence;
///
/// let p = [0.5, 0.5];
/// let q = [0.9, 0.1];
///
/// let chi2 = chi_squared_divergence(&p, &q);
/// assert!(chi2 > 0.0);
/// ```
pub fn chi_squared_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    p.iter()
        .zip(q.iter())
        .filter(|(_, &qi)| qi > EPSILON)
        .map(|(&pi, &qi)| (pi - qi).powi(2) / qi)
        .sum()
}

/// Symmetric chi-squared divergence: (χ²(P||Q) + χ²(Q||P)) / 2
pub fn chi_squared_symmetric(p: &[f64], q: &[f64]) -> f64 {
    (chi_squared_divergence(p, q) + chi_squared_divergence(q, p)) / 2.0
}

/// Hellinger distance: H(P, Q) = (1/√2) √Σ (√p(x) - √q(x))²
///
/// A symmetric, bounded metric on probability distributions.
///
/// # Properties
///
/// - 0 ≤ H(P, Q) ≤ 1
/// - H(P, Q) = H(Q, P)
/// - H² is an f-divergence with f(t) = (√t - 1)²
///
/// # Relationship to Other Divergences
///
/// - H²(P,Q) ≤ KL(P||Q)
/// - H²(P,Q) ≤ TV(P,Q) ≤ H(P,Q)√(2 - H²(P,Q))
/// - H²(P,Q) = 1 - BC(P,Q), where BC is Bhattacharyya coefficient
///
/// # Example
///
/// ```rust
/// use surp::fdiv::hellinger_distance;
///
/// let p = [0.5, 0.5];
/// let q = [0.9, 0.1];
///
/// let h = hellinger_distance(&p, &q);
/// assert!(h >= 0.0 && h <= 1.0);
/// ```
pub fn hellinger_distance(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    let sum: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi.sqrt() - qi.sqrt()).powi(2))
        .sum();

    (sum / 2.0).sqrt()
}

/// Hellinger distance squared (the f-divergence form).
pub fn hellinger_squared(p: &[f64], q: &[f64]) -> f64 {
    let h = hellinger_distance(p, q);
    h * h
}

/// Tensorization of squared Hellinger distance for i.i.d. product measures.
///
/// If \(H^2(P,Q)\) is the squared Hellinger distance between single-sample distributions,
/// then for \(n\) i.i.d. samples:
///
/// ```text
/// H^2(P^{⊗n}, Q^{⊗n}) = 1 - (1 - H^2(P,Q))^n
/// ```
///
/// This identity is a workhorse behind two-point testing lower bounds. One common corollary is:
/// when \(n \cdot H^2(P,Q)\) is small, the product measures \(P^{\otimes n}\) and \(Q^{\otimes n}\)
/// remain close in total variation, hence hard to distinguish with constant advantage.
///
/// # Arguments
///
/// * `h2` - Squared Hellinger distance in [0, 1]
/// * `n` - Number of i.i.d. samples
pub fn hellinger_squared_tensorized(h2: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    // Be conservative: clamp to a valid range rather than panic in a metric helper.
    let h2 = h2.clamp(0.0, 1.0);
    if h2 == 0.0 {
        return 0.0;
    }
    if h2 == 1.0 {
        return 1.0;
    }

    // Compute 1 - (1 - h2)^n in a numerically stable-ish way.
    // For small h2 and large n, exp(n * ln(1-h2)) behaves better than powi on f64.
    let log_term = (1.0 - h2).ln();
    let a = (n as f64) * log_term; // <= 0
                                   // 1 - exp(a) = -expm1(a), which is more stable when a is close to 0.
    (-a.exp_m1()).clamp(0.0, 1.0)
}

/// Bhattacharyya coefficient: BC(P, Q) = Σ √(p(x) q(x))
///
/// Measures overlap between distributions. BC ∈ [0, 1].
///
/// # Properties
///
/// - BC(P, Q) = 1 iff P = Q
/// - BC(P, Q) = 0 iff P and Q have disjoint support
/// - Related to Hellinger: H²(P,Q) = 1 - BC(P,Q)
///
/// # Example
///
/// ```rust
/// use surp::fdiv::bhattacharyya_coefficient;
///
/// let p = [0.5, 0.5];
/// let q = [0.5, 0.5];
///
/// let bc = bhattacharyya_coefficient(&p, &q);
/// assert!((bc - 1.0).abs() < 1e-10);  // Identical distributions
/// ```
pub fn bhattacharyya_coefficient(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi * qi).sqrt())
        .sum()
}

/// Bhattacharyya distance: -ln(BC(P, Q))
pub fn bhattacharyya_distance(p: &[f64], q: &[f64]) -> f64 {
    -bhattacharyya_coefficient(p, q).ln()
}

/// Total variation distance: TV(P, Q) = (1/2) Σ |p(x) - q(x)|
///
/// The most natural metric on probability distributions.
///
/// # Properties
///
/// - TV(P, Q) ∈ [0, 1]
/// - TV(P, Q) = sup_A |P(A) - Q(A)| (variational form)
/// - TV is a proper metric (satisfies triangle inequality)
///
/// # Relationship to Other Divergences
///
/// - TV²(P,Q) ≤ KL(P||Q)/2 (Pinsker's inequality)
/// - TV(P,Q) ≤ H(P,Q) √(2 - H²(P,Q))
///
/// # Example
///
/// ```rust
/// use surp::fdiv::total_variation;
///
/// let p = [0.5, 0.5];
/// let q = [0.9, 0.1];
///
/// let tv = total_variation(&p, &q);
/// assert!((tv - 0.4).abs() < 1e-10);  // |0.5-0.9|/2 + |0.5-0.1|/2 = 0.4
/// ```
pub fn total_variation(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).abs())
        .sum::<f64>()
        / 2.0
}

/// α-divergence: D_α(P || Q) = (1/(α(α-1))) (Σ p^α q^(1-α) - 1)
///
/// A family that interpolates between forward and reverse KL.
///
/// # Special Cases
///
/// - α → 0: KL(Q || P) (reverse KL)
/// - α = 0.5: Symmetric, related to Hellinger
/// - α → 1: KL(P || Q) (forward KL)
///
/// # Example
///
/// ```rust
/// use surp::fdiv::alpha_divergence;
///
/// let p = [0.5, 0.5];
/// let q = [0.9, 0.1];
///
/// let d_half = alpha_divergence(&p, &q, 0.5);
/// let d_2 = alpha_divergence(&p, &q, 2.0);
/// ```
pub fn alpha_divergence(p: &[f64], q: &[f64], alpha: f64) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    if alpha.abs() < EPSILON {
        // α → 0: reverse KL
        return q
            .iter()
            .zip(p.iter())
            .filter(|(&qi, &pi)| qi > EPSILON && pi > EPSILON)
            .map(|(&qi, &pi)| qi * (qi / pi).ln())
            .sum::<f64>()
            / LN_2;
    }

    if (alpha - 1.0).abs() < EPSILON {
        // α → 1: forward KL
        return p
            .iter()
            .zip(q.iter())
            .filter(|(&pi, &qi)| pi > EPSILON && qi > EPSILON)
            .map(|(&pi, &qi)| pi * (pi / qi).ln())
            .sum::<f64>()
            / LN_2;
    }

    let sum: f64 = p
        .iter()
        .zip(q.iter())
        .filter(|(&pi, &qi)| pi > EPSILON && qi > EPSILON)
        .map(|(&pi, &qi)| pi.powf(alpha) * qi.powf(1.0 - alpha))
        .sum();

    (sum - 1.0) / (alpha * (alpha - 1.0))
}

/// Fisher information distance (approximation via Bhattacharyya).
///
/// The Fisher information metric defines a Riemannian geometry on
/// the space of probability distributions. The geodesic distance
/// under this metric is related to the Hellinger distance:
///
/// ```text
/// d_Fisher ≈ 2 * arccos(BC(P, Q))  (for distributions with same support)
/// ```
///
/// # Connection to Statistical Estimation
///
/// Fisher information governs the Cramér-Rao bound for unbiased estimators.
/// For smooth parametric families (e.g. a location family \(p(x-\mu)\)), the metric is
/// *local*: for small parameter shifts Δ, both Hellinger and Fisher-Rao distances scale
/// like \(\sqrt{I}\,|\Delta|\) where \(I\) is the Fisher information.
///
/// This is the regime where you can connect “estimation rate” heuristics to
/// two-point testing lower bounds. Outside that local regime, you should not treat
/// Fisher-Rao distance as a universal proxy for test difficulty.
///
/// See also: `hellinger_squared_tensorized` for the tensorization identity
/// that underlies many statistical lower bounds.
pub fn fisher_distance_approx(p: &[f64], q: &[f64]) -> f64 {
    let bc = bhattacharyya_coefficient(p, q);
    2.0 * bc.clamp(-1.0, 1.0).acos()
}

/// f-divergence with custom generator function.
///
/// D_f(P || Q) = Σ q(x) f(p(x)/q(x))
///
/// # Arguments
///
/// * `p` - Distribution P
/// * `q` - Distribution Q
/// * `f` - Generator function (convex, f(1) = 0)
///
/// # Example
///
/// ```rust
/// use surp::fdiv::f_divergence;
///
/// let p = [0.5, 0.5];
/// let q = [0.9, 0.1];
///
/// // Chi-squared: f(t) = (t-1)²
/// let chi2 = f_divergence(&p, &q, |t| (t - 1.0).powi(2));
/// ```
pub fn f_divergence<F>(p: &[f64], q: &[f64], f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    p.iter()
        .zip(q.iter())
        .filter(|(_, &qi)| qi > EPSILON)
        .map(|(&pi, &qi)| qi * f(pi / qi))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi_squared_non_negative() {
        let p = [0.3, 0.4, 0.3];
        let q = [0.5, 0.25, 0.25];
        let chi2 = chi_squared_divergence(&p, &q);
        assert!(chi2 >= 0.0);
    }

    #[test]
    fn test_chi_squared_zero_identical() {
        let p = [0.3, 0.4, 0.3];
        let chi2 = chi_squared_divergence(&p, &p);
        assert!(chi2 < 1e-10);
    }

    #[test]
    fn test_hellinger_bounds() {
        let p = [0.5, 0.5];
        let q = [0.9, 0.1];
        let h = hellinger_distance(&p, &q);
        assert!(h >= 0.0 && h <= 1.0);
    }

    #[test]
    fn test_hellinger_symmetric() {
        let p = [0.3, 0.4, 0.3];
        let q = [0.5, 0.25, 0.25];
        let h_pq = hellinger_distance(&p, &q);
        let h_qp = hellinger_distance(&q, &p);
        assert!((h_pq - h_qp).abs() < 1e-10);
    }

    #[test]
    fn test_total_variation_bounds() {
        let p = [0.5, 0.5];
        let q = [0.9, 0.1];
        let tv = total_variation(&p, &q);
        assert!(tv >= 0.0 && tv <= 1.0);
    }

    #[test]
    fn test_total_variation_value() {
        let p = [0.5, 0.5];
        let q = [0.9, 0.1];
        let tv = total_variation(&p, &q);
        // |0.5-0.9| + |0.5-0.1| = 0.4 + 0.4 = 0.8, then /2 = 0.4
        assert!((tv - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_bhattacharyya_identical() {
        let p = [0.3, 0.4, 0.3];
        let bc = bhattacharyya_coefficient(&p, &p);
        assert!((bc - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_hellinger_bhattacharyya_relation() {
        let p = [0.3, 0.4, 0.3];
        let q = [0.5, 0.25, 0.25];

        let h2 = hellinger_squared(&p, &q);
        let bc = bhattacharyya_coefficient(&p, &q);

        // H² = 1 - BC
        assert!((h2 - (1.0 - bc)).abs() < 1e-10);
    }

    #[test]
    fn test_hellinger_tensorization_edges() {
        assert_eq!(hellinger_squared_tensorized(0.0, 0), 0.0);
        assert_eq!(hellinger_squared_tensorized(0.0, 10), 0.0);
        assert_eq!(hellinger_squared_tensorized(1.0, 10), 1.0);
    }

    #[test]
    fn test_hellinger_tensorization_increases_with_n() {
        let h2 = 0.01;
        let a = hellinger_squared_tensorized(h2, 1);
        let b = hellinger_squared_tensorized(h2, 10);
        let c = hellinger_squared_tensorized(h2, 100);
        assert!(a <= b && b <= c);
        assert!(c <= 1.0);
    }

    #[test]
    fn test_hellinger_tensorization_stable_for_tiny_h2_large_n() {
        // This should exercise the expm1-based path without underflowing to 0.
        let h2 = 1e-12;
        let n = 1_000_000_000usize; // 1e9
        let got = hellinger_squared_tensorized(h2, n);
        assert!(got.is_finite());
        assert!(got >= 0.0 && got <= 1.0);

        // For tiny h2, log(1-h2) ≈ -h2, so: 1 - (1-h2)^n ≈ 1 - exp(-n*h2).
        let approx = -((-((n as f64) * h2)).exp_m1());
        // We only expect the approximation to be good when n*h2 is small (here: 1e-3).
        // The implementation uses `ln(1-h2)` (not `ln_1p(-h2)`), so don't demand ulp-level agreement.
        assert!((got - approx).abs() < 1e-7, "got={got} approx={approx}");
    }

    #[test]
    fn test_hellinger_tensorization_saturates_for_large_n() {
        let h2 = 0.01;
        let n = 1_000_000_000usize;
        let got = hellinger_squared_tensorized(h2, n);
        assert!(got.is_finite());
        assert!(got > 0.999_999, "expected near-1 saturation, got={got}");
    }

    #[test]
    fn test_renyi_positive() {
        let p = [0.3, 0.4, 0.3];
        let q = [0.5, 0.25, 0.25];

        let d = renyi_divergence(&p, &q, 2.0);
        assert!(d >= 0.0);
    }

    #[test]
    fn test_f_divergence_chi_squared() {
        let p = [0.5, 0.5];
        let q = [0.9, 0.1];

        let chi2_direct = chi_squared_divergence(&p, &q);
        let chi2_via_f = f_divergence(&p, &q, |t| (t - 1.0).powi(2));

        assert!((chi2_direct - chi2_via_f).abs() < 1e-10);
    }
}
