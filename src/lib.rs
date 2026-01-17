//! # logp
//!
//! Information theory primitives: entropies and divergences.
//!
//! ## Scope
//!
//! This crate is **L1 (Logic)** in the Tekne stack: it should stay small and reusable.
//! It provides *scalar* information measures that appear across clustering, ranking,
//! evaluation, and geometry:
//!
//! - Shannon entropy and cross-entropy
//! - KL / Jensen–Shannon divergences
//! - Csiszár \(f\)-divergences (a.k.a. *information monotone* divergences)
//! - Bhattacharyya coefficient, Rényi/Tsallis families
//! - Bregman divergences (convex-analytic, not generally monotone)
//!
//! ## Distances vs divergences (terminology that prevents bugs)
//!
//! A **divergence** \(D(p:q)\) is usually required to satisfy:
//!
//! - \(D(p:q) \ge 0\)
//! - \(D(p:p) = 0\)
//!
//! but it is typically **not** symmetric and **not** a metric (no triangle inequality).
//! Many failures in downstream code are caused by treating a divergence as a distance metric.
//!
//! ## Key invariants (what tests should enforce)
//!
//! - **Jensen–Shannon** is bounded on the simplex:
//!   \(0 \le JS(p,q) \le \ln 2\) (nats).
//! - **Csiszár \(f\)-divergences** are monotone under coarse-graining (Markov kernels):
//!   merging bins cannot increase the divergence.
//!
//! ## Further reading
//!
//! - Frank Nielsen, “Divergences” portal (taxonomy diagrams + references):
//!   <https://franknielsen.github.io/Divergence/index.html>
//! - `nocotan/awesome-information-geometry` (curated reading list):
//!   <https://github.com/nocotan/awesome-information-geometry>
//! - Csiszár (1967): \(f\)-divergences and information monotonicity.
//! - Amari & Nagaoka (2000): *Methods of Information Geometry*.
//!
//! ## Taxonomy of Divergences (Nielsen)
//!
//! | Family | Generator | Key Property |
//! |---|---|---|
//! | **f-divergences** | Convex \(f(t)\) with \(f(1)=0\) | Monotone under Markov morphisms (coarse-graining) |
//! | **Bregman** | Convex \(F(x)\) | Dually flat geometry; generalized Pythagorean theorem |
//! | **Jensen-Shannon** | \(f\)-div + metric | Symmetric, bounded \([0, \ln 2]\), \(\sqrt{JS}\) is a metric |
//! | **Alpha** | \(\rho_\alpha = \int p^\alpha q^{1-\alpha}\) | Encodes Rényi, Tsallis, Bhattacharyya, Hellinger |
//!
//! ## Connections
//!
//! - [`rkhs`](../rkhs): MMD and KL both measure distribution "distance"
//! - [`wass`](../wass): Wasserstein vs entropy-based divergences
//! - [`stratify`](../stratify): NMI for cluster evaluation uses this crate
//! - [`fynch`](../fynch): Temperature scaling affects entropy calibration
//!
//! ## References
//!
//! - Shannon (1948). "A Mathematical Theory of Communication"
//! - Cover & Thomas (2006). "Elements of Information Theory"

#![forbid(unsafe_code)]

use thiserror::Error;

/// Natural log of 2. Useful when converting nats ↔ bits or bounding Jensen–Shannon.
pub const LN_2: f64 = core::f64::consts::LN_2;

/// KL Divergence between two diagonal Multivariate Gaussians.
///
/// Used for Variational Information Bottleneck (VIB) to regularize latent spaces.
///
/// Returns 0.5 * Σ [ (std1/std2)^2 + (mu2-mu1)^2 / std2^2 - 1 + 2*ln(std2/std1) ]
pub fn kl_divergence_gaussians(mu1: &[f64], std1: &[f64], mu2: &[f64], std2: &[f64]) -> Result<f64> {
    ensure_same_len(mu1, std1)?;
    ensure_same_len(mu1, mu2)?;
    ensure_same_len(mu1, std2)?;

    let mut kl = 0.0;
    for (((&m1, &s1), &m2), &s2) in mu1.iter().zip(std1).zip(mu2).zip(std2) {
        if s1 <= 0.0 || s2 <= 0.0 {
            return Err(Error::Domain("standard deviation must be positive"));
        }
        let v1 = s1 * s1;
        let v2 = s2 * s2;
        kl += (v1 / v2) + (m2 - m1).powi(2) / v2 - 1.0 + 2.0 * (s2.ln() - s1.ln());
    }
    Ok(0.5 * kl)
}

/// Errors for information-measure computations.
#[derive(Debug, Error)]
pub enum Error {
    #[error("length mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),

    #[error("empty input")]
    Empty,

    #[error("non-finite entry at index {idx}: {value}")]
    NonFinite { idx: usize, value: f64 },

    #[error("negative entry at index {idx}: {value}")]
    Negative { idx: usize, value: f64 },

    #[error("not normalized (expected sum≈1): sum={sum}")]
    NotNormalized { sum: f64 },

    #[error("invalid alpha: {alpha} (must be finite and not equal to {forbidden})")]
    InvalidAlpha { alpha: f64, forbidden: f64 },

    #[error("domain error: {0}")]
    Domain(&'static str),
}

pub type Result<T> = core::result::Result<T, Error>;

fn ensure_nonempty(x: &[f64]) -> Result<()> {
    if x.is_empty() {
        return Err(Error::Empty);
    }
    Ok(())
}

fn ensure_same_len(a: &[f64], b: &[f64]) -> Result<()> {
    if a.len() != b.len() {
        return Err(Error::LengthMismatch(a.len(), b.len()));
    }
    Ok(())
}

fn ensure_nonnegative(x: &[f64]) -> Result<()> {
    for (i, &v) in x.iter().enumerate() {
        if !v.is_finite() {
            return Err(Error::NonFinite { idx: i, value: v });
        }
        if v < 0.0 {
            return Err(Error::Negative { idx: i, value: v });
        }
    }
    Ok(())
}

fn sum(x: &[f64]) -> f64 {
    x.iter().sum()
}

/// Validate that `p` is a probability distribution on the simplex (within `tol`).
pub fn validate_simplex(p: &[f64], tol: f64) -> Result<()> {
    ensure_nonempty(p)?;
    ensure_nonnegative(p)?;
    let s = sum(p);
    if (s - 1.0).abs() > tol {
        return Err(Error::NotNormalized { sum: s });
    }
    Ok(())
}

/// Normalize a nonnegative vector in-place to sum to 1.
///
/// Returns the original sum.
pub fn normalize_in_place(p: &mut [f64]) -> Result<f64> {
    ensure_nonempty(p)?;
    ensure_nonnegative(p)?;
    let s = sum(p);
    if s <= 0.0 {
        return Err(Error::Domain("cannot normalize: sum <= 0"));
    }
    for v in p.iter_mut() {
        *v /= s;
    }
    Ok(s)
}

/// Shannon entropy \(H(p) = -\sum_i p_i \ln p_i\) (nats).
///
/// Requires `p` to be a valid simplex distribution (within `tol`).
pub fn entropy_nats(p: &[f64], tol: f64) -> Result<f64> {
    validate_simplex(p, tol)?;
    let mut h = 0.0;
    for &pi in p {
        if pi > 0.0 {
            h -= pi * pi.ln();
        }
    }
    Ok(h)
}

/// Shannon entropy in bits.
pub fn entropy_bits(p: &[f64], tol: f64) -> Result<f64> {
    Ok(entropy_nats(p, tol)? / LN_2)
}

/// Fast Shannon entropy calculation without simplex validation.
///
/// Used in performance-critical loops like Sinkhorn iteration for Optimal Transport.
///
/// # Invariant
/// Assumes `p` is non-negative and normalized.
#[inline]
pub fn entropy_unchecked(p: &[f64]) -> f64 {
    let mut h = 0.0;
    for &pi in p {
        if pi > 0.0 {
            h -= pi * pi.ln();
        }
    }
    h
}

/// Cross-entropy \(H(p,q) = -\sum_i p_i \ln q_i\) (nats).
///
/// Domain: `p` must be on the simplex; `q` must be nonnegative and normalized; and
/// whenever `p_i > 0`, we require `q_i > 0` (otherwise cross-entropy is infinite).
pub fn cross_entropy_nats(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    let mut h = 0.0;
    for (i, (&pi, &qi)) in p.iter().zip(q.iter()).enumerate() {
        if pi == 0.0 {
            continue;
        }
        if qi <= 0.0 {
            return Err(Error::Domain(match i {
                _ => "cross-entropy undefined: q_i=0 while p_i>0",
            }));
        }
        h -= pi * qi.ln();
    }
    Ok(h)
}

/// Kullback–Leibler divergence \(D_{KL}(p\|q) = \sum_i p_i \ln(p_i/q_i)\) (nats).
///
/// Domain: `p` and `q` must be valid simplex distributions; and whenever `p_i > 0`,
/// we require `q_i > 0`.
pub fn kl_divergence(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    let mut d = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi == 0.0 {
            continue;
        }
        if qi <= 0.0 {
            return Err(Error::Domain("KL undefined: q_i=0 while p_i>0"));
        }
        d += pi * (pi / qi).ln();
    }
    Ok(d)
}

/// Jensen–Shannon divergence (nats), defined as:
///
/// \(JS(p,q) = \tfrac12 KL(p\|m) + \tfrac12 KL(q\|m)\), where \(m = \tfrac12(p+q)\).
///
/// Domain: `p`, `q` must be simplex distributions.
///
/// Bound: \(0 \le JS(p,q) \le \ln 2\).
pub fn jensen_shannon_divergence(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;

    let mut m = vec![0.0; p.len()];
    for i in 0..p.len() {
        m[i] = 0.5 * (p[i] + q[i]);
    }

    Ok(0.5 * kl_divergence(p, &m, tol)? + 0.5 * kl_divergence(q, &m, tol)?)
}

/// Bhattacharyya coefficient \(BC(p,q) = \sum_i \sqrt{p_i q_i}\).
///
/// For simplex distributions, \(BC \in [0,1]\).
pub fn bhattacharyya_coeff(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    let bc: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi * qi).sqrt())
        .sum();
    Ok(bc)
}

/// Bhattacharyya distance \(D_B(p,q) = -\ln BC(p,q)\).
pub fn bhattacharyya_distance(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    let bc = bhattacharyya_coeff(p, q, tol)?;
    // When supports are disjoint, bc can be 0 (=> +∞ distance). Keep it explicit.
    if bc == 0.0 {
        return Err(Error::Domain("Bhattacharyya distance is infinite (BC=0)"));
    }
    Ok(-bc.ln())
}

/// Squared Hellinger distance: \(H^2(p,q) = 1 - \sum_i \sqrt{p_i q_i}\).
pub fn hellinger_squared(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    let bc = bhattacharyya_coeff(p, q, tol)?;
    Ok((1.0 - bc).max(0.0))
}

/// Hellinger distance \(H(p,q) = \sqrt{H^2(p,q)}\).
pub fn hellinger(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    Ok(hellinger_squared(p, q, tol)?.sqrt())
}

fn pow_nonneg(x: f64, a: f64) -> Result<f64> {
    if x < 0.0 || !x.is_finite() || !a.is_finite() {
        return Err(Error::Domain("pow_nonneg: invalid input"));
    }
    if x == 0.0 {
        if a == 0.0 {
            // By continuity in the divergence formulas, treat 0^0 as 1.
            return Ok(1.0);
        }
        if a > 0.0 {
            return Ok(0.0);
        }
        return Err(Error::Domain("0^a for a<0 is infinite"));
    }
    Ok(x.powf(a))
}

/// \(\rho_\alpha[p:q] = \sum_i p_i^\alpha q_i^{1-\alpha}\).
///
/// This coefficient underlies Rényi/Tsallis/Bhattacharyya/Chernoff families.
pub fn rho_alpha(p: &[f64], q: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    if !alpha.is_finite() {
        return Err(Error::InvalidAlpha {
            alpha,
            forbidden: f64::NAN,
        });
    }
    let mut s = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        let a = pow_nonneg(pi, alpha)?;
        let b = pow_nonneg(qi, 1.0 - alpha)?;
        s += a * b;
    }
    Ok(s)
}

/// Rényi divergence (nats):
///
/// \(D_\alpha^R(p\|q) = \frac{1}{\alpha-1}\ln \rho_\alpha[p:q]\), \(\alpha>0, \alpha \ne 1\).
pub fn renyi_divergence(p: &[f64], q: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    if alpha == 1.0 {
        return Err(Error::InvalidAlpha { alpha, forbidden: 1.0 });
    }
    let rho = rho_alpha(p, q, alpha, tol)?;
    if rho <= 0.0 {
        return Err(Error::Domain("rho_alpha <= 0"));
    }
    Ok(rho.ln() / (alpha - 1.0))
}

/// Tsallis divergence:
///
/// \(D_\alpha^T(p\|q) = \frac{\rho_\alpha[p:q] - 1}{\alpha-1}\), \(\alpha \ne 1\).
pub fn tsallis_divergence(p: &[f64], q: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    if alpha == 1.0 {
        return Err(Error::InvalidAlpha { alpha, forbidden: 1.0 });
    }
    Ok((rho_alpha(p, q, alpha, tol)? - 1.0) / (alpha - 1.0))
}

/// Amari \(\alpha\)-divergence (Amari parameter \(\alpha\in\mathbb{R}\)).
///
/// For \(\alpha \notin \{-1,1\}\):
/// \(D^\alpha[p:q] = \frac{4}{1-\alpha^2}\left(1 - \rho_{\frac{1-\alpha}{2}}[p:q]\right)\).
///
/// Limits:
/// - \(D^{-1}[p:q] = KL(p\|q)\)
/// - \(D^{1}[p:q] = KL(q\|p)\)
pub fn amari_alpha_divergence(p: &[f64], q: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    if !alpha.is_finite() {
        return Err(Error::InvalidAlpha {
            alpha,
            forbidden: f64::NAN,
        });
    }
    // Numerically stable handling near ±1.
    let eps = 1e-10;
    if (alpha + 1.0).abs() <= eps {
        return kl_divergence(p, q, tol);
    }
    if (alpha - 1.0).abs() <= eps {
        return kl_divergence(q, p, tol);
    }
    let t = (1.0 - alpha) / 2.0;
    let rho = rho_alpha(p, q, t, tol)?;
    Ok((4.0 / (1.0 - alpha * alpha)) * (1.0 - rho))
}

/// A Csiszár \(f\)-divergence with the standard form:
///
/// \(D_f(p\|q) = \sum_i q_i f(p_i / q_i)\).
///
/// When `q_i = 0`:
/// - if `p_i = 0`, the contribution is treated as 0 (by continuity).
/// - if `p_i > 0`, the divergence is infinite; we return an error.
pub fn csiszar_f_divergence(
    p: &[f64],
    q: &[f64],
    f: impl Fn(f64) -> f64,
    tol: f64,
) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;

    let mut d = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if qi == 0.0 {
            if pi == 0.0 {
                continue;
            }
            return Err(Error::Domain("f-divergence undefined: q_i=0 while p_i>0"));
        }
        d += qi * f(pi / qi);
    }
    Ok(d)
}

/// Bregman generator: a convex function \(F\) and its gradient.
pub trait BregmanGenerator {
    /// Evaluate the potential \(F(x)\).
    fn f(&self, x: &[f64]) -> Result<f64>;

    /// Write \(\nabla F(x)\) into `out`.
    fn grad_into(&self, x: &[f64], out: &mut [f64]) -> Result<()>;
}

/// Bregman divergence \(B_F(p,q) = F(p) - F(q) - \langle p-q, \nabla F(q)\rangle\).
pub fn bregman_divergence(
    gen: &impl BregmanGenerator,
    p: &[f64],
    q: &[f64],
    grad_q: &mut [f64],
) -> Result<f64> {
    ensure_nonempty(p)?;
    ensure_same_len(p, q)?;
    if grad_q.len() != q.len() {
        return Err(Error::LengthMismatch(grad_q.len(), q.len()));
    }
    gen.grad_into(q, grad_q)?;
    let fp = gen.f(p)?;
    let fq = gen.f(q)?;
    let mut inner = 0.0;
    for i in 0..p.len() {
        inner += (p[i] - q[i]) * grad_q[i];
    }
    Ok(fp - fq - inner)
}

/// Total Bregman divergence as shown in Nielsen’s taxonomy diagram:
///
/// \(tB_F(p,q) = \frac{B_F(p,q)}{\sqrt{1 + \|\nabla F(q)\|^2}}\).
pub fn total_bregman_divergence(
    gen: &impl BregmanGenerator,
    p: &[f64],
    q: &[f64],
    grad_q: &mut [f64],
) -> Result<f64> {
    let b = bregman_divergence(gen, p, q, grad_q)?;
    let grad_norm_sq: f64 = grad_q.iter().map(|&x| x * x).sum();
    Ok(b / (1.0 + grad_norm_sq).sqrt())
}

/// Squared Euclidean Bregman generator: \(F(x)=\tfrac12\|x\|_2^2\), \(\nabla F(x)=x\).
#[derive(Debug, Clone, Copy, Default)]
pub struct SquaredL2;

impl BregmanGenerator for SquaredL2 {
    fn f(&self, x: &[f64]) -> Result<f64> {
        ensure_nonempty(x)?;
        Ok(0.5 * x.iter().map(|&v| v * v).sum::<f64>())
    }

    fn grad_into(&self, x: &[f64], out: &mut [f64]) -> Result<()> {
        ensure_nonempty(x)?;
        if out.len() != x.len() {
            return Err(Error::LengthMismatch(out.len(), x.len()));
        }
        out.copy_from_slice(x);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const TOL: f64 = 1e-9;

    fn simplex_vec(len: usize) -> impl Strategy<Value = Vec<f64>> {
        // Draw nonnegative weights then normalize.
        prop::collection::vec(0.0f64..10.0, len).prop_map(|mut v| {
            let s: f64 = v.iter().sum();
            if s == 0.0 {
                v[0] = 1.0;
                return v;
            }
            for x in v.iter_mut() {
                *x /= s;
            }
            v
        })
    }

    fn simplex_vec_pos(len: usize, eps: f64) -> impl Strategy<Value = Vec<f64>> {
        prop::collection::vec(0.0f64..10.0, len).prop_map(move |mut v| {
            // Add a small floor to avoid exact zeros (needed for KL-style domains).
            for x in v.iter_mut() {
                *x += eps;
            }
            let s: f64 = v.iter().sum();
            for x in v.iter_mut() {
                *x /= s;
            }
            v
        })
    }

    fn random_partition(n: usize) -> impl Strategy<Value = Vec<usize>> {
        // Partition indices into k buckets (k chosen implicitly).
        // We generate a label in [0, n) for each index and later reindex to compact labels.
        prop::collection::vec(0usize..n, n).prop_map(|labels| {
            // Compress labels to 0..k-1 while preserving equality pattern.
            use std::collections::BTreeMap;
            let mut map = BTreeMap::<usize, usize>::new();
            let mut next = 0usize;
            labels
                .into_iter()
                .map(|l| {
                    *map.entry(l).or_insert_with(|| {
                        let id = next;
                        next += 1;
                        id
                    })
                })
                .collect::<Vec<_>>()
        })
    }

    fn coarse_grain(p: &[f64], labels: &[usize]) -> Vec<f64> {
        let k = labels.iter().copied().max().unwrap_or(0) + 1;
        let mut out = vec![0.0; k];
        for (i, &lab) in labels.iter().enumerate() {
            out[lab] += p[i];
        }
        out
    }

    #[test]
    fn test_entropy_unchecked() {
        let p = [0.5, 0.5];
        let h = entropy_unchecked(&p);
        // -0.5*ln(0.5) - 0.5*ln(0.5) = -ln(0.5) = ln(2)
        assert!((h - LN_2).abs() < 1e-12);
    }

    #[test]
    fn js_is_bounded_by_ln2() {
        let p = [1.0, 0.0];
        let q = [0.0, 1.0];
        let js = jensen_shannon_divergence(&p, &q, TOL).unwrap();
        assert!(js <= LN_2 + 1e-12);
        assert!(js >= 0.0);
    }

    #[test]
    fn bregman_squared_l2_matches_half_l2() {
        let gen = SquaredL2;
        let p = [1.0, 2.0, 3.0];
        let q = [1.5, 1.5, 2.5];
        let mut grad = [0.0; 3];
        let b = bregman_divergence(&gen, &p, &q, &mut grad).unwrap();
        let expected = 0.5 * p
            .iter()
            .zip(q.iter())
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f64>();
        assert!((b - expected).abs() < 1e-12);
    }

    proptest! {
        #[test]
        fn kl_is_nonnegative(p in simplex_vec_pos(8, 1e-6), q in simplex_vec_pos(8, 1e-6)) {
            let d = kl_divergence(&p, &q, 1e-6).unwrap();
            prop_assert!(d >= -1e-12);
        }

        #[test]
        fn js_is_bounded(p in simplex_vec(16), q in simplex_vec(16)) {
            let js = jensen_shannon_divergence(&p, &q, 1e-6).unwrap();
            prop_assert!(js >= -1e-12);
            prop_assert!(js <= LN_2 + 1e-9);
        }

        #[test]
        fn prop_kl_gaussians_is_nonnegative(
            mu1 in prop::collection::vec(-10.0f64..10.0, 1..16),
            std1 in prop::collection::vec(0.1f64..5.0, 1..16),
            mu2 in prop::collection::vec(-10.0f64..10.0, 1..16),
            std2 in prop::collection::vec(0.1f64..5.0, 1..16),
        ) {
            let n = mu1.len().min(std1.len()).min(mu2.len()).min(std2.len());
            let d = kl_divergence_gaussians(&mu1[..n], &std1[..n], &mu2[..n], &std2[..n]).unwrap();
            // KL divergence is always non-negative.
            prop_assert!(d >= -1e-12);
        }

        #[test]
        fn prop_kl_gaussians_is_zero_for_identical(
            mu in prop::collection::vec(-10.0f64..10.0, 1..16),
            std in prop::collection::vec(0.1f64..5.0, 1..16),
        ) {
            let n = mu.len().min(std.len());
            let d = kl_divergence_gaussians(&mu[..n], &std[..n], &mu[..n], &std[..n]).unwrap();
            prop_assert!(d.abs() < 1e-12);
        }

        #[test]
        fn f_divergence_monotone_under_coarse_graining(
            p in simplex_vec_pos(12, 1e-6),
            q in simplex_vec_pos(12, 1e-6),
            labels in random_partition(12),
        ) {
            // Use KL as an f-divergence instance: f(t)=t ln t.
            // D_KL(p||q) = Σ q_i f(p_i/q_i).
            let f = |t: f64| if t == 0.0 { 0.0 } else { t * t.ln() };
            let d_f = csiszar_f_divergence(&p, &q, f, 1e-6).unwrap();

            let pc = coarse_grain(&p, &labels);
            let qc = coarse_grain(&q, &labels);
            let d_fc = csiszar_f_divergence(&pc, &qc, f, 1e-6).unwrap();

            // Coarse graining should not increase.
            prop_assert!(d_fc <= d_f + 1e-9);
        }
    }
}
