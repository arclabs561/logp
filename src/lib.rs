//! # logp
//!
//! Information theory primitives: entropies and divergences.
//!
//! ## Scope
//!
//! Scalar information measures that appear across clustering, ranking,
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
//! | **Jensen-Shannon** | \(f\)-div + metric | Symmetric, bounded \(\[0, \ln 2\]\), \(\sqrt{JS}\) is a metric |
//! | **Alpha** | \(\rho_\alpha = \int p^\alpha q^{1-\alpha}\) | Encodes Rényi, Tsallis, Bhattacharyya, Hellinger |
//!
//! ## References
//!
//! - Shannon (1948). "A Mathematical Theory of Communication"
//! - Cover & Thomas (2006). "Elements of Information Theory"

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use thiserror::Error;

mod ksg;
pub use ksg::{mutual_information_ksg, KsgVariant};

use core::f64::consts::LN_2;

/// KL Divergence between two diagonal Multivariate Gaussians.
///
/// Assumes **diagonal** covariance (independent dimensions): each dimension is
/// parameterized by a mean and a standard deviation, with no cross-covariance terms.
/// For full-covariance Gaussians, the formula involves log-determinant ratios and
/// matrix inverses; this function does not handle that case.
///
/// Returns 0.5 * Σ [ (std1/std2)^2 + (mu2-mu1)^2 / std2^2 - 1 + 2*ln(std2/std1) ]
///
/// # Examples
///
/// ```
/// # use logp::kl_divergence_gaussians;
/// // KL(N || N) = 0 (identical Gaussians).
/// let mu = [0.0, 1.0];
/// let std = [1.0, 2.0];
/// let kl = kl_divergence_gaussians(&mu, &std, &mu, &std).unwrap();
/// assert!(kl.abs() < 1e-12);
///
/// // KL is non-negative for distinct Gaussians.
/// let kl = kl_divergence_gaussians(&[0.0], &[1.0], &[1.0], &[1.0]).unwrap();
/// assert!(kl >= 0.0);
/// ```
pub fn kl_divergence_gaussians(
    mu1: &[f64],
    std1: &[f64],
    mu2: &[f64],
    std2: &[f64],
) -> Result<f64> {
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
    /// Two input slices have different lengths (e.g., `p` and `q` in a divergence).
    #[error("length mismatch: {0} vs {1}")]
    LengthMismatch(usize, usize),

    /// An input slice is empty where at least one element is required.
    #[error("empty input")]
    Empty,

    /// An entry is NaN or infinite where a finite value is required.
    #[error("non-finite entry at index {idx}: {value}")]
    NonFinite {
        /// Position of the offending entry.
        idx: usize,
        /// The non-finite value found.
        value: f64,
    },

    /// An entry is negative where a nonnegative value is required (probability distributions).
    #[error("negative entry at index {idx}: {value}")]
    Negative {
        /// Position of the offending entry.
        idx: usize,
        /// The negative value found.
        value: f64,
    },

    /// The distribution does not sum to 1 within the specified tolerance.
    #[error("not normalized (expected sum≈1): sum={sum}")]
    NotNormalized {
        /// Actual sum of the distribution.
        sum: f64,
    },

    /// The alpha parameter is outside its valid domain (e.g., negative or non-finite).
    #[error("invalid alpha: {alpha} (must be finite and not equal to {forbidden})")]
    InvalidAlpha {
        /// The invalid alpha value provided.
        alpha: f64,
        /// The value alpha must not equal (e.g., 0.0).
        forbidden: f64,
    },

    /// Catch-all for domain violations: zero standard deviation, q_i=0 while p_i>0,
    /// insufficient sample size for KSG, and similar precondition failures.
    #[error("domain error: {0}")]
    Domain(&'static str),
}

/// Convenience alias for `Result<T, logp::Error>`.
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
///
/// # Examples
///
/// ```
/// # use logp::validate_simplex;
/// // Valid simplex.
/// assert!(validate_simplex(&[0.3, 0.7], 1e-9).is_ok());
/// assert!(validate_simplex(&[1.0], 1e-9).is_ok());
///
/// // Rejects bad sum.
/// assert!(validate_simplex(&[0.3, 0.6], 1e-9).is_err());
///
/// // Rejects negative entries.
/// assert!(validate_simplex(&[1.5, -0.5], 1e-9).is_err());
///
/// // Rejects empty input.
/// assert!(validate_simplex(&[], 1e-9).is_err());
/// ```
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
///
/// # Examples
///
/// ```
/// # use logp::normalize_in_place;
/// let mut v = vec![2.0, 3.0, 5.0];
/// let original_sum = normalize_in_place(&mut v).unwrap();
/// assert!((original_sum - 10.0).abs() < 1e-12);
/// assert!((v[0] - 0.2).abs() < 1e-12);
/// assert!((v[1] - 0.3).abs() < 1e-12);
/// assert!((v[2] - 0.5).abs() < 1e-12);
///
/// // Rejects all-zero input.
/// assert!(normalize_in_place(&mut vec![0.0, 0.0]).is_err());
/// ```
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

/// Shannon entropy in nats: the expected surprise under distribution \(p\).
///
/// \[H(p) = -\sum_i p_i \ln p_i\]
///
/// # Key properties
///
/// - **Non-negative**: \(H(p) \ge 0\), with equality iff \(p\) is a delta (point mass).
/// - **Maximized by uniform**: among distributions on \(n\) outcomes,
///   \(H(p) \le \ln n\), with equality iff \(p_i = 1/n\) for all \(i\).
/// - **Concavity**: \(H\) is a concave function of \(p\) on the simplex.
///   Mixing distributions never decreases entropy.
/// - **Units**: result is in nats (base \(e\)); divide by \(\ln 2\) for bits.
/// - **Joint distributions**: for joint entropy \(H(X,Y)\), flatten the joint
///   distribution to a 1D slice and pass it directly. Shannon entropy of the
///   flattened joint is mathematically identical to joint entropy.
///
/// # Domain
///
/// Requires `p` to be a valid simplex distribution (within `tol`).
///
/// # Examples
///
/// ```
/// # use logp::entropy_nats;
/// // Uniform distribution over 4 outcomes: H = ln(4).
/// let p = [0.25, 0.25, 0.25, 0.25];
/// let h = entropy_nats(&p, 1e-9).unwrap();
/// assert!((h - 4.0_f64.ln()).abs() < 1e-12);
///
/// // Delta (point mass): H = 0.
/// let delta = [1.0, 0.0, 0.0];
/// assert!(entropy_nats(&delta, 1e-9).unwrap().abs() < 1e-15);
/// ```
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
///
/// # Examples
///
/// ```
/// # use logp::{entropy_bits, entropy_nats};
/// // Fair coin: H = 1 bit.
/// let p = [0.5, 0.5];
/// let bits = entropy_bits(&p, 1e-9).unwrap();
/// assert!((bits - 1.0).abs() < 1e-12);
///
/// // Consistent with nats / ln(2).
/// let nats = entropy_nats(&p, 1e-9).unwrap();
/// assert!((bits - nats / core::f64::consts::LN_2).abs() < 1e-12);
/// ```
pub fn entropy_bits(p: &[f64], tol: f64) -> Result<f64> {
    Ok(entropy_nats(p, tol)? / LN_2)
}

/// Fast Shannon entropy calculation without simplex validation.
///
/// Used in performance-critical loops like Sinkhorn iteration for Optimal Transport.
///
/// # Invariant
/// Assumes `p` is non-negative and normalized.
///
/// # Examples
///
/// ```
/// # use logp::entropy_unchecked;
/// // Fair coin: H = ln(2).
/// let h = entropy_unchecked(&[0.5, 0.5]);
/// assert!((h - core::f64::consts::LN_2).abs() < 1e-12);
///
/// // Agrees with the checked version on valid input.
/// let p = [0.3, 0.7];
/// let h_checked = logp::entropy_nats(&p, 1e-9).unwrap();
/// assert!((entropy_unchecked(&p) - h_checked).abs() < 1e-15);
/// ```
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

/// Renyi entropy in nats: a one-parameter generalization of Shannon entropy.
///
/// \[H_\alpha(p) = \frac{1}{1-\alpha} \ln \sum_i p_i^\alpha, \quad \alpha > 0,\;\alpha \ne 1\]
///
/// # Key properties
///
/// - **Limit to Shannon**: \(\lim_{\alpha \to 1} H_\alpha(p) = H(p)\) (Shannon entropy).
/// - **Alpha = 0**: \(H_0(p) = \ln |\text{supp}(p)|\), the log of the support size (Hartley entropy).
/// - **Alpha = 2**: \(H_2(p) = -\ln \sum_i p_i^2\), the collision entropy (negative log of
///   the probability that two independent draws match).
/// - **Alpha = infinity**: \(H_\infty(p) = -\ln \max_i p_i\), the min-entropy (worst-case surprise).
/// - **Monotone in alpha**: \(H_\alpha(p)\) is non-increasing in \(\alpha\).
/// - **Non-negative**: \(H_\alpha(p) \ge 0\).
///
/// # Examples
///
/// ```
/// # use logp::renyi_entropy;
/// // Uniform over 4: H_alpha = ln(4) for all alpha.
/// let p = [0.25, 0.25, 0.25, 0.25];
/// let h = renyi_entropy(&p, 2.0, 1e-9).unwrap();
/// assert!((h - 4.0_f64.ln()).abs() < 1e-12);
///
/// // Collision entropy: H_2 = -ln(sum(p_i^2)).
/// let q = [0.3, 0.7];
/// let h2 = renyi_entropy(&q, 2.0, 1e-9).unwrap();
/// let expected = -(0.3_f64.powi(2) + 0.7_f64.powi(2)).ln();
/// assert!((h2 - expected).abs() < 1e-12);
/// ```
pub fn renyi_entropy(p: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    validate_simplex(p, tol)?;
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(Error::InvalidAlpha {
            alpha,
            forbidden: f64::NAN,
        });
    }
    if (alpha - 1.0).abs() < 1e-12 {
        return entropy_nats(p, tol);
    }
    let mut s = 0.0;
    for &pi in p {
        if pi > 0.0 {
            s += pi.powf(alpha);
        }
    }
    if s <= 0.0 {
        return Err(Error::Domain("renyi_entropy: sum of p_i^alpha <= 0"));
    }
    Ok(s.ln() / (1.0 - alpha))
}

/// Tsallis entropy: a non-extensive generalization of Shannon entropy from
/// statistical mechanics.
///
/// \[S_\alpha(p) = \frac{1}{\alpha - 1}\left(1 - \sum_i p_i^\alpha\right), \quad \alpha \ne 1\]
///
/// # Key properties
///
/// - **Limit to Shannon**: \(\lim_{\alpha \to 1} S_\alpha(p) = H(p)\).
/// - **Non-extensive**: for independent systems \(A, B\):
///   \(S_\alpha(A \otimes B) = S_\alpha(A) + S_\alpha(B) + (1-\alpha)\,S_\alpha(A)\,S_\alpha(B)\).
/// - **Non-negative**: \(S_\alpha(p) \ge 0\).
/// - **Connection to Renyi**: related via \(S_\alpha = \frac{e^{(1-\alpha)H_\alpha} - 1}{\alpha - 1}\).
///
/// # Examples
///
/// ```
/// # use logp::tsallis_entropy;
/// // Uniform over 4: S_2 = 1 - 1/4 = 0.75.
/// let p = [0.25, 0.25, 0.25, 0.25];
/// let s = tsallis_entropy(&p, 2.0, 1e-9).unwrap();
/// assert!((s - 0.75).abs() < 1e-12);
///
/// // Delta: S_alpha = 0 for any alpha.
/// let delta = [1.0, 0.0, 0.0];
/// assert!(tsallis_entropy(&delta, 2.0, 1e-9).unwrap().abs() < 1e-12);
/// ```
pub fn tsallis_entropy(p: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    validate_simplex(p, tol)?;
    if !alpha.is_finite() || alpha < 0.0 {
        return Err(Error::InvalidAlpha {
            alpha,
            forbidden: f64::NAN,
        });
    }
    if (alpha - 1.0).abs() < 1e-12 {
        return entropy_nats(p, tol);
    }
    let mut s = 0.0;
    for &pi in p {
        if pi > 0.0 {
            s += pi.powf(alpha);
        }
    }
    Ok((1.0 - s) / (alpha - 1.0))
}

/// Cross-entropy in nats: the expected code length when using model \(q\) to encode
/// data drawn from true distribution \(p\).
///
/// \[H(p, q) = -\sum_i p_i \ln q_i\]
///
/// # Key properties
///
/// - **Decomposition identity**: cross-entropy splits into entropy plus KL divergence:
///   \(H(p, q) = H(p) + D_{KL}(p \| q)\).
///   This means \(H(p, q) \ge H(p)\) with equality iff \(p = q\).
/// - **Loss function**: minimizing \(H(p, q)\) over \(q\) is equivalent to minimizing
///   \(D_{KL}(p \| q)\), which is why cross-entropy is the standard classification loss.
/// - **Not symmetric**: \(H(p, q) \ne H(q, p)\) in general.
///
/// # Domain
///
/// `p` must be on the simplex; `q` must be nonnegative and normalized; and
/// whenever `p_i > 0`, we require `q_i > 0` (otherwise cross-entropy is infinite).
///
/// # Examples
///
/// ```
/// # use logp::{cross_entropy_nats, entropy_nats, kl_divergence};
/// let p = [0.3, 0.7];
/// let q = [0.5, 0.5];
/// let h_pq = cross_entropy_nats(&p, &q, 1e-9).unwrap();
///
/// // Decomposition: H(p,q) = H(p) + KL(p||q).
/// let h_p = entropy_nats(&p, 1e-9).unwrap();
/// let kl = kl_divergence(&p, &q, 1e-9).unwrap();
/// assert!((h_pq - (h_p + kl)).abs() < 1e-12);
///
/// // Self-cross-entropy equals entropy: H(p,p) = H(p).
/// let h_pp = cross_entropy_nats(&p, &p, 1e-9).unwrap();
/// assert!((h_pp - h_p).abs() < 1e-12);
/// ```
pub fn cross_entropy_nats(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    let mut h = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if pi == 0.0 {
            continue;
        }
        if qi <= 0.0 {
            return Err(Error::Domain("cross-entropy undefined: q_i=0 while p_i>0"));
        }
        h -= pi * qi.ln();
    }
    Ok(h)
}

/// Kullback--Leibler divergence in nats: the information lost when \(q\) is used to
/// approximate \(p\).
///
/// \[D_{KL}(p \| q) = \sum_i p_i \ln \frac{p_i}{q_i}\]
///
/// # Key properties
///
/// - **Gibbs' inequality**: \(D_{KL}(p \| q) \ge 0\), with equality iff \(p = q\).
///   This follows directly from Jensen's inequality applied to \(-\ln\).
/// - **Not symmetric**: \(D_{KL}(p \| q) \ne D_{KL}(q \| p)\) in general;
///   this is why KL is a divergence, not a distance.
/// - **Not bounded above**: KL can be arbitrarily large when supports differ.
/// - **Connection to MLE**: minimizing \(D_{KL}(p_{data} \| q_\theta)\) over \(\theta\)
///   is equivalent to maximum likelihood estimation.
/// - **Additive for independent distributions**: if \(p = p_1 \otimes p_2\) and
///   \(q = q_1 \otimes q_2\), then
///   \(D_{KL}(p \| q) = D_{KL}(p_1 \| q_1) + D_{KL}(p_2 \| q_2)\).
///
/// # Domain
///
/// `p` and `q` must be valid simplex distributions; and whenever `p_i > 0`,
/// we require `q_i > 0`.
///
/// # Examples
///
/// ```
/// # use logp::kl_divergence;
/// // KL(p || p) = 0 (Gibbs' inequality, tight case).
/// let p = [0.2, 0.3, 0.5];
/// assert!(kl_divergence(&p, &p, 1e-9).unwrap().abs() < 1e-15);
///
/// // KL is non-negative.
/// let q = [0.5, 0.25, 0.25];
/// assert!(kl_divergence(&p, &q, 1e-9).unwrap() >= 0.0);
///
/// // Not symmetric in general.
/// let kl_pq = kl_divergence(&p, &q, 1e-9).unwrap();
/// let kl_qp = kl_divergence(&q, &p, 1e-9).unwrap();
/// assert!((kl_pq - kl_qp).abs() > 1e-6);
/// ```
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

/// Jensen--Shannon divergence in nats: a symmetric, bounded smoothing of KL divergence.
///
/// \[JS(p, q) = \tfrac{1}{2} D_{KL}(p \| m) + \tfrac{1}{2} D_{KL}(q \| m), \quad m = \tfrac{1}{2}(p + q)\]
///
/// # Key properties
///
/// - **Symmetric**: \(JS(p, q) = JS(q, p)\), unlike KL.
/// - **Bounded**: \(0 \le JS(p, q) \le \ln 2\). The upper bound is attained when \(p\)
///   and \(q\) have disjoint supports.
/// - **Square root is a metric**: \(\sqrt{JS(p, q)}\) satisfies the triangle inequality
///   (Endres & Schindelin, 2003), so it can be used as a proper distance function.
/// - **Connection to mutual information**: \(JS(p, q) = I(X; Z)\) where \(Z\) is a
///   fair coin selecting between \(p\) and \(q\), and \(X\) is drawn from the selected
///   distribution.
/// - **Always finite**: because \(m_i > 0\) whenever \(p_i > 0\) or \(q_i > 0\), the
///   KL terms are always well-defined (no division by zero).
///
/// # Domain
///
/// `p`, `q` must be simplex distributions.
///
/// # Examples
///
/// ```
/// # use logp::jensen_shannon_divergence;
/// // JS(p, p) = 0.
/// let p = [0.3, 0.7];
/// assert!(jensen_shannon_divergence(&p, &p, 1e-9).unwrap().abs() < 1e-15);
///
/// // Disjoint supports: JS = ln(2).
/// let a = [1.0, 0.0];
/// let b = [0.0, 1.0];
/// let js = jensen_shannon_divergence(&a, &b, 1e-9).unwrap();
/// assert!((js - core::f64::consts::LN_2).abs() < 1e-12);
///
/// // Symmetric: JS(p, q) = JS(q, p).
/// let q = [0.5, 0.5];
/// let js_pq = jensen_shannon_divergence(&p, &q, 1e-9).unwrap();
/// let js_qp = jensen_shannon_divergence(&q, &p, 1e-9).unwrap();
/// assert!((js_pq - js_qp).abs() < 1e-15);
/// ```
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

/// Weighted Jensen--Shannon divergence: a generalization that allows unequal mixture
/// weights.
///
/// \[JS_\pi(p, q) = \pi_1\,D_{KL}(p \| m) + \pi_2\,D_{KL}(q \| m), \quad
///   m = \pi_1 p + \pi_2 q\]
///
/// where \(\pi_1 + \pi_2 = 1\). At \(\pi_1 = \pi_2 = 0.5\) this reduces to
/// [`jensen_shannon_divergence`].
///
/// # Key properties
///
/// - **Bounded**: \(0 \le JS_\pi \le H(\pi)\) where \(H(\pi) = -\pi_1 \ln \pi_1 - \pi_2 \ln \pi_2\).
///   The standard bound \(\ln 2\) is the special case \(\pi_1 = \pi_2 = 0.5\).
/// - **Symmetric in \((p, \pi_1)\) and \((q, \pi_2)\)**: swapping both distributions
///   and weights gives the same value.
///
/// # Examples
///
/// ```
/// # use logp::{jensen_shannon_weighted, jensen_shannon_divergence};
/// let p = [0.3, 0.7];
/// let q = [0.5, 0.5];
///
/// // Equal weights recovers standard JS.
/// let jsw = jensen_shannon_weighted(&p, &q, 0.5, 1e-9).unwrap();
/// let js = jensen_shannon_divergence(&p, &q, 1e-9).unwrap();
/// assert!((jsw - js).abs() < 1e-12);
///
/// // Extreme weight: pi1=1 means m=p, so JS=0.
/// let js1 = jensen_shannon_weighted(&p, &q, 1.0, 1e-9).unwrap();
/// assert!(js1.abs() < 1e-12);
/// ```
pub fn jensen_shannon_weighted(p: &[f64], q: &[f64], pi1: f64, tol: f64) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    if !(0.0..=1.0).contains(&pi1) || !pi1.is_finite() {
        return Err(Error::Domain("pi1 must be in [0, 1]"));
    }
    let pi2 = 1.0 - pi1;

    let mut m = vec![0.0; p.len()];
    for i in 0..p.len() {
        m[i] = pi1 * p[i] + pi2 * q[i];
    }

    // When pi1 or pi2 is zero, the corresponding KL term is 0 * D_KL = 0.
    let kl_p = if pi1 > 0.0 {
        kl_divergence(p, &m, tol)?
    } else {
        0.0
    };
    let kl_q = if pi2 > 0.0 {
        kl_divergence(q, &m, tol)?
    } else {
        0.0
    };
    Ok(pi1 * kl_p + pi2 * kl_q)
}

/// Mutual information in nats: how much knowing \(Y\) reduces uncertainty about \(X\).
///
/// \[I(X; Y) = \sum_{x,y} p(x,y) \ln \frac{p(x,y)}{p(x)\,p(y)}\]
///
/// # Key properties
///
/// - **KL form**: \(I(X; Y) = D_{KL}\bigl(p(x,y) \;\|\; p(x)\,p(y)\bigr)\), measuring
///   how far the joint distribution is from the product of its marginals.
/// - **Non-negative**: \(I(X; Y) \ge 0\), with equality iff \(X\) and \(Y\) are
///   independent.
/// - **Symmetric**: \(I(X; Y) = I(Y; X)\).
/// - **Bounded by entropy**: \(I(X; Y) \le \min\{H(X),\, H(Y)\}\).
/// - **Data processing inequality**: for any Markov chain \(X \to Y \to Z\),
///   \(I(X; Z) \le I(X; Y)\). Processing cannot create information.
/// - **Entropy decomposition**: \(I(X; Y) = H(X) + H(Y) - H(X, Y) = H(X) - H(X|Y)\).
///
/// # Layout
///
/// For discrete distributions, given a **row-major** joint distribution table `p_xy`
/// with shape `(n_x, n_y)`.
///
/// Public invariant (this is the important one): this API is **backend-agnostic**.
/// It does not force `ndarray` into the public surface of a leaf crate.
///
/// # Examples
///
/// ```
/// # use logp::{mutual_information, entropy_nats};
/// // Independent joint: p(x,y) = p(x)*p(y), so I(X;Y) = 0.
/// let p_xy = [0.15, 0.35, 0.15, 0.35]; // 2x2, marginals [0.5,0.5] x [0.3,0.7]
/// let mi = mutual_information(&p_xy, 2, 2, 1e-9).unwrap();
/// assert!(mi.abs() < 1e-12);
///
/// // Perfect correlation (Y = X, uniform bit): I(X;Y) = H(X) = ln(2).
/// let diag = [0.5, 0.0, 0.0, 0.5];
/// let mi = mutual_information(&diag, 2, 2, 1e-9).unwrap();
/// assert!((mi - core::f64::consts::LN_2).abs() < 1e-12);
/// ```
pub fn mutual_information(p_xy: &[f64], n_x: usize, n_y: usize, tol: f64) -> Result<f64> {
    if n_x == 0 || n_y == 0 {
        return Err(Error::Domain(
            "mutual_information: n_x and n_y must be >= 1",
        ));
    }
    if p_xy.len() != n_x * n_y {
        return Err(Error::LengthMismatch(p_xy.len(), n_x * n_y));
    }
    validate_simplex(p_xy, tol)?;

    let mut p_x = vec![0.0; n_x];
    let mut p_y = vec![0.0; n_y];
    for i in 0..n_x {
        for j in 0..n_y {
            let p = p_xy[i * n_y + j];
            p_x[i] += p;
            p_y[j] += p;
        }
    }

    let mut mi = 0.0;
    for i in 0..n_x {
        for j in 0..n_y {
            let pxy = p_xy[i * n_y + j];
            if pxy > 0.0 {
                let px = p_x[i];
                let py = p_y[j];
                if px <= 0.0 || py <= 0.0 {
                    return Err(Error::Domain(
                        "mutual_information: p(x)=0 or p(y)=0 while p(x,y)>0",
                    ));
                }
                mi += pxy * (pxy / (px * py)).ln();
            }
        }
    }
    Ok(mi)
}

/// `ndarray` adapter for discrete mutual information.
///
/// Requires `logp` feature `ndarray`.
#[cfg(feature = "ndarray")]
pub fn mutual_information_ndarray(p_xy: &ndarray::Array2<f64>, tol: f64) -> Result<f64> {
    let (n_x, n_y) = p_xy.dim();
    let flat: Vec<f64> = p_xy.iter().copied().collect();
    mutual_information(&flat, n_x, n_y, tol)
}

/// Conditional entropy in nats: the remaining uncertainty about \(X\) after observing \(Y\).
///
/// \[H(X|Y) = H(X, Y) - H(Y) = H(X) - I(X; Y)\]
///
/// # Key properties
///
/// - **Non-negative**: \(H(X|Y) \ge 0\).
/// - **Bounded**: \(H(X|Y) \le H(X)\), with equality iff \(X\) and \(Y\) are independent.
/// - **Zero iff deterministic**: \(H(X|Y) = 0\) iff \(X\) is a function of \(Y\).
///
/// # Layout
///
/// Row-major joint distribution `p_xy` with shape `(n_x, n_y)`, same as
/// [`mutual_information`].
///
/// # Examples
///
/// ```
/// # use logp::conditional_entropy;
/// // Independent: H(X|Y) = H(X).
/// let p_xy = [0.15, 0.35, 0.15, 0.35]; // 2x2
/// let h_x_given_y = conditional_entropy(&p_xy, 2, 2, 1e-9).unwrap();
/// // H(X) for marginal [0.5, 0.5] = ln(2)
/// assert!((h_x_given_y - 2.0_f64.ln()).abs() < 1e-10);
///
/// // Deterministic (Y = X): H(X|Y) = 0.
/// let diag = [0.5, 0.0, 0.0, 0.5];
/// assert!(conditional_entropy(&diag, 2, 2, 1e-9).unwrap().abs() < 1e-10);
/// ```
pub fn conditional_entropy(p_xy: &[f64], n_x: usize, n_y: usize, tol: f64) -> Result<f64> {
    let mi = mutual_information(p_xy, n_x, n_y, tol)?;
    // Compute H(X) from marginal.
    let mut p_x = vec![0.0; n_x];
    for i in 0..n_x {
        for j in 0..n_y {
            p_x[i] += p_xy[i * n_y + j];
        }
    }
    let h_x = entropy_nats(&p_x, tol)?;
    Ok(h_x - mi)
}

/// Normalized mutual information: MI scaled to \([0, 1]\) for comparing clusterings
/// of different sizes.
///
/// \[NMI(X, Y) = \frac{2\,I(X; Y)}{H(X) + H(Y)}\]
///
/// # Key properties
///
/// - **Bounded**: \(NMI \in [0, 1]\). Equals 0 for independent variables;
///   equals 1 for perfectly correlated (identical clustering).
/// - **Symmetric**: \(NMI(X, Y) = NMI(Y, X)\).
///
/// Returns 0 when both marginal entropies are zero (trivial single-cluster case).
///
/// # Examples
///
/// ```
/// # use logp::normalized_mutual_information;
/// // Perfect correlation: NMI = 1.
/// let diag = [0.5, 0.0, 0.0, 0.5];
/// let nmi = normalized_mutual_information(&diag, 2, 2, 1e-9).unwrap();
/// assert!((nmi - 1.0).abs() < 1e-10);
///
/// // Independent: NMI = 0.
/// let indep = [0.15, 0.35, 0.15, 0.35];
/// let nmi = normalized_mutual_information(&indep, 2, 2, 1e-9).unwrap();
/// assert!(nmi.abs() < 1e-10);
/// ```
pub fn normalized_mutual_information(
    p_xy: &[f64],
    n_x: usize,
    n_y: usize,
    tol: f64,
) -> Result<f64> {
    if n_x == 0 || n_y == 0 {
        return Err(Error::Domain("nmi: n_x and n_y must be >= 1"));
    }
    if p_xy.len() != n_x * n_y {
        return Err(Error::LengthMismatch(p_xy.len(), n_x * n_y));
    }
    validate_simplex(p_xy, tol)?;

    let mut p_x = vec![0.0; n_x];
    let mut p_y = vec![0.0; n_y];
    for i in 0..n_x {
        for j in 0..n_y {
            let p = p_xy[i * n_y + j];
            p_x[i] += p;
            p_y[j] += p;
        }
    }

    let h_x = entropy_nats(&p_x, tol)?;
    let h_y = entropy_nats(&p_y, tol)?;
    let denom = h_x + h_y;
    if denom <= 0.0 {
        return Ok(0.0);
    }

    let mi = mutual_information(p_xy, n_x, n_y, tol)?;
    Ok(2.0 * mi / denom)
}

/// Pointwise mutual information: the log-ratio measuring how much more (or less)
/// likely two specific outcomes co-occur than if they were independent.
///
/// \[PMI(x; y) = \ln \frac{p(x, y)}{p(x)\,p(y)}\]
///
/// # Key properties
///
/// - **Sign**: positive when \(x\) and \(y\) co-occur more than chance; negative when
///   less; zero when independent.
/// - **Unbounded**: \(PMI \in (-\infty, -\ln p(x,y)]\). In practice, rare events yield
///   very large PMI, which is why PPMI (positive PMI, clamped at 0) is common.
/// - **Connection to mutual information**: \(I(X; Y) = \mathbb{E}_{p(x,y)}[PMI(x; y)]\).
///   MI is the expected value of PMI over the joint distribution.
/// - **Connection to word2vec**: Levy & Goldberg (2014) showed that Skip-gram with
///   negative sampling implicitly factorizes a PMI matrix (shifted by \(\ln k\)).
///
/// # Examples
///
/// ```
/// # use logp::pmi;
/// // Independent events: p(x,y) = p(x)*p(y), so PMI = 0.
/// let val = pmi(0.06, 0.3, 0.2).unwrap();
/// assert!(val.abs() < 1e-10);
///
/// // Positive correlation: p(x,y) > p(x)*p(y).
/// assert!(pmi(0.4, 0.5, 0.5).unwrap() > 0.0);
///
/// // Negative correlation: p(x,y) < p(x)*p(y).
/// assert!(pmi(0.1, 0.5, 0.5).unwrap() < 0.0);
///
/// // Zero joint probability returns 0 by convention.
/// assert_eq!(pmi(0.0, 0.5, 0.5).unwrap(), 0.0);
///
/// // Impossible: p(x,y) > 0 but p(x) = 0.
/// assert!(pmi(0.1, 0.0, 0.5).is_err());
/// ```
pub fn pmi(pxy: f64, px: f64, py: f64) -> Result<f64> {
    if pxy > 0.0 && px == 0.0 {
        return Err(Error::Domain("pmi: p(x,y)>0 but p(x)=0 is impossible"));
    }
    if pxy > 0.0 && py == 0.0 {
        return Err(Error::Domain("pmi: p(x,y)>0 but p(y)=0 is impossible"));
    }
    if pxy <= 0.0 || px <= 0.0 || py <= 0.0 {
        Ok(0.0)
    } else {
        Ok((pxy / (px * py)).ln())
    }
}

/// Log-sum-exp: numerically stable computation of `ln(exp(a_1) + ... + exp(a_n))`.
///
/// This is the fundamental primitive for working in log-probability space.
/// The naive `values.iter().map(|v| v.exp()).sum::<f64>().ln()` overflows
/// for large values and underflows for small ones; the max-shift trick
/// avoids both.
///
/// Returns `NEG_INFINITY` for an empty slice.
///
/// # Examples
///
/// ```
/// # use logp::log_sum_exp;
/// // ln(e^0 + e^0) = ln(2)
/// let lse = log_sum_exp(&[0.0, 0.0]);
/// assert!((lse - 2.0_f64.ln()).abs() < 1e-12);
///
/// // Dominated term: ln(e^1000 + e^0) ≈ 1000
/// let lse = log_sum_exp(&[1000.0, 0.0]);
/// assert!((lse - 1000.0).abs() < 1e-10);
///
/// // Single element: identity.
/// assert_eq!(log_sum_exp(&[42.0]), 42.0);
///
/// // Empty: -inf.
/// assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
/// ```
#[inline]
pub fn log_sum_exp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Log-sum-exp for two values (common special case).
///
/// Equivalent to `log_sum_exp(&[a, b])` but avoids the slice overhead.
///
/// # Examples
///
/// ```
/// # use logp::log_sum_exp2;
/// let lse = log_sum_exp2(0.0, 0.0);
/// assert!((lse - 2.0_f64.ln()).abs() < 1e-12);
/// ```
#[inline]
pub fn log_sum_exp2(a: f64, b: f64) -> f64 {
    let max = a.max(b);
    if max.is_infinite() {
        return max;
    }
    max + ((a - max).exp() + (b - max).exp()).ln()
}

/// Streaming log-sum-exp: single-pass, O(1) memory computation over an iterator.
///
/// Equivalent to `log_sum_exp` but processes elements one at a time without
/// materializing a slice. Useful for large-scale or iterator-chain workloads.
///
/// Uses a running `(max, sum_exp)` pair that rescales when a new maximum arrives.
///
/// Returns `NEG_INFINITY` for an empty iterator.
///
/// # Examples
///
/// ```
/// # use logp::{log_sum_exp_iter, log_sum_exp};
/// let values = vec![1.0, 2.0, 3.0];
/// let lse_iter = log_sum_exp_iter(values.iter().copied());
/// let lse_slice = log_sum_exp(&values);
/// assert!((lse_iter - lse_slice).abs() < 1e-12);
///
/// // Works with any iterator.
/// let lse = log_sum_exp_iter((0..5).map(|i| i as f64));
/// assert!(lse.is_finite());
///
/// // Empty iterator: -inf.
/// assert_eq!(log_sum_exp_iter(std::iter::empty::<f64>()), f64::NEG_INFINITY);
/// ```
#[inline]
pub fn log_sum_exp_iter(iter: impl Iterator<Item = f64>) -> f64 {
    let mut max = f64::NEG_INFINITY;
    let mut sum_exp = 0.0;

    for v in iter {
        if v > max {
            if max.is_finite() {
                // Rescale the running sum when a new max arrives.
                sum_exp *= (max - v).exp();
            }
            max = v;
        }
        // (v - max).exp() is <= 1.0 since v <= max.
        sum_exp += (v - max).exp();
    }

    if max.is_infinite() {
        return max; // Handles NEG_INFINITY (empty) and POS_INFINITY.
    }
    max + sum_exp.ln()
}

/// Digamma function: the logarithmic derivative of the Gamma function.
///
/// \[\psi(x) = \frac{d}{dx} \ln \Gamma(x) = \frac{\Gamma'(x)}{\Gamma(x)}\]
///
/// # Key properties
///
/// - **Recurrence**: \(\psi(x+1) = \psi(x) + \frac{1}{x}\), which follows from
///   \(\Gamma(x+1) = x\,\Gamma(x)\).
/// - **Special value**: \(\psi(1) = -\gamma \approx -0.5772\), where \(\gamma\) is
///   the Euler--Mascheroni constant.
/// - **Asymptotic**: \(\psi(x) \sim \ln x - \frac{1}{2x}\) for large \(x\).
/// - **Why it appears here**: the KSG estimator for mutual information
///   ([`mutual_information_ksg`]) uses digamma to correct for the bias of
///   nearest-neighbor density estimates.
///
/// # Domain
///
/// Defined for \(x > 0\). Returns `NaN` for \(x \le 0\).
///
/// # Implementation
///
/// Uses the recurrence to shift small \(x\) up to \(x \ge 10\), then applies the
/// asymptotic expansion with 4 Bernoulli-number correction terms (~1e-14 accuracy).
///
/// # Examples
///
/// ```
/// # use logp::digamma;
/// // psi(1) = -gamma (Euler-Mascheroni constant).
/// let psi1 = digamma(1.0);
/// assert!((psi1 - (-0.5772156649)).abs() < 1e-12);
///
/// // Recurrence: psi(x+1) = psi(x) + 1/x.
/// let x = 3.5;
/// assert!((digamma(x + 1.0) - digamma(x) - 1.0 / x).abs() < 1e-12);
///
/// // Non-positive input returns NaN.
/// assert!(digamma(0.0).is_nan());
/// assert!(digamma(-1.0).is_nan());
/// ```
pub fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let mut result = 0.0;
    // Shift x upward via the recurrence psi(x+1) = psi(x) + 1/x until x >= 10,
    // where the asymptotic expansion converges to ~1e-14 with 4 Bernoulli terms.
    // (Previous threshold of 7 with 3 terms gave ~1e-10 accuracy.)
    while x < 10.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let r = 1.0 / x;
    result += x.ln() - 0.5 * r;
    let r2 = r * r;
    // Bernoulli-number correction terms: B_{2k} / (2k * x^{2k}).
    // B2/2 = 1/12, B4/4 = 1/120, B6/6 = 1/252, B8/8 = 1/240.
    result -= r2 * (1.0 / 12.0 - r2 * (1.0 / 120.0 - r2 * (1.0 / 252.0 - r2 / 240.0)));
    result
}

/// Bhattacharyya coefficient: the geometric-mean overlap between two distributions.
///
/// \[BC(p, q) = \sum_i \sqrt{p_i \, q_i}\]
///
/// # Key properties
///
/// - **Geometric mean interpretation**: each term \(\sqrt{p_i q_i}\) is the geometric
///   mean of the two probabilities at bin \(i\). BC sums these, measuring how much
///   the distributions overlap.
/// - **Bounded**: \(BC \in [0, 1]\). Equals 1 iff \(p = q\); equals 0 iff supports
///   are disjoint.
/// - **Relationship to Hellinger**: \(H^2(p, q) = 1 - BC(p, q)\), so the squared
///   Hellinger distance is just one minus the Bhattacharyya coefficient.
/// - **Relationship to Renyi**: at \(\alpha = \tfrac{1}{2}\), the Renyi divergence
///   gives \(D_{1/2}^R(p \| q) = -2 \ln BC(p, q)\).
/// - **Connection to alpha family**: \(BC = \rho_{1/2}(p, q)\), a special case of
///   [`rho_alpha`].
///
/// # Examples
///
/// ```
/// # use logp::bhattacharyya_coeff;
/// // BC(p, p) = 1.
/// let p = [0.3, 0.7];
/// assert!((bhattacharyya_coeff(&p, &p, 1e-9).unwrap() - 1.0).abs() < 1e-12);
///
/// // Disjoint supports: BC = 0.
/// let a = [1.0, 0.0];
/// let b = [0.0, 1.0];
/// assert!(bhattacharyya_coeff(&a, &b, 1e-9).unwrap().abs() < 1e-15);
///
/// // BC is in [0, 1].
/// let q = [0.5, 0.5];
/// let bc = bhattacharyya_coeff(&p, &q, 1e-9).unwrap();
/// assert!((0.0..=1.0).contains(&bc));
/// ```
pub fn bhattacharyya_coeff(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    let bc: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| pi.sqrt() * qi.sqrt())
        .sum();
    Ok(bc)
}

/// Bhattacharyya distance \(D_B(p,q) = -\ln BC(p,q)\).
///
/// # Examples
///
/// ```
/// # use logp::bhattacharyya_distance;
/// // D_B(p, p) = 0.
/// let p = [0.4, 0.6];
/// assert!(bhattacharyya_distance(&p, &p, 1e-9).unwrap().abs() < 1e-12);
///
/// // Non-negative for distinct distributions.
/// let q = [0.5, 0.5];
/// assert!(bhattacharyya_distance(&p, &q, 1e-9).unwrap() >= 0.0);
/// ```
pub fn bhattacharyya_distance(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    let bc = bhattacharyya_coeff(p, q, tol)?;
    // When supports are disjoint, bc can be 0 (=> +∞ distance). Keep it explicit.
    if bc == 0.0 {
        return Err(Error::Domain("Bhattacharyya distance is infinite (BC=0)"));
    }
    Ok(-bc.ln())
}

/// Squared Hellinger distance.
///
/// \[H^2(p, q) = \frac{1}{2}\sum_i \left(\sqrt{p_i} - \sqrt{q_i}\right)^2\]
///
/// Equivalent to \(1 - BC(p,q)\) but computed via the sum-of-squared-differences
/// form to avoid catastrophic cancellation when \(p \approx q\).
///
/// Bounded in \([0, 1]\). Equals the Amari \(\alpha\)-divergence at \(\alpha = 0\)
/// (up to a factor of 2).
///
/// # Examples
///
/// ```
/// # use logp::hellinger_squared;
/// // H^2(p, p) = 0.
/// let p = [0.25, 0.75];
/// assert!(hellinger_squared(&p, &p, 1e-9).unwrap().abs() < 1e-15);
///
/// // Disjoint supports: H^2 = 1.
/// let a = [1.0, 0.0];
/// let b = [0.0, 1.0];
/// assert!((hellinger_squared(&a, &b, 1e-9).unwrap() - 1.0).abs() < 1e-12);
/// ```
pub fn hellinger_squared(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    let h2: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| {
            let diff = pi.sqrt() - qi.sqrt();
            diff * diff
        })
        .sum();
    Ok((0.5 * h2).max(0.0))
}

/// Hellinger distance: the square root of the squared Hellinger distance.
///
/// \[H(p, q) = \sqrt{1 - BC(p, q)}\]
///
/// Unlike KL divergence, Hellinger is a **proper metric**: it is symmetric, satisfies
/// the triangle inequality, and is bounded in \([0, 1]\).
///
/// # Examples
///
/// ```
/// # use logp::hellinger;
/// // H(p, p) = 0.
/// let p = [0.3, 0.7];
/// assert!(hellinger(&p, &p, 1e-9).unwrap().abs() < 1e-15);
///
/// // Symmetric: H(p, q) = H(q, p).
/// let q = [0.5, 0.5];
/// let h_pq = hellinger(&p, &q, 1e-9).unwrap();
/// let h_qp = hellinger(&q, &p, 1e-9).unwrap();
/// assert!((h_pq - h_qp).abs() < 1e-15);
///
/// // Bounded in [0, 1].
/// assert!((0.0..=1.0).contains(&h_pq));
/// ```
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

/// Alpha-integral: the workhorse behind the entire alpha-family of divergences.
///
/// \[\rho_\alpha(p, q) = \sum_i p_i^\alpha \, q_i^{1-\alpha}\]
///
/// # Why this matters
///
/// This single quantity generates multiple divergence families via simple transforms:
///
/// - **Renyi**: \(D_\alpha^R = \frac{1}{\alpha - 1} \ln \rho_\alpha\)
/// - **Tsallis**: \(D_\alpha^T = \frac{\rho_\alpha - 1}{\alpha - 1}\)
/// - **Bhattacharyya coefficient**: \(BC = \rho_{1/2}\)
/// - **Chernoff information**: \(\min_\alpha (-\ln \rho_\alpha)\)
///
/// # Key properties
///
/// - \(\rho_\alpha(p, p) = 1\) for all \(\alpha\) (since \(\sum p_i = 1\)).
/// - By Holder's inequality, \(\rho_\alpha(p, q) \le 1\) for \(\alpha \in [0, 1]\).
/// - Continuous and log-convex in \(\alpha\).
///
/// # Examples
///
/// ```
/// # use logp::rho_alpha;
/// // rho_alpha(p, p, alpha) = 1 for any alpha (since sum(p) = 1).
/// let p = [0.2, 0.3, 0.5];
/// assert!((rho_alpha(&p, &p, 0.5, 1e-9).unwrap() - 1.0).abs() < 1e-12);
/// assert!((rho_alpha(&p, &p, 2.0, 1e-9).unwrap() - 1.0).abs() < 1e-12);
///
/// // At alpha = 0.5, rho equals the Bhattacharyya coefficient.
/// let q = [0.5, 0.25, 0.25];
/// let rho = rho_alpha(&p, &q, 0.5, 1e-9).unwrap();
/// let bc = logp::bhattacharyya_coeff(&p, &q, 1e-9).unwrap();
/// assert!((rho - bc).abs() < 1e-12);
/// ```
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

/// Renyi divergence in nats: a one-parameter family that interpolates between
/// different notions of distributional difference.
///
/// \[D_\alpha^R(p \| q) = \frac{1}{\alpha - 1} \ln \rho_\alpha(p, q), \quad \alpha > 0,\; \alpha \ne 1\]
///
/// # Key properties
///
/// - **Limit to KL**: \(\lim_{\alpha \to 1} D_\alpha^R(p \| q) = D_{KL}(p \| q)\)
///   by L'Hopital's rule (the logarithm and denominator both vanish).
/// - **Alpha = 1/2**: \(D_{1/2}^R = -2 \ln BC(p, q)\), twice the negative log
///   Bhattacharyya coefficient.
/// - **Alpha = infinity**: \(D_\infty^R = \ln \max_i (p_i / q_i)\), the log of the
///   maximum likelihood ratio. This bounds all other Renyi orders.
/// - **Monotone in alpha**: \(D_\alpha^R\) is non-decreasing in \(\alpha\).
/// - **Non-negative**: \(D_\alpha^R(p \| q) \ge 0\), with equality iff \(p = q\).
///
/// # Domain
///
/// \(\alpha > 0\), \(\alpha \ne 1\). Both `p` and `q` must be simplex distributions.
///
/// # Examples
///
/// ```
/// # use logp::renyi_divergence;
/// // D_alpha(p || p) = 0 for any valid alpha.
/// let p = [0.3, 0.7];
/// assert!(renyi_divergence(&p, &p, 2.0, 1e-9).unwrap().abs() < 1e-12);
///
/// // Non-negative.
/// let q = [0.5, 0.5];
/// assert!(renyi_divergence(&p, &q, 0.5, 1e-9).unwrap() >= -1e-12);
///
/// // alpha = 1.0 returns KL divergence (Shannon limit).
/// let kl = logp::kl_divergence(&p, &q, 1e-9).unwrap();
/// let r1 = renyi_divergence(&p, &q, 1.0, 1e-9).unwrap();
/// assert!((r1 - kl).abs() < 1e-12);
/// ```
pub fn renyi_divergence(p: &[f64], q: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    if (alpha - 1.0).abs() < 1e-12 {
        return kl_divergence(p, q, tol);
    }
    let rho = rho_alpha(p, q, alpha, tol)?;
    if rho <= 0.0 {
        return Err(Error::Domain("rho_alpha <= 0"));
    }
    Ok(rho.ln() / (alpha - 1.0))
}

/// Tsallis divergence: a non-extensive generalization of KL divergence from
/// statistical mechanics.
///
/// \[D_\alpha^T(p \| q) = \frac{\rho_\alpha(p, q) - 1}{\alpha - 1}, \quad \alpha \ne 1\]
///
/// # Key properties
///
/// - **Limit to KL**: \(\lim_{\alpha \to 1} D_\alpha^T(p \| q) = D_{KL}(p \| q)\),
///   same limit as Renyi but via a different path.
/// - **Connection to Renyi via deformed logarithm**: Tsallis uses the q-logarithm
///   \(\ln_q(x) = \frac{x^{1-q} - 1}{1-q}\) where Renyi uses the ordinary log.
///   Formally: \(D_\alpha^T = \frac{e^{(\alpha-1) D_\alpha^R} - 1}{\alpha - 1}\).
/// - **Non-extensive**: for independent systems, Tsallis divergence is **not** additive
///   (unlike KL and Renyi). This property is intentional and models systems with
///   long-range correlations in statistical physics.
/// - **Non-negative**: \(D_\alpha^T(p \| q) \ge 0\), with equality iff \(p = q\).
///
/// # Domain
///
/// \(\alpha \ne 1\). Both `p` and `q` must be simplex distributions.
///
/// # Examples
///
/// ```
/// # use logp::tsallis_divergence;
/// // D_alpha^T(p || p) = 0 for any valid alpha.
/// let p = [0.4, 0.6];
/// assert!(tsallis_divergence(&p, &p, 2.0, 1e-9).unwrap().abs() < 1e-12);
///
/// // Non-negative.
/// let q = [0.5, 0.5];
/// assert!(tsallis_divergence(&p, &q, 0.5, 1e-9).unwrap() >= -1e-12);
///
/// // alpha = 1.0 returns KL divergence (Shannon limit).
/// let kl = logp::kl_divergence(&p, &q, 1e-9).unwrap();
/// let t1 = tsallis_divergence(&p, &q, 1.0, 1e-9).unwrap();
/// assert!((t1 - kl).abs() < 1e-12);
/// ```
pub fn tsallis_divergence(p: &[f64], q: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    if (alpha - 1.0).abs() < 1e-12 {
        return kl_divergence(p, q, tol);
    }
    Ok((rho_alpha(p, q, alpha, tol)? - 1.0) / (alpha - 1.0))
}

/// Amari alpha-divergence: a one-parameter family from information geometry that
/// continuously interpolates between forward KL, reverse KL, and squared Hellinger.
///
/// For \(\alpha \notin \{-1, 1\}\):
///
/// \[D^\alpha(p : q) = \frac{4}{1 - \alpha^2}\left(1 - \rho_{\frac{1-\alpha}{2}}(p, q)\right)\]
///
/// # Key properties
///
/// - **\(\alpha = -1\)**: recovers \(D_{KL}(p \| q)\), the forward KL divergence.
/// - **\(\alpha = +1\)**: recovers \(D_{KL}(q \| p)\), the reverse KL divergence.
/// - **\(\alpha = 0\)**: gives \(4(1 - BC(p,q)) = 4\,H^2(p,q)\), proportional to
///   the squared Hellinger distance.
/// - **Duality**: \(D^\alpha(p : q) = D^{-\alpha}(q : p)\). Swapping the sign of
///   \(\alpha\) is the same as swapping the arguments.
/// - **Non-negative**: \(D^\alpha(p : q) \ge 0\), with equality iff \(p = q\).
/// - **Information geometry**: the Amari family parameterizes the \(\alpha\)-connections
///   on the statistical manifold (Amari & Nagaoka, 2000).
///
/// # Examples
///
/// ```
/// # use logp::{amari_alpha_divergence, kl_divergence, hellinger_squared};
/// let p = [0.3, 0.7];
/// let q = [0.5, 0.5];
/// let tol = 1e-9;
///
/// // alpha = -1 gives forward KL(p || q).
/// let amari_neg1 = amari_alpha_divergence(&p, &q, -1.0, tol).unwrap();
/// let kl_fwd = kl_divergence(&p, &q, tol).unwrap();
/// assert!((amari_neg1 - kl_fwd).abs() < 1e-6);
///
/// // alpha = 0 gives 4 * H^2(p, q).
/// let amari_0 = amari_alpha_divergence(&p, &q, 0.0, tol).unwrap();
/// let h2 = hellinger_squared(&p, &q, tol).unwrap();
/// assert!((amari_0 - 4.0 * h2).abs() < 1e-10);
/// ```
pub fn amari_alpha_divergence(p: &[f64], q: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    if !alpha.is_finite() {
        return Err(Error::InvalidAlpha {
            alpha,
            forbidden: f64::NAN,
        });
    }
    // Numerically stable handling near ±1.
    let eps = tol.sqrt();
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

/// Csiszar f-divergence: the most general class of divergences that respect
/// sufficient statistics (information monotonicity).
///
/// \[D_f(p \| q) = \sum_i q_i \, f\!\left(\frac{p_i}{q_i}\right)\]
///
/// where \(f\) is a convex function with \(f(1) = 0\).
///
/// # Information monotonicity theorem
///
/// The defining property of f-divergences (Csiszar, 1967): for any Markov kernel
/// (stochastic map) \(T\),
///
/// \[D_f(Tp \| Tq) \le D_f(p \| q)\]
///
/// Coarse-graining (merging bins) cannot increase the divergence. This is the
/// information-theoretic analogue of the data processing inequality.
///
/// # Common f-generators
///
/// | Divergence | \(f(t)\) |
/// |---|---|
/// | KL divergence | \(t \ln t\) |
/// | Reverse KL | \(-\ln t\) |
/// | Squared Hellinger | \((\sqrt{t} - 1)^2\) |
/// | Total variation | \(\tfrac{1}{2} |t - 1|\) |
/// | Chi-squared | \((t - 1)^2\) |
/// | Jensen-Shannon | \(t \ln t - (1+t) \ln \tfrac{1+t}{2}\) |
///
/// # Convention
///
/// This function uses Csiszar's original convention: \(D_f(p \| q) = \sum q_i f(p_i/q_i)\).
/// Some textbooks reverse the roles of \(p\) and \(q\), writing
/// \(\sum p_i f(q_i/p_i)\), which silently computes the *conjugate* divergence
/// \(D_{f^*}\) where \(f^*(t) = t\,f(1/t)\). The generators in the table above
/// follow this function's convention.
///
/// # Edge cases
///
/// When `q_i = 0`:
/// - if `p_i = 0`, the contribution is treated as 0 (by continuity).
/// - if `p_i > 0`, the divergence is infinite; we return an error.
///
/// # Examples
///
/// ```
/// # use logp::{csiszar_f_divergence, kl_divergence};
/// let p = [0.3, 0.7];
/// let q = [0.5, 0.5];
///
/// // f(t) = t*ln(t) recovers KL divergence.
/// let cs = csiszar_f_divergence(&p, &q, |t| t * t.ln(), 1e-9).unwrap();
/// let kl = kl_divergence(&p, &q, 1e-9).unwrap();
/// assert!((cs - kl).abs() < 1e-10);
///
/// // f(t) = (t - 1)^2 gives chi-squared divergence.
/// let chi2 = csiszar_f_divergence(&p, &q, |t| (t - 1.0).powi(2), 1e-9).unwrap();
/// assert!(chi2 >= 0.0);
/// ```
pub fn csiszar_f_divergence(p: &[f64], q: &[f64], f: impl Fn(f64) -> f64, tol: f64) -> Result<f64> {
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

/// Total variation distance: half the L1 norm between two distributions.
///
/// \[TV(p, q) = \frac{1}{2} \sum_i |p_i - q_i|\]
///
/// Equivalently, this is the Csiszar f-divergence with generator
/// \(f(t) = \frac{1}{2} |t - 1|\).
///
/// # Key properties
///
/// - **Metric**: symmetric, satisfies the triangle inequality, and \(TV(p, p) = 0\).
/// - **Bounded**: \(TV \in [0, 1]\).
/// - **Pinsker's inequality**: \(TV(p, q) \le \sqrt{\frac{1}{2} D_{KL}(p \| q)}\).
///
/// # Examples
///
/// ```
/// # use logp::total_variation;
/// // TV(p, p) = 0.
/// let p = [0.3, 0.7];
/// assert!(total_variation(&p, &p, 1e-9).unwrap().abs() < 1e-15);
///
/// // Disjoint supports: TV = 1.
/// let a = [1.0, 0.0];
/// let b = [0.0, 1.0];
/// assert!((total_variation(&a, &b, 1e-9).unwrap() - 1.0).abs() < 1e-12);
///
/// // Symmetric.
/// let q = [0.5, 0.5];
/// let tv_pq = total_variation(&p, &q, 1e-9).unwrap();
/// let tv_qp = total_variation(&q, &p, 1e-9).unwrap();
/// assert!((tv_pq - tv_qp).abs() < 1e-15);
/// ```
pub fn total_variation(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    let tv: f64 = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).abs())
        .sum();
    Ok(0.5 * tv)
}

/// Chi-squared divergence: a member of the Csiszar f-divergence family that is
/// particularly sensitive to tail differences.
///
/// \[\chi^2(p \| q) = \sum_i \frac{(p_i - q_i)^2}{q_i}\]
///
/// Equivalently, the f-divergence with generator \(f(t) = (t - 1)^2\).
///
/// # Key properties
///
/// - **Non-negative**: \(\chi^2 \ge 0\), with equality iff \(p = q\).
/// - **Upper bounds KL**: \(D_{KL}(p \| q) \le \ln(1 + \chi^2(p \| q))\).
/// - **Sensitivity warning**: unbounded and extremely sensitive to small \(q_i\).
///   When \(q_i\) is near zero but \(p_i\) is not, the ratio \((p_i - q_i)^2 / q_i\)
///   can be very large even when KL divergence would be moderate.
///
/// # Domain
///
/// Requires \(q_i > 0\) whenever \(p_i > 0\) (same as KL).
///
/// # Examples
///
/// ```
/// # use logp::chi_squared_divergence;
/// // chi^2(p, p) = 0.
/// let p = [0.3, 0.7];
/// assert!(chi_squared_divergence(&p, &p, 1e-9).unwrap().abs() < 1e-15);
///
/// // Non-negative.
/// let q = [0.5, 0.5];
/// assert!(chi_squared_divergence(&p, &q, 1e-9).unwrap() >= 0.0);
/// ```
pub fn chi_squared_divergence(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    ensure_same_len(p, q)?;
    validate_simplex(p, tol)?;
    validate_simplex(q, tol)?;
    let mut d = 0.0;
    for (&pi, &qi) in p.iter().zip(q.iter()) {
        if qi == 0.0 {
            if pi == 0.0 {
                continue;
            }
            return Err(Error::Domain("chi-squared undefined: q_i=0 while p_i>0"));
        }
        let diff = pi - qi;
        d += diff * diff / qi;
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

/// Bregman divergence: the gap between a convex function and its tangent approximation.
///
/// \[B_F(p, q) = F(p) - F(q) - \langle p - q,\, \nabla F(q) \rangle\]
///
/// # Key properties
///
/// - **Non-negative**: \(B_F(p, q) \ge 0\) by convexity of \(F\), with equality iff
///   \(p = q\).
/// - **Not symmetric** in general: \(B_F(p, q) \ne B_F(q, p)\).
/// - **Generalized Pythagorean theorem**: for an affine subspace \(S\) and its
///   Bregman projection \(q^* = \arg\min_{q \in S} B_F(p, q)\), the three-point
///   identity holds: \(B_F(p, q) = B_F(p, q^*) + B_F(q^*, q)\) for all \(q \in S\).
///   This is the foundation of dually flat geometry (Amari).
/// - **Not an f-divergence**: Bregman divergences are **not** information monotone
///   in general. They live in a different branch of the divergence taxonomy.
/// - **Examples**: squared Euclidean (\(F = \tfrac{1}{2}\|x\|^2\)) gives
///   \(B_F(p,q) = \tfrac{1}{2}\|p - q\|^2\); negative entropy
///   (\(F = \sum x_i \ln x_i\)) gives the KL divergence.
///
/// # Examples
///
/// ```
/// # use logp::{bregman_divergence, SquaredL2};
/// // Squared-L2 generator: B_F(p, q) = 0.5 * ||p - q||^2.
/// let gen = SquaredL2;
/// let p = [1.0, 2.0, 3.0];
/// let q = [1.5, 1.5, 2.5];
/// let b = bregman_divergence(&gen, &p, &q).unwrap();
/// let expected = 0.5 * ((0.5_f64).powi(2) + (0.5_f64).powi(2) + (0.5_f64).powi(2));
/// assert!((b - expected).abs() < 1e-12);
///
/// // B_F(p, p) = 0.
/// assert!(bregman_divergence(&gen, &p, &p).unwrap().abs() < 1e-15);
/// ```
pub fn bregman_divergence(gen: &impl BregmanGenerator, p: &[f64], q: &[f64]) -> Result<f64> {
    ensure_nonempty(p)?;
    ensure_same_len(p, q)?;
    let mut grad_q = vec![0.0; q.len()];
    gen.grad_into(q, &mut grad_q)?;
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
///
/// # Examples
///
/// ```
/// # use logp::{total_bregman_divergence, bregman_divergence, SquaredL2};
/// let gen = SquaredL2;
/// let p = [1.0, 2.0];
/// let q = [3.0, 4.0];
///
/// let tb = total_bregman_divergence(&gen, &p, &q).unwrap();
///
/// // Total Bregman <= Bregman (normalization divides by >= 1).
/// let b = bregman_divergence(&gen, &p, &q).unwrap();
/// assert!(tb <= b + 1e-12);
/// ```
pub fn total_bregman_divergence(gen: &impl BregmanGenerator, p: &[f64], q: &[f64]) -> Result<f64> {
    ensure_nonempty(p)?;
    ensure_same_len(p, q)?;
    let mut grad_q = vec![0.0; q.len()];
    gen.grad_into(q, &mut grad_q)?;
    let fp = gen.f(p)?;
    let fq = gen.f(q)?;
    let mut inner = 0.0;
    for i in 0..p.len() {
        inner += (p[i] - q[i]) * grad_q[i];
    }
    let b = fp - fq - inner;
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

/// Negative-entropy Bregman generator: \(F(x) = \sum_i x_i \ln x_i\),
/// \(\nabla F(x)_i = 1 + \ln x_i\).
///
/// The Bregman divergence with this generator is the (unnormalized) KL divergence:
/// \(B_F(p, q) = \sum_i p_i \ln(p_i / q_i) - \sum_i (p_i - q_i)\).
/// When \(p\) and \(q\) are normalized (simplex), the second sum vanishes and
/// \(B_F = D_{KL}(p \| q)\).
///
/// This connects the information-theoretic (f-divergence) and geometric (Bregman)
/// views of KL divergence. The dually-flat structure of the probability simplex
/// under this generator is the foundation of information geometry (Amari & Nagaoka, 2000).
#[derive(Debug, Clone, Copy, Default)]
pub struct NegEntropy;

impl BregmanGenerator for NegEntropy {
    fn f(&self, x: &[f64]) -> Result<f64> {
        ensure_nonempty(x)?;
        let mut s = 0.0;
        for &xi in x {
            if xi < 0.0 {
                return Err(Error::Domain("NegEntropy: input must be nonnegative"));
            }
            if xi > 0.0 {
                s += xi * xi.ln();
            }
        }
        Ok(s)
    }

    fn grad_into(&self, x: &[f64], out: &mut [f64]) -> Result<()> {
        ensure_nonempty(x)?;
        if out.len() != x.len() {
            return Err(Error::LengthMismatch(out.len(), x.len()));
        }
        for (o, &xi) in out.iter_mut().zip(x.iter()) {
            if xi <= 0.0 {
                return Err(Error::Domain("NegEntropy grad: input must be positive"));
            }
            *o = 1.0 + xi.ln();
        }
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

    fn l1(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q.iter()).map(|(&a, &b)| (a - b).abs()).sum()
    }

    #[test]
    fn test_entropy_unchecked() {
        let p = [0.5, 0.5];
        let h = entropy_unchecked(&p);
        // -0.5*ln(0.5) - 0.5*ln(0.5) = -ln(0.5) = ln(2)
        assert!((h - core::f64::consts::LN_2).abs() < 1e-12);
    }

    #[test]
    fn js_is_bounded_by_ln2() {
        let p = [1.0, 0.0];
        let q = [0.0, 1.0];
        let js = jensen_shannon_divergence(&p, &q, TOL).unwrap();
        assert!(js <= core::f64::consts::LN_2 + 1e-12);
        assert!(js >= 0.0);
    }

    #[test]
    fn mutual_information_independent_is_zero() {
        // p(x,y) = p(x)p(y) ⇒ I(X;Y)=0
        let p_x = [0.5, 0.5];
        let p_y = [0.25, 0.75];
        // Row-major 2x2:
        // [0.125, 0.375,
        //  0.125, 0.375]
        let p_xy = [
            p_x[0] * p_y[0],
            p_x[0] * p_y[1],
            p_x[1] * p_y[0],
            p_x[1] * p_y[1],
        ];
        let mi = mutual_information(&p_xy, 2, 2, TOL).unwrap();
        assert!(mi.abs() < 1e-12, "mi={}", mi);
    }

    #[test]
    fn mutual_information_perfect_correlation_is_ln2() {
        // X=Y uniform bit ⇒ I(X;Y)=ln 2 (nats)
        let p_xy = [0.5, 0.0, 0.0, 0.5]; // 2x2 diagonal
        let mi = mutual_information(&p_xy, 2, 2, TOL).unwrap();
        assert!((mi - core::f64::consts::LN_2).abs() < 1e-12, "mi={}", mi);
    }

    #[test]
    fn bregman_squared_l2_matches_half_l2() {
        let gen = SquaredL2;
        let p = [1.0, 2.0, 3.0];
        let q = [1.5, 1.5, 2.5];
        let b = bregman_divergence(&gen, &p, &q).unwrap();
        let expected = 0.5
            * p.iter()
                .zip(q.iter())
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f64>();
        assert!((b - expected).abs() < 1e-12);
    }

    // --- Entropy tests ---

    #[test]
    fn entropy_nats_uniform_is_ln_n() {
        // Uniform distribution over n items: H = ln(n)
        for n in [2, 4, 8, 16] {
            let p: Vec<f64> = vec![1.0 / n as f64; n];
            let h = entropy_nats(&p, TOL).unwrap();
            let expected = (n as f64).ln();
            assert!(
                (h - expected).abs() < 1e-12,
                "n={n}: h={h} expected={expected}"
            );
        }
    }

    #[test]
    fn entropy_nats_singleton_is_zero() {
        let h = entropy_nats(&[1.0], TOL).unwrap();
        assert!(h.abs() < 1e-15);
    }

    #[test]
    fn entropy_bits_converts_correctly() {
        let p = [0.25, 0.75];
        let nats = entropy_nats(&p, TOL).unwrap();
        let bits = entropy_bits(&p, TOL).unwrap();
        assert!((bits - nats / core::f64::consts::LN_2).abs() < 1e-12);
    }

    // --- Cross-entropy tests ---

    #[test]
    fn cross_entropy_identity_h_pq_eq_h_p_plus_kl() {
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        let h_pq = cross_entropy_nats(&p, &q, TOL).unwrap();
        let h_p = entropy_nats(&p, TOL).unwrap();
        let kl = kl_divergence(&p, &q, TOL).unwrap();
        assert!((h_pq - (h_p + kl)).abs() < 1e-12);
    }

    #[test]
    fn cross_entropy_rejects_zero_q_with_positive_p() {
        let p = [0.5, 0.5];
        let q = [1.0, 0.0]; // q[1]=0 but p[1]=0.5
        assert!(cross_entropy_nats(&p, &q, TOL).is_err());
    }

    // --- Validate / normalize tests ---

    #[test]
    fn validate_simplex_accepts_valid() {
        assert!(validate_simplex(&[0.3, 0.7], TOL).is_ok());
        assert!(validate_simplex(&[1.0], TOL).is_ok());
    }

    #[test]
    fn validate_simplex_rejects_bad_sum() {
        assert!(validate_simplex(&[0.3, 0.6], TOL).is_err()); // sum=0.9
    }

    #[test]
    fn validate_simplex_rejects_negative() {
        assert!(validate_simplex(&[1.5, -0.5], TOL).is_err());
    }

    #[test]
    fn validate_simplex_rejects_empty() {
        assert!(validate_simplex(&[], TOL).is_err());
    }

    #[test]
    fn normalize_in_place_works() {
        let mut v = vec![2.0, 3.0];
        let s = normalize_in_place(&mut v).unwrap();
        assert!((s - 5.0).abs() < 1e-12);
        assert!((v[0] - 0.4).abs() < 1e-12);
        assert!((v[1] - 0.6).abs() < 1e-12);
    }

    #[test]
    fn normalize_in_place_rejects_zero_sum() {
        let mut v = vec![0.0, 0.0];
        assert!(normalize_in_place(&mut v).is_err());
    }

    // --- Hellinger / Bhattacharyya tests ---

    #[test]
    fn hellinger_identical_is_zero() {
        let p = [0.25, 0.75];
        let h = hellinger(&p, &p, TOL).unwrap();
        assert!(h.abs() < 1e-12);
    }

    #[test]
    fn hellinger_squared_in_unit_interval() {
        let p = [0.1, 0.9];
        let q = [0.9, 0.1];
        let h2 = hellinger_squared(&p, &q, TOL).unwrap();
        assert!((-1e-12..=1.0 + 1e-12).contains(&h2), "h2={h2}");
    }

    #[test]
    fn bhattacharyya_coeff_identical_is_one() {
        let p = [0.3, 0.7];
        let bc = bhattacharyya_coeff(&p, &p, TOL).unwrap();
        assert!((bc - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bhattacharyya_distance_identical_is_zero() {
        let p = [0.5, 0.5];
        let d = bhattacharyya_distance(&p, &p, TOL).unwrap();
        assert!(d.abs() < 1e-12);
    }

    // --- Renyi / Tsallis tests ---

    #[test]
    fn renyi_alpha_half_on_simple_case() {
        let p = [0.5, 0.5];
        let q = [0.25, 0.75];
        // alpha=0.5 should be well-defined and non-negative
        let r = renyi_divergence(&p, &q, 0.5, TOL).unwrap();
        assert!(r >= -1e-12, "renyi={r}");
    }

    #[test]
    fn renyi_identical_is_zero() {
        let p = [0.3, 0.7];
        let r = renyi_divergence(&p, &p, 2.0, TOL).unwrap();
        assert!(r.abs() < 1e-12, "renyi(p,p)={r}");
    }

    #[test]
    fn tsallis_identical_is_zero() {
        let p = [0.4, 0.6];
        let t = tsallis_divergence(&p, &p, 2.0, TOL).unwrap();
        assert!(t.abs() < 1e-12, "tsallis(p,p)={t}");
    }

    // --- Digamma test ---

    #[test]
    fn digamma_at_one_is_neg_euler_mascheroni() {
        let psi1 = digamma(1.0);
        // digamma(1) = -gamma where gamma ~= 0.57721566490153286
        assert!(
            (psi1 - (-0.57721566490153286)).abs() < 1e-12,
            "psi(1)={psi1}"
        );
    }

    #[test]
    fn digamma_recurrence_relation() {
        // digamma(x+1) = digamma(x) + 1/x
        for &x in &[1.0, 2.0, 3.5, 10.0] {
            let lhs = digamma(x + 1.0);
            let rhs = digamma(x) + 1.0 / x;
            assert!(
                (lhs - rhs).abs() < 1e-12,
                "recurrence at x={x}: {lhs} vs {rhs}"
            );
        }
    }

    #[test]
    fn pmi_independent_is_zero() {
        // PMI(x,y) = log(p(x,y) / (p(x)*p(y))). If independent: p(x,y) = p(x)*p(y)
        let pmi_val = pmi(0.06, 0.3, 0.2).unwrap(); // 0.3 * 0.2 = 0.06
        assert!(
            pmi_val.abs() < 1e-10,
            "PMI of independent events should be 0: {pmi_val}"
        );
    }

    #[test]
    fn pmi_positive_for_correlated() {
        // If p(x,y) > p(x)*p(y), events are positively correlated
        let pmi_val = pmi(0.4, 0.5, 0.5).unwrap(); // 0.4 > 0.5*0.5 = 0.25
        assert!(
            pmi_val > 0.0,
            "correlated events should have positive PMI: {pmi_val}"
        );
    }

    #[test]
    fn renyi_approaches_kl_as_alpha_to_one() {
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        let tol = 1e-9;
        let kl = kl_divergence(&p, &q, tol).unwrap();
        // Renyi(alpha) -> KL as alpha -> 1
        let r099 = renyi_divergence(&p, &q, 0.99, tol).unwrap();
        let r0999 = renyi_divergence(&p, &q, 0.999, tol).unwrap();
        assert!((r099 - kl).abs() < 0.01, "Renyi(0.99)={r099}, KL={kl}");
        assert!((r0999 - kl).abs() < 0.001, "Renyi(0.999)={r0999}, KL={kl}");
    }

    #[test]
    fn amari_alpha_neg1_is_kl_forward() {
        // Amari alpha=-1 returns KL(p||q) per the implementation
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        let tol = 1e-9;
        let kl_pq = kl_divergence(&p, &q, tol).unwrap();
        let amari = amari_alpha_divergence(&p, &q, -1.0, tol).unwrap();
        assert!(
            (amari - kl_pq).abs() < 1e-6,
            "Amari(-1)={amari}, KL(p||q)={kl_pq}"
        );
    }

    #[test]
    fn amari_alpha_pos1_is_kl_reverse() {
        // Amari alpha=+1 returns KL(q||p) per the implementation
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        let tol = 1e-9;
        let kl_qp = kl_divergence(&q, &p, tol).unwrap();
        let amari = amari_alpha_divergence(&p, &q, 1.0, tol).unwrap();
        assert!(
            (amari - kl_qp).abs() < 1e-6,
            "Amari(1)={amari}, KL(q||p)={kl_qp}"
        );
    }

    #[test]
    fn csiszar_with_kl_generator_matches_kl() {
        // f(t) = t*ln(t) gives KL divergence
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        let tol = 1e-9;
        let kl = kl_divergence(&p, &q, tol).unwrap();
        let cs = csiszar_f_divergence(&p, &q, |t| t * t.ln(), tol).unwrap();
        assert!((cs - kl).abs() < 1e-6, "Csiszar(t*ln(t))={cs}, KL={kl}");
    }

    #[test]
    fn mutual_information_deterministic_equals_entropy() {
        // If Y = f(X), MI(X;Y) = H(X)
        // Joint: p(x=0,y=0)=0.3, p(x=1,y=1)=0.7
        let p_xy = [0.3, 0.0, 0.0, 0.7]; // 2x2 joint
        let mi = mutual_information(&p_xy, 2, 2, 1e-9).unwrap();
        let h_x = entropy_nats(&[0.3, 0.7], 1e-9).unwrap();
        assert!((mi - h_x).abs() < 1e-6, "MI={mi}, H(X)={h_x}");
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
            prop_assert!(js <= core::f64::consts::LN_2 + 1e-9);
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

    // Heavier “theorem-ish” checks: keep case count modest so `cargo test` stays fast.
    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

        #[test]
        fn pinsker_kl_lower_bounds_l1_squared(
            p in simplex_vec_pos(16, 1e-6),
            q in simplex_vec_pos(16, 1e-6),
        ) {
            // Pinsker: TV(p,q)^2 <= (1/2) KL(p||q)
            // where TV = (1/2)||p-q||_1. Rearranged: KL(p||q) >= 0.5 * ||p-q||_1^2.
            let kl = kl_divergence(&p, &q, 1e-6).unwrap();
            let d1 = l1(&p, &q);
            prop_assert!(kl + 1e-9 >= 0.5 * d1 * d1, "kl={kl} l1={d1}");
        }

        #[test]
        fn sqrt_js_satisfies_triangle_inequality(
            p in simplex_vec(12),
            q in simplex_vec(12),
            r in simplex_vec(12),
        ) {
            // Known fact: sqrt(JS) is a metric on the simplex.
            let js_pq = jensen_shannon_divergence(&p, &q, 1e-6).unwrap().max(0.0).sqrt();
            let js_qr = jensen_shannon_divergence(&q, &r, 1e-6).unwrap().max(0.0).sqrt();
            let js_pr = jensen_shannon_divergence(&p, &r, 1e-6).unwrap().max(0.0).sqrt();
            prop_assert!(js_pr <= js_pq + js_qr + 1e-7, "js_pr={js_pr} js_pq+js_qr={}", js_pq+js_qr);
        }

        #[test]
        fn mutual_information_equals_kl_to_product(
            // Ensure strictly positive so KL domains are satisfied.
            p_xy in simplex_vec_pos(16, 1e-6),
            nx in 2usize..=4,
            ny in 2usize..=4,
        ) {
            // We need p_xy to have length nx*ny; we will truncate/renormalize a fixed-length draw.
            let n = nx * ny;
            let mut joint = p_xy;
            joint.truncate(n);
            // Renormalize after truncation.
            let _ = normalize_in_place(&mut joint).unwrap();

            // Compute MI via the dedicated function.
            let mi = mutual_information(&joint, nx, ny, 1e-6).unwrap();

            // Compute product of marginals and KL(joint || product).
            let mut p_x = vec![0.0; nx];
            let mut p_y = vec![0.0; ny];
            for i in 0..nx {
                for j in 0..ny {
                    let p = joint[i * ny + j];
                    p_x[i] += p;
                    p_y[j] += p;
                }
            }
            let mut prod = vec![0.0; n];
            for i in 0..nx {
                for j in 0..ny {
                    prod[i * ny + j] = p_x[i] * p_y[j];
                }
            }
            let kl = kl_divergence(&joint, &prod, 1e-6).unwrap();

            prop_assert!((mi - kl).abs() < 1e-9, "mi={mi} kl={kl}");
        }

        #[test]
        fn hellinger_satisfies_triangle_inequality(
            p in simplex_vec(8),
            q in simplex_vec(8),
            r in simplex_vec(8),
        ) {
            let h_pq = hellinger(&p, &q, 1e-6).unwrap();
            let h_qr = hellinger(&q, &r, 1e-6).unwrap();
            let h_pr = hellinger(&p, &r, 1e-6).unwrap();
            prop_assert!(h_pr <= h_pq + h_qr + 1e-7, "h_pr={h_pr} h_pq+h_qr={}", h_pq + h_qr);
        }
    }

    // --- total_bregman_divergence ---

    #[test]
    fn total_bregman_le_bregman() {
        // tB_F(p, q) <= B_F(p, q) because the denominator sqrt(1 + ||grad||^2) >= 1.
        let gen = SquaredL2;
        let p = [1.0, 2.0, 3.0];
        let q = [4.0, 5.0, 6.0];
        let b = bregman_divergence(&gen, &p, &q).unwrap();
        let tb = total_bregman_divergence(&gen, &p, &q).unwrap();
        assert!(tb <= b + 1e-12, "total_bregman={tb} > bregman={b}");
        assert!(tb >= 0.0);
    }

    #[test]
    fn total_bregman_is_zero_for_identical() {
        let gen = SquaredL2;
        let p = [1.0, 2.0];
        let tb = total_bregman_divergence(&gen, &p, &p).unwrap();
        assert!(tb.abs() < 1e-15);
    }

    // --- rho_alpha ---

    #[test]
    fn rho_alpha_self_is_one() {
        let p = [0.1, 0.2, 0.3, 0.4];
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0, 2.0, -1.0] {
            let r = rho_alpha(&p, &p, alpha, TOL).unwrap();
            assert!((r - 1.0).abs() < 1e-10, "rho_alpha(p,p,{alpha})={r}");
        }
    }

    // --- digamma negative domain ---

    #[test]
    fn digamma_nonpositive_is_nan() {
        assert!(digamma(0.0).is_nan());
        assert!(digamma(-1.0).is_nan());
        assert!(digamma(-100.0).is_nan());
    }

    // --- pmi edge cases ---

    #[test]
    fn pmi_zero_joint_returns_zero() {
        assert_eq!(pmi(0.0, 0.5, 0.5).unwrap(), 0.0);
    }

    #[test]
    fn pmi_zero_marginal_with_zero_joint_returns_zero() {
        // When px or py is zero AND pxy is also zero, return 0 by convention.
        assert_eq!(pmi(0.0, 0.0, 0.5).unwrap(), 0.0);
        assert_eq!(pmi(0.0, 0.5, 0.0).unwrap(), 0.0);
    }

    #[test]
    fn pmi_all_zero_returns_zero() {
        assert_eq!(pmi(0.0, 0.0, 0.0).unwrap(), 0.0);
    }

    // --- Enrich-motivated tests ---

    #[test]
    fn digamma_at_dlmf_reference_values() {
        // psi(0.5) = -gamma - 2*ln(2)
        let gamma = 0.57721566490153286;
        let expected_half = -gamma - 2.0 * core::f64::consts::LN_2;
        let psi_half = digamma(0.5);
        assert!(
            (psi_half - expected_half).abs() < 1e-12,
            "psi(0.5)={psi_half} expected={expected_half}"
        );

        // psi(2) = 1 - gamma
        let expected_2 = 1.0 - gamma;
        let psi_2 = digamma(2.0);
        assert!(
            (psi_2 - expected_2).abs() < 1e-12,
            "psi(2)={psi_2} expected={expected_2}"
        );

        // psi(3) = 1 + 1/2 - gamma
        let expected_3 = 1.5 - gamma;
        let psi_3 = digamma(3.0);
        assert!(
            (psi_3 - expected_3).abs() < 1e-12,
            "psi(3)={psi_3} expected={expected_3}"
        );

        // psi(4) = 1 + 1/2 + 1/3 - gamma
        let expected_4 = 1.0 + 0.5 + 1.0 / 3.0 - gamma;
        let psi_4 = digamma(4.0);
        assert!(
            (psi_4 - expected_4).abs() < 1e-12,
            "psi(4)={psi_4} expected={expected_4}"
        );
    }

    #[test]
    fn tsallis_approaches_kl_as_alpha_to_one() {
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        let tol = 1e-9;
        let kl = kl_divergence(&p, &q, tol).unwrap();
        // Tsallis(alpha) -> KL as alpha -> 1 (from both sides)
        let t099 = tsallis_divergence(&p, &q, 0.99, tol).unwrap();
        let t0999 = tsallis_divergence(&p, &q, 0.999, tol).unwrap();
        let t101 = tsallis_divergence(&p, &q, 1.01, tol).unwrap();
        let t1001 = tsallis_divergence(&p, &q, 1.001, tol).unwrap();
        assert!((t099 - kl).abs() < 0.01, "Tsallis(0.99)={t099}, KL={kl}");
        assert!(
            (t0999 - kl).abs() < 0.001,
            "Tsallis(0.999)={t0999}, KL={kl}"
        );
        assert!((t101 - kl).abs() < 0.01, "Tsallis(1.01)={t101}, KL={kl}");
        assert!(
            (t1001 - kl).abs() < 0.001,
            "Tsallis(1.001)={t1001}, KL={kl}"
        );
    }

    #[test]
    fn renyi_approaches_kl_from_above() {
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        let tol = 1e-9;
        let kl = kl_divergence(&p, &q, tol).unwrap();
        let r101 = renyi_divergence(&p, &q, 1.01, tol).unwrap();
        let r1001 = renyi_divergence(&p, &q, 1.001, tol).unwrap();
        assert!((r101 - kl).abs() < 0.01, "Renyi(1.01)={r101}, KL={kl}");
        assert!((r1001 - kl).abs() < 0.001, "Renyi(1.001)={r1001}, KL={kl}");
    }

    #[test]
    fn renyi_at_half_equals_neg2_ln_bc() {
        // D_{1/2}^R(p || q) = -2 * ln(BC(p, q))
        let p = [0.2, 0.3, 0.5];
        let q = [0.4, 0.4, 0.2];
        let tol = 1e-9;
        let renyi_half = renyi_divergence(&p, &q, 0.5, tol).unwrap();
        let bc = bhattacharyya_coeff(&p, &q, tol).unwrap();
        let expected = -2.0 * bc.ln();
        assert!(
            (renyi_half - expected).abs() < 1e-10,
            "Renyi(0.5)={renyi_half}, -2*ln(BC)={expected}"
        );
    }

    #[test]
    fn hellinger_squared_equals_one_minus_bc() {
        let p = [0.1, 0.4, 0.5];
        let q = [0.3, 0.3, 0.4];
        let tol = 1e-9;
        let h2 = hellinger_squared(&p, &q, tol).unwrap();
        let bc = bhattacharyya_coeff(&p, &q, tol).unwrap();
        assert!(
            (h2 - (1.0 - bc)).abs() < 1e-12,
            "H^2={h2}, 1-BC={}",
            1.0 - bc
        );
    }

    #[test]
    fn csiszar_hellinger_generator_matches_twice_hellinger_squared() {
        // f(t) = (sqrt(t) - 1)^2 gives sum(q_i*(sqrt(p_i/q_i)-1)^2)
        // = sum((sqrt(p_i) - sqrt(q_i))^2) = 2*(1 - BC) = 2*H^2.
        let p = [0.2, 0.3, 0.5];
        let q = [0.4, 0.4, 0.2];
        let tol = 1e-9;
        let h2 = hellinger_squared(&p, &q, tol).unwrap();
        let cs = csiszar_f_divergence(&p, &q, |t| (t.sqrt() - 1.0).powi(2), tol).unwrap();
        assert!(
            (cs - 2.0 * h2).abs() < 1e-10,
            "Csiszar(Hellinger)={cs}, 2*H^2={}",
            2.0 * h2
        );
    }

    #[test]
    fn csiszar_chi_squared_generator_is_nonneg() {
        // f(t) = (t - 1)^2 gives chi-squared divergence.
        let p = [0.2, 0.3, 0.5];
        let q = [0.4, 0.4, 0.2];
        let tol = 1e-9;
        let chi2 = csiszar_f_divergence(&p, &q, |t| (t - 1.0).powi(2), tol).unwrap();
        assert!(chi2 >= 0.0, "chi2={chi2}");
        // Chi-squared(p,p) = 0
        let chi2_self = csiszar_f_divergence(&p, &p, |t| (t - 1.0).powi(2), tol).unwrap();
        assert!(chi2_self.abs() < 1e-12, "chi2(p,p)={chi2_self}");
    }

    #[test]
    fn near_boundary_inputs_no_nan() {
        // Distributions with entries near machine epsilon.
        let tiny = 1e-300;
        let p = [tiny, 1.0 - tiny];
        let q = [tiny * 2.0, 1.0 - tiny * 2.0];
        let tol = 1e-6;

        let kl = kl_divergence(&p, &q, tol).unwrap();
        assert!(kl.is_finite(), "kl={kl}");
        // Allow tiny negative from floating-point at extreme values.
        assert!(kl >= -1e-12, "kl negative: {kl}");

        let js = jensen_shannon_divergence(&p, &q, tol).unwrap();
        assert!(js.is_finite(), "js={js}");

        let h = hellinger(&p, &q, tol).unwrap();
        assert!(h.is_finite(), "hellinger={h}");

        let bc = bhattacharyya_coeff(&p, &q, tol).unwrap();
        assert!(bc.is_finite(), "bc={bc}");

        let ent = entropy_nats(&p, tol).unwrap();
        assert!(ent.is_finite(), "entropy={ent}");
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

        #[test]
        fn entropy_is_concave(
            p in simplex_vec(8),
            q in simplex_vec(8),
            lambda in 0.0f64..=1.0,
        ) {
            // H(lambda*p + (1-lambda)*q) >= lambda*H(p) + (1-lambda)*H(q)
            let mix: Vec<f64> = p.iter().zip(q.iter())
                .map(|(&pi, &qi)| lambda * pi + (1.0 - lambda) * qi)
                .collect();
            let h_mix = entropy_nats(&mix, 1e-6).unwrap();
            let h_p = entropy_nats(&p, 1e-6).unwrap();
            let h_q = entropy_nats(&q, 1e-6).unwrap();
            let rhs = lambda * h_p + (1.0 - lambda) * h_q;
            prop_assert!(h_mix + 1e-10 >= rhs, "h_mix={h_mix} rhs={rhs}");
        }

        #[test]
        fn renyi_monotone_in_alpha(
            p in simplex_vec_pos(8, 1e-6),
            q in simplex_vec_pos(8, 1e-6),
        ) {
            // Renyi divergence is non-decreasing in alpha for fixed p, q.
            let alphas = [0.1, 0.25, 0.5, 0.75, 0.99];
            let vals: Vec<f64> = alphas.iter()
                .map(|&a| renyi_divergence(&p, &q, a, 1e-6).unwrap())
                .collect();
            for i in 1..vals.len() {
                prop_assert!(
                    vals[i] + 1e-9 >= vals[i - 1],
                    "Renyi not monotone: D({})={} < D({})={}",
                    alphas[i], vals[i], alphas[i - 1], vals[i - 1]
                );
            }
        }

        #[test]
        fn cross_entropy_decomposition(
            p in simplex_vec_pos(8, 1e-6),
            q in simplex_vec_pos(8, 1e-6),
        ) {
            // H(p, q) = H(p) + KL(p || q)
            let h_pq = cross_entropy_nats(&p, &q, 1e-6).unwrap();
            let h_p = entropy_nats(&p, 1e-6).unwrap();
            let kl = kl_divergence(&p, &q, 1e-6).unwrap();
            prop_assert!(
                (h_pq - (h_p + kl)).abs() < 1e-9,
                "H(p,q)={h_pq} != H(p)+KL={}", h_p + kl
            );
        }

        #[test]
        fn bhattacharyya_renyi_consistency(
            p in simplex_vec_pos(8, 1e-6),
            q in simplex_vec_pos(8, 1e-6),
        ) {
            // D_{1/2}^R(p || q) = -2 * ln(BC(p, q))
            let renyi_half = renyi_divergence(&p, &q, 0.5, 1e-6).unwrap();
            let bc = bhattacharyya_coeff(&p, &q, 1e-6).unwrap();
            let expected = -2.0 * bc.ln();
            prop_assert!(
                (renyi_half - expected).abs() < 1e-8,
                "Renyi(0.5)={renyi_half}, -2*ln(BC)={expected}"
            );
        }

        #[test]
        fn csiszar_hellinger_consistency(
            p in simplex_vec_pos(8, 1e-6),
            q in simplex_vec_pos(8, 1e-6),
        ) {
            // Csiszar with f(t)=(sqrt(t)-1)^2 = 2 * squared Hellinger
            let h2 = hellinger_squared(&p, &q, 1e-6).unwrap();
            let cs = csiszar_f_divergence(&p, &q, |t| (t.sqrt() - 1.0).powi(2), 1e-6).unwrap();
            prop_assert!(
                (cs - 2.0 * h2).abs() < 1e-8,
                "Csiszar(Hellinger)={cs}, 2*H^2={}", 2.0 * h2
            );
        }

        #[test]
        fn pinsker_tightness_for_nearby_distributions(
            p in simplex_vec_pos(8, 1e-6),
        ) {
            // For a distribution slightly perturbed from p, check Pinsker is non-vacuous.
            // q = 0.99*p + 0.01*uniform (epsilon-perturbation).
            let n = p.len();
            let q: Vec<f64> = p.iter().map(|&pi| 0.99 * pi + 0.01 / n as f64).collect();
            let kl = kl_divergence(&p, &q, 1e-6).unwrap();
            let d1: f64 = p.iter().zip(q.iter()).map(|(&a, &b)| (a - b).abs()).sum();
            let pinsker_rhs = 0.5 * d1 * d1;
            // Pinsker: KL >= 0.5 * L1^2
            prop_assert!(kl + 1e-12 >= pinsker_rhs, "kl={kl} pinsker_rhs={pinsker_rhs}");
            // Non-vacuity: for nearby distributions, KL / (0.5*L1^2) should be O(1),
            // not astronomically large (would indicate the bound is useful).
            if pinsker_rhs > 1e-20 {
                let ratio = kl / pinsker_rhs;
                prop_assert!(ratio < 1000.0, "Pinsker ratio too large: {ratio}");
            }
        }

        #[test]
        fn total_variation_satisfies_triangle(
            p in simplex_vec(8),
            q in simplex_vec(8),
            r in simplex_vec(8),
        ) {
            let tv_pq = total_variation(&p, &q, 1e-6).unwrap();
            let tv_qr = total_variation(&q, &r, 1e-6).unwrap();
            let tv_pr = total_variation(&p, &r, 1e-6).unwrap();
            prop_assert!(tv_pr <= tv_pq + tv_qr + 1e-10);
        }

        #[test]
        fn chi_squared_matches_csiszar(
            p in simplex_vec_pos(8, 1e-6),
            q in simplex_vec_pos(8, 1e-6),
        ) {
            let chi2 = chi_squared_divergence(&p, &q, 1e-6).unwrap();
            let cs = csiszar_f_divergence(&p, &q, |t| (t - 1.0).powi(2), 1e-6).unwrap();
            prop_assert!(
                (chi2 - cs).abs() < 1e-8,
                "chi2={chi2}, csiszar={cs}"
            );
        }

        #[test]
        fn renyi_entropy_monotone_in_alpha(
            p in simplex_vec_pos(8, 1e-6),
        ) {
            // H_alpha is non-increasing in alpha.
            let alphas = [0.1, 0.25, 0.5, 0.75, 0.99];
            let vals: Vec<f64> = alphas.iter()
                .map(|&a| renyi_entropy(&p, a, 1e-6).unwrap())
                .collect();
            for i in 1..vals.len() {
                prop_assert!(
                    vals[i] <= vals[i - 1] + 1e-9,
                    "H_alpha not monotone: H({})={} > H({})={}",
                    alphas[i], vals[i], alphas[i - 1], vals[i - 1]
                );
            }
        }
    }

    // --- New function unit tests ---

    #[test]
    fn weighted_js_at_extreme_weights() {
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        // pi1 = 0 means m = q, KL(q || q) = 0
        let js0 = jensen_shannon_weighted(&p, &q, 0.0, TOL).unwrap();
        assert!(js0.abs() < 1e-12, "JS(pi=0)={js0}");
    }

    #[test]
    fn conditional_entropy_chain_rule() {
        // H(X|Y) = H(X,Y) - H(Y)
        let p_xy = [0.2, 0.1, 0.3, 0.4]; // 2x2
        let h_xy = entropy_nats(&p_xy, TOL).unwrap();
        let p_y = [p_xy[0] + p_xy[2], p_xy[1] + p_xy[3]];
        let h_y = entropy_nats(&p_y, TOL).unwrap();
        let h_x_given_y = conditional_entropy(&p_xy, 2, 2, TOL).unwrap();
        assert!(
            (h_x_given_y - (h_xy - h_y)).abs() < 1e-10,
            "H(X|Y)={h_x_given_y}, H(X,Y)-H(Y)={}",
            h_xy - h_y
        );
    }

    #[test]
    fn conditional_entropy_nonnegative() {
        let p_xy = [0.1, 0.2, 0.3, 0.4];
        let h = conditional_entropy(&p_xy, 2, 2, TOL).unwrap();
        assert!(h >= -1e-12, "H(X|Y) negative: {h}");
    }

    #[test]
    fn nmi_bounds() {
        // NMI for a general joint: should be in [0, 1].
        let p_xy = [0.1, 0.2, 0.3, 0.4];
        let nmi = normalized_mutual_information(&p_xy, 2, 2, TOL).unwrap();
        assert!((-1e-12..=1.0 + 1e-12).contains(&nmi), "nmi={nmi}");
    }

    #[test]
    fn total_variation_self_is_zero() {
        let p = [0.3, 0.7];
        assert!(total_variation(&p, &p, TOL).unwrap().abs() < 1e-15);
    }

    #[test]
    fn total_variation_disjoint_is_one() {
        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((total_variation(&a, &b, TOL).unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn chi_squared_self_is_zero() {
        let p = [0.3, 0.7];
        assert!(chi_squared_divergence(&p, &p, TOL).unwrap().abs() < 1e-15);
    }

    #[test]
    fn chi_squared_upper_bounds_kl() {
        // KL(p||q) <= ln(1 + chi2(p||q))
        let p = [0.2, 0.3, 0.5];
        let q = [0.4, 0.4, 0.2];
        let kl = kl_divergence(&p, &q, TOL).unwrap();
        let chi2 = chi_squared_divergence(&p, &q, TOL).unwrap();
        assert!(
            kl <= (1.0 + chi2).ln() + 1e-10,
            "kl={kl} > ln(1+chi2)={}",
            (1.0 + chi2).ln()
        );
    }

    #[test]
    fn renyi_entropy_uniform_is_ln_n() {
        let p = [0.25, 0.25, 0.25, 0.25];
        for alpha in [0.5, 2.0, 3.0, 10.0] {
            let h = renyi_entropy(&p, alpha, TOL).unwrap();
            let expected = 4.0_f64.ln();
            assert!(
                (h - expected).abs() < 1e-12,
                "H_{alpha}(uniform) = {h}, expected {expected}"
            );
        }
    }

    #[test]
    fn tsallis_entropy_delta_is_zero() {
        let delta = [1.0, 0.0, 0.0];
        for alpha in [0.5, 2.0, 3.0] {
            let s = tsallis_entropy(&delta, alpha, TOL).unwrap();
            assert!(s.abs() < 1e-12, "Tsallis({alpha}) of delta = {s}");
        }
    }

    #[test]
    fn renyi_entropy_collision() {
        // H_2(p) = -ln(sum(p_i^2))
        let p = [0.3, 0.7];
        let h2 = renyi_entropy(&p, 2.0, TOL).unwrap();
        let expected = -(0.3_f64.powi(2) + 0.7_f64.powi(2)).ln();
        assert!(
            (h2 - expected).abs() < 1e-12,
            "H_2={h2} expected={expected}"
        );
    }

    #[test]
    fn neg_entropy_bregman_matches_kl_on_simplex() {
        // For normalized p, q: B_NegEntropy(p, q) = KL(p || q).
        let p = [0.2, 0.3, 0.5];
        let q = [0.4, 0.4, 0.2];
        let kl = kl_divergence(&p, &q, TOL).unwrap();
        let gen = NegEntropy;
        let breg = bregman_divergence(&gen, &p, &q).unwrap();
        assert!(
            (breg - kl).abs() < 1e-10,
            "Bregman(NegEntropy)={breg}, KL={kl}"
        );
    }

    #[test]
    fn neg_entropy_bregman_self_is_zero() {
        let p = [0.3, 0.7];
        let gen = NegEntropy;
        let breg = bregman_divergence(&gen, &p, &p).unwrap();
        assert!(breg.abs() < 1e-14, "Bregman(p,p)={breg}");
    }

    // --- Streaming log-sum-exp tests ---

    #[test]
    fn log_sum_exp_iter_matches_slice() {
        let values = [1.0, 2.0, 3.0, -1.0, 0.5];
        let lse_slice = log_sum_exp(&values);
        let lse_iter = log_sum_exp_iter(values.iter().copied());
        assert!(
            (lse_slice - lse_iter).abs() < 1e-12,
            "slice={lse_slice} iter={lse_iter}"
        );
    }

    #[test]
    fn log_sum_exp_iter_empty() {
        assert_eq!(log_sum_exp_iter(std::iter::empty()), f64::NEG_INFINITY);
    }

    #[test]
    fn log_sum_exp_iter_single() {
        assert_eq!(log_sum_exp_iter(std::iter::once(42.0)), 42.0);
    }

    #[test]
    fn log_sum_exp_iter_large_values() {
        // Same stability test as log_sum_exp: dominated term.
        let lse = log_sum_exp_iter([1000.0, 0.0].iter().copied());
        assert!((lse - 1000.0).abs() < 1e-10);
    }

    // --- Data processing inequality for discrete MI ---

    #[test]
    fn data_processing_inequality_mi() {
        // Apply a deterministic coarse-graining (Markov kernel) to Y.
        // MI(X; f(Y)) <= MI(X; Y).
        let p_xy = [0.3, 0.1, 0.05, 0.05, 0.1, 0.2, 0.05, 0.15]; // 2x4
        let n_x = 2;
        let n_y = 4;
        let mi_full = mutual_information(&p_xy, n_x, n_y, TOL).unwrap();

        // Coarse-grain Y: merge bins 0+1 and 2+3 -> 2x2.
        let mut p_coarse = [0.0; 4]; // 2x2
        for i in 0..n_x {
            p_coarse[i * 2] = p_xy[i * n_y] + p_xy[i * n_y + 1];
            p_coarse[i * 2 + 1] = p_xy[i * n_y + 2] + p_xy[i * n_y + 3];
        }
        let mi_coarse = mutual_information(&p_coarse, n_x, 2, TOL).unwrap();

        assert!(
            mi_coarse <= mi_full + 1e-10,
            "DPI violated: MI(coarse)={mi_coarse} > MI(full)={mi_full}"
        );
    }

    // --- Weighted JS additional tests ---

    #[test]
    fn weighted_js_bounded_by_entropy_of_weights() {
        // JS_pi(p, q) <= H(pi) = -pi1*ln(pi1) - pi2*ln(pi2)
        let p = [0.1, 0.9];
        let q = [0.9, 0.1];
        let pi1 = 0.3;
        let jsw = jensen_shannon_weighted(&p, &q, pi1, TOL).unwrap();
        let pi2 = 1.0 - pi1;
        let h_pi = -(pi1 * pi1.ln() + pi2 * pi2.ln());
        assert!(jsw <= h_pi + 1e-10, "JS_pi={jsw} > H(pi)={h_pi}");
    }

    // --- Renyi/Tsallis entropy approach Shannon ---

    #[test]
    fn renyi_entropy_approaches_shannon_as_alpha_to_one() {
        let p = [0.2, 0.3, 0.5];
        let h_shannon = entropy_nats(&p, TOL).unwrap();
        let h_099 = renyi_entropy(&p, 0.99, TOL).unwrap();
        let h_0999 = renyi_entropy(&p, 0.999, TOL).unwrap();
        let h_101 = renyi_entropy(&p, 1.01, TOL).unwrap();
        assert!((h_099 - h_shannon).abs() < 0.01);
        assert!((h_0999 - h_shannon).abs() < 0.001);
        assert!((h_101 - h_shannon).abs() < 0.01);
    }

    #[test]
    fn tsallis_entropy_approaches_shannon_as_alpha_to_one() {
        let p = [0.2, 0.3, 0.5];
        let h_shannon = entropy_nats(&p, TOL).unwrap();
        let s_099 = tsallis_entropy(&p, 0.99, TOL).unwrap();
        let s_0999 = tsallis_entropy(&p, 0.999, TOL).unwrap();
        assert!((s_099 - h_shannon).abs() < 0.01);
        assert!((s_0999 - h_shannon).abs() < 0.001);
    }

    // --- New tests for 0.2.0 ---

    #[test]
    fn bhattacharyya_precision_near_identical() {
        let p = [0.5 + 1e-15, 0.5 - 1e-15];
        let q = [0.5, 0.5];
        let bc = bhattacharyya_coeff(&p, &q, 1e-6).unwrap();
        assert!(
            (bc - 1.0).abs() < 1e-14,
            "BC should be very close to 1.0: {bc}"
        );
        let h2 = hellinger_squared(&p, &q, 1e-6).unwrap();
        // h2 should be tiny but not exactly zero (numerics permitting).
        assert!(h2 < 1e-14, "h2 should be tiny: {h2}");
        assert!(h2.is_finite(), "h2 should be finite");
    }

    #[test]
    fn renyi_alpha_sweep_continuity() {
        let p = [0.2, 0.3, 0.5];
        let tol = 1e-9;
        let mut prev_renyi = renyi_entropy(&p, 0.5, tol).unwrap();
        let mut prev_tsallis = tsallis_entropy(&p, 0.5, tol).unwrap();
        let mut alpha = 0.6;
        while alpha <= 2.0 + 1e-9 {
            let r = renyi_entropy(&p, alpha, tol).unwrap();
            let t = tsallis_entropy(&p, alpha, tol).unwrap();
            let jump_r = (r - prev_renyi).abs();
            let jump_t = (t - prev_tsallis).abs();
            assert!(
                jump_r < 0.5,
                "Renyi discontinuity at alpha={alpha}: jump={jump_r}"
            );
            assert!(
                jump_t < 0.5,
                "Tsallis discontinuity at alpha={alpha}: jump={jump_t}"
            );
            prev_renyi = r;
            prev_tsallis = t;
            alpha += 0.1;
        }
    }

    #[test]
    fn ksg_ties_finite() {
        // Integer-valued data with exact ties.
        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![(i % 5) as f64]).collect();
        let y: Vec<Vec<f64>> = (0..50).map(|i| vec![(i % 3) as f64]).collect();
        let mi1 = mutual_information_ksg(&x, &y, 3, KsgVariant::Alg1).unwrap();
        let mi2 = mutual_information_ksg(&x, &y, 3, KsgVariant::Alg2).unwrap();
        assert!(
            mi1.is_finite(),
            "KSG Alg1 with ties returned NaN/Inf: {mi1}"
        );
        assert!(
            mi2.is_finite(),
            "KSG Alg2 with ties returned NaN/Inf: {mi2}"
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 64, .. ProptestConfig::default() })]

        #[test]
        fn bregman_nonnegative(
            p in simplex_vec_pos(8, 1e-6),
            q in simplex_vec_pos(8, 1e-6),
        ) {
            let gen = NegEntropy;
            let b = bregman_divergence(&gen, &p, &q).unwrap();
            prop_assert!(b >= -1e-12, "Bregman(NegEntropy) negative: {b}");
        }

        #[test]
        fn renyi_divergence_alpha1_equals_kl(
            p in simplex_vec_pos(8, 1e-6),
            q in simplex_vec_pos(8, 1e-6),
        ) {
            let tol = 1e-6;
            let kl = kl_divergence(&p, &q, tol).unwrap();
            let r1 = renyi_divergence(&p, &q, 1.0, tol).unwrap();
            prop_assert!(
                (r1 - kl).abs() < 1e-9,
                "renyi(alpha=1)={r1} != kl={kl}"
            );
        }
    }

    #[test]
    fn pmi_impossible_input_errors() {
        // p(x,y)>0 but p(x)=0 is impossible.
        assert!(pmi(0.1, 0.0, 0.5).is_err());
        // p(x,y)>0 but p(y)=0 is impossible.
        assert!(pmi(0.1, 0.5, 0.0).is_err());
    }
}
