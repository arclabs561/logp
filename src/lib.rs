//! # logp
//!
//! Information theory primitives: entropies and divergences.
//!
//! ## Scope
//!
//! This crate is **L1 (Logic)** in the mathematical foundation: it should stay small and reusable.
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
//! | **Jensen-Shannon** | \(f\)-div + metric | Symmetric, bounded \(\[0, \ln 2\]\), \(\sqrt{JS}\) is a metric |
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

mod ksg;
pub use ksg::{mutual_information_ksg, KsgVariant};

/// Natural log of 2. Useful when converting nats ↔ bits or bounding Jensen–Shannon.
pub const LN_2: f64 = core::f64::consts::LN_2;

/// KL Divergence between two diagonal Multivariate Gaussians.
///
/// Used for Variational Information Bottleneck (VIB) to regularize latent spaces.
///
/// Returns 0.5 * Σ [ (std1/std2)^2 + (mu2-mu1)^2 / std2^2 - 1 + 2*ln(std2/std1) ]
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
///
/// # Domain
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
/// It does not force `ndarray` into the public surface of an L1 crate.
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
pub fn pmi(pxy: f64, px: f64, py: f64) -> f64 {
    if pxy <= 0.0 || px <= 0.0 || py <= 0.0 {
        0.0
    } else {
        (pxy / (px * py)).ln()
    }
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
/// Uses the recurrence to shift small \(x\) up to \(x \ge 7\), then applies the
/// asymptotic expansion with Bernoulli-number correction terms.
pub fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    let mut result = 0.0;
    while x < 7.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    let r = 1.0 / x;
    result += x.ln() - 0.5 * r;
    let r2 = r * r;
    result -= r2 * (1.0 / 12.0 - r2 * (1.0 / 120.0 - r2 / 252.0));
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

/// Squared Hellinger distance: one minus the Bhattacharyya coefficient.
///
/// \[H^2(p, q) = 1 - \sum_i \sqrt{p_i \, q_i} = 1 - BC(p, q)\]
///
/// Bounded in \([0, 1]\). Equals the Amari \(\alpha\)-divergence at \(\alpha = 0\)
/// (up to a factor of 2).
pub fn hellinger_squared(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
    let bc = bhattacharyya_coeff(p, q, tol)?;
    Ok((1.0 - bc).max(0.0))
}

/// Hellinger distance: the square root of the squared Hellinger distance.
///
/// \[H(p, q) = \sqrt{1 - BC(p, q)}\]
///
/// Unlike KL divergence, Hellinger is a **proper metric**: it is symmetric, satisfies
/// the triangle inequality, and is bounded in \([0, 1]\).
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
pub fn renyi_divergence(p: &[f64], q: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    if alpha == 1.0 {
        return Err(Error::InvalidAlpha {
            alpha,
            forbidden: 1.0,
        });
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
pub fn tsallis_divergence(p: &[f64], q: &[f64], alpha: f64, tol: f64) -> Result<f64> {
    if alpha == 1.0 {
        return Err(Error::InvalidAlpha {
            alpha,
            forbidden: 1.0,
        });
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
/// # Edge cases
///
/// When `q_i = 0`:
/// - if `p_i = 0`, the contribution is treated as 0 (by continuity).
/// - if `p_i > 0`, the divergence is infinite; we return an error.
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

    fn l1(p: &[f64], q: &[f64]) -> f64 {
        p.iter().zip(q.iter()).map(|(&a, &b)| (a - b).abs()).sum()
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
        assert!((mi - LN_2).abs() < 1e-12, "mi={}", mi);
    }

    #[test]
    fn bregman_squared_l2_matches_half_l2() {
        let gen = SquaredL2;
        let p = [1.0, 2.0, 3.0];
        let q = [1.5, 1.5, 2.5];
        let mut grad = [0.0; 3];
        let b = bregman_divergence(&gen, &p, &q, &mut grad).unwrap();
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
            assert!((h - expected).abs() < 1e-12, "n={n}: h={h} expected={expected}");
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
        assert!((bits - nats / LN_2).abs() < 1e-12);
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
        assert!(h2 >= -1e-12 && h2 <= 1.0 + 1e-12, "h2={h2}");
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
        // digamma(1) = -gamma where gamma ~= 0.5772156649
        assert!((psi1 - (-0.5772156649)).abs() < 1e-8, "psi(1)={psi1}");
    }

    #[test]
    fn digamma_recurrence_relation() {
        // digamma(x+1) = digamma(x) + 1/x
        for &x in &[1.0, 2.0, 3.5, 10.0] {
            let lhs = digamma(x + 1.0);
            let rhs = digamma(x) + 1.0 / x;
            assert!((lhs - rhs).abs() < 1e-8, "recurrence at x={x}: {lhs} vs {rhs}");
        }
    }

    #[test]
    fn pmi_independent_is_zero() {
        // PMI(x,y) = log(p(x,y) / (p(x)*p(y))). If independent: p(x,y) = p(x)*p(y)
        let pmi_val = pmi(0.06, 0.3, 0.2); // 0.3 * 0.2 = 0.06
        assert!(pmi_val.abs() < 1e-10, "PMI of independent events should be 0: {pmi_val}");
    }

    #[test]
    fn pmi_positive_for_correlated() {
        // If p(x,y) > p(x)*p(y), events are positively correlated
        let pmi_val = pmi(0.4, 0.5, 0.5); // 0.4 > 0.5*0.5 = 0.25
        assert!(pmi_val > 0.0, "correlated events should have positive PMI: {pmi_val}");
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
        assert!((amari - kl_pq).abs() < 1e-6, "Amari(-1)={amari}, KL(p||q)={kl_pq}");
    }

    #[test]
    fn amari_alpha_pos1_is_kl_reverse() {
        // Amari alpha=+1 returns KL(q||p) per the implementation
        let p = [0.3, 0.7];
        let q = [0.5, 0.5];
        let tol = 1e-9;
        let kl_qp = kl_divergence(&q, &p, tol).unwrap();
        let amari = amari_alpha_divergence(&p, &q, 1.0, tol).unwrap();
        assert!((amari - kl_qp).abs() < 1e-6, "Amari(1)={amari}, KL(q||p)={kl_qp}");
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
    }
}
