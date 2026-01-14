//! # surp
//!
//! Information theory primitives: surprisal, entropy, KL divergence.
//!
//! (surp: from "surprisal", the fundamental unit of information) If an event
//! is certain, learning it happened conveys no information. If it's rare,
//! learning it happened is highly informative.
//!
//! ## Key Functions
//!
//! | Function | Measures | Formula |
//! |----------|----------|---------|
//! | [`entropy`] | Uncertainty in distribution | H(p) = -Σ p(x) log p(x) |
//! | [`cross_entropy`] | Expected bits using wrong code | H(p,q) = -Σ p(x) log q(x) |
//! | [`kl_divergence`] | "Distance" from q to p | KL(p‖q) = Σ p(x) log(p(x)/q(x)) |
//! | [`js_divergence`] | Symmetric "distance" | JS(p,q) = ½KL(p‖m) + ½KL(q‖m) |
//! | [`mutual_information`] | Shared information | I(X;Y) = H(X) + H(Y) - H(X,Y) |
//!
//! ## Quick Start
//!
//! ```rust
//! use surp::{entropy, kl_divergence, js_divergence};
//!
//! let p = [0.25, 0.25, 0.25, 0.25];  // uniform
//! let q = [0.5, 0.25, 0.125, 0.125]; // skewed
//!
//! let h = entropy(&p);           // 2.0 bits (maximum for 4 outcomes)
//! let kl = kl_divergence(&p, &q); // asymmetric divergence
//! let js = js_divergence(&p, &q); // symmetric, bounded [0, 1]
//! ```
//!
//! ## Why These Matter for ML
//!
//! - **Cross-entropy loss**: Standard classification loss is cross-entropy
//!   between true labels (one-hot) and predicted probabilities
//! - **KL divergence**: VAE latent space regularization, knowledge distillation
//! - **Mutual information**: Feature selection, representation learning bounds
//! - **JS divergence**: GAN training (original formulation)
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
//!
//! ## Sublinear Estimation
//!
//! The [`unseen`] module implements the Valiant-Valiant estimators
//! for entropy and support size from sublinear samples.
//!
//! **Key insight**: Given O(k/log k) samples from a distribution over k elements,
//! we can accurately estimate entropy, support size, and distances. This is
//! optimal up to constant factors.
//!
//! **References**:
//! - Valiant & Valiant (2011). "Estimating the Unseen: An n/log(n)-Sample Estimator"
//! - Valiant & Valiant (2017). "Estimating the Unseen: Improved Estimators" (JACM)
//!
//! ## What Can Go Wrong
//!
//! 1. **Zero probabilities in KL**: KL(p||q) is infinite if q(x)=0 where p(x)>0.
//!    Use JS divergence or add smoothing.
//! 2. **Not normalized**: Many functions assume probabilities sum to 1. Check inputs.
//! 3. **Entropy of continuous distributions**: Discrete entropy ≠ differential entropy.
//!    Discretization bin width affects results.
//! 4. **Sample size too small**: Entropy estimators are biased for small samples.
//!    Miller-Madow correction or Valiant-Valiant for sublinear regime.
//! 5. **Floating point issues**: Very small probabilities can cause log(0) = -inf.

use thiserror::Error;

pub mod entropy_calibration;
pub mod fdiv;
pub mod unseen;
pub mod zipf;

// Re-export commonly used f-divergences at crate root
pub use fdiv::{
    alpha_divergence, bhattacharyya_coefficient, bhattacharyya_distance, chi_squared_divergence,
    f_divergence, hellinger_distance, hellinger_squared, hellinger_squared_tensorized,
    renyi_divergence, total_variation,
};

/// Error types for information theory operations.
#[derive(Debug, Error)]
pub enum Error {
    #[error("distributions have different lengths: {0} vs {1}")]
    LengthMismatch(usize, usize),

    #[error("distribution does not sum to 1.0 (sum = {0})")]
    NotNormalized(f64),

    #[error("negative probability: {0}")]
    NegativeProbability(f64),

    #[error("KL divergence undefined: q[{index}] = 0 but p[{index}] = {p_val} > 0")]
    KLUndefined { index: usize, p_val: f64 },
}

pub type Result<T> = std::result::Result<T, Error>;

const LN_2: f64 = std::f64::consts::LN_2;
const EPSILON: f64 = 1e-12;

// =============================================================================
// Fingerprint: the "histogram of histograms" from sample data
// =============================================================================

/// Compute the fingerprint of a sample: F[i] = count of elements appearing exactly i times.
///
/// The fingerprint captures the "shape" of a distribution without the labels.
/// It's the key data structure for sublinear entropy estimation.
///
/// # Arguments
///
/// * `samples` - Slice of sample values (any Eq + Hash + Clone type)
///
/// # Returns
///
/// Vector where index i contains count of elements appearing exactly i+1 times.
/// (Index 0 = elements appearing once, index 1 = appearing twice, etc.)
///
/// # Example
///
/// ```rust
/// use surp::fingerprint;
///
/// let samples = ["a", "a", "b", "c", "c", "c"];
/// let fp = fingerprint(&samples);
///
/// // "b" appears 1 time, "a" appears 2 times, "c" appears 3 times
/// assert_eq!(fp[0], 1);  // F_1: one element appears once
/// assert_eq!(fp[1], 1);  // F_2: one element appears twice  
/// assert_eq!(fp[2], 1);  // F_3: one element appears three times
/// ```
pub fn fingerprint<T: std::hash::Hash + Eq + Clone>(samples: &[T]) -> Vec<usize> {
    use std::collections::HashMap;

    // Count occurrences
    let mut counts: HashMap<T, usize> = HashMap::new();
    for s in samples {
        *counts.entry(s.clone()).or_insert(0) += 1;
    }

    // Build fingerprint
    let max_count = counts.values().copied().max().unwrap_or(0);
    let mut fp = vec![0usize; max_count];

    for &count in counts.values() {
        if count > 0 {
            fp[count - 1] += 1;
        }
    }

    fp
}

/// Count distinct elements in a sample.
///
/// This is the naive estimator - actual support size may be larger.
pub fn distinct_count<T: std::hash::Hash + Eq>(samples: &[T]) -> usize {
    use std::collections::HashSet;
    samples.iter().collect::<HashSet<_>>().len()
}

/// Good-Turing estimate of unseen probability mass.
///
/// Estimates the total probability mass of elements not seen in the sample.
/// Uses the simple formula: P(unseen) ≈ F₁ / n where F₁ is the count of
/// elements appearing exactly once (singletons) and n is sample size.
///
/// # Arguments
///
/// * `fingerprint` - The fingerprint of the sample
/// * `sample_size` - Total number of samples
///
/// # Returns
///
/// Estimated probability mass of unseen elements
///
/// # Example
///
/// ```rust
/// use surp::{fingerprint, good_turing_unseen_mass};
///
/// let samples: Vec<u32> = vec![1, 1, 2, 3, 4, 5, 5, 5];
/// let fp = fingerprint(&samples);
/// let unseen = good_turing_unseen_mass(&fp, samples.len());
///
/// // F_1 = 2 (elements 2,3,4 appear once - wait, that's 3)
/// // Actually: 2 appears 1x, 3 appears 1x, 4 appears 1x = F_1 = 3
/// // unseen ≈ 3/8 = 0.375
/// assert!(unseen > 0.0);
/// ```
pub fn good_turing_unseen_mass(fingerprint: &[usize], sample_size: usize) -> f64 {
    if sample_size == 0 || fingerprint.is_empty() {
        return 0.0;
    }
    fingerprint[0] as f64 / sample_size as f64
}

/// Shannon entropy: H(p) = -Σ p(x) log₂ p(x)
///
/// Measures the average "surprise" or uncertainty in a distribution.
/// Maximum entropy = log₂(n) for uniform distribution over n outcomes.
///
/// # Arguments
///
/// * `p` - Probability distribution (must sum to ~1.0)
///
/// # Returns
///
/// Entropy in bits (base-2 logarithm)
///
/// # Example
///
/// ```rust
/// use surp::entropy;
///
/// let uniform = [0.25, 0.25, 0.25, 0.25];
/// let certain = [1.0, 0.0, 0.0, 0.0];
///
/// assert!((entropy(&uniform) - 2.0).abs() < 1e-10);  // max entropy
/// assert!(entropy(&certain) < 1e-10);                 // zero entropy
/// ```
pub fn entropy(p: &[f64]) -> f64 {
    p.iter()
        .filter(|&&x| x > EPSILON)
        .map(|&x| -x * x.ln() / LN_2)
        .sum()
}

/// Cross-entropy: H(p, q) = -Σ p(x) log₂ q(x)
///
/// Expected number of bits needed to encode samples from p using
/// a code optimized for q. Always ≥ H(p), with equality iff p = q.
///
/// # Arguments
///
/// * `p` - True distribution
/// * `q` - Model distribution
///
/// # Returns
///
/// Cross-entropy in bits
///
/// # Example
///
/// ```rust
/// use surp::{entropy, cross_entropy};
///
/// let p = [0.5, 0.5];
/// let q = [0.9, 0.1];
///
/// let h_p = entropy(&p);
/// let h_pq = cross_entropy(&p, &q);
/// assert!(h_pq >= h_p);  // cross-entropy ≥ entropy
/// ```
pub fn cross_entropy(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    p.iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > EPSILON)
        .map(|(&pi, &qi)| -pi * (qi.max(EPSILON)).ln() / LN_2)
        .sum()
}

/// KL divergence: KL(p ‖ q) = Σ p(x) log₂(p(x) / q(x))
///
/// Measures how much information is lost when q is used to approximate p.
/// **Not symmetric**: KL(p‖q) ≠ KL(q‖p).
///
/// # Properties
///
/// - KL(p‖q) ≥ 0 (Gibbs' inequality)
/// - KL(p‖q) = 0 iff p = q
/// - Undefined if q(x) = 0 where p(x) > 0
///
/// # Arguments
///
/// * `p` - True distribution
/// * `q` - Approximate distribution
///
/// # Returns
///
/// KL divergence in bits
///
/// # Example
///
/// ```rust
/// use surp::kl_divergence;
///
/// let p = [0.5, 0.5];
/// let q = [0.9, 0.1];
///
/// let kl_pq = kl_divergence(&p, &q);
/// let kl_qp = kl_divergence(&q, &p);
///
/// assert!(kl_pq >= 0.0);
/// assert!((kl_pq - kl_qp).abs() > 0.1);  // asymmetric!
/// ```
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    p.iter()
        .zip(q.iter())
        .filter(|(&pi, _)| pi > EPSILON)
        .map(|(&pi, &qi)| pi * (pi / qi.max(EPSILON)).ln() / LN_2)
        .sum()
}

/// Jensen-Shannon divergence: JS(p, q) = ½KL(p‖m) + ½KL(q‖m)
///
/// Symmetric, bounded version of KL divergence where m = (p + q) / 2.
///
/// # Properties
///
/// - 0 ≤ JS(p,q) ≤ 1 (in bits)
/// - JS(p,q) = JS(q,p) (symmetric)
/// - √JS is a proper metric
///
/// # Arguments
///
/// * `p` - First distribution
/// * `q` - Second distribution
///
/// # Returns
///
/// JS divergence in bits, bounded [0, 1]
///
/// # Example
///
/// ```rust
/// use surp::js_divergence;
///
/// let p = [0.5, 0.5];
/// let q = [0.9, 0.1];
///
/// let js = js_divergence(&p, &q);
/// assert!(js >= 0.0 && js <= 1.0);
/// assert!((js_divergence(&p, &q) - js_divergence(&q, &p)).abs() < 1e-10);
/// ```
pub fn js_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "distributions must have same length");

    let m: Vec<f64> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi + qi) / 2.0)
        .collect();

    0.5 * kl_divergence(p, &m) + 0.5 * kl_divergence(q, &m)
}

/// Mutual information from joint distribution: I(X;Y) = H(X) + H(Y) - H(X,Y)
///
/// Measures how much knowing X reduces uncertainty about Y (and vice versa).
///
/// # Arguments
///
/// * `joint` - Joint probability matrix P(X,Y), flattened row-major
/// * `rows` - Number of X outcomes
/// * `cols` - Number of Y outcomes
///
/// # Returns
///
/// Mutual information in bits
///
/// # Example
///
/// ```rust
/// use surp::mutual_information;
///
/// // Independent: P(X,Y) = P(X)P(Y)
/// let independent = [0.25, 0.25, 0.25, 0.25];  // uniform 2x2
///
/// // Perfectly dependent: X = Y
/// let dependent = [0.5, 0.0, 0.0, 0.5];
///
/// let mi_indep = mutual_information(&independent, 2, 2);
/// let mi_dep = mutual_information(&dependent, 2, 2);
///
/// assert!(mi_indep < 0.01);  // ~0 for independent
/// assert!(mi_dep > 0.9);     // ~1 bit for perfect dependence
/// ```
pub fn mutual_information(joint: &[f64], rows: usize, cols: usize) -> f64 {
    assert_eq!(
        joint.len(),
        rows * cols,
        "joint size must match rows * cols"
    );

    // Marginals
    let mut p_x = vec![0.0; rows];
    let mut p_y = vec![0.0; cols];

    for i in 0..rows {
        for j in 0..cols {
            let p_xy = joint[i * cols + j];
            p_x[i] += p_xy;
            p_y[j] += p_xy;
        }
    }

    // I(X;Y) = Σ P(x,y) log(P(x,y) / (P(x)P(y)))
    let mut mi = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            let p_xy = joint[i * cols + j];
            if p_xy > EPSILON && p_x[i] > EPSILON && p_y[j] > EPSILON {
                mi += p_xy * (p_xy / (p_x[i] * p_y[j])).ln() / LN_2;
            }
        }
    }

    mi.max(0.0) // Numerical errors can make it slightly negative
}

/// Conditional entropy: H(Y|X) = H(X,Y) - H(X)
///
/// Average uncertainty remaining in Y after observing X.
///
/// # Arguments
///
/// * `joint` - Joint probability matrix P(X,Y), flattened row-major
/// * `rows` - Number of X outcomes
/// * `cols` - Number of Y outcomes
///
/// # Returns
///
/// Conditional entropy in bits
pub fn conditional_entropy(joint: &[f64], rows: usize, cols: usize) -> f64 {
    let h_xy = entropy(joint);

    // Marginal H(X)
    let mut p_x = vec![0.0; rows];
    for i in 0..rows {
        for j in 0..cols {
            p_x[i] += joint[i * cols + j];
        }
    }
    let h_x = entropy(&p_x);

    h_xy - h_x
}

/// Normalized mutual information: NMI(X,Y) = 2*I(X;Y) / (H(X) + H(Y))
///
/// Scaled to [0, 1] for easier interpretation and comparison.
///
/// # Arguments
///
/// * `joint` - Joint probability matrix P(X,Y), flattened row-major
/// * `rows` - Number of X outcomes
/// * `cols` - Number of Y outcomes
///
/// # Returns
///
/// NMI in [0, 1]
pub fn normalized_mutual_information(joint: &[f64], rows: usize, cols: usize) -> f64 {
    // Marginals
    let mut p_x = vec![0.0; rows];
    let mut p_y = vec![0.0; cols];

    for i in 0..rows {
        for j in 0..cols {
            let p_xy = joint[i * cols + j];
            p_x[i] += p_xy;
            p_y[j] += p_xy;
        }
    }

    let h_x = entropy(&p_x);
    let h_y = entropy(&p_y);

    if h_x + h_y < EPSILON {
        return 1.0; // Both constant → perfectly "correlated"
    }

    let mi = mutual_information(joint, rows, cols);
    2.0 * mi / (h_x + h_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entropy_uniform() {
        let uniform = [0.25, 0.25, 0.25, 0.25];
        let h = entropy(&uniform);
        assert!((h - 2.0).abs() < 1e-10, "uniform entropy should be 2 bits");
    }

    #[test]
    fn test_entropy_certain() {
        let certain = [1.0, 0.0, 0.0, 0.0];
        let h = entropy(&certain);
        assert!(h < 1e-10, "certain distribution should have 0 entropy");
    }

    #[test]
    fn test_kl_non_negative() {
        let p = [0.3, 0.4, 0.3];
        let q = [0.5, 0.25, 0.25];
        let kl = kl_divergence(&p, &q);
        assert!(kl >= 0.0, "KL divergence must be non-negative");
    }

    #[test]
    fn test_kl_zero_for_identical() {
        let p = [0.3, 0.4, 0.3];
        let kl = kl_divergence(&p, &p);
        assert!(kl < 1e-10, "KL(p||p) should be 0");
    }

    #[test]
    fn test_js_symmetric() {
        let p = [0.3, 0.4, 0.3];
        let q = [0.5, 0.25, 0.25];
        let js_pq = js_divergence(&p, &q);
        let js_qp = js_divergence(&q, &p);
        assert!(
            (js_pq - js_qp).abs() < 1e-10,
            "JS divergence should be symmetric"
        );
    }

    #[test]
    fn test_js_bounded() {
        let p = [1.0, 0.0];
        let q = [0.0, 1.0];
        let js = js_divergence(&p, &q);
        assert!(js >= 0.0 && js <= 1.0 + 1e-10, "JS should be in [0, 1]");
    }

    #[test]
    fn test_mutual_information_independent() {
        // P(X,Y) = P(X)P(Y) for uniform
        let independent = [0.25, 0.25, 0.25, 0.25];
        let mi = mutual_information(&independent, 2, 2);
        assert!(mi < 0.01, "independent variables should have ~0 MI");
    }

    #[test]
    fn test_mutual_information_dependent() {
        // X = Y (diagonal)
        let dependent = [0.5, 0.0, 0.0, 0.5];
        let mi = mutual_information(&dependent, 2, 2);
        assert!(
            (mi - 1.0).abs() < 0.01,
            "perfectly dependent should have 1 bit MI"
        );
    }

    #[test]
    fn test_cross_entropy_geq_entropy() {
        let p = [0.3, 0.4, 0.3];
        let q = [0.5, 0.25, 0.25];
        let h_p = entropy(&p);
        let h_pq = cross_entropy(&p, &q);
        assert!(h_pq >= h_p - 1e-10, "cross-entropy should be >= entropy");
    }

    #[test]
    fn test_nmi_bounds() {
        let joint = [0.25, 0.25, 0.25, 0.25];
        let nmi = normalized_mutual_information(&joint, 2, 2);
        assert!(nmi >= 0.0 && nmi <= 1.0 + 1e-10, "NMI should be in [0, 1]");
    }
}
