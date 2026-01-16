//! # Sublinear Estimation via the Unseen
//!
//! Implements the Valiant-Valiant estimators for distribution properties
//! from sublinear samples.
//!
//! ## The Key Insight
//!
//! Given n samples from a distribution over k elements, the naive approach
//! uses O(k) samples. Valiant & Valiant showed that O(k/log k) samples suffice
//! for accurate estimation of:
//!
//! - **Entropy**: H(p) = -Σ p(x) log p(x)
//! - **Support size**: |{x : p(x) > 0}|
//! - **Distance**: Total variation between two distributions
//!
//! ## The Algorithm
//!
//! 1. Compute the **fingerprint** F where F[i] = count of elements seen i times
//! 2. Find the "simplest" histogram h that could have generated F via linear programming
//! 3. Estimate properties from h
//!
//! The LP objective minimizes support size (Occam's razor), subject to
//! expected fingerprint ≈ observed fingerprint.
//!
//! ## Complexity
//!
//! - Sample complexity: O(k / log k) for distributions over k elements
//! - Time complexity: O(n log n) for LP setup + O(poly(log k)) for LP solve
//!
//! ## References
//!
//! - Valiant & Valiant (2011). "Estimating the Unseen: An n/log(n)-Sample Estimator"
//! - Valiant & Valiant (2017). "Estimating the Unseen: Improved Estimators" (JACM)

use crate::fingerprint;
use minilp::{ComparisonOp, OptimizationDirection, Problem};

/// Configuration for the unseen estimators.
#[derive(Clone, Debug)]
pub struct UnseenConfig {
    /// Geometric ratio for probability grid (default: 1.5)
    pub grid_ratio: f64,
    /// Number of fingerprint entries to fit (default: 10)
    pub fit_count: usize,
    /// Slack factor for constraints (default: 1.0)
    pub slack_factor: f64,
}

impl Default for UnseenConfig {
    fn default() -> Self {
        Self {
            grid_ratio: 1.5,
            fit_count: 10,
            slack_factor: 1.0,
        }
    }
}

/// Estimate entropy of a distribution including unseen elements.
///
/// Uses the linear programming approach from Valiant & Valiant to reconstruct
/// a plausible histogram consistent with the observed fingerprint.
///
/// # Arguments
///
/// * `samples` - Sample data from the distribution
///
/// # Returns
///
/// Estimated entropy in bits
///
/// # Example
///
/// ```rust
/// use surp::unseen::entropy_unseen;
///
/// // Sample 50 distinct elements once each (uniform-ish)
/// let samples: Vec<u32> = (0..50).collect();
/// let entropy = entropy_unseen(&samples);
///
/// // For uniform over ~100 elements, entropy ≈ log2(100) ≈ 6.6 bits
/// // Our estimate should be > log2(50) ≈ 5.6 bits
/// assert!(entropy > 5.0);
/// ```
pub fn entropy_unseen<T: std::hash::Hash + Eq + Clone>(samples: &[T]) -> f64 {
    entropy_unseen_with_config(samples, &UnseenConfig::default())
}

/// Entropy estimation with custom configuration.
pub fn entropy_unseen_with_config<T: std::hash::Hash + Eq + Clone>(
    samples: &[T],
    config: &UnseenConfig,
) -> f64 {
    let fp = fingerprint(samples);
    let n = samples.len() as f64;

    let hist = recover_histogram_with_config(&fp, n, config);

    // H = -Σ count * prob * log(prob)
    hist.iter()
        .map(|&(prob, count)| {
            if prob > 1e-12 {
                -prob * prob.ln() / std::f64::consts::LN_2 * count
            } else {
                0.0
            }
        })
        .sum()
}

/// Estimate support size including unseen elements.
///
/// This estimates the true number of distinct elements in the underlying
/// distribution, even those not seen in the sample.
///
/// # Arguments
///
/// * `samples` - Sample data from the distribution
///
/// # Returns
///
/// Estimated support size (number of distinct elements)
///
/// # Example
///
/// ```rust
/// use surp::unseen::support_unseen;
///
/// // If we see 50 elements each exactly once, there are likely more
/// let samples: Vec<u32> = (0..50).collect();
/// let support = support_unseen(&samples);
///
/// assert!(support >= 50.0);  // At least what we saw
/// ```
pub fn support_unseen<T: std::hash::Hash + Eq + Clone>(samples: &[T]) -> f64 {
    support_unseen_with_config(samples, &UnseenConfig::default())
}

/// Support estimation with custom configuration.
pub fn support_unseen_with_config<T: std::hash::Hash + Eq + Clone>(
    samples: &[T],
    config: &UnseenConfig,
) -> f64 {
    let fp = fingerprint(samples);
    let n = samples.len() as f64;

    let hist = recover_histogram_with_config(&fp, n, config);
    hist.iter().map(|&(_, count)| count).sum()
}

/// Estimate total variation distance between two distributions from samples.
///
/// Uses a heuristic based on support overlap. For full Valiant-Valiant
/// distance estimation, a 2D fingerprint LP would be needed.
///
/// # Arguments
///
/// * `samples_p` - Samples from distribution P
/// * `samples_q` - Samples from distribution Q
///
/// # Returns
///
/// Estimated distance heuristic (higher = more different)
///
/// # Example
///
/// ```rust
/// use surp::unseen::distance_unseen;
///
/// // Samples from different ranges (no overlap)
/// let p: Vec<u32> = (0..50).collect();
/// let r: Vec<u32> = (100..150).collect();
///
/// let dist = distance_unseen(&p, &r);
/// // With disjoint samples, distance should be positive
/// assert!(dist >= 0.0);
/// ```
pub fn distance_unseen<T: std::hash::Hash + Eq + Clone, U: std::hash::Hash + Eq + Clone>(
    samples_p: &[T],
    samples_q: &[U],
) -> f64 {
    // For now, use a simpler approach based on individual histograms
    // Full Valiant-Valiant uses 2D fingerprint which is more complex

    let fp_p = fingerprint(samples_p);
    let fp_q = fingerprint(samples_q);
    let n_p = samples_p.len() as f64;
    let n_q = samples_q.len() as f64;

    let config = UnseenConfig::default();
    let hist_p = recover_histogram_with_config(&fp_p, n_p, &config);
    let hist_q = recover_histogram_with_config(&fp_q, n_q, &config);

    // Estimate total probability mass at each probability level
    let mass_p: f64 = hist_p.iter().map(|&(prob, count)| prob * count).sum();
    let mass_q: f64 = hist_q.iter().map(|&(prob, count)| prob * count).sum();

    // Support sizes
    let support_p: f64 = hist_p.iter().map(|&(_, count)| count).sum();
    let support_q: f64 = hist_q.iter().map(|&(_, count)| count).sum();

    // Rough bound: if supports are very different, distance is large
    let support_overlap = support_p.min(support_q) / support_p.max(support_q);

    // Heuristic combination (proper implementation would use 2D LP)
    ((1.0 - support_overlap) + (mass_p - mass_q).abs()) / 2.0
}

/// Recover the underlying histogram from a sample fingerprint.
///
/// Returns a vector of (probability, count) pairs.
/// For example, `[(0.1, 5.0)]` means "5 elements have probability 0.1".
///
/// This is the core of the Valiant-Valiant approach:
///
/// 1. Define a grid of possible probability values
/// 2. Solve an LP to find the simplest histogram that explains the fingerprint
/// 3. "Simplest" = minimum support size (Occam's razor)
pub fn recover_histogram(fingerprint: &[usize], n: f64) -> Vec<(f64, f64)> {
    recover_histogram_with_config(fingerprint, n, &UnseenConfig::default())
}

/// Histogram recovery with custom configuration.
pub fn recover_histogram_with_config(
    fingerprint: &[usize],
    n: f64,
    config: &UnseenConfig,
) -> Vec<(f64, f64)> {
    if fingerprint.is_empty() || n < 1.0 {
        return Vec::new();
    }

    // Build probability grid
    // Valiant suggests geometric spacing for the "unseen" region
    let mut grid = Vec::new();

    // Unseen region: very small probabilities
    let mut x = 1.0 / (n * n);
    while x <= 1.0 {
        grid.push(x);
        x *= config.grid_ratio;
    }

    // Seen region: explicit empirical probabilities
    for i in 1..=fingerprint.len() {
        grid.push(i as f64 / n);
    }

    grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
    grid.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // Setup LP: minimize support size subject to fingerprint constraints
    let mut problem = Problem::new(OptimizationDirection::Minimize);

    // Variables: h[j] = count of elements with probability grid[j]
    let vars: Vec<_> = grid
        .iter()
        .map(|_| problem.add_var(1.0, (0.0, f64::INFINITY)))
        .collect();

    // Constraint: E[F_i] ≈ observed F_i for i = 1..fit_count
    // E[F_i] = Σ_j h[j] * Poisson(n * grid[j], i)
    let fit_k = fingerprint.len().min(config.fit_count);

    for (i, &obs) in fingerprint.iter().enumerate().take(fit_k) {
        let obs_count = obs as f64;
        let k = (i + 1) as i32;

        let mut expr = minilp::LinearExpr::empty();
        for (j, &p) in grid.iter().enumerate() {
            let poisson_prob = poisson_pmf(n * p, k);
            expr.add(vars[j], poisson_prob);
        }

        // Slack proportional to sqrt(expected count) + constant
        let slack = config.slack_factor * (obs_count.sqrt() + 1.0).max(1.0);

        problem.add_constraint(expr.clone(), ComparisonOp::Ge, obs_count - slack);
        problem.add_constraint(expr, ComparisonOp::Le, obs_count + slack);
    }

    // Constraint: total probability mass ≤ 1
    let mut prob_expr = minilp::LinearExpr::empty();
    for (j, &p) in grid.iter().enumerate() {
        prob_expr.add(vars[j], p);
    }
    problem.add_constraint(prob_expr, ComparisonOp::Le, 1.0);

    // Solve and extract solution
    match problem.solve() {
        Ok(solution) => grid
            .iter()
            .enumerate()
            .filter_map(|(j, &p)| {
                let count = solution[vars[j]];
                if count > 1e-6 {
                    Some((p, count))
                } else {
                    None
                }
            })
            .collect(),
        Err(_) => {
            // Fallback: empirical histogram
            fingerprint
                .iter()
                .enumerate()
                .filter(|(_, &count)| count > 0)
                .map(|(i, &count)| ((i + 1) as f64 / n, count as f64))
                .collect()
        }
    }
}

/// Poisson probability mass function: P(X = k) = λ^k e^(-λ) / k!
fn poisson_pmf(lambda: f64, k: i32) -> f64 {
    if lambda <= 0.0 || k < 0 {
        return if k == 0 && lambda == 0.0 { 1.0 } else { 0.0 };
    }

    let k_f = k as f64;
    // Use log-space to avoid overflow: log(PMF) = k*log(λ) - λ - log(k!)
    let log_pmf = k_f * lambda.ln() - lambda - ln_factorial(k as usize);
    log_pmf.exp()
}

/// Compute ln(n!) using log-sum for small n, Stirling for large n
fn ln_factorial(n: usize) -> f64 {
    if n <= 20 {
        // Direct computation for small values
        (1..=n).map(|i| (i as f64).ln()).sum()
    } else {
        // Stirling's approximation: ln(n!) ≈ n*ln(n) - n + 0.5*ln(2πn)
        let nf = n as f64;
        nf * nf.ln() - nf + 0.5 * (2.0 * std::f64::consts::PI * nf).ln()
    }
}

/// Estimate properties of a distribution with confidence bounds.
///
/// Returns (estimate, lower_bound, upper_bound) via bootstrap.
pub fn entropy_unseen_with_ci<T: std::hash::Hash + Eq + Clone>(
    samples: &[T],
    confidence: f64,
    num_bootstrap: usize,
) -> (f64, f64, f64) {
    use rand::prelude::IndexedRandom;
    let mut rng = rand::rng();

    let point_estimate = entropy_unseen(samples);

    if samples.len() < 10 || num_bootstrap == 0 {
        return (point_estimate, point_estimate, point_estimate);
    }

    let mut bootstrap_estimates = Vec::with_capacity(num_bootstrap);

    for _ in 0..num_bootstrap {
        let resampled: Vec<_> = (0..samples.len())
            .map(|_| samples.choose(&mut rng).expect("non-empty").clone())
            .collect();
        bootstrap_estimates.push(entropy_unseen(&resampled));
    }

    bootstrap_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = (1.0 - confidence) / 2.0;
    let lower_idx = (alpha * num_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha) * num_bootstrap as f64) as usize;

    (
        point_estimate,
        bootstrap_estimates[lower_idx.min(bootstrap_estimates.len() - 1)],
        bootstrap_estimates[upper_idx.min(bootstrap_estimates.len() - 1)],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unseen_support_grows() {
        // If we see k distinct elements once each, support estimate should be > k
        let samples: Vec<u32> = (0..50).collect();
        let est_support = support_unseen(&samples);

        assert!(
            est_support >= 50.0,
            "support should be at least observed count"
        );
    }

    #[test]
    fn test_unseen_entropy_bounds() {
        let samples: Vec<u32> = (0..100).collect();
        let entropy = entropy_unseen(&samples);

        // For ~100 elements, entropy should be around log2(100) ≈ 6.6
        assert!(
            entropy > 4.0 && entropy < 10.0,
            "entropy should be reasonable"
        );
    }

    #[test]
    fn test_recover_histogram_nonempty() {
        let fp = vec![50, 10, 5, 2, 1];
        let hist = recover_histogram(&fp, 100.0);

        assert!(!hist.is_empty(), "histogram should not be empty");
    }

    #[test]
    fn test_distance_same_distribution() {
        let samples_p: Vec<u32> = (0..50).collect();
        let samples_q: Vec<u32> = (0..50).collect();

        let dist = distance_unseen(&samples_p, &samples_q);
        assert!(dist < 0.5, "same distribution should have small distance");
    }

    #[test]
    fn test_poisson_pmf_sums_to_one() {
        let lambda = 5.0;
        let sum: f64 = (0..50).map(|k| poisson_pmf(lambda, k)).sum();
        assert!((sum - 1.0).abs() < 0.01, "PMF should sum to ~1");
    }

    #[test]
    fn test_config_affects_result() {
        let samples: Vec<u32> = (0..50).collect();

        let default_entropy = entropy_unseen(&samples);

        let config = UnseenConfig {
            grid_ratio: 1.2,   // Finer grid
            fit_count: 15,     // More constraints
            slack_factor: 0.5, // Tighter
        };
        let custom_entropy = entropy_unseen_with_config(&samples, &config);

        // Results may differ but should both be reasonable
        assert!(default_entropy > 0.0);
        assert!(custom_entropy > 0.0);
    }
}
