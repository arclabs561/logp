//! Kraskov-Stögbauer-Grassberger (KSG) Mutual Information Estimator.
//!
//! Gold standard for non-parametric MI estimation from continuous samples.

use crate::{digamma, Error, Result};

/// Algorithm variant for KSG estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KsgVariant {
    /// Algorithm 1: Neighbors within radius epsilon.
    Alg1,
    /// Algorithm 2: Neighbors within radius epsilon/2. More robust for independence tests.
    Alg2,
}

/// Estimate Mutual Information \(I(X;Y)\) using the KSG estimator.
///
/// # Arguments
/// * `x` - Samples from X (N x Dx)
/// * `y` - Samples from Y (N x Dy)
/// * `k` - Number of neighbors (typically 3-5)
/// * `variant` - KSG Algorithm 1 or 2
///
/// # Examples
///
/// ```
/// # use logp::{mutual_information_ksg, KsgVariant};
/// // Perfectly correlated (Y = X): MI should be substantially positive.
/// let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 / 30.0]).collect();
/// let y_corr = x.clone();
/// let mi = mutual_information_ksg(&x, &y_corr, 3, KsgVariant::Alg1).unwrap();
/// assert!(mi > 0.0);
///
/// // Result is finite.
/// assert!(mi.is_finite());
/// ```
pub fn mutual_information_ksg(
    x: &[Vec<f64>],
    y: &[Vec<f64>],
    k: usize,
    variant: KsgVariant,
) -> Result<f64> {
    let n = x.len();
    if n != y.len() {
        return Err(Error::LengthMismatch(n, y.len()));
    }
    if n <= k {
        return Err(Error::Domain("Sample size must be greater than k"));
    }
    if k == 0 {
        return Err(Error::Domain("k must be >= 1"));
    }

    // KSG expects fixed-dimensional vectors in each space.
    // If dimensions drift, dist_inf would silently ignore trailing coordinates.
    let dx = x.first().map(|v| v.len()).unwrap_or(0);
    let dy = y.first().map(|v| v.len()).unwrap_or(0);
    if dx == 0 || dy == 0 {
        return Err(Error::Domain("x and y must have non-empty feature vectors"));
    }
    if x.iter().any(|v| v.len() != dx) {
        return Err(Error::Domain("x has inconsistent sample dimensionality"));
    }
    if y.iter().any(|v| v.len() != dy) {
        return Err(Error::Domain("y has inconsistent sample dimensionality"));
    }

    let mut nx = vec![0usize; n];
    let mut ny = vec![0usize; n];

    for i in 0..n {
        // 1. Find k-th neighbor distance in joint space (infinity norm)
        let mut joint_dists = Vec::with_capacity(n);
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = dist_inf(&x[i], &x[j]);
            let dy = dist_inf(&y[i], &y[j]);
            joint_dists.push(dx.max(dy));
        }
        joint_dists.sort_by(|a, b| a.total_cmp(b));
        let eps = joint_dists[k - 1];

        match variant {
            KsgVariant::Alg1 => {
                // Count neighbors in marginal spaces with distance < eps
                nx[i] = x
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| i != *j && dist_inf(&x[i], &x[*j]) < eps)
                    .count();
                ny[i] = y
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| i != *j && dist_inf(&y[i], &y[*j]) < eps)
                    .count();
            }
            KsgVariant::Alg2 => {
                // Count neighbors in marginal spaces with distance <= eps.
                //
                // KSG Alg2's closed form uses ψ(n_x) and ψ(n_y). For that to be well-defined,
                // n_x and n_y must be ≥ 1. The standard convention is that Alg2 counts include
                // the point itself (so the minimum count is 1).
                nx[i] = x
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| dist_inf(&x[i], &x[*j]) <= eps || i == *j)
                    .count();
                ny[i] = y
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| dist_inf(&y[i], &y[*j]) <= eps || i == *j)
                    .count();
            }
        }
    }

    match variant {
        KsgVariant::Alg1 => {
            // I(1) = psi(k) - <psi(nx + 1) + psi(ny + 1)> + psi(N)
            let avg_psi: f64 = nx
                .iter()
                .zip(ny.iter())
                .map(|(&nxi, &nyi)| digamma(nxi as f64 + 1.0) + digamma(nyi as f64 + 1.0))
                .sum::<f64>()
                / n as f64;
            Ok(digamma(k as f64) - avg_psi + digamma(n as f64))
        }
        KsgVariant::Alg2 => {
            // I(2) = psi(k) - 1/k - <psi(nx) + psi(ny)> + psi(N)
            let avg_psi: f64 = nx
                .iter()
                .zip(ny.iter())
                .map(|(&nxi, &nyi)| digamma(nxi as f64) + digamma(nyi as f64))
                .sum::<f64>()
                / n as f64;
            Ok(digamma(k as f64) - 1.0 / k as f64 - avg_psi + digamma(n as f64))
        }
    }
}

fn dist_inf(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ksg_independent_alg1() {
        let x = vec![vec![0.1], vec![0.5], vec![0.9], vec![0.2], vec![0.8]];
        let y = vec![vec![0.9], vec![0.2], vec![0.1], vec![0.8], vec![0.5]];
        let mi = mutual_information_ksg(&x, &y, 2, KsgVariant::Alg1).unwrap();
        assert!(mi.is_finite());
        assert!(mi.abs() < 2.0);
    }

    #[test]
    fn test_ksg_independent_alg2() {
        let x = vec![vec![0.1], vec![0.5], vec![0.9], vec![0.2], vec![0.8]];
        let y = vec![vec![0.9], vec![0.2], vec![0.1], vec![0.8], vec![0.5]];
        let mi = mutual_information_ksg(&x, &y, 2, KsgVariant::Alg2).unwrap();
        assert!(mi.is_finite());
        assert!(mi.abs() < 2.0);
    }

    #[test]
    fn test_ksg_correlated_is_larger_than_shuffled() {
        // Deterministic “correlated vs shuffled” sanity check.
        // (Not a precise benchmark; just catches obvious sign/NaN issues.)
        let x: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64 / 40.0]).collect();
        let y_corr = x.clone();
        let y_shuf: Vec<Vec<f64>> = (0..40).rev().map(|i| vec![i as f64 / 40.0]).collect();

        let mi_corr = mutual_information_ksg(&x, &y_corr, 3, KsgVariant::Alg1).unwrap();
        let mi_shuf = mutual_information_ksg(&x, &y_shuf, 3, KsgVariant::Alg1).unwrap();
        assert!(mi_corr.is_finite() && mi_shuf.is_finite());
        assert!(mi_corr > mi_shuf);
    }

    #[test]
    fn test_ksg_rejects_inconsistent_dims() {
        let x = vec![vec![0.1], vec![0.2, 0.3]];
        let y = vec![vec![0.1], vec![0.2]];
        let err = mutual_information_ksg(&x, &y, 1, KsgVariant::Alg1).unwrap_err();
        assert!(matches!(err, Error::Domain(_)));
    }
}
