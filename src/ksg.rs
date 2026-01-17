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

    let mut nx = vec![0usize; n];
    let mut ny = vec![0usize; n];

    for i in 0..n {
        // 1. Find k-th neighbor distance in joint space (infinity norm)
        let mut joint_dists = Vec::with_capacity(n);
        for j in 0..n {
            if i == j { continue; }
            let dx = dist_inf(&x[i], &x[j]);
            let dy = dist_inf(&y[i], &y[j]);
            joint_dists.push(dx.max(dy));
        }
        joint_dists.sort_by(|a, b| a.total_cmp(b));
        let eps = joint_dists[k - 1];

        match variant {
            KsgVariant::Alg1 => {
                // Count neighbors in marginal spaces with distance < eps
                nx[i] = x.iter().enumerate()
                    .filter(|(j, _)| i != *j && dist_inf(&x[i], &x[*j]) < eps)
                    .count();
                ny[i] = y.iter().enumerate()
                    .filter(|(j, _)| i != *j && dist_inf(&y[i], &y[*j]) < eps)
                    .count();
            }
            KsgVariant::Alg2 => {
                // Count neighbors in marginal spaces with distance <= eps
                nx[i] = x.iter().enumerate()
                    .filter(|(j, _)| i != *j && dist_inf(&x[i], &x[*j]) <= eps)
                    .count();
                ny[i] = y.iter().enumerate()
                    .filter(|(j, _)| i != *j && dist_inf(&y[i], &y[*j]) <= eps)
                    .count();
            }
        }
    }

    match variant {
        KsgVariant::Alg1 => {
            // I(1) = psi(k) - <psi(nx + 1) + psi(ny + 1)> + psi(N)
            let avg_psi: f64 = nx.iter().zip(ny.iter())
                .map(|(&nxi, &nyi)| digamma(nxi as f64 + 1.0) + digamma(nyi as f64 + 1.0))
                .sum::<f64>() / n as f64;
            Ok(digamma(k as f64) - avg_psi + digamma(n as f64))
        }
        KsgVariant::Alg2 => {
            // I(2) = psi(k) - 1/k - <psi(nx) + psi(ny)> + psi(N)
            let avg_psi: f64 = nx.iter().zip(ny.iter())
                .map(|(&nxi, &nyi)| digamma(nxi as f64) + digamma(nyi as f64))
                .sum::<f64>() / n as f64;
            Ok(digamma(k as f64) - 1.0 / k as f64 - avg_psi + digamma(n as f64))
        }
    }
}

fn dist_inf(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter())
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
        assert!(mi.abs() < 1.0);
    }

    #[test]
    fn test_ksg_independent_alg2() {
        let x = vec![vec![0.1], vec![0.5], vec![0.9], vec![0.2], vec![0.8]];
        let y = vec![vec![0.9], vec![0.2], vec![0.1], vec![0.8], vec![0.5]];
        let mi = mutual_information_ksg(&x, &y, 2, KsgVariant::Alg2).unwrap();
        assert!(mi.abs() < 1.0);
    }
}
