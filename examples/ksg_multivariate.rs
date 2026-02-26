//! KSG mutual information estimation in high dimensions.
//!
//! Histogram-based MI estimation fails in high dimensions because bins become
//! exponentially sparse. The KSG estimator uses k-nearest neighbors instead,
//! avoiding the curse of dimensionality for bin-based methods.
//!
//! This example estimates MI between two halves of a 10D multivariate Gaussian
//! with a known correlation structure. The analytical MI is:
//!
//!   MI(X; Y) = -0.5 * ln(det(Sigma) / (det(Sigma_X) * det(Sigma_Y)))
//!
//! For a block-diagonal-off-diagonal structure with uniform cross-correlation rho,
//! the determinants have closed forms.
//!
//! We show convergence as sample size increases: n = 200, 500, 1000, 2000, 5000.
//!
//! Run: cargo run --example ksg_multivariate

use logp::{mutual_information_ksg, KsgVariant};

fn main() {
    let dx = 5; // dimension of X
    let dy = 5; // dimension of Y
    let rho = 0.5_f64; // cross-correlation between each X_i and each Y_j pair
    let k = 6;

    // Analytical MI for this structure.
    // We use a specific correlation matrix:
    //   Sigma_XX = I_dx, Sigma_YY = I_dy, Sigma_XY = rho * I (when dx == dy)
    // So Sigma is a 2d x 2d block matrix:
    //   [ I      rho*I ]
    //   [ rho*I  I     ]
    // det(Sigma) = det(I - rho^2 * I)^d = (1 - rho^2)^d  (for dx == dy == d)
    // det(Sigma_X) = 1, det(Sigma_Y) = 1
    // MI = -0.5 * d * ln(1 - rho^2)
    let d = dx; // dx == dy
    let theoretical_mi = -0.5 * d as f64 * (1.0 - rho * rho).ln();

    println!("KSG Mutual Information: Multivariate Gaussian ({dx}D + {dy}D = {}D joint)", dx + dy);
    println!("  cross-correlation rho = {rho}");
    println!("  theoretical MI = {theoretical_mi:.4} nats");
    println!("  k = {k} neighbors");
    println!();

    let sample_sizes = [200, 500, 1000, 2000, 5000];

    println!(
        "{:>6} {:>12} {:>12} {:>10} {:>10}",
        "n", "MI (theory)", "MI (KSG)", "abs_err", "rel_err%"
    );
    println!("{}", "-".repeat(54));

    for &n in &sample_sizes {
        let (xs, ys) = generate_correlated_gaussian(n, dx, dy, rho, 42);
        let mi = mutual_information_ksg(&xs, &ys, k, KsgVariant::Alg1).unwrap();
        let abs_err = (mi - theoretical_mi).abs();
        let rel_err = abs_err / theoretical_mi * 100.0;

        println!(
            "{:>6} {:>12.4} {:>12.4} {:>10.4} {:>9.1}%",
            n, theoretical_mi, mi, abs_err, rel_err
        );
    }

    println!();
    println!("The estimate converges toward the true value as n increases.");
    println!("A histogram estimator with even 10 bins per dimension would need");
    println!("10^{} bins for the joint space -- infeasible at these sample sizes.", dx + dy);
}

/// Generate n samples from a multivariate Gaussian where:
///   X ~ N(0, I_dx)
///   Y_j = rho * X_j + sqrt(1 - rho^2) * Z_j,  Z ~ N(0, I_dy)
///
/// This gives Cov(X_i, Y_j) = rho if i == j, 0 otherwise (when dx == dy).
/// Uses LCG + Box-Muller for reproducibility without external RNG dependencies.
fn generate_correlated_gaussian(
    n: usize,
    dx: usize,
    dy: usize,
    rho: f64,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut state = seed;
    let mut next_uniform = || -> f64 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut next_normal = || -> f64 {
        let u1 = next_uniform().max(1e-15);
        let u2 = next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let scale = (1.0 - rho * rho).sqrt();
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);

    for _ in 0..n {
        let x: Vec<f64> = (0..dx).map(|_| next_normal()).collect();
        let y: Vec<f64> = (0..dy)
            .map(|j| {
                let z = next_normal();
                if j < dx {
                    rho * x[j] + scale * z
                } else {
                    z // independent dimensions if dy > dx
                }
            })
            .collect();
        xs.push(x);
        ys.push(y);
    }

    (xs, ys)
}
