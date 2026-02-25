//! Estimate mutual information between continuous variables using the KSG estimator.
//!
//! The Kraskov-Stogbauer-Grassberger (KSG) estimator uses k-nearest neighbors
//! to estimate MI from samples, without binning. This makes it work well for
//! continuous and high-dimensional data where histograms would be sparse.
//!
//! We demonstrate on bivariate Gaussians with varying correlation:
//! - rho = 0: independent, MI should be ~0
//! - rho = 0.5: moderate correlation
//! - rho = 0.9: strong correlation, high MI
//! - rho = 0.99: near-deterministic, MI approaches H(X)
//!
//! Theoretical MI for bivariate Gaussian: MI = -0.5 * ln(1 - rho^2)
//!
//! Run: cargo run --example ksg_mutual_information

use logp::{mutual_information_ksg, KsgVariant};

fn main() {
    let n = 2000;
    let k = 6; // KSG neighbor count
    let seed = 42u64;

    println!("KSG Mutual Information Estimator");
    println!("================================");
    println!("n={} samples, k={} neighbors\n", n, k);
    println!("{:<8} {:>12} {:>12} {:>8}", "rho", "MI (theory)", "MI (KSG)", "error");
    println!("{}", "-".repeat(44));

    for &rho in &[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99] {
        let theoretical = -0.5 * (1.0_f64 - rho * rho).ln();

        // Generate correlated Gaussian samples using Box-Muller + Cholesky
        let (xs, ys) = correlated_gaussian_samples(n, rho, seed);
        let mi = mutual_information_ksg(&xs, &ys, k, KsgVariant::Alg1).unwrap();

        let err = (mi - theoretical).abs();
        println!(
            "{:<8.2} {:>12.4} {:>12.4} {:>8.4}",
            rho, theoretical, mi, err
        );
    }

    println!();
    println!("Note: KSG estimates are unbiased for large n but noisy for small n.");
    println!("The estimator works without binning, making it ideal for continuous data.");
}

/// Generate n samples from a bivariate Gaussian with correlation rho.
/// Uses a simple LCG + Box-Muller for reproducibility (no external RNG dep).
fn correlated_gaussian_samples(n: usize, rho: f64, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut state = seed;
    let mut next_uniform = || -> f64 {
        // LCG (Numerical Recipes)
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };

    let mut next_normal = || -> f64 {
        // Box-Muller
        let u1 = next_uniform().max(1e-15);
        let u2 = next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);

    for _ in 0..n {
        let z1 = next_normal();
        let z2 = next_normal();
        let x = z1;
        let y = rho * z1 + (1.0 - rho * rho).sqrt() * z2; // Cholesky decomposition
        xs.push(vec![x]);
        ys.push(vec![y]);
    }

    (xs, ys)
}
