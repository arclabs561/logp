//! Feature selection via KSG mutual information.
//!
//! Generates a synthetic dataset (1000 samples, 8 features, 1 target) and ranks
//! features by their estimated MI with the target.
//!
//! - Features 0-2: linearly correlated with target (high MI expected)
//! - Feature 3: nonlinearly related (quadratic, moderate MI expected)
//! - Features 4-7: independent noise (MI near zero)
//!
//! KSG handles the nonlinear feature without binning, unlike histogram estimators.
//!
//! Run: cargo run --example feature_selection

use logp::{mutual_information_ksg, KsgVariant};

fn main() {
    let n = 1000;
    let k = 5;
    let num_features = 8;

    // -- Deterministic RNG (LCG + Box-Muller, same as other examples) ----------
    let mut state = 31415u64;
    let mut next_uniform = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut next_normal = || -> f64 {
        let u1 = next_uniform().max(1e-15);
        let u2 = next_uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    // -- Generate dataset ------------------------------------------------------
    // Target: standard normal
    let target: Vec<f64> = (0..n).map(|_| next_normal()).collect();

    // Features: each is a 1-D series of n samples.
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(num_features);

    // Features 0-2: linear relationship with target + noise
    let coeffs = [0.9, 0.7, 0.5];
    for &c in &coeffs {
        let col: Vec<f64> = target
            .iter()
            .map(|&t| c * t + (1.0 - c * c).sqrt() * next_normal())
            .collect();
        features.push(col);
    }

    // Feature 3: quadratic relationship (target^2 + noise)
    let col3: Vec<f64> = target
        .iter()
        .map(|&t| t * t + 0.3 * next_normal())
        .collect();
    features.push(col3);

    // Features 4-7: independent noise
    for _ in 4..num_features {
        let col: Vec<f64> = (0..n).map(|_| next_normal()).collect();
        features.push(col);
    }

    // -- Estimate MI for each feature ------------------------------------------
    let target_vecs: Vec<Vec<f64>> = target.iter().map(|&t| vec![t]).collect();
    let mut mi_estimates: Vec<(usize, f64)> = Vec::with_capacity(num_features);

    for (idx, feat) in features.iter().enumerate() {
        let feat_vecs: Vec<Vec<f64>> = feat.iter().map(|&v| vec![v]).collect();
        let mi = mutual_information_ksg(&feat_vecs, &target_vecs, k, KsgVariant::Alg1).unwrap();
        mi_estimates.push((idx, mi));
    }

    // -- Rank by MI (descending) -----------------------------------------------
    mi_estimates.sort_by(|a, b| b.1.total_cmp(&a.1));

    let labels = [
        "linear (c=0.9)",
        "linear (c=0.7)",
        "linear (c=0.5)",
        "quadratic",
        "noise",
        "noise",
        "noise",
        "noise",
    ];

    println!("Feature Selection via KSG Mutual Information");
    println!("=============================================");
    println!("n={n} samples, k={k} neighbors, {num_features} features\n");
    println!(
        "{:>4}  {:>10}  {:<18}  {}",
        "rank", "MI (KSG)", "feature", "type"
    );
    println!("{}", "-".repeat(52));

    for (rank, &(idx, mi)) in mi_estimates.iter().enumerate() {
        println!(
            "{:>4}  {:>10.4}  {:<18}  {}",
            rank + 1,
            mi,
            format!("feature_{idx}"),
            labels[idx],
        );
    }

    // -- Sanity check: informative features outrank noise ----------------------
    let top3: Vec<usize> = mi_estimates.iter().take(3).map(|&(idx, _)| idx).collect();
    let all_informative = top3.iter().all(|&idx| idx <= 3);
    println!();
    if all_informative {
        println!("OK: top 3 features are all from the informative set (0-3).");
    } else {
        println!("UNEXPECTED: a noise feature ranked in the top 3.");
    }
}
