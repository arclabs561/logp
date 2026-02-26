//! Detecting distribution shift in categorical features using KL and JS divergence.
//!
//! A common production problem: a model was trained on data with a certain feature
//! distribution, and you want to detect when incoming batches have shifted away
//! from that reference. This is useful for:
//! - data quality monitoring (ETL errors, schema changes)
//! - concept drift detection (changing user populations)
//! - A/B test sanity checks (are treatment/control populations balanced?)
//!
//! This example simulates a categorical feature (e.g., HTTP status codes, user
//! country, product category) with a known reference distribution, then generates
//! batches with increasing levels of drift. KL and JS divergence quantify the shift.
//!
//! Run: cargo run --example text_similarity

use logp::{entropy_nats, jensen_shannon_divergence, kl_divergence};

fn main() {
    // Reference distribution: 6 categories (e.g., HTTP status buckets).
    let categories = ["2xx", "3xx", "4xx", "403", "5xx", "timeout"];
    let reference = [0.70, 0.10, 0.10, 0.03, 0.05, 0.02];
    let tol = 1e-12;

    println!("Distribution shift detection via divergences");
    println!();
    println!("Reference distribution (e.g., from training data):");
    for (cat, &p) in categories.iter().zip(&reference) {
        println!("  {cat:<8} {p:.2}");
    }
    let h_ref = entropy_nats(&reference, tol).unwrap();
    println!("  entropy: {h_ref:.4} nats");
    println!();

    // Simulate batches with increasing drift.
    // Drift scenario: 5xx and timeout rates increase, 2xx decreases.
    let batches: Vec<(&str, [f64; 6])> = vec![
        ("no_drift    ", [0.70, 0.10, 0.10, 0.03, 0.05, 0.02]),
        ("mild_drift  ", [0.65, 0.10, 0.10, 0.04, 0.08, 0.03]),
        ("moderate    ", [0.55, 0.10, 0.12, 0.05, 0.12, 0.06]),
        ("severe_drift", [0.40, 0.08, 0.15, 0.07, 0.20, 0.10]),
        ("broken_etl  ", [0.10, 0.05, 0.30, 0.15, 0.25, 0.15]),
    ];

    println!(
        "{:<14} {:>8} {:>8} {:>10} {:>10}",
        "batch", "JS", "KL(b|r)", "KL(r|b)", "alert?"
    );
    println!("{}", "-".repeat(54));

    // Threshold: JS > 0.01 nats is a common heuristic for "worth investigating".
    let js_threshold = 0.01;

    for (name, batch_dist) in &batches {
        // Smooth to avoid log(0): add epsilon and renormalize.
        let smoothed = smooth(&reference, 1e-10);
        let batch_smoothed = smooth(batch_dist, 1e-10);

        let js = jensen_shannon_divergence(&smoothed, &batch_smoothed, tol).unwrap();
        let kl_batch_ref = kl_divergence(&batch_smoothed, &smoothed, tol).unwrap();
        let kl_ref_batch = kl_divergence(&smoothed, &batch_smoothed, tol).unwrap();

        let alert = if js > js_threshold { "YES" } else { "no" };
        println!(
            "{:<14} {:>8.5} {:>8.5} {:>10.5} {:>10}",
            name, js, kl_batch_ref, kl_ref_batch, alert
        );
    }

    println!();
    println!("JS divergence is symmetric and bounded [0, ln(2) ~ 0.693].");
    println!("KL(batch || ref) measures the information lost by using the reference");
    println!("to model the batch. Note KL is asymmetric: KL(b|r) != KL(r|b).");
    println!();
    println!("In practice, alert thresholds depend on batch size and domain.");
    println!("JS > {js_threshold} is a starting heuristic; calibrate on historical data.");
}

/// Add epsilon to each entry and renormalize to stay on the simplex.
fn smooth(p: &[f64], eps: f64) -> Vec<f64> {
    let shifted: Vec<f64> = p.iter().map(|&x| x + eps).collect();
    let s: f64 = shifted.iter().sum();
    shifted.iter().map(|&x| x / s).collect()
}
