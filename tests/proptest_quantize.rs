use logp::quantize::{dequantize, optimal_codebook, quantize, QuantizeDist};
use proptest::prelude::*;
use std::f64::consts::PI;

fn dist_strategy() -> impl Strategy<Value = QuantizeDist> {
    prop_oneof![
        Just(QuantizeDist::Gaussian),
        Just(QuantizeDist::Logistic),
        Just(QuantizeDist::Cauchy),
    ]
}

/// Generate a random probability simplex of size `n`.
fn simplex_strategy(n: usize) -> impl Strategy<Value = Vec<f64>> {
    proptest::collection::vec(0.001f64..100.0, n).prop_map(|raw| {
        let sum: f64 = raw.iter().sum();
        raw.into_iter().map(|x| x / sum).collect()
    })
}

// ── Quantization properties ─────────────────────────────────────────────────

proptest! {
    #[test]
    fn boundaries_are_sorted(dist in dist_strategy(), levels in 2usize..=32) {
        let cb = optimal_codebook(dist, levels).unwrap();
        for w in cb.boundaries.windows(2) {
            prop_assert!(w[0] < w[1], "unsorted boundaries for {dist:?}: {w:?}");
        }
    }

    #[test]
    fn representatives_within_boundaries(dist in dist_strategy(), levels in 2usize..=32) {
        let cb = optimal_codebook(dist, levels).unwrap();
        let n = cb.representatives.len();
        for i in 0..n {
            let lo = if i == 0 { f64::NEG_INFINITY } else { cb.boundaries[i - 1] };
            let hi = if i == n - 1 { f64::INFINITY } else { cb.boundaries[i] };
            let r = cb.representatives[i];
            prop_assert!(r > lo && r < hi, "rep {r} not in ({lo}, {hi}) for {dist:?} level {i}");
        }
    }

    #[test]
    fn level_counts_match(dist in dist_strategy(), levels in 1usize..=32) {
        let cb = optimal_codebook(dist, levels).unwrap();
        prop_assert_eq!(cb.representatives.len(), levels);
        prop_assert_eq!(cb.boundaries.len(), levels.saturating_sub(1));
    }

    #[test]
    fn symmetric_codebook_for_even_levels(dist in dist_strategy(), half in 1usize..=16) {
        let levels = half * 2;
        let cb = optimal_codebook(dist, levels).unwrap();
        let n = cb.representatives.len();
        for i in 0..n / 2 {
            let lo = cb.representatives[i];
            let hi = cb.representatives[n - 1 - i];
            prop_assert!((lo + hi).abs() < 1e-6, "{dist:?}: reps not symmetric: {lo} vs {hi}");
        }
        for i in 0..cb.boundaries.len() / 2 {
            let lo = cb.boundaries[i];
            let hi = cb.boundaries[cb.boundaries.len() - 1 - i];
            prop_assert!((lo + hi).abs() < 1e-6, "{dist:?}: bounds not symmetric: {lo} vs {hi}");
        }
    }

    #[test]
    fn roundtrip_on_representatives(dist in dist_strategy(), levels in 2usize..=32) {
        let cb = optimal_codebook(dist, levels).unwrap();
        for (i, &r) in cb.representatives.iter().enumerate() {
            let idx = quantize(r, &cb.boundaries);
            prop_assert_eq!(idx, i, "representative {} should map to level {}, got {}", r, i, idx);
            let recovered = dequantize(idx, &cb.representatives);
            prop_assert!((recovered - r).abs() < 1e-14, "dequantize(quantize(rep)) not exact");
        }
    }
}

/// Increasing levels should decrease quantization error (measured by expected MSE
/// via numerical integration over the Gaussian case).
#[test]
fn increasing_levels_decreasing_error_gaussian() {
    let mut prev_error = f64::INFINITY;
    for levels in [2, 4, 8, 16] {
        let cb = optimal_codebook(QuantizeDist::Gaussian, levels).unwrap();
        // Approximate expected MSE by sampling
        let n_samples = 10_000;
        let mut mse = 0.0;
        for i in 0..n_samples {
            // Use deterministic quasi-random points via inverse CDF
            let u = (i as f64 + 0.5) / n_samples as f64;
            // Inverse CDF of standard normal (probit) via rational approximation
            let x = probit(u);
            let idx = quantize(x, &cb.boundaries);
            let r = dequantize(idx, &cb.representatives);
            mse += (x - r).powi(2);
        }
        mse /= n_samples as f64;
        assert!(
            mse < prev_error,
            "MSE did not decrease: {levels} levels gave {mse}, previous was {prev_error}"
        );
        prev_error = mse;
    }
}

/// Gaussian 2-level codebook: representatives should match E[|X|] = sqrt(2/pi).
#[test]
fn gaussian_two_level_matches_expected_abs() {
    let cb = optimal_codebook(QuantizeDist::Gaussian, 2).unwrap();
    let expected = (2.0 / PI).sqrt();
    assert!(
        (cb.representatives[1] - expected).abs() < 1e-5,
        "expected {expected}, got {}",
        cb.representatives[1]
    );
}

// ── Divergence properties ───────────────────────────────────────────────────

proptest! {
    #[test]
    fn entropy_non_negative(p in simplex_strategy(5)) {
        let h = logp::entropy_nats(&p, 1e-6).unwrap();
        prop_assert!(h >= -1e-12, "entropy should be non-negative, got {h}");
    }

    #[test]
    fn kl_self_is_zero(p in simplex_strategy(5)) {
        let kl = logp::kl_divergence(&p, &p, 1e-6).unwrap();
        prop_assert!(kl.abs() < 1e-10, "KL(p, p) should be 0, got {kl}");
    }

    #[test]
    fn kl_non_negative(
        p in simplex_strategy(4),
        q in simplex_strategy(4),
    ) {
        // Both have strictly positive entries (from simplex_strategy min 0.001)
        let kl = logp::kl_divergence(&p, &q, 1e-6).unwrap();
        prop_assert!(kl >= -1e-12, "KL should be non-negative, got {kl}");
    }

    #[test]
    fn jensen_shannon_symmetric(
        p in simplex_strategy(4),
        q in simplex_strategy(4),
    ) {
        let js_pq = logp::jensen_shannon_divergence(&p, &q, 1e-6).unwrap();
        let js_qp = logp::jensen_shannon_divergence(&q, &p, 1e-6).unwrap();
        prop_assert!(
            (js_pq - js_qp).abs() < 1e-12,
            "JS not symmetric: {js_pq} vs {js_qp}"
        );
    }

    #[test]
    fn hellinger_bounded_zero_one(
        p in simplex_strategy(4),
        q in simplex_strategy(4),
    ) {
        let h = logp::hellinger(&p, &q, 1e-6).unwrap();
        prop_assert!(h >= -1e-12 && h <= 1.0 + 1e-12, "hellinger not in [0,1]: {h}");
    }

    #[test]
    fn total_variation_bounded_zero_one(
        p in simplex_strategy(4),
        q in simplex_strategy(4),
    ) {
        let tv = logp::total_variation(&p, &q, 1e-6).unwrap();
        prop_assert!(tv >= -1e-12 && tv <= 1.0 + 1e-12, "TV not in [0,1]: {tv}");
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Rational approximation of the probit function (inverse CDF of N(0,1)).
/// Abramowitz & Stegun 26.2.23. Accurate to ~4.5e-4.
fn probit(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p < 0.5 {
        return -probit(1.0 - p);
    }
    let t = (-2.0 * (1.0 - p).ln()).sqrt();
    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;
    t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
}
