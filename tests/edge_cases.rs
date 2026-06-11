//! Edge-case pins for the most-used entropy / divergence functions.
//!
//! Each test asserts a specific value or a specific error variant so that a
//! change to validation or zero-handling conventions fails loudly. External
//! consumers depend on these behaviors (empty input, zero probabilities,
//! unnormalized distributions, exact error variants) staying as documented.

use core::f64::consts::LN_2;
use logp::{
    cross_entropy_nats, entropy_nats, entropy_unchecked, jensen_shannon_divergence, kl_divergence,
    kl_divergence_gaussians, mutual_information, mutual_information_ksg, Error, KsgVariant,
};

const TOL: f64 = 1e-9;

// --- Empty input ---

#[test]
fn entropy_empty_is_error() {
    assert!(matches!(entropy_nats(&[], TOL), Err(Error::Empty)));
}

#[test]
fn kl_empty_is_error() {
    assert!(matches!(kl_divergence(&[], &[], TOL), Err(Error::Empty)));
}

#[test]
fn cross_entropy_empty_is_error() {
    assert!(matches!(
        cross_entropy_nats(&[], &[], TOL),
        Err(Error::Empty)
    ));
}

// --- Zero probability: the 0 ln 0 = 0 convention, exact values ---

#[test]
fn entropy_with_zero_entry_is_exactly_zero() {
    // 0 ln 0 = 0 by continuity: a delta with explicit zero bins has H = 0.
    let h = entropy_nats(&[0.0, 1.0], TOL).unwrap();
    assert_eq!(h, 0.0);
}

#[test]
fn kl_skips_zero_p_entries() {
    // p_i = 0 entries contribute nothing even where q_i > 0:
    // KL([0,1] || [0.5,0.5]) = 1 * ln(1/0.5) = ln 2.
    let kl = kl_divergence(&[0.0, 1.0], &[0.5, 0.5], TOL).unwrap();
    assert!((kl - LN_2).abs() < 1e-15, "kl={kl}");
}

#[test]
fn cross_entropy_skips_zero_p_entries() {
    // Only the support of p needs q > 0: H([0,1], [0.5,0.5]) = -ln(0.5) = ln 2.
    let h = cross_entropy_nats(&[0.0, 1.0], &[0.5, 0.5], TOL).unwrap();
    assert!((h - LN_2).abs() < 1e-15, "h={h}");
    // q may be zero exactly where p is zero.
    let h = cross_entropy_nats(&[0.0, 1.0], &[0.0, 1.0], TOL).unwrap();
    assert_eq!(h, 0.0);
}

#[test]
fn kl_zero_q_on_p_support_is_domain_error() {
    assert!(matches!(
        kl_divergence(&[0.5, 0.5], &[1.0, 0.0], TOL),
        Err(Error::Domain(_))
    ));
}

#[test]
fn js_with_disjoint_supports_is_exactly_ln2() {
    // m_i > 0 wherever p_i or q_i > 0, so JS stays finite even at disjoint
    // supports, where it attains its upper bound ln 2.
    let js = jensen_shannon_divergence(&[1.0, 0.0], &[0.0, 1.0], TOL).unwrap();
    assert!((js - LN_2).abs() < 1e-12, "js={js}");
}

// --- Unnormalized / invalid distributions ---

#[test]
fn entropy_unnormalized_is_not_normalized_error() {
    match entropy_nats(&[0.3, 0.3], TOL) {
        Err(Error::NotNormalized { sum }) => assert!((sum - 0.6).abs() < 1e-12, "sum={sum}"),
        other => panic!("expected NotNormalized, got {other:?}"),
    }
}

#[test]
fn entropy_negative_entry_reports_index_and_value() {
    match entropy_nats(&[1.5, -0.5], TOL) {
        Err(Error::Negative { idx, value }) => {
            assert_eq!(idx, 1);
            assert!((value - (-0.5)).abs() < 1e-15, "value={value}");
        }
        other => panic!("expected Negative, got {other:?}"),
    }
}

#[test]
fn entropy_nan_entry_is_non_finite_error() {
    assert!(matches!(
        entropy_nats(&[f64::NAN, 1.0], TOL),
        Err(Error::NonFinite { idx: 0, .. })
    ));
}

#[test]
fn kl_length_mismatch_reports_lengths() {
    assert!(matches!(
        kl_divergence(&[0.5, 0.5], &[0.3, 0.3, 0.4], TOL),
        Err(Error::LengthMismatch(2, 3))
    ));
}

// --- entropy_unchecked: documented non-validating behavior ---

#[test]
fn entropy_unchecked_empty_is_zero() {
    assert_eq!(entropy_unchecked(&[]), 0.0);
}

#[test]
fn entropy_unchecked_unnormalized_goes_negative() {
    // No validation: an entry > 1 produces a negative "entropy".
    let h = entropy_unchecked(&[2.0]);
    let expected = -(2.0 * 2.0_f64.ln());
    assert!((h - expected).abs() < 1e-15, "h={h}");
}

// --- Diagonal-Gaussian KL ---

#[test]
fn kl_gaussians_empty_is_zero() {
    // Zero-dimensional Gaussians: empty sum, KL = 0.
    let kl = kl_divergence_gaussians(&[], &[], &[], &[]).unwrap();
    assert_eq!(kl, 0.0);
}

#[test]
fn kl_gaussians_zero_std_is_domain_error() {
    assert!(matches!(
        kl_divergence_gaussians(&[0.0], &[0.0], &[0.0], &[1.0]),
        Err(Error::Domain(_))
    ));
}

// --- Discrete MI shape errors ---

#[test]
fn mutual_information_shape_mismatch() {
    let p_xy = [0.25, 0.25, 0.25, 0.25];
    assert!(matches!(
        mutual_information(&p_xy, 2, 3, TOL),
        Err(Error::LengthMismatch(4, 6))
    ));
    assert!(matches!(
        mutual_information(&p_xy, 0, 4, TOL),
        Err(Error::Domain(_))
    ));
}

// --- KSG argument contract ---

#[test]
fn ksg_rejects_empty_input_and_bad_k() {
    let empty: Vec<Vec<f64>> = vec![];
    // Empty input: n = 0 <= k.
    assert!(matches!(
        mutual_information_ksg(&empty, &empty, 3, KsgVariant::Alg1),
        Err(Error::Domain(_))
    ));

    let x: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
    let y = x.clone();
    // k = 0 is rejected.
    assert!(matches!(
        mutual_information_ksg(&x, &y, 0, KsgVariant::Alg1),
        Err(Error::Domain(_))
    ));
    // n == k is rejected (need n > k).
    assert!(matches!(
        mutual_information_ksg(&x, &y, 5, KsgVariant::Alg2),
        Err(Error::Domain(_))
    ));
    // Sample-count mismatch.
    assert!(matches!(
        mutual_information_ksg(&x, &y[..4], 2, KsgVariant::Alg1),
        Err(Error::LengthMismatch(5, 4))
    ));
}

#[test]
fn cross_entropy_rejects_length_mismatch() {
    // Regression: previously zip-truncated and returned a silently wrong value.
    let p = [0.5, 0.5];
    let q = [0.3, 0.3, 0.4];
    assert!(matches!(
        logp::cross_entropy_nats(&p, &q, 1e-9),
        Err(logp::Error::LengthMismatch(2, 3))
    ));
}

#[test]
fn renyi_tsallis_divergence_reject_nonpositive_alpha() {
    let p = [0.5, 0.5];
    let q = [0.4, 0.6];
    for alpha in [0.0, -1.0, f64::NAN] {
        assert!(matches!(
            logp::renyi_divergence(&p, &q, alpha, 1e-9),
            Err(logp::Error::InvalidAlpha { .. })
        ));
        assert!(matches!(
            logp::tsallis_divergence(&p, &q, alpha, 1e-9),
            Err(logp::Error::InvalidAlpha { .. })
        ));
    }
}
