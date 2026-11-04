//! Regression tests for fixed numerical bugs. Each test states the correct
//! behavior and the reference it comes from.

use logp::{
    amari_alpha_divergence, digamma, jensen_shannon_divergence, kl_divergence, normalize_in_place,
    rho_alpha,
};

/// KL(p || q) is finite whenever q_i > 0 for every p_i > 0. With q_i = 1e-310
/// the ratio p_i / q_i overflows f64, but ln p_i - ln q_i does not.
#[test]
fn kl_with_subnormal_q_is_finite() {
    let p = [0.5, 0.5];
    let q = [1.0, 1e-310];
    let kl = kl_divergence(&p, &q, 1e-9).unwrap();
    let want = 0.5 * 0.5f64.ln() + 0.5 * (0.5f64.ln() - 1e-310f64.ln());
    assert!(
        kl.is_finite() && (kl - want).abs() < 1e-9,
        "kl = {kl}, want {want}"
    );
}

/// JS is finite on simplex inputs: m_i = (p_i + q_i) / 2 is positive whenever
/// p_i or q_i is, even when halving the smallest subnormal rounds to zero.
#[test]
fn js_with_subnormal_entry_is_finite() {
    let p = [1.0, 5e-324];
    let q = [1.0, 0.0];
    let js = jensen_shannon_divergence(&p, &q, 1e-9).unwrap();
    assert!(js.is_finite() && js.abs() < 1e-300, "js = {js}");
}

/// `normalize_in_place` either returns a vector that sums to 1 or an error.
/// Two entries of 1e308 have a sum that overflows f64; dividing by it would
/// zero the vector.
#[test]
fn normalize_in_place_rejects_overflowing_sum() {
    let mut v = [1e308, 1e308];
    let r = normalize_in_place(&mut v);
    assert!(r.is_err(), "r = {r:?}, v = {v:?}");
    assert_eq!(v, [1e308, 1e308], "input modified on error");
}

/// rho_alpha is continuous in alpha. At alpha = 0 it equals the limit from
/// above, sum over {p_i > 0} of q_i (van Erven and Harremoës 2014, Rényi
/// divergence of order 0 is -ln Q(p > 0)), so disjoint supports give 0.
/// Symmetrically, alpha = 1 gives sum over {q_i > 0} of p_i.
#[test]
fn rho_alpha_is_continuous_at_zero_and_one_with_zero_entries() {
    let p = [1.0, 0.0];
    let q = [0.0, 1.0];
    for (alpha, near) in [(0.0, 1e-9), (1.0, 1.0 - 1e-9)] {
        let at = rho_alpha(&p, &q, alpha, 1e-9).unwrap();
        let near = rho_alpha(&p, &q, near, 1e-9).unwrap();
        assert!(
            (at - near).abs() < 1e-6,
            "alpha = {alpha}: {at}, nearby: {near}"
        );
    }
    let p = [0.5, 0.5, 0.0];
    let q = [0.2, 0.3, 0.5];
    let at = rho_alpha(&p, &q, 0.0, 1e-9).unwrap();
    assert!(
        (at - 0.5).abs() < 1e-12,
        "rho_0 = {at}, want Q(p > 0) = 0.5"
    );
}

/// The simplex validation tolerance does not change which divergence
/// `amari_alpha_divergence` computes: alpha = -0.95 is not alpha = -1.
#[test]
fn amari_alpha_does_not_depend_on_validation_tolerance() {
    let p = [0.3, 0.7];
    let q = [0.5, 0.5];
    let alpha = -0.95;
    let strict = amari_alpha_divergence(&p, &q, alpha, 1e-12).unwrap();
    let loose = amari_alpha_divergence(&p, &q, alpha, 1e-2).unwrap();
    assert!(
        (strict - loose).abs() < 1e-12,
        "tol 1e-12: {strict}, tol 1e-2: {loose}"
    );
}

/// `digamma` matches `scipy.special.digamma` to the documented ~1e-14.
#[test]
fn digamma_matches_scipy_reference() {
    let reference = [
        (0.001, -1_000.575_571_931_810_3),
        (0.1, -10.423_754_940_411_076),
        (0.5, -1.963_510_026_021_423_5),
        (1.0, -0.577_215_664_901_532_9),
        (2.0, 0.422_784_335_098_467_13),
        (5.0, 1.506_117_668_431_800_3),
        (9.99, 2.250_700_372_831_201),
        (10.0, 2.251_752_589_066_721),
        (10.5, 2.303_001_034_297_686_5),
        (20.0, 2.970_523_992_242_149),
        (100.0, 4.600_161_852_738_088),
        (1e6, 13.815_510_057_964_19),
    ];
    for (x, want) in reference {
        let got = digamma(x);
        let err = (got - want).abs() / want.abs().max(1.0);
        assert!(
            err < 1e-14,
            "digamma({x}) = {got}, scipy {want}, error {err:.2e}"
        );
    }
}
