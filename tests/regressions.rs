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

#[test]
fn positive_bins_survive_extreme_orders() {
    let p = [0.25, 0.75];
    let q = [0.5, 0.5];
    let tol = 1e-9;
    for order in [-600.0, 600.0, f64::MAX] {
        assert_eq!(rho_alpha(&p, &p, order, tol).unwrap(), 1.0);
    }

    // rho = (0.5^a + 1.5^a)/2. At 1600 the sum fits in f64 and
    // the smaller term is negligible. At 1760 only the divergences fit.
    let log_expected = |a: f64| a * 1.5_f64.ln() - 2.0_f64.ln();
    let rho = rho_alpha(&p, &q, 1600.0, tol).unwrap();
    assert!((rho / log_expected(1600.0).exp() - 1.0).abs() < 2e-13);
    assert_eq!(rho_alpha(&q, &p, -1599.0, tol).unwrap(), rho);
    assert!(rho_alpha(&p, &q, 1760.0, tol).unwrap().is_infinite());
    let log_rho = log_expected(1760.0);
    let renyi = logp::renyi_divergence(&p, &q, 1760.0, tol).unwrap();
    assert!((renyi - log_rho / 1759.0).abs() < 1e-14);
    let tsallis = logp::tsallis_divergence(&p, &q, 1760.0, tol).unwrap();
    assert!((tsallis / (log_rho - 1759.0_f64.ln()).exp() - 1.0).abs() < 2e-13);
    let amari = amari_alpha_divergence(&p, &q, -3519.0, tol).unwrap();
    assert!((amari / (log_rho - (1759.0_f64 * 1760.0).ln()).exp() - 1.0).abs() < 3e-13);

    // Even log(rho) exceeds MAX here. The finite divergence approaches ln(4).
    let renyi = logp::renyi_divergence(&q, &[0.125, 0.875], f64::MAX, tol).unwrap();
    assert!((renyi - 4.0_f64.ln()).abs() < 1e-14);
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

#[test]
fn weighted_js_retains_positive_mass_below_mixture_precision() {
    use logp::jensen_shannon_weighted;
    let tiny = f64::from_bits(1);
    let identical = [tiny, 1.0];
    let equal = jensen_shannon_divergence(&identical, &identical, 0.0).unwrap();
    assert_eq!(
        jensen_shannon_weighted(&identical, &identical, 0.5, 0.0).unwrap(),
        equal
    );

    let p = [tiny, 0.75, 0.25];
    let q = [0.0, 0.25, 0.75];
    for weight in [0.125, 0.25, 0.5] {
        let expected = jensen_shannon_weighted(&p[1..], &q[1..], weight, 0.0).unwrap();
        let forward = jensen_shannon_weighted(&p, &q, weight, 0.0).unwrap();
        let reverse = jensen_shannon_weighted(&q, &p, 1.0 - weight, 0.0).unwrap();
        assert_eq!(forward, expected);
        assert_eq!(reverse, expected);
    }

    // At the smallest positive weight, the first mixture entry is tiny / 2.
    // The exact result is about 372.37346 units of the smallest subnormal.
    let js = jensen_shannon_weighted(&[0.5, 0.5], &[0.0, 1.0], tiny, 0.0).unwrap();
    assert!((js / tiny - 372.37346).abs() <= 1.0);
}

#[test]
fn sparse_divergences_stay_finite_just_below_order_one() {
    let (p, q) = ([0.5, 0.5], [1.0, 0.0]);
    let alpha: f64 = 1.0 - 5e-13;
    let log_rho = alpha * 0.5_f64.ln();
    let renyi = logp::renyi_divergence(&p, &q, alpha, 1e-12).unwrap();
    let tsallis = logp::tsallis_divergence(&p, &q, alpha, 1e-12).unwrap();
    assert!((renyi / (log_rho / (alpha - 1.0)) - 1.0).abs() < 1e-12);
    assert!((tsallis / (log_rho.exp_m1() / (alpha - 1.0)) - 1.0).abs() < 1e-12);
    for order in [1.0, 1.0 + 5e-13] {
        assert!(logp::renyi_divergence(&p, &q, order, 1e-12).is_err());
        assert!(logp::tsallis_divergence(&p, &q, order, 1e-12).is_err());
    }
}
