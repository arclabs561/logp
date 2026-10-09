//! The crate docs promise that validated functions report bad input through
//! `Err` and never panic. These properties feed arbitrary `f64` values
//! (NaN, infinities, subnormals, negatives, huge magnitudes) and arbitrary
//! shape parameters into every validated entry point and only require that
//! the call returns.

use logp::*;
use proptest::prelude::*;

fn any_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        Just(f64::NAN),
        Just(f64::INFINITY),
        Just(f64::NEG_INFINITY),
        Just(0.0),
        Just(-0.0),
        Just(f64::MIN_POSITIVE),
        Just(f64::MAX),
        -2.0..2.0f64,
        any::<f64>(),
    ]
}

fn any_vec() -> impl Strategy<Value = Vec<f64>> {
    prop::collection::vec(any_f64(), 0..8)
}

fn any_shape() -> impl Strategy<Value = usize> {
    prop_oneof![0usize..6, Just(usize::MAX), Just(usize::MAX / 2 + 1)]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn single_distribution_functions_never_panic(p in any_vec(), tol in any_f64(), alpha in any_f64()) {
        let _ = validate_simplex(&p, tol);
        let _ = entropy_nats(&p, tol);
        let _ = entropy_bits(&p, tol);
        let _ = renyi_entropy(&p, alpha, tol);
        let _ = tsallis_entropy(&p, alpha, tol);
        let mut q = p.clone();
        let _ = normalize_in_place(&mut q);
    }

    #[test]
    fn two_distribution_functions_never_panic(
        p in any_vec(),
        q in any_vec(),
        tol in any_f64(),
        alpha in any_f64(),
    ) {
        let _ = cross_entropy_nats(&p, &q, tol);
        let _ = kl_divergence(&p, &q, tol);
        let _ = jensen_shannon_divergence(&p, &q, tol);
        let _ = jensen_shannon_weighted(&p, &q, alpha, tol);
        let _ = bhattacharyya_coeff(&p, &q, tol);
        let _ = bhattacharyya_distance(&p, &q, tol);
        let _ = hellinger_squared(&p, &q, tol);
        let _ = hellinger(&p, &q, tol);
        let _ = rho_alpha(&p, &q, alpha, tol);
        let _ = renyi_divergence(&p, &q, alpha, tol);
        let _ = tsallis_divergence(&p, &q, alpha, tol);
        let _ = amari_alpha_divergence(&p, &q, alpha, tol);
        let _ = total_variation(&p, &q, tol);
        let _ = chi_squared_divergence(&p, &q, tol);
        let _ = csiszar_f_divergence(&p, &q, |t| t * t.ln(), tol);
        let _ = bregman_divergence(&SquaredL2, &p, &q);
        let _ = total_bregman_divergence(&SquaredL2, &p, &q);
    }

    #[test]
    fn joint_distribution_functions_never_panic(
        p in any_vec(),
        n_x in any_shape(),
        n_y in any_shape(),
        tol in any_f64(),
    ) {
        let _ = mutual_information(&p, n_x, n_y, tol);
        let _ = conditional_entropy(&p, n_x, n_y, tol);
        let _ = normalized_mutual_information(&p, n_x, n_y, tol);
    }

    #[test]
    fn scalar_functions_never_panic(a in any_f64(), b in any_f64(), c in any_f64()) {
        let _ = pmi(a, b, c);
        let _ = log_sum_exp(&[a, b, c]);
        let _ = log_sum_exp2(a, b);
        let _ = digamma(a);
        let _ = distprop::Gaussian::new(a, b);
        let _ = distprop::Gaussian::point(a);
    }

    #[test]
    fn gaussian_kl_never_panics(
        mu1 in any_vec(),
        s1 in any_vec(),
        mu2 in any_vec(),
        s2 in any_vec(),
    ) {
        let _ = kl_divergence_gaussians(&mu1, &s1, &mu2, &s2);
    }

    #[test]
    fn ksg_never_panics(
        x in prop::collection::vec(prop::collection::vec(any_f64(), 0..3), 0..8),
        y in prop::collection::vec(prop::collection::vec(any_f64(), 0..3), 0..8),
        k in 0usize..10,
    ) {
        let _ = mutual_information_ksg(&x, &y, k, KsgVariant::Alg1);
        let _ = mutual_information_ksg(&x, &y, k, KsgVariant::Alg2);
    }
}
