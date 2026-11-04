//! Quantities bounded in exact arithmetic must stay in range after rounding.

use logp::{bhattacharyya_coeff, bhattacharyya_distance, renyi_divergence};

// For this distribution the coefficient of p with itself sums to
// 1.0000000000000002 before clamping.
const P: [f64; 6] = [
    0.010397885443187215,
    0.3022611833901705,
    0.022568907883676773,
    0.2780781927108418,
    0.1451126177032422,
    0.2415812128688816,
];

#[test]
fn coefficient_of_a_distribution_with_itself_is_at_most_one() {
    assert_eq!(bhattacharyya_coeff(&P, &P, 1e-9).unwrap(), 1.0);
}

#[test]
fn self_distances_are_positive_zero() {
    let d = bhattacharyya_distance(&P, &P, 1e-9).unwrap();
    assert_eq!(d.to_bits(), 0.0f64.to_bits(), "d={d:e}");
    // Renyi of p with itself may still round to a tiny positive value; it must
    // not be negative or -0.0.
    for alpha in [0.5, 2.0] {
        let r = renyi_divergence(&P, &P, alpha, 1e-9).unwrap();
        assert!(r.is_sign_positive() && r < 1e-12, "alpha={alpha} r={r:e}");
    }
}

#[test]
fn random_pairs_stay_in_range() {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for _ in 0..5_000 {
        let n = 2 + (next() * 30.0) as usize;
        let mut p: Vec<f64> = (0..n).map(|_| next()).collect();
        let mut q: Vec<f64> = (0..n).map(|_| next()).collect();
        let (sp, sq): (f64, f64) = (p.iter().sum(), q.iter().sum());
        p.iter_mut().for_each(|x| *x /= sp);
        q.iter_mut().for_each(|x| *x /= sq);
        for (a, b) in [(&p, &q), (&p, &p)] {
            let bc = bhattacharyya_coeff(a, b, 1e-9).unwrap();
            assert!((0.0..=1.0).contains(&bc), "bc={bc:e}");
            let d = bhattacharyya_distance(a, b, 1e-9).unwrap();
            assert!(d.is_sign_positive(), "d={d:e}");
            for alpha in [0.5, 2.0] {
                let r = renyi_divergence(a, b, alpha, 1e-9).unwrap();
                assert!(r.is_sign_positive(), "alpha={alpha} r={r:e}");
            }
        }
    }
}
