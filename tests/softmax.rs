//! `softmax` replaces private max-shifted copies in fynch, hopfield and rankit.
//! It must equal that formula bitwise so those crates can delegate without a
//! change in output.

use logp::{log_softmax, softmax, softmax_in_place};

// The formula as fynch's `softmax` and hopfield's `softmax` write it.
fn copied(x: &[f64]) -> Vec<f64> {
    if x.is_empty() {
        return Vec::new();
    }
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn values(seed: &mut u64, n: usize, scale: f64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            *seed ^= *seed << 13;
            *seed ^= *seed >> 7;
            *seed ^= *seed << 17;
            ((*seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * scale
        })
        .collect()
}

#[test]
fn equals_the_copied_formula_bitwise() {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    for n in [0, 1, 2, 5, 64, 1000] {
        for scale in [1e-3, 1.0, 40.0, 2000.0] {
            let x = values(&mut seed, n, scale);
            let got: Vec<u64> = softmax(&x).iter().map(|v| v.to_bits()).collect();
            let want: Vec<u64> = copied(&x).iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "n={n} scale={scale}");
            let mut in_place = x.clone();
            softmax_in_place(&mut in_place);
            assert_eq!(in_place, softmax(&x));
        }
    }
}

#[test]
fn is_a_distribution_and_shift_invariant() {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..200 {
        let x = values(&mut seed, 17, 30.0);
        let p = softmax(&x);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p.iter().all(|&v| (0.0..=1.0).contains(&v)));
        let shifted: Vec<f64> = x.iter().map(|v| v + 500.0).collect();
        for (a, b) in p.iter().zip(softmax(&shifted)) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
fn log_softmax_is_the_log_of_softmax_and_stays_finite() {
    let mut seed = 0x0dd_b1a5e_u64;
    for _ in 0..200 {
        let x = values(&mut seed, 9, 20.0);
        for (l, p) in log_softmax(&x).iter().zip(softmax(&x)) {
            assert!((l - p.ln()).abs() < 1e-10);
        }
    }
    let lp = log_softmax(&[0.0, 800.0]);
    assert_eq!(softmax(&[0.0, 800.0])[0], 0.0);
    assert!((lp[0] + 800.0).abs() < 1e-9 && lp[1].abs() < 1e-12);
    assert!(log_softmax(&[]).is_empty());
}
