//! `mutual_information_ksg` now selects the k nearest neighbours partially
//! and reuses the stored marginal distances for its counts. The reference
//! below is the previous implementation (full stable sort, recomputed
//! distances); results must be bitwise equal, including with tied distances.

use logp::{digamma, mutual_information_ksg, KsgVariant};

fn dist_inf(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(ai, bi)| (ai - bi).abs())
        .fold(0.0, f64::max)
}

fn old_ksg(x: &[Vec<f64>], y: &[Vec<f64>], k: usize, alg2: bool) -> f64 {
    let n = x.len();
    let mut nx = vec![0usize; n];
    let mut ny = vec![0usize; n];
    for i in 0..n {
        let mut neighbours = Vec::with_capacity(n);
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = dist_inf(&x[i], &x[j]);
            let dy = dist_inf(&y[i], &y[j]);
            neighbours.push((dx.max(dy), dx, dy));
        }
        neighbours.sort_by(|a, b| a.0.total_cmp(&b.0));
        let eps = neighbours[k - 1].0;
        if !alg2 {
            nx[i] = (0..n)
                .filter(|&j| i != j && dist_inf(&x[i], &x[j]) < eps)
                .count();
            ny[i] = (0..n)
                .filter(|&j| i != j && dist_inf(&y[i], &y[j]) < eps)
                .count();
        } else {
            let (ex, ey) = neighbours[..k]
                .iter()
                .fold((0.0_f64, 0.0_f64), |(ex, ey), &(_, dx, dy)| {
                    (ex.max(dx), ey.max(dy))
                });
            nx[i] = (0..n)
                .filter(|&j| i != j && dist_inf(&x[i], &x[j]) <= ex)
                .count();
            ny[i] = (0..n)
                .filter(|&j| i != j && dist_inf(&y[i], &y[j]) <= ey)
                .count();
        }
    }
    let shift = if alg2 { 0.0 } else { 1.0 };
    let avg: f64 = nx
        .iter()
        .zip(&ny)
        .map(|(&a, &b)| digamma(a as f64 + shift) + digamma(b as f64 + shift))
        .sum::<f64>()
        / n as f64;
    let correction = if alg2 { 1.0 / k as f64 } else { 0.0 };
    digamma(k as f64) - correction - avg + digamma(n as f64)
}

fn samples(seed: &mut u64, n: usize, d: usize, levels: Option<u64>) -> Vec<Vec<f64>> {
    let mut next = || {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    };
    (0..n)
        .map(|_| {
            (0..d)
                .map(|_| match levels {
                    // Few distinct values force many tied distances.
                    Some(l) => (next() % l) as f64,
                    None => (next() >> 11) as f64 / (1u64 << 53) as f64,
                })
                .collect()
        })
        .collect()
}

#[test]
fn matches_the_previous_implementation_bitwise() {
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    for &(n, dx, dy) in &[(12, 1, 1), (40, 2, 1), (90, 3, 2)] {
        for levels in [None, Some(3), Some(5)] {
            let x = samples(&mut seed, n, dx, levels);
            // Correlate y with x so the estimate is not trivially near zero.
            let mut y = samples(&mut seed, n, dy, levels);
            for (yi, xi) in y.iter_mut().zip(&x) {
                yi[0] += xi[0];
            }
            for k in [1, 3, 5] {
                for (variant, alg2) in [(KsgVariant::Alg1, false), (KsgVariant::Alg2, true)] {
                    let new = mutual_information_ksg(&x, &y, k, variant).unwrap();
                    let old = old_ksg(&x, &y, k, alg2);
                    assert_eq!(
                        new.to_bits(),
                        old.to_bits(),
                        "n={n} k={k} levels={levels:?} alg2={alg2}: {new} vs {old}"
                    );
                }
            }
        }
    }
}

/// Timing only: `cargo test --release --test ksg_equivalence -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing_against_the_previous_implementation() {
    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    for n in [500, 1500, 3000] {
        let x = samples(&mut seed, n, 2, None);
        let y = samples(&mut seed, n, 2, None);
        let best = |f: &dyn Fn() -> f64| {
            (0..5)
                .map(|_| {
                    let t = std::time::Instant::now();
                    std::hint::black_box(f());
                    t.elapsed().as_secs_f64()
                })
                .fold(f64::INFINITY, f64::min)
        };
        let new = best(&|| mutual_information_ksg(&x, &y, 3, KsgVariant::Alg2).unwrap());
        let old = best(&|| old_ksg(&x, &y, 3, true));
        eprintln!(
            "n={n}: old {:.1} ms, new {:.1} ms, {:.2}x",
            old * 1e3,
            new * 1e3,
            old / new
        );
    }
}
