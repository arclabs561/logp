//! Timing harness for the hot primitives, in the shapes downstream crates use.
//!
//! No external benchmark framework: each case runs a calibrated batch of calls
//! per sample and reports the minimum and median ns/call over `SAMPLES` samples.
//!
//! ```text
//! cargo bench --bench primitives            # all cases
//! cargo bench --bench primitives -- js      # cases whose name contains "js"
//! ```
//!
//! Output is tab-separated: `case  n  min_ns  median_ns`.

use std::hint::black_box;
use std::time::{Duration, Instant};

use logp::{
    bregman_divergence, cross_entropy_nats, entropy_nats, hellinger, jensen_shannon_divergence,
    kl_divergence, log_sum_exp, validate_simplex, NegEntropy, SquaredL2,
};

const SAMPLES: usize = 15;
const SAMPLE_TARGET: Duration = Duration::from_millis(4);
const TOL: f64 = 1e-9;

/// Deterministic xorshift64* generator (no `rand` dependency).
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let x = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A categorical distribution of length `n`; about `zero_frac` of entries are 0.
fn simplex(rng: &mut Rng, n: usize, zero_frac: f64) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            let u = rng.next_f64();
            if rng.next_f64() < zero_frac {
                0.0
            } else {
                u + 1e-3
            }
        })
        .collect();
    if v.iter().all(|&x| x == 0.0) {
        v[0] = 1.0;
    }
    let s: f64 = v.iter().sum();
    for x in &mut v {
        *x /= s;
    }
    v
}

fn time_case(name: &str, n: usize, filter: Option<&str>, mut f: impl FnMut()) {
    if let Some(pat) = filter {
        if !name.contains(pat) {
            return;
        }
    }
    // Calibrate the batch size so one sample takes about SAMPLE_TARGET.
    let mut iters: u64 = 1;
    loop {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        let el = t.elapsed();
        if el >= SAMPLE_TARGET / 4 || iters >= 1 << 30 {
            let per = el.as_secs_f64() / iters as f64;
            iters = ((SAMPLE_TARGET.as_secs_f64() / per.max(1e-12)) as u64).max(1);
            break;
        }
        iters *= 2;
    }
    let mut per_call: Vec<f64> = (0..SAMPLES)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_secs_f64() * 1e9 / iters as f64
        })
        .collect();
    per_call.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "{name}\t{n}\t{:.2}\t{:.2}",
        per_call[0],
        per_call[SAMPLES / 2]
    );
}

fn main() {
    // `cargo bench` passes `--bench`; the first other argument is a name filter.
    let filter = std::env::args().skip(1).find(|a| !a.starts_with("--"));
    let filter = filter.as_deref();
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    for &n in &[2usize, 10, 100, 1000] {
        let p = simplex(&mut rng, n, 0.0);
        let q = simplex(&mut rng, n, 0.0);
        // Drift-monitor shape: sparse histograms with some empty bins.
        let ps = simplex(&mut rng, n, 0.2);
        let qs = simplex(&mut rng, n, 0.2);

        time_case("validate_simplex", n, filter, || {
            black_box(validate_simplex(black_box(&p), TOL)).unwrap();
        });
        time_case("entropy_nats", n, filter, || {
            black_box(entropy_nats(black_box(&ps), TOL).unwrap());
        });
        time_case("kl_divergence", n, filter, || {
            black_box(kl_divergence(black_box(&p), black_box(&q), TOL).unwrap());
        });
        time_case("cross_entropy_nats", n, filter, || {
            black_box(cross_entropy_nats(black_box(&p), black_box(&q), TOL).unwrap());
        });
        time_case("js_dense", n, filter, || {
            black_box(jensen_shannon_divergence(black_box(&p), black_box(&q), TOL).unwrap());
        });
        time_case("js_sparse", n, filter, || {
            black_box(jensen_shannon_divergence(black_box(&ps), black_box(&qs), TOL).unwrap());
        });
        time_case("hellinger", n, filter, || {
            black_box(hellinger(black_box(&ps), black_box(&qs), TOL).unwrap());
        });
    }

    for &n in &[2usize, 10, 100, 1000, 10000] {
        let xs: Vec<f64> = (0..n).map(|_| rng.next_f64() * 40.0 - 20.0).collect();
        time_case("log_sum_exp", n, filter, || {
            black_box(log_sum_exp(black_box(&xs)));
        });
    }

    for &n in &[2usize, 16, 128, 1024] {
        let x: Vec<f64> = (0..n).map(|_| rng.next_f64() + 0.01).collect();
        let c: Vec<f64> = (0..n).map(|_| rng.next_f64() + 0.01).collect();
        time_case("bregman_sql2", n, filter, || {
            black_box(bregman_divergence(&SquaredL2, black_box(&x), black_box(&c)).unwrap());
        });
        time_case("bregman_negent", n, filter, || {
            black_box(bregman_divergence(&NegEntropy, black_box(&x), black_box(&c)).unwrap());
        });
    }
}
