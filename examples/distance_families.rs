//! Cross-crate comparison of distance families on discrete distributions.
//!
//! Three families: f-divergences (logp), optimal transport (wass), kernel MMD (rkhs).
//! Each imposes different geometry; this example shows where they agree and disagree
//! on ranking distribution pairs.

use ndarray::{Array1, Array2};

/// Build a 4x4 cost matrix C[i][j] = |i - j| (L1 ground metric on bin indices).
fn l1_cost_4() -> Array2<f32> {
    let mut c = Array2::<f32>::zeros((4, 4));
    for i in 0..4 {
        for j in 0..4 {
            c[[i, j]] = (i as f32 - j as f32).abs();
        }
    }
    c
}

fn to_f32(p: &[f64]) -> Array1<f32> {
    Array1::from_vec(p.iter().map(|&v| v as f32).collect())
}

/// Discrete distribution -> weighted point samples at bin positions for MMD.
fn to_samples(p: &[f64], n: usize) -> Vec<Vec<f64>> {
    let mut s = Vec::with_capacity(n);
    for (i, &pi) in p.iter().enumerate() {
        for _ in 0..(pi * n as f64).round() as usize {
            s.push(vec![i as f64]);
        }
    }
    while s.len() < n {
        s.push(vec![0.0]);
    }
    s.truncate(n);
    s
}

fn main() {
    let dists: [(&str, [f64; 4]); 4] = [
        ("uniform", [0.25, 0.25, 0.25, 0.25]),
        ("skewed", [0.7, 0.1, 0.1, 0.1]),
        ("bimodal", [0.4, 0.1, 0.1, 0.4]),
        ("peaked", [0.05, 0.9, 0.025, 0.025]),
    ];

    let cost = l1_cost_4();
    let tol = 1e-9;
    let (reg, sink_iter, sink_tol) = (0.05_f32, 200, 1e-6_f32);
    let (n_samp, sigma) = (100, 1.0);

    println!(
        "{:<18} {:>8} {:>8} {:>10} {:>10} {:>8}",
        "pair", "KL(p||q)", "JS", "Hellinger", "Sinkhorn", "MMD^2"
    );
    println!("{:-<68}", "");

    let mut rows: Vec<(String, [f64; 5])> = Vec::new();

    for i in 0..dists.len() {
        for j in (i + 1)..dists.len() {
            let (np, p) = &dists[i];
            let (nq, q) = &dists[j];
            let label = format!("{np}-{nq}");

            let kl = logp::kl_divergence(p, q, tol).unwrap_or(f64::NAN);
            let js = logp::jensen_shannon_divergence(p, q, tol).unwrap_or(f64::NAN);
            let hell = logp::hellinger(p, q, tol).unwrap_or(f64::NAN);

            let sink = wass::sinkhorn_divergence_same_support(
                &to_f32(p),
                &to_f32(q),
                &cost,
                reg,
                sink_iter,
                sink_tol,
            )
            .unwrap_or(f32::NAN) as f64;

            let mmd2 = rkhs::mmd_biased(&to_samples(p, n_samp), &to_samples(q, n_samp), |a, b| {
                rkhs::rbf(a, b, sigma)
            });

            println!(
                "{:<18} {:>8.4} {:>8.4} {:>10.4} {:>10.4} {:>8.4}",
                label, kl, js, hell, sink, mmd2
            );
            rows.push((label, [kl, js, hell, sink, mmd2]));
        }
    }

    // Rank each metric (1 = most distant pair) and print ranking table.
    let names = ["KL(p||q)", "JS", "Hellinger", "Sinkhorn", "MMD^2"];
    println!("\n{:-<68}", "");
    println!("Rankings (1 = most distant):\n");
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "pair", names[0], names[1], names[2], names[3], names[4]
    );
    println!("{:-<68}", "");

    for col in 0..5 {
        let mut ord: Vec<(usize, f64)> = rows
            .iter()
            .enumerate()
            .map(|(i, r)| (i, r.1[col]))
            .collect();
        ord.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (rank, &(idx, _)) in ord.iter().enumerate() {
            rows[idx].1[col] = (rank + 1) as f64;
        }
    }
    for (label, r) in &rows {
        println!(
            "{:<18} {:>10} {:>10} {:>10} {:>10} {:>10}",
            label, r[0] as u32, r[1] as u32, r[2] as u32, r[3] as u32, r[4] as u32
        );
    }

    let unanimous: Vec<&str> = rows
        .iter()
        .filter(|(_, r)| r.iter().all(|&v| v == 1.0))
        .map(|(l, _)| l.as_str())
        .collect();
    if unanimous.is_empty() {
        println!("\nMetrics disagree on the most distant pair.");
    } else {
        println!(
            "\nAll metrics agree on most distant pair: {}",
            unanimous.join(", ")
        );
    }
}
