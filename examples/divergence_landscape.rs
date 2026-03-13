//! Compare divergence families on a simple 1D sweep.
//!
//! Fixes q = [0.5, 0.5] and sweeps p = [t, 1-t] for t in (0, 1).
//! Prints a table of KL, JS, Hellinger, Bhattacharyya, Renyi(0.5), and Tsallis(2)
//! so you can see how each divergence responds to the same distributional shift.
//!
//! Also includes a Renyi alpha sweep showing continuity at alpha = 1, where
//! renyi_divergence falls back to KL divergence (a 0.2.0 feature).
//!
//! Run: cargo run --example divergence_landscape

use logp::{
    bhattacharyya_distance, hellinger, jensen_shannon_divergence, kl_divergence, renyi_divergence,
    tsallis_divergence,
};

fn main() {
    let q = [0.5, 0.5];
    let tol = 1e-12;
    let steps = 19;

    println!(
        "{:<6} {:>8} {:>8} {:>8} {:>8} {:>10} {:>10}",
        "t", "KL", "JS", "Hell", "Bhatt", "Renyi0.5", "Tsallis2"
    );
    println!("{}", "-".repeat(70));

    for i in 1..=steps {
        let t = i as f64 / (steps + 1) as f64;
        let p = [t, 1.0 - t];

        let kl = kl_divergence(&p, &q, tol).unwrap();
        let js = jensen_shannon_divergence(&p, &q, tol).unwrap();
        let hel = hellinger(&p, &q, tol).unwrap();
        let bhat = bhattacharyya_distance(&p, &q, tol).unwrap();
        let ren = renyi_divergence(&p, &q, 0.5, tol).unwrap();
        let tsa = tsallis_divergence(&p, &q, 2.0, tol).unwrap();

        println!(
            "{:<6.3} {:>8.5} {:>8.5} {:>8.5} {:>8.5} {:>10.5} {:>10.5}",
            t, kl, js, hel, bhat, ren, tsa
        );
    }

    println!();
    println!("Observations:");
    println!("  - KL diverges as p -> delta; JS stays bounded by ln(2) ~ 0.693");
    println!("  - Hellinger saturates at 1.0; Bhattacharyya diverges");
    println!("  - Renyi(alpha<1) is gentler than KL; Tsallis(alpha>1) is sharper near extremes");

    // --- Renyi alpha sweep: continuity at alpha = 1 (0.2.0 feature) ----------
    //
    // In 0.2.0, renyi_divergence handles alpha = 1 by falling back to KL,
    // so the function is continuous across the full alpha range.
    println!();
    println!("Renyi divergence alpha sweep (p = [0.7, 0.3], q = [0.5, 0.5]):");
    println!("{:<8} {:>12}", "alpha", "Renyi_alpha");
    println!("{}", "-".repeat(22));

    let p_sweep = [0.7, 0.3];
    let q_sweep = [0.5, 0.5];
    let alphas = [
        0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0, 1.01, 1.1, 1.5, 2.0, 5.0,
    ];

    for &a in &alphas {
        let r = renyi_divergence(&p_sweep, &q_sweep, a, tol).unwrap();
        println!("{:<8.2} {:>12.6}", a, r);
    }

    println!();
    println!("The column is continuous through alpha = 1.0 (where Renyi = KL).");
    println!("Tsallis divergence has the same alpha = 1 continuity.");
}
