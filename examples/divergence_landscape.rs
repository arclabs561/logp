//! Compare divergence families on a simple 1D sweep.
//!
//! Fixes q = [0.5, 0.5] and sweeps p = [t, 1-t] for t in (0, 1).
//! Prints a table of KL, JS, Hellinger, Bhattacharyya, Renyi(0.5), and Tsallis(2)
//! so you can see how each divergence responds to the same distributional shift.
//!
//! Run: cargo run --example divergence_landscape

use logp::{
    bhattacharyya_distance, hellinger, jensen_shannon_divergence, kl_divergence,
    renyi_divergence, tsallis_divergence,
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
}
