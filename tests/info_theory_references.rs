//! Exact information-theoretic references for logp.
//!
//! Deterministic closed forms (uniform entropy = ln n, discrete MI, digamma at
//! ψ(1)=-γ etc., Gaussian KL, divergence invariants) plus a seeded statistical
//! check: the KSG estimator recovers the analytic mutual information of a
//! correlated bivariate Gaussian, I(X;Y) = -½ln(1-ρ²). All RNG is a fixed-seed
//! Box-Muller stream so the statistical check is reproducible.

use logp::{
    digamma, entropy_bits, entropy_nats, jensen_shannon_divergence, kl_divergence,
    kl_divergence_gaussians, mutual_information, mutual_information_ksg, total_variation,
    KsgVariant,
};

const GAMMA: f64 = 0.577_215_664_901_532_9;
const LN2: f64 = std::f64::consts::LN_2;

/// Fixed-seed standard-normal stream (Box-Muller over xorshift).
struct Normals {
    s: u64,
}
impl Normals {
    fn new(seed: u64) -> Self {
        Self { s: seed }
    }
    fn u(&mut self) -> f64 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 7;
        self.s ^= self.s << 17;
        ((self.s >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 2.0)
    }
    fn normal(&mut self) -> f64 {
        let (u1, u2) = (self.u(), self.u());
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

#[test]
fn ksg_recovers_correlated_gaussian_mi() {
    let mut rng = Normals::new(0x243F6A8885A308D3);
    let n = 2000;
    for rho in [0.0_f64, 0.5, 0.8] {
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        for _ in 0..n {
            let x = rng.normal();
            let z = rng.normal();
            let y = rho * x + (1.0 - rho * rho).sqrt() * z;
            xs.push(vec![x]);
            ys.push(vec![y]);
        }
        let true_mi = -0.5 * (1.0 - rho * rho).ln();
        let est = mutual_information_ksg(&xs, &ys, 5, KsgVariant::Alg1).expect("ksg");
        assert!(
            (est - true_mi).abs() < 0.08,
            "KSG MI ρ={rho}: est {est:.4} vs true {true_mi:.4} (|Δ|≥0.08)"
        );
    }
}

#[test]
fn uniform_entropy_closed_form() {
    for n in [2usize, 4, 8, 10] {
        let p = vec![1.0 / n as f64; n];
        let h_nats = entropy_nats(&p, 1e-9).expect("entropy");
        let h_bits = entropy_bits(&p, 1e-9).expect("entropy_bits");
        assert!(
            (h_nats - (n as f64).ln()).abs() < 1e-9,
            "uniform({n}) {h_nats} != ln(n)"
        );
        assert!(
            (h_bits - (n as f64).log2()).abs() < 1e-9,
            "uniform({n}) {h_bits} != log2(n)"
        );
    }
}

#[test]
fn discrete_mi_independent_and_diagonal() {
    let (nx, ny) = (4usize, 4usize);
    let indep = vec![1.0 / (nx * ny) as f64; nx * ny];
    let mi = mutual_information(&indep, nx, ny, 1e-9).expect("mi");
    assert!(mi.abs() < 1e-9, "independent MI {mi} != 0");

    let mut diag = vec![0.0; nx * ny];
    for i in 0..nx {
        diag[i * ny + i] = 1.0 / nx as f64;
    }
    let mi = mutual_information(&diag, nx, ny, 1e-9).expect("mi");
    assert!(
        (mi - (nx as f64).ln()).abs() < 1e-9,
        "diagonal MI {mi} != ln(n)"
    );
}

#[test]
fn digamma_known_points() {
    for (x, want) in [
        (1.0, -GAMMA),
        (2.0, 1.0 - GAMMA),
        (0.5, -GAMMA - 2.0 * LN2),
        (3.0, 1.5 - GAMMA),
    ] {
        let got = digamma(x);
        assert!(
            (got - want).abs() < 1e-6,
            "digamma({x}) = {got:.8}, want {want:.8}"
        );
    }
}

#[test]
fn gaussian_kl_closed_form() {
    let kl_self = kl_divergence_gaussians(&[0.0], &[1.0], &[0.0], &[1.0]).expect("klg");
    assert!(
        kl_self.abs() < 1e-12,
        "Gaussian KL(p||p) {kl_self:.2e} != 0"
    );

    let (m1, s1, m2, s2) = (1.0_f64, 2.0_f64, -1.0_f64, 3.0_f64);
    let want = (s2 / s1).ln() + (s1 * s1 + (m1 - m2).powi(2)) / (2.0 * s2 * s2) - 0.5;
    let got = kl_divergence_gaussians(&[m1], &[s1], &[m2], &[s2]).expect("klg");
    assert!(
        (got - want).abs() < 1e-9,
        "Gaussian KL {got:.6} != {want:.6}"
    );
}

#[test]
fn divergence_invariants() {
    let mut rs = 0xCAFEBABEu64;
    let mut nextp = |k: usize| -> Vec<f64> {
        let mut v = Vec::with_capacity(k);
        let mut sum = 0.0;
        for _ in 0..k {
            rs ^= rs << 13;
            rs ^= rs >> 7;
            rs ^= rs << 17;
            let x = (rs >> 11) as f64 / (1u64 << 53) as f64 + 1e-3;
            v.push(x);
            sum += x;
        }
        for e in &mut v {
            *e /= sum;
        }
        v
    };
    for _ in 0..200 {
        let p = nextp(6);
        let q = nextp(6);
        let kl = kl_divergence(&p, &q, 1e-9).expect("kl");
        assert!(kl >= -1e-12, "KL negative: {kl}");
        let js = jensen_shannon_divergence(&p, &q, 1e-9).expect("js");
        assert!(
            (-1e-12..=LN2 + 1e-9).contains(&js),
            "JS {js} out of [0,ln2]"
        );
        let js_rev = jensen_shannon_divergence(&q, &p, 1e-9).expect("js");
        assert!(
            (js - js_rev).abs() < 1e-12,
            "JS asymmetric: {js} vs {js_rev}"
        );
        let tv = total_variation(&p, &q, 1e-9).expect("tv");
        let l1: f64 = p.iter().zip(&q).map(|(a, b)| (a - b).abs()).sum();
        assert!((tv - 0.5 * l1).abs() < 1e-9, "TV {tv} != ½·L1 {}", 0.5 * l1);
    }
}
