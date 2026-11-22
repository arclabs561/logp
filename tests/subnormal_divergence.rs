use logp::{jensen_shannon_divergence, kl_divergence};

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
