//! Rosetta correctness fixtures: logp information-theory primitives asserted
//! against scipy and scikit-learn.
//!
//! Reference values in `fixtures/rosetta/logp_infotheory.json` come from
//! `gen_logp.py` (their provenance). logp works in nats (natural log), so every
//! reference uses base e. Library oracles: entropy_nats (scipy.stats.entropy),
//! kl_divergence (scipy.stats.entropy(p, q)), jensen_shannon_divergence
//! (scipy.spatial.distance.jensenshannon squared), mutual_information
//! (sklearn.metrics.mutual_info_score over a contingency table). The remaining
//! checks (cross_entropy, bhattacharyya, hellinger, renyi, tsallis) have no
//! scipy/sklearn function, so their reference is the canonical formula computed
//! in numpy: a cross-implementation check, not a library oracle. The Bregman,
//! Csiszar f-divergence, and alpha families are deferred (no clean oracle).
//!
//! Regenerate the fixture: `uv run tests/fixtures/rosetta/gen_logp.py`.

use serde::Deserialize;

const FIXTURE: &str = include_str!("fixtures/rosetta/logp_infotheory.json");
const TOL: f64 = 1e-9;

#[derive(Deserialize)]
struct Fixture {
    p: Vec<f64>,
    q: Vec<f64>,
    p_xy: Vec<f64>,
    n_x: usize,
    n_y: usize,
    alpha: f64,
    expected: Expected,
}

#[derive(Deserialize)]
struct Expected {
    entropy_p: f64,
    kl_pq: f64,
    js_pq: f64,
    mutual_information: f64,
    cross_entropy_pq: f64,
    bhattacharyya_pq: f64,
    hellinger_squared_pq: f64,
    renyi_p: f64,
    tsallis_p: f64,
}

fn close(got: f64, want: f64, label: &str) {
    let tol = 1e-9 * (1.0 + want.abs());
    let diff = (got - want).abs();
    assert!(
        diff <= tol,
        "{label}: logp={got} ref={want} diff={diff} tol={tol}"
    );
}

#[test]
fn rosetta_infotheory_matches_scipy_sklearn() {
    let fx: Fixture = serde_json::from_str(FIXTURE).expect("parse rosetta fixture");
    let p = &fx.p;
    let q = &fx.q;
    let e = &fx.expected;

    // Library oracles (scipy / sklearn).
    close(
        logp::entropy_nats(p, TOL).unwrap(),
        e.entropy_p,
        "entropy_p",
    );
    close(logp::kl_divergence(p, q, TOL).unwrap(), e.kl_pq, "kl_pq");
    close(
        logp::jensen_shannon_divergence(p, q, TOL).unwrap(),
        e.js_pq,
        "js_pq",
    );
    close(
        logp::mutual_information(&fx.p_xy, fx.n_x, fx.n_y, TOL).unwrap(),
        e.mutual_information,
        "mutual_information",
    );

    // Cross-implementation checks (numpy formula).
    close(
        logp::cross_entropy_nats(p, q, TOL).unwrap(),
        e.cross_entropy_pq,
        "cross_entropy_pq",
    );
    close(
        logp::bhattacharyya_coeff(p, q, TOL).unwrap(),
        e.bhattacharyya_pq,
        "bhattacharyya_pq",
    );
    close(
        logp::hellinger_squared(p, q, TOL).unwrap(),
        e.hellinger_squared_pq,
        "hellinger_squared_pq",
    );
    close(
        logp::renyi_entropy(p, fx.alpha, TOL).unwrap(),
        e.renyi_p,
        "renyi_p",
    );
    close(
        logp::tsallis_entropy(p, fx.alpha, TOL).unwrap(),
        e.tsallis_p,
        "tsallis_p",
    );
}
