//! Differential tests for the single-pass (fused validation) implementations.
//!
//! `old` holds verbatim copies of the pre-0.2.5 implementations, which made
//! separate validation passes (and, for Jensen-Shannon, allocated `m` and
//! called `kl_divergence` twice). Each public function must agree with its old
//! copy on every input: `Ok` values bit for bit, and errors with identical
//! `Debug` output (variant, index, offending value, reported sum).

use logp::{BregmanGenerator, Error, NegEntropy, Result, SquaredL2};

mod old {
    use super::*;

    fn ensure_nonempty(x: &[f64]) -> Result<()> {
        if x.is_empty() {
            return Err(Error::Empty);
        }
        Ok(())
    }

    fn ensure_same_len(a: &[f64], b: &[f64]) -> Result<()> {
        if a.len() != b.len() {
            return Err(Error::LengthMismatch(a.len(), b.len()));
        }
        Ok(())
    }

    fn ensure_nonnegative(x: &[f64]) -> Result<()> {
        for (i, &v) in x.iter().enumerate() {
            if !v.is_finite() {
                return Err(Error::NonFinite { idx: i, value: v });
            }
            if v < 0.0 {
                return Err(Error::Negative { idx: i, value: v });
            }
        }
        Ok(())
    }

    fn sum(x: &[f64]) -> f64 {
        x.iter().sum()
    }

    pub fn validate_simplex(p: &[f64], tol: f64) -> Result<()> {
        ensure_nonempty(p)?;
        ensure_nonnegative(p)?;
        let s = sum(p);
        if (s - 1.0).abs() > tol {
            return Err(Error::NotNormalized { sum: s });
        }
        Ok(())
    }

    pub fn entropy_nats(p: &[f64], tol: f64) -> Result<f64> {
        validate_simplex(p, tol)?;
        let mut h = 0.0;
        for &pi in p {
            if pi > 0.0 {
                h -= pi * pi.ln();
            }
        }
        Ok(h)
    }

    pub fn cross_entropy_nats(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
        ensure_same_len(p, q)?;
        validate_simplex(p, tol)?;
        validate_simplex(q, tol)?;
        let mut h = 0.0;
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi == 0.0 {
                continue;
            }
            if qi <= 0.0 {
                return Err(Error::Domain("cross-entropy undefined: q_i=0 while p_i>0"));
            }
            h -= pi * qi.ln();
        }
        Ok(h)
    }

    pub fn kl_divergence(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
        ensure_same_len(p, q)?;
        validate_simplex(p, tol)?;
        validate_simplex(q, tol)?;
        let mut d = 0.0;
        for (&pi, &qi) in p.iter().zip(q.iter()) {
            if pi == 0.0 {
                continue;
            }
            if qi <= 0.0 {
                return Err(Error::Domain("KL undefined: q_i=0 while p_i>0"));
            }
            d += pi * (pi / qi).ln();
        }
        Ok(d)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn jensen_shannon_divergence(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
        ensure_same_len(p, q)?;
        validate_simplex(p, tol)?;
        validate_simplex(q, tol)?;

        let mut m = vec![0.0; p.len()];
        for i in 0..p.len() {
            m[i] = 0.5 * (p[i] + q[i]);
        }

        Ok(0.5 * kl_divergence(p, &m, tol)? + 0.5 * kl_divergence(q, &m, tol)?)
    }

    pub fn bhattacharyya_coeff(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
        ensure_same_len(p, q)?;
        validate_simplex(p, tol)?;
        validate_simplex(q, tol)?;
        let bc: f64 = p
            .iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| pi.sqrt() * qi.sqrt())
            .sum();
        // The pre-fusion function returned `bc` unclamped. Clamping to 1 is a
        // separate, intended fix (pinned in tests/rounding_bounds.rs), so the
        // reference applies it to keep this a test of the fusion alone.
        Ok(bc.min(1.0))
    }

    pub fn hellinger_squared(p: &[f64], q: &[f64], tol: f64) -> Result<f64> {
        ensure_same_len(p, q)?;
        validate_simplex(p, tol)?;
        validate_simplex(q, tol)?;
        let h2: f64 = p
            .iter()
            .zip(q.iter())
            .map(|(&pi, &qi)| {
                let diff = pi.sqrt() - qi.sqrt();
                diff * diff
            })
            .sum();
        Ok((0.5 * h2).max(0.0))
    }

    #[allow(clippy::needless_range_loop)]
    pub fn bregman_divergence(gen: &impl BregmanGenerator, p: &[f64], q: &[f64]) -> Result<f64> {
        ensure_nonempty(p)?;
        ensure_same_len(p, q)?;
        let mut grad_q = vec![0.0; q.len()];
        gen.grad_into(q, &mut grad_q)?;
        let fp = gen.f(p)?;
        let fq = gen.f(q)?;
        let mut inner = 0.0;
        for i in 0..p.len() {
            inner += (p[i] - q[i]) * grad_q[i];
        }
        Ok(fp - fq - inner)
    }

    #[allow(clippy::needless_range_loop)]
    pub fn total_bregman_divergence(
        gen: &impl BregmanGenerator,
        p: &[f64],
        q: &[f64],
    ) -> Result<f64> {
        ensure_nonempty(p)?;
        ensure_same_len(p, q)?;
        let mut grad_q = vec![0.0; q.len()];
        gen.grad_into(q, &mut grad_q)?;
        let fp = gen.f(p)?;
        let fq = gen.f(q)?;
        let mut inner = 0.0;
        for i in 0..p.len() {
            inner += (p[i] - q[i]) * grad_q[i];
        }
        let b = fp - fq - inner;
        let grad_norm_sq: f64 = grad_q.iter().map(|&x| x * x).sum();
        Ok(b / (1.0 + grad_norm_sq).sqrt())
    }
}

/// `Ok` values must match bit for bit; errors must match in `Debug` form.
fn same(new: Result<f64>, old: Result<f64>) -> bool {
    match (new, old) {
        (Ok(a), Ok(b)) => a.to_bits() == b.to_bits(),
        (Err(a), Err(b)) => format!("{a:?}") == format!("{b:?}"),
        _ => false,
    }
}

fn same_unit(new: Result<()>, old: Result<()>) -> bool {
    same(new.map(|_| 0.0), old.map(|_| 0.0))
}

/// Deterministic xorshift64* generator.
struct Rng(u64);

impl Rng {
    fn u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f64(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: u64) -> u64 {
        self.u64() % n
    }
}

/// An entry that is usually an ordinary probability weight and sometimes an
/// edge value: exact zeros of both signs, subnormals, non-finite, negative,
/// or huge.
fn entry(rng: &mut Rng) -> f64 {
    match rng.below(40) {
        0..=3 => 0.0,
        4 => -0.0,
        5 => 5e-324,
        6 => 1e-310,
        7 if rng.below(8) == 0 => f64::NAN,
        8 if rng.below(8) == 0 => f64::INFINITY,
        9 if rng.below(8) == 0 => f64::NEG_INFINITY,
        10 if rng.below(4) == 0 => -rng.f64(),
        11 if rng.below(8) == 0 => 1e308,
        _ => rng.f64(),
    }
}

fn vector(rng: &mut Rng, n: usize, tol: f64) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|_| entry(rng)).collect();
    // Usually normalize so most cases reach the computation, not just the
    // validation errors.
    let s: f64 = v.iter().sum();
    if rng.below(5) != 0 && s.is_finite() && s > 0.0 {
        for x in &mut v {
            *x /= s;
        }
    }
    // Sometimes push the sum to the edge of `tol` to exercise the comparison.
    if !v.is_empty() && rng.below(6) == 0 && tol.is_finite() {
        let i = rng.below(v.len() as u64) as usize;
        let delta = match rng.below(3) {
            0 => tol,
            1 => -tol,
            _ => tol * (1.0 + f64::EPSILON * 4.0),
        };
        v[i] += delta;
    }
    v
}

fn tol(rng: &mut Rng) -> f64 {
    const TOLS: [f64; 8] = [1e-9, 1e-12, 1e-6, 0.0, 0.5, f64::NAN, f64::INFINITY, 1e300];
    TOLS[rng.below(TOLS.len() as u64) as usize]
}

/// Cases per run. Lengths span 0..=40 so both sides of the validate
/// threshold (16) are covered.
const CASES: usize = 60_000;

fn pairs(seed: u64) -> impl Iterator<Item = (Vec<f64>, Vec<f64>, f64)> {
    let mut rng = Rng(seed);
    (0..CASES).map(move |_| {
        let t = tol(&mut rng);
        let n = rng.below(41) as usize;
        let m = if rng.below(20) == 0 {
            rng.below(41) as usize
        } else {
            n
        };
        let p = vector(&mut rng, n, t);
        // Sometimes reuse p (identical-argument fast cases) or a copy with
        // zeros moved around (support mismatch, the KL domain error).
        let q = match rng.below(8) {
            0 if m == n => p.clone(),
            1 if m == n => {
                let mut q = p.clone();
                q.reverse();
                q
            }
            _ => vector(&mut rng, m, t),
        };
        (p, q, t)
    })
}

#[test]
fn validate_simplex_matches_old() {
    let mut rng = Rng(0xA5A5_1234_5678_9ABC);
    for _ in 0..CASES {
        let t = tol(&mut rng);
        let n = rng.below(41) as usize;
        let p = vector(&mut rng, n, t);
        assert!(
            same_unit(logp::validate_simplex(&p, t), old::validate_simplex(&p, t)),
            "p={p:?} tol={t}"
        );
    }
}

#[test]
fn entropy_nats_matches_old() {
    let mut rng = Rng(0x1357_9BDF_2468_ACE0);
    for _ in 0..CASES {
        let t = tol(&mut rng);
        let n = rng.below(41) as usize;
        let p = vector(&mut rng, n, t);
        assert!(
            same(logp::entropy_nats(&p, t), old::entropy_nats(&p, t)),
            "p={p:?} tol={t}"
        );
    }
}

macro_rules! pairwise_matches_old {
    ($test:ident, $f:ident, $seed:expr) => {
        #[test]
        fn $test() {
            let mut ok_cases = 0usize;
            for (p, q, t) in pairs($seed) {
                let new = logp::$f(&p, &q, t);
                ok_cases += new.is_ok() as usize;
                assert!(same(new, old::$f(&p, &q, t)), "p={p:?} q={q:?} tol={t}");
            }
            // Guard against a generator that only produces errors.
            assert!(ok_cases > CASES / 20, "only {ok_cases} Ok cases");
        }
    };
}

pairwise_matches_old!(
    kl_divergence_matches_old,
    kl_divergence,
    0x0DDB_1A5E_5BAD_5EED
);
pairwise_matches_old!(
    cross_entropy_matches_old,
    cross_entropy_nats,
    0xFEED_FACE_CAFE_BEEF
);
pairwise_matches_old!(
    jensen_shannon_matches_old,
    jensen_shannon_divergence,
    0x5EED_0F15
);
pairwise_matches_old!(bhattacharyya_matches_old, bhattacharyya_coeff, 0xB0B0_CAFE);
pairwise_matches_old!(
    hellinger_squared_matches_old,
    hellinger_squared,
    0x4E11_1A6E
);

/// Generator whose gradient reads the incoming buffer, so a buffer that is
/// not zeroed (or has the wrong length) would change the result.
struct ReadsOut;

impl BregmanGenerator for ReadsOut {
    fn f(&self, x: &[f64]) -> Result<f64> {
        Ok(x.iter().map(|v| v * v * v).sum())
    }
    fn grad_into(&self, x: &[f64], out: &mut [f64]) -> Result<()> {
        if out.len() != x.len() {
            return Err(Error::LengthMismatch(out.len(), x.len()));
        }
        for (o, &v) in out.iter_mut().zip(x) {
            *o += 3.0 * v * v + 1.0;
        }
        Ok(())
    }
}

#[test]
fn bregman_matches_old_across_stack_threshold() {
    let mut rng = Rng(0xB4E6_3A11);
    for _ in 0..5_000 {
        let n = rng.below(70) as usize;
        let m = if rng.below(10) == 0 {
            rng.below(70) as usize
        } else {
            n
        };
        let p: Vec<f64> = (0..n).map(|_| entry(&mut rng).abs()).collect();
        let q: Vec<f64> = (0..m).map(|_| entry(&mut rng).abs()).collect();
        let ctx = format!("p={p:?} q={q:?}");
        assert!(
            same(
                logp::bregman_divergence(&SquaredL2, &p, &q),
                old::bregman_divergence(&SquaredL2, &p, &q)
            ),
            "SquaredL2 {ctx}"
        );
        assert!(
            same(
                logp::bregman_divergence(&NegEntropy, &p, &q),
                old::bregman_divergence(&NegEntropy, &p, &q)
            ),
            "NegEntropy {ctx}"
        );
        assert!(
            same(
                logp::bregman_divergence(&ReadsOut, &p, &q),
                old::bregman_divergence(&ReadsOut, &p, &q)
            ),
            "ReadsOut {ctx}"
        );
        assert!(
            same(
                logp::total_bregman_divergence(&ReadsOut, &p, &q),
                old::total_bregman_divergence(&ReadsOut, &p, &q)
            ),
            "total ReadsOut {ctx}"
        );
    }
}

// --- Pinned edge cases the random generator may hit only rarely ---

#[test]
fn js_subnormal_midpoint_underflow_is_domain_error() {
    // m_1 = 0.5 * (5e-324 + 0) rounds to 0, so KL(p || m) is undefined. The
    // old implementation reported this from the inner kl_divergence call.
    let p = [1.0, 5e-324];
    let q = [1.0, 0.0];
    let r = logp::jensen_shannon_divergence(&p, &q, 1e-9);
    assert!(matches!(r, Err(Error::Domain(_))), "{r:?}");
    assert!(same(r, old::jensen_shannon_divergence(&p, &q, 1e-9)));
}

#[test]
fn js_midpoint_overflow_reports_nonfinite_m() {
    // With a huge tol, p and q validate but m_0 = 0.5 * (1e308 + 1e308)
    // overflows; the old inner validation of m reported NonFinite at index 0.
    let p = [1e308, 0.0];
    let q = [1e308, 0.0];
    let r = logp::jensen_shannon_divergence(&p, &q, f64::INFINITY);
    assert!(matches!(r, Err(Error::NonFinite { idx: 0, .. })), "{r:?}");
    assert!(same(
        r,
        old::jensen_shannon_divergence(&p, &q, f64::INFINITY)
    ));
}

#[test]
fn validate_simplex_reports_first_offending_index_on_long_input() {
    // Long input takes the fused path; the error must still name the first
    // bad index and its value.
    let mut p = vec![1.0 / 32.0; 32];
    p[5] = -0.25;
    p[9] = f64::NAN;
    let r = logp::validate_simplex(&p, 1e-9);
    assert!(
        matches!(r, Err(Error::Negative { idx: 5, value }) if value == -0.25),
        "{r:?}"
    );
}

#[test]
fn negative_zero_entries_are_accepted_as_before() {
    let p = [1.0, -0.0];
    let q = [-0.0, 1.0];
    assert!(same(
        logp::bhattacharyya_coeff(&p, &q, 1e-9),
        old::bhattacharyya_coeff(&p, &q, 1e-9)
    ));
    assert!(same(
        logp::jensen_shannon_divergence(&p, &q, 1e-9),
        old::jensen_shannon_divergence(&p, &q, 1e-9)
    ));
}
