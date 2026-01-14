//! # Entropy calibration for generative models
//!
//! This module implements the core metrics used in
//! *Cao, Valiant, Liang (NeurIPS 2025)* “On the Entropy Calibration of Language Models”.
//!
//! ## Intuition (why two numbers?)
//!
//! There are two relevant expectations for an autoregressive model \(\hat p\):
//!
//! - **Entropy over generations**: sample a continuation from the model, then score the sampled
//!   tokens under the same model.
//!   \[
//!   H(\hat p) = \mathbb{E}_{\hat Y \sim \hat p}[ -\log \hat p(\hat Y) ].
//!   \]
//!
//! - **Log loss on human text**: sample a continuation from the *data* distribution \(p^\*\), then
//!   score it under the model.
//!   \[
//!   L(p^\* \Vert \hat p) = \mathbb{E}_{Y \sim p^\*}[ -\log \hat p(Y) ].
//!   \]
//!
//! For a well-calibrated model, these should match (per token, and over time), but empirically
//! models often have **entropy that grows with generation length** (“error accumulation”).
//!
//! ## Metric
//!
//! We report **entropy calibration error** (EntCE) in bits per token:
//!
//! \[
//! \mathrm{EntCE} = H(\hat p) - L(p^\* \Vert \hat p).
//! \]
//!
//! - **EntCE > 0**: sampled generations have higher per-token entropy than the model’s per-token
//!   log loss on reference text. This is commonly associated with error accumulation, but the sign
//!   alone is not a complete quality metric.
//! - **EntCE < 0**: sampled generations have lower per-token entropy than the model’s per-token
//!   log loss on reference text. This can happen under strong truncation or alignment, and can
//!   correlate with repetitiveness, but again the sign alone is not definitive.
//!
//! ## Inputs
//!
//! This module consumes **per-token log-probabilities** (natural log / ln), typically produced by
//! model inference code:
//!
//! - For \(H(\hat p)\): logprobs of *generated tokens* under the model given their model-generated
//!   prefixes.
//! - For \(L(p^\* \Vert \hat p)\): logprobs of *reference tokens* (human/code) under the model given
//!   their reference prefixes.
//!
//! If you already have negative log-likelihoods (NLLs), just pass `-nll` as log-probabilities.
//!
//! ## What could go wrong
//!
//! - **Mismatched conditioning**: for log loss, the prefix should be the reference prefix
//!   (“teacher forcing”), not the model’s own prefix.
//! - **Different truncation/filters**: entropy is about the distribution you sample from. If you
//!   apply truncation (top-p, min-p, temperature), EntCE changes even if the base model is fixed.
//! - **Non-finite logprobs**: `-inf` logprob means the model assigns probability 0; EntCE becomes
//!   infinite / undefined for that sample.
//! - **Length effects**: if you compare across different maximum lengths, EntCE is not apples to
//!   apples. Prefer per-step curves or fixed-length evaluation.
//!
//! # Examples
//!
//! ```rust
//! use surp::entropy_calibration::{entropy_calibration_bits, mean_nll_bits_from_ln};
//!
//! // Suppose a model assigns probability 0.5 to each sampled token
//! // (logprob = ln(0.5) for each token).
//! let gen = vec![(0.5_f64).ln(); 10];
//! let ref_ = vec![(0.5_f64).ln(); 10];
//!
//! let h_bits = mean_nll_bits_from_ln(&gen).unwrap();
//! assert!((h_bits - 1.0).abs() < 1e-12);
//!
//! let stats = entropy_calibration_bits(&gen, &ref_).unwrap();
//! assert!(stats.entce_bits_per_token.abs() < 1e-12);
//! ```
//!
//! (Note: in real evaluation, `gen` and `ref_` will typically differ.)

use std::f64::consts::LN_2;
use thiserror::Error;

/// Errors for entropy calibration computations.
#[derive(Debug, Error)]
pub enum EntropyCalibrationError {
    /// Input slice is empty.
    #[error("empty input")]
    EmptyInput,

    /// Two inputs were expected to have equal lengths (e.g. aligned per-step comparisons).
    #[error("length mismatch: {a} vs {b}")]
    LengthMismatch { a: usize, b: usize },

    /// A log-probability was NaN or infinite.
    #[error("non-finite log probability at index {index}: {value}")]
    NonFiniteLogProb { index: usize, value: f64 },

    /// Internal consistency error: a step had no contributing sequences.
    ///
    /// This should not happen if `max_len` is computed as the maximum sequence length,
    /// but returning an error is safer than propagating NaNs.
    #[error("no observations at step {step}")]
    MissingStep { step: usize },
}

/// Summary statistics for entropy calibration (bits per token).
#[derive(Clone, Debug, PartialEq)]
pub struct EntropyCalibrationStats {
    /// Mean entropy over model generations, estimated by mean NLL of sampled tokens (bits/token).
    pub entropy_bits_per_token: f64,
    /// Mean log loss on reference text, estimated by mean NLL of reference tokens (bits/token).
    pub log_loss_bits_per_token: f64,
    /// Entropy calibration error: entropy - log loss (bits/token).
    pub entce_bits_per_token: f64,
}

/// Compute mean negative log-probability in bits/token from per-token log-probabilities in ln.
///
/// This is the core estimator used for both:
/// - entropy over generations (use generated-token logprobs), and
/// - log loss on reference text (use reference-token logprobs under teacher forcing).
pub fn mean_nll_bits_from_ln(log_probs_ln: &[f64]) -> Result<f64, EntropyCalibrationError> {
    if log_probs_ln.is_empty() {
        return Err(EntropyCalibrationError::EmptyInput);
    }

    let mut sum_ln = 0.0;
    for (i, &lp) in log_probs_ln.iter().enumerate() {
        if !lp.is_finite() {
            return Err(EntropyCalibrationError::NonFiniteLogProb {
                index: i,
                value: lp,
            });
        }
        sum_ln += lp;
    }

    // NLL (nats/token) = -mean(log p) ; convert to bits/token by dividing by ln 2.
    let mean_ln = sum_ln / (log_probs_ln.len() as f64);
    Ok((-mean_ln) / LN_2)
}

/// Compute entropy calibration stats in bits/token from two sets of per-token log-probabilities.
///
/// - `gen_log_probs_ln`: logprobs (ln) of tokens sampled from the model, scored under the model.
/// - `ref_log_probs_ln`: logprobs (ln) of reference tokens, scored under the model.
pub fn entropy_calibration_bits(
    gen_log_probs_ln: &[f64],
    ref_log_probs_ln: &[f64],
) -> Result<EntropyCalibrationStats, EntropyCalibrationError> {
    let h = mean_nll_bits_from_ln(gen_log_probs_ln)?;
    let l = mean_nll_bits_from_ln(ref_log_probs_ln)?;
    Ok(EntropyCalibrationStats {
        entropy_bits_per_token: h,
        log_loss_bits_per_token: l,
        entce_bits_per_token: h - l,
    })
}

/// Compute per-step mean NLL (bits/token) from a batch of sequences of log-probabilities (ln).
///
/// Returns a vector `out` where `out[t]` is the mean NLL at step `t` over sequences that have a
/// token at that step.
pub fn mean_nll_bits_by_step_from_ln(
    sequences_log_probs_ln: &[Vec<f64>],
) -> Result<Vec<f64>, EntropyCalibrationError> {
    if sequences_log_probs_ln.is_empty() {
        return Err(EntropyCalibrationError::EmptyInput);
    }

    let max_len = sequences_log_probs_ln
        .iter()
        .map(|s| s.len())
        .max()
        .unwrap_or(0);

    if max_len == 0 {
        return Err(EntropyCalibrationError::EmptyInput);
    }

    let mut sums = vec![0.0_f64; max_len];
    let mut counts = vec![0usize; max_len];

    for seq in sequences_log_probs_ln {
        for (t, &lp) in seq.iter().enumerate() {
            if !lp.is_finite() {
                return Err(EntropyCalibrationError::NonFiniteLogProb {
                    index: t,
                    value: lp,
                });
            }
            sums[t] += lp;
            counts[t] += 1;
        }
    }

    let mut out = Vec::with_capacity(max_len);
    for t in 0..max_len {
        if counts[t] == 0 {
            return Err(EntropyCalibrationError::MissingStep { step: t });
        }
        let mean_ln = sums[t] / (counts[t] as f64);
        out.push((-mean_ln) / LN_2);
    }

    Ok(out)
}

/// Compute per-step EntCE (bits/token) from aligned per-step curves.
///
/// This is the most “paper-faithful” usage when you have fixed-length evaluations.
pub fn entce_bits_by_step(
    entropy_bits_by_step: &[f64],
    log_loss_bits_by_step: &[f64],
) -> Result<Vec<f64>, EntropyCalibrationError> {
    if entropy_bits_by_step.is_empty() || log_loss_bits_by_step.is_empty() {
        return Err(EntropyCalibrationError::EmptyInput);
    }
    if entropy_bits_by_step.len() != log_loss_bits_by_step.len() {
        return Err(EntropyCalibrationError::LengthMismatch {
            a: entropy_bits_by_step.len(),
            b: log_loss_bits_by_step.len(),
        });
    }

    Ok(entropy_bits_by_step
        .iter()
        .zip(log_loss_bits_by_step.iter())
        .map(|(&h, &l)| h - l)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn mean_nll_bits_ln_half_is_one_bit() {
        let lp = (0.5_f64).ln();
        let xs = vec![lp; 8];
        let nll = mean_nll_bits_from_ln(&xs).unwrap();
        assert!((nll - 1.0).abs() < 1e-12);
    }

    #[test]
    fn entropy_calibration_zero_when_arrays_equal() {
        let lp = (0.25_f64).ln();
        let gen = vec![lp; 16];
        let ref_ = vec![lp; 16];
        let stats = entropy_calibration_bits(&gen, &ref_).unwrap();
        assert!(stats.entce_bits_per_token.abs() < 1e-12);
    }

    #[test]
    fn by_step_mean_handles_ragged_lengths() {
        let a = vec![(0.5_f64).ln(), (0.25_f64).ln()];
        let b = vec![(0.5_f64).ln()];
        let curve = mean_nll_bits_by_step_from_ln(&[a, b]).unwrap();
        // step 0: both sequences contribute log(0.5) -> 1 bit
        assert!((curve[0] - 1.0).abs() < 1e-12);
        // step 1: only first sequence contributes log(0.25) -> 2 bits
        assert!((curve[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn entce_by_step_requires_alignment() {
        let err = entce_bits_by_step(&[1.0, 2.0], &[1.0]).unwrap_err();
        match err {
            EntropyCalibrationError::LengthMismatch { .. } => {}
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn non_finite_logprob_errors() {
        let xs = vec![0.0_f64, f64::NEG_INFINITY];
        let err = mean_nll_bits_from_ln(&xs).unwrap_err();
        match err {
            EntropyCalibrationError::NonFiniteLogProb { index, .. } => assert_eq!(index, 1),
            other => panic!("unexpected error: {other:?}"),
        }
    }

    proptest! {
        #[test]
        fn nll_bits_is_non_negative_for_log_probs_leq_zero(xs in prop::collection::vec(-20.0f64..=0.0, 1..256)) {
            let nll = mean_nll_bits_from_ln(&xs).unwrap();
            prop_assert!(nll.is_finite());
            prop_assert!(nll >= 0.0);
        }

        #[test]
        fn nll_bits_monotone_in_log_probs(
            base in prop::collection::vec(-20.0f64..=0.0, 1..256),
            deltas in prop::collection::vec(0.0f64..=5.0, 1..256),
        ) {
            let n = base.len().min(deltas.len());
            let lp1: Vec<f64> = base[..n].to_vec();
            let lp2: Vec<f64> = (0..n).map(|i| (lp1[i] + deltas[i]).min(0.0)).collect();

            // If lp1 <= lp2 elementwise (since deltas >= 0), then -mean(lp1) >= -mean(lp2).
            let nll1 = mean_nll_bits_from_ln(&lp1).unwrap();
            let nll2 = mean_nll_bits_from_ln(&lp2).unwrap();
            prop_assert!(nll1 + 1e-12 >= nll2);
        }
    }
}
