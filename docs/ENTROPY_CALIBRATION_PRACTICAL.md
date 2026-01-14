# Practical Integration: Entropy Calibration Paper

Sensible, testable improvements based on Cao, Valiant, Liang (2025).

## Current State

✅ **Already implemented:**
- `surp::entropy_calibration` - Core metrics (EntCE, per-step tracking)
- `surp::zipf` - Power law fitting with `singleton_mass_scaling_exponent()`
- `surp::fdiv::hellinger_distance` - Used in paper's theory

✅ **Tests passing:** 41 tests, all green

## What's Missing (Testable)

### 1. Explicit Connection: Zipf → Entropy Calibration Scaling

The paper's Section 3 shows scaling exponent depends on power law α. We have both pieces but they're not connected.

**Add to `surp/src/entropy_calibration.rs`:**

```rust
use crate::zipf::{zipf_fit_from_counts, ZipfError};

/// Predict calibration scaling from token distribution.
///
/// From paper Section 3: EntCE scales as n^(1/α - 1) where:
/// - n = dataset/model size
/// - α = power law exponent from token distribution
///
/// Returns predicted scaling exponent (negative = improves with scale).
///
/// # Example
///
/// ```rust
/// use surp::entropy_calibration::predict_calibration_scaling;
///
/// // Text-like distribution (α ≈ 1) → very slow improvement
/// let text_counts = vec![10000, 5000, 3333, 2500, 2000]; // roughly α=1
/// let scaling = predict_calibration_scaling(&text_counts).unwrap();
/// assert!(scaling > -0.1, "text should have slow scaling");
///
/// // Code-like distribution (α ≈ 1.5) → moderate improvement  
/// let code_counts = vec![10000, 2828, 1547, 1000, 707]; // roughly α=1.5
/// let scaling = predict_calibration_scaling(&code_counts).unwrap();
/// assert!(scaling < -0.2, "code should have faster scaling");
/// ```
pub fn predict_calibration_scaling(
    token_counts: &[usize],
) -> Result<f64, ZipfError> {
    let zipf = zipf_fit_from_counts(token_counts, 5, 5000)?;
    match zipf {
        Some(fit) => Ok(crate::zipf::singleton_mass_scaling_exponent(fit.alpha)),
        None => Err(ZipfError::NotEnoughPoints(0)),
    }
}
```

**Tests needed:**
- Text distribution (α≈1) → scaling ≈ 0
- Code distribution (α≈1.5) → scaling ≈ -0.33
- Edge cases: too few tokens, invalid counts

### 2. Per-Step Accumulation Detection

The paper shows entropy grows with generation length. Add simple slope detection.

**Add to `surp/src/entropy_calibration.rs`:**

```rust
/// Detect if entropy is accumulating over generation steps.
///
/// Returns slope (bits/token²) and whether accumulation detected.
///
/// # Example
///
/// ```rust
/// use surp::entropy_calibration::detect_entropy_accumulation;
///
/// // Entropy increasing: [1.0, 1.1, 1.2, 1.3, 1.4]
/// let increasing = vec![1.0, 1.1, 1.2, 1.3, 1.4];
/// let (slope, is_accumulating) = detect_entropy_accumulation(&increasing);
/// assert!(is_accumulating);
/// assert!(slope > 0.0);
///
/// // Flat entropy: [1.0, 1.0, 1.0, 1.0]
/// let flat = vec![1.0; 4];
/// let (slope, is_accumulating) = detect_entropy_accumulation(&flat);
/// assert!(!is_accumulating);
/// ```
pub fn detect_entropy_accumulation(entropy_by_step: &[f64]) -> (f64, bool) {
    if entropy_by_step.len() < 2 {
        return (0.0, false);
    }
    
    // Simple linear regression: entropy[t] ≈ a + b*t
    let n = entropy_by_step.len() as f64;
    let mean_t = (n - 1.0) / 2.0;
    let mean_h: f64 = entropy_by_step.iter().sum::<f64>() / n;
    
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (t, &h) in entropy_by_step.iter().enumerate() {
        let dt = t as f64 - mean_t;
        let dh = h - mean_h;
        sxy += dt * dh;
        sxx += dt * dt;
    }
    
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let threshold = 0.01; // bits/token² (from paper: noticeable accumulation)
    (slope, slope > threshold)
}
```

**Tests needed:**
- Increasing entropy → positive slope, detected
- Flat entropy → zero slope, not detected
- Decreasing entropy → negative slope, not detected
- Edge cases: empty, single value, all NaN

### 3. Better Documentation Cross-References

**Update `surp/src/entropy_calibration.rs` module docs:**

```rust
//! # Entropy calibration for generative models
//!
//! This module implements the core metrics used in
//! *Cao, Valiant, Liang (NeurIPS 2025)* "On the Entropy Calibration of Language Models".
//!
//! ## Connection to Power Law Scaling
//!
//! The paper's Section 3 shows that calibration improvement with scale depends on
//! the power law exponent α of the token distribution:
//!
//! - Text (α ≈ 1): Very slow improvement (scaling exponent ≈ 0)
//! - Code (α ≈ 1.5): Moderate improvement (scaling exponent ≈ -0.33)
//!
//! Use [`crate::zipf::zipf_fit_from_counts`] to estimate α, then
//! [`crate::zipf::singleton_mass_scaling_exponent`] to get the scaling exponent.
//!
//! See also: [`predict_calibration_scaling`] for a convenience function.
```

**Update `surp/src/zipf.rs` module docs:**

```rust
//! # Zipf / power-law tails
//!
//! A lot of discrete data (text tokens, code tokens, entity IDs) has heavy tails.
//!
//! ## Connection to Entropy Calibration
//!
//! The entropy calibration paper (Cao, Valiant, Liang 2025) shows that the
//! power law exponent α predicts how quickly generation stability improves with scale.
//! See [`singleton_mass_scaling_exponent`] and [`crate::entropy_calibration`].
```

## Testing Strategy

### Property Tests (Add to existing `proptest!` blocks)

```rust
// In entropy_calibration::tests
proptest! {
    #[test]
    fn entce_bounded_when_logprobs_reasonable(
        gen in prop::collection::vec(-10.0f64..=0.0, 10..100),
        ref_ in prop::collection::vec(-10.0f64..=0.0, 10..100),
    ) {
        if gen.len() == ref_.len() {
            let stats = entropy_calibration_bits(&gen, &ref_).unwrap();
            // EntCE should be finite
            prop_assert!(stats.entce_bits_per_token.is_finite());
            // Entropy and log loss should be non-negative
            prop_assert!(stats.entropy_bits_per_token >= 0.0);
            prop_assert!(stats.log_loss_bits_per_token >= 0.0);
        }
    }
    
    #[test]
    fn accumulation_detection_consistent(
        xs in prop::collection::vec(0.0f64..=10.0, 5..50),
    ) {
        let (slope1, acc1) = detect_entropy_accumulation(&xs);
        // Reverse should give opposite slope
        let mut reversed = xs.clone();
        reversed.reverse();
        let (slope2, acc2) = detect_entropy_accumulation(&reversed);
        prop_assert!((slope1 + slope2).abs() < 0.1);
    }
}

// In zipf::tests
#[test]
fn scaling_exponent_connects_to_calibration() {
    // From paper: text (α≈1) has scaling ≈ 0, code (α≈1.5) has scaling ≈ -0.33
    let text_alpha = 1.0;
    let code_alpha = 1.5;
    
    let text_scaling = singleton_mass_scaling_exponent(text_alpha);
    let code_scaling = singleton_mass_scaling_exponent(code_alpha);
    
    assert!((text_scaling - 0.0).abs() < 1e-10);
    assert!((code_scaling - (-1.0/3.0)).abs() < 1e-10);
    assert!(code_scaling < text_scaling, "code should improve faster");
}
```

### Integration Tests

```rust
// tests/integration_test.rs (new file)
#[test]
fn zipf_to_calibration_scaling_roundtrip() {
    // Generate synthetic token counts with known α
    let alpha = 1.5;
    let scale = 1_000_000.0;
    let counts: Vec<usize> = (1..=1000)
        .map(|r| (scale / (r as f64).powf(alpha)).round() as usize)
        .collect();
    
    // Fit Zipf
    let zipf = zipf_fit_from_counts(&counts, 5, 1000).unwrap().unwrap();
    assert!((zipf.alpha - alpha).abs() < 0.1);
    
    // Predict scaling
    let scaling = predict_calibration_scaling(&counts).unwrap();
    let expected = singleton_mass_scaling_exponent(alpha);
    assert!((scaling - expected).abs() < 0.1);
}
```

## Crate Topology Assessment

### Current Structure: ✅ Good

```
surp/
├── lib.rs              (core: entropy, KL, MI)
├── fdiv.rs             (Hellinger, Rényi, χ²)
├── unseen.rs           (Valiant-Valiant)
├── zipf.rs             (power law fitting)
└── entropy_calibration.rs  (LLM calibration)
```

**Verdict:** No refactor needed. Structure is clean:
- Each module has clear purpose
- Dependencies are minimal (zipf → entropy_calibration is fine)
- Tests are organized per module

### Potential New Crate? ❌ No

**Why not:**
- `entropy_calibration` is small (~300 lines)
- Tightly coupled to `surp`'s core (entropy, log loss)
- Would create unnecessary dependency overhead

**If it grows:** Consider `surp-calibration` sub-crate only if:
- It exceeds 1000 lines
- It needs heavy dependencies (e.g., plotting, ML frameworks)
- It's used independently of `surp` core

## Documentation Improvements

### 1. Add Paper Reference to README

**Update `surp/README.md`:**

```markdown
## Entropy Calibration

Metrics for evaluating whether an LLM's generation entropy matches its log loss
on reference text. Based on Cao, Valiant, Liang (2025).

**Key insight:** Entropy often grows with generation length (error accumulation),
even when log loss on human text is flat. The gap (EntCE) measures this.

**Scaling behavior:** Calibration improvement with model/data scale depends on
the power law exponent α of the token distribution:
- Text (α ≈ 1): Very slow improvement
- Code (α ≈ 1.5): Moderate improvement

See [`zipf`] module for power law fitting and scaling predictions.
```

### 2. Add Cross-Reference in `zipf` Module

**Update `surp/src/zipf.rs` header:**

```rust
//! # Zipf / power-law tails
//!
//! Heavy-tailed distributions are common in discrete data (text, code, entity IDs).
//!
//! ## Connection to Entropy Calibration
//!
//! The entropy calibration paper (Cao, Valiant, Liang 2025) shows that power law
//! exponent α predicts scaling behavior:
//!
//! - α ≈ 1 (text): Singleton mass scales as m^0 → no improvement with scale
//! - α ≈ 1.5 (code): Singleton mass scales as m^(-0.33) → moderate improvement
//!
//! Use [`singleton_mass_scaling_exponent`] to compute the scaling exponent.
//! See [`crate::entropy_calibration`] for the calibration metrics.
```

## Implementation Priority

### High (Do First)

1. ✅ **Add `predict_calibration_scaling()`** - Simple, testable, connects existing pieces
2. ✅ **Add `detect_entropy_accumulation()`** - Useful diagnostic, easy to test
3. ✅ **Add property tests** - Ensure invariants hold
4. ✅ **Update cross-references in docs** - Low effort, high value

### Medium (If Time)

5. **Add integration test** - Verify zipf → calibration connection
6. **Add edge case tests** - Empty inputs, NaN handling, etc.

### Low (Nice to Have)

7. **Add example in README** - Show zipf + calibration together
8. **Add benchmark** - Track performance of new functions

## What NOT to Do

❌ **Don't create new crate** - `entropy_calibration` is fine in `surp`
❌ **Don't implement Algorithm 1** - Too complex, not practical
❌ **Don't add plotting** - Out of scope for `surp`
❌ **Don't integrate with `hop` yet** - Wait until we have solid tests

## Testing Checklist

Before merging any changes:

- [ ] All new functions have unit tests
- [ ] Property tests cover edge cases
- [ ] Integration test verifies zipf → calibration connection
- [ ] Docs updated with cross-references
- [ ] `cargo test` passes
- [ ] `cargo clippy` passes
- [ ] `cargo fmt` applied
