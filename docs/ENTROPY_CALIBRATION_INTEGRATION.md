# Entropy Calibration Integration Plan

How the Cao, Valiant, Liang (2025) paper connects to the Tekne stack.

## Current State

### What We Have

1. **`surp::entropy_calibration`** - Core metrics implemented:
   - `entropy_calibration_bits()` - Main EntCE computation
   - `mean_nll_bits_by_step_from_ln()` - Per-step entropy tracking
   - `entce_bits_by_step()` - Per-step calibration error

2. **`surp::fdiv::hellinger_distance`** - Used in paper's theoretical analysis

3. **`surp::zipf::zipf_fit_from_counts`** - Power law exponent estimation (paper Section 3)

4. **`cerno`** - Uses entropy in Sinkhorn, cross-entropy losses, but not for generation calibration

5. **`hop`** - LLM generation (summarization, QA, query generation) but no entropy tracking

## Integration Opportunities

### 1. Enhanced `surp::entropy_calibration` Module

#### A. Power Law Analysis (Section 3)

The paper shows that scaling behavior depends on power law exponent α:
- Text: α ≈ 1 → very slow improvement with scale
- Code: α ≈ 1.5 → moderate improvement

**Add to `surp/src/entropy_calibration.rs`:**

```rust
/// Analyze power law exponent from token distribution to predict scaling behavior.
///
/// Returns predicted scaling exponent: for α close to 1, expect slow improvement.
pub fn predict_scaling_exponent(
    token_counts: &[usize],
) -> Result<f64, ZipfError> {
    let zipf_fit = zipf_fit_from_counts(token_counts, 5, 5000)?;
    match zipf_fit {
        Some(fit) => {
            // From paper: scaling exponent ≈ 1/α - 1
            // For α=1: exponent ≈ 0 (very slow)
            // For α=1.5: exponent ≈ -0.33 (moderate)
            Ok(1.0 / fit.alpha - 1.0)
        }
        None => Err(ZipfError::NotEnoughPoints(0)),
    }
}
```

#### B. Per-Step Entropy Tracking with Diagnostics

The paper shows entropy grows with generation length (error accumulation).

**Enhance existing `mean_nll_bits_by_step_from_ln()` with:**

```rust
/// Per-step entropy analysis with accumulation detection.
#[derive(Debug, Clone)]
pub struct PerStepAnalysis {
    pub entropy_by_step: Vec<f64>,
    pub log_loss_by_step: Vec<f64>,
    pub entce_by_step: Vec<f64>,
    /// Slope of entropy growth (bits/token²)
    pub entropy_slope: f64,
    /// Whether entropy is accumulating (slope > threshold)
    pub is_accumulating: bool,
    /// Predicted entropy at step T (extrapolation)
    pub predicted_entropy_at_t: Option<f64>,
}

pub fn analyze_per_step_accumulation(
    gen_sequences: &[Vec<f64>],
    ref_sequences: &[Vec<f64>],
) -> Result<PerStepAnalysis, EntropyCalibrationError> {
    let h_by_step = mean_nll_bits_by_step_from_ln(gen_sequences)?;
    let l_by_step = mean_nll_bits_by_step_from_ln(ref_sequences)?;
    let entce = entce_bits_by_step(&h_by_step, &l_by_step)?;
    
    // Linear regression on entropy growth
    let slope = linear_slope(&h_by_step);
    let is_accumulating = slope > 0.01; // threshold from paper
    
    Ok(PerStepAnalysis {
        entropy_by_step: h_by_step,
        log_loss_by_step: l_by_step,
        entce_by_step: entce,
        entropy_slope: slope,
        is_accumulating,
        predicted_entropy_at_t: if is_accumulating {
            Some(extrapolate_entropy(&h_by_step))
        } else {
            None
        },
    })
}
```

#### C. Future Entropy Heuristics (Section 5)

The paper's Algorithm 1 is impractical, but we can extract heuristics:

```rust
/// Heuristic: adjust token probabilities based on expected future entropy.
///
/// This is a simplified version of the paper's future entropy scaling.
/// Instead of fitting full predictors, we use a lookahead window.
pub fn future_entropy_adjusted_logprobs(
    current_logprobs: &[f64],
    lookahead_window: usize,
    alpha: f64, // adjustment strength
) -> Vec<f64> {
    // Simplified: penalize tokens that lead to high entropy in next k steps
    // This is a heuristic approximation of the paper's Algorithm 1
    current_logprobs.iter()
        .map(|&lp| {
            // In practice, you'd compute future entropy via sampling
            // For now, this is a placeholder structure
            lp * (1.0 + alpha)
        })
        .collect()
}
```

### 2. Integration with `hop` RAG Pipeline

#### A. Track Entropy During Generation

**Add to `hop-core/src/extractors/qa_llm.rs` or similar:**

```rust
use surp::entropy_calibration::{entropy_calibration_bits, EntropyCalibrationStats};

pub struct GenerationWithCalibration {
    pub text: String,
    pub log_probs: Vec<f64>, // per-token log probabilities
    pub calibration: Option<EntropyCalibrationStats>,
}

impl LLMExtractor {
    pub fn generate_with_calibration(
        &self,
        prompt: &str,
        reference: Option<&str>, // for log loss computation
    ) -> Result<GenerationWithCalibration> {
        // Generate with logprobs
        let (text, log_probs) = self.generate_with_logprobs(prompt)?;
        
        // If reference available, compute calibration
        let calibration = if let Some(ref_text) = reference {
            let ref_log_probs = self.logprobs_for_sequence(ref_text)?;
            Some(entropy_calibration_bits(&log_probs, &ref_log_probs)?)
        } else {
            None
        };
        
        Ok(GenerationWithCalibration {
            text,
            log_probs,
            calibration,
        })
    }
}
```

#### B. Add to RAG Evaluation Metrics

**Enhance `hop/src/hop/rag_metrics.py`:**

```python
from surp import entropy_calibration_bits

class RAGMetrics:
    # ... existing fields ...
    
    # Generation calibration (new)
    entropy_calibration_error: float = 0.0  # EntCE in bits/token
    entropy_per_token: float = 0.0
    log_loss_per_token: float = 0.0
    is_well_calibrated: bool = False  # EntCE < threshold
```

**In `RAGEvaluator.evaluate()`:**

```python
def evaluate(self, ..., generation_logprobs: Optional[List[float]] = None,
             reference_logprobs: Optional[List[float]] = None):
    # ... existing metrics ...
    
    # Entropy calibration (if logprobs available)
    if generation_logprobs and reference_logprobs:
        stats = entropy_calibration_bits(
            generation_logprobs, 
            reference_logprobs
        )
        metrics.entropy_calibration_error = stats.entce_bits_per_token
        metrics.entropy_per_token = stats.entropy_bits_per_token
        metrics.log_loss_per_token = stats.log_loss_bits_per_token
        metrics.is_well_calibrated = abs(stats.entce_bits_per_token) < 0.5
```

### 3. Integration with `cerno` Retrieval/Reranking

#### A. Calibration-Aware Reranking

The paper shows instruction tuning reduces entropy (Section 4.3). This suggests rerankers might benefit from entropy-aware scoring.

**In `cerno-rerank`:**

```rust
use surp::entropy_calibration::mean_nll_bits_from_ln;

/// Rerank with entropy calibration penalty.
///
/// Penalizes candidates that would lead to high entropy generations.
pub fn rerank_with_entropy_penalty(
    candidates: &[Candidate],
    query: &str,
    entropy_threshold: f64,
) -> Vec<Candidate> {
    candidates.iter()
        .map(|c| {
            // Estimate future entropy if we generate from this candidate
            let estimated_entropy = estimate_generation_entropy(c, query);
            let penalty = if estimated_entropy > entropy_threshold {
                (estimated_entropy - entropy_threshold) * 0.1
            } else {
                0.0
            };
            (c.clone(), penalty)
        })
        .sorted_by(|(a, pa), (b, pb)| {
            (a.score - pa).partial_cmp(&(b.score - pb)).unwrap()
        })
        .map(|(c, _)| c)
        .collect()
}
```

#### B. Cross-Entropy Loss Calibration

`cerno-train` uses cross-entropy for ranking. The paper's insights about entropy calibration could inform loss design.

**Connection point:** The paper shows truncation (top-k, top-p) reduces entropy but increases log loss. This tradeoff is relevant for ranking where we want diversity but also quality.

### 4. New Utility: `surp::entropy_calibration::scaling`

**New module for scaling analysis:**

```rust
//! Scaling behavior analysis from entropy calibration paper.
//!
//! Predicts how calibration improves with model/data scale based on
//! power law exponent of the data distribution.

use crate::zipf::zipf_fit_from_counts;

/// Predict calibration scaling exponent from token distribution.
///
/// From paper Section 3: scaling exponent ≈ 1/α - 1 where α is power law exponent.
pub fn predict_calibration_scaling(
    token_counts: &[usize],
) -> Result<ScalingPrediction, ZipfError> {
    let zipf = zipf_fit_from_counts(token_counts, 5, 5000)?;
    match zipf {
        Some(fit) => {
            let scaling_exp = 1.0 / fit.alpha - 1.0;
            Ok(ScalingPrediction {
                power_law_exponent: fit.alpha,
                scaling_exponent: scaling_exp,
                improvement_rate: match scaling_exp {
                    x if x > -0.1 => ImprovementRate::VerySlow,
                    x if x > -0.3 => ImprovementRate::Slow,
                    x if x > -0.5 => ImprovementRate::Moderate,
                    _ => ImprovementRate::Fast,
                },
            })
        }
        None => Err(ZipfError::NotEnoughPoints(0)),
    }
}

#[derive(Debug, Clone)]
pub struct ScalingPrediction {
    pub power_law_exponent: f64,
    pub scaling_exponent: f64,
    pub improvement_rate: ImprovementRate,
}

#[derive(Debug, Clone, Copy)]
pub enum ImprovementRate {
    VerySlow,  // α ≈ 1 (text)
    Slow,      // α ≈ 1.1-1.2
    Moderate,  // α ≈ 1.5 (code)
    Fast,      // α > 1.5
}
```

### 5. Integration with `jin` ANN Search

The paper doesn't directly connect to ANN, but there's an indirect link:

**Error accumulation in generation** → **Embedding quality** → **ANN search quality**

If a model's generations have high entropy (incoherent), the embeddings might be less useful for retrieval.

**Potential diagnostic in `hop`:**

```rust
// After generating summaries/answers, check if high entropy correlates
// with poor retrieval performance
pub fn diagnose_retrieval_quality_vs_entropy(
    queries: &[String],
    retrieved_docs: &[Vec<Document>],
    generation_entropies: &[f64],
) -> Correlation {
    // High entropy generations might indicate:
    // 1. Model confusion → poor embeddings → worse retrieval
    // 2. Or: high entropy is fine if log loss is also high (calibrated)
}
```

## Practical Workflow Integration

### Example: Evaluating LLM in `hop`

```rust
use surp::entropy_calibration::{analyze_per_step_accumulation, predict_calibration_scaling};
use surp::zipf::zipf_fit_from_counts;

// 1. Generate with logprobs
let generations: Vec<(String, Vec<f64>)> = llm.generate_batch(prompts, return_logprobs=true)?;

// 2. Get reference logprobs (from ground truth or human text)
let references: Vec<Vec<f64>> = get_reference_logprobs(&ground_truth)?;

// 3. Analyze calibration
let analysis = analyze_per_step_accumulation(
    &generations.iter().map(|(_, lp)| lp.clone()).collect::<Vec<_>>(),
    &references,
)?;

// 4. Check if entropy is accumulating
if analysis.is_accumulating {
    warn!("Entropy accumulating: slope = {:.3} bits/token²", analysis.entropy_slope);
}

// 5. Predict scaling behavior from token distribution
let token_counts = count_tokens_in_corpus(&corpus);
let scaling = predict_calibration_scaling(&token_counts)?;
info!("Power law α = {:.2}, scaling exponent = {:.2}", 
      scaling.power_law_exponent, scaling.scaling_exponent);
```

### Example: Calibration-Aware RAG

```python
# In hop/src/hop/rag_metrics.py

def evaluate_with_calibration(
    self,
    query: str,
    retrieved_docs: List[Dict],
    generated_answer: str,
    generation_logprobs: List[float],
    reference_logprobs: Optional[List[float]] = None,
) -> RAGMetrics:
    metrics = self.evaluate(query, retrieved_docs, generated_answer)
    
    # Add entropy calibration
    if reference_logprobs:
        from surp import entropy_calibration_bits
        stats = entropy_calibration_bits(generation_logprobs, reference_logprobs)
        metrics.entropy_calibration_error = stats.entce_bits_per_token
        
        # High EntCE might indicate:
        # - Model derailment (EntCE >> 0)
        # - Repetition/mode collapse (EntCE << 0)
        if stats.entce_bits_per_token > 1.0:
            metrics.warnings.append("High entropy: model may be derailing")
        elif stats.entce_bits_per_token < -1.0:
            metrics.warnings.append("Low entropy: possible repetition")
    
    return metrics
```

## Missing Pieces to Add

### 1. Per-Step Visualization

The paper's Figure 2 shows entropy over time. Add plotting utilities:

```rust
// surp/src/entropy_calibration/visualization.rs (new)

pub fn plot_entropy_over_time(
    analysis: &PerStepAnalysis,
    output_path: &Path,
) -> Result<()> {
    // Generate plot showing:
    // - Entropy per step (solid line)
    // - Log loss per step (dashed line)
    // - EntCE = gap between them
}
```

### 2. Temperature/Truncation Analysis

Paper Section 4.3 shows temperature scaling reduces entropy but increases log loss.

**Add:**

```rust
/// Analyze entropy-log loss tradeoff across temperature settings.
pub fn analyze_temperature_tradeoff(
    base_logprobs: &[Vec<f64>],
    temperatures: &[f64],
) -> Vec<TemperatureAnalysis> {
    temperatures.iter()
        .map(|&temp| {
            let adjusted = apply_temperature(base_logprobs, temp);
            let stats = entropy_calibration_bits(&adjusted, &reference)?;
            TemperatureAnalysis {
                temperature: temp,
                entropy: stats.entropy_bits_per_token,
                log_loss: stats.log_loss_bits_per_token,
                entce: stats.entce_bits_per_token,
            }
        })
        .collect()
}
```

### 3. Dataset-Specific Scaling Predictions

The paper shows different scaling for WikiText (α≈0.92) vs CodeContests (α≈1.5).

**Add diagnostic:**

```rust
/// Predict optimal model size for target calibration error.
///
/// From paper: EntCE scales as n^(1/α - 1) where n is dataset size.
pub fn predict_model_size_for_target_entce(
    current_entce: f64,
    target_entce: f64,
    power_law_alpha: f64,
) -> f64 {
    let scaling_exp = 1.0 / power_law_alpha - 1.0;
    // target_entce / current_entce = (n_target / n_current)^scaling_exp
    // Solve for n_target / n_current
    (target_entce / current_entce).powf(1.0 / scaling_exp)
}
```

## Cross-Crate Synergies

### `surp` ↔ `cerno`

- **Sinkhorn entropy** (in `cerno-train`) vs **generation entropy** (in `surp`)
- Both use entropy regularization, but for different purposes
- Could unify entropy-based loss functions (see `cerno/docs/FYNCH_CONNECTIONS.md`)

### `surp` ↔ `hop`

- **Entropy calibration** as a quality metric for RAG generation
- Track entropy during query generation, summarization, QA
- Use EntCE to detect when LLM is derailing

### `surp` ↔ `jin`

- Indirect: poor calibration → poor embeddings → worse ANN search
- Could add diagnostic: "If EntCE > threshold, expect lower retrieval recall"

### `surp` ↔ `anno`

- `anno` has confidence calibration for NER (different from entropy calibration)
- Both measure "does model uncertainty match reality?"
- Could add entropy calibration for sequence labeling tasks

## Research Directions

### 1. Practical Future Entropy Approximation

The paper's Algorithm 1 is impractical (requires fitting predictors for each token). But we could:

- Use a small lookahead window (next 5-10 tokens)
- Approximate future entropy via sampling
- Apply heuristic adjustments based on token frequency

### 2. Calibration-Aware Decoding

Instead of fixed temperature/top-p, adjust dynamically based on:
- Current entropy vs target
- Predicted future entropy (heuristic)
- Power law exponent of the domain

### 3. RAG-Specific Calibration

For RAG, "reference" could be:
- Retrieved context (how well does generation match retrieved docs?)
- Ground truth answers
- Human-written continuations

Different references → different calibration interpretations.

## Implementation Priority

1. **High**: Per-step accumulation analysis (enhance existing `mean_nll_bits_by_step_from_ln`)
2. **High**: Power law scaling prediction (connect `zipf` to `entropy_calibration`)
3. **Medium**: Integration with `hop` generation tracking
4. **Medium**: Temperature tradeoff analysis
5. **Low**: Future entropy heuristics (simplified version of Algorithm 1)
6. **Low**: Visualization utilities

## References

- Cao, Valiant, Liang (2025). "On the Entropy Calibration of Language Models" (arXiv:2511.11966)
- Your existing `surp/src/entropy_calibration.rs` implements the core metrics
- `surp/src/zipf.rs` provides power law fitting needed for scaling analysis
- `surp/src/fdiv.rs` provides Hellinger distance used in paper's theory
