//! Distribution propagation through nonlinear functions.
//!
//! Given a Gaussian input N(mu, sigma^2), propagate through a nonlinear
//! function f to approximate the output distribution using moment matching.
//!
//! Two approximation methods:
//! - **Linearization** (first-order Taylor): output ~ N(f(mu), f'(mu)^2 * sigma^2)
//! - **Unscented transform**: sigma-point propagation (more accurate for nonlinear f)
//!
//! Based on: Petersen et al., "Distribution Propagation" -- propagating
//! distributions through computational graphs via moment matching.
//!
//! ## When to use
//!
//! Distribution propagation is useful when you have uncertain inputs and need
//! to estimate the distribution of outputs without Monte Carlo sampling.
//! Common in uncertainty quantification, Bayesian neural networks, and
//! probabilistic programming.

use crate::{Error, Result};

/// Gaussian distribution parameterized by mean and standard deviation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Gaussian {
    /// Mean of the distribution.
    pub mean: f64,
    /// Standard deviation (must be non-negative).
    pub std: f64,
}

impl Gaussian {
    /// Create a new Gaussian. Returns error if std is negative or non-finite.
    pub fn new(mean: f64, std: f64) -> Result<Self> {
        if !mean.is_finite() {
            return Err(Error::Domain("mean must be finite"));
        }
        if !std.is_finite() || std < 0.0 {
            return Err(Error::Domain("std must be finite and non-negative"));
        }
        Ok(Self { mean, std })
    }

    /// Point mass (zero variance).
    pub fn point(value: f64) -> Result<Self> {
        Self::new(value, 0.0)
    }
}

/// A nonlinear function with its derivative, for linearization-based propagation.
pub struct DifferentiableFunc {
    /// The function f(x).
    pub f: Box<dyn Fn(f64) -> f64>,
    /// The derivative f'(x).
    pub df: Box<dyn Fn(f64) -> f64>,
}

/// Propagate a Gaussian through a nonlinear function using first-order linearization.
///
/// Approximates: if X ~ N(mu, sigma^2) and Y = f(X), then
/// Y ~ N(f(mu), f'(mu)^2 * sigma^2).
///
/// Accurate when f is nearly linear over the range [mu - 2*sigma, mu + 2*sigma].
pub fn propagate_linearized(input: &Gaussian, func: &DifferentiableFunc) -> Gaussian {
    let output_mean = (func.f)(input.mean);
    let jacobian = (func.df)(input.mean);
    let output_std = (jacobian * input.std).abs();
    Gaussian {
        mean: output_mean,
        std: output_std,
    }
}

/// Propagate a Gaussian through any function using the unscented transform.
///
/// Uses 2n+1 sigma points (here n=1, so 3 points) with tunable spread parameter
/// kappa. The sigma points capture mean and covariance exactly for linear
/// transforms, and approximate well for mildly nonlinear functions.
///
/// `kappa` controls spread: 0.0 is standard, negative values pull sigma
/// points closer to the mean. For Gaussian inputs, kappa = 0.0 is typical.
pub fn propagate_unscented(input: &Gaussian, f: impl Fn(f64) -> f64, kappa: f64) -> Gaussian {
    let n = 1.0; // dimension
    let lambda = kappa; // for n=1, lambda = n + kappa - n = kappa

    let scale = (n + lambda).sqrt();
    let spread = scale * input.std;

    // Sigma points.
    let x0 = input.mean;
    let x1 = input.mean + spread;
    let x2 = input.mean - spread;

    // Propagate sigma points.
    let y0 = f(x0);
    let y1 = f(x1);
    let y2 = f(x2);

    // Weights.
    let w0_mean = lambda / (n + lambda);
    let w0_cov = w0_mean; // can differ in higher-order variants
    let wi = 1.0 / (2.0 * (n + lambda));

    // Output mean.
    let output_mean = w0_mean * y0 + wi * y1 + wi * y2;

    // Output variance.
    let output_var =
        w0_cov * (y0 - output_mean).powi(2) + wi * (y1 - output_mean).powi(2) + wi * (y2 - output_mean).powi(2);

    Gaussian {
        mean: output_mean,
        std: output_var.sqrt(),
    }
}

/// Propagate a multivariate diagonal Gaussian through an element-wise function.
///
/// Each dimension is propagated independently using linearization.
/// Input: vectors of means and stds. Output: vectors of output means and stds.
pub fn propagate_elementwise(
    means: &[f64],
    stds: &[f64],
    func: &DifferentiableFunc,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if means.len() != stds.len() {
        return Err(Error::LengthMismatch(means.len(), stds.len()));
    }

    let mut out_means = Vec::with_capacity(means.len());
    let mut out_stds = Vec::with_capacity(stds.len());

    for (&m, &s) in means.iter().zip(stds.iter()) {
        let g = Gaussian { mean: m, std: s };
        let out = propagate_linearized(&g, func);
        out_means.push(out.mean);
        out_stds.push(out.std);
    }

    Ok((out_means, out_stds))
}

/// Common activation functions with derivatives for distribution propagation.
pub mod activations {
    use super::DifferentiableFunc;

    /// ReLU: f(x) = max(0, x).
    ///
    /// Note: derivative at 0 is set to 0 (subgradient convention).
    pub fn relu() -> DifferentiableFunc {
        DifferentiableFunc {
            f: Box::new(|x| x.max(0.0)),
            df: Box::new(|x| if x > 0.0 { 1.0 } else { 0.0 }),
        }
    }

    /// Sigmoid: f(x) = 1 / (1 + exp(-x)).
    pub fn sigmoid() -> DifferentiableFunc {
        DifferentiableFunc {
            f: Box::new(|x| 1.0 / (1.0 + (-x).exp())),
            df: Box::new(|x| {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }),
        }
    }

    /// Tanh: f(x) = tanh(x).
    pub fn tanh_act() -> DifferentiableFunc {
        DifferentiableFunc {
            f: Box::new(|x| x.tanh()),
            df: Box::new(|x| 1.0 - x.tanh().powi(2)),
        }
    }

    /// Softplus: f(x) = ln(1 + exp(x)). Smooth approximation to ReLU.
    pub fn softplus() -> DifferentiableFunc {
        DifferentiableFunc {
            f: Box::new(|x| (1.0 + x.exp()).ln()),
            df: Box::new(|x| 1.0 / (1.0 + (-x).exp())),
        }
    }

    /// Square: f(x) = x^2.
    pub fn square() -> DifferentiableFunc {
        DifferentiableFunc {
            f: Box::new(|x| x * x),
            df: Box::new(|x| 2.0 * x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_propagation() {
        let g = Gaussian::new(1.0, 0.5).unwrap();
        let id = DifferentiableFunc {
            f: Box::new(|x| x),
            df: Box::new(|_| 1.0),
        };
        let out = propagate_linearized(&g, &id);
        assert!((out.mean - 1.0).abs() < 1e-12);
        assert!((out.std - 0.5).abs() < 1e-12);
    }

    #[test]
    fn linear_scaling() {
        let g = Gaussian::new(2.0, 1.0).unwrap();
        let scale3 = DifferentiableFunc {
            f: Box::new(|x| 3.0 * x),
            df: Box::new(|_| 3.0),
        };
        let out = propagate_linearized(&g, &scale3);
        assert!((out.mean - 6.0).abs() < 1e-12);
        assert!((out.std - 3.0).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_narrows_variance() {
        let g = Gaussian::new(0.0, 1.0).unwrap();
        let sig = activations::sigmoid();
        let out = propagate_linearized(&g, &sig);
        // sigmoid(0) = 0.5, sigmoid'(0) = 0.25
        assert!((out.mean - 0.5).abs() < 1e-12);
        assert!((out.std - 0.25).abs() < 1e-12);
    }

    #[test]
    fn relu_at_positive_mean() {
        let g = Gaussian::new(3.0, 0.5).unwrap();
        let relu = activations::relu();
        let out = propagate_linearized(&g, &relu);
        // ReLU(3) = 3, ReLU'(3) = 1
        assert!((out.mean - 3.0).abs() < 1e-12);
        assert!((out.std - 0.5).abs() < 1e-12);
    }

    #[test]
    fn unscented_linear_exact() {
        let g = Gaussian::new(2.0, 1.5).unwrap();
        let out = propagate_unscented(&g, |x| 3.0 * x + 1.0, 1.0);
        assert!((out.mean - 7.0).abs() < 1e-10);
        assert!((out.std - 4.5).abs() < 1e-10);
    }

    #[test]
    fn unscented_vs_linearized_square() {
        // For x^2 with X ~ N(mu, sigma^2):
        // E[X^2] = mu^2 + sigma^2 (exact)
        // Linearized: f(mu) = mu^2 (misses sigma^2 correction)
        let g = Gaussian::new(0.0, 1.0).unwrap();

        let lin = propagate_linearized(&g, &activations::square());
        let uns = propagate_unscented(&g, |x| x * x, 1.0);

        // Linearized mean = 0^2 = 0 (wrong for zero-mean case)
        assert!((lin.mean - 0.0).abs() < 1e-12);
        // Unscented mean should be close to 0^2 + 1^2 = 1.0
        assert!((uns.mean - 1.0).abs() < 0.1, "unscented mean = {}", uns.mean);
    }

    #[test]
    fn elementwise_propagation() {
        let means = [1.0, 2.0, 3.0];
        let stds = [0.1, 0.2, 0.3];
        let scale = DifferentiableFunc {
            f: Box::new(|x| 2.0 * x),
            df: Box::new(|_| 2.0),
        };
        let (out_m, out_s) = propagate_elementwise(&means, &stds, &scale).unwrap();
        assert!((out_m[0] - 2.0).abs() < 1e-12);
        assert!((out_s[1] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn point_mass_propagation() {
        let g = Gaussian::point(5.0).unwrap();
        let sig = activations::sigmoid();
        let out = propagate_linearized(&g, &sig);
        assert!(out.std.abs() < 1e-15, "point mass should have zero std");
    }

    #[test]
    fn length_mismatch_error() {
        let scale = DifferentiableFunc {
            f: Box::new(|x| x),
            df: Box::new(|_| 1.0),
        };
        let r = propagate_elementwise(&[1.0, 2.0], &[1.0], &scale);
        assert!(r.is_err());
    }
}
