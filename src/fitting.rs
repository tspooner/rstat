use crate::{Distribution, Sample};
use failure::Fail;

#[derive(Debug, Fail)]
pub enum FittingError {
    #[fail(display = "Insufficient data samples provided ({}).", _0)]
    InsufficientSamples(usize),
}

/// Trait for [distributions](trait.Distribution.html) with a well-defined
/// likelihood.
///
/// # Note
/// All observations are treated as independent.
pub trait Likelihood: Distribution {
    /// Computes the likelihood \\(\\mathcal{L}(\\theta \\mid \\bm{x})\\).
    fn likelihood(&self, samples: &[Sample<Self>]) -> f64 { self.log_likelihood(samples).exp() }

    /// Computes the log-likelihood \\(l(\\theta \\mid \\bm{x})\\).
    fn log_likelihood(&self, samples: &[Sample<Self>]) -> f64 { self.likelihood(samples).ln() }
}

/// Trait for [distributions](trait.Distribution.html) with a well-defined score
/// function.
///
/// # Note
/// All observations are treated as independent.
pub trait Score: Likelihood {
    type Grad;

    /// Computes the score function \\(s(\\theta) = \\frac{\\partial l(\\theta
    /// \mid \\bm{x})}{\\partial \\theta}\\).
    fn score(&self, samples: &[Sample<Self>]) -> Self::Grad;
}

/// Trait for [distributions](trait.Distribution.html) implementing maximum
/// likelihood estimation.
///
/// # Note
/// All observations are treated as independent.
pub trait MLE: Likelihood {
    fn fit_mle(samples: &[Sample<Self>]) -> Result<Self, failure::Error>;
}
