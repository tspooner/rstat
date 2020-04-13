use crate::Distribution;
use spaces::Space;
use failure::Fail;

#[derive(Debug, Fail)]
pub enum FittingError {
    #[fail(display = "Insufficient data samples provided ({}).", _0)]
    InsufficientData(usize),
}

/// Trait for [distributions](trait.Distribution.html) implementing maximum likelihood estimation.
pub trait MLE: Distribution where Self: Sized {
    fn fit_mle(samples: &[<Self::Support as Space>::Value]) -> Result<Self, failure::Error>;
}
