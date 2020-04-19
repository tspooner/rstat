//! A generalised implementation of the multivariate Normal distribution.
use crate::linalg::{Matrix, Vector};
use std::fmt;

/// Parameter set for [Normal](struct.Normal.html).
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Params<S = Matrix<f64>> {
    pub mu: Loc<Vector<f64>>,
    pub sigma: Covariance<S>,
}

pub use crate::params::Loc;

/// Covariance matrix parameter \\(\\bm{\\Sigma}\\).
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Covariance<T = Matrix<f64>>(pub T);

/// Multivariate Normal distribution with mean \\(\\bm{\\mu}\\) covariance
/// matrix \\(\\bm{\\Sigma}\\).
#[derive(Debug, Clone)]
pub struct Normal<S = Matrix<f64>> {
    pub(crate) params: Params<S>,

    sigma_det: f64,
    sigma_lt: S,
    precision: S,
}

impl<S: fmt::Display> fmt::Display for Normal<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.params.mu.0, self.params.sigma.0)
    }
}

mod isotropic;
pub use self::isotropic::*;

mod diagonal;
pub use self::diagonal::*;

mod general;
pub use self::general::*;

#[cfg(test)]
mod tests {
    use super::Normal;
    use crate::ContinuousDistribution;
    use failure::Error;

    #[test]
    fn test_pdf() -> Result<(), Error> {
        let m = Normal::standard(5)?;
        let prob = m.pdf(&vec![0.0; 5]);

        assert!((prob - 0.010105326013811651).abs() < 1e-7);

        Ok(())
    }
}
