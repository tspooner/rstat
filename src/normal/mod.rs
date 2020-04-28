//! A generalised implementation of the multivariate Normal distribution.
#![allow(non_snake_case)]
use crate::{
    params::{
        Param, constraints::{
            self, Constraint, Constraints,
            UnsatisfiedConstraintError as UCE,
        }
    },
    linalg::{Vector, Matrix},
};

pub use crate::params::Loc;

/// Covariance matrix parameter \\(\\bm{\\Sigma}\\).
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Covariance<S>(pub S);

/// Dense covariance matrix parameter \\(\\bm{\\Sigma}\\).
pub type MatrixCovariance = Covariance<Matrix<f64>>;

impl MatrixCovariance {
    /// Construct an \\(n\\)-dimensional covariance matrix parameter \\(\\bm{\\Sigma}\\).
    ///
    /// # Arguments
    /// * `Sigma` - the covariance matrix \\(\\bm{\\Sigma}\\).
    ///
    /// # Constraints
    /// 1. The covariance matrix is square.
    /// 2. The covariance matrix is positive semi-definite.
    pub fn matrix(Sigma: Matrix<f64>) -> Result<Self, failure::Error> {
        let c = constraints::And((constraints::Square, constraints::All(constraints::Positive)));

        Ok(Covariance(c.check(Sigma)?))
    }
}

impl Param for MatrixCovariance {
    type Value = Matrix<f64>;

    fn value(&self) -> &Self::Value { &self.0 }

    fn into_value(self) -> Self::Value { self.0 }

    fn constraints() -> Constraints<Self::Value> {
        let c = constraints::All(constraints::Positive);

        vec![Box::new(c)]
    }
}

/// Diagonal covariance matrix parameter \\(\\mathrm{diag}(\\sigma_1^2, \\ldots, \\sigma_n^2)\\).
pub type DiagonalCovariance = Covariance<Vector<f64>>;

impl DiagonalCovariance {
    /// Construct an \\(n\\)-dimensional diagonal covariance matrix parameter
    /// \\(\\mathrm{diag}(\\sigma_1^2, \\ldots, \\sigma_n^2)\\).
    ///
    /// # Arguments
    /// * `sigma2_diag` - the covariance matrix diagonal entries \\(\\langle\\sigma_1^2, \\ldots,
    ///     \\sigma_n^2\\rangle\\).
    ///
    /// # Constraints
    /// 1. All variance terms are positive real.
    pub fn diagonal(sigma2_diag: Vector<f64>) -> Result<Self, failure::Error> {
        let c = constraints::All(constraints::Positive);

        Ok(Covariance(c.check(sigma2_diag)?))
    }
}

impl Param for Covariance<Vector<f64>> {
    type Value = Vector<f64>;

    fn value(&self) -> &Self::Value { &self.0 }

    fn into_value(self) -> Self::Value { self.0 }

    fn constraints() -> Constraints<Self::Value> {
        let c = constraints::All(constraints::Positive);

        vec![Box::new(c)]
    }
}

/// Bivariate covariance matrix parameter with variance terms \\(\\sigma_1^2\\) and
/// \\(\\sigma_2^2\\), and correlation coefficient \\(\\rho\\).
pub type BivariateCovariance = Covariance<([f64; 2], f64)>;

impl BivariateCovariance {
    /// Construct an bivariate covariance matrix parameter with variance terms \\(\\sigma_1^2\\)
    /// and \\(\\sigma_2^2\\), and correlation coefficient \\(\\rho\\).
    ///
    /// # Arguments
    /// * `sigma_diag` - the covariance matrix diagonal entries \\(\\sigma_1^2\\) and
    ///     \\(\\sigma_2^2\\).
    /// * `rho` - the correlation coefficient \\(\\rho\\).
    ///
    /// # Constraints
    /// 1. The variance terms are positive real.
    /// 2. The correlation coefficient is in the interval \\([-1, 1]\\).
    pub fn bivariate(sigma2_diag: [f64; 2], rho: f64) -> Result<Self, failure::Error> {
        let c_sigma = constraints::All(constraints::Positive);
        let c_rho = constraints::Interval { lb: -1.0, ub: 1.0, };

        Ok(Covariance((c_sigma.check(sigma2_diag)?, c_rho.check(rho)?)))
    }
}

impl Param for BivariateCovariance {
    type Value = ([f64; 2], f64);

    fn value(&self) -> &Self::Value { &self.0 }

    fn into_value(self) -> Self::Value { self.0 }

    fn constraints() -> Constraints<Self::Value> { todo!() }
}

/// Diagonal covariance matrix parameter \\(\\mathrm{diag}(\\sigma_1^2, \\sigma_2^2)\\) in
/// 2-dimensions.
pub type PairedCovariance = Covariance<[f64; 2]>;

impl PairedCovariance {
    /// Construct a diagonal covariance matrix parameter \\(\\mathrm{diag}(\\sigma_1^2,
    /// \\sigma_2^2)\\) in 2-dimensions.
    ///
    /// # Arguments
    /// * `sigma2_diag` - the covariance matrix diagonal entries \\(\\sigma_1^2\\) and
    ///     \\(\\sigma_2^2\\).
    ///
    /// # Constraints
    /// 1. The variance terms are positive real.
    /// 2. The correlation coefficient is in the interval \\([-1, 1]\\).
    pub fn paired(sigma2_diag: [f64; 2]) -> Result<Self, failure::Error> {
        let c = constraints::All(constraints::Positive);

        Ok(Covariance(c.check(sigma2_diag)?))
    }
}

impl Param for PairedCovariance {
    type Value = [f64; 2];

    fn value(&self) -> &Self::Value { &self.0 }

    fn into_value(self) -> Self::Value { self.0 }

    fn constraints() -> Constraints<Self::Value> { todo!() }
}

/// Isotropic covariance matrix parameter \\(\\sigma^2\\bm{I}\\).
pub type IsotropicCovariance = Covariance<f64>;

impl Covariance<f64> {
    /// Construct an isotropic covariance matrix parameter \\((\\sigma^2\\bm{I}\\).
    ///
    /// # Arguments
    /// * `sigma2` - the covariance matrix diagonal entry \\(\\sigma^2\\).
    ///
    /// # Constraints
    /// 1. The variance term is positive real.
    pub fn isotropic(sigma2: f64) -> Result<Self, UCE<f64>> {
        Ok(Covariance(assert_constraint!(sigma2+)?))
    }
}

impl Param for Covariance<f64> {
    type Value = f64;

    fn value(&self) -> &Self::Value { &self.0 }

    fn into_value(self) -> Self::Value { self.0 }

    fn constraints() -> Constraints<Self::Value> { vec![Box::new(constraints::Positive)] }
}

/// Parameter set for Normal distributions.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Params<M, S> {
    /// The mean parameter \\(\\bm{\\mu}\\).
    pub mu: Loc<M>,

    /// The covariance parameter \\(\\bm{\\Sigma}\\).
    pub Sigma: Covariance<S>,
}

/// Gradient of parameters of a Normal distribution.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Grad<M, S> {
    /// Gradient of the mean parameter \\(\\bm{\\mu}\\).
    pub mu: M,

    /// Gradient of the covariance parameter \\(\\bm{\\Sigma}\\).
    pub Sigma: S,
}

mod univariate;
pub use self::univariate::*;

mod paired;
pub use self::paired::*;

mod bivariate;
pub use self::bivariate::*;

mod multivariate;
pub use self::multivariate::*;
