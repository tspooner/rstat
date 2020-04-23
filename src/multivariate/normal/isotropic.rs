use super::{Covariance, Loc, Normal, Params};
use crate::{
    consts::PI_2,
    linalg::{Matrix, Vector},
    metrics::Mahalanobis,
    params::{
        constraints::{self, Constraint, Constraints, UnsatisfiedConstraintError as UCE},
        Param,
    },
    statistics::MultivariateMoments,
    ContinuousDistribution,
    Distribution,
    Probability,
};
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{real::Reals, ProductSpace};

/// Isotropic covariance matrix parameter \\(\\sigma^2\\bm{I}\\).
pub type IsotropicCovariance = Covariance<f64>;

impl Covariance<f64> {
    /// Construct an isotropic covariance matrix parameter
    /// \\((\\sigma^2\\bm{I}\\).
    ///
    /// # Constraints
    /// 1. The variance term is positive real.
    pub fn isotropic(value: f64) -> Result<Self, UCE<f64>> {
        Ok(Covariance(assert_constraint!(value+)?))
    }
}

impl Param for Covariance<f64> {
    type Value = f64;

    fn value(&self) -> &Self::Value { &self.0 }

    fn into_value(self) -> Self::Value { self.0 }

    fn constraints() -> Constraints<Self::Value> { vec![Box::new(constraints::Positive)] }
}

/// Parameter set for [IsotropicNormal](type.IsotropicNormal.html).
pub type IsotropicNormalParams = Params<f64>;

impl IsotropicNormalParams {
    /// Construct a parameter set for
    /// [IsotropicNormal](type.DiagonalNormal.html): \\(\\langle\\bm{\\mu},
    /// \\sigma^2\\bm{I}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. All variance terms are positive real.
    pub fn isotropic<M: Into<Vector<f64>>>(mu: M, sigma: f64) -> Result<Self, failure::Error> {
        let mu = Loc::new(mu.into())?;
        let sigma = Covariance::isotropic(sigma)?;

        Ok(Params { mu, sigma })
    }

    /// Construct a parameter set for
    /// [IsotropicNormal](type.DiagonalNormal.html): \\(\\langle\\bm{\\mu},
    /// \\sigma^2\\bm{I}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The dimensionality is a positive integer.
    /// 2. The variance term is positive real.
    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> Result<Self, failure::Error> {
        let n = constraints::Natural.check(n)?;

        Self::isotropic(Vector::from_elem((n,), mu), sigma)
    }

    /// Construct a parameter set for
    /// [IsotropicNormal](type.DiagonalNormal.html): \\(\\langle\\bm{\\mu},
    /// \\sigma^2\\bm{I}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The dimensionality is a positive integer.
    pub fn standard(n: usize) -> Result<Self, failure::Error> { Self::homogeneous(n, 0.0, 1.0) }
}

/// Multivariate Normal distribution with mean \\(\\bm{\\mu}\\) and isotropic
/// covariance matrix \\(\\sigma^2\\bm{I}\\).
pub type IsotropicNormal = Normal<f64>;

impl IsotropicNormal {
    /// Construct an \\(n\\)-dimensional
    /// [IsotropicNormal](type.IsotropicNormal.html) distribution
    /// with mean \\(\\bm{\\mu}\\) and variance \\(\\sigma^2\\bm{I}\\).
    ///
    /// # Constraints
    /// 1. The variance term is positive real.
    ///
    /// # Examples
    /// ```
    /// # use rstat::multivariate::normal::Normal;
    /// let dist = Normal::isotropic(vec![0.0, 1.0].into(), 1.0);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn isotropic<M: Into<Vector<f64>>>(mu: M, sigma: f64) -> Result<Self, failure::Error> {
        let params = Params::isotropic(mu, sigma)?;

        Ok(Normal::isotropic_unchecked(params.mu.0, params.sigma.0))
    }

    /// Construct an \\(n\\)-dimensional
    /// [IsotropicNormal](type.IsotropicNormal.html) distribution
    /// with mean \\(\\bm{\\mu}\\) and variance \\(\\sigma^2\\bm{I}\\), without
    /// checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use rstat::multivariate::normal::{Normal, IsotropicNormal};
    /// let dist: IsotropicNormal = Normal::isotropic_unchecked(vec![0.0, 1.0].into(), 1.0);
    /// ```
    pub fn isotropic_unchecked<M: Into<Vector<f64>>>(mu: M, sigma: f64) -> Self {
        let params = Params {
            mu: Loc(mu.into()),
            sigma: Covariance(sigma),
        };

        Normal {
            sigma_lt: params.sigma.0,
            sigma_det: params.sigma.0.powi(params.mu.0.len() as i32),
            precision: 1.0 / params.sigma.0,
            params,
        }
    }

    /// Construct an \\(n\\)-dimensional
    /// [IsotropicNormal](type.IsotropicNormal.html) distribution
    /// with mean \\(\\mu\\) and variance \\(\\sigma^2\\) in each dimension.
    ///
    /// # Constraints
    /// 1. The dimensionality is a positive integer.
    ///
    /// # Examples
    /// ```
    /// # use rstat::multivariate::normal::Normal;
    /// let dist = Normal::homogeneous(2, 0.0, 1.0);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> Result<Self, failure::Error> {
        let params = Params::homogeneous(n, mu, sigma)?;

        Ok(Normal::isotropic_unchecked(params.mu.0, params.sigma.0))
    }

    /// Construct an \\(n\\)-dimensional
    /// [IsotropicNormal](type.IsotropicNormal.html) distribution
    /// with mean \\(\\mu\\) and variance \\(\\sigma^2\\) in each dimension,
    /// without checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use rstat::multivariate::normal::{Normal, IsotropicNormal};
    /// let dist: IsotropicNormal = Normal::homogeneous_unchecked(2, 0.0, 1.0);
    /// ```
    pub fn homogeneous_unchecked(n: usize, mu: f64, sigma: f64) -> Self {
        Self::isotropic_unchecked(Vector::from_elem((n,), mu), sigma)
    }

    /// Construct an \\(n\\)-dimensional
    /// [IsotropicNormal](type.IsotropicNormal.html) distribution
    /// with mean 0 and unit variance \\(\\sigma^2\\) in each dimension.
    ///
    /// # Constraints
    /// 1. The dimensionality is a positive integer.
    ///
    /// # Examples
    /// ```
    /// # use rstat::multivariate::normal::Normal;
    /// let dist = Normal::standard(2);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn standard(n: usize) -> Result<Self, failure::Error> {
        let params = Params::standard(n)?;

        Ok(Normal::isotropic_unchecked(params.mu.0, params.sigma.0))
    }

    /// Construct an \\(n\\)-dimensional
    /// [IsotropicNormal](type.IsotropicNormal.html) distribution
    /// with mean 0 and unit variance \\(\\sigma^2\\) in each dimension, without
    /// checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use rstat::multivariate::normal::{Normal, IsotropicNormal};
    /// let dist: IsotropicNormal = Normal::standard_unchecked(2);
    /// ```
    pub fn standard_unchecked(n: usize) -> Self { Normal::homogeneous_unchecked(n, 0.0, 1.0) }
}

impl From<IsotropicNormalParams> for IsotropicNormal {
    fn from(params: IsotropicNormalParams) -> IsotropicNormal {
        IsotropicNormal::isotropic_unchecked(params.mu.0, params.sigma.0)
    }
}

impl Distribution for IsotropicNormal {
    type Support = ProductSpace<Reals>;
    type Params = IsotropicNormalParams;

    fn support(&self) -> ProductSpace<Reals> { self.params.mu.0.iter().map(|_| Reals).collect() }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.params
            .mu
            .0
            .iter()
            .map(|m| {
                let z: f64 = rng.sample(RandSN);

                m + z * self.sigma_lt
            })
            .collect()
    }
}

impl ContinuousDistribution for IsotropicNormal {
    fn pdf(&self, x: &Vec<f64>) -> f64 {
        let z = self.d_mahalanobis_squared(x);
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for IsotropicNormal {
    fn mean(&self) -> Vector<f64> { self.params.mu.0.clone() }

    fn covariance(&self) -> Matrix<f64> {
        let n = self.params.mu.0.len();
        let mut cov = Matrix::zeros((n, n));

        cov.diag_mut().fill(self.params.sigma.0);

        cov
    }

    fn variance(&self) -> Vector<f64> {
        Vector::from_elem((self.params.mu.0.len(),), self.params.sigma.0)
    }
}

impl Mahalanobis for IsotropicNormal {
    fn d_mahalanobis_squared(&self, x: &Vec<f64>) -> f64 {
        x.into_iter()
            .zip(self.params.mu.0.iter())
            .map(|(x, y)| (x - y) * self.precision * self.precision)
            .sum()
    }
}
