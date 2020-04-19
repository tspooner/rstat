use super::{Covariance, Loc, Normal, Params};
use crate::{
    consts::PI_2,
    linalg::Vector,
    params::constraints::{self, Constraint, Constraints},
    prelude::*,
};
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{real::Reals, ProductSpace};

/// Parameter set for [DiagonalNormal](type.DiagonalNormal.html).
pub type DiagonalNormalParams = Params<Vector<f64>>;

impl DiagonalNormalParams {
    /// Construct a parameter set for
    /// [DiagonalNormal](type.DiagonalNormal.html): \\(\\langle\\bm{\\mu},
    /// \\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The mean and variance vectors are the same length.
    /// 2. All variance terms are positive real.
    pub fn diagonal(mu: Vector<f64>, sigma: Vector<f64>) -> Result<Self, failure::Error> {
        let mu = Loc::new(mu)?;
        let sigma = Covariance::diagonal(sigma)?;

        let n_mu = mu.0.len();
        let n_sigma = mu.0.len();

        assert_constraint!(n_mu == n_sigma)?;

        Ok(DiagonalNormalParams { mu, sigma })
    }
}

/// Diagonal covariance matrix parameter \\(\\mathrm{diag}(\\sigma_1, \\ldots,
/// \\sigma_n)\\).
pub type DiagonalCovariance = Covariance<Vector<f64>>;

impl DiagonalCovariance {
    /// Construct an \\(n\\)-dimensional diagonal covariance matrix parameter
    /// \\((\\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\).
    ///
    /// # Constraints
    /// 1. All variance terms are positive real.
    pub fn diagonal(value: Vector<f64>) -> Result<Self, failure::Error> {
        let c = constraints::All(constraints::Positive);

        Ok(Covariance(c.check(value)?))
    }
}

impl Param for Covariance<Vector<f64>> {
    type Value = Vector<f64>;

    fn value(&self) -> &Self::Value { &self.0 }

    fn constraints() -> Constraints<Self::Value> {
        let c = constraints::All(constraints::Positive);

        vec![Box::new(c)]
    }
}

/// Multivariate Normal distribution with mean \\(\\bm{\\mu}\\) and diagonal
/// covariance matrix \\(\\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\).
pub type DiagonalNormal = Normal<Vector<f64>>;

impl DiagonalNormal {
    /// Construct an \\(n\\)-dimensional
    /// [DiagonalNormal](type.DiagonalNormal.html) distribution with mean \\
    /// (\\bm{\\mu}\\) and diagonal covariance matrix
    /// \\((\\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\).
    ///
    /// # Constraints
    /// 1. All entries in the covariance are positive real.
    ///
    /// # Examples
    /// ```
    /// # use rstat::multivariate::normal::Normal;
    /// let dist = Normal::diagonal(vec![0.0, 1.0].into(), vec![0.5, 2.0].into());
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn diagonal(mu: Vector<f64>, sigma: Vector<f64>) -> Result<Self, failure::Error> {
        let params = Params::diagonal(mu, sigma)?;

        Ok(Normal::diagonal_unchecked(params.mu.0, params.sigma.0))
    }

    /// Construct an \\(n\\)-dimensional
    /// [DiagonalNormal](type.DiagonalNormal.html) distribution with mean \\
    /// (\\bm{\\mu}\\) and diagonal covariance matrix
    /// \\((\\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\), without
    /// checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use rstat::multivariate::normal::{Normal, DiagonalNormal};
    /// let dist: DiagonalNormal =
    ///     Normal::diagonal_unchecked(vec![0.0, 1.0].into(), vec![0.5, 2.0].into());
    /// ```
    pub fn diagonal_unchecked(mu: Vector<f64>, sigma: Vector<f64>) -> Self {
        let params = Params {
            mu: Loc(mu),
            sigma: Covariance(sigma),
        };

        Normal {
            sigma_lt: params.sigma.0.clone(),
            sigma_det: params.sigma.0.iter().product(),
            precision: params.sigma.0.iter().cloned().map(|v| 1.0 / v).collect(),
            params,
        }
    }
}

impl From<DiagonalNormalParams> for DiagonalNormal {
    fn from(params: DiagonalNormalParams) -> DiagonalNormal {
        DiagonalNormal::diagonal_unchecked(params.mu.0, params.sigma.0)
    }
}

impl Distribution for DiagonalNormal {
    type Support = ProductSpace<Reals>;
    type Params = Params<Vector<f64>>;

    fn support(&self) -> ProductSpace<Reals> { self.params.mu.0.iter().map(|_| Reals).collect() }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.params
            .mu
            .0
            .iter()
            .zip(self.sigma_lt.iter())
            .map(|(m, s)| {
                let z: f64 = rng.sample(RandSN);

                m + z * s
            })
            .collect()
    }
}

impl ContinuousDistribution for DiagonalNormal {
    fn pdf(&self, x: &Vec<f64>) -> f64 {
        let z = self.d_mahalanobis_squared(x);
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for DiagonalNormal {
    fn mean(&self) -> Vector<f64> { self.params.mu.0.clone() }

    fn covariance(&self) -> Matrix<f64> { Matrix::from_diag(&self.params.sigma.0) }

    fn variance(&self) -> Vector<f64> { self.params.sigma.0.clone() }
}

impl MahalanobisDistance for DiagonalNormal {
    fn d_mahalanobis_squared(&self, x: &Vec<f64>) -> f64 {
        x.into_iter()
            .zip(self.params.mu.0.iter())
            .map(|(x, y)| x - y)
            .zip(self.precision.iter())
            .map(|(d, si)| d * si * si)
            .sum()
    }
}
