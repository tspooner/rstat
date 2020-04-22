use super::{Covariance, Loc, Normal, Params};
use crate::{
    consts::PI_2,
    linalg::{cholesky, inverse_lt, Matrix, Vector},
    metrics::Mahalanobis,
    params::{
        constraints::{self, Constraint, Constraints},
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

impl Covariance {
    /// Construct an \\(n\\)-dimensional covariance matrix parameter
    /// \\(\\bm{\\Sigma}\\).
    ///
    /// # Constraints
    /// 1. The covariance matrix is not square.
    /// 2. The covariance matrix is not positive semi-definite.
    pub fn new(value: Matrix<f64>) -> Result<Self, failure::Error> {
        let c = constraints::And((constraints::Square, constraints::All(constraints::Positive)));

        Ok(Covariance(c.check(value)?))
    }
}

impl Param for Covariance {
    type Value = Matrix<f64>;

    fn value(&self) -> &Self::Value { &self.0 }

    fn into_value(self) -> Self::Value { self.0 }

    fn constraints() -> Constraints<Self::Value> {
        let c = constraints::All(constraints::Positive);

        vec![Box::new(c)]
    }
}

impl Params {
    /// Construct a parameter set for [Normal](struct.Normal.html):
    /// \\(\\langle\\bm{\\mu}, \\bm{\\Sigma}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The covariance matrix is not square.
    /// 2. The covariance matrix is not positive semi-definite.
    pub fn new(mu: Vector<f64>, sigma: Matrix<f64>) -> Result<Self, failure::Error> {
        let mu = Loc::new(mu)?;
        let sigma = Covariance::new(sigma)?;

        let n_mu = mu.0.len();
        let n_sigma = sigma.0.nrows();

        assert_constraint!(n_mu == n_sigma)?;

        Ok(Params { mu, sigma })
    }
}

impl Normal {
    /// Construct an \\(n\\)-dimensional [Normal](struct.Normal.html)
    /// distribution with mean \\(\\bm{\\mu}\\) and covariance matrix
    /// \\(\\bm{\\Sigma}\\).
    ///
    /// # Constraints
    /// 1. The covariance matrix is not square.
    /// 2. The covariance matrix is not positive semi-definite.
    ///
    /// # Examples
    /// ```
    /// # use ndarray::arr2;
    /// # use rstat::multivariate::normal::Normal;
    /// let cov = arr2(&[
    ///     [0.5, 2.0],
    ///     [1.0, 1.0]
    /// ]);
    /// let dist = Normal::new(vec![0.0, 1.0].into(), cov);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn new(mu: Vector<f64>, sigma: Matrix<f64>) -> Result<Self, failure::Error> {
        let params = Params::new(mu, sigma)?;

        Ok(Normal::new_unchecked(params.mu.0, params.sigma.0))
    }

    /// Construct an \\(n\\)-dimensional [Normal](struct.Normal.html)
    /// distribution with mean \\(\\bm{\\mu}\\) and covariance matrix
    /// \\(\\bm{\\Sigma}\\), without checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use ndarray::arr2;
    /// # use rstat::multivariate::normal::Normal;
    /// let cov = arr2(&[
    ///     [0.5, 2.0],
    ///     [1.0, 1.0]
    /// ]);
    /// let dist: Normal = Normal::new_unchecked(vec![0.0, 1.0].into(), cov);
    /// ```
    pub fn new_unchecked(mu: Vector<f64>, sigma: Matrix<f64>) -> Self {
        let params = Params {
            mu: Loc(mu),
            sigma: Covariance(sigma),
        };

        #[cfg(feature = "ndarray-linalg")]
        {
            use ndarray_linalg::{
                cholesky::{Cholesky, UPLO},
                Inverse,
            };

            let sigma_lt = params.sigma.0.cholesky(UPLO::Lower).expect(
                "Covariance matrix must be positive-definite to apply Cholesky decomposition.",
            );
            let sigma_lt_inv = sigma_lt
                .inv()
                .expect("Covariance matrix must be positive-definite to compute an inverse.");
            let precision = &sigma_lt_inv * &sigma_lt_inv.t();

            Normal {
                params,
                sigma_det: sigma_lt.diag().product(),
                sigma_lt,
                precision,
            }
        }

        #[cfg(not(feature = "ndarray-linalg"))]
        {
            let sigma_lt = unsafe { cholesky(&params.sigma.0) };
            let sigma_det = sigma_lt.diag().product();

            let sigma_lt_inv = unsafe { inverse_lt(&sigma_lt) };
            let precision = &sigma_lt_inv * &sigma_lt_inv.t();

            Normal {
                params,
                sigma_lt,
                sigma_det,
                precision,
            }
        }
    }
}

impl From<Params> for Normal {
    fn from(params: Params) -> Normal { Normal::new_unchecked(params.mu.0, params.sigma.0) }
}

impl Distribution for Normal {
    type Support = ProductSpace<Reals>;
    type Params = Params;

    fn support(&self) -> ProductSpace<Reals> { self.params.mu.0.iter().map(|_| Reals).collect() }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let z = Vector::from_shape_fn((self.params.mu.0.len(),), |_| rng.sample(RandSN));
        let az = self.sigma_lt.dot(&z);

        (az + &self.params.mu.0).into_raw_vec()
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: &Vec<f64>) -> f64 {
        let z = self.d_mahalanobis_squared(x);
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for Normal {
    fn mean(&self) -> Vector<f64> { self.params.mu.0.clone() }

    fn covariance(&self) -> Matrix<f64> { self.params.sigma.0.clone() }

    fn variance(&self) -> Vector<f64> { self.params.sigma.0.diag().to_owned() }
}

impl Mahalanobis for Normal {
    fn d_mahalanobis_squared(&self, x: &Vec<f64>) -> f64 {
        let diff: Vector<f64> = x
            .into_iter()
            .zip(self.params.mu.0.iter())
            .map(|(x, y)| x - y)
            .collect();

        diff.dot(&self.precision).dot(&diff)
    }
}
