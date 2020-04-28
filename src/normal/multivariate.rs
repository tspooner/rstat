use super::{Covariance, Loc, Params, Grad};
use crate::{
    consts::PI_2,
    linalg::{cholesky, inverse_lt, Matrix, Vector},
    metrics::Mahalanobis,
    params::constraints::{Natural, Constraint},
    statistics::{MultivariateMoments, Modes, ShannonEntropy},
    ContinuousDistribution,
    Distribution,
    Probability,
};
// use ndarray::ArrayView1;
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{real::Reals, ProductSpace};
use std::fmt;

/// Multivariate Normal distribution with mean \\(\\bm{\\mu}\\) and  covariance
/// matrix \\(\\bm{\\Sigma}\\).
#[derive(Debug, Clone)]
pub struct MvNormal<M = Vector<f64>, S = Matrix<f64>> {
    pub(crate) params: Params<M, S>,

    Sigma_det: f64,
    Sigma_lt: S,
    precision: S,
}

impl<S> Modes for MvNormal<Vector<f64>, S>
where
    MvNormal<Vector<f64>, S>: Distribution<Support = ProductSpace<Reals>>,
{
    fn modes(&self) -> Vec<Vec<f64>> { vec![self.params.mu.0.to_vec()] }
}

impl<S> ShannonEntropy for MvNormal<Vector<f64>, S>
where
    MvNormal<Vector<f64>, S>: Distribution<Support = ProductSpace<Reals>>,
{
    fn shannon_entropy(&self) -> f64 {
        let k = self.params.mu.len() as f64;

        0.5 * (k * (1.0 + PI_2.ln()) + self.Sigma_det.ln())
    }
}

impl<M: fmt::Display, S: fmt::Display> fmt::Display for MvNormal<M, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.params.mu.0, self.params.Sigma.0)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// General
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type MvNormalParams = Params<Vector<f64>, Matrix<f64>>;

impl MvNormalParams {
    /// Construct a parameter set for [MvNormal](struct.MvNormal.html):
    /// \\(\\langle\\bm{\\mu}, \\bm{\\Sigma}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The covariance matrix is not square.
    /// 2. The covariance matrix is not positive semi-definite.
    pub fn multivariate<M, S>(mu: M, Sigma: S) -> Result<Self, failure::Error>
    where
        M: Into<Vector<f64>>,
        S: Into<Matrix<f64>>,
    {
        let mu = Loc::new(mu.into())?;
        let Sigma = Covariance::matrix(Sigma.into())?;

        let n_mu = mu.0.len();
        let n_sigma = Sigma.0.nrows();

        assert_constraint!(n_mu == n_sigma)?;

        Ok(Params { mu, Sigma, })
    }
}

pub type MvNormalGrad = Grad<Vector<f64>, Matrix<f64>>;

impl std::ops::Mul<f64> for MvNormalGrad {
    type Output = MvNormalGrad;

    fn mul(self, sf: f64) -> MvNormalGrad {
        MvNormalGrad {
            mu: self.mu * sf,
            Sigma: self.Sigma * sf,
        }
    }
}

impl MvNormal<Vector<f64>> {
    /// Construct an \\(n\\)-dimensional [MvNormal](struct.MvNormal.html)
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
    /// # use rstat::normal::MvNormal;
    /// let cov = arr2(&[
    ///     [0.5, 2.0],
    ///     [1.0, 1.0]
    /// ]);
    /// let dist = MvNormal::new(vec![0.0, 1.0], cov);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn new<M, S>(mu: M, Sigma: S) -> Result<Self, failure::Error>
    where
        M: Into<Vector<f64>>,
        S: Into<Matrix<f64>>,
    {
        let params = Params::multivariate(mu, Sigma)?;

        Ok(MvNormal::new_unchecked(params.mu.0, params.Sigma.0))
    }

    /// Construct an \\(n\\)-dimensional [MvNormal](struct.MvNormal.html)
    /// distribution with mean \\(\\bm{\\mu}\\) and covariance matrix
    /// \\(\\bm{\\Sigma}\\), without checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use ndarray::arr2;
    /// # use rstat::normal::MvNormal;
    /// let cov = arr2(&[
    ///     [0.5, 2.0],
    ///     [1.0, 1.0]
    /// ]);
    /// let dist: MvNormal = MvNormal::new_unchecked(vec![0.0, 1.0], cov);
    /// ```
    pub fn new_unchecked<M, S>(mu: M, Sigma: S) -> Self
    where
        M: Into<Vector<f64>>,
        S: Into<Matrix<f64>>,
    {
        let params = Params {
            mu: Loc(mu.into()),
            Sigma: Covariance(Sigma.into()),
        };

        #[cfg(feature = "ndarray-linalg")]
        {
            use ndarray_linalg::{
                cholesky::{Cholesky, UPLO},
                Inverse,
            };

            let Sigma_lt = params.Sigma.0.cholesky(UPLO::Lower).expect(
                "Covariance matrix must be positive-definite to apply Cholesky decomposition.",
            );
            let Sigma_lt_inv = Sigma_lt
                .inv()
                .expect("Covariance matrix must be positive-definite to compute an inverse.");
            let precision = &Sigma_lt_inv * &Sigma_lt_inv.t();

            MvNormal {
                params,
                Sigma_det: Sigma_lt.diag().product(),
                Sigma_lt,
                precision,
            }
        }

        #[cfg(not(feature = "ndarray-linalg"))]
        {
            let Sigma_lt = unsafe { cholesky(&params.Sigma.0) };
            let Sigma_det = Sigma_lt.diag().product();

            let Sigma_lt_inv = unsafe { inverse_lt(&Sigma_lt) };
            let precision = &Sigma_lt_inv * &Sigma_lt_inv.t();

            MvNormal {
                params,
                Sigma_lt,
                Sigma_det,
                precision,
            }
        }
    }
}

impl From<MvNormalParams> for MvNormal {
    fn from(params: MvNormalParams) -> MvNormal {
        MvNormal::new_unchecked(params.mu.0, params.Sigma.0)
    }
}

impl Distribution for MvNormal {
    type Support = ProductSpace<Reals>;
    type Params = MvNormalParams;

    fn support(&self) -> ProductSpace<Reals> { self.params.mu.0.iter().map(|_| Reals).collect() }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let z = Vector::from_shape_fn((self.params.mu.0.len(),), |_| rng.sample(RandSN));
        let az = self.Sigma_lt.dot(&z);

        (az + &self.params.mu.0).into_raw_vec()
    }
}

impl ContinuousDistribution for MvNormal {
    fn pdf(&self, x: &Vec<f64>) -> f64 {
        let z = self.d_mahalanobis_squared(x);
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.Sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for MvNormal {
    fn mean(&self) -> Vector<f64> { self.params.mu.0.clone() }

    fn covariance(&self) -> Matrix<f64> { self.params.Sigma.0.clone() }

    fn variance(&self) -> Vector<f64> { self.params.Sigma.0.diag().to_owned() }
}

// impl Likelihood for MvNormal {
    // fn log_likelihood(&self, samples: &[Vec<f64>]) -> f64 {
        // let (mu, Sigma) = get_params!(self);

        // let k = mu.len();
        // let no2 = (samples.len() as f64) / 2.0;
        // let ln_det = self.Sigma_det.ln();

        // -no2 * (ln_det + k as f64 * PI_2.ln() + samples.into_iter().map(|xs| {
            // let xs = unsafe { ArrayView1::from_shape_ptr(k, &xs) };
            // let diff = xs - mu;

            // diff.insert_axis(Axis(0)).dot(&sigma_inv)
        // }).sum::<f64>())
    // }
// }

impl Mahalanobis for MvNormal {
    fn d_mahalanobis_squared(&self, x: &Vec<f64>) -> f64 {
        let diff: Vector<f64> = x
            .into_iter()
            .zip(self.params.mu.0.iter())
            .map(|(x, y)| x - y)
            .collect();

        diff.dot(&self.precision).dot(&diff)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Diagonal
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Parameter set for [DiagonalNormal](type.DiagonalNormal.html).
pub type DiagonalNormalParams = Params<Vector<f64>, Vector<f64>>;

impl DiagonalNormalParams {
    /// Construct a parameter set for
    /// [DiagonalNormal](type.DiagonalNormal.html): \\(\\langle\\bm{\\mu},
    /// \\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The mean and variance vectors are the same length.
    /// 2. All variance terms are positive real.
    pub fn diagonal<M, S>(mu: M, sigma2_diag: S) -> Result<Self, failure::Error>
    where
        M: Into<Vector<f64>>,
        S: Into<Vector<f64>>,
    {
        let mu = Loc::new(mu.into())?;
        let Sigma = Covariance::diagonal(sigma2_diag.into())?;

        let n_mu = mu.0.len();
        let n_sigma = mu.0.len();

        assert_constraint!(n_mu == n_sigma)?;

        Ok(Params { mu, Sigma, })
    }
}

pub type DiagonalNormalGrad = Grad<Vector<f64>, Vector<f64>>;

impl std::ops::Mul<f64> for DiagonalNormalGrad {
    type Output = DiagonalNormalGrad;

    fn mul(self, sf: f64) -> DiagonalNormalGrad {
        DiagonalNormalGrad {
            mu: self.mu * sf,
            Sigma: self.Sigma * sf,
        }
    }
}

/// Multivariate Normal distribution with mean \\(\\bm{\\mu}\\) and diagonal
/// covariance matrix \\(\\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\).
pub type DiagonalNormal = MvNormal<Vector<f64>, Vector<f64>>;

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
    /// # use rstat::normal::DiagonalNormal;
    /// let dist = DiagonalNormal::diagonal(
    ///     vec![0.0, 1.0], vec![0.5, 2.0]
    /// );
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn diagonal<M, S>(mu: M, sigma2_diag: S) -> Result<Self, failure::Error>
    where
        M: Into<Vector<f64>>,
        S: Into<Vector<f64>>,
    {
        let params = Params::diagonal(mu, sigma2_diag)?;

        Ok(MvNormal::diagonal_unchecked(params.mu.0, params.Sigma.0))
    }

    /// Construct an \\(n\\)-dimensional
    /// [DiagonalNormal](type.DiagonalNormal.html) distribution with mean \\
    /// (\\bm{\\mu}\\) and diagonal covariance matrix
    /// \\((\\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\), without
    /// checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use rstat::normal::DiagonalNormal;
    /// let dist = DiagonalNormal::diagonal_unchecked(
    ///     vec![0.0, 1.0], vec![0.5, 2.0]
    /// );
    /// ```
    pub fn diagonal_unchecked<M, S>(mu: M, sigma2_diag: S) -> Self
    where
        M: Into<Vector<f64>>,
        S: Into<Vector<f64>>,
    {
        let params = Params {
            mu: Loc(mu.into()),
            Sigma: Covariance(sigma2_diag.into()),
        };

        MvNormal {
            Sigma_lt: params.Sigma.0.clone(),
            Sigma_det: params.Sigma.0.iter().product(),
            precision: params.Sigma.0.iter().cloned().map(|v| 1.0 / v).collect(),
            params,
        }
    }
}

impl From<DiagonalNormalParams> for DiagonalNormal {
    fn from(params: DiagonalNormalParams) -> DiagonalNormal {
        DiagonalNormal::diagonal_unchecked(params.mu.0, params.Sigma.0)
    }
}

impl Distribution for DiagonalNormal {
    type Support = ProductSpace<Reals>;
    type Params = DiagonalNormalParams;

    fn support(&self) -> ProductSpace<Reals> { self.params.mu.0.iter().map(|_| Reals).collect() }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.params
            .mu
            .0
            .iter()
            .zip(self.Sigma_lt.iter())
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
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.Sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for DiagonalNormal {
    fn mean(&self) -> Vector<f64> { self.params.mu.0.clone() }

    fn covariance(&self) -> Matrix<f64> { Matrix::from_diag(&self.params.Sigma.0) }

    fn variance(&self) -> Vector<f64> { self.params.Sigma.0.clone() }
}

impl Mahalanobis for DiagonalNormal {
    fn d_mahalanobis_squared(&self, x: &Vec<f64>) -> f64 {
        x.into_iter()
            .zip(self.params.mu.0.iter())
            .map(|(x, y)| x - y)
            .zip(self.precision.iter())
            .map(|(d, si)| d * si * si)
            .sum()
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Isotropic
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameter set for [IsotropicNormal](type.IsotropicNormal.html).
pub type IsotropicNormalParams = Params<Vector<f64>, f64>;

impl IsotropicNormalParams {
    /// Construct a parameter set for
    /// [IsotropicNormal](type.IsotropicNormal.html): \\(\\langle\\bm{\\mu},
    /// \\sigma^2\\bm{I}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. All variance terms are positive real.
    pub fn isotropic<M: Into<Vector<f64>>>(mu: M, sigma2: f64) -> Result<Self, failure::Error> {
        Ok(IsotropicNormalParams {
            mu: Loc::new(mu.into())?,
            Sigma: Covariance::isotropic(sigma2)?,
        })
    }

    /// Construct a parameter set for
    /// [IsotropicNormal](type.IsotropicNormal.html): \\(\\langle\\bm{\\mu},
    /// \\sigma^2\\bm{I}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The dimensionality is a positive integer.
    /// 2. The variance term is positive real.
    pub fn homogeneous(n: usize, mu: f64, sigma2: f64) -> Result<Self, failure::Error> {
        let n = Natural.check(n)?;

        Self::isotropic(Vector::from_elem((n,), mu), sigma2)
    }

    /// Construct a parameter set for
    /// [IsotropicNormal](type.IsotropicNormal.html): \\(\\langle\\bm{\\mu},
    /// \\sigma^2\\bm{I}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The dimensionality is a positive integer.
    pub fn standard(n: usize) -> Result<Self, failure::Error> { Self::homogeneous(n, 0.0, 1.0) }
}

pub type IsotropicNormalGrad = Grad<Vector<f64>, f64>;

impl std::ops::Mul<f64> for IsotropicNormalGrad {
    type Output = IsotropicNormalGrad;

    fn mul(self, sf: f64) -> IsotropicNormalGrad {
        IsotropicNormalGrad {
            mu: self.mu * sf,
            Sigma: self.Sigma * sf,
        }
    }
}

/// Multivariate Normal distribution with mean \\(\\bm{\\mu}\\) and isotropic
/// covariance matrix \\(\\sigma^2\\bm{I}\\).
pub type IsotropicNormal = MvNormal<Vector<f64>, f64>;

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
    /// # use rstat::normal::IsotropicNormal;
    /// let dist = IsotropicNormal::isotropic(vec![0.0, 1.0], 1.0);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn isotropic<M: Into<Vector<f64>>>(mu: M, sigma2: f64) -> Result<Self, failure::Error> {
        let params = Params::isotropic(mu, sigma2)?;

        Ok(MvNormal::isotropic_unchecked(params.mu.0, params.Sigma.0))
    }

    /// Construct an \\(n\\)-dimensional
    /// [IsotropicNormal](type.IsotropicNormal.html) distribution
    /// with mean \\(\\bm{\\mu}\\) and variance \\(\\sigma^2\\bm{I}\\), without
    /// checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use rstat::normal::IsotropicNormal;
    /// let dist = IsotropicNormal::isotropic_unchecked(vec![0.0, 1.0], 1.0);
    /// ```
    pub fn isotropic_unchecked<M: Into<Vector<f64>>>(mu: M, sigma2: f64) -> Self {
        let params = Params {
            mu: Loc(mu.into()),
            Sigma: Covariance(sigma2),
        };

        MvNormal {
            Sigma_lt: params.Sigma.0,
            Sigma_det: params.Sigma.0.powi(params.mu.0.len() as i32),
            precision: 1.0 / params.Sigma.0,
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
    /// # use rstat::normal::IsotropicNormal;
    /// let dist = IsotropicNormal::homogeneous(2, 0.0, 1.0);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn homogeneous(n: usize, mu: f64, sigma2: f64) -> Result<Self, failure::Error> {
        let params = Params::homogeneous(n, mu, sigma2)?;

        Ok(MvNormal::isotropic_unchecked(params.mu.0, params.Sigma.0))
    }

    /// Construct an \\(n\\)-dimensional
    /// [IsotropicNormal](type.IsotropicNormal.html) distribution
    /// with mean \\(\\mu\\) and variance \\(\\sigma^2\\) in each dimension,
    /// without checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use rstat::normal::IsotropicNormal;
    /// let dist = IsotropicNormal::homogeneous_unchecked(2, 0.0, 1.0);
    /// ```
    pub fn homogeneous_unchecked(n: usize, mu: f64, sigma2: f64) -> Self {
        Self::isotropic_unchecked(Vector::from_elem((n,), mu), sigma2)
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
    /// # use rstat::normal::IsotropicNormal;
    /// let dist = IsotropicNormal::standard(2);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn standard(n: usize) -> Result<Self, failure::Error> {
        let params = Params::standard(n)?;

        Ok(IsotropicNormal::isotropic_unchecked(params.mu.0, params.Sigma.0))
    }

    /// Construct an \\(n\\)-dimensional
    /// [IsotropicNormal](type.IsotropicNormal.html) distribution
    /// with mean 0 and unit variance \\(\\sigma^2\\) in each dimension, without
    /// checking for correctness.
    ///
    /// # Examples
    /// ```
    /// # use rstat::normal::IsotropicNormal;
    /// let dist = IsotropicNormal::standard_unchecked(2);
    /// ```
    pub fn standard_unchecked(n: usize) -> Self {
        IsotropicNormal::homogeneous_unchecked(n, 0.0, 1.0)
    }
}

impl From<IsotropicNormalParams> for IsotropicNormal {
    fn from(params: IsotropicNormalParams) -> IsotropicNormal {
        IsotropicNormal::isotropic_unchecked(params.mu.0, params.Sigma.0)
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

                m + z * self.Sigma_lt
            })
            .collect()
    }
}

impl ContinuousDistribution for IsotropicNormal {
    fn pdf(&self, x: &Vec<f64>) -> f64 {
        let z = self.d_mahalanobis_squared(x);
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.Sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for IsotropicNormal {
    fn mean(&self) -> Vector<f64> { self.params.mu.0.clone() }

    fn covariance(&self) -> Matrix<f64> {
        let n = self.params.mu.0.len();
        let mut cov = Matrix::zeros((n, n));

        cov.diag_mut().fill(self.params.Sigma.0);

        cov
    }

    fn variance(&self) -> Vector<f64> {
        Vector::from_elem((self.params.mu.0.len(),), self.params.Sigma.0)
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

#[cfg(test)]
mod tests {
    use super::IsotropicNormal;
    use crate::ContinuousDistribution;
    use failure::Error;

    #[test]
    fn test_pdf() -> Result<(), Error> {
        let m = IsotropicNormal::standard(5)?;
        let prob = m.pdf(&vec![0.0; 5]);

        assert!((prob - 0.010105326013811651).abs() < 1e-7);

        Ok(())
    }
}
