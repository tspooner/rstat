use super::{Covariance, Loc, Params, Grad, linalg};
use crate::{
    consts::PI_2,
    distance::Mahalanobis,
    params::constraints::{Natural, Constraint},
    statistics::{MvMoments, Modes, ShannonEntropy},
    ContinuousDistribution,
    Distribution,
    Probability,
    Multivariate,
};
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::real::{Reals, reals};
use std::fmt;

type Supp<const N: usize> = [Reals<f64>; N];

/// Multivariate Normal distribution with mean \\(\\bm{\\mu}\\) and  covariance
/// matrix \\(\\bm{\\Sigma}\\).
#[derive(Debug, Clone)]
pub struct MvNormal<const N: usize, M = [f64; N], S = [[f64; N]; N]> {
    pub(crate) params: Params<M, S>,

    Sigma_det: f64,
    Sigma_lt: S,
    precision: S,
}

impl<const N: usize, S> Modes for MvNormal<N, [f64; N], S>
where
    MvNormal<N, [f64; N], S>: Distribution<Support = Supp<N>>,
{
    fn modes(&self) -> Vec<[f64; N]> { vec![self.params.mu.0.clone()] }
}

impl<const N: usize, S> ShannonEntropy for MvNormal<N, [f64; N], S>
where
    MvNormal<N, [f64; N], S>: Distribution<Support = Supp<N>>,
{
    fn shannon_entropy(&self) -> f64 {
        let k = self.params.mu.len() as f64;

        0.5 * (k * (1.0 + PI_2.ln()) + self.Sigma_det.ln())
    }
}

impl<const N: usize, M: fmt::Display, S: fmt::Display> fmt::Display for MvNormal<N, M, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.params.mu.0, self.params.Sigma.0)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// General
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type MvNormalParams<const N: usize> = Params<[f64; N], [[f64; N]; N]>;

impl<const N: usize> MvNormalParams<N> {
    /// Construct a parameter set for [MvNormal](struct.MvNormal.html):
    /// \\(\\langle\\bm{\\mu}, \\bm{\\Sigma}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The covariance matrix is not square.
    /// 2. The covariance matrix is not positive semi-definite.
    pub fn multivariate<M, S>(mu: M, Sigma: S) -> Result<Self, failure::Error>
    where
        M: Into<[f64; N]>,
        S: Into<[[f64; N]; N]>,
    {
        let mu = Loc::new(mu.into())?;
        let Sigma = Covariance::matrix(Sigma.into())?;

        Ok(Params { mu, Sigma, })
    }
}

pub type MvNormalGrad<const N: usize> = Grad<[f64; N], [[f64; N]; N]>;

impl<const N: usize> std::ops::Mul<f64> for MvNormalGrad<N> {
    type Output = MvNormalGrad<N>;

    fn mul(self, sf: f64) -> MvNormalGrad<N> {
        MvNormalGrad {
            mu: self.mu.map(|x| x * sf),
            Sigma: self.Sigma.map(|xs| xs.map(|x| x * sf)),
        }
    }
}

impl<const N: usize> MvNormal<N> {
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
    /// # extern crate blas_src;
    /// # use rstat::normal::MvNormal;
    /// let cov = [
    ///     [0.5, 2.0],
    ///     [1.0, 1.0]
    /// ];
    /// let dist = MvNormal::new([0.0, 1.0], cov);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn new<M, S>(mu: M, Sigma: S) -> Result<Self, failure::Error>
    where
        M: Into<[f64; N]>,
        S: Into<[[f64; N]; N]>,
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
    /// # extern crate blas_src;
    /// # use rstat::normal::MvNormal;
    /// let cov =[
    ///     [0.5, 2.0],
    ///     [1.0, 1.0]
    /// ];
    /// let dist: MvNormal<2> = MvNormal::new_unchecked([0.0, 1.0], cov);
    /// ```
    pub fn new_unchecked<M, S>(mu: M, Sigma: S) -> Self
    where
        M: Into<[f64; N]>,
        S: Into<[[f64; N]; N]>,
    {
        let params = Params {
            mu: Loc(mu.into()),
            Sigma: Covariance(Sigma.into()),
        };

        unsafe {
            let Sigma_lt = linalg::cholesky(params.Sigma.0.clone());
            let Sigma_det = (0..N).map(|i| Sigma_lt[i][i]).product::<f64>().powi(2);

            let precision = {
                let lt_inv = linalg::inverse_lt(Sigma_lt.clone());
                let lt_inv_mm = linalg::mm_square(&lt_inv);

                lt_inv_mm
            };

            MvNormal {
                params,
                Sigma_lt,
                Sigma_det,
                precision,
            }
        }
    }
}

impl<const N: usize> From<MvNormalParams<N>> for MvNormal<N, [f64; N], [[f64; N]; N]> {
    fn from(params: MvNormalParams<N>) -> Self {
        MvNormal::new_unchecked(params.mu.0, params.Sigma.0)
    }
}

impl<const N: usize> Distribution for MvNormal<N, [f64; N], [[f64; N]; N]> {
    type Support = Supp<N>;
    type Params = MvNormalParams<N>;

    fn support(&self) -> Supp<N> { [reals(); N] }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &[f64; N]) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; N] {
        let mut z = [0.0f64; N];

        for i in 0..N {
            z[i] = rng.sample(RandSN);
        }

        let mut Az = unsafe { linalg::mv_mult(&self.Sigma_lt, z) };

        for i in 0..N {
            Az[i] += self.params.mu.0[i];
        }

        Az
    }
}

impl<const N: usize> ContinuousDistribution for MvNormal<N> {
    fn pdf(&self, x: &[f64; N]) -> f64 {
        let z = self.d_mahalanobis_squared(x);
        let norm = (PI_2.powi(N as i32) * self.Sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl<const N: usize> Multivariate<N> for MvNormal<N> {}

impl<const N: usize> MvMoments<N> for MvNormal<N> {
    fn mean(&self) -> [f64; N] { self.params.mu.0.clone() }

    fn covariance(&self) -> [[f64; N]; N] { self.params.Sigma.0.clone() }

    fn variance(&self) -> [f64; N] {
        let mut var = [0.0; N];

        for i in 0..N {
            var[i] = self.params.Sigma.0[i][i];
        }

        var
    }
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

impl<const N: usize> Mahalanobis for MvNormal<N> {
    fn d_mahalanobis_squared(&self, x: &[f64; N]) -> f64 {
        let z = {
            let mut res = self.params.mu.0.clone();

            for i in 0..N {
                res[i] -= x[i];
            }

            res
        };
        let Az = unsafe { linalg::mv_mult(&self.precision, z.clone()) };

        IntoIterator::into_iter(z)
            .zip(IntoIterator::into_iter(Az))
            .map(|(zi, Azi)| zi * Azi)
            .sum()
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Diagonal
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Parameter set for [DiagonalNormal](type.DiagonalNormal.html).
pub type DiagonalNormalParams<const N: usize> = Params<[f64; N], [f64; N]>;

impl<const N: usize> DiagonalNormalParams<N> {
    /// Construct a parameter set for
    /// [DiagonalNormal](type.DiagonalNormal.html): \\(\\langle\\bm{\\mu},
    /// \\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The mean and variance vectors are the same length.
    /// 2. All variance terms are positive real.
    pub fn diagonal<M, S>(mu: M, sigma2_diag: S) -> Result<Self, failure::Error>
    where
        M: Into<[f64; N]>,
        S: Into<[f64; N]>,
    {
        let mu = Loc::new(mu.into())?;
        let Sigma = Covariance::diagonal(sigma2_diag.into())?;

        let n_mu = mu.0.len();
        let n_sigma = mu.0.len();

        assert_constraint!(n_mu == n_sigma)?;

        Ok(Params { mu, Sigma, })
    }
}

pub type DiagonalNormalGrad<const N: usize> = Grad<[f64; N], [f64; N]>;

impl<const N: usize> std::ops::Mul<f64> for DiagonalNormalGrad<N> {
    type Output = DiagonalNormalGrad<N>;

    fn mul(self, sf: f64) -> DiagonalNormalGrad<N> {
        DiagonalNormalGrad {
            mu: self.mu.map(|x| x  * sf),
            Sigma: self.Sigma.map(|x| x * sf),
        }
    }
}

/// Multivariate Normal distribution with mean \\(\\bm{\\mu}\\) and diagonal
/// covariance matrix \\(\\mathrm{diag}(\\sigma_1, \\ldots, \\sigma_n)\\).
pub type DiagonalNormal<const N: usize> = MvNormal<N, [f64; N], [f64; N]>;

impl<const N: usize> DiagonalNormal<N> {
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
    ///     [0.0, 1.0], [0.5, 2.0]
    /// );
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn diagonal<M, S>(mu: M, sigma2_diag: S) -> Result<Self, failure::Error>
    where
        M: Into<[f64; N]>,
        S: Into<[f64; N]>,
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
    ///     [0.0, 1.0], [0.5, 2.0]
    /// );
    /// ```
    pub fn diagonal_unchecked<M, S>(mu: M, sigma2_diag: S) -> Self
    where
        M: Into<[f64; N]>,
        S: Into<[f64; N]>,
    {
        let params = Params {
            mu: Loc(mu.into()),
            Sigma: Covariance(sigma2_diag.into()),
        };

        MvNormal {
            Sigma_lt: params.Sigma.0.clone(),
            Sigma_det: params.Sigma.0.iter().product(),
            precision: params.Sigma.0.map(|v| 1.0 / v),
            params,
        }
    }
}

impl<const N: usize> From<DiagonalNormalParams<N>> for DiagonalNormal<N> {
    fn from(params: DiagonalNormalParams<N>) -> DiagonalNormal<N> {
        DiagonalNormal::diagonal_unchecked(params.mu.0, params.Sigma.0)
    }
}

impl<const N: usize> Distribution for DiagonalNormal<N> {
    type Support = Supp<N>;
    type Params = DiagonalNormalParams<N>;

    fn support(&self) -> Supp<N> { [reals(); N] }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &[f64; N]) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; N] {
        let mut s = [0.0; N];

        for i in 0..N {
            let z: f64 = rng.sample(RandSN);

            s[i] = self.params.mu.0[i] + self.Sigma_lt[i] * z;
        }

        s
    }
}

impl<const N: usize> ContinuousDistribution for DiagonalNormal<N> {
    fn pdf(&self, x: &[f64; N]) -> f64 {
        let z = self.d_mahalanobis_squared(x);
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.Sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl<const N: usize> Multivariate<N> for DiagonalNormal<N> {}

impl<const N: usize> MvMoments<N> for DiagonalNormal<N> {
    fn mean(&self) -> [f64; N] { self.params.mu.0.clone() }

    fn covariance(&self) -> [[f64; N]; N] {
        let mut cov = [[0.0; N]; N];

        for i in 0..N {
            cov[i][i] = self.params.Sigma.0[i];
        }

        cov
    }

    fn variance(&self) -> [f64; N] { self.params.Sigma.0.clone() }
}

impl<const N: usize> Mahalanobis for DiagonalNormal<N> {
    fn d_mahalanobis_squared(&self, x: &[f64; N]) -> f64 {
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
pub type IsotropicNormalParams<const N: usize> = Params<[f64; N], f64>;

impl<const N: usize> IsotropicNormalParams<N> {
    /// Construct a parameter set for
    /// [IsotropicNormal](type.IsotropicNormal.html): \\(\\langle\\bm{\\mu},
    /// \\sigma^2\\bm{I}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. All variance terms are positive real.
    pub fn isotropic<M: Into<[f64; N]>>(mu: M, sigma2: f64) -> Result<Self, failure::Error> {
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
    pub fn homogeneous(mu: f64, sigma2: f64) -> Result<Self, failure::Error> {
        let _ = Natural.check(N)?;

        Self::isotropic([mu; N], sigma2)
    }

    /// Construct a parameter set for
    /// [IsotropicNormal](type.IsotropicNormal.html): \\(\\langle\\bm{\\mu},
    /// \\sigma^2\\bm{I}\\rangle.\\)
    ///
    /// # Constraints
    /// 1. The dimensionality is a positive integer.
    pub fn standard() -> Result<Self, failure::Error> { Self::homogeneous(0.0, 1.0) }
}

pub type IsotropicNormalGrad<const N: usize> = Grad<[f64; N], f64>;

impl<const N: usize> std::ops::Mul<f64> for IsotropicNormalGrad<N> {
    type Output = IsotropicNormalGrad<N>;

    fn mul(self, sf: f64) -> IsotropicNormalGrad<N> {
        IsotropicNormalGrad {
            mu: self.mu.map(|x| x * sf),
            Sigma: self.Sigma * sf,
        }
    }
}

/// Multivariate Normal distribution with mean \\(\\bm{\\mu}\\) and isotropic
/// covariance matrix \\(\\sigma^2\\bm{I}\\).
pub type IsotropicNormal<const N: usize> = MvNormal<N, [f64; N], f64>;

impl<const N: usize> IsotropicNormal<N> {
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
    /// let dist = IsotropicNormal::isotropic([0.0, 1.0], 1.0);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn isotropic<M: Into<[f64; N]>>(mu: M, sigma2: f64) -> Result<Self, failure::Error> {
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
    /// let dist = IsotropicNormal::isotropic_unchecked([0.0, 1.0], 1.0);
    /// ```
    pub fn isotropic_unchecked<M: Into<[f64; N]>>(mu: M, sigma2: f64) -> Self {
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
    /// let dist = IsotropicNormal::<2>::homogeneous(0.0, 1.0);
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn homogeneous(mu: f64, sigma2: f64) -> Result<Self, failure::Error> {
        let params = Params::homogeneous(mu, sigma2)?;

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
    /// let dist = IsotropicNormal::<2>::homogeneous_unchecked(0.0, 1.0);
    /// ```
    pub fn homogeneous_unchecked(mu: f64, sigma2: f64) -> Self {
        Self::isotropic_unchecked([mu; N], sigma2)
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
    /// let dist = IsotropicNormal::<2>::standard();
    ///
    /// assert!(dist.is_ok());
    /// ```
    pub fn standard() -> Result<Self, failure::Error> {
        let params = Params::standard()?;

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
    /// let dist = IsotropicNormal::<2>::standard_unchecked();
    /// ```
    pub fn standard_unchecked() -> Self {
        IsotropicNormal::homogeneous_unchecked(0.0, 1.0)
    }
}

impl<const N: usize> From<IsotropicNormalParams<N>> for IsotropicNormal<N> {
    fn from(params: IsotropicNormalParams<N>) -> IsotropicNormal<N> {
        IsotropicNormal::isotropic_unchecked(params.mu.0, params.Sigma.0)
    }
}

impl<const N: usize> Distribution for IsotropicNormal<N> {
    type Support = Supp<N>;
    type Params = IsotropicNormalParams<N>;

    fn support(&self) -> Supp<N> { [reals(); N] }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &[f64; N]) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; N] {
        let mut s = [0.0; N];

        for i in 0..N {
            let z: f64 = rng.sample(RandSN);

            s[i] = self.params.mu.0[i] + self.Sigma_lt * z;
        }

        s
    }
}

impl<const N: usize> ContinuousDistribution for IsotropicNormal<N> {
    fn pdf(&self, x: &[f64; N]) -> f64 {
        let z = self.d_mahalanobis_squared(x);
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.Sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl<const N: usize> Multivariate<N> for IsotropicNormal<N> {}

impl<const N: usize> MvMoments<N> for IsotropicNormal<N> {
    fn mean(&self) -> [f64; N] { self.params.mu.0.clone() }

    fn covariance(&self) -> [[f64; N]; N] {
        let mut cov = [[0.0; N]; N];

        for i in 0..N {
            cov[i][i] = self.params.Sigma.0;
        }

        cov
    }

    fn variance(&self) -> [f64; N] {
        [self.params.Sigma.0; N]
    }
}

impl<const N: usize> Mahalanobis for IsotropicNormal<N> {
    fn d_mahalanobis_squared(&self, x: &[f64; N]) -> f64 {
        x.into_iter()
            .zip(self.params.mu.0.iter())
            .map(|(x, y)| (x - y) * self.precision * self.precision)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{MvNormal, IsotropicNormal};
    use crate::{ContinuousDistribution, distance::Mahalanobis};
    use failure::Error;

    #[test]
    fn test_general_pdf() -> Result<(), Error> {
        let m = MvNormal::<2>::new([0.0, 1.0], [
            [1.0, 0.5],
            [0.5, 1.0]
        ])?;

        assert!((m.precision[0][0] - 4.0 / 3.0).abs() < 1e-7);
        assert!((m.precision[0][1] - -2.0 / 3.0).abs() < 1e-7);
        assert!((m.precision[1][0] - -2.0 / 3.0).abs() < 1e-7);
        assert!((m.precision[1][1] - 4.0 / 3.0).abs() < 1e-7);

        let prob = m.pdf(&[0.0; 2]);

        assert!((prob - 0.09435389770895924).abs() < 1e-7);

        Ok(())
    }

    #[test]
    fn test_isotropic_pdf() -> Result<(), Error> {
        let m = IsotropicNormal::<5>::standard()?;
        let prob = m.pdf(&[0.0; 5]);

        assert!((prob - 0.010105326013811651).abs() < 1e-7);

        Ok(())
    }
}
