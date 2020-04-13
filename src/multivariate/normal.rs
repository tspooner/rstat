use crate::{
    consts::PI_2,
    params::{Param, Loc, constraints::{self, Constraint, Constraints}},
    prelude::*,
};
use failure::Error;
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{ProductSpace, real::Reals};
use std::fmt;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Parameters
///////////////////////////////////////////////////////////////////////////////////////////////////
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Covariance<T>(pub T);

impl Covariance<Matrix<f64>> {
    pub fn new(value: Matrix<f64>) -> Result<Self, failure::Error> {
        let c = constraints::And((
            constraints::Square,
            constraints::All(constraints::Positive)
        ));

        Ok(Covariance(c.check(value)?))
    }
}

impl Param for Covariance<Matrix<f64>> {
    type Value = Matrix<f64>;

    fn value(&self) -> &Self::Value { &self.0 }

    fn constraints() -> Constraints<Self::Value> {
        let c = constraints::All(constraints::Positive);

        vec![Box::new(c)]
    }
}

impl Covariance<Vector<f64>> {
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

impl Covariance<f64> {
    pub fn isotropic(value: f64) -> Result<Self, failure::Error> {
        Ok(Covariance(assert_constraint!(value+)?))
    }
}

impl Param for Covariance<f64> {
    type Value = f64;

    fn value(&self) -> &Self::Value { &self.0 }

    fn constraints() -> Constraints<Self::Value> {
        vec![Box::new(constraints::Positive)]
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Params<S> {
    pub mu: Loc<Vector<f64>>,
    pub sigma: Covariance<S>,
}

impl Params<Matrix<f64>> {
    pub fn new(mu: Vector<f64>, sigma: Matrix<f64>) -> Result<Self, Error> {
        let mu = Loc::new(mu)?;
        let sigma = Covariance::new(sigma)?;

        let n_mu = mu.0.len();
        let n_sigma = sigma.0.nrows();

        assert_constraint!(n_mu == n_sigma)?;

        Ok(Params { mu, sigma, })
    }
}

impl Params<Vector<f64>> {
    pub fn diagonal(mu: Vector<f64>, sigma: Vector<f64>) -> Result<Self, Error> {
        let mu = Loc::new(mu)?;
        let sigma = Covariance::diagonal(sigma)?;

        let n_mu = mu.0.len();
        let n_sigma = mu.0.len();

        assert_constraint!(n_mu == n_sigma)?;

        Ok(Params { mu, sigma, })
    }
}

impl Params<f64> {
    pub fn isotropic(mu: Vector<f64>, sigma: f64) -> Result<Self, Error> {
        let mu = Loc::new(mu)?;
        let sigma = Covariance::isotropic(sigma)?;

        Ok(Params { mu, sigma, })
    }

    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> Result<Self, Error> {
        let n = constraints::Natural.check(n)?;

        Self::isotropic(Vector::from_elem((n,), mu), sigma)
    }

    pub fn standard(n: usize) -> Result<Self, Error> {
        Self::homogeneous(n, 0.0, 1.0)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Base struct
///////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone)]
pub struct Normal<S> {
    pub(crate) params: Params<S>,

    sigma_det: f64,
    sigma_lt: S,
    sigma_inv: S,
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Diagonal Normal
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type FullNormal = Normal<Matrix<f64>>;

impl FullNormal {
    pub fn new(mu: Vector<f64>, sigma: Matrix<f64>) -> Result<Self, Error> {
        let params = Params::new(mu, sigma)?;

        Ok(Normal::new_unchecked(params.mu.0, params.sigma.0))
    }

    pub fn new_unchecked(mu: Vector<f64>, sigma: Matrix<f64>) -> Self {
        let params = Params { mu: Loc(mu), sigma: Covariance(sigma), };

        #[cfg(feature = "ndarray-linalg")]
        {
            use ndarray_linalg::{Inverse, cholesky::{Cholesky, UPLO}};

            let sigma_lt = params.sigma.0.cholesky(UPLO::Lower).expect(
                "Covariance matrix must be positive-definite to apply Cholesky decomposition.");
            let sigma_lt_inv = sigma_lt.inv()
                .expect("Covariance matrix must be positive-definite to compute an inverse.");
            let sigma_inv = &sigma_lt_inv * &sigma_lt_inv.t();

            Normal {
                params,
                sigma_det: sigma_lt.diag().product(),
                sigma_lt,
                sigma_inv,
            }
        }

        #[cfg(not(feature = "ndarray-linalg"))]
        {
            let sigma_lt = unsafe { cholesky(&params.sigma.0) };
            let sigma_det = sigma_lt.diag().product();

            let sigma_lt_inv = unsafe { inverse_lt(&sigma_lt) };
            let sigma_inv = &sigma_lt_inv * &sigma_lt_inv.t();

            Normal {
                params,
                sigma_lt,
                sigma_det,
                sigma_inv,
            }
        }
    }

    pub fn precision(&self) -> Matrix<f64> { self.sigma_inv.clone() }

    #[inline]
    pub fn z(&self, x: &[f64]) -> f64 {
        let diff: Vector<f64> =
            x.into_iter().zip(self.params.mu.0.iter()).map(|(x, y)| x - y).collect();

        diff.dot(&self.sigma_inv).dot(&diff)
    }
}

impl From<Params<Matrix<f64>>> for FullNormal {
    fn from(params: Params<Matrix<f64>>) -> FullNormal {
        FullNormal::new_unchecked(params.mu.0, params.sigma.0)
    }
}

impl Distribution for FullNormal {
    type Support = ProductSpace<Reals>;
    type Params = Params<Matrix<f64>>;

    fn support(&self) -> ProductSpace<Reals> {
        self.params.mu.0.iter().map(|_| Reals).collect()
    }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let z = Vector::from_shape_fn((self.params.mu.0.len(),), |_| rng.sample(RandSN));
        let az = self.sigma_lt.dot(&z);

        (az + &self.params.mu.0).into_raw_vec()
    }
}

impl ContinuousDistribution for FullNormal {
    fn pdf(&self, x: &Vec<f64>) -> f64 {
        let z = self.z(x);
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for FullNormal {
    fn mean(&self) -> Vector<f64> { self.params.mu.0.clone() }

    fn covariance(&self) -> Matrix<f64> {
        self.params.sigma.0.clone()
    }

    fn variance(&self) -> Vector<f64> {
        self.params.sigma.0.diag().to_owned()
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Diagonal Normal
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type DiagonalNormal = Normal<Vector<f64>>;

impl DiagonalNormal {
    pub fn diagonal(mu: Vector<f64>, sigma: Vector<f64>) -> Result<Self, Error> {
        let params = Params::diagonal(mu, sigma)?;

        Ok(Normal::diagonal_unchecked(params.mu.0, params.sigma.0))
    }

    pub fn diagonal_unchecked(mu: Vector<f64>, sigma: Vector<f64>) -> Self {
        let params = Params { mu: Loc(mu), sigma: Covariance(sigma), };

        Normal {
            sigma_lt: params.sigma.0.clone(),
            sigma_det: params.sigma.0.iter().product(),
            sigma_inv: params.sigma.0.iter().cloned().map(|v| 1.0 / v).collect(),
            params,
        }
    }

    pub fn precision(&self) -> Vector<f64> { self.sigma_inv.clone() }

    #[inline]
    pub fn z(&self, x: &[f64]) -> f64 {
        x.into_iter()
            .zip(self.params.mu.0.iter()).map(|(x, y)| x - y)
            .zip(self.sigma_inv.iter())
            .map(|(d, si)| d * si * si)
            .sum()
    }
}

impl From<Params<Vector<f64>>> for DiagonalNormal {
    fn from(params: Params<Vector<f64>>) -> DiagonalNormal {
        DiagonalNormal::diagonal_unchecked(params.mu.0, params.sigma.0)
    }
}

impl Distribution for DiagonalNormal {
    type Support = ProductSpace<Reals>;
    type Params = Params<Vector<f64>>;

    fn support(&self) -> ProductSpace<Reals> {
        self.params.mu.0.iter().map(|_| Reals).collect()
    }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.params.mu.0.iter().zip(self.sigma_lt.iter()).map(|(m, s)| {
            let z: f64 = rng.sample(RandSN);

            m + z * s
        }).collect()
    }
}

impl ContinuousDistribution for DiagonalNormal {
    fn pdf(&self, x: &Vec<f64>) -> f64 {
        let z = self.z(x);
        let norm = (PI_2.powi(self.params.mu.0.len() as i32) * self.sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for DiagonalNormal {
    fn mean(&self) -> Vector<f64> { self.params.mu.0.clone() }

    fn covariance(&self) -> Matrix<f64> {
        Matrix::from_diag(&self.params.sigma.0)
    }

    fn variance(&self) -> Vector<f64> {
        self.params.sigma.0.clone()
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Isotropic Normal
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type IsotropicNormal = Normal<f64>;

impl IsotropicNormal {
    pub fn isotropic(mu: Vector<f64>, sigma: f64) -> Result<Self, Error> {
        let params = Params::isotropic(mu, sigma)?;

        Ok(Normal::isotropic_unchecked(params.mu.0, params.sigma.0))
    }

    pub fn isotropic_unchecked(mu: Vector<f64>, sigma: f64) -> Self {
        let params = Params { mu: Loc(mu), sigma: Covariance(sigma), };

        Normal {
            sigma_lt: params.sigma.0,
            sigma_det: params.sigma.0.powi(params.mu.0.len() as i32),
            sigma_inv: 1.0 / params.sigma.0,
            params,
        }
    }

    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> Result<Self, Error> {
        let params = Params::homogeneous(n, mu, sigma)?;

        Ok(Normal::isotropic_unchecked(params.mu.0, params.sigma.0))
    }

    pub fn homogenous_unchecked(n: usize, mu: f64, sigma: f64) -> Self {
        Self::isotropic_unchecked(Vector::from_elem((n,), mu), sigma)
    }

    pub fn standard(n: usize) -> Result<Self, Error> {
        let params = Params::standard(n)?;

        Ok(Normal::isotropic_unchecked(params.mu.0, params.sigma.0))
    }

    pub fn standard_unchecked(n: usize) -> Self {
        Normal::homogenous_unchecked(n, 0.0, 1.0)
    }

    pub fn precision(&self) -> f64 { self.sigma_inv }

    #[inline]
    pub fn z(&self, x: &[f64]) -> f64 {
        x.into_iter()
            .zip(self.params.mu.0.iter())
            .map(|(x, y)| (x - y) * self.sigma_inv * self.sigma_inv)
            .sum()
    }
}

impl From<Params<f64>> for IsotropicNormal {
    fn from(params: Params<f64>) -> IsotropicNormal {
        IsotropicNormal::isotropic_unchecked(params.mu.0, params.sigma.0)
    }
}

impl Distribution for IsotropicNormal {
    type Support = ProductSpace<Reals>;
    type Params = Params<f64>;

    fn support(&self) -> ProductSpace<Reals> {
        self.params.mu.0.iter().map(|_| Reals).collect()
    }

    fn params(&self) -> Self::Params { self.params.clone() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.params.mu.0.iter().map(|m| {
            let z: f64 = rng.sample(RandSN);

            m + z * self.sigma_lt
        }).collect()
    }
}

impl ContinuousDistribution for IsotropicNormal {
    fn pdf(&self, x: &Vec<f64>) -> f64 {
        let z = self.z(x);
        println!("{:?}", z);
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

///////////////////////////////////////////////////////////////////////////////////////////////////
// Common
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<S: fmt::Display> fmt::Display for Normal<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.params.mu.0, self.params.sigma.0)
    }
}

#[cfg(test)]
mod tests {
    use crate::ContinuousDistribution;
    use failure::Error;
    use super::Normal;

    #[test]
    fn test_pdf() -> Result<(), Error> {
        let m = Normal::standard(5)?;
        let prob = m.pdf(&vec![0.0; 5]);

        assert!((prob - 0.010105326013811651).abs() < 1e-7);

        Ok(())
    }
}
