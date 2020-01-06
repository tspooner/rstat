use crate::{
    consts::PI_2,
    prelude::*,
    validation::{Validator, Result},
};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Determinant, solve::Inverse, cholesky::{Cholesky, UPLO}};
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{ProductSpace, real::Reals};
use std::fmt;

pub type Gaussian = Normal;

#[derive(Debug, Clone)]
pub struct Normal {
    pub mu: Array1<f64>,
    pub sigma: Array2<f64>,

    sigma_det: f64,
    sigma_inv: Array2<f64>,
    sigma_cholesky: Array2<f64>,
}

impl Normal {
    pub fn new(mu: Array1<f64>, sigma: Array2<f64>) -> Result<Normal> {
        Validator
            .require_square(&sigma)
            .and_then(|v| {
                sigma
                    .iter()
                    .map(|&x| v.require_non_negative(x))
                    .collect::<Result<Validator>>()
                    .map(|_| Normal::new_unchecked(mu, sigma))
            })
    }

    pub fn new_unchecked(mu: Array1<f64>, sigma: Array2<f64>) -> Normal {
        Normal {
            mu,
            sigma_det: sigma.det()
                .expect("Covariance matrix must have a well-defined determinant."),
            sigma_inv: sigma.inv()
                .expect("Covariance matrix must be positive-definite to compute an inverse."),
            sigma_cholesky: sigma.cholesky(UPLO::Lower)
                .expect("Covariance matrix must be positive-definite to apply Cholesky decomposition."),
            sigma,
        }
    }

    pub fn isotropic(mu: Array1<f64>, sigma: f64) -> Result<Normal> {
        Validator
            .require_non_negative(sigma)
            .map(|_| {
                let mut sigma_mat = Array2::eye(mu.len());
                sigma_mat.diag_mut().fill(sigma);

                Self::new_unchecked(mu, sigma_mat)
            })
    }

    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> Result<Normal> {
        Validator
            .require_natural(n)
            .and_then(|_| Self::isotropic(Array1::from_elem((n,), mu), sigma))

    }

    pub fn standard(n: usize) -> Result<Normal> {
        Self::homogeneous(n, 0.0, 1.0)
    }

    pub fn precision(&self) -> Array2<f64> { self.sigma_inv.clone() }

    #[inline]
    pub fn z(&self, x: Vec<f64>) -> f64 {
        let x = Array1::from(x);
        let diff = x - &self.mu;

        diff.dot(&self.sigma_inv).dot(&diff)
    }
}

impl Distribution for Normal {
    type Support = ProductSpace<Reals>;

    fn support(&self) -> ProductSpace<Reals> {
        ProductSpace::new(vec![Reals; self.mu.len()])
    }

    fn cdf(&self, x: Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let z = Array1::from_shape_fn((self.mu.len(),), |_| rng.sample(RandSN));
        let az = self.sigma_cholesky.dot(&z);

        (az + &self.mu).into_raw_vec()
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: Vec<f64>) -> f64 {
        let z = self.z(x);
        let norm = (PI_2.powi(self.mu.len() as i32) * self.sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for Normal {
    fn mean(&self) -> Array1<f64> {
        self.mu.clone()
    }

    fn covariance(&self) -> Array2<f64> {
        self.sigma.clone()
    }

    fn variance(&self) -> Array1<f64> {
        self.sigma.diag().to_owned()
    }
}

impl fmt::Display for Normal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.mean(), self.covariance())
    }
}

#[cfg(test)]
mod tests {
    use crate::ContinuousDistribution;
    use super::Normal;

    #[test]
    fn test_pdf() {
        let m = Normal::standard(5).unwrap();
        let prob = m.pdf(vec![0.0; 5]);

        assert!((prob - 0.010105326013811646).abs() < 1e-7);
    }
}
