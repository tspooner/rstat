use crate::{
    consts::PI_2,
    core::*,
};
use ndarray::{Array1, Array2};
use ndarray_linalg::{Determinant, solve::Inverse, cholesky::{Cholesky, UPLO}};
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{ProductSpace, real::Reals};
use std::fmt;

pub type MultivariateGaussian = MultivariateNormal;

#[derive(Debug, Clone)]
pub struct MultivariateNormal {
    pub mu: Array1<f64>,
    pub sigma: Array2<f64>,

    sigma_det: f64,
    sigma_inv: Array2<f64>,
    sigma_cholesky: Array2<f64>,
}

impl MultivariateNormal {
    fn new_unchecked(mu: Array1<f64>, sigma: Array2<f64>) -> MultivariateNormal {
        MultivariateNormal {
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

    pub fn new(mu: Array1<f64>, sigma: Array2<f64>) -> MultivariateNormal {
        assert!(sigma.is_square());

        sigma.iter().for_each(|&v| assert_positive_real!(v));

        Self::new_unchecked(mu, sigma)
    }

    pub fn isotropic(mu: Array1<f64>, sigma: f64) -> MultivariateNormal {
        assert_positive_real!(sigma);

        let mut sigma_mat = Array2::eye(mu.len());
        sigma_mat.diag_mut().fill(sigma);

        Self::new_unchecked(mu, sigma_mat)
    }

    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> MultivariateNormal {
        Self::isotropic(Array1::from_elem((n,), mu), sigma)
    }

    pub fn standard(n: usize) -> MultivariateNormal {
        Self::homogeneous(n, 0.0, 1.0)
    }

    pub fn precision(&self) -> Array2<f64> { self.sigma_inv.clone() }

    #[inline]
    pub fn z(&self, x: Vec<f64>) -> f64 {
        let x = Array1::from_vec(x);
        let diff = x - &self.mu;

        diff.dot(&self.sigma_inv).dot(&diff)
    }
}

impl Distribution for MultivariateNormal {
    type Support = ProductSpace<Reals>;

    fn support(&self) -> ProductSpace<Reals> {
        ProductSpace::new(vec![Reals; self.mu.len()])
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        let z = Array1::from_shape_fn((self.mu.len(),), |_| rng.sample(RandSN));
        let az = self.sigma_cholesky.dot(&z);

        (az + &self.mu).into_raw_vec()
    }
}

impl ContinuousDistribution for MultivariateNormal {
    fn pdf(&self, x: Vec<f64>) -> f64 {
        let z = self.z(x);
        let norm = (PI_2.powi(self.mu.len() as i32) * self.sigma_det).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for MultivariateNormal {
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

impl fmt::Display for MultivariateNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.mean(), self.covariance())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Distribution, ContinuousDistribution};
    use rand::{Rng, thread_rng};
    use super::MultivariateNormal;

    #[test]
    fn test_pdf() {
        let m = MultivariateNormal::standard(5);
        let mut rng = thread_rng();
        let prob = m.pdf(vec![0.0; 5]);

        assert!((prob - 0.010105326013811646).abs() < 1e-7);
    }
}
