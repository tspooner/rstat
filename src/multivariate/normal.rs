use crate::{
    constraints::{self, Constraint},
    consts::PI_2,
    prelude::*,
};
use failure::Error;
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{ProductSpace, real::Reals};
use std::fmt;

pub type Gaussian = Normal;

#[derive(Debug, Clone)]
pub struct Normal {
    pub mu: Vector<f64>,
    pub sigma: Matrix<f64>,

    sigma_lt: Matrix<f64>,
    sigma_det: f64,
    sigma_inv: Matrix<f64>,
}

impl Normal {
    pub fn new(mu: Vector<f64>, sigma: Matrix<f64>) -> Result<Normal, Error> {
        // TODO: Rewrite this to actually test for positive-semidefiniteness!
        let sigma = constraints::Square.check(sigma)?;

         for x in sigma.iter().cloned() {
             assert_constraint!(x+)?;
         }

        Ok(Normal::new_unchecked(mu, sigma))
    }

    pub fn new_unchecked(mu: Vector<f64>, sigma: Matrix<f64>) -> Normal {
        #[cfg(not(backend))]
        {
            let sigma_lt = unsafe { cholesky(&sigma) };
            let sigma_det = sigma_lt.diag().product();

            let sigma_lt_inv = unsafe { inverse_lt(&sigma_lt) };
            let sigma_inv = &sigma_lt_inv * &sigma_lt_inv.t();

            Normal {
                mu,
                sigma,
                sigma_lt,
                sigma_det,
                sigma_inv,
            }
        }

        #[cfg(backend)]
        {
            use ndarray_linalg::{Inverse, cholesky::{Cholesky, UPLO}};

            let sigma_lt = sigma.cholesky(UPLO::Lower).expect(
                "Covariance matrix must be positive-definite to apply Cholesky decomposition.");
            let sigma_lt_inv = sigma_lt.inv()
                .expect("Covariance matrix must be positive-definite to compute an inverse.");
            let sigma_inv = &sigma_lt_inv * &sigma_lt_inv.t();

            Normal {
                mu,
                sigma,
                sigma_det: sigma_lt.diag().product(),
                sigma_lt,
                sigma_inv,
            }
        }
    }

    pub fn isotropic(mu: Vector<f64>, sigma: f64) -> Result<Normal, Error> {
        let sigma = assert_constraint!(sigma+)?;

        let mut sigma_mat = Matrix::eye(mu.len());
        sigma_mat.diag_mut().fill(sigma);

        Ok(Self::new_unchecked(mu, sigma_mat))
    }

    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> Result<Normal, Error> {
        let n = constraints::Natural.check(n)?;

        Self::isotropic(Vector::from_elem((n,), mu), sigma)

    }

    pub fn standard(n: usize) -> Result<Normal, Error> {
        Self::homogeneous(n, 0.0, 1.0)
    }

    pub fn precision(&self) -> Matrix<f64> { self.sigma_inv.clone() }

    #[inline]
    pub fn z(&self, x: Vec<f64>) -> f64 {
        let x = Vector::from(x);
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
        let z = Vector::from_shape_fn((self.mu.len(),), |_| rng.sample(RandSN));
        let az = self.sigma_lt.dot(&z);

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
    fn mean(&self) -> Vector<f64> {
        self.mu.clone()
    }

    fn covariance(&self) -> Matrix<f64> {
        self.sigma.clone()
    }

    fn variance(&self) -> Vector<f64> {
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
