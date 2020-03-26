use crate::{
    consts::PI_2,
    linalg::{Vector, Matrix},
    prelude::*,
};
use failure::Error;
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{ProductSpace, real::Reals};
use std::fmt;

pub type DiagonalGaussian = DiagonalNormal;

#[derive(Debug, Clone)]
pub struct DiagonalNormal {
    pub mu: Vector<f64>,
    pub sigma: Vector<f64>,
}

impl DiagonalNormal {
    pub fn new(mu: Vector<f64>, sigma: Vector<f64>) -> Result<DiagonalNormal, Error> {
        let mu_len = mu.len();

        #[allow(unused_parens)]
        assert_constraint!(mu_len == (sigma.len()))?;

        for s in sigma.iter().cloned() {
            assert_constraint!(s >= 0.0)?;
        }

        Ok(DiagonalNormal::new_unchecked(mu, sigma))
    }

    pub fn new_unchecked(mu: Vector<f64>, sigma: Vector<f64>) -> DiagonalNormal {
        DiagonalNormal {
            mu,
            sigma,
        }
    }

    pub fn isotropic(mu: Vector<f64>, sigma: f64) -> Result<DiagonalNormal, Error> {
        let sigma = assert_constraint!(sigma >= 0.0)?;

        Ok(Self::new_unchecked(mu, Vector::from(vec![sigma; 2])))
    }

    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> Result<DiagonalNormal, Error> {
        let n = assert_constraint!(n > 0)?;

        Self::isotropic(Vector::from_elem((n,), mu), sigma)

    }

    pub fn standard(n: usize) -> Result<DiagonalNormal, Error> {
        Self::homogeneous(n, 0.0, 1.0)
    }

    pub fn precision(&self) -> Vector<f64> { self.sigma.mapv(|x| 1.0 / x) }

    #[inline]
    pub fn z(&self, xs: Vec<f64>) -> f64 {
        xs.into_iter()
            .zip(self.mu.iter().zip(self.sigma.iter()))
            .fold(0.0, |acc, (x, (m, s))| {
                let diff = x - m;

                acc + diff * diff / s
            })
    }
}

impl Distribution for DiagonalNormal {
    type Support = ProductSpace<Reals>;

    fn support(&self) -> ProductSpace<Reals> {
        ProductSpace::new(vec![Reals; self.mu.len()])
    }

    fn cdf(&self, _: Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.mu.iter()
            .zip(self.sigma.iter())
            .map(|(m, s)| {
                let z: f64 = rng.sample(RandSN);

                m + s * z
            })
            .collect()
    }
}

impl ContinuousDistribution for DiagonalNormal {
    fn pdf(&self, x: Vec<f64>) -> f64 {
        let z = self.z(x);
        let norm = (PI_2.powi(self.mu.len() as i32) * self.sigma.product()).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for DiagonalNormal {
    fn mean(&self) -> Vector<f64> {
        self.mu.clone()
    }

    fn covariance(&self) -> Matrix<f64> {
        let mut cov = Matrix::eye(self.sigma.len());

        cov.diag_mut().assign(&self.sigma);

        cov
    }

    fn variance(&self) -> Vector<f64> {
        self.sigma.clone()
    }
}

impl fmt::Display for DiagonalNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.mean(), self.covariance())
    }
}
