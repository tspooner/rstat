use crate::{
    constraints::{Positive, Interval, Constraint},
    consts::PI_2,
    linalg::{Vector, Matrix},
    prelude::*,
};
use failure::Error;
use ndarray::array;
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{TwoSpace, real::Reals};
use std::fmt;

pub type BivariateGaussian = BivariateNormal;

#[derive(Debug, Clone, Copy)]
pub struct BivariateNormal {
    pub mu: [f64; 2],
    pub sigma: [f64; 2],
    pub rho: f64,
}

impl BivariateNormal {
    pub fn new(mu: [f64; 2], sigma: [f64; 2], rho: f64) -> Result<BivariateNormal, Error> {
        let s0 = Positive.check(sigma[0])?;
        let s1 = Positive.check(sigma[1])?;
        let rho = Interval { lb: -1.0, ub: 1.0, }.check(rho)?;

        Ok(BivariateNormal::new_unchecked(mu, [s0, s1], rho))
    }

    pub fn independent(mu: [f64; 2], sigma: [f64; 2]) -> Result<BivariateNormal, Error> {
        BivariateNormal::new(mu, sigma, 0.0)
    }

    pub fn isotropic(mu: [f64; 2], sigma: f64) -> Result<BivariateNormal, Error> {
        BivariateNormal::independent(mu, [sigma, sigma])
    }

    pub fn standard() -> BivariateNormal {
        BivariateNormal::new_unchecked([0.0; 2], [1.0; 2], 0.0)
    }

    pub fn new_unchecked(mu: [f64; 2], sigma: [f64; 2], rho: f64) -> BivariateNormal {
        BivariateNormal { mu, sigma, rho, }
    }

    #[inline]
    pub fn z(&self, x: [f64; 2]) -> f64 {
        let diff_0 = x[0] - self.mu[0];
        let diff_1 = x[1] - self.mu[1];

        let q_term = diff_0.powi(2) / self.sigma[0] / self.sigma[0] -
            2.0 * self.rho * diff_0 * diff_1 / self.sigma[0] / self.sigma[1] +
            diff_1.powi(2) / self.sigma[1] / self.sigma[1];
        let rho_term = 1.0 - self.rho * self.rho;

        q_term / rho_term
    }
}

impl Default for BivariateNormal {
    fn default() -> BivariateNormal {
        BivariateNormal::standard()
    }
}

impl Distribution for BivariateNormal {
    type Support = TwoSpace<Reals>;

    fn support(&self) -> TwoSpace<Reals> {
        TwoSpace::new([Reals, Reals])
    }

    fn cdf(&self, x: [f64; 2]) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 2] {
        let z1: f64 = rng.sample(RandSN);
        let z2: f64 = rng.sample(RandSN);

        [
            self.mu[0] + self.sigma[0] * z1,
            self.mu[1] + self.sigma[1] * (z1 * self.rho + z2 * (1.0 - self.rho * self.rho).sqrt())
        ]
    }
}

impl ContinuousDistribution for BivariateNormal {
    fn pdf(&self, x: [f64; 2]) -> f64 {
        let z = self.z(x);
        let norm = PI_2 * self.sigma[0] * self.sigma[1] * (1.0 - self.rho * self.rho).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for BivariateNormal {
    fn mean(&self) -> Vector<f64> {
        vec![self.mu[0], self.mu[1]].into()
    }

    fn covariance(&self) -> Matrix<f64> {
        let cross_sigma = self.sigma[0] * self.sigma[1];

        array![
            [self.sigma[0] * self.sigma[0], self.rho * cross_sigma],
            [self.rho * cross_sigma, self.sigma[1] * self.sigma[1]],
        ]
    }

    fn variance(&self) -> Vector<f64> {
        vec![
            self.sigma[0] * self.sigma[0],
            self.sigma[1] * self.sigma[1]
        ].into()
    }

    fn correlation(&self) -> Matrix<f64> {
        array![
            [1.0, self.rho],
            [self.rho, 1.0],
        ]
    }
}

impl fmt::Display for BivariateNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.mean(), self.covariance())
    }
}
