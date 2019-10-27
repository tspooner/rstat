use crate::{
    consts::PI_2,
    prelude::*,
};
use ndarray::{array, Array1, Array2};
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
    pub fn new(mu: [f64; 2], sigma: [f64; 2], rho: f64) -> BivariateNormal {
        assert_positive_real!(sigma[0]);
        assert_positive_real!(sigma[1]);

        assert_bounded!(-1.0; rho; 1.0);

        BivariateNormal {
            mu,
            sigma,
            rho,
        }
    }

    pub fn independent(mu: [f64; 2], sigma: [f64; 2]) -> BivariateNormal {
        assert_positive_real!(sigma[1]);
        assert_positive_real!(sigma[1]);

        BivariateNormal {
            mu,
            sigma,
            rho: 0.0
        }
    }

    pub fn isotropic(mu: [f64; 2], sigma: f64) -> BivariateNormal {
        assert_positive_real!(sigma);

        BivariateNormal {
            mu,
            sigma: [sigma; 2],
            rho: 0.0
        }
    }

    pub fn standard() -> BivariateNormal {
        BivariateNormal {
            mu: [0.0; 2],
            sigma: [1.0; 2],
            rho: 0.0
        }
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
    fn mean(&self) -> Array1<f64> {
        vec![self.mu[0], self.mu[1]].into()
    }

    fn covariance(&self) -> Array2<f64> {
        let cross_sigma = self.sigma[0] * self.sigma[1];

        array![
            [self.sigma[0] * self.sigma[0], self.rho * cross_sigma],
            [self.rho * cross_sigma, self.sigma[1] * self.sigma[1]],
        ]
    }

    fn variance(&self) -> Array1<f64> {
        vec![
            self.sigma[0] * self.sigma[0],
            self.sigma[1] * self.sigma[1]
        ].into()
    }

    fn correlation(&self) -> Array2<f64> {
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
