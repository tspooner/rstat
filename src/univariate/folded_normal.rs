use crate::{
    consts::{PI_2, TWO_OVER_PI},
    prelude::*,
};
use rand::Rng;
use spaces::real::NonNegativeReals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct FoldedNormal {
    pub mu: f64,
    pub sigma: f64,
}

impl FoldedNormal {
    pub fn new(mu: f64, sigma: f64) -> FoldedNormal {
        assert_positive_real!(sigma);

        FoldedNormal { mu, sigma }
    }

    pub fn half_normal(sigma: f64) -> FoldedNormal {
        FoldedNormal::new(0.0, sigma)
    }

    pub fn standard() -> FoldedNormal {
        FoldedNormal::new(0.0, 1.0)
    }

    #[inline(always)]
    pub fn precision(&self) -> f64 {
        1.0 / self.sigma / self.sigma
    }
}

impl Default for FoldedNormal {
    fn default() -> FoldedNormal {
        FoldedNormal::standard()
    }
}

impl Distribution for FoldedNormal {
    type Support = NonNegativeReals;

    fn support(&self) -> NonNegativeReals {
        NonNegativeReals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        let sqrt_2: f64 = 2.0f64.sqrt();

        Probability::new_unchecked(0.5 * (
            ((x + self.mu) / self.sigma / sqrt_2).erf() +
            ((x - self.mu) / self.sigma / sqrt_2).erf()
        ))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        rand_distr::Normal::new(self.mu, self.sigma).unwrap().sample(rng).abs()
    }
}

impl ContinuousDistribution for FoldedNormal {
    fn pdf(&self, x: f64) -> f64 {
        let z_pos = (x + self.mu) / self.sigma;
        let z_neg = (x - self.mu) / self.sigma;

        let norm = PI_2.sqrt() * self.sigma;

        (-z_pos * z_pos / 2.0).exp() / norm + (-z_neg * z_neg / 2.0).exp() / norm
    }
}

impl UnivariateMoments for FoldedNormal {
    fn mean(&self) -> f64 {
        use special_fun::FloatSpecial;

        let z = self.mu / self.sigma / 2.0f64.sqrt();

        self.sigma * TWO_OVER_PI.sqrt() * (-z * z).exp() + self.mu * z.erf()
    }

    fn variance(&self) -> f64 {
        let mean = self.mean();

        self.mu * self.mu + self.sigma * self.sigma - mean * mean
    }

    fn skewness(&self) -> f64 { unimplemented!() }

    fn kurtosis(&self) -> f64 { unimplemented!() }

    fn excess_kurtosis(&self) -> f64 { unimplemented!() }
}

impl Modes for FoldedNormal {
    fn modes(&self) -> Vec<f64> {
        if self.mu < self.sigma {
            vec![0.0]
        } else {
            vec![self.mu]
        }
    }
}

impl fmt::Display for FoldedNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FN({}, {})", self.mu, self.variance())
    }
}
