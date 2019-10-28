use crate::{
    consts::{PI, PI2, PI_OVER_2, THREE_HALVES},
    prelude::*,
    validation::{Result, ValidationError},
};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

const TWO_PI_MINUS_3: f64 = 2.0 * (PI - 3.0);
const FOUR_MINUS_PI: f64 = (4.0 - PI);
const FOUR_MINUS_PI_OVER_2: f64 = FOUR_MINUS_PI / 2.0;

const EXCESS_KURTOSIS: f64 =
    1.5 * PI2 - 6.0 * PI + 16.0 / FOUR_MINUS_PI_OVER_2 / FOUR_MINUS_PI_OVER_2;
const KURTOSIS: f64 = EXCESS_KURTOSIS + 3.0;

#[derive(Debug, Clone, Copy)]
pub struct Rayleigh {
    pub sigma: f64,
}

impl Rayleigh {
    pub fn new(sigma: f64) -> Result<Rayleigh> {
        ValidationError::assert_positive_real(sigma)
            .map(Rayleigh::new_unchecked)
    }

    pub fn new_unchecked(sigma: f64) -> Rayleigh {
        Rayleigh { sigma }
    }
}

impl Default for Rayleigh {
    fn default() -> Rayleigh {
        Rayleigh { sigma: 1.0 }
    }
}

impl Distribution for Rayleigh {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        let sigma2 = self.sigma * self.sigma;

        Probability::new_unchecked(1.0 - (-x * x / sigma2 / 2.0).exp())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Rayleigh {
    fn pdf(&self, x: f64) -> f64 {
        let sigma2 = self.sigma * self.sigma;

        x / sigma2 * (-x * x / sigma2 / 2.0).exp()
    }
}

impl UnivariateMoments for Rayleigh {
    fn mean(&self) -> f64 {
        self.sigma * PI_OVER_2.sqrt()
    }

    fn variance(&self) -> f64 {
        FOUR_MINUS_PI_OVER_2 * self.sigma * self.sigma
    }

    fn skewness(&self) -> f64 {
        TWO_PI_MINUS_3 * PI.sqrt() / FOUR_MINUS_PI.powf(THREE_HALVES)
    }

    fn kurtosis(&self) -> f64 {
        KURTOSIS
    }

    fn excess_kurtosis(&self) -> f64 {
        EXCESS_KURTOSIS
    }
}

impl Quantiles for Rayleigh {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.sigma * (2.0 * 2.0f64.ln()).sqrt()
    }
}

impl Modes for Rayleigh {
    fn modes(&self) -> Vec<f64> {
        vec![self.sigma]
    }
}

impl Entropy for Rayleigh {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let gamma = -(1.0f64.digamma());

        1.0 + (self.sigma / 2.0f64.sqrt()).ln() + gamma / 2.0
    }
}

impl fmt::Display for Rayleigh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rayleigh({})", self.sigma)
    }
}
