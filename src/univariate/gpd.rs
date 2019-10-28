use crate::{
    consts::ONE_THIRD,
    prelude::*,
    validation::{Result, ValidationError},
};
use rand::Rng;
use spaces::real::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct GeneralisedPareto {
    pub mu: f64,
    pub sigma: f64,
    pub zeta: f64,
}

impl GeneralisedPareto {
    pub fn new(mu: f64, sigma: f64, zeta: f64) -> Result<GeneralisedPareto> {
        ValidationError::assert_positive_real(sigma)
            .map(|sigma| GeneralisedPareto::new_unchecked(mu, sigma, zeta))
    }

    pub fn new_unchecked(mu: f64, sigma: f64, zeta: f64) -> GeneralisedPareto {
        GeneralisedPareto { mu, sigma, zeta }
    }
}

impl Distribution for GeneralisedPareto {
    type Support = Interval;

    fn support(&self) -> Interval {
        use std::cmp::Ordering::*;

        match self
            .zeta
            .partial_cmp(&0.0)
            .expect("Invalid value provided for `zeta`.")
        {
            Less => Interval::bounded(self.mu, self.mu - self.sigma / self.zeta),
            Equal | Greater => Interval::left_bounded(self.mu),
        }
    }

    fn cdf(&self, x: f64) -> Probability {
        let z = (x - self.mu) / self.sigma;

        Probability::new_unchecked(1.0 - (1.0 + self.zeta * z).powf(-1.0 / self.zeta))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for GeneralisedPareto {
    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;

        (1.0 + self.zeta * z).powf(-1.0 / self.zeta - 1.0) / self.sigma
    }
}

impl UnivariateMoments for GeneralisedPareto {
    fn mean(&self) -> f64 {
        if self.zeta <= 1.0 {
            unimplemented!("Mean is undefined for zeta <= 1.")
        } else {
            self.mu + self.sigma / (1.0 - self.zeta)
        }
    }

    fn variance(&self) -> f64 {
        if self.zeta <= 0.5 {
            unimplemented!("Variance is undefined for zeta <= 1/2.")
        } else {
            self.sigma * self.sigma / (1.0 - self.zeta).powi(2) / (1.0 - 2.0 * self.zeta)
        }
    }

    fn skewness(&self) -> f64 {
        if self.zeta <= ONE_THIRD {
            unimplemented!("Skewness is undefined for zeta <= 1/3.")
        } else {
            2.0 * (1.0 + self.zeta) * (1.0 - 2.0 * self.zeta).sqrt() / (1.0 - 3.0 * self.zeta)
        }
    }

    fn excess_kurtosis(&self) -> f64 {
        if self.zeta <= ONE_THIRD {
            unimplemented!("Skewness is undefined for zeta <= 1/3.")
        } else {
            3.0 * (1.0 - 2.0 * self.zeta) * (2.0 * self.zeta * self.zeta + self.zeta + 3.0)
                / (1.0 - 3.0 * self.zeta)
                / (1.0 - 4.0 * self.zeta)
                - 3.0
        }
    }
}

impl Quantiles for GeneralisedPareto {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.mu + self.sigma * (2.0f64.powf(self.zeta) - 1.0) / self.zeta
    }
}

impl Entropy for GeneralisedPareto {
    fn entropy(&self) -> f64 {
        self.sigma.ln() + self.zeta + 1.0
    }
}

impl fmt::Display for GeneralisedPareto {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GPD({}, {}, {})", self.mu, self.sigma, self.zeta)
    }
}
