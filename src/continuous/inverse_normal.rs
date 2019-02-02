use super::Gaussian;
use consts::{PI_2, PI_E_2};
use core::*;
use rand::Rng;
use spaces::continuous::PositiveReals;
use std::fmt;

pub type InvGaussian = InvNormal;

#[derive(Debug, Clone, Copy)]
pub struct InvNormal {
    pub mu: f64,
    pub lambda: f64,

    sgd: Gaussian,
}

impl InvNormal {
    pub fn new(mu: f64, lambda: f64) -> InvNormal {
        assert_positive_real!(mu);
        assert_positive_real!(lambda);

        InvNormal {
            mu,
            lambda,
            sgd: Gaussian::standard(),
        }
    }
}

impl Default for InvNormal {
    fn default() -> InvNormal {
        InvNormal {
            mu: 1.0,
            lambda: 1.0,
            sgd: Gaussian::standard(),
        }
    }
}

impl Distribution for InvNormal {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        let xom = x / self.mu;
        let lox_sqrt = (self.lambda / x).sqrt();

        let term1 = f64::from(self.sgd.cdf(lox_sqrt * (xom - 1.0)));
        let term2 = (2.0 * self.lambda / self.mu).exp() *
            f64::from(self.sgd.cdf(-lox_sqrt * (xom + 1.0)));

        (term1 + term2).into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for InvNormal {
    fn pdf(&self, x: f64) -> Probability {
        let z = (self.lambda / PI_2 / x / x / x).sqrt();

        let diff = x - self.mu;
        let exponent = (-self.lambda * diff * diff) / 2.0 / self.mu / self.mu / x;

        (z * exponent.exp()).into()
    }
}

impl UnivariateMoments for InvNormal {
    fn mean(&self) -> f64 {
        self.mu
    }

    fn variance(&self) -> f64 {
        self.mu * self.mu * self.mu / self.lambda
    }

    fn skewness(&self) -> f64 {
        3.0 * (self.mu / self.lambda).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        16.0 * self.mu / self.lambda
    }
}

impl Modes for InvNormal {
    fn modes(&self) -> Vec<f64> {
        let term1 = (1.0 + 9.0 * self.mu * self.mu / 4.0 / self.lambda / self.lambda).sqrt();
        let term2 = 3.0 * self.mu / 2.0 / self.lambda;

        vec![self.mu * (term1 - term2)]
    }
}

impl Entropy for InvNormal {
    fn entropy(&self) -> f64 {
        (PI_E_2 * self.variance()).ln() / 2.0
    }
}

impl fmt::Display for InvNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "IG({}, {})", self.mu, self.lambda)
    }
}
