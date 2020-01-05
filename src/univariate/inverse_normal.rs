use crate::{
    consts::{PI_2, PI_E_2},
    prelude::*,
    validation::{Validator, Result},
};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;
use super::{Gaussian, Uniform};

pub type InvGaussian = InvNormal;

#[derive(Debug, Clone, Copy)]
pub struct InvNormal {
    pub mu: f64,
    pub lambda: f64,

    sgd: Gaussian,
}

impl InvNormal {
    pub fn new(mu: f64, lambda: f64) -> Result<InvNormal> {
        Validator
            .require_positive_real(mu)?
            .require_positive_real(lambda)
            .map(|_| InvNormal::new_unchecked(mu, lambda))
    }

    pub fn new_unchecked(mu: f64, lambda: f64) -> InvNormal {
        InvNormal { mu, lambda, sgd: Gaussian::standard() }
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

        Probability::new_unchecked(term1 + term2)
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let mu = self.mu;
        let lambda = self.lambda;

        let nu = Gaussian::standard().sample(rng);

        let y = nu * nu;
        let x =
            mu +
            mu * mu * y / 2.0 / self.lambda -
            mu / 2.0 / lambda * (4.0 * mu * lambda * y + mu * mu * y * y).sqrt();

        let z = Uniform::<f64>::new_unchecked(0.0, 1.0).sample(rng);

        if z < mu / (mu + x) { x } else { mu * mu / x }
    }
}

impl ContinuousDistribution for InvNormal {
    fn pdf(&self, x: f64) -> f64 {
        let z = (self.lambda / PI_2 / x / x / x).sqrt();

        let diff = x - self.mu;
        let exponent = (-self.lambda * diff * diff) / 2.0 / self.mu / self.mu / x;

        z * exponent.exp()
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
