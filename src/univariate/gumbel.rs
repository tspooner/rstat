use crate::{
    consts::{PI2, PI3, TWELVE_FIFTHS, TWENTY_SEVEN_FIFTHS},
    prelude::*,
    validation::{Validator, Result},
};
use rand::Rng;
use spaces::real::Reals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Gumbel {
    pub mu: f64,
    pub beta: f64,
}

impl Gumbel {
    pub fn new(mu: f64, beta: f64) -> Result<Gumbel> {
        Validator
            .require_positive_real(beta)
            .map(|_| Gumbel::new_unchecked(mu, beta))
    }

    pub fn new_unchecked(mu: f64, beta: f64) -> Gumbel {
        Gumbel { mu, beta }
    }

    #[inline(always)]
    pub fn z(&self, x: f64) -> f64 {
        (x - self.mu) / self.beta
    }
}

impl Default for Gumbel {
    fn default() -> Gumbel {
        Gumbel { mu: 0.0, beta: 1.0 }
    }
}

impl Distribution for Gumbel {
    type Support = Reals;

    fn support(&self) -> Reals {
        Reals
    }

    fn cdf(&self, x: f64) -> Probability {
        let z = self.z(x);

        Probability::new_unchecked((-(-z).exp()).exp())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Gumbel {
    fn pdf(&self, x: f64) -> f64 {
        let z = self.z(x);

        (-z - (-z).exp()).exp() / self.beta
    }
}

impl UnivariateMoments for Gumbel {
    fn mean(&self) -> f64 {
        use special_fun::FloatSpecial;

        self.mu + self.beta * -(1.0f64.digamma())
    }

    fn variance(&self) -> f64 {
        PI2 * self.beta * self.beta / 6.0
    }

    fn skewness(&self) -> f64 {
        use special_fun::FloatSpecial;

        12.0 * 6.0f64.sqrt() * 3.0f64.riemann_zeta() / PI3
    }

    fn kurtosis(&self) -> f64 {
        TWENTY_SEVEN_FIFTHS
    }

    fn excess_kurtosis(&self) -> f64 {
        TWELVE_FIFTHS
    }
}

impl Quantiles for Gumbel {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.mu - self.beta * 2.0f64.ln().ln()
    }
}

impl Modes for Gumbel {
    fn modes(&self) -> Vec<f64> {
        vec![self.mu]
    }
}

impl Entropy for Gumbel {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        self.beta.ln() + -(1.0f64.digamma()) + 1.0
    }
}

impl fmt::Display for Gumbel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Gumbel({}, {})", self.mu, self.beta)
    }
}
