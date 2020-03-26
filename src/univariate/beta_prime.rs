use crate::prelude::*;
use failure::Error;
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct BetaPrime {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaPrime {
    pub fn new(alpha: f64, beta: f64) -> Result<BetaPrime, Error> {
        let alpha = assert_constraint!(alpha+)?;
        let beta = assert_constraint!(beta+)?;

        Ok(BetaPrime::new_unchecked(alpha, beta))
    }

    pub fn new_unchecked(alpha: f64, beta: f64) -> BetaPrime {
        BetaPrime { alpha, beta }
    }
}

impl Default for BetaPrime {
    fn default() -> BetaPrime {
        BetaPrime {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl Distribution for BetaPrime {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        Probability::new_unchecked((x / (1.0 + x)).betainc(self.alpha, self.beta))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for BetaPrime {
    fn pdf(&self, x: f64) -> f64 {
        use special_fun::FloatSpecial;

        let numerator = x.powf(self.alpha - 1.0) * (1.0 + x).powf(-self.alpha - self.beta);
        let denominator = self.alpha.beta(self.beta);

        numerator / denominator
    }
}

impl UnivariateMoments for BetaPrime {
    fn mean(&self) -> f64 {
        if self.beta <= 1.0 {
            unimplemented!("Mean is undefined for values of beta <= 1.")
        }

        self.alpha / (self.beta - 1.0)
    }

    fn variance(&self) -> f64 {
        if self.beta <= 2.0 {
            unimplemented!("Variance is undefined for values of beta <= 2.")
        }

        let bm1 = self.beta - 1.0;

        self.alpha * (self.alpha + bm1) / (self.beta - 2.0) / bm1 / bm1
    }

    fn skewness(&self) -> f64 {
        if self.beta <= 3.0 {
            unimplemented!("Skewness is undefined for values of beta <= 3.")
        }

        let bm1 = self.beta - 1.0;

        2.0 * (2.0 * self.alpha + bm1) / (self.beta - 3.0)
            * ((self.beta - 2.0) / (self.alpha * (self.alpha + bm1))).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let bm1 = self.beta - 1.0;

        let numerator =
            6.0 * (self.alpha + bm1) * (5.0 * self.beta - 11.0) + bm1 * bm1 * (self.beta - 2.0);
        let denominator = self.alpha * (self.alpha + bm1) * (self.beta - 3.0) * (self.beta - 4.0);

        numerator / denominator
    }
}

impl Modes for BetaPrime {
    fn modes(&self) -> Vec<f64> {
        if self.alpha >= 1.0 {
            vec![(self.alpha - 1.0) / (self.beta + 1.0)]
        } else {
            vec![0.0]
        }
    }
}

impl Entropy for BetaPrime {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let apb = self.alpha + self.beta;

        self.alpha.logbeta(self.beta)
            - (self.alpha - 1.0) * self.alpha.digamma()
            - (self.beta - 1.0) * self.beta.digamma()
            + (apb - 2.0) * apb.digamma()
    }
}

impl fmt::Display for BetaPrime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "BetaPrime({}, {})", self.alpha, self.beta)
    }
}
