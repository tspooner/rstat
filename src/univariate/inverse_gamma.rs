use crate::{prelude::*, validation::{Validator, Result}};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct InvGamma {
    pub alpha: f64,
    pub beta: f64,
}

impl InvGamma {
    pub fn new(alpha: f64, beta: f64) -> Result<InvGamma> {
        Validator
            .require_positive_real(alpha)?
            .require_positive_real(beta)
            .map(|_| InvGamma::new_unchecked(alpha, beta))
    }

    pub fn new_unchecked(alpha: f64, beta: f64) -> InvGamma {
        InvGamma { alpha, beta }
    }

    pub fn with_scale(k: f64, theta: f64) -> Result<InvGamma> {
        InvGamma::new(k, 1.0 / theta)
    }
}

impl Default for InvGamma {
    fn default() -> InvGamma {
        InvGamma {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl Distribution for InvGamma {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        Probability::new_unchecked(self.alpha.gammainc(self.beta / x) / self.alpha.gamma())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for InvGamma {
    fn pdf(&self, x: f64) -> f64 {
        use special_fun::FloatSpecial;

        self.beta.powf(self.alpha) * x.powf(-self.alpha - 1.0) * (-self.beta / x).exp()
            / self.alpha.gamma()
    }
}

impl UnivariateMoments for InvGamma {
    fn mean(&self) -> f64 {
        if self.alpha <= 1.0 {
            unimplemented!("Mean is undefined for alpha <= 1.")
        }

        self.beta / (self.alpha - 1.0)
    }

    fn variance(&self) -> f64 {
        if self.alpha <= 2.0 {
            unimplemented!("Variance is undefined for alpha <= 2.")
        }

        let am1 = self.alpha - 1.0;

        self.beta * self.beta / am1 / am1 / (self.alpha - 2.0)
    }

    fn skewness(&self) -> f64 {
        if self.alpha <= 3.0 {
            unimplemented!("Skewness is undefined for alpha <= 3.")
        }

        4.0 * (self.alpha - 2.0).sqrt() / (self.alpha - 3.0)
    }

    fn excess_kurtosis(&self) -> f64 {
        if self.alpha <= 4.0 {
            unimplemented!("Kurtosis is undefined for alpha <= 4.")
        }

        (30.0 * self.alpha - 66.0) / (self.alpha - 3.0) / (self.alpha - 4.0)
    }
}

impl Modes for InvGamma {
    fn modes(&self) -> Vec<f64> {
        vec![self.beta / (self.alpha + 1.0)]
    }
}

impl Entropy for InvGamma {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        self.alpha + (self.beta * self.alpha.gamma()).ln()
            - (1.0 + self.alpha) * self.alpha.digamma()
    }
}

impl fmt::Display for InvGamma {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Inv-Gamma({}, {})", self.alpha, self.beta)
    }
}
