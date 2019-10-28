use crate::{prelude::*, validation::{Result, ValidationError}};
use rand::Rng;
use spaces::real::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Frechet {
    pub alpha: f64,
    pub s: f64,
    pub m: f64,
}

impl Frechet {
    pub fn new(alpha: f64, s: f64, m: f64) -> Result<Frechet> {
        let alpha = ValidationError::assert_positive_real(alpha)?;
        let s = ValidationError::assert_positive_real(s)?;

        Ok(Frechet::new_unchecked(alpha, s, m))
    }

    pub fn new_unchecked(alpha: f64, s: f64, m: f64) -> Frechet {
        Frechet { alpha, s, m }
    }

    #[inline(always)]
    fn z(&self, x: f64) -> f64 {
        (x - self.m) / self.s
    }
}

impl Default for Frechet {
    fn default() -> Frechet {
        Frechet {
            alpha: 1.0,
            s: 1.0,
            m: 1.0,
        }
    }
}

impl Distribution for Frechet {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::left_bounded(self.m)
    }

    fn cdf(&self, x: f64) -> Probability {
        let z = self.z(x);

        Probability::new_unchecked((-(z.powf(-self.alpha))).exp())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Frechet {
    fn pdf(&self, x: f64) -> f64 {
        let z = self.z(x);
        let cdf = f64::from(self.cdf(x));

        self.alpha / self.s * z.powf(-1.0 - self.alpha) * cdf
    }
}

impl UnivariateMoments for Frechet {
    fn mean(&self) -> f64 {
        if self.alpha <= 1.0 {
            unimplemented!("Mean is infinite for alpha <= 1.")
        }

        use special_fun::FloatSpecial;

        self.m + self.s * (1.0 - 1.0 / self.alpha).gamma()
    }

    fn variance(&self) -> f64 {
        if self.alpha <= 2.0 {
            unimplemented!("Variance is infinite for alpha <= 2.")
        }

        use special_fun::FloatSpecial;

        let gamma_1m1oa = (1.0 - 1.0 / self.alpha).gamma();

        self.s * self.s * ((1.0 - 2.0 / self.alpha).gamma() - gamma_1m1oa * gamma_1m1oa)
    }

    fn skewness(&self) -> f64 {
        if self.alpha <= 3.0 {
            unimplemented!("Skewness is infinite for alpha <= 3.")
        }

        use special_fun::FloatSpecial;

        let gamma_1m1oa = (1.0 - 1.0 / self.alpha).gamma();
        let gamma_1m2oa = (1.0 - 2.0 / self.alpha).gamma();

        let numerator = (1.0 - 3.0 / self.alpha).gamma() - 3.0 * gamma_1m2oa * gamma_1m1oa
            + 2.0 * gamma_1m1oa * gamma_1m1oa * gamma_1m1oa;
        let denominator_inner = gamma_1m2oa - gamma_1m1oa * gamma_1m1oa;
        let denominator = (denominator_inner * denominator_inner * denominator_inner).sqrt();

        (numerator / denominator).into()
    }

    fn excess_kurtosis(&self) -> f64 {
        if self.alpha <= 4.0 {
            unimplemented!("Kurtosis is infinite for alpha <= 4.")
        }

        use special_fun::FloatSpecial;

        let gamma_1m1oa = (1.0 - 1.0 / self.alpha).gamma();
        let gamma_1m2oa = (1.0 - 2.0 / self.alpha).gamma();

        let numerator = (1.0 - 4.0 / self.alpha).gamma()
            - 4.0 * (1.0 - 3.0 / self.alpha).gamma() * gamma_1m1oa
            + 3.0 * gamma_1m2oa * gamma_1m2oa;
        let denominator_inner = gamma_1m2oa - gamma_1m1oa * gamma_1m1oa;
        let denominator = denominator_inner * denominator_inner;

        numerator / denominator - 6.0
    }
}

impl Quantiles for Frechet {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.m + self.s / 2.0f64.ln().powf(1.0 / self.alpha)
    }
}

impl Modes for Frechet {
    fn modes(&self) -> Vec<f64> {
        vec![self.m + self.s * (self.alpha / (1.0 + self.alpha)).powf(1.0 / self.alpha)]
    }
}

impl Entropy for Frechet {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let gamma = -(1.0f64.digamma());

        1.0 + gamma / self.alpha + gamma + (self.s / self.alpha).ln()
    }
}

impl fmt::Display for Frechet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Frechet({}, {}, {})", self.alpha, self.s, self.m)
    }
}
