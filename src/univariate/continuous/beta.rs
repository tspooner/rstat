use crate::{
    consts::{ONE_THIRD, TWO_THIRDS},
    core::*,
};
use rand;
use spaces::real::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Beta {
        assert_positive_real!(alpha);
        assert_positive_real!(beta);

        Beta { alpha, beta }
    }
}

impl Default for Beta {
    fn default() -> Beta {
        Beta {
            alpha: 1.0,
            beta: 1.0,
        }
    }
}

impl Into<rand_distr::Beta<f64>> for Beta {
    fn into(self) -> rand_distr::Beta<f64> {
        rand_distr::Beta::new(self.alpha, self.beta).unwrap()
    }
}

impl Into<rand_distr::Beta<f64>> for &Beta {
    fn into(self) -> rand_distr::Beta<f64> {
        rand_distr::Beta::new(self.alpha, self.beta).unwrap()
    }
}

impl Distribution for Beta {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::bounded(0.0, 1.0)
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        x.betainc(self.alpha, self.beta).into()
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        let sampler: rand_distr::Beta<f64> = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for Beta {
    fn pdf(&self, x: f64) -> f64 {
        use special_fun::FloatSpecial;

        let numerator = x.powf(self.alpha - 1.0) * (1.0 - x).powf(self.beta - 1.0);
        let denominator = self.alpha.beta(self.beta);

        numerator / denominator
    }
}

impl UnivariateMoments for Beta {
    fn mean(&self) -> f64 {
        1.0 / (1.0 + self.beta / self.alpha)
    }

    fn variance(&self) -> f64 {
        let apb = self.alpha + self.beta;

        self.alpha * self.beta / (apb * apb * (apb + 1.0))
    }

    fn skewness(&self) -> f64 {
        let apb = self.alpha + self.beta;

        2.0 * (self.beta - self.alpha) * (apb + 1.0).sqrt()
            / (apb + 2.0)
            / (self.alpha * self.beta).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let apb = self.alpha + self.beta;
        let asb = self.alpha - self.beta;
        let amb = self.alpha * self.beta;

        let apbp2 = apb + 2.0;

        3.0 * asb * asb * (apb + 1.0) - amb * apbp2 / amb / apbp2 / (apb + 3.0)
    }
}

impl Quantiles for Beta {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        if (self.alpha - self.beta).abs() < 1e-7 {
            0.5
        } else if self.alpha > 1.0 && self.beta > 1.0 {
            (self.alpha - ONE_THIRD) / (self.alpha + self.beta - TWO_THIRDS)
        } else if (self.alpha - 1.0).abs() < 1e-7 {
            1.0 - 2.0f64.powf(-1.0 / self.beta)
        } else if (self.beta - 1.0).abs() < 1e-7 {
            2.0f64.powf(-1.0 / self.alpha)
        } else if (self.alpha - 3.0).abs() < 1e-7 && (self.beta - 2.0).abs() < 1e-7 {
            0.6142724318676105
        } else if (self.alpha - 2.0).abs() < 1e-7 && (self.beta - 3.0).abs() < 1e-7 {
            0.38572756813238945
        } else {
            unimplemented!()
        }
    }
}

impl Modes for Beta {
    fn modes(&self) -> Vec<f64> {
        if self.alpha > 1.0 && self.beta > 1.0 {
            vec![(self.alpha - 1.0) / (self.alpha + self.beta - 2.0)]
        } else if self.alpha < 1.0 && self.beta < 1.0 {
            vec![0.0, 1.0]
        } else {
            vec![]
        }
    }
}

impl Entropy for Beta {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let apb = self.alpha + self.beta;

        self.alpha.logbeta(self.beta)
            - (self.alpha - 1.0) * self.alpha.digamma()
            - (self.beta - 1.0) * self.beta.digamma()
            + (apb - 2.0) * apb.digamma()
    }
}

impl fmt::Display for Beta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Beta({}, {})", self.alpha, self.beta)
    }
}
