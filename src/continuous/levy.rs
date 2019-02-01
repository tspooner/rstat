use consts::{PI, PI_16, THREE_HALVES};
use core::*;
use rand::Rng;
use spaces::continuous::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Levy {
    pub mu: f64,
    pub c: f64,
}

impl Levy {
    pub fn new(mu: f64, c: f64) -> Levy {
        assert_positive_real!(c);

        Levy { mu, c }
    }
}

impl Default for Levy {
    fn default() -> Levy {
        Levy { mu: 0.0, c: 1.0 }
    }
}

impl Distribution for Levy {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::left_bounded(self.mu)
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        (self.c / 2.0 / (x - self.mu)).erfc().into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Levy {
    fn pdf(&self, x: f64) -> Probability {
        let diff = x - self.mu;
        let c_over_2 = self.c / 2.0;

        ((c_over_2 / PI).sqrt() * (-c_over_2 / diff).exp() / diff.powf(THREE_HALVES)).into()
    }
}

impl UnivariateMoments for Levy {
    fn mean(&self) -> f64 {
        unimplemented!("Mean of the Levy distribution always diverges to infinity.")
    }

    fn variance(&self) -> f64 {
        unimplemented!("Variance of the Levy distribution always diverges to infinity.")
    }

    fn skewness(&self) -> f64 {
        unimplemented!()
    }

    fn kurtosis(&self) -> f64 {
        unimplemented!()
    }
}

// impl Quantiles for Levy {
    // fn quantile(&self, p: Probability) -> f64 {
        // unimplemented!()
    // }

    // fn median(&self) -> f64 {
        // if self.mu.abs() < 1e-7 {
            // unimplemented!()
        // } else {
            // unimplemented!("Median of the Levy distribution is defined only for mu = 0.")
        // }
    // }
// }

impl Modes for Levy {
    fn modes(&self) -> Vec<f64> {
        if self.mu.abs() < 1e-7 {
            vec![self.c / 3.0]
        } else {
            unimplemented!("Median of the Levy distribution is defined only for mu = 0.")
        }
    }
}

impl Entropy for Levy {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let gamma = -(1.0f64.digamma());

        (1.0 + 3.0 * gamma + (PI_16 * self.c * self.c).ln()) / 2.0
    }
}

impl fmt::Display for Levy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Levy({}, {})", self.mu, self.c)
    }
}
