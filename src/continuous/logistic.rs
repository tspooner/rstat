use consts::{PI2, SIX_FIFTHS, TWENTY_ONE_FIFTHS};
use core::*;
use rand::Rng;
use spaces::continuous::Reals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Logistic {
    pub mu: f64,
    pub s: f64,
}

impl Logistic {
    pub fn new(mu: f64, s: f64) -> Logistic {
        assert_positive_real!(s);

        Logistic { mu, s }
    }

    fn z(&self, x: f64) -> f64 {
        (x - self.mu) / self.s
    }
}

impl Default for Logistic {
    fn default() -> Logistic {
        Logistic { mu: 0.0, s: 1.0 }
    }
}

impl Distribution for Logistic {
    type Support = Reals;

    fn support(&self) -> Reals { Reals }

    fn cdf(&self, x: f64) -> Probability {
        (1.0 / (1.0 + (-self.z(x)).exp())).into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Logistic {
    fn pdf(&self, x: f64) -> Probability {
        let exp_term = (-self.z(x)).exp();
        let exp_term_p1 = 1.0 + exp_term;

        (exp_term / self.s / exp_term_p1 / exp_term_p1).into()
    }
}

impl UnivariateMoments for Logistic {
    fn mean(&self) -> f64 { self.mu }

    fn variance(&self) -> f64 { self.s * self.s * PI2 / 3.0 }

    fn skewness(&self) -> f64 { unimplemented!() }

    fn kurtosis(&self) -> f64 { TWENTY_ONE_FIFTHS }

    fn excess_kurtosis(&self) -> f64 { SIX_FIFTHS }
}

impl Quantiles for Logistic {
    fn quantile(&self, p: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 { self.mu }
}

impl Modes for Logistic {
    fn modes(&self) -> Vec<f64> { vec![self.mu] }
}

impl Entropy for Logistic {
    fn entropy(&self) -> f64 { self.s.ln() + 2.0 }
}

impl fmt::Display for Logistic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Logistic({}, {})", self.mu, self.s)
    }
}
