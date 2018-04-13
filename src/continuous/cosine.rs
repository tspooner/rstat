use consts::{ONE_THIRD, TWO_OVER_PI2, PI2, PI4};
use core::*;
use rand::Rng;
use spaces::continuous::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Cosine {
    pub mu: f64,
    pub s: f64,
}

impl Cosine {
    pub fn new(mu: f64, s: f64) -> Cosine {
        Cosine { mu, s }
    }
}

impl Distribution for Cosine {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::bounded(self.mu - self.s, self.mu + self.s)
    }

    fn cdf(&self, x: f64) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Cosine {
    fn pdf(&self, x: f64) -> Probability {
        unimplemented!()
    }
}

impl UnivariateMoments for Cosine {
    fn mean(&self) -> f64 { self.mu }

    fn variance(&self) -> f64 {
        self.s * self.s * (ONE_THIRD - TWO_OVER_PI2)
    }

    fn skewness(&self) -> f64 { 0.0 }

    fn excess_kurtosis(&self) -> f64 {
        let v = PI2 - 6.0;

        6.0 * (60.0 - PI4) / 5.0 / v / v
    }
}

impl Quantiles for Cosine {
    fn quantile(&self, p: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 { self.mu }
}

impl Modes for Cosine {
    fn modes(&self) -> Vec<f64> { vec![self.mu] }
}

impl fmt::Display for Cosine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cosine({}, {})", self.mu, self.s)
    }
}
