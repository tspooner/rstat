use crate::{
    consts::{ONE_EIGHTH, ONE_OVER_PI, PI_OVER_4, THREE_HALVES, TWO_OVER_PI},
    core::*,
};
use rand::Rng;
use spaces::continuous::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Arcsine {
    pub a: f64,
    pub b: f64,
}

impl Arcsine {
    pub fn new(a: f64, b: f64) -> Arcsine {
        Arcsine { a, b }
    }
}

impl Default for Arcsine {
    fn default() -> Arcsine {
        Arcsine { a: 0.0, b: 1.0 }
    }
}

impl Distribution for Arcsine {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::bounded(self.a, self.b)
    }

    fn cdf(&self, x: f64) -> Probability {
        let xab = (x - self.a) / (self.b - self.a);

        (TWO_OVER_PI * xab.sqrt().asin()).into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Arcsine {
    fn pdf(&self, x: f64) -> Probability {
        let xab = (x - self.a) * (self.b - x);

        (ONE_OVER_PI / xab.sqrt()).into()
    }
}

impl UnivariateMoments for Arcsine {
    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }

    fn variance(&self) -> f64 {
        let diff = self.b - self.a;

        ONE_EIGHTH * diff * diff
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        THREE_HALVES
    }

    fn excess_kurtosis(&self) -> f64 {
        -THREE_HALVES
    }
}

impl Quantiles for Arcsine {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        (self.a + self.b) / 2.0
    }
}

impl Modes for Arcsine {
    fn modes(&self) -> Vec<f64> {
        vec![self.a, self.b]
    }
}

impl Entropy for Arcsine {
    fn entropy(&self) -> f64 {
        PI_OVER_4.ln()
    }
}

impl fmt::Display for Arcsine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Arcsine({}, {})", self.a, self.b)
    }
}
