use consts::{NINE_FIFTHS, SIX_FIFTHS};
use core::*;
use rand::Rng;
use spaces::continuous::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    pub a: f64,
    pub b: f64,

    prob: f64,
}

impl Uniform {
    pub fn new(a: f64, b: f64) -> Uniform {
        if b <= a {
            panic!("b must be strictly greater than a.")
        }

        Uniform {
            a,
            b,
            prob: 1.0 / (b - a),
        }
    }
}

impl Default for Uniform {
    fn default() -> Uniform {
        Uniform {
            a: 0.0,
            b: 1.0,
            prob: 1.0,
        }
    }
}

impl Distribution for Uniform {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::bounded(self.a, self.b)
    }

    fn cdf(&self, x: f64) -> Probability {
        if x < self.a {
            0.0
        } else if x >= self.b {
            1.0
        } else {
            (x - self.a) * self.prob
        }
        .into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Uniform {
    fn pdf(&self, x: f64) -> Probability {
        if x < self.a || x > self.b {
            0.0
        } else {
            self.prob
        }
        .into()
    }
}

impl UnivariateMoments for Uniform {
    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }

    fn variance(&self) -> f64 {
        let width = self.b - self.a;

        width * width / 12.0
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        NINE_FIFTHS
    }

    fn excess_kurtosis(&self) -> f64 {
        -SIX_FIFTHS
    }
}

impl Quantiles for Uniform {
    fn quantile(&self, p: Probability) -> f64 {
        self.a + f64::from(p) * (self.b - self.a)
    }

    fn median(&self) -> f64 {
        (self.a + self.b) / 2.0
    }
}

impl Entropy for Uniform {
    fn entropy(&self) -> f64 {
        (self.b - self.a).ln()
    }
}

impl fmt::Display for Uniform {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "U({}, {})", self.a, self.b)
    }
}
