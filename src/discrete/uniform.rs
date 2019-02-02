use consts::SIX_FIFTHS;
use core::*;
use rand::Rng;
use spaces::discrete::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Uniform {
    pub a: i64,
    pub b: i64,

    prob: f64,
}

impl Uniform {
    pub fn new(a: i64, b: i64) -> Uniform {
        if b <= a {
            panic!("b must be strictly greater than a.")
        }

        Uniform { a, b, prob: 1.0 / (b - a + 1) as f64 }
    }

    #[inline]
    pub fn span(&self) -> u64 {
        (self.b - self.a + 1) as u64
    }
}

impl Distribution for Uniform {
    type Support = Interval;

    fn support(&self) -> Interval { Interval::bounded(self.a, self.b) }

    fn cdf(&self, k: i64) -> Probability {
        if k < self.a {
            0.0
        } else if k >= self.b {
            1.0
        } else {
            (k - self.a + 1) as f64 * self.prob
        }.into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> i64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Uniform {
    fn pdf(&self, x: i64) -> Probability {
        if x < self.a || x > self.b {
            0.0
        } else {
            self.prob
        }.into()
    }
}

impl UnivariateMoments for Uniform {
    fn mean(&self) -> f64 {
        (self.a + self.b) as f64 / 2.0
    }

    fn variance(&self) -> f64 {
        let n = self.span() as f64;

        (n * n - 1.0) / 12.0
    }

    fn skewness(&self) -> f64 { 0.0 }

    fn excess_kurtosis(&self) -> f64 {
        let n = self.span() as f64;
        let n2 = n * n;

        -SIX_FIFTHS * (n2 + 1.0) / (n2 - 1.0)
    }
}

impl Quantiles for Uniform {
    fn quantile(&self, p: Probability) -> f64 {
        let n = self.span() as f64;

        self.a as f64 + (f64::from(p) * n).floor()
    }

    fn median(&self) -> f64 {
        (self.a + self.b) as f64 / 2.0
    }
}

impl Entropy for Uniform {
    fn entropy(&self) -> f64 {
        let n = (self.b - self.a + 1) as f64;

        n.ln()
    }
}

impl fmt::Display for Uniform {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "U{{{}, {}}}", self.a, self.b)
    }
}
