use crate::{
    consts::{THREE_FIFTHS, THREE_HALVES, TWELVE_FIFTHS},
    core::*,
};
use rand::Rng;
use spaces::continuous::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Triangular {
    pub a: f64,
    pub b: f64,
    pub c: f64,
}

impl Triangular {
    pub fn new(a: f64, b: f64, c: f64) -> Triangular {
        if b <= a {
            panic!("b must be strictly greater than a.")
        }

        if c < a || c > b {
            panic!("c must lie in the interval [a, b].")
        }

        Triangular { a, b, c }
    }

    pub fn symmetric(a: f64, b: f64) -> Triangular {
        Triangular::new(a, b, (a + b) / 2.0)
    }
}

impl Distribution for Triangular {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::bounded(self.a, self.b)
    }

    fn cdf(&self, x: f64) -> Probability {
        if x <= self.a {
            0.0
        } else if x <= self.c {
            (x - self.a) * (x - self.a) / (self.b - self.a) / (self.c - self.a)
        } else if x <= self.b {
            1.0 - (self.b - x) * (self.b - x) / (self.b - self.a) / (self.b - self.c)
        } else {
            1.0
        }
        .into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::distributions::{Triangular as TriangularSampler, Distribution as DistSampler};

        TriangularSampler::new(self.a, self.b, self.c).sample(rng)
    }
}

impl ContinuousDistribution for Triangular {
    fn pdf(&self, x: f64) -> Probability {
        if x <= self.a {
            0.0
        } else if x < self.c {
            2.0 * (x - self.a) / (self.b - self.a) / (self.c - self.a)
        } else if (x - self.c).abs() < 1e-7 {
            2.0 / (self.b - self.a)
        } else if x <= self.b {
            2.0 * (self.b - x) / (self.b - self.a) / (self.b - self.c)
        } else {
            0.0
        }
        .into()
    }
}

impl UnivariateMoments for Triangular {
    fn mean(&self) -> f64 {
        (self.a + self.b + self.c) / 2.0
    }

    fn variance(&self) -> f64 {
        let sq_terms = self.a * self.a + self.b * self.b + self.c * self.c;
        let cross_terms = self.a * self.b + self.a * self.c + self.b * self.c;

        (sq_terms - cross_terms) / 18.0
    }

    fn skewness(&self) -> f64 {
        let sq_terms = self.a * self.a + self.b * self.b + self.c * self.c;
        let cross_terms = self.a * self.b + self.a * self.c + self.b * self.c;

        let numerator = 2.0f64.sqrt()
            * (self.a + self.b - 2.0 * self.c)
            * (2.0 * self.a - self.b - self.c)
            * (self.a - 2.0 * self.b + self.c);
        let denominator = 5.0 * (sq_terms - cross_terms).powf(THREE_HALVES);

        numerator / denominator
    }

    fn kurtosis(&self) -> f64 {
        TWELVE_FIFTHS
    }

    fn excess_kurtosis(&self) -> f64 {
        -THREE_FIFTHS
    }
}

impl Quantiles for Triangular {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        let midpoint = (self.a + self.b) / 2.0;

        if self.c >= midpoint {
            self.a + ((self.b - self.a) * (self.c - self.a) / 2.0).sqrt()
        } else {
            self.b - ((self.b - self.a) * (self.b - self.c) / 2.0).sqrt()
        }
    }
}

impl Modes for Triangular {
    fn modes(&self) -> Vec<f64> {
        vec![self.c]
    }
}

impl Entropy for Triangular {
    fn entropy(&self) -> f64 {
        1.0 + ((self.b - self.a) / 2.0).ln()
    }
}

impl fmt::Display for Triangular {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Triangular({}, {}, {})", self.a, self.b, self.c)
    }
}
