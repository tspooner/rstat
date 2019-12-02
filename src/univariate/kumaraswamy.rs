use crate::prelude::*;
use rand;
use spaces::real::Interval;
use std::fmt;

#[inline]
fn harmonic_n(n: usize) -> f64 { (1..=n).map(|i| 1.0 / i as f64).sum() }

#[derive(Debug, Clone, Copy)]
pub struct Kumaraswamy {
    pub a: f64,
    pub b: f64,
}

impl Kumaraswamy {
    pub fn new(a: f64, b: f64) -> Kumaraswamy {
        assert_positive_real!(a);
        assert_positive_real!(b);

        Kumaraswamy { a, b }
    }

    fn moment_n(&self, n: usize) -> f64 {
        use special_fun::FloatSpecial;

        self.b * (1.0 + n as f64 / self.a).beta(self.b)
    }
}

impl Default for Kumaraswamy {
    fn default() -> Kumaraswamy {
        Kumaraswamy {
            a: 1.0,
            b: 1.0,
        }
    }
}

impl Distribution for Kumaraswamy {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::bounded(0.0, 1.0)
    }

    fn cdf(&self, x: f64) -> Probability {
        Probability::new_unchecked(1.0 - (1.0 - x.powf(self.a)).powf(self.b))
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::{Distribution, Uniform};

        let x: f64 = Uniform::new_inclusive(0.0, 1.0).sample(rng);

        (1.0 - (1.0 - x).powf(1.0 / self.b)).powf(1.0 / self.a)
    }
}

impl ContinuousDistribution for Kumaraswamy {
    fn pdf(&self, x: f64) -> f64 {
        self.a * self.b * x.powf(self.a - 1.0) * (1.0 - x.powf(self.a)).powf(self.b - 1.0)
    }
}

impl UnivariateMoments for Kumaraswamy {
    fn mean(&self) -> f64 { self.moment_n(1) }

    fn variance(&self) -> f64 {
        let m1 = self.moment_n(1);

        self.moment_n(2) - m1 * m1
    }

    fn skewness(&self) -> f64 {
        let m3 = self.moment_n(3);
        let var = self.variance();
        let mean = self.mean();

        (m3 - 3.0 * mean * var - mean.powi(3)) / var.powf(3.0 / 2.0)
    }

    fn excess_kurtosis(&self) -> f64 {
        unimplemented!()
    }
}

impl Quantiles for Kumaraswamy {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        (1.0 - 2.0f64.powf(-1.0 / self.b)).powf(1.0 / self.a)
    }
}

impl Modes for Kumaraswamy {
    fn modes(&self) -> Vec<f64> {
        if (self.a > 1.0 && self.b >= 1.0) || (self.a >= 1.0 && self.b > 1.0) {
            vec![((self.a - 1.0) / (self.a * self.b - 1.0)).powf(1.0 / self.a)]
        } else {
            vec![]
        }
    }
}

impl Entropy for Kumaraswamy {
    fn entropy(&self) -> f64 {
        let hb = harmonic_n(self.b.floor() as usize);

        (1.0 - 1.0 / self.b) + (1.0 - 1.0 / self.a) * hb - (self.a * self.b).ln()
    }
}

impl fmt::Display for Kumaraswamy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Kumaraswamy({}, {})", self.a, self.b)
    }
}
