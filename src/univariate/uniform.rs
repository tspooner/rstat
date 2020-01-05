use crate::{consts::{NINE_FIFTHS, SIX_FIFTHS}, prelude::*, validation::{Validator, Result}};
use rand::Rng;
use spaces::{
    real::Interval as RealInterval,
    discrete::Interval as DiscreteInterval,
};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Uniform<T> {
    pub a: T,
    pub b: T,

    prob: f64,
}

impl<T> Into<rand_distr::Uniform<T>> for Uniform<T>
where
    T: rand_distr::uniform::SampleUniform,
{
    fn into(self) -> rand_distr::Uniform<T> {
        rand_distr::Uniform::new(self.a, self.b)
    }
}

impl<T: Clone> Into<rand_distr::Uniform<T>> for &Uniform<T>
where
    T: rand_distr::uniform::SampleUniform,
{
    fn into(self) -> rand_distr::Uniform<T> {
        rand_distr::Uniform::new(self.a.clone(), self.b.clone())
    }
}

// Continuous:
impl Uniform<f64> {
    pub fn new(a: f64, b: f64) -> Result<Uniform<f64>> {
        Validator
            .require_lte(a, b)
            .map(|_| Self::new_unchecked(a, b))
    }

    pub fn new_unchecked(a: f64, b: f64) -> Uniform<f64> {
        Uniform {
            a, b,
            prob: 1.0 / (b - a),
        }
    }
}

impl Default for Uniform<f64> {
    fn default() -> Uniform<f64> {
        Uniform {
            a: 0.0,
            b: 1.0,
            prob: 1.0,
        }
    }
}

impl Distribution for Uniform<f64> {
    type Support = RealInterval;

    fn support(&self) -> RealInterval {
        RealInterval::bounded(self.a, self.b)
    }

    fn cdf(&self, x: f64) -> Probability {
        if x < self.a {
            Probability::zero()
        } else if x >= self.b {
            Probability::one()
        } else {
            Probability::new_unchecked((x - self.a) * self.prob)
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        let sampler: rand_distr::Uniform<f64> = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for Uniform<f64> {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.a || x > self.b {
            0.0
        } else {
            self.prob
        }
    }
}

impl UnivariateMoments for Uniform<f64> {
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

impl Quantiles for Uniform<f64> {
    fn quantile(&self, p: Probability) -> f64 {
        self.a + f64::from(p) * (self.b - self.a)
    }

    fn median(&self) -> f64 {
        (self.a + self.b) / 2.0
    }
}

impl Entropy for Uniform<f64> {
    fn entropy(&self) -> f64 {
        (self.b - self.a).ln()
    }
}

impl fmt::Display for Uniform<f64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "U({}, {})", self.a, self.b)
    }
}

// Discrete:
impl Uniform<i64> {
    pub fn new(a: i64, b: i64) -> Result<Uniform<i64>> {
        Validator
            .require_lte(a, b)
            .map(|_| Self::new_unchecked(a, b))
    }

    pub fn new_unchecked(a: i64, b: i64) -> Uniform<i64> {
        Uniform {
            a, b,
            prob: 1.0 / (b - a + 1) as f64
        }
    }

    #[inline]
    pub fn span(&self) -> u64 {
        (self.b - self.a + 1) as u64
    }
}

impl Distribution for Uniform<i64> {
    type Support = DiscreteInterval;

    fn support(&self) -> DiscreteInterval {
        DiscreteInterval::bounded(self.a, self.b)
    }

    fn cdf(&self, k: i64) -> Probability {
        if k < self.a {
            Probability::zero()
        } else if k >= self.b {
            Probability::one()
        } else {
            Probability::new_unchecked((k - self.a + 1) as f64 * self.prob)
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i64 {
        use rand_distr::Distribution;

        let sampler: rand_distr::Uniform<i64> = self.into();

        sampler.sample(rng)
    }
}

impl DiscreteDistribution for Uniform<i64> {
    fn pmf(&self, x: i64) -> Probability {
        if x < self.a || x > self.b {
            Probability::zero()
        } else {
            Probability::new_unchecked(self.prob)
        }
    }
}

impl UnivariateMoments for Uniform<i64> {
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

impl Quantiles for Uniform<i64> {
    fn quantile(&self, p: Probability) -> f64 {
        let n = self.span() as f64;

        self.a as f64 + (f64::from(p) * n).floor()
    }

    fn median(&self) -> f64 {
        (self.a + self.b) as f64 / 2.0
    }
}

impl Entropy for Uniform<i64> {
    fn entropy(&self) -> f64 {
        let n = (self.b - self.a + 1) as f64;

        n.ln()
    }
}

impl fmt::Display for Uniform<i64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "U{{{}, {}}}", self.a, self.b)
    }
}
