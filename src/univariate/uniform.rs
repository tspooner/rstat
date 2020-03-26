use crate::{
    consts::{NINE_FIFTHS, SIX_FIFTHS},
    prelude::*,
    constraints::{self, Constraint},
};
use failure::Error;
use rand::Rng;
use spaces::{
    real::Interval as RealInterval,
    discrete::Interval as DiscreteInterval,
};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Uniform<T> {
    loc: T,
    scale: T,
    prob: f64,
}

impl<T> Into<rand_distr::Uniform<T>> for Uniform<T>
where
    T: rand_distr::uniform::SampleUniform + Clone + std::ops::Add<Output = T>,
{
    fn into(self) -> rand_distr::Uniform<T> {
        rand_distr::Uniform::new(self.loc.clone(), self.loc.clone() + self.scale.clone())
    }
}

impl<T> Into<rand_distr::Uniform<T>> for &Uniform<T>
where
    T: rand_distr::uniform::SampleUniform + Clone + std::ops::Add<Output = T>,
{
    fn into(self) -> rand_distr::Uniform<T> {
        rand_distr::Uniform::new(self.loc.clone(), self.loc.clone() + self.scale.clone())
    }
}

// Continuous:
impl Uniform<f64> {
    pub fn new(loc: f64, scale: f64) -> Result<Uniform<f64>, Error> {
        let scale = constraints::Positive.check(scale)?;

        Ok(Uniform::new_unchecked(loc, scale))
    }

    pub fn new_unchecked(loc: f64, scale: f64) -> Uniform<f64> {
        Uniform {
            loc,
            scale,
            prob: 1.0 / scale,
        }
    }
}

impl Default for Uniform<f64> {
    fn default() -> Uniform<f64> {
        Uniform {
            loc: 0.0,
            scale: 1.0,
            prob: 1.0,
        }
    }
}

impl Distribution for Uniform<f64> {
    type Support = RealInterval;

    fn support(&self) -> RealInterval {
        RealInterval::bounded(self.loc, self.loc + self.scale)
    }

    fn cdf(&self, x: f64) -> Probability {
        if x < self.loc {
            Probability::zero()
        } else if x >= (self.loc + self.scale) {
            Probability::one()
        } else {
            Probability::new_unchecked((x - self.loc) * self.prob)
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
        if x < self.loc || x > (self.loc + self.scale) {
            0.0
        } else {
            self.prob
        }
    }
}

impl UnivariateMoments for Uniform<f64> {
    fn mean(&self) -> f64 {
        self.loc + self.scale / 2.0
    }

    fn variance(&self) -> f64 {
        self.scale * self.scale / 12.0
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
        self.loc + p * self.scale
    }

    fn median(&self) -> f64 {
        self.loc + self.scale / 2.0
    }
}

impl Entropy for Uniform<f64> {
    fn entropy(&self) -> f64 {
        self.scale.ln()
    }
}

impl fmt::Display for Uniform<f64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "U({}, {})", self.loc, self.loc + self.scale)
    }
}

// Discrete:
impl Uniform<i64> {
    pub fn new(loc: i64, scale: u64) -> Uniform<i64> {
        Uniform {
            loc,
            scale: scale as i64,
            prob: 1.0 / (scale + 1) as f64
        }
    }

    #[inline]
    pub fn span(&self) -> u64 {
        (self.scale + 1) as u64
    }
}

impl Distribution for Uniform<i64> {
    type Support = DiscreteInterval;

    fn support(&self) -> DiscreteInterval {
        DiscreteInterval::bounded(self.loc, self.loc + self.scale)
    }

    fn cdf(&self, k: i64) -> Probability {
        if k < self.loc {
            Probability::zero()
        } else if k >= self.loc + self.scale {
            Probability::one()
        } else {
            Probability::new_unchecked((k - self.loc + 1) as f64 * self.prob)
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
        if x < self.loc || x > self.loc + self.scale {
            Probability::zero()
        } else {
            Probability::new_unchecked(self.prob)
        }
    }
}

impl UnivariateMoments for Uniform<i64> {
    fn mean(&self) -> f64 {
        self.loc as f64 + self.scale as f64 / 2.0
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

        self.loc as f64 + (p * n).floor()
    }

    fn median(&self) -> f64 {
        self.loc as f64 + self.scale as f64 / 2.0
    }
}

impl Entropy for Uniform<i64> {
    fn entropy(&self) -> f64 {
        let n = (self.scale + 1) as f64;

        n.ln()
    }
}

impl fmt::Display for Uniform<i64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "U{{{}, {}}}", self.loc, self.loc + self.scale)
    }
}
