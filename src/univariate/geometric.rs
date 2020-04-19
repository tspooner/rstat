use crate::prelude::*;
use rand::Rng;
use spaces::discrete::NonNegativeIntegers;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Geometric {
    pub p: Probability,

    q: Probability,
}

impl Geometric {
    pub fn new(p: Probability) -> Geometric { Geometric { p, q: !p } }
}

impl From<Probability> for Geometric {
    fn from(p: Probability) -> Geometric { Geometric::new(p) }
}

impl Distribution for Geometric {
    type Support = NonNegativeIntegers;
    type Params = Probability;

    fn support(&self) -> NonNegativeIntegers { NonNegativeIntegers }

    fn params(&self) -> Probability { self.p }

    fn cdf(&self, k: &u64) -> Probability {
        Probability::new_unchecked(1.0 - self.q.powi(*k as i32 + 1))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> u64 { unimplemented!() }
}

impl DiscreteDistribution for Geometric {
    fn pmf(&self, k: &u64) -> Probability {
        Probability::new_unchecked(self.p * self.q.powi(*k as i32))
    }
}

impl UnivariateMoments for Geometric {
    fn mean(&self) -> f64 { self.q.unwrap() / self.p.unwrap() }

    fn variance(&self) -> f64 {
        let (p, q) = (self.p.unwrap(), self.q.unwrap());

        q / p / p
    }

    fn skewness(&self) -> f64 {
        let (p, q) = (self.p.unwrap(), self.q.unwrap());

        (2.0 - p) / q.sqrt()
    }

    fn excess_kurtosis(&self) -> f64 { 6.0 + 1.0 / self.variance() }
}

impl Quantiles for Geometric {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 { (-1.0 / self.q.log2()) - 1.0 }
}

impl Modes for Geometric {
    fn modes(&self) -> Vec<u64> { vec![0] }
}

impl ShannonEntropy for Geometric {
    fn shannon_entropy(&self) -> f64 {
        let (p, q) = (self.p.unwrap(), self.q.unwrap());

        (-q * q.log2() - p * p.log2()) / p
    }
}

impl fmt::Display for Geometric {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Geometric({})", self.p) }
}
