use crate::{
    statistics::{Modes, Quantiles, ShannonEntropy, UnivariateMoments},
    DiscreteDistribution,
    Distribution,
};
use rand::Rng;
use spaces::discrete::NonNegativeIntegers;
use std::fmt;

pub use crate::Probability;

params! {
    #[derive(Copy)]
    Params {
        p: Probability<>
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Geometric {
    params: Params,
    q: Probability,
}

impl Geometric {
    pub fn new(p: f64) -> Result<Geometric, failure::Error> { Params::new(p).map(|ps| ps.into()) }

    pub fn new_unchecked(p: f64) -> Geometric { Params::new_unchecked(p).into() }
}

impl From<Params> for Geometric {
    fn from(params: Params) -> Geometric {
        Geometric {
            q: !params.p,
            params,
        }
    }
}

impl Distribution for Geometric {
    type Support = NonNegativeIntegers;
    type Params = Params;

    fn support(&self) -> NonNegativeIntegers { NonNegativeIntegers }

    fn params(&self) -> Params { self.params }

    fn cdf(&self, k: &u64) -> Probability {
        Probability::new_unchecked(1.0 - self.q.powi(*k as i32 + 1))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> u64 { unimplemented!() }
}

impl DiscreteDistribution for Geometric {
    fn pmf(&self, k: &u64) -> Probability {
        Probability::new_unchecked(self.params.p * self.q.powi(*k as i32))
    }
}

impl UnivariateMoments for Geometric {
    fn mean(&self) -> f64 { self.q.unwrap() / self.params.p.unwrap() }

    fn variance(&self) -> f64 {
        let (p, q) = (self.params.p.unwrap(), self.q.unwrap());

        q / p / p
    }

    fn skewness(&self) -> f64 {
        let (p, q) = (self.params.p.unwrap(), self.q.unwrap());

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
        let (p, q) = (self.params.p.unwrap(), self.q.unwrap());

        (-q * q.log2() - p * p.log2()) / p
    }
}

impl fmt::Display for Geometric {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Geometric({})", self.params.p)
    }
}
