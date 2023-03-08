use crate::{
    ContinuousDistribution,
    Distribution,
    DiscreteDistribution,
    Probability,
    Univariate,
    statistics::{UvMoments, Quantiles, Modes, ShannonEntropy},
};
use rand::Rng;
use spaces::{
    real::Reals,
    discrete::Integers,
};
use std::fmt;

pub use crate::params::Loc;

#[derive(Debug, Clone)]
pub struct Degenerate<T: num::Zero + PartialOrd> {
    pub k: Loc<T>,
}

impl<T: num::Zero + PartialOrd> Degenerate<T> {
    pub fn new(k: T) -> Degenerate<T> {
        Degenerate { k: Loc(k), }
    }
}

impl<T: num::Zero + PartialOrd> From<Loc<T>> for Degenerate<T> {
    fn from(loc: Loc<T>) -> Degenerate<T> {
        Degenerate { k: loc, }
    }
}

impl<T: num::Zero + PartialOrd + Default> Default for Degenerate<T> {
    fn default() -> Degenerate<T> {
        Degenerate::new(T::default())
    }
}

impl<T: num::Zero + PartialOrd> Univariate for Degenerate<T>
where
    Degenerate<T>: Distribution,
{}

// Continuous variant:
impl Distribution for Degenerate<f64> {
    type Support = Reals<f64>;
    type Params = Loc<f64>;

    fn support(&self) -> Reals<f64> { Reals::unbounded() }

    fn params(&self) -> Loc<f64> { self.k }

    fn cdf(&self, x: &f64) -> Probability {
        if *x < self.k.0 {
            Probability::zero()
        } else {
            Probability::one()
        }
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { self.k.0 }
}

impl ContinuousDistribution for Degenerate<f64> {
    fn pdf(&self, x: &f64) -> f64 {
        if (x - self.k.0).abs() < 1e-7 { 1.0 } else { 0.0 }
    }
}

impl UvMoments for Degenerate<f64> {
    fn mean(&self) -> f64 { self.k.0 }

    fn variance(&self) -> f64 { 0.0 }

    fn skewness(&self) -> f64 { 0.0 }

    fn kurtosis(&self) -> f64 {
        undefined!("Kurtosis is undefined for the degenerate distribution.")
    }

    fn excess_kurtosis(&self) -> f64 {
        undefined!("Kurtosis is undefined for the degenerate distribution.")
    }
}

impl Quantiles for Degenerate<f64> {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 { self.k.0 }
}

impl Modes for Degenerate<f64> {
    fn modes(&self) -> Vec<f64> { vec![self.k.0] }
}

impl ShannonEntropy for Degenerate<f64> {
    fn shannon_entropy(&self) -> f64 { 0.0 }
}

// Discrete variant:
impl Distribution for Degenerate<i64> {
    type Support = Integers<i64>;
    type Params = Loc<i64>;

    fn support(&self) -> Integers<i64> { Integers::unbounded() }

    fn params(&self) -> Loc<i64> { self.k }

    fn cdf(&self, x: &i64) -> Probability {
        if *x < self.k.0 {
            Probability::zero()
        } else {
            Probability::one()
        }
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> i64 { self.k.0 }
}

impl DiscreteDistribution for Degenerate<i64> {
    fn pmf(&self, x: &i64) -> Probability {
        if *x == self.k.0 {
            Probability::one()
        } else {
            Probability::zero()
        }
    }
}

impl UvMoments for Degenerate<i64> {
    fn mean(&self) -> f64 { self.k.0 as f64 }

    fn variance(&self) -> f64 { 0.0 }

    fn skewness(&self) -> f64 { 0.0 }

    fn kurtosis(&self) -> f64 {
        undefined!("Kurtosis is undefined for the degenerate distribution.")
    }

    fn excess_kurtosis(&self) -> f64 {
        undefined!("Kurtosis is undefined for the degenerate distribution.")
    }
}

impl Quantiles for Degenerate<i64> {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 { self.k.0 as f64 }
}

impl Modes for Degenerate<i64> {
    fn modes(&self) -> Vec<i64> { vec![self.k.0] }
}

impl ShannonEntropy for Degenerate<i64> {
    fn shannon_entropy(&self) -> f64 { 0.0 }
}

impl<T: num::Zero + PartialOrd + fmt::Display> fmt::Display for Degenerate<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "δ(x - {})", self.k.0)
    }
}
