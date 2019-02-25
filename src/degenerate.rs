use crate::core::*;
use rand::Rng;
use spaces::{
    continuous::Reals,
    discrete::Integers,
};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Degenerate<T = f64> {
    pub k: T,
}

impl<T> Degenerate<T> {
    pub fn new(k: T) -> Degenerate<T> {
        Degenerate { k, }
    }
}

// Continuous variant:
impl Default for Degenerate<f64> {
    fn default() -> Degenerate<f64> {
        Degenerate::new(0.0)
    }
}

impl Distribution for Degenerate<f64> {
    type Support = Reals;

    fn support(&self) -> Reals {
        Reals
    }

    fn cdf(&self, x: f64) -> Probability {
        if x < self.k {
            0.0
        } else {
            1.0
        }.into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        self.k
    }
}

impl ContinuousDistribution for Degenerate<f64> {
    fn pdf(&self, x: f64) -> Probability {
        if (x - self.k).abs() < 1e-7 {
            1.0
        } else {
            0.0
        }.into()
    }
}

impl UnivariateMoments for Degenerate<f64> {
    fn mean(&self) -> f64 {
        self.k
    }

    fn variance(&self) -> f64 {
        0.0
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        panic!("Kurtosis is undefined for the degenerate distribution.")
    }

    fn excess_kurtosis(&self) -> f64 {
        panic!("Kurtosis is undefined for the degenerate distribution.")
    }
}

impl Quantiles for Degenerate<f64> {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.k
    }
}

impl Modes for Degenerate<f64> {
    fn modes(&self) -> Vec<f64> {
        vec![self.k]
    }
}

// Discrete variant:
impl Default for Degenerate<i64> {
    fn default() -> Degenerate<i64> {
        Degenerate::new(0)
    }
}

impl Distribution for Degenerate<i64> {
    type Support = Integers;

    fn support(&self) -> Integers {
        Integers
    }

    fn cdf(&self, x: i64) -> Probability {
        if x < self.k {
            0.0
        } else {
            1.0
        }.into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> i64 {
        self.k
    }
}

impl DiscreteDistribution for Degenerate<i64> {
    fn pmf(&self, x: i64) -> Probability {
        if x == self.k {
            1.0
        } else {
            0.0
        }.into()
    }
}

impl UnivariateMoments for Degenerate<i64> {
    fn mean(&self) -> f64 {
        self.k as f64
    }

    fn variance(&self) -> f64 {
        0.0
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        panic!("Kurtosis is undefined for the degenerate distribution.")
    }

    fn excess_kurtosis(&self) -> f64 {
        panic!("Kurtosis is undefined for the degenerate distribution.")
    }
}

impl Quantiles for Degenerate<i64> {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.k as f64
    }
}

impl Modes for Degenerate<i64> {
    fn modes(&self) -> Vec<i64> {
        vec![self.k]
    }
}

// Generalised:
impl<T> Entropy for Degenerate<T> {
    fn entropy(&self) -> f64 {
        0.0
    }
}

impl fmt::Display for Degenerate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "δ(x - {})", self.k)
    }
}
