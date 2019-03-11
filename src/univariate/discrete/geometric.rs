use crate::core::*;
use rand::Rng;
use spaces::discrete::NonNegativeIntegers;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Geometric {
    pub p: Probability,

    q: Probability,
}

impl Geometric {
    pub fn new<P: Into<Probability>>(p: P) -> Geometric {
        let p: Probability = p.into();

        Geometric {
            p,
            q: !p,
        }
    }
}

impl Distribution for Geometric {
    type Support = NonNegativeIntegers;

    fn support(&self) -> NonNegativeIntegers { NonNegativeIntegers }

    fn cdf(&self, k: u64) -> Probability {
        (1.0 - f64::from(self.q).powi(k as i32 + 1)).into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> u64 {
        unimplemented!()
    }
}

impl DiscreteDistribution for Geometric {
    fn pmf(&self, k: u64) -> Probability {
        (self.p * self.q.powi(k as i32)).into()
    }
}

impl UnivariateMoments for Geometric {
    fn mean(&self) -> f64 {
        f64::from(self.q / self.p)
    }

    fn variance(&self) -> f64 {
        f64::from(self.q / self.p / self.p)
    }

    fn skewness(&self) -> f64 {
        (2.0 - f64::from(self.p)) / f64::from(self.q).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        6.0 + 1.0 / self.variance()
    }
}

impl Quantiles for Geometric {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        (-1.0 / f64::from(self.q).log2()) - 1.0
    }
}

impl Modes for Geometric {
    fn modes(&self) -> Vec<u64> {
        vec![0]
    }
}

impl Entropy for Geometric {
    fn entropy(&self) -> f64 {
        let p = f64::from(self.p);
        let q = f64::from(self.q);

        (-q * q.log2() - p * p.log2()) / p
    }
}

impl fmt::Display for Geometric {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Geometric({})", f64::from(self.p))
    }
}
