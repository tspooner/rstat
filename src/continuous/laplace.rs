use consts::E;
use core::*;
use rand::Rng;
use spaces::continuous::Reals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Laplace {
    pub mu: f64,
    pub b: f64,
}

impl Laplace {
    pub fn new(mu: f64, b: f64) -> Laplace {
        assert_positive_real!(b);

        Laplace { mu, b }
    }
}

impl Default for Laplace {
    fn default() -> Laplace {
        Laplace { mu: 0.0, b: 1.0 }
    }
}

impl Distribution for Laplace {
    type Support = Reals;

    fn support(&self) -> Reals { Reals }

    fn cdf(&self, x: f64) -> Probability {
        ((-((x - self.mu).abs() / self.b).abs()).exp() / 2.0 / self.b).into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Laplace {
    fn pdf(&self, x: f64) -> Probability {
        use std::cmp::Ordering::*;

        match x.partial_cmp(&self.mu) {
            Some(Less) | Some(Equal) => {
                ((x - self.mu) / self.b).exp() / 2.0
            },
            Some(Greater) => {
                1.0 - ((self.mu - x) / self.b).exp() / 2.0
            },
            None => unreachable!(),
        }.into()
    }
}

impl UnivariateMoments for Laplace {
    fn mean(&self) -> f64 { self.mu }

    fn variance(&self) -> f64 { 2.0 * self.b * self.b }

    fn skewness(&self) -> f64 { unimplemented!() }

    fn kurtosis(&self) -> f64 { 6.0 }

    fn excess_kurtosis(&self) -> f64 { 3.0 }
}

impl Quantiles for Laplace {
    fn quantile(&self, p: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 { self.mu }
}

impl Modes for Laplace {
    fn modes(&self) -> Vec<f64> { vec![self.mu] }
}

impl Entropy for Laplace {
    fn entropy(&self) -> f64 { (2.0 * self.b * E).ln() }
}

impl fmt::Display for Laplace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Laplace({}, {})", self.mu, self.b)
    }
}
