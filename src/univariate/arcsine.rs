use crate::{
    consts::{ONE_OVER_PI, PI_OVER_4, THREE_HALVES, TWO_OVER_PI},
    prelude::*,
    validation::{Validator, Result},
};
use rand::Rng;
use spaces::real::Interval;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Arcsine {
    pub a: f64,
    pub b: f64,
}

impl Arcsine {
    pub fn new(a: f64, b: f64) -> Result<Arcsine> {
        Validator.require_lte(a, b).map(|_| Arcsine::new_unchecked(a, b))
    }

    pub fn new_unchecked(a: f64, b: f64) -> Arcsine { Arcsine { a, b } }
}

impl Default for Arcsine {
    fn default() -> Arcsine {
        Arcsine::new_unchecked(0.0, 1.0)
    }
}

impl Distribution for Arcsine {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::bounded(self.a, self.b)
    }

    fn cdf(&self, x: f64) -> Probability {
        let xab = (x - self.a) / (self.b - self.a);

        Probability::new_unchecked(TWO_OVER_PI * xab.sqrt().asin())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Arcsine {
    fn pdf(&self, x: f64) -> f64 {
        let xab = (x - self.a) * (self.b - x);

        ONE_OVER_PI / xab.sqrt()
    }
}

impl UnivariateMoments for Arcsine {
    fn mean(&self) -> f64 {
        (self.a + self.b) / 2.0
    }

    fn variance(&self) -> f64 {
        let diff = self.b - self.a;

        diff * diff / 8.0
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        THREE_HALVES
    }

    fn excess_kurtosis(&self) -> f64 {
        -THREE_HALVES
    }
}

impl Quantiles for Arcsine {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        (self.a + self.b) / 2.0
    }
}

impl Modes for Arcsine {
    fn modes(&self) -> Vec<f64> {
        vec![self.a, self.b]
    }
}

impl Entropy for Arcsine {
    fn entropy(&self) -> f64 {
        PI_OVER_4.ln()
    }
}

impl fmt::Display for Arcsine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Arcsine({}, {})", self.a, self.b)
    }
}


#[cfg(test)]
mod tests {
    use super::{Arcsine, UnivariateMoments, Quantiles, Modes};

    #[test]
    fn test_mean() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).mean(), 0.5);
        assert_eq!(Arcsine::new_unchecked(-1.0, 0.0).mean(), -0.5);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).mean(), 0.0);
    }

    #[test]
    fn test_variance() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).variance(), 1.0 / 8.0);
        assert_eq!(Arcsine::new_unchecked(-1.0, 0.0).variance(), 1.0 / 8.0);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).variance(), 1.0 / 2.0);
    }

    #[test]
    fn test_skewness() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).skewness(), 0.0);
        assert_eq!(Arcsine::new_unchecked(-1.0, 0.0).skewness(), 0.0);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).skewness(), 0.0);
    }

    #[test]
    fn test_median() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).median(), 0.5);
        assert_eq!(Arcsine::new_unchecked(-1.0, 0.0).median(), -0.5);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).median(), 0.0);
    }

    #[test]
    fn test_mode() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).modes(), vec![0.0, 1.0]);
        assert_eq!(Arcsine::new_unchecked(-1.0, 0.0).modes(), vec![-1.0, 0.0]);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).modes(), vec![-1.0, 1.0]);
    }
}
