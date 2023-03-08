use crate::{
    consts::{ONE_OVER_PI, PI_OVER_4, THREE_HALVES, TWO_OVER_PI},
    statistics::{Modes, Quantiles, ShannonEntropy, UvMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::intervals::Closed;
use std::fmt;

locscale_params! {
    #[derive(Copy)]
    Params { a<f64>, w<f64> }
}

impl Params {
    #[inline(always)]
    pub fn b(&self) -> crate::params::Loc<f64> { crate::params::Loc(self.a.0 + self.w.0) }
}

new_dist!(Arcsine<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.a.0, $self.0.b().0)
    };
}

impl Arcsine {
    pub fn new(a: f64, w: f64) -> Result<Arcsine, failure::Error> {
        Params::new(a, w).map(Arcsine)
    }

    pub fn new_unchecked(a: f64, w: f64) -> Arcsine { Arcsine(Params::new_unchecked(a, w)) }
}

impl Default for Arcsine {
    fn default() -> Arcsine { Arcsine::new_unchecked(0.0, 1.0) }
}

impl Distribution for Arcsine {
    type Support = Closed<f64>;
    type Params = Params;

    fn support(&self) -> Closed<f64> {
        let (a, b) = get_params!(self);

        Closed::closed_unchecked(a, b)
    }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let a = self.0.a.0;
        let w = self.0.w.0;

        let xab = (x - a) / w;

        Probability::new_unchecked(TWO_OVER_PI * xab.sqrt().asin())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for Arcsine {
    fn pdf(&self, x: &f64) -> f64 {
        let (a, b) = get_params!(self);
        let xab = (x - a) * (b - x);

        ONE_OVER_PI / xab.sqrt()
    }
}

impl Univariate for Arcsine {}

impl UvMoments for Arcsine {
    fn mean(&self) -> f64 {
        let (a, b) = get_params!(self);

        (a + b) / 2.0
    }

    fn variance(&self) -> f64 {
        let w = self.0.w.0;

        w * w / 8.0
    }

    fn skewness(&self) -> f64 { 0.0 }

    fn kurtosis(&self) -> f64 { THREE_HALVES }

    fn excess_kurtosis(&self) -> f64 { -THREE_HALVES }
}

impl Quantiles for Arcsine {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 { self.mean() }
}

impl Modes for Arcsine {
    fn modes(&self) -> Vec<f64> {
        let (a, b) = get_params!(self);

        vec![a, b]
    }
}

impl ShannonEntropy for Arcsine {
    fn shannon_entropy(&self) -> f64 { PI_OVER_4.ln() }
}

impl fmt::Display for Arcsine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (a, b) = get_params!(self);

        write!(f, "Arcsine({}, {})", a, b)
    }
}

#[cfg(test)]
mod tests {
    use super::{Arcsine, Modes, Quantiles, UvMoments};

    #[test]
    fn test_mean() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).mean(), 0.5);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).mean(), -0.5);
        assert_eq!(Arcsine::new_unchecked(-1.0, 2.0).mean(), 0.0);
    }

    #[test]
    fn test_variance() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).variance(), 1.0 / 8.0);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).variance(), 1.0 / 8.0);
        assert_eq!(Arcsine::new_unchecked(-1.0, 2.0).variance(), 1.0 / 2.0);
    }

    #[test]
    fn test_skewness() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).skewness(), 0.0);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).skewness(), 0.0);
        assert_eq!(Arcsine::new_unchecked(-1.0, 2.0).skewness(), 0.0);
    }

    #[test]
    fn test_median() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).median(), 0.5);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).median(), -0.5);
        assert_eq!(Arcsine::new_unchecked(-1.0, 2.0).median(), 0.0);
    }

    #[test]
    fn test_modes() {
        assert_eq!(Arcsine::new_unchecked(0.0, 1.0).modes(), vec![0.0, 1.0]);
        assert_eq!(Arcsine::new_unchecked(-1.0, 1.0).modes(), vec![-1.0, 0.0]);
        assert_eq!(Arcsine::new_unchecked(-1.0, 2.0).modes(), vec![-1.0, 1.0]);
    }
}
