use crate::{
    consts::{ONE_OVER_PI, ONE_THIRD, PI, PI2, PI4, TWO_OVER_PI2},
    statistics::{Modes, Quantiles, UnivariateMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
};
use rand::Rng;
use spaces::real::Interval;
use std::fmt;

locscale_params! {
    #[derive(Copy)]
    Params {
        mu<f64>,
        s<f64>
    }
}

new_dist!(Cosine<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.s.0)
    };
}

impl Cosine {
    pub fn new(mu: f64, s: f64) -> Result<Cosine, failure::Error> {
        Params::new(mu, s).map(Cosine)
    }

    pub fn new_unchecked(mu: f64, s: f64) -> Cosine { Cosine(Params::new_unchecked(mu, s)) }

    #[inline]
    fn z(&self, x: f64) -> f64 {
        let (mu, s) = get_params!(self);

        (x - mu) / s
    }

    #[inline]
    fn hvc(&self, x: f64) -> f64 { ((self.z(x) * PI).cos() + 1.0) / 2.0 }
}

impl Distribution for Cosine {
    type Support = Interval;
    type Params = Params;

    fn support(&self) -> Interval {
        let (mu, s) = get_params!(self);

        Interval::bounded(mu - s, mu + s)
    }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let z = self.z(*x);

        Probability::new_unchecked(0.5 * (1.0 + z + ONE_OVER_PI * (z * PI).sin()))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for Cosine {
    fn pdf(&self, x: &f64) -> f64 { 0.5 * self.hvc(*x) }
}

impl UnivariateMoments for Cosine {
    fn mean(&self) -> f64 { self.0.mu.0 }

    fn variance(&self) -> f64 {
        let s = self.0.s.0;

        s * s * (ONE_THIRD - TWO_OVER_PI2)
    }

    fn skewness(&self) -> f64 { 0.0 }

    fn excess_kurtosis(&self) -> f64 {
        let v = PI2 - 6.0;

        6.0 * (60.0 - PI4) / 5.0 / v / v
    }
}

impl Quantiles for Cosine {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 { self.0.mu.0 }
}

impl Modes for Cosine {
    fn modes(&self) -> Vec<f64> { vec![self.0.mu.0] }
}

impl fmt::Display for Cosine {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mu, s) = get_params!(self);

        write!(f, "Cosine({}, {})", mu, s)
    }
}
