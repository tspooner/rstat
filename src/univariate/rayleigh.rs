use crate::{
    consts::{PI, PI2, PI_OVER_2, THREE_HALVES},
    statistics::{Modes, Quantiles, ShannonEntropy, UvMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::real::{PositiveReals, positive_reals};
use std::fmt;

const TWO_PI_MINUS_3: f64 = 2.0 * (PI - 3.0);
const FOUR_MINUS_PI: f64 = 4.0 - PI;
const FOUR_MINUS_PI_OVER_2: f64 = FOUR_MINUS_PI / 2.0;

const EXCESS_KURTOSIS: f64 =
    1.5 * PI2 - 6.0 * PI + 16.0 / FOUR_MINUS_PI_OVER_2 / FOUR_MINUS_PI_OVER_2;
const KURTOSIS: f64 = EXCESS_KURTOSIS + 3.0;

pub use crate::params::Shape;

params! {
    #[derive(Copy)]
    Params {
        sigma: Shape<f64>
    }
}

new_dist!(Rayleigh<Params>);

macro_rules! get_sigma {
    ($self:ident) => {
        $self.0.sigma.0
    };
}

impl Rayleigh {
    pub fn new(sigma: f64) -> Result<Rayleigh, failure::Error> { Params::new(sigma).map(Rayleigh) }

    pub fn new_unchecked(sigma: f64) -> Rayleigh { Rayleigh(Params::new_unchecked(sigma)) }
}

impl Distribution for Rayleigh {
    type Support = PositiveReals<f64>;
    type Params = Params;

    fn support(&self) -> PositiveReals<f64> { positive_reals() }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let sigma = get_sigma!(self);

        Probability::new_unchecked(1.0 - (-x * x / sigma * sigma / 2.0).exp())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for Rayleigh {
    fn pdf(&self, x: &f64) -> f64 {
        let sigma = get_sigma!(self);
        let sigma2 = sigma * sigma;

        x / sigma2 * (-x * x / sigma2 / 2.0).exp()
    }
}

impl Univariate for Rayleigh {}

impl UvMoments for Rayleigh {
    fn mean(&self) -> f64 { get_sigma!(self) * PI_OVER_2.sqrt() }

    fn variance(&self) -> f64 {
        let sigma = get_sigma!(self);

        FOUR_MINUS_PI_OVER_2 * sigma * sigma
    }

    fn skewness(&self) -> f64 { TWO_PI_MINUS_3 * PI.sqrt() / FOUR_MINUS_PI.powf(THREE_HALVES) }

    fn kurtosis(&self) -> f64 { KURTOSIS }

    fn excess_kurtosis(&self) -> f64 { EXCESS_KURTOSIS }
}

impl Quantiles for Rayleigh {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 { get_sigma!(self) * (2.0 * 2.0f64.ln()).sqrt() }
}

impl Modes for Rayleigh {
    fn modes(&self) -> Vec<f64> { vec![get_sigma!(self)] }
}

impl ShannonEntropy for Rayleigh {
    fn shannon_entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let gamma = -(1.0f64.digamma());

        1.0 + (get_sigma!(self) / 2.0f64.sqrt()).ln() + gamma / 2.0
    }
}

impl fmt::Display for Rayleigh {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Rayleigh({})", get_sigma!(self))
    }
}
