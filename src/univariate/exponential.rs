use crate::{
    statistics::{FisherInformation, Modes, Quantiles, ShannonEntropy, UvMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::real::{PositiveReals, positive_reals};
use std::{fmt, ops::Not};

pub use crate::params::Rate;

params! {
    #[derive(Copy)]
    Params {
        lambda: Rate<f64>
    }
}

new_dist!(Exponential<Params>);

macro_rules! get_lambda {
    ($self:ident) => {
        $self.0.lambda.0
    };
}

impl Exponential {
    pub fn new(lambda: f64) -> Result<Exponential, failure::Error> {
        Params::new(lambda).map(Exponential)
    }

    pub fn new_unchecked(lambda: f64) -> Exponential { Exponential(Params::new_unchecked(lambda)) }
}

impl Default for Exponential {
    fn default() -> Exponential { Exponential::new_unchecked(1.0) }
}

impl Distribution for Exponential {
    type Support = PositiveReals<f64>;
    type Params = Params;

    fn support(&self) -> PositiveReals<f64> { positive_reals() }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        Probability::new_unchecked(1.0 - (-get_lambda!(self) * x).exp())
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        rand_distr::Exp::new(get_lambda!(self)).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for Exponential {
    fn pdf(&self, x: &f64) -> f64 {
        let l = get_lambda!(self);

        l * (-l * x).exp()
    }
}

impl Univariate for Exponential {}

impl UvMoments for Exponential {
    fn mean(&self) -> f64 { 1.0 / get_lambda!(self) }

    fn variance(&self) -> f64 {
        let l = get_lambda!(self);

        1.0 / l / l
    }

    fn skewness(&self) -> f64 { 2.0 }

    fn kurtosis(&self) -> f64 { 3.0 }

    fn excess_kurtosis(&self) -> f64 { 6.0 }
}

impl Quantiles for Exponential {
    fn quantile(&self, p: Probability) -> f64 { -(p.not().unwrap()) / get_lambda!(self) }

    fn median(&self) -> f64 { self.mean() * 2.0f64.ln() }
}

impl Modes for Exponential {
    fn modes(&self) -> Vec<f64> { vec![0.0] }
}

impl ShannonEntropy for Exponential {
    fn shannon_entropy(&self) -> f64 { 1.0 - get_lambda!(self).ln() }
}

impl FisherInformation<1> for Exponential {
    fn fisher_information(&self) -> [[f64; 1]; 1] {
        let l = get_lambda!(self);

        [[l * l]]
    }
}

impl fmt::Display for Exponential {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Exp({})", self.0.lambda.0) }
}
