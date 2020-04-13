use crate::prelude::*;
use ndarray::Array2;
use rand::Rng;
use spaces::real::PositiveReals;
use std::{fmt, ops::Not};

pub use crate::params::Rate;

new_dist!(Exponential<Rate<f64>>);

macro_rules! get_lambda {
    ($self:ident) => { ($self.0).0 }
}

impl Exponential {
    pub fn new(lambda: f64) -> Result<Exponential, failure::Error> {
        Ok(Exponential(Rate::new(lambda)?))
    }

    pub fn new_unchecked(lambda: f64) -> Exponential {
        Exponential(Rate(lambda))
    }

    #[inline(always)]
    pub fn lambda(&self) -> f64 { get_lambda!(self) }

    #[inline(always)]
    pub fn mu(&self) -> f64 { 1.0 / get_lambda!(self) }
}

impl Default for Exponential {
    fn default() -> Exponential { Exponential(Rate(1.0)) }
}

impl Distribution for Exponential {
    type Support = PositiveReals;
    type Params = Rate<f64>;

    fn support(&self) -> PositiveReals { PositiveReals }

    fn params(&self) -> Rate<f64> { self.0 }

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

impl UnivariateMoments for Exponential {
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
    fn quantile(&self, p: Probability) -> f64 {
        -(p.not().unwrap()) / get_lambda!(self)
    }

    fn median(&self) -> f64 {
        self.mean() * 2.0f64.ln()
    }
}

impl Modes for Exponential {
    fn modes(&self) -> Vec<f64> { vec![0.0] }
}

impl Entropy for Exponential {
    fn entropy(&self) -> f64 {
        1.0 - get_lambda!(self).ln()
    }
}

impl FisherInformation for Exponential {
    fn fisher_information(&self) -> Array2<f64> {
        let l = get_lambda!(self);

        Array2::from_elem((1, 1), l * l)
    }
}

impl fmt::Display for Exponential {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Exp({})", self.lambda())
    }
}
