use crate::{
    statistics::{Modes, UvMoments},
    ContinuousDistribution,
    Distribution,
    Univariate,
};
use rand::Rng;
use spaces::real::{PositiveReals, positive_reals};
use special_fun::FloatSpecial;
use std::fmt;

pub use crate::params::{Rate, Shape};

params! {
    #[derive(Copy)]
    Params {
        k: Shape<usize>,
        lambda: Rate<f64>
    }
}

new_dist!(Erlang<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.k.0, $self.0.lambda.0)
    };
}

impl Erlang {
    pub fn new(mu: usize, s: f64) -> Result<Erlang, failure::Error> {
        Params::new(mu, s).map(Erlang)
    }

    pub fn new_unchecked(mu: usize, s: f64) -> Erlang { Erlang(Params::new_unchecked(mu, s)) }

    #[inline(always)]
    pub fn mu(&self) -> f64 { 1.0 / self.0.lambda.0 }
}

impl Default for Erlang {
    fn default() -> Erlang { Erlang(Params::new_unchecked(1, 1.0)) }
}

impl Distribution for Erlang {
    type Support = PositiveReals<f64>;
    type Params = Params;

    fn support(&self) -> PositiveReals<f64> { positive_reals() }

    fn params(&self) -> Params { self.0 }

    fn log_cdf(&self, x: &f64) -> f64 {
        let (k, lambda) = get_params!(self);

        (k as f64).gammainc(lambda * x).ln() - crate::utils::log_factorial_stirling(k as u64)
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for Erlang {
    fn log_pdf(&self, x: &f64) -> f64 {
        let (k, lambda) = get_params!(self);

        k as f64 * lambda.ln() + (k - 1) as f64 * x
            - lambda * x
            - crate::utils::log_factorial_stirling(k as u64 - 1)
    }
}

impl Univariate for Erlang {}

impl UvMoments for Erlang {
    fn mean(&self) -> f64 {
        let (k, lambda) = get_params!(self);

        k as f64 / lambda
    }

    fn variance(&self) -> f64 {
        let (k, lambda) = get_params!(self);

        k as f64 / lambda / lambda
    }

    fn skewness(&self) -> f64 { 2.0 / (self.0.k.0 as f64).sqrt() }

    fn excess_kurtosis(&self) -> f64 { 6.0 / self.0.k.0 as f64 }
}

impl Modes for Erlang {
    fn modes(&self) -> Vec<f64> {
        let (k, lambda) = get_params!(self);

        vec![(k - 1) as f64 / lambda]
    }
}

impl fmt::Display for Erlang {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (k, lambda) = get_params!(self);

        write!(f, "Erlang({}, {})", k, lambda)
    }
}
