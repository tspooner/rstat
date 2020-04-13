use crate::{params::{Shape, Rate}, prelude::*};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

params! {
    Params {
        k: Shape<usize>,
        lambda: Rate<f64>
    }
}

new_dist!(Erlang<Params>);

macro_rules! get_params {
    ($self:ident) => { ($self.0.k.0, $self.0.lambda.0) }
}

impl Erlang {
    pub fn new(mu: usize, s: f64) -> Result<Erlang, failure::Error> {
        Params::new(mu, s).map(|p| Erlang(p))
    }

    pub fn new_unchecked(mu: usize, s: f64) -> Erlang {
        Erlang(Params::new_unchecked(mu, s))
    }

    #[inline(always)]
    pub fn mu(&self) -> f64 { 1.0 / self.0.lambda.0 }
}

impl Default for Erlang {
    fn default() -> Erlang { Erlang(Params::new_unchecked(1, 1.0)) }
}

impl Distribution for Erlang {
    type Support = PositiveReals;
    type Params = Params;

    fn support(&self) -> PositiveReals { PositiveReals }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;

        let (k, lambda) = get_params!(self);
        let k = k as f64;

        Probability::new_unchecked(k.gammainc(lambda * x) / k.factorial())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Erlang {
    fn pdf(&self, x: &f64) -> f64 {
        use special_fun::FloatSpecial;

        let (k, lambda) = get_params!(self);

        lambda.powi(k as i32) * x.powi(k as i32 - 1) * (-lambda * x).exp() / (k as f64).factorial()
    }
}

impl UnivariateMoments for Erlang {
    fn mean(&self) -> f64 {
        let (k, lambda) = get_params!(self);

        k as f64 / lambda
    }

    fn variance(&self) -> f64 {
        let (k, lambda) = get_params!(self);

        k as f64 / lambda / lambda
    }

    fn skewness(&self) -> f64 {
        2.0 / (self.0.k.0 as f64).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        6.0 / self.0.k.0 as f64
    }
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
