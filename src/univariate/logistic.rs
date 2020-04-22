use crate::{
    consts::{PI2, SIX_FIFTHS, TWENTY_ONE_FIFTHS},
    statistics::{Modes, Quantiles, ShannonEntropy, UnivariateMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
};
use rand::Rng;
use spaces::real::Reals;
use std::fmt;

locscale_params! {
    #[derive(Copy)]
    Params {
        mu<f64>,
        s<f64>
    }
}

new_dist!(Logistic<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.s.0)
    };
}

impl Logistic {
    pub fn new(mu: f64, s: f64) -> Result<Logistic, failure::Error> {
        Params::new(mu, s).map(Logistic)
    }

    pub fn new_unchecked(mu: f64, s: f64) -> Logistic { Logistic(Params::new_unchecked(mu, s)) }

    #[inline(always)]
    fn z(&self, x: f64) -> f64 {
        let (mu, s) = get_params!(self);

        (x - mu) / s
    }
}

impl Distribution for Logistic {
    type Support = Reals;
    type Params = Params;

    fn support(&self) -> Reals { Reals }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        Probability::new_unchecked(1.0 / (1.0 + (-self.z(*x)).exp()))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for Logistic {
    fn pdf(&self, x: &f64) -> f64 {
        let exp_term = (-self.z(*x)).exp();
        let exp_term_p1 = 1.0 + exp_term;

        exp_term / self.0.s.0 / exp_term_p1 / exp_term_p1
    }
}

impl UnivariateMoments for Logistic {
    fn mean(&self) -> f64 { self.0.mu.0 }

    fn variance(&self) -> f64 { self.0.s.0 * self.0.s.0 * PI2 / 3.0 }

    fn skewness(&self) -> f64 { 0.0 }

    fn kurtosis(&self) -> f64 { TWENTY_ONE_FIFTHS }

    fn excess_kurtosis(&self) -> f64 { SIX_FIFTHS }
}

impl Quantiles for Logistic {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 { self.0.mu.0 }
}

impl Modes for Logistic {
    fn modes(&self) -> Vec<f64> { vec![self.0.mu.0] }
}

impl ShannonEntropy for Logistic {
    fn shannon_entropy(&self) -> f64 { self.0.s.0.ln() + 2.0 }
}

impl fmt::Display for Logistic {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mu, s) = get_params!(self);

        write!(f, "Logistic({}, {})", mu, s)
    }
}
