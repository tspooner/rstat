use crate::{
    consts::ONE_THIRD,
    params::{Loc, Shape},
    prelude::*,
};
use rand::Rng;
use spaces::real::Interval;
use std::fmt;

params! {
    Params {
        mu: Loc<f64>,
        sigma: Shape<f64>,
        zeta: Shape<f64>
    }
}

new_dist!(GeneralisedPareto<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.sigma.0, $self.0.zeta.0)
    };
}

impl GeneralisedPareto {
    pub fn new(mu: f64, sigma: f64, zeta: f64) -> Result<GeneralisedPareto, failure::Error> {
        Params::new(mu, sigma, zeta).map(|p| GeneralisedPareto(p))
    }

    pub fn new_unchecked(mu: f64, sigma: f64, zeta: f64) -> GeneralisedPareto {
        GeneralisedPareto(Params::new_unchecked(mu, sigma, zeta))
    }
}

impl Distribution for GeneralisedPareto {
    type Support = Interval;
    type Params = Params;

    fn support(&self) -> Interval {
        use std::cmp::Ordering::*;

        let (mu, sigma, zeta) = get_params!(self);

        match zeta
            .partial_cmp(&0.0)
            .expect("Invalid value provided for `zeta`.")
        {
            Less => Interval::bounded(mu, mu - sigma / zeta),
            Equal | Greater => Interval::left_bounded(mu),
        }
    }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let (mu, sigma, zeta) = get_params!(self);
        let z = (x - mu) / sigma;

        Probability::new_unchecked(1.0 - (1.0 + zeta * z).powf(-1.0 / zeta))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for GeneralisedPareto {
    fn pdf(&self, x: &f64) -> f64 {
        let (mu, sigma, zeta) = get_params!(self);
        let z = (x - mu) / sigma;

        (1.0 + zeta * z).powf(-1.0 / zeta - 1.0) / sigma
    }
}

impl UnivariateMoments for GeneralisedPareto {
    fn mean(&self) -> f64 {
        let (mu, sigma, zeta) = get_params!(self);

        if zeta <= 1.0 {
            undefined!("Mean is undefined for zeta <= 1.")
        } else {
            mu + sigma / (1.0 - zeta)
        }
    }

    fn variance(&self) -> f64 {
        let zeta = self.0.zeta.0;
        let sigma = self.0.sigma.0;

        if zeta <= 0.5 {
            undefined!("Variance is undefined for zeta <= 1/2.")
        } else {
            sigma * sigma / (1.0 - zeta).powi(2) / (1.0 - 2.0 * zeta)
        }
    }

    fn skewness(&self) -> f64 {
        let zeta = self.0.zeta.0;

        if zeta <= ONE_THIRD {
            undefined!("Skewness is undefined for zeta <= 1/3.")
        } else {
            2.0 * (1.0 + zeta) * (1.0 - 2.0 * zeta).sqrt() / (1.0 - 3.0 * zeta)
        }
    }

    fn excess_kurtosis(&self) -> f64 {
        let zeta = self.0.zeta.0;

        if zeta <= ONE_THIRD {
            undefined!("Skewness is undefined for zeta <= 1/3.")
        } else {
            3.0 * (1.0 - 2.0 * zeta) * (2.0 * zeta * zeta + zeta + 3.0)
                / (1.0 - 3.0 * zeta)
                / (1.0 - 4.0 * zeta)
                - 3.0
        }
    }
}

impl Quantiles for GeneralisedPareto {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        let (mu, sigma, zeta) = get_params!(self);

        mu + sigma * (2.0f64.powf(zeta) - 1.0) / zeta
    }
}

impl ShannonEntropy for GeneralisedPareto {
    fn shannon_entropy(&self) -> f64 { self.0.sigma.0.ln() + self.0.zeta.0 + 1.0 }
}

impl fmt::Display for GeneralisedPareto {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mu, sigma, zeta) = get_params!(self);

        write!(f, "GPD({}, {}, {})", mu, sigma, zeta)
    }
}
