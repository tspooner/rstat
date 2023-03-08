use crate::{
    consts::{PI, PI_16, THREE_HALVES},
    statistics::{Modes, Quantiles, ShannonEntropy, UvMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::intervals::LeftClosed;
use std::{f64::INFINITY, fmt};

locscale_params! {
    #[derive(Copy)]
    Params {
        mu<f64>,
        c<f64>
    }
}

new_dist!(Levy<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.c.0)
    };
}

impl Levy {
    pub fn new(mu: f64, c: f64) -> Result<Levy, failure::Error> { Params::new(mu, c).map(Levy) }

    pub fn new_unchecked(mu: f64, c: f64) -> Levy { Levy(Params::new_unchecked(mu, c)) }
}

impl Distribution for Levy {
    type Support = LeftClosed<f64>;
    type Params = Params;

    fn support(&self) -> LeftClosed<f64> { LeftClosed::left_closed(self.0.mu.0) }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;

        let (mu, c) = get_params!(self);

        Probability::new_unchecked((c / 2.0 / (x - mu)).erfc())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for Levy {
    fn pdf(&self, x: &f64) -> f64 {
        let (mu, c) = get_params!(self);

        let diff = x - mu;
        let c_over_2 = c / 2.0;

        (c_over_2 / PI).sqrt() * (-c_over_2 / diff).exp() / diff.powf(THREE_HALVES)
    }
}

impl Univariate for Levy {}

impl UvMoments for Levy {
    fn mean(&self) -> f64 { INFINITY }

    fn variance(&self) -> f64 { INFINITY }

    fn skewness(&self) -> f64 { undefined!() }

    fn kurtosis(&self) -> f64 { undefined!() }
}

impl Quantiles for Levy {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        if self.0.mu.0.abs() < 1e-7 {
            // TODO
            unimplemented!("Need an implementation of the inverse ERFC function.")
        } else {
            undefined!("median of the Levy distribution is defined only for mu = 0.")
        }
    }
}

impl Modes for Levy {
    fn modes(&self) -> Vec<f64> {
        if self.0.mu.0.abs() < 1e-7 {
            vec![self.0.c.0 / 3.0]
        } else {
            undefined!("mode of the Levy distribution is defined only for mu = 0.")
        }
    }
}

impl ShannonEntropy for Levy {
    fn shannon_entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let c = self.0.c.0;
        let gamma = -(1.0f64.digamma());

        (1.0 + 3.0 * gamma + (PI_16 * c * c).ln()) / 2.0
    }
}

impl fmt::Display for Levy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mu, c) = get_params!(self);

        write!(f, "Levy({}, {})", mu, c)
    }
}
