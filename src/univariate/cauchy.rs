use crate::{
    consts::{ONE_OVER_PI, PI},
    statistics::{Modes, Quantiles, ShannonEntropy},
    ContinuousDistribution,
    Convolution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::real::{Reals, reals};
use std::fmt;

locscale_params! {
    #[derive(Copy)]
    Params {
        x0<f64>,
        gamma<f64>
    }
}

new_dist!(Cauchy<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.x0.0, $self.0.gamma.0)
    };
}

impl Cauchy {
    pub fn new(x0: f64, gamma: f64) -> Result<Cauchy, failure::Error> {
        Params::new(x0, gamma).map(Cauchy)
    }

    pub fn new_unchecked(x0: f64, gamma: f64) -> Cauchy {
        Cauchy(Params::new_unchecked(x0, gamma))
    }

    #[inline(always)]
    pub fn fwhm(&self) -> f64 { 2.0 * self.0.gamma.0 }

    #[inline(always)]
    fn z(&self, x: f64) -> f64 {
        let (x0, gamma) = get_params!(self);

        (x - x0) / gamma
    }
}

impl Default for Cauchy {
    fn default() -> Cauchy { Cauchy(Params::new_unchecked(0.0, 1.0)) }
}

impl Distribution for Cauchy {
    type Support = Reals<f64>;
    type Params = Params;

    fn support(&self) -> Reals<f64> { reals() }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        Probability::new_unchecked(ONE_OVER_PI * self.z(*x).atan() + 0.5)
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let (x0, gamma) = get_params!(self);

        rand_distr::Cauchy::new(x0, gamma).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for Cauchy {
    fn pdf(&self, x: &f64) -> f64 {
        let z = self.z(*x);

        1.0 / PI / self.0.gamma.0 / (1.0 + z * z)
    }
}

impl Univariate for Cauchy {}

impl Quantiles for Cauchy {
    fn quantile(&self, p: Probability) -> f64 {
        let (x0, gamma) = get_params!(self);

        x0 + gamma * (PI * (p - 0.5)).tan()
    }

    fn median(&self) -> f64 { self.0.x0.0 }
}

impl Modes for Cauchy {
    fn modes(&self) -> Vec<f64> { vec![self.0.x0.0] }
}

impl ShannonEntropy for Cauchy {
    fn shannon_entropy(&self) -> f64 { (4.0 * PI * self.0.gamma.0).ln() }
}

impl Convolution<Cauchy> for Cauchy {
    type Output = Cauchy;

    fn convolve(self, rv: Cauchy) -> Result<Cauchy, failure::Error> {
        let x0 = self.0.x0.0 + rv.0.x0.0;
        let gamma = self.0.gamma.0 + rv.0.gamma.0;

        Ok(Cauchy::new_unchecked(x0, gamma))
    }
}

impl fmt::Display for Cauchy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (x0, gamma) = get_params!(self);

        write!(f, "Cauchy({}, {})", x0, gamma)
    }
}
