use crate::{
    consts::{PI_2, TWO_OVER_PI},
    statistics::{Modes, UnivariateMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
};
use rand::Rng;
use spaces::real::NonNegativeReals;
use std::fmt;

locscale_params! {
    #[derive(Copy)]
    Params {
        mu<f64>,
        sigma<f64>
    }
}

new_dist!(FoldedNormal<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.sigma.0)
    };
}

impl FoldedNormal {
    pub fn new(mu: f64, sigma: f64) -> Result<FoldedNormal, failure::Error> {
        Params::new(mu, sigma).map(FoldedNormal)
    }

    pub fn new_unchecked(mu: f64, sigma: f64) -> FoldedNormal {
        FoldedNormal(Params::new_unchecked(mu, sigma))
    }

    pub fn half_normal(scale: Scale<f64>) -> FoldedNormal {
        FoldedNormal(Params::new_unchecked(0.0, scale.0))
    }

    pub fn standard() -> FoldedNormal { FoldedNormal(Params::new_unchecked(0.0, 1.0)) }

    #[inline(always)]
    pub fn precision(&self) -> f64 {
        let s = self.0.sigma.0;

        1.0 / s / s
    }
}

impl Default for FoldedNormal {
    fn default() -> FoldedNormal { FoldedNormal::standard() }
}

impl Distribution for FoldedNormal {
    type Support = NonNegativeReals;
    type Params = Params;

    fn support(&self) -> NonNegativeReals { NonNegativeReals }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;

        let (mu, sigma) = get_params!(self);
        let sqrt_2: f64 = 2.0f64.sqrt();

        Probability::new_unchecked(
            0.5 * (((x + mu) / sigma / sqrt_2).erf() + ((x - mu) / sigma / sqrt_2).erf()),
        )
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        let (mu, sigma) = get_params!(self);

        rand_distr::Normal::new(mu, sigma)
            .unwrap()
            .sample(rng)
            .abs()
    }
}

impl ContinuousDistribution for FoldedNormal {
    fn pdf(&self, x: &f64) -> f64 {
        let (mu, sigma) = get_params!(self);

        let norm = PI_2.sqrt() * sigma;
        let z_pos = (x + mu) / sigma;
        let z_neg = (x - mu) / sigma;

        (-z_pos * z_pos / 2.0).exp() / norm + (-z_neg * z_neg / 2.0).exp() / norm
    }
}

impl UnivariateMoments for FoldedNormal {
    fn mean(&self) -> f64 {
        use special_fun::FloatSpecial;

        let (mu, sigma) = get_params!(self);
        let z = mu / sigma / 2.0f64.sqrt();

        sigma * TWO_OVER_PI.sqrt() * (-z * z).exp() + mu * z.erf()
    }

    fn variance(&self) -> f64 {
        let (mu, sigma) = get_params!(self);
        let mean = self.mean();

        mu * mu + sigma * sigma - mean * mean
    }

    fn skewness(&self) -> f64 { unimplemented!() }

    fn kurtosis(&self) -> f64 { unimplemented!() }

    fn excess_kurtosis(&self) -> f64 { unimplemented!() }
}

impl Modes for FoldedNormal {
    fn modes(&self) -> Vec<f64> {
        let (mu, sigma) = get_params!(self);

        if mu < sigma {
            vec![0.0]
        } else {
            vec![mu]
        }
    }
}

impl fmt::Display for FoldedNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FN({}, {})", self.0.mu.0, self.variance())
    }
}
