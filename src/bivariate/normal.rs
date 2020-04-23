use crate::{
    consts::PI_2,
    linalg::{Matrix, Vector},
    statistics::MultivariateMoments,
    ContinuousDistribution,
    Distribution,
    Probability,
};
use ndarray::array;
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{real::Reals, TwoSpace};
use std::fmt;

pub use crate::params::{Corr, Loc, Scale};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Params
///////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Params {
    pub mu: Loc<[f64; 2]>,
    pub sigma: Scale<[f64; 2]>,
    pub rho: Corr,
}

impl Params {
    pub fn new(mu: [f64; 2], sigma: [f64; 2], rho: f64) -> Result<Params, failure::Error> {
        let mu = Loc::new(mu)?;
        let sigma = Scale::new(sigma)?;
        let rho = Corr::new(rho)?;

        Ok(Params { mu, sigma, rho })
    }

    pub fn independent(mu: [f64; 2], sigma: [f64; 2]) -> Result<Params, failure::Error> {
        Params::new(mu, sigma, 0.0)
    }

    pub fn isotropic(mu: [f64; 2], sigma: f64) -> Result<Params, failure::Error> {
        Params::independent(mu, [sigma, sigma])
    }

    pub fn standard() -> Params {
        Params {
            mu: Loc([0.0; 2]),
            sigma: Scale([1.0; 2]),
            rho: Corr(0.0),
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Distribution
///////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone)]
pub struct Normal(Params);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.sigma.0, $self.0.rho.0)
    };
}

impl Normal {
    pub fn new(mu: [f64; 2], sigma: [f64; 2], rho: f64) -> Result<Normal, failure::Error> {
        Params::new(mu, sigma, rho).map(Normal)
    }

    pub fn new_unchecked(mu: [f64; 2], sigma: [f64; 2], rho: f64) -> Normal {
        Normal(Params {
            mu: Loc(mu),
            sigma: Scale(sigma),
            rho: Corr(rho),
        })
    }

    pub fn independent(mu: [f64; 2], sigma: [f64; 2]) -> Result<Normal, failure::Error> {
        Params::independent(mu, sigma).map(Normal)
    }

    pub fn independent_unchecked(mu: [f64; 2], sigma: [f64; 2]) -> Normal {
        Normal(Params {
            mu: Loc(mu),
            sigma: Scale(sigma),
            rho: Corr(0.0),
        })
    }

    pub fn isotropic(mu: [f64; 2], sigma: f64) -> Result<Normal, failure::Error> {
        Params::isotropic(mu, sigma).map(Normal)
    }

    pub fn isotropic_unchecked(mu: [f64; 2], sigma: f64) -> Normal {
        Normal(Params {
            mu: Loc(mu),
            sigma: Scale([sigma, sigma]),
            rho: Corr(0.0),
        })
    }

    pub fn standard() -> Normal { Normal(Params::standard()) }

    #[inline]
    pub fn z(&self, x: &[f64; 2]) -> f64 {
        let (mu, sigma, rho) = get_params!(self);

        let diff_0 = x[0] - mu[0];
        let diff_1 = x[1] - mu[1];

        let q_term = diff_0.powi(2) / sigma[0] / sigma[0]
            - 2.0 * rho * diff_0 * diff_1 / sigma[0] / sigma[1]
            + diff_1.powi(2) / sigma[1] / sigma[1];
        let rho_term = 1.0 - rho * rho;

        q_term / rho_term
    }
}

impl Default for Normal {
    fn default() -> Normal { Normal::standard() }
}

impl From<Params> for Normal {
    fn from(params: Params) -> Normal { Normal(params) }
}

impl Distribution for Normal {
    type Support = TwoSpace<Reals>;
    type Params = Params;

    fn support(&self) -> TwoSpace<Reals> { TwoSpace::new([Reals, Reals]) }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, _: &[f64; 2]) -> Probability { unimplemented!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 2] {
        let z1: f64 = rng.sample(RandSN);
        let z2: f64 = rng.sample(RandSN);
        let (mu, sigma, rho) = get_params!(self);

        [
            mu[0] + sigma[0] * z1,
            mu[1] + sigma[1] * (z1 * rho + z2 * (1.0 - rho * rho).sqrt()),
        ]
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: &[f64; 2]) -> f64 {
        let z = self.z(x);
        let sigma = self.0.sigma.0;
        let rho = self.0.rho.0;

        let norm = PI_2 * sigma[0] * sigma[1] * (1.0 - rho * rho).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for Normal {
    fn mean(&self) -> Vector<f64> { vec![self.0.mu.0[0], self.0.mu.0[1]].into() }

    fn covariance(&self) -> Matrix<f64> {
        let [s0, s1] = self.0.sigma.0;
        let cross_sigma = s0 * s1;

        array![
            [s0 * s0, self.0.rho.0 * cross_sigma],
            [self.0.rho.0 * cross_sigma, s1 * s1],
        ]
    }

    fn variance(&self) -> Vector<f64> {
        let [s0, s1] = self.0.sigma.0;

        vec![s0 * s0, s1 * s1].into()
    }

    fn correlation(&self) -> Matrix<f64> { array![[1.0, self.0.rho.0], [self.0.rho.0, 1.0],] }
}

impl fmt::Display for Normal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.mean(), self.covariance())
    }
}
