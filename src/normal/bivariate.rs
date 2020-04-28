use super::{Params, Grad, Loc, Covariance};
use crate::{
    consts::PI_2,
    linalg::{Matrix, Vector},
    statistics::{MultivariateMoments, Modes, ShannonEntropy},
    ContinuousDistribution,
    Distribution,
    Probability,
};
use ndarray::array;
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::{real::Reals, TwoSpace};
use std::fmt;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Params/Grad
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type BvNormalParams = Params<[f64; 2], ([f64; 2], f64)>;

impl BvNormalParams {
    pub fn bivariate(mu: [f64; 2], sigma2_diag: [f64; 2], rho: f64) -> Result<Self, failure::Error> {
        Ok(Params {
            mu: Loc::new(mu)?,
            Sigma: Covariance::bivariate(sigma2_diag, rho)?,
        })
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Grad
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type BvNormalGrad = Grad<[f64; 2], ([f64; 2], f64)>;

impl std::ops::Mul<f64> for BvNormalGrad {
    type Output = Self;

    fn mul(self, sf: f64) -> Self {
        Grad {
            mu: [self.mu[0] * sf, self.mu[1] * sf],
            Sigma: ([self.Sigma.0[0] * sf, self.Sigma.0[1]], self.Sigma.1 * sf),
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Distribution
///////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone, Copy)]
pub struct BvNormal(BvNormalParams);

macro_rules! get_params {
    ($self:ident) => { ($self.0.mu.0, ($self.0.Sigma.0).0, ($self.0.Sigma.0).1) };
}

impl BvNormal {
    pub fn new(mu: [f64; 2], sigma2_diag: [f64; 2], rho: f64) -> Result<BvNormal, failure::Error> {
        Params::bivariate(mu, sigma2_diag, rho).map(BvNormal)
    }

    pub fn new_unchecked(mu: [f64; 2], sigma2_diag: [f64; 2], rho: f64) -> BvNormal {
        BvNormal(Params {
            mu: Loc(mu),
            Sigma: Covariance((sigma2_diag, rho)),
        })
    }

    pub fn independent(mu: [f64; 2], sigma2_diag: [f64; 2]) -> Result<BvNormal, failure::Error> {
        Params::bivariate(mu, sigma2_diag, 0.0).map(BvNormal)
    }

    pub fn independent_unchecked(mu: [f64; 2], sigma2_diag: [f64; 2]) -> BvNormal {
        BvNormal::new_unchecked(mu, sigma2_diag, 0.0)
    }

    pub fn isotropic(mu: [f64; 2], sigma2: f64) -> Result<BvNormal, failure::Error> {
        BvNormal::independent(mu, [sigma2; 2])
    }

    pub fn isotropic_unchecked(mu: [f64; 2], sigma2: f64) -> BvNormal {
        BvNormal::independent_unchecked(mu, [sigma2; 2])
    }

    pub fn standard() -> BvNormal { BvNormal::isotropic_unchecked([0.0; 2], 1.0) }

    #[inline]
    pub fn z(&self, x: &[f64; 2]) -> f64 {
        let (mu, [sigma2_0, sigma2_1], rho) = get_params!(self);

        let diff_0 = x[0] - mu[0];
        let diff_1 = x[1] - mu[1];

        let q_term = diff_0.powi(2) / sigma2_0
            - 2.0 * rho * diff_0 * diff_1 / sigma2_0.sqrt() / sigma2_1.sqrt()
            + diff_1.powi(2) / sigma2_1;
        let rho_term = 1.0 - rho * rho;

        q_term / rho_term
    }
}

impl Default for BvNormal {
    fn default() -> BvNormal { BvNormal::standard() }
}

impl From<BvNormalParams> for BvNormal {
    fn from(params: BvNormalParams) -> BvNormal { BvNormal(params) }
}

impl Distribution for BvNormal {
    type Support = TwoSpace<Reals>;
    type Params = BvNormalParams;

    fn support(&self) -> TwoSpace<Reals> { TwoSpace::new([Reals, Reals]) }

    fn params(&self) -> BvNormalParams { self.0 }

    fn cdf(&self, _: &[f64; 2]) -> Probability { unimplemented!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 2] {
        let z1: f64 = rng.sample(RandSN);
        let z2: f64 = rng.sample(RandSN);
        let (mu, [sigma2_0, sigma2_1], rho) = get_params!(self);

        [
            mu[0] + sigma2_0.sqrt() * z1,
            mu[1] + sigma2_1.sqrt() * (z1 * rho + z2 * (1.0 - rho * rho).sqrt()),
        ]
    }
}

impl ContinuousDistribution for BvNormal {
    fn pdf(&self, x: &[f64; 2]) -> f64 {
        let z = self.z(x);
        let (_, [sigma2_0, sigma2_1], rho) = get_params!(self);

        let norm = PI_2 * sigma2_0.sqrt() * sigma2_1.sqrt() * (1.0 - rho * rho).sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl MultivariateMoments for BvNormal {
    fn mean(&self) -> Vector<f64> { vec![self.0.mu.0[0], self.0.mu.0[1]].into() }

    fn covariance(&self) -> Matrix<f64> {
        let (_, [sigma2_0, sigma2_1], rho) = get_params!(self);
        let cross_sigma = sigma2_0.sqrt() * sigma2_1.sqrt();

        array![
            [sigma2_0, rho * cross_sigma],
            [rho * cross_sigma, sigma2_1],
        ]
    }

    fn variance(&self) -> Vector<f64> {
        let (_, [sigma2_0, sigma2_1], _) = get_params!(self);

        vec![sigma2_0, sigma2_1].into()
    }

    fn correlation(&self) -> Matrix<f64> {
        let (_, _, rho) = get_params!(self);

        array![[1.0, rho], [rho, 1.0]]
    }
}

impl Modes for BvNormal {
    fn modes(&self) -> Vec<[f64; 2]> { vec![self.0.mu.0] }
}

impl ShannonEntropy for BvNormal {
    fn shannon_entropy(&self) -> f64 {
        let (_, [sigma2_0, sigma2_1], rho) = get_params!(self);

        1.0 + PI_2.ln() + 0.5 * sigma2_0 * sigma2_1 * (1.0 - rho)
    }
}

impl fmt::Display for BvNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({:?}, {})", self.mean(), self.covariance())
    }
}
