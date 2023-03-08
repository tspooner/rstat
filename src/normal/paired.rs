use super::{Params, Grad, Loc, Covariance};
use crate::{
    consts::PI_2,
    fitting::{Likelihood, Score},
    statistics::{MvMoments, Modes, ShannonEntropy},
    ContinuousDistribution,
    Distribution,
    Probability,
    Multivariate,
};
use rand::Rng;
use rand_distr::StandardNormal as RandSN;
use spaces::real::{Reals, reals};
use std::fmt;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Params
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type PairedNormalParams = Params<[f64; 2], [f64; 2]>;

impl PairedNormalParams {
    pub fn paired(mu: [f64; 2], sigma2_diag: [f64; 2]) -> Result<Self, failure::Error> {
        Ok(Params {
            mu: Loc::new(mu)?,
            Sigma: Covariance::paired(sigma2_diag)?,
        })
    }
}

pub type PairedNormalGrad = Grad<[f64; 2], [f64; 2]>;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Distribution
///////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug, Clone, Copy)]
pub struct PairedNormal(PairedNormalParams);

macro_rules! get_params {
    ($self:ident) => { ($self.0.mu.0, $self.0.Sigma.0) };
}

impl PairedNormal {
    pub fn new(mu: [f64; 2], sigma2_diag: [f64; 2]) -> Result<PairedNormal, failure::Error> {
        Params::paired(mu, sigma2_diag).map(PairedNormal)
    }

    pub fn new_unchecked(mu: [f64; 2], sigma2_diag: [f64; 2]) -> PairedNormal {
        PairedNormal(Params {
            mu: Loc(mu),
            Sigma: Covariance(sigma2_diag),
        })
    }

    pub fn isotropic(mu: [f64; 2], sigma2: f64) -> Result<PairedNormal, failure::Error> {
        PairedNormal::new(mu, [sigma2; 2])
    }

    pub fn isotropic_unchecked(mu: [f64; 2], sigma2: f64) -> PairedNormal {
        PairedNormal::new_unchecked(mu, [sigma2; 2])
    }

    pub fn standard() -> PairedNormal { PairedNormal::isotropic_unchecked([0.0; 2], 1.0) }

    #[inline]
    pub fn z(&self, x: &[f64; 2]) -> f64 {
        let (mu, Sigma) = get_params!(self);

        let diff_0 = x[0] - mu[0];
        let diff_1 = x[1] - mu[1];

        diff_0.powi(2) / Sigma[0] + diff_1.powi(2) / Sigma[1]
    }
}

impl Default for PairedNormal {
    fn default() -> PairedNormal { PairedNormal::standard() }
}

impl From<PairedNormalParams> for PairedNormal {
    fn from(params: PairedNormalParams) -> PairedNormal { PairedNormal(params) }
}

impl Distribution for PairedNormal {
    type Support = [Reals<f64>; 2];
    type Params = PairedNormalParams;

    fn support(&self) -> Self::Support { [reals(); 2] }

    fn params(&self) -> PairedNormalParams { self.0 }

    fn cdf(&self, _: &[f64; 2]) -> Probability { unimplemented!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; 2] {
        let z1: f64 = rng.sample(RandSN);
        let z2: f64 = rng.sample(RandSN);
        let (mu, Sigma) = get_params!(self);

        [
            mu[0] + Sigma[0].sqrt() * z1,
            mu[1] + Sigma[1].sqrt() * z2
        ]
    }
}

impl ContinuousDistribution for PairedNormal {
    fn pdf(&self, x: &[f64; 2]) -> f64 {
        let z = self.z(x);
        let (_, Sigma) = get_params!(self);

        let norm = PI_2 * Sigma[0].sqrt() * Sigma[1].sqrt();

        (-z / 2.0).exp() / norm
    }
}

impl Multivariate<2> for PairedNormal {}

impl MvMoments<2> for PairedNormal {
    fn mean(&self) -> [f64; 2] { self.0.mu.0 }

    fn covariance(&self) -> [[f64; 2]; 2] {
        let (_, Sigma) = get_params!(self);

        [
            [Sigma[0], 0.0],
            [0.0, Sigma[1]],
        ]
    }

    fn variance(&self) -> [f64; 2] { get_params!(self).1 }

    fn correlation(&self) -> [[f64; 2]; 2] { [[1.0, 0.0], [0.0, 1.0]] }
}

impl Modes for PairedNormal {
    fn modes(&self) -> Vec<[f64; 2]> { vec![self.0.mu.0] }
}

impl ShannonEntropy for PairedNormal {
    fn shannon_entropy(&self) -> f64 {
        let (_, Sigma) = get_params!(self);

        1.0 + PI_2.ln() + 0.5 * Sigma[0] * Sigma[1]
    }
}

impl Likelihood for PairedNormal {
    fn log_likelihood(&self, samples: &[[f64; 2]]) -> f64 {
        let (mu, [sigma2_0, sigma2_1]) = get_params!(self);

        let ln_det = (sigma2_0 * sigma2_1).ln();
        let no2 = (samples.len() as f64) / 2.0;

        -no2 * (ln_det + 2.0 * PI_2.ln() + samples.into_iter().map(|x| {
            (x[0] - mu[0]).powi(2) / sigma2_0 + (x[1] - mu[1]).powi(2) / sigma2_1
        }).sum::<f64>())
    }
}

impl Score for PairedNormal {
    type Grad = PairedNormalGrad;

    fn score(&self, samples: &[[f64; 2]]) -> PairedNormalGrad {
        let (mu, [sigma2_0, sigma2_1]) = get_params!(self);

        let n = samples.len() as f64;
        let [sum, sum_sq] = samples.into_iter().fold([[0.0; 2]; 2], |[s, ss], x| {
            let d = [x[0] - mu[0], x[1] - mu[1]];

            [
                [s[0] + d[0], s[1] + d[1]],
                [ss[0] + d[0] * d[0], ss[1] + d[1] * d[1]]
            ]
        });

        Grad {
            mu: [sum[0] / sigma2_0, sum[1] / sigma2_1],
            Sigma: [
                (sum_sq[0] / sigma2_0 - n) / sigma2_0.sqrt(),
                (sum_sq[1] / sigma2_1 - n) / sigma2_1.sqrt()
            ],
        }
    }
}

impl fmt::Display for PairedNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({:?}, {:?})", self.mean(), self.covariance())
    }
}
