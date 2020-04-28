use super::{Params, Grad, Loc, Covariance};
use crate::{
    consts::{PI_2, PI_E_2},
    fitting::{Likelihood, Score, MLE},
    statistics::{FisherInformation, Modes, Quantiles, ShannonEntropy, UnivariateMoments},
    ContinuousDistribution,
    Convolution,
    Distribution,
    Probability,
};
use ndarray::Array2;
use rand::Rng;
use spaces::real::Reals;
use std::fmt;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Params
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type UvNormalParams = Params<f64, f64>;

impl UvNormalParams {
    pub fn new(mu: f64, sigma2: f64) -> Result<Self, failure::Error> {
        Ok(Params {
            mu: Loc::new(mu)?,
            Sigma: Covariance::isotropic(sigma2)?,
        })
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Grad
///////////////////////////////////////////////////////////////////////////////////////////////////
pub type UvNormalGrad = Grad<f64, f64>;

impl std::ops::Mul<f64> for UvNormalGrad {
    type Output = Self;

    fn mul(self, sf: f64) -> Self {
        Grad {
            mu: self.mu * sf,
            Sigma: self.Sigma * sf,
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Distribution
///////////////////////////////////////////////////////////////////////////////////////////////////
new_dist!(#[derive(Copy)] UvNormal<UvNormalParams>);

macro_rules! get_params {
    ($self:ident) => { ($self.0.mu.0, $self.0.Sigma.0) };
}

impl UvNormal {
    pub fn new(mu: f64, sigma2: f64) -> Result<UvNormal, failure::Error> {
        Params::new(mu, sigma2).map(UvNormal)
    }

    pub fn new_unchecked(mu: f64, sigma2: f64) -> UvNormal {
        UvNormal(Params {
            mu: Loc(mu),
            Sigma: Covariance(sigma2),
        })
    }

    pub fn standard() -> UvNormal { UvNormal::new_unchecked(0.0, 1.0) }

    #[inline(always)]
    pub fn z(&self, x: f64) -> f64 {
        let (mu, Sigma) = get_params!(self);

        (x - mu) / Sigma.sqrt()
    }

    #[inline(always)]
    pub fn precision(&self) -> f64 { 1.0 / self.0.Sigma.0 }

    #[inline(always)]
    pub fn width(&self) -> f64 { 2.0 * self.precision() }
}

impl Distribution for UvNormal {
    type Support = Reals;
    type Params = UvNormalParams;

    fn support(&self) -> Reals { Reals }

    fn params(&self) -> Self::Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;
        use std::num::FpCategory;

        Probability::new_unchecked(match x.classify() {
            FpCategory::Infinite => {
                if x.is_sign_negative() {
                    0.0
                } else {
                    1.0
                }
            },
            _ => self.z(*x).norm(),
        })
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let (mu, Sigma) = get_params!(self);

        rand_distr::Normal::new(mu, Sigma.sqrt()).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for UvNormal {
    fn pdf(&self, x: &f64) -> f64 {
        let z = self.z(*x);
        let norm = PI_2.sqrt() * self.0.Sigma.0.sqrt();

        (-z * z / 2.0).exp() / norm
    }
}

impl UnivariateMoments for UvNormal {
    fn mean(&self) -> f64 { self.0.mu.0 }

    fn variance(&self) -> f64 { self.0.Sigma.0 }

    fn skewness(&self) -> f64 { 0.0 }

    fn kurtosis(&self) -> f64 { 0.0 }

    fn excess_kurtosis(&self) -> f64 { -3.0 }
}

impl Quantiles for UvNormal {
    fn quantile(&self, p: Probability) -> f64 {
        use special_fun::FloatSpecial;

        let (mu, Sigma) = get_params!(self);

        mu + Sigma.sqrt() * p.unwrap().norm_inv()
    }

    fn median(&self) -> f64 { self.0.mu.0 }
}

impl Modes for UvNormal {
    fn modes(&self) -> Vec<f64> { vec![self.0.mu.0] }
}

impl ShannonEntropy for UvNormal {
    fn shannon_entropy(&self) -> f64 { (PI_E_2 * self.0.Sigma.0).ln() / 2.0 }
}

impl FisherInformation for UvNormal {
    fn fisher_information(&self) -> Array2<f64> {
        let precision = self.precision();

        unsafe {
            Array2::from_shape_vec_unchecked(
                (2, 2),
                vec![precision, 0.0, 0.0, precision * precision / 2.0],
            )
        }
    }
}

impl Likelihood for UvNormal {
    fn log_likelihood(&self, samples: &[f64]) -> f64 {
        let (mu, Sigma) = get_params!(self);

        let ls2 = Sigma.ln();
        let no2 = (samples.len() as f64) / 2.0;

        -no2 / 2.0 * (PI_2.ln() + ls2)
            - samples
                .into_iter()
                .map(|x| (*x - mu).powi(2) / 2.0 / Sigma)
                .sum::<f64>()
    }
}

impl Score for UvNormal {
    type Grad = UvNormalGrad;

    fn score(&self, samples: &[f64]) -> UvNormalGrad {
        let n = samples.len() as f64;
        let (mu, Sigma) = get_params!(self);
        let [sum, sum_sq] = samples.into_iter().fold([0.0; 2], |[s, ss], x| {
            let d = x - mu;

            [s + d, ss + d * d]
        });


        Grad {
            mu: sum / Sigma,
            Sigma: (sum_sq / Sigma - n) / Sigma.sqrt(),
        }
    }
}

impl MLE for UvNormal {
    fn fit_mle(xs: &[f64]) -> Result<Self, failure::Error> {
        let n = xs.len() as f64;

        let mean = xs.iter().fold(0.0, |acc, &x| acc + x) / n;
        let var = xs
            .into_iter()
            .map(|x| x - mean)
            .fold(0.0, |acc, r| acc + r * r)
            / (n - 1.0);

        UvNormal::new(mean, var.sqrt())
    }
}

impl Convolution<UvNormal> for UvNormal {
    type Output = UvNormal;

    fn convolve(self, rv: UvNormal) -> Result<UvNormal, failure::Error> {
        let new_mu = self.0.mu.0 + rv.0.mu.0;
        let new_Sigma = self.variance() + rv.variance();

        Ok(UvNormal::new_unchecked(new_mu, new_Sigma))
    }
}

impl fmt::Display for UvNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.0.mu.0, self.variance())
    }
}
