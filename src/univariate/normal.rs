use crate::{
    Convolution, ConvolutionResult,
    consts::{PI_2, PI_E_2},
    fitting::MLE,
    prelude::*,
    validation::{Validator, Result},
};
use ndarray::Array2;
use rand::Rng;
use spaces::real::Reals;
use std::fmt;

pub type Gaussian = Normal;

#[derive(Debug, Clone, Copy)]
pub struct Normal {
    pub mu: f64,
    pub sigma: f64,
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Result<Normal> {
        Validator
            .require_positive_real(sigma)
            .map(|_| Normal::new_unchecked(mu, sigma))
    }

    pub fn new_unchecked(mu: f64, sigma: f64) -> Normal {
        Normal { mu, sigma }
    }

    pub fn standard() -> Normal {
        Normal {
            mu: 0.0,
            sigma: 1.0,
        }
    }

    #[inline(always)]
    pub fn z(&self, x: f64) -> f64 {
        (x - self.mu) / self.sigma
    }

    #[inline(always)]
    pub fn precision(&self) -> f64 {
        1.0 / self.sigma / self.sigma
    }

    #[inline(always)]
    pub fn width(&self) -> f64 {
        2.0 * self.precision()
    }
}

impl Default for Normal {
    fn default() -> Normal {
        Normal::standard()
    }
}

impl Into<rand_distr::Normal<f64>> for Normal {
    fn into(self) -> rand_distr::Normal<f64> {
        rand_distr::Normal::new(self.mu, self.sigma).unwrap()
    }
}

impl Into<rand_distr::Normal<f64>> for &Normal {
    fn into(self) -> rand_distr::Normal<f64> {
        rand_distr::Normal::new(self.mu, self.sigma).unwrap()
    }
}

impl Distribution for Normal {
    type Support = Reals;

    fn support(&self) -> Reals {
        Reals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        Probability::new_unchecked(0.5 + (self.z(x) / 2.0f64.sqrt()).erf() / 2.0)
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        let sampler: rand_distr::Normal<f64> = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: f64) -> f64 {
        let z = self.z(x);
        let norm = PI_2.sqrt() * self.sigma;

        (-z * z / 2.0).exp() / norm
    }
}

impl UnivariateMoments for Normal {
    fn mean(&self) -> f64 {
        self.mu
    }

    fn variance(&self) -> f64 {
        self.sigma * self.sigma
    }

    fn skewness(&self) -> f64 {
        0.0
    }

    fn kurtosis(&self) -> f64 {
        0.0
    }

    fn excess_kurtosis(&self) -> f64 {
        -3.0
    }
}

impl Quantiles for Normal {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.mu
    }
}

impl Modes for Normal {
    fn modes(&self) -> Vec<f64> {
        vec![self.mu]
    }
}

impl Entropy for Normal {
    fn entropy(&self) -> f64 {
        (PI_E_2 * self.variance()).ln() / 2.0
    }
}

impl FisherInformation for Normal {
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

impl Convolution<Normal> for Normal {
    fn convolve(self, rv: Normal) -> ConvolutionResult<Normal> {
        Self::convolve_pair(self, rv)
    }

    fn convolve_pair(a: Normal, b: Normal) -> ConvolutionResult<Normal> {
        let new_mu = a.mu + b.mu;
        let new_var = (a.variance() + b.variance()).sqrt();

        Ok(Normal::new_unchecked(new_mu, new_var))
    }
}

impl MLE for Normal {
    fn fit_mle(xs: Vec<f64>) -> Self {
        let n = xs.len() as f64;

        let mean = xs.iter().fold(0.0, |acc, &x| acc + x) / n;
        let var = xs.into_iter().map(|x| {
            x - mean
        }).fold(0.0, |acc, r| acc + r * r) / (n - 1.0);

        Normal::new_unchecked(mean, var.sqrt())
    }
}

impl fmt::Display for Normal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.mu, self.variance())
    }
}
