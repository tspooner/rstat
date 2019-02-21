use crate::{
    consts::{PI_2, PI_E_2},
    core::*,
};
use rand::Rng;
use spaces::{continuous::Reals, Matrix, Vector};
use std::fmt;

pub type Gaussian = Normal;

#[derive(Debug, Clone, Copy)]
pub struct Normal {
    pub mu: f64,
    pub sigma: f64,
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Normal {
        assert_positive_real!(sigma);

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

impl Distribution for Normal {
    type Support = Reals;

    fn support(&self) -> Reals {
        Reals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        (0.5 + (self.z(x) / 2.0f64.sqrt()).erf() / 2.0).into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::distributions::{Distribution as DistSampler, Normal as NormalSampler};

        NormalSampler::new(self.mu, self.sigma).sample(rng)
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: f64) -> Probability {
        let z = self.z(x);
        let norm = 1.0 / PI_2.sqrt() / self.sigma;

        (norm * (-z * z / 2.0).exp()).into()
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
    fn fisher_information(&self) -> Matrix {
        let precision = self.precision();

        unsafe {
            Matrix::from_shape_vec_unchecked(
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

        Ok(Normal::new(new_mu, new_var))
    }
}

impl MLE for Normal {
    fn fit_mle(samples: Vector<f64>) -> Self {
        let n = samples.len() as f64;

        let sample_mean = samples.scalar_sum() / n;

        let residuals = samples - sample_mean;
        let sample_var = residuals.fold(0.0, |acc, v| acc + v * v) / n;

        Normal::new(sample_mean, sample_var.sqrt())
    }
}

impl fmt::Display for Normal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.mu, self.variance())
    }
}
