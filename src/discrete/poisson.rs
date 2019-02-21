use crate::{
    consts::{PI_E_2, ONE_HALF, ONE_THIRD, ONE_TWELTH, ONE_TWENTY_FOURTH, NINETEEN_OVER_360},
    core::*,
};
use rand::Rng;
use spaces::{Vector, Matrix, discrete::Naturals};
use std::fmt;
use super::factorial;

#[derive(Debug, Clone, Copy)]
pub struct Poisson {
    pub lambda: f64,
}

impl Poisson {
    pub fn new(lambda: f64) -> Poisson {
        if lambda <= 0.0 {
            panic!("The rate (lambda) of a Poisson distribution must be a positive real value.")
        }

        Poisson { lambda }
    }
}

impl Distribution for Poisson {
    type Support = Naturals;

    fn support(&self) -> Naturals { Naturals }

    fn cdf(&self, _: u64) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> u64 {
        unimplemented!()
    }
}

impl DiscreteDistribution for Poisson {
    fn pmf(&self, k: u64) -> Probability {
        (self.lambda.powi(k as i32) * (-self.lambda).exp() / factorial(k) as f64).into()
    }
}

impl UnivariateMoments for Poisson {
    fn mean(&self) -> f64 { self.lambda }

    fn variance(&self) -> f64 { self.lambda }

    fn skewness(&self) -> f64 { self.lambda.powf(-ONE_HALF) }

    fn kurtosis(&self) -> f64 { 1.0 / self.lambda }
}

impl Quantiles for Poisson {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        (self.lambda + ONE_THIRD - 0.02 / self.lambda).floor()
    }
}

impl Modes for Poisson {
    fn modes(&self) -> Vec<u64> {
        vec![self.lambda.floor() as u64]
    }
}

impl Entropy for Poisson {
    fn entropy(&self) -> f64 {
        (PI_E_2 * self.lambda).ln() / 2.0 -
            ONE_TWELTH / self.lambda -
            ONE_TWENTY_FOURTH / self.lambda / self.lambda -
            NINETEEN_OVER_360 / self.lambda / self.lambda / self.lambda
    }
}

impl FisherInformation for Poisson {
    fn fisher_information(&self) -> Matrix {
        Matrix::from_elem((1, 1), self.lambda)
    }
}

impl Convolution<Poisson> for Poisson {
    fn convolve(self, rv: Poisson) -> ConvolutionResult<Poisson> {
        Self::convolve_pair(self, rv)
    }

    fn convolve_pair(a: Poisson, b: Poisson) -> ConvolutionResult<Poisson> {
        Ok(Poisson::new(a.lambda + b.lambda))
    }
}

impl MLE for Poisson {
    fn fit_mle(samples: Vector<u64>) -> Self {
        let n = samples.len() as f64;

        Poisson::new(samples.scalar_sum() as f64 / n)
    }
}

impl fmt::Display for Poisson {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Poi({})", self.lambda)
    }
}
