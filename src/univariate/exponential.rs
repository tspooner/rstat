use crate::{prelude::*, validation::{Result, ValidationError}};
use ndarray::Array2;
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    pub lambda: f64,
}

impl Exponential {
    pub fn new(lambda: f64) -> Result<Exponential> {
        ValidationError::assert_positive_real(lambda)
            .map(|lambda| Exponential::new_unchecked(lambda))
    }

    pub fn new_unchecked(lambda: f64) -> Exponential {
        Exponential { lambda }
    }

    pub fn mu(&self) -> f64 {
        1.0 / self.lambda
    }
}

impl Default for Exponential {
    fn default() -> Exponential {
        Exponential { lambda: 1.0 }
    }
}

impl Into<rand_distr::Exp<f64>> for Exponential {
    fn into(self) -> rand_distr::Exp<f64> {
        rand_distr::Exp::new(self.lambda).unwrap()
    }
}

impl Into<rand_distr::Exp<f64>> for &Exponential {
    fn into(self) -> rand_distr::Exp<f64> {
        rand_distr::Exp::new(self.lambda).unwrap()
    }
}

impl Distribution for Exponential {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        Probability::new_unchecked(1.0 - (-self.lambda * x).exp())
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        let sampler: rand_distr::Exp<f64> = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for Exponential {
    fn pdf(&self, x: f64) -> f64 {
        self.lambda * (-self.lambda * x).exp()
    }
}

impl UnivariateMoments for Exponential {
    fn mean(&self) -> f64 {
        1.0 / self.lambda
    }

    fn variance(&self) -> f64 {
        1.0 / self.lambda / self.lambda
    }

    fn skewness(&self) -> f64 {
        2.0
    }

    fn kurtosis(&self) -> f64 {
        3.0
    }

    fn excess_kurtosis(&self) -> f64 {
        6.0
    }
}

impl Quantiles for Exponential {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.mean() * 2.0f64.ln()
    }
}

impl Modes for Exponential {
    fn modes(&self) -> Vec<f64> {
        vec![0.0]
    }
}

impl Entropy for Exponential {
    fn entropy(&self) -> f64 {
        1.0 - self.lambda.ln()
    }
}

impl FisherInformation for Exponential {
    fn fisher_information(&self) -> Array2<f64> {
        Array2::from_elem((1, 1), self.lambda * self.lambda)
    }
}

impl fmt::Display for Exponential {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Exp({})", self.lambda)
    }
}
