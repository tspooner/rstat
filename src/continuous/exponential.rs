use crate::core::*;
use rand::Rng;
use spaces::{continuous::PositiveReals, Matrix};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    pub lambda: f64,
}

impl Exponential {
    pub fn new(lambda: f64) -> Exponential {
        assert_positive_real!(lambda);

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

impl Distribution for Exponential {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        (1.0 - (-self.lambda * x).exp()).into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand::distributions::{Exp as ExpSampler, Distribution as DistSampler};

        ExpSampler::new(self.lambda).sample(rng)
    }
}

impl ContinuousDistribution for Exponential {
    fn pdf(&self, x: f64) -> Probability {
        (self.lambda * (-self.lambda * x).exp()).into()
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
    fn fisher_information(&self) -> Matrix {
        Matrix::from_elem((1, 1), self.lambda * self.lambda)
    }
}

impl fmt::Display for Exponential {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Exp({})", self.lambda)
    }
}
