use consts::PI_2;
use core::*;
use rand::Rng;
use spaces::{continuous::PositiveReals, Matrix};
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct LogNormal {
    pub mu: f64,
    pub sigma: f64,
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> LogNormal {
        assert_positive_real!(sigma);

        LogNormal { mu, sigma }
    }

    fn z(&self, x: f64) -> f64 {
        (x.ln() - self.mu) / self.sigma
    }
}

impl Default for LogNormal {
    fn default() -> LogNormal {
        LogNormal {
            mu: 0.0,
            sigma: 1.0,
        }
    }
}

impl Distribution for LogNormal {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        (0.5 + (self.z(x) / 2.0f64.sqrt()).erf() / 2.0).into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for LogNormal {
    fn pdf(&self, x: f64) -> Probability {
        let z = self.z(x);
        let norm = 1.0 / PI_2.sqrt() / self.sigma;

        (norm * (-z * z / 2.0).exp() / x).into()
    }
}

impl UnivariateMoments for LogNormal {
    fn mean(&self) -> f64 {
        (self.mu + self.sigma * self.sigma / 2.0).exp()
    }

    fn variance(&self) -> f64 {
        let sigma2 = self.sigma * self.sigma;

        (sigma2.exp() - 1.0) * (2.0 * self.mu + sigma2).exp()
    }

    fn skewness(&self) -> f64 {
        let sigma2 = self.sigma * self.sigma;

        (sigma2.exp() + 2.0) * (sigma2.exp() - 1.0).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let sigma2 = self.sigma * self.sigma;

        (4.0 * sigma2).exp() + 2.0 * (3.0 * sigma2).exp() + 3.0 * (2.0 * sigma2).exp() - 6.0
    }
}

impl Quantiles for LogNormal {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.mu.exp()
    }
}

impl Modes for LogNormal {
    fn modes(&self) -> Vec<f64> {
        vec![(self.mu - self.sigma * self.sigma).exp()]
    }
}

impl Entropy for LogNormal {
    fn entropy(&self) -> f64 {
        (self.sigma * (self.mu + 0.5).exp() * PI_2.sqrt()).log2()
    }
}

impl FisherInformation for LogNormal {
    fn fisher_information(&self) -> Matrix {
        let one_over_sigma2 = 1.0 / self.sigma / self.sigma;

        unsafe {
            Matrix::from_shape_vec_unchecked(
                (2, 2),
                vec![
                    one_over_sigma2,
                    0.0,
                    0.0,
                    one_over_sigma2 * one_over_sigma2 / 2.0,
                ],
            )
        }
    }
}

impl fmt::Display for LogNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lognormal({}, {})", self.mu, self.variance())
    }
}
