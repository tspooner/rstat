use crate::{
    prelude::*,
    multivariate::Normal,
};
use rand::Rng;
use spaces::{ProductSpace, real::Reals};
use std::fmt;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct LogNormal(Normal);

impl LogNormal {
    pub fn new_unchecked(mu: Array1<f64>, sigma: Array2<f64>) -> LogNormal {
        LogNormal(Normal::new_unchecked(mu, sigma))
    }

    pub fn isotropic(mu: Array1<f64>, sigma: f64) -> LogNormal {
        LogNormal(Normal::isotropic(mu, sigma))
    }

    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> LogNormal {
        LogNormal(Normal::homogeneous(n, mu, sigma))
    }

    pub fn standard(n: usize) -> LogNormal {
        LogNormal(Normal::standard(n))
    }

    pub fn precision(&self) -> Array2<f64> { self.0.precision() }
}

impl Distribution for LogNormal {
    type Support = ProductSpace<Reals>;

    fn support(&self) -> ProductSpace<Reals> {
        self.0.support()
    }

    fn cdf(&self, x: Vec<f64>) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.0.sample(rng).into_iter().map(|v| v.exp()).collect()
    }
}

impl ContinuousDistribution for LogNormal {
    fn pdf(&self, x: Vec<f64>) -> f64 {
        self.0.pdf(x.into_iter().map(|v| v.ln()).collect())
    }
}

impl MultivariateMoments for LogNormal {
    fn mean(&self) -> Array1<f64> {
        let mu = self.0.mean();
        let var = self.0.variance();

        (mu + var / 2.0).mapv(|v| v.exp())
    }

    fn covariance(&self) -> Array2<f64> {
        let mu = self.0.mean();
        let cov = self.0.covariance();
        let var = cov.diag();

        let n = mu.len();

        Array2::from_shape_fn((n, n), |(i, j)| {
            (mu[i] + mu[j] + (var[i] + var[j]) / 2.0).exp() * (cov[(i, j)].exp() - 1.0)
        })
    }

    fn variance(&self) -> Array1<f64> {
        let mu = self.0.mean();
        let var = self.0.variance();

        Array1::from_shape_fn(mu.len(), |i| {
            (2.0 * mu[i] + var[i]).exp() * (var[i].exp() - 1.0)
        })
    }
}

impl fmt::Display for LogNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LogNormal({}, {})", self.mean(), self.covariance())
    }
}
