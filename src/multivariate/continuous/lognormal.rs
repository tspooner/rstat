use crate::{
    core::*,
    multivariate::continuous::MultivariateNormal,
};
use rand::Rng;
use spaces::{continuous::Reals, product::LinearSpace, Matrix, Vector};
use std::fmt;

#[derive(Debug, Clone)]
pub struct MultivariateLogNormal(MultivariateNormal);

impl MultivariateLogNormal {
    pub fn new(mu: Vector<f64>, sigma: Matrix<f64>) -> MultivariateLogNormal {
        MultivariateLogNormal(MultivariateNormal::new(mu, sigma))
    }

    pub fn isotropic(mu: Vector<f64>, sigma: f64) -> MultivariateLogNormal {
        MultivariateLogNormal(MultivariateNormal::isotropic(mu, sigma))
    }

    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> MultivariateLogNormal {
        MultivariateLogNormal(MultivariateNormal::homogeneous(n, mu, sigma))
    }

    pub fn standard(n: usize) -> MultivariateLogNormal {
        MultivariateLogNormal(MultivariateNormal::standard(n))
    }

    pub fn precision(&self) -> Matrix<f64> { self.0.precision() }
}

impl Distribution for MultivariateLogNormal {
    type Support = LinearSpace<Reals>;

    fn support(&self) -> LinearSpace<Reals> {
        self.0.support()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector<f64> {
        self.0.sample(rng).mapv(|v| v.exp())
    }
}

impl ContinuousDistribution for MultivariateLogNormal {
    fn pdf(&self, x: Vector<f64>) -> f64 {
        self.0.pdf(x.mapv(|v| v.ln()))
    }
}

impl MultivariateMoments for MultivariateLogNormal {
    fn mean(&self) -> Vector<f64> {
        let mu = self.0.mean();
        let var = self.0.variance();

        (mu + var / 2.0).mapv(|v| v.exp())
    }

    fn covariance(&self) -> Matrix<f64> {
        let mu = self.0.mean();
        let cov = self.0.covariance();
        let var = cov.diag();

        let n = mu.len();

        Matrix::from_shape_fn((n, n), |(i, j)| {
            (mu[i] + mu[j] + (var[i] + var[j]) / 2.0).exp() * (cov[(i, j)].exp() - 1.0)
        })
    }

    fn variance(&self) -> Vector<f64> {
        let mu = self.0.mean();
        let var = self.0.variance();

        Vector::from_shape_fn(mu.len(), |i| {
            (2.0 * mu[i] + var[i]).exp() * (var[i].exp() - 1.0)
        })
    }
}

impl fmt::Display for MultivariateLogNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Lognormal({}, {})", self.mean(), self.covariance())
    }
}
