use crate::{
    linalg::{Matrix, Vector},
    multivariate::normal::Normal,
    statistics::MultivariateMoments,
    ContinuousDistribution,
    Distribution,
    Probability,
};
use failure::Error;
use rand::Rng;
use spaces::{real::Reals, ProductSpace};
use std::fmt;

pub use super::normal::Params;

#[derive(Debug, Clone)]
pub struct LogNormal<S>(Normal<S>);

impl LogNormal<Matrix<f64>> {
    pub fn new(mu: Vector<f64>, sigma: Matrix<f64>) -> Result<Self, Error> {
        Normal::new(mu, sigma).map(LogNormal)
    }

    pub fn new_unchecked(mu: Vector<f64>, sigma: Matrix<f64>) -> Self {
        LogNormal(Normal::new_unchecked(mu, sigma))
    }
}

impl LogNormal<Vector<f64>> {
    pub fn diagonal(mu: Vector<f64>, sigma: Vector<f64>) -> Result<Self, Error> {
        Normal::diagonal(mu, sigma).map(LogNormal)
    }

    pub fn diagonal_unchecked(mu: Vector<f64>, sigma: Vector<f64>) -> Self {
        LogNormal(Normal::diagonal_unchecked(mu, sigma))
    }
}

impl LogNormal<f64> {
    pub fn isotropic(mu: Vector<f64>, sigma: f64) -> Result<Self, Error> {
        Normal::isotropic(mu, sigma).map(LogNormal)
    }

    pub fn isotropic_unchecked(mu: Vector<f64>, sigma: f64) -> Self {
        LogNormal(Normal::isotropic_unchecked(mu, sigma))
    }

    pub fn homogeneous(n: usize, mu: f64, sigma: f64) -> Result<Self, Error> {
        Normal::homogeneous(n, mu, sigma).map(LogNormal)
    }

    pub fn homogeneous_unchecked(n: usize, mu: f64, sigma: f64) -> Self {
        LogNormal(Normal::homogeneous_unchecked(n, mu, sigma))
    }

    pub fn standard(n: usize) -> Result<Self, Error> { Normal::standard(n).map(LogNormal) }

    pub fn standard_unchecked(n: usize) -> Self { LogNormal(Normal::standard_unchecked(n)) }
}

impl<S> From<Normal<S>> for LogNormal<S> {
    fn from(normal: Normal<S>) -> LogNormal<S> { LogNormal(normal) }
}

impl<S> From<Params<S>> for LogNormal<S>
where Normal<S>: From<Params<S>>
{
    fn from(params: Params<S>) -> LogNormal<S> { LogNormal(params.into()) }
}

impl<S> Distribution for LogNormal<S>
where Normal<S>: From<Params<S>> + Distribution<Support = ProductSpace<Reals>, Params = Params<S>>
{
    type Support = ProductSpace<Reals>;
    type Params = Params<S>;

    fn support(&self) -> ProductSpace<Reals> { self.0.support() }

    fn params(&self) -> Params<S> { self.0.params() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        self.0.sample(rng).into_iter().map(|v| v.exp()).collect()
    }
}

impl<S> ContinuousDistribution for LogNormal<S>
where
    Normal<S>: From<Params<S>> + Distribution<Support = ProductSpace<Reals>, Params = Params<S>>,
    Normal<S>: ContinuousDistribution,
{
    fn pdf(&self, x: &Vec<f64>) -> f64 {
        let log_x = x.into_iter().map(|v| v.ln()).collect();

        self.0.pdf(&log_x)
    }
}

impl<S> MultivariateMoments for LogNormal<S>
where
    Normal<S>: From<Params<S>> + Distribution<Support = ProductSpace<Reals>, Params = Params<S>>,
    Normal<S>: MultivariateMoments,
{
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

impl<S: fmt::Display> fmt::Display for LogNormal<S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "LogNormal({}, {})",
            self.0.params.mu.0, self.0.params.sigma.0
        )
    }
}
