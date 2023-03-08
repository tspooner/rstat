#![allow(non_snake_case)]
use crate::{
    multivariate::normal::Normal,
    normal::Params as NormalParams,
    statistics::MvMoments,
    ContinuousDistribution,
    Distribution,
    Probability,
    Multivariate,
};
use failure::Error;
use rand::Rng;
use spaces::real::{Reals, PositiveReals, positive_reals};
use std::fmt;

pub type Params<const N: usize, S> = NormalParams<[f64; N], S>;

#[derive(Debug, Clone)]
pub struct LogNormal<const N: usize, S>(Normal<N, [f64; N], S>);

impl<const N: usize> LogNormal<N, [[f64; N]; N]> {
    pub fn new(mu: [f64; N], Sigma: [[f64; N]; N]) -> Result<Self, Error> {
        Normal::new(mu, Sigma).map(LogNormal)
    }

    pub fn new_unchecked(mu: [f64; N], Sigma: [[f64; N]; N]) -> Self {
        LogNormal(Normal::new_unchecked(mu, Sigma))
    }
}

impl<const N: usize> LogNormal<N, [f64; N]> {
    pub fn diagonal(mu: [f64; N], sigma2_diag: [f64; N]) -> Result<Self, Error> {
        Normal::diagonal(mu, sigma2_diag).map(LogNormal)
    }

    pub fn diagonal_unchecked(mu: [f64; N], sigma2_diag: [f64; N]) -> Self {
        LogNormal(Normal::diagonal_unchecked(mu, sigma2_diag))
    }
}

impl<const N: usize> LogNormal<N, f64> {
    pub fn isotropic(mu: [f64; N], sigma2: f64) -> Result<Self, Error> {
        Normal::isotropic(mu, sigma2).map(LogNormal)
    }

    pub fn isotropic_unchecked(mu: [f64; N], sigma2: f64) -> Self {
        LogNormal(Normal::isotropic_unchecked(mu, sigma2))
    }

    pub fn homogeneous(mu: f64, sigma2: f64) -> Result<Self, Error> {
        Normal::homogeneous(mu, sigma2).map(LogNormal)
    }

    pub fn homogeneous_unchecked(mu: f64, sigma2: f64) -> Self {
        LogNormal(Normal::homogeneous_unchecked(mu, sigma2))
    }

    pub fn standard() -> Result<Self, Error> { Normal::standard().map(LogNormal) }

    pub fn standard_unchecked() -> Self { LogNormal(Normal::standard_unchecked()) }
}

impl<const N: usize, S> From<Normal<N, [f64; N], S>> for LogNormal<N, S> {
    fn from(normal: Normal<N, [f64; N], S>) -> LogNormal<N, S> { LogNormal(normal) }
}

impl<const N: usize, S> From<Params<N, S>> for LogNormal<N, S>
where
    Normal<N, [f64; N], S>: From<Params<N, S>>
{
    fn from(params: Params<N, S>) -> LogNormal<N, S> { LogNormal(params.into()) }
}

impl<const N: usize, S> Distribution for LogNormal<N, S>
where
    Normal<N, [f64; N], S>: From<Params<N, S>> + Distribution<Support = [Reals<f64>; N], Params = Params<N, S>>
{
    type Support = [PositiveReals<f64>; N];
    type Params = Params<N, S>;

    fn support(&self) -> [PositiveReals<f64>; N] { [positive_reals(); N] }

    fn params(&self) -> Params<N, S> { self.0.params() }

    fn cdf(&self, _: &[f64; N]) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; N] {
        self.0.sample(rng).map(|v| v.exp())
    }
}

impl<const N: usize, S> ContinuousDistribution for LogNormal<N, S>
where
    Normal<N, [f64; N], S>: From<Params<N, S>> + Distribution<Support = [Reals<f64>; N], Params = Params<N, S>>,
    Normal<N, [f64; N], S>: ContinuousDistribution,
{
    fn pdf(&self, x: &[f64; N]) -> f64 {
        let log_x = x.map(|v| v.ln());

        self.0.pdf(&log_x)
    }
}

impl<const N: usize, S> Multivariate<N> for LogNormal<N, S>
where
    Normal<N, [f64; N], S>: From<Params<N, S>> + Distribution<Support = [Reals<f64>; N], Params = Params<N, S>>
{}

impl<const N: usize, S> MvMoments<N> for LogNormal<N, S>
where
    Normal<N, [f64; N], S>: From<Params<N, S>> + Distribution<Support = [Reals<f64>; N], Params = Params<N, S>>,
    Normal<N, [f64; N], S>: MvMoments<N>,
{
    fn mean(&self) -> [f64; N] {
        let mut out = self.0.variance();

        for i in 0..N {
            out[i] = (self.0.params.mu.0[i] + out[i] / 2.0).exp();
        }

        out
    }

    fn covariance(&self) -> [[f64; N]; N] {
        let mu = self.0.mean();
        let mut cov = self.0.covariance();

        macro_rules! update_index {
            ($i:ident, $j:ident) => {
                let x = self.0.params.mu.0[$i] + self.0.params.mu.0[$j];
                let y = (cov[$i][$i] + cov[$j][$j]) / 2.0;
                let z = cov[$i][$j].exp() - 1.0;

                cov[$i][$j] = (x + y).exp() * z;
            }
        }

        // Upper/Lower triangular:
        for i in 0..N {
            for j in 0..i {
                update_index!(i, j);
            }

            for j in (i + 1)..N {
                update_index!(i, j);
            }

            update_index!(i, i);
        }

        cov
    }

    fn variance(&self) -> [f64; N] {
        let mut var = self.0.variance();

        for i in 0..N {
            var[i] = (2.0 * self.0.params.mu.0[i] + var[i]).exp() * (var[i].exp() - 1.0)
        }

        var
    }
}

impl<const N: usize, S: fmt::Display> fmt::Display for LogNormal<N, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "LogNormal({:?}, {})",
            self.0.params.mu.0, self.0.params.Sigma.0
        )
    }
}
