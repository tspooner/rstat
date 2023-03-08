use crate::{
    statistics::{Modes, MvMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
    Multivariate,
};
use failure::Error;
use rand::Rng;
use spaces::intervals::Closed;
use std::fmt;

pub use crate::params::Concentrations;

#[derive(Debug, Clone)]
pub struct Dirichlet<const N: usize> {
    pub alphas: Concentrations<N>,

    alpha0: f64,
    ln_beta_alphas: f64,
    alphas_normed: [f64; N],
}

impl<const N: usize> Dirichlet<N> {
    pub fn new(alphas: [f64; N]) -> Result<Dirichlet<N>, Error> {
        let alphas = Concentrations::new(alphas)?;

        Ok(Dirichlet::new_unchecked(alphas.0))
    }

    pub fn new_unchecked(alphas: [f64; N]) -> Dirichlet<N> {
        use special_fun::FloatSpecial;

        let alpha0: f64 = alphas.iter().sum();

        Dirichlet {
            ln_beta_alphas: alphas
                .iter()
                .fold(-alpha0.loggamma(), |acc, a| acc + a.loggamma()),

            alpha0,
            alphas_normed: alphas.map(|a| a / alpha0),
            alphas: Concentrations(alphas),
        }
    }
}

impl<const N: usize> From<Concentrations<N>> for Dirichlet<N> {
    fn from(alphas: Concentrations<N>) -> Dirichlet<N> { Dirichlet::new_unchecked(alphas.0) }
}

impl<const N: usize> Distribution for Dirichlet<N> {
    type Support = [Closed<f64>; N];
    type Params = Concentrations<N>;

    fn support(&self) -> Self::Support { [Closed::unit(); N] }

    fn params(&self) -> Concentrations<N> { self.alphas.clone() }

    fn cdf(&self, _: &[f64; N]) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; N] {
        use rand_distr::Gamma as GammaSampler;

        let mut sum = 0.0f64;
        let mut sample = [0.0; N];

        for i in 0..N {
            let a = self.alphas.0[i];
            let s = rng.sample(GammaSampler::new(a, 1.0).unwrap());

            sum += a;
            sample[i] = s;
        }

        for i in 0..N {
            sample[i] /= sum;
        }

        sample
    }
}

impl<const N: usize> ContinuousDistribution for Dirichlet<N> {
    fn pdf(&self, xs: &[f64; N]) -> f64 { self.log_pdf(xs).exp() }

    fn log_pdf(&self, xs: &[f64; N]) -> f64 {
        assert!(xs.len() == N, "Input `xs` must have length {}.", N);

        xs.iter()
            .zip(self.alphas.0.iter())
            .fold(-self.ln_beta_alphas, |acc, (x, a)| acc + (a - 1.0) * x.ln())
    }
}

impl<const N: usize> Multivariate<N> for Dirichlet<N> {}

impl<const N: usize> MvMoments<N> for Dirichlet<N> {
    fn mean(&self) -> [f64; N] { self.alphas_normed.clone() }

    fn variance(&self) -> [f64; N] {
        let norm = self.alpha0 * self.alpha0 * (self.alpha0 + 1.0);

        self.alphas.0.map(|a| -a * (a - self.alpha0) / norm)
    }

    fn covariance(&self) -> [[f64; N]; N] {
        let d = self.alphas.0.len();
        let norm = self.alpha0 * self.alpha0 * (self.alpha0 + 1.0);

        let mut cov = [[0.0; N]; N];

        for i in 0..N {
            for j in 0..i {
                cov[i][j] = -self.alphas.0[i] * self.alphas.0[j] / norm;
            }

            for j in (i + 1)..N {
                cov[i][j] = -self.alphas.0[i] * self.alphas.0[j] / norm;
            }

            cov[i][i] = self.alphas.0[i] * (self.alpha0 - self.alphas.0[i]) / norm;
        }

        cov
    }
}

impl<const N: usize> Modes for Dirichlet<N> {
    fn modes(&self) -> Vec<[f64; N]> {
        let k = self.alphas.0.len() as f64;

        vec![self.alphas.0.map(|a| (a - 1.0) / (self.alpha0 - k))]
    }
}

impl<const N: usize> fmt::Display for Dirichlet<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Dir({:?})", self.alphas.0) }
}
