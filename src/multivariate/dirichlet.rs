use crate::{params::Concentrations, prelude::*};
use failure::Error;
use rand::Rng;
use spaces::{Interval, ProductSpace};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Dirichlet {
    pub alphas: Concentrations,

    alpha0: f64,
    ln_beta_alphas: f64,
    alphas_normed: Vector<f64>,
}

impl Dirichlet {
    pub fn new(alphas: Vector<f64>) -> Result<Dirichlet, Error> {
        let alphas = Concentrations::new(alphas)?;

        Ok(Dirichlet::new_unchecked(alphas.0))
    }

    pub fn new_unchecked(alphas: Vector<f64>) -> Dirichlet {
        use special_fun::FloatSpecial;

        let alpha0 = alphas.scalar_sum();

        Dirichlet {
            ln_beta_alphas: alphas
                .iter()
                .fold(-alpha0.loggamma(), |acc, a| acc + a.loggamma()),

            alpha0,
            alphas_normed: alphas.clone() / alpha0,
            alphas: Concentrations(alphas),
        }
    }
}

impl From<Concentrations> for Dirichlet {
    fn from(alphas: Concentrations) -> Dirichlet {
        Dirichlet::new_unchecked(alphas.0)
    }
}

impl Distribution for Dirichlet {
    type Support = ProductSpace<Interval>;
    type Params = Concentrations;

    fn support(&self) -> ProductSpace<Interval> {
        std::iter::repeat(Interval::unit())
            .take(self.alphas.0.len())
            .collect()
    }

    fn params(&self) -> Concentrations { self.alphas.clone() }

    fn cdf(&self, _: &Vec<f64>) -> Probability { todo!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        use rand_distr::Gamma as GammaSampler;

        let mut sum = 0.0f64;

        (self.alphas.0.iter().map(|&alpha| {
            let sample = rng.sample(GammaSampler::new(alpha, 1.0).unwrap());
            sum += sample;
            sample
        }).collect::<Vector<f64>>() / sum).into_raw_vec()
    }
}

impl ContinuousDistribution for Dirichlet {
    fn pdf(&self, xs: &Vec<f64>) -> f64 {
        self.log_pdf(xs).exp()
    }

    fn log_pdf(&self, xs: &Vec<f64>) -> f64 {
        let n = self.alphas.0.len();

        assert!(
            xs.len() == n,
            format!("Input `xs` must have length {}.", n)
        );

        xs.iter()
            .zip(self.alphas.0.iter())
            .fold(-self.ln_beta_alphas, |acc, (x, a)| acc + (a - 1.0) * x.ln())
    }
}

impl MultivariateMoments for Dirichlet {
    fn mean(&self) -> Vector<f64> { self.alphas_normed.clone() }

    fn variance(&self) -> Vector<f64> {
        let norm = self.alpha0 * self.alpha0 * (self.alpha0 + 1.0);

        self.alphas.0.mapv(|a| -a * (a - self.alpha0) / norm)
    }

    fn covariance(&self) -> Matrix<f64> {
        let d = self.alphas.0.len();
        let norm = self.alpha0 * self.alpha0 * (self.alpha0 + 1.0);

        Matrix::from_shape_fn((d, d), |(i, j)| {
            if i == j {
                let alpha = self.alphas.0[i];

                alpha * (self.alpha0 - alpha) / norm
            } else {
                -self.alphas.0[i] * self.alphas.0[j] / norm
            }
        })
    }
}

impl Modes for Dirichlet {
    fn modes(&self) -> Vec<Vec<f64>> {
        let k = self.alphas.0.len() as f64;

        vec![self.alphas.0.mapv(|a| (a - 1.0) / (self.alpha0 - k)).into_raw_vec()]
    }
}

impl fmt::Display for Dirichlet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Dir({})", self.alphas.0)
    }
}
