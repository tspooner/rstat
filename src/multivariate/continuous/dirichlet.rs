use crate::prelude::*;
use rand::Rng;
use spaces::{Interval, ProductSpace};
use std::fmt;
use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct Dirichlet {
    pub alphas: Array1<f64>,

    alpha0: f64,
    ln_beta_alphas: f64,
}

impl Dirichlet {
    pub fn new(alphas: Vec<f64>) -> Dirichlet {
        use special_fun::FloatSpecial;

        if alphas.len() < 2 {
            panic!("A Dirichlet distribution requires at least 2 concentration parameters.")
        }

        if !alphas.iter().all(|v| v > &0.0) {
            panic!("Concentration parameters must all be positive real.")
        }

        if (alphas.iter().fold(0.0, |acc, a| acc + a) - 1.0f64).abs() > 1e-7 {
            panic!("Concentration parameters must sum to 1.")
        }

        let alphas = Array1::from(alphas);
        let alpha0 = alphas.scalar_sum();

        Dirichlet {
            ln_beta_alphas: alphas
                .iter()
                .fold(-alpha0.loggamma(), |acc, a| acc + a.loggamma()),

            alphas,
            alpha0,
        }
    }
}

impl Into<rand_distr::Dirichlet<f64>> for Dirichlet {
    fn into(self) -> rand_distr::Dirichlet<f64> {
        rand_distr::Dirichlet::new(self.alphas.to_vec()).unwrap()
    }
}

impl Into<rand_distr::Dirichlet<f64>> for &Dirichlet {
    fn into(self) -> rand_distr::Dirichlet<f64> {
        rand_distr::Dirichlet::new(self.alphas.to_vec()).unwrap()
    }
}

impl Distribution for Dirichlet {
    type Support = ProductSpace<Interval>;

    fn support(&self) -> ProductSpace<Interval> {
        ProductSpace::new(vec![Interval::bounded(0.0, 1.0); self.alphas.len()])
    }

    fn cdf(&self, _: Vec<f64>) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vec<f64> {
        use rand_distr::Gamma as GammaSampler;

        let mut sum = 0.0f64;

        (self.alphas.iter().map(|&alpha| {
            let sample = rng.sample(GammaSampler::new(alpha, 1.0).unwrap());
            sum += sample;
            sample
        }).collect::<Array1<f64>>() / sum).into_raw_vec()
    }
}

impl ContinuousDistribution for Dirichlet {
    fn pdf(&self, xs: Vec<f64>) -> f64 {
        self.logpdf(xs).exp()
    }

    fn logpdf(&self, xs: Vec<f64>) -> f64 {
        assert_len!(xs => self.alphas.len(); K);

        xs.iter()
            .zip(self.alphas.iter())
            .fold(-self.ln_beta_alphas, |acc, (x, a)| acc + (a - 1.0) * x.ln())
    }
}

impl MultivariateMoments for Dirichlet {
    fn mean(&self) -> Array1<f64> {
        self.alphas.clone() / self.alpha0
    }

    fn variance(&self) -> Array1<f64> {
        let alphas = self.alphas.clone();

        let norm = self.alpha0 * self.alpha0 * (self.alpha0 + 1.0);

        alphas.map(|a| -a * (a - self.alpha0) / norm)
    }

    fn covariance(&self) -> Array2<f64> {
        let d = self.alphas.len();
        let norm = self.alpha0 * self.alpha0 * (self.alpha0 + 1.0);

        Array2::from_shape_fn((d, d), |(i, j)| {
            if i == j {
                let alpha = self.alphas[i];

                alpha * (self.alpha0 - alpha) / norm
            } else {
                -self.alphas[i] * self.alphas[j] / norm
            }
        })
    }
}

impl Modes for Dirichlet {
    fn modes(&self) -> Vec<Vec<f64>> {
        let k = self.alphas.len() as f64;

        vec![self.alphas.map(|a| (a - 1.0) / (self.alpha0 - k)).into_raw_vec()]
    }
}

impl fmt::Display for Dirichlet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Dir({})", self.alphas)
    }
}
