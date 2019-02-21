use crate::core::*;
use rand::Rng;
use spaces::{continuous::Interval, product::LinearSpace, Matrix, Vector};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Dirichlet {
    pub alphas: Vector<f64>,

    alpha0: f64,
    ln_beta_alphas: f64,
}

impl Dirichlet {
    pub fn new(alphas: Vector<f64>) -> Dirichlet {
        use special_fun::FloatSpecial;

        if alphas.len() < 2 {
            panic!("A Dirichlet distribution requires at least 2 concentration parameters.")
        }

        if !alphas.fold(true, |acc, v| acc && (v > &0.0)) {
            panic!("Concentration parameters must all be positive real.")
        }

        if (alphas.scalar_sum() - 1.0) > 1e-7 {
            panic!("Concentration parameters must sum to 1.")
        }

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

impl Distribution for Dirichlet {
    type Support = LinearSpace<Interval>;

    fn support(&self) -> LinearSpace<Interval> {
        LinearSpace::new(vec![Interval::bounded(0.0, 1.0); self.alphas.len()])
    }

    fn cdf(&self, _: Vector<f64>) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Vector<f64> {
        use rand::distributions::{Gamma as GammaSampler, Distribution as DistSampler};

        let mut sum = 0.0f64;

        self.alphas.iter().map(|&alpha| {
            let sample = GammaSampler::new(alpha, 1.0).sample(rng);
            sum += sample;
            sample
        }).collect::<Vector<f64>>() / sum
    }
}

impl ContinuousDistribution for Dirichlet {
    fn pdf(&self, xs: Vector<f64>) -> Probability {
        self.logpdf(xs).exp().into()
    }

    fn logpdf(&self, xs: Vector<f64>) -> f64 {
        assert_len!(xs => self.alphas.len(); K);

        xs.iter()
            .zip(self.alphas.iter())
            .fold(-self.ln_beta_alphas, |acc, (x, a)| acc + (a - 1.0) * x.ln())
    }
}

impl MultivariateMoments for Dirichlet {
    fn mean(&self) -> Vector<f64> {
        self.alphas.clone() / self.alpha0
    }

    fn variance(&self) -> Vector<f64> {
        let alphas = self.alphas.clone();

        let norm = self.alpha0 * self.alpha0 * (self.alpha0 + 1.0);

        alphas.map(|a| -a * (a - self.alpha0) / norm)
    }

    fn covariance(&self) -> Matrix<f64> {
        let d = self.alphas.len();
        let norm = self.alpha0 * self.alpha0 * (self.alpha0 + 1.0);

        Matrix::from_shape_fn((d, d), |(i, j)| {
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
    fn modes(&self) -> Vec<Vector<f64>> {
        let k = self.alphas.len() as f64;

        vec![self.alphas.map(|a| (a - 1.0) / (self.alpha0 - k))]
    }
}

impl fmt::Display for Dirichlet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Dir({})", self.alphas)
    }
}
