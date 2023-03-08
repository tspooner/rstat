use crate::{
    fitting::{Likelihood, Score},
    statistics::{Modes, ShannonEntropy, UvMoments},
    univariate::exponential::Exponential,
    ContinuousDistribution,
    Convolution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::real::{PositiveReals, positive_reals};
use special_fun::FloatSpecial;
use std::fmt;

shape_params! {
    #[derive(Copy)]
    Params<f64> {
        alpha,
        beta
    }
}

pub struct Grad {
    pub alpha: f64,
    pub beta: f64,
}

new_dist!(Gamma<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.alpha.0, $self.0.beta.0)
    };
}

impl Gamma {
    pub fn new(alpha: f64, beta: f64) -> Result<Gamma, failure::Error> {
        Params::new(alpha, beta).map(Gamma)
    }

    pub fn new_unchecked(alpha: f64, beta: f64) -> Gamma {
        Gamma(Params::new_unchecked(alpha, beta))
    }
}

impl Default for Gamma {
    fn default() -> Gamma { Gamma(Params::new_unchecked(1.0, 1.0)) }
}

impl Into<rand_distr::Gamma<f64>> for &Gamma {
    fn into(self) -> rand_distr::Gamma<f64> {
        let (a, b) = get_params!(self);

        rand_distr::Gamma::new(a, b).unwrap()
    }
}

impl Distribution for Gamma {
    type Support = PositiveReals<f64>;
    type Params = Params;

    fn support(&self) -> PositiveReals<f64> { positive_reals() }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let (a, b) = get_params!(self);

        Probability::new_unchecked(a.gammainc(b * x) / a.gamma())
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        let sampler: rand_distr::Gamma<f64> = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for Gamma {
    fn pdf(&self, x: &f64) -> f64 {
        let (a, b) = get_params!(self);

        b.powf(a) * x.powf(a - 1.0) * (-b * x).exp() / a.gamma()
    }
}

impl Univariate for Gamma {}

impl UvMoments for Gamma {
    fn mean(&self) -> f64 {
        let (a, b) = get_params!(self);

        a / b
    }

    fn variance(&self) -> f64 {
        let (a, b) = get_params!(self);

        a / b / b
    }

    fn skewness(&self) -> f64 { 2.0 / self.0.alpha.0.sqrt() }

    fn excess_kurtosis(&self) -> f64 { 6.0 / self.0.alpha.0 }
}

impl Modes for Gamma {
    fn modes(&self) -> Vec<f64> {
        match self.0.alpha.0 {
            a if a < 1.0 => undefined!("Mode is undefined for alpha < 1."),
            a => vec![(a - 1.0) / self.0.beta.0],
        }
    }
}

impl ShannonEntropy for Gamma {
    fn shannon_entropy(&self) -> f64 {
        let (a, b) = get_params!(self);

        a - b.ln() + a.gamma().ln() + (1.0 - a) * a.digamma()
    }
}

impl Likelihood for Gamma {
    fn log_likelihood(&self, samples: &[f64]) -> f64 {
        const JITTER: f64 = 1e-9;

        let n = samples.len() as f64;

        let (a, b) = get_params!(self);
        let am1 = a - 1.0;

        let t1 = samples
            .into_iter()
            .map(|x| am1 * (x + JITTER).ln() - b * (1.0 - x))
            .sum::<f64>();

        t1 - n * (a * (1.0 / b).max(JITTER) + a.gamma().ln())
    }
}

impl Score for Gamma {
    type Grad = Grad;

    fn score(&self, samples: &[f64]) -> Grad {
        const JITTER: f64 = 1e-9;

        let n = samples.len() as f64;

        let (a, b) = get_params!(self);
        let a_digamma = a.digamma();

        let [grad_a, grad_b] = samples.into_iter().fold([0.0; 2], |acc, x| {
            [acc[0] + (x / b).max(JITTER).ln(), acc[1] - x]
        });

        Grad {
            alpha: grad_a - n * a_digamma,
            beta: grad_b + n * a / b,
        }
    }
}

impl Convolution<Gamma> for Gamma {
    type Output = Gamma;

    fn convolve(self, rv: Gamma) -> Result<Gamma, failure::Error> {
        let (a1, beta) = get_params!(self);
        let a2 = rv.0.alpha.0;

        assert_constraint!(a1 == a2)?;

        Ok(Gamma::new_unchecked(a1 + a2, beta))
    }
}

impl Convolution<Exponential> for Gamma {
    type Output = Gamma;

    fn convolve(self, rv: Exponential) -> Result<Gamma, failure::Error> {
        let (alpha, beta) = get_params!(self);
        let lambda = rv.0.lambda.0;

        assert_constraint!(lambda == beta)?;

        Ok(Gamma::new_unchecked(alpha + 1.0, beta))
    }
}

impl fmt::Display for Gamma {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (a, b) = get_params!(self);

        write!(f, "Gamma({}, {})", a, b)
    }
}
