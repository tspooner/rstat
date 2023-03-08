use crate::{
    consts::{NINETEEN_OVER_360, ONE_HALF, ONE_THIRD, ONE_TWELTH, ONE_TWENTY_FOURTH, PI_E_2},
    fitting::{Likelihood, Score, MLE},
    statistics::{FisherInformation, Modes, Quantiles, ShannonEntropy, UvMoments},
    utils::log_factorial_stirling,
    Convolution,
    DiscreteDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::discrete::{Naturals, naturals};
use std::fmt;

pub use crate::params::Rate;

params! {
    #[derive(Copy)]
    Params {
        lambda: Rate<f64>
    }
}

pub struct Grad {
    pub lambda: f64,
}

new_dist!(Poisson<Params>);

macro_rules! get_lambda {
    ($self:ident) => {
        $self.0.lambda.0
    };
}

impl Poisson {
    pub fn new(lambda: f64) -> Result<Poisson, failure::Error> {
        Ok(Poisson(Params::new(lambda)?))
    }

    pub fn new_unchecked(lambda: f64) -> Poisson { Poisson(Params::new_unchecked(lambda)) }
}

impl Distribution for Poisson {
    type Support = Naturals<u64>;
    type Params = Params;

    fn support(&self) -> Naturals<u64> { naturals() }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, _: &u64) -> Probability { unimplemented!() }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        use rand_distr::Distribution as _;

        rand_distr::Poisson::<f64>::new(get_lambda!(self))
            .unwrap()
            .sample(rng) as u64
    }
}

impl DiscreteDistribution for Poisson {
    fn log_pmf(&self, k: &u64) -> f64 {
        let l = get_lambda!(self);

        *k as f64 * l.ln() - l - crate::utils::log_factorial_stirling(*k)
    }
}

impl Univariate for Poisson {}

impl UvMoments for Poisson {
    fn mean(&self) -> f64 { get_lambda!(self) }

    fn variance(&self) -> f64 { get_lambda!(self) }

    fn skewness(&self) -> f64 { get_lambda!(self).powf(-ONE_HALF) }

    fn kurtosis(&self) -> f64 { 1.0 / get_lambda!(self) }
}

impl Quantiles for Poisson {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        let l = get_lambda!(self);

        (l + ONE_THIRD - 0.02 / l).floor()
    }
}

impl Modes for Poisson {
    fn modes(&self) -> Vec<u64> { vec![get_lambda!(self).floor() as u64] }
}

impl ShannonEntropy for Poisson {
    fn shannon_entropy(&self) -> f64 {
        let l = get_lambda!(self);

        (PI_E_2 * l).ln() / 2.0
            - ONE_TWELTH / l
            - ONE_TWENTY_FOURTH / l / l
            - NINETEEN_OVER_360 / l / l / l
    }
}

impl FisherInformation<1> for Poisson {
    fn fisher_information(&self) -> [[f64; 1]; 1] { [[get_lambda!(self)]] }
}

impl Likelihood for Poisson {
    fn log_likelihood(&self, samples: &[u64]) -> f64 {
        let n = samples.len() as f64;
        let l = get_lambda!(self);
        let l_ln = l.ln();

        -n * l
            + samples
                .into_iter()
                .map(|k| l_ln * *k as f64 - log_factorial_stirling(*k))
                .sum::<f64>()
    }
}

impl Score for Poisson {
    type Grad = Grad;

    fn score(&self, samples: &[u64]) -> Grad {
        let n = samples.len() as f64;
        let l = get_lambda!(self);

        Grad {
            lambda: n + samples.into_iter().sum::<u64>() as f64 / l,
        }
    }
}

impl MLE for Poisson {
    fn fit_mle(xs: &[u64]) -> Result<Self, failure::Error> {
        let n = xs.len() as f64;

        Poisson::new(xs.iter().fold(0.0, |acc, &x| acc + x as f64) as f64 / n)
    }
}

impl Convolution<Poisson> for Poisson {
    type Output = Poisson;

    fn convolve(self, rv: Poisson) -> Result<Poisson, failure::Error> {
        Ok(Poisson(Params::new_unchecked(
            get_lambda!(self) + get_lambda!(rv),
        )))
    }
}

impl fmt::Display for Poisson {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Poi({})", get_lambda!(self))
    }
}
