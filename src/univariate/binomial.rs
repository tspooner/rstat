use crate::{
    Convolution, ConvolutionError, ConvolutionResult,
    consts::PI_E,
    prelude::*,
    univariate::Bernoulli,
};
use ndarray::Array2;
use rand;
use spaces::discrete::Ordinal;
use std::fmt;
use super::choose;

#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    pub n: usize,
    pub p: Probability,

    q: Probability,
}

impl Binomial {
    pub fn new(n: usize, p: Probability) -> Binomial {
        Binomial {
            n, p,
            q: !p,
        }
    }
}

impl Into<rand_distr::Binomial> for Binomial {
    fn into(self) -> rand_distr::Binomial {
        rand_distr::Binomial::new(self.n as u64, self.p.unwrap()).unwrap()
    }
}

impl Into<rand_distr::Binomial> for &Binomial {
    fn into(self) -> rand_distr::Binomial {
        rand_distr::Binomial::new(self.n as u64, self.p.unwrap()).unwrap()
    }
}

impl Distribution for Binomial {
    type Support = Ordinal;

    fn support(&self) -> Ordinal { Ordinal::new(self.n as usize) }

    fn cdf(&self, k: usize) -> Probability {
        use special_fun::FloatSpecial;

        let a = (self.n - k) as f64;
        let b = (k + 1) as f64;

        Probability::new_unchecked(self.q.unwrap().betainc(a, b))
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        use rand_distr::Distribution;

        let sampler: rand_distr::Binomial = self.into();

        sampler.sample(rng) as usize
    }
}

impl DiscreteDistribution for Binomial {
    fn pmf(&self, k: usize) -> Probability {
        let bc = choose(self.n as u64, k as u64) as f64;

        let prob_successes = self.p.powi(k as i32);
        let prob_failures = self.q.powi((self.n - k) as i32);
        let prob = prob_successes * prob_failures;

        Probability::new_unchecked(bc * prob)
    }
}

impl UnivariateMoments for Binomial {
    fn mean(&self) -> f64 {
        self.p * self.n as f64
    }

    fn variance(&self) -> f64 {
        let (p, q) = (self.p.unwrap(), self.q.unwrap());

        p * q * self.n as f64
    }

    fn skewness(&self) -> f64 {
        (1.0 - self.p * 2.0) / self.variance().sqrt()
    }

    fn kurtosis(&self) -> f64 {
        let (p, q) = (self.p.unwrap(), self.q.unwrap());

        (1.0 - 6.0 * p * q) / (self.n as f64 * p * q)
    }
}

impl Quantiles for Binomial {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.mean().round()
    }
}

impl Modes for Binomial {
    fn modes(&self) -> Vec<usize> {
        vec![(self.p * self.n as f64).floor() as usize]
    }
}

impl Entropy for Binomial {
    fn entropy(&self) -> f64 {
        (2.0 * PI_E * self.variance()).log2() / 2.0
    }
}

impl FisherInformation for Binomial {
    fn fisher_information(&self) -> Array2<f64> {
        Array2::from_elem((1, 1), self.n as f64 / self.p.unwrap() / self.q.unwrap())
    }
}

impl Convolution<Bernoulli> for Binomial {
    fn convolve(self, rv: Bernoulli) -> ConvolutionResult<Binomial> {
        if self.p == rv.p {
            Ok(Binomial::new(self.n + 1, self.p))
        } else {
            Err(ConvolutionError::MixedParameters)
        }
    }

    fn convolve_pair(a: Bernoulli, b: Bernoulli) -> ConvolutionResult<Binomial> {
        if a.p == b.p {
            Ok(Binomial::new(2, a.p))
        } else {
            Err(ConvolutionError::MixedParameters)
        }
    }
}

impl Convolution<Binomial> for Binomial {
    fn convolve(self, b: Binomial) -> ConvolutionResult<Binomial> {
        Self::convolve_pair(self, b)
    }

    fn convolve_pair(a: Binomial, b: Binomial) -> ConvolutionResult<Binomial> {
        if a.p == b.p {
            Ok(Binomial::new(a.n + b.n, a.p))
        } else {
            Err(ConvolutionError::MixedParameters)
        }
    }
}

impl fmt::Display for Binomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Bin({}, {})", self.n, self.p)
    }
}
