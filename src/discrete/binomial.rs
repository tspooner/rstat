use consts::PI_E;
use core::*;
use discrete::Bernoulli;
use rand::Rng;
use spaces::{Matrix, discrete::Discrete};
use std::fmt;
use super::choose;

// TODO XXX: Replace usize with u64 after the new version of `spaces`
#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    pub n: usize,
    pub p: Probability,

    q: Probability,
}

impl Binomial {
    pub fn new<P: Into<Probability>>(n: usize, p: P) -> Binomial {
        let p: Probability = p.into();

        Binomial {
            n, p,
            q: !p,
        }
    }
}

impl Distribution for Binomial {
    type Support = Discrete;

    fn support(&self) -> Discrete { Discrete::new(self.n as usize) }

    fn cdf(&self, k: usize) -> Probability {
        use special_fun::FloatSpecial;

        let a = (self.n - k) as f64;
        let b = (k + 1) as f64;

        f64::from(self.q).betainc(a, b).into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> usize {
        unimplemented!()
    }
}

impl DiscreteDistribution for Binomial {
    fn pmf(&self, k: usize) -> Probability {
        let bc = choose(self.n as u64, k as u64) as f64;

        let prob_successes = self.p.powi(k as i32);
        let prob_failures = self.q.powi((self.n - k) as i32);
        let prob = f64::from(prob_successes * prob_failures);

        (bc * prob).into()
    }
}

impl UnivariateMoments for Binomial {
    fn mean(&self) -> f64 {
        self.n as f64 * f64::from(self.p)
    }

    fn variance(&self) -> f64 {
        self.n as f64 * f64::from(self.p * self.q)
    }

    fn skewness(&self) -> f64 {
        (1.0 - 2.0 * f64::from(self.p)) / self.variance().sqrt()
    }

    fn kurtosis(&self) -> f64 {
        let pnp = f64::from(self.p * self.q);

        (1.0 - 6.0 * pnp) / (self.n as f64 * pnp)
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
        vec![(self.n as f64 * f64::from(self.p)).floor() as usize]
    }
}

impl Entropy for Binomial {
    fn entropy(&self) -> f64 {
        (2.0 * PI_E * self.variance()).log2() / 2.0
    }
}

impl FisherInformation for Binomial {
    fn fisher_information(&self) -> Matrix {
        Matrix::from_elem((1, 1), self.n as f64 / f64::from(self.p * self.q))
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
        write!(f, "Bin({}, {})", self.n, f64::from(self.p))
    }
}
