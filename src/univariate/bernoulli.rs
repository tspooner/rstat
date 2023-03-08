use crate::{
    fitting::{Likelihood, Score, MLE},
    statistics::{FisherInformation, Modes, Quantiles, ShannonEntropy, UvMoments},
    univariate::binomial::Binomial,
    Convolution,
    DiscreteDistribution,
    Distribution,
    Probability,
    Univariate,
};
use spaces::discrete::{Binary, binary};
use std::fmt;

params! {
    #[derive(Copy)]
    Params {
        p: Probability<>
    }
}

pub struct Grad {
    pub p: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Bernoulli {
    pub(crate) params: Params,

    q: Probability,
    variance: f64,
}

impl Bernoulli {
    pub fn new(p: f64) -> Result<Bernoulli, failure::Error> {
        Params::new(p).map(|ps| Bernoulli {
            q: !ps.p,
            params: ps,
            variance: p * (1.0 - p),
        })
    }

    pub fn new_unchecked(p: f64) -> Bernoulli { Params::new_unchecked(p).into() }
}

impl From<Params> for Bernoulli {
    fn from(params: Params) -> Bernoulli {
        Bernoulli {
            q: !params.p,
            variance: params.p.0 * (1.0 - params.p.0),

            params,
        }
    }
}

impl Distribution for Bernoulli {
    type Support = Binary;
    type Params = Params;

    fn support(&self) -> Binary { binary() }

    fn params(&self) -> Params { self.params }

    fn cdf(&self, k: &bool) -> Probability {
        if *k {
            Probability::one()
        } else {
            Probability::zero()
        }
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.gen_bool(self.params.p.unwrap())
    }
}

impl DiscreteDistribution for Bernoulli {
    fn pmf(&self, k: &bool) -> Probability {
        match k {
            true => self.params.p,
            false => self.q,
        }
    }
}

impl Univariate for Bernoulli {}

impl UvMoments for Bernoulli {
    fn mean(&self) -> f64 { self.params.p.into() }

    fn variance(&self) -> f64 { self.variance }

    fn skewness(&self) -> f64 { (1.0 - 2.0 * self.params.p.unwrap()) / self.variance.sqrt() }

    fn kurtosis(&self) -> f64 { 1.0 / self.variance - 6.0 }

    fn excess_kurtosis(&self) -> f64 { 1.0 / self.variance - 9.0 }
}

impl Quantiles for Bernoulli {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        match self.params.p.unwrap() {
            p if (p - 0.5).abs() < 1e-7 => 0.5,
            p if (p < 0.5) => 0.0,
            _ => 1.0,
        }
    }
}

impl Modes for Bernoulli {
    fn modes(&self) -> Vec<bool> {
        use std::cmp::Ordering::*;

        match self.params.p.partial_cmp(&self.q) {
            Some(Less) => vec![false],
            Some(Equal) => vec![false, true],
            Some(Greater) => vec![false],
            None => unreachable!(),
        }
    }
}

impl ShannonEntropy for Bernoulli {
    fn shannon_entropy(&self) -> f64 {
        let p = self.params.p.unwrap();
        let q = self.q.unwrap();

        if q.abs() < 1e-7 || (q - 1.0).abs() < 1e-7 {
            0.0
        } else {
            -q * q.ln() - p * p.ln()
        }
    }
}

impl FisherInformation<1> for Bernoulli {
    fn fisher_information(&self) -> [[f64; 1]; 1] { [[1.0 / self.variance]] }
}

impl Likelihood for Bernoulli {
    fn log_likelihood(&self, samples: &[bool]) -> f64 {
        samples.into_iter().map(|x| self.log_pmf(x)).sum()
    }
}

impl Score for Bernoulli {
    type Grad = Grad;

    fn score(&self, samples: &[bool]) -> Grad {
        Grad {
            p: samples.into_iter().map(|x| 1.0 / self.pmf(x)).sum(),
        }
    }
}

impl MLE for Bernoulli {
    fn fit_mle(xs: &[bool]) -> Result<Self, failure::Error> {
        let n = xs.len() as f64;
        let p = xs.iter().fold(0, |acc, &x| acc + x as u64) as f64 / n;

        Bernoulli::new(p)
    }
}

impl Convolution<Bernoulli> for Bernoulli {
    type Output = Binomial;

    fn convolve(self, rv: Bernoulli) -> Result<Binomial, failure::Error> {
        let p1 = self.params.p;
        let p2 = rv.params.p;

        assert_constraint!(p1 == p2)?;

        Ok(Binomial::new_unchecked(2, self.params.p))
    }
}

impl fmt::Display for Bernoulli {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Ber({})", self.params.p) }
}

#[cfg(test)]
mod tests {
    use super::{Bernoulli, Modes, Quantiles, UvMoments};
    use std::f64::INFINITY;

    macro_rules! test_bernoulli {
        ($fn:ident {
            $($p:literal => $v:expr),+
        }) => {
            $(
                assert_eq!(Bernoulli::new_unchecked($p).$fn(), $v);
            )+
        }
    }

    #[test]
    fn test_mean() {
        test_bernoulli!(mean {
            0.0 => 0.0,
            0.5 => 0.5,
            1.0 => 1.0
        });
    }

    #[test]
    fn test_variance() {
        test_bernoulli!(variance {
            0.0 => 0.0,
            0.5 => 0.25,
            1.0 => 0.0
        });
    }

    #[test]
    fn test_skewness() {
        test_bernoulli!(skewness {
            0.0 => INFINITY,
            0.25 => 0.5 / 0.1875f64.sqrt(),
            0.5 => 0.0,
            0.75 => -0.5 / 0.1875f64.sqrt(),
            1.0 => -INFINITY
        });
    }

    #[test]
    fn test_median() {
        test_bernoulli!(median {
            0.0 => 0.0,
            0.25 => 0.0,
            0.5 => 0.5,
            0.75 => 1.0,
            1.0 => 1.0
        });
    }
}
