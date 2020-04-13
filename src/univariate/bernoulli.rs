use crate::{
    fitting::MLE,
    prelude::*,
    univariate::binomial::Binomial,
};
use ndarray::Array2;
use spaces::discrete::Binary;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Bernoulli {
    pub p: Probability,
    q: Probability,

    variance: f64,
}

impl Bernoulli {
    pub fn new(p: Probability) -> Bernoulli {
        let pf64 = p.unwrap();

        Bernoulli {
            p: p,
            q: !p,

            variance: pf64 * (1.0 - pf64),
        }
    }
}

impl From<Probability> for Bernoulli {
    fn from(p: Probability) -> Bernoulli {
        Bernoulli::new(p)
    }
}

impl Distribution for Bernoulli {
    type Support = Binary;
    type Params = Probability;

    fn support(&self) -> Binary { Binary }

    fn params(&self) -> Probability { self.p }

    fn cdf(&self, k: &bool) -> Probability {
        if *k { Probability::one() } else { Probability::zero() }
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> bool {
        rng.gen_bool(self.p.into())
    }
}

impl DiscreteDistribution for Bernoulli {
    fn pmf(&self, k: &bool) -> Probability {
        match k {
            true => self.p,
            false => self.q,
        }
    }
}

impl UnivariateMoments for Bernoulli {
    fn mean(&self) -> f64 { self.p.into() }

    fn variance(&self) -> f64 {
        self.variance
    }

    fn skewness(&self) -> f64 {
        (1.0 - 2.0 * self.p.unwrap()) / self.variance.sqrt()
    }

    fn kurtosis(&self) -> f64 {
        1.0 / self.variance - 6.0
    }

    fn excess_kurtosis(&self) -> f64 {
        1.0 / self.variance - 9.0
    }
}

impl Quantiles for Bernoulli {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        match self.p.unwrap() {
            p if (p - 0.5).abs() < 1e-7 => 0.5,
            p if (p < 0.5) => 0.0,
            _ => 1.0,
        }
    }
}

impl Modes for Bernoulli {
    fn modes(&self) -> Vec<bool> {
        use std::cmp::Ordering::*;

        match self.p.partial_cmp(&self.q) {
            Some(Less) => vec![false],
            Some(Equal) => vec![false, true],
            Some(Greater) => vec![false],
            None => unreachable!(),
        }
    }
}

impl Entropy for Bernoulli {
    fn entropy(&self) -> f64 {
        let p: f64 = self.p.into();
        let q: f64 = self.q.into();

        if q.abs() < 1e-7 || (q - 1.0).abs() < 1e-7 {
            0.0
        } else {
            -q * q.ln() - p*p.ln()
        }
    }
}

impl FisherInformation for Bernoulli {
    fn fisher_information(&self) -> Array2<f64> {
        Array2::from_elem((1, 1), 1.0 / self.variance)
    }
}

impl Convolution<Bernoulli> for Bernoulli {
    type Output = Binomial;

    fn convolve(self, rv: Bernoulli) -> Result<Binomial, failure::Error> {
        let p1 = self.p;
        let p2 = rv.p;

        assert_constraint!(p1 == p2)?;

        Ok(Binomial::new_unchecked(2, self.p))
    }
}

impl MLE for Bernoulli {
    fn fit_mle(xs: &[bool]) -> Result<Self, failure::Error> {
        let n = xs.len() as f64;
        let p = Probability::new(
            xs.iter().fold(0, |acc, &x| acc + x as u64) as f64 / n
        )?;

        Ok(Bernoulli::new(p))
    }
}

impl fmt::Display for Bernoulli {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Ber({})", self.p)
    }
}
