use crate::{
    consts::PI_E,
    statistics::{FisherInformation, Modes, Quantiles, ShannonEntropy, UnivariateMoments},
    univariate::bernoulli::Bernoulli,
    utils::*,
    Convolution,
    DiscreteDistribution,
    Distribution,
    Probability,
};
use ndarray::Array2;
use rand;
use spaces::discrete::Ordinal;
use std::fmt;

pub use crate::params::Count;

params! {
    #[derive(Copy)]
    Params {
        n: Count<usize>,
        p: Probability<>
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Binomial {
    params: Params,
    q: Probability,
}

macro_rules! get_params {
    ($self:ident) => {
        ($self.params.n.0, $self.params.p)
    };
}

impl Binomial {
    pub fn new(n: usize, p: Probability) -> Result<Binomial, failure::Error> {
        Params::new(n, p.0).map(|params| Binomial {
            q: !(params.p),
            params,
        })
    }

    pub fn new_unchecked(n: usize, p: Probability) -> Binomial {
        Binomial {
            q: !p,
            params: Params::new_unchecked(n, p.0),
        }
    }
}

impl From<Params> for Binomial {
    fn from(params: Params) -> Binomial { Binomial::new_unchecked(params.n.0, params.p) }
}

impl Distribution for Binomial {
    type Support = Ordinal;
    type Params = Params;

    fn support(&self) -> Ordinal { Ordinal::new(self.params.n.0) }

    fn params(&self) -> Params { self.params }

    fn cdf(&self, k: &usize) -> Probability {
        use special_fun::FloatSpecial;

        let a = (self.params.n.0 - k) as f64;
        let b = (k + 1) as f64;

        Probability::new_unchecked(self.q.unwrap().betainc(a, b))
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> usize {
        use rand_distr::Distribution as _;

        let (n, p) = get_params!(self);
        let dist = rand_distr::Binomial::new(n as u64, p.unwrap()).unwrap();

        dist.sample(rng) as usize
    }
}

impl DiscreteDistribution for Binomial {
    fn pmf(&self, k: &usize) -> Probability {
        let (n, p) = get_params!(self);

        let bc = choose(n as u64, *k as u64) as f64;

        let prob_successes = p.powi(*k as i32);
        let prob_failures = self.q.powi((n - *k) as i32);
        let prob = prob_successes * prob_failures;

        Probability::new_unchecked(bc * prob)
    }
}

impl UnivariateMoments for Binomial {
    fn mean(&self) -> f64 {
        let (n, p) = get_params!(self);

        p.unwrap() * n as f64
    }

    fn variance(&self) -> f64 {
        let (n, p) = get_params!(self);
        let (p, q) = (p.unwrap(), self.q.unwrap());

        p * q * n as f64
    }

    fn skewness(&self) -> f64 { (1.0 - self.params.p * 2.0) / self.variance().sqrt() }

    fn kurtosis(&self) -> f64 {
        let (n, p) = get_params!(self);
        let (p, q) = (p.unwrap(), self.q.unwrap());

        (1.0 - 6.0 * p * q) / (n as f64 * p * q)
    }
}

impl Quantiles for Binomial {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 { self.mean().round() }
}

impl Modes for Binomial {
    fn modes(&self) -> Vec<usize> {
        let (n, p) = get_params!(self);

        vec![(p * n as f64).floor() as usize]
    }
}

impl ShannonEntropy for Binomial {
    fn shannon_entropy(&self) -> f64 { (2.0 * PI_E * self.variance()).log2() / 2.0 }
}

impl FisherInformation for Binomial {
    fn fisher_information(&self) -> Array2<f64> {
        let (n, p) = get_params!(self);

        Array2::from_elem((1, 1), n as f64 / p.unwrap() / self.q.unwrap())
    }
}

impl Convolution<Bernoulli> for Binomial {
    type Output = Binomial;

    fn convolve(self, rv: Bernoulli) -> Result<Binomial, failure::Error> {
        let p1 = self.params.p;
        let p2 = rv.params.p;

        Binomial::new(self.params.n.0 + 1, assert_constraint!(p1 == p2)?)
    }
}

impl Convolution<Binomial> for Binomial {
    type Output = Binomial;

    fn convolve(self, rv: Binomial) -> Result<Binomial, failure::Error> {
        let p1 = self.params.p;
        let p2 = rv.params.p;

        assert_constraint!(p1 == p2)?;

        Binomial::new(self.params.n.0 + rv.params.n.0, p1)
    }
}

impl fmt::Display for Binomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (n, p) = get_params!(self);

        write!(f, "Bin({}, {})", n, p)
    }
}
