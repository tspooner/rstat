use super::choose;
use crate::{
    params::{Count, Shape},
    prelude::*,
};
use rand::Rng;
use spaces::discrete::Ordinal;
use special_fun::FloatSpecial;
use std::fmt;

params! {
    Params {
        n: Count<usize>,
        alpha: Shape<f64>,
        beta: Shape<f64>
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BetaBinomial {
    params: Params,

    pi: f64,
    rho: f64,
}

macro_rules! get_params {
    ($self:ident) => {
        ($self.params.n.0, $self.params.alpha.0, $self.params.beta.0)
    };
}

impl BetaBinomial {
    pub fn new(n: usize, alpha: f64, beta: f64) -> Result<BetaBinomial, failure::Error> {
        Params::new(n, alpha, beta).map(|p| p.into())
    }

    pub fn new_unchecked(n: usize, alpha: f64, beta: f64) -> BetaBinomial {
        BetaBinomial {
            pi: alpha / (alpha + beta),
            rho: 1.0 / (alpha + beta + 1.0),

            params: Params::new_unchecked(n, alpha, beta),
        }
    }

    fn pmf_raw(&self, k: usize) -> f64 {
        let (n, alpha, beta) = get_params!(self);

        let c = choose(n as u64, k as u64) as f64;
        let z = alpha.beta(beta);
        let k = k as f64;

        c * (k + alpha).beta(n as f64 - k + beta) / z
    }
}

impl From<Params> for BetaBinomial {
    fn from(params: Params) -> BetaBinomial {
        BetaBinomial::new_unchecked(params.n.0, params.alpha.0, params.beta.0)
    }
}

impl Distribution for BetaBinomial {
    type Support = Ordinal;
    type Params = Params;

    fn support(&self) -> Ordinal { Ordinal::new(self.params.n.0) }

    fn params(&self) -> Params { self.params }

    fn cdf(&self, k: &usize) -> Probability {
        if *k >= self.params.n.0 {
            Probability::one()
        } else {
            unimplemented!("Need an implmentation of the 3F2 generalised hypergeometric function.")
        }
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> usize { unimplemented!() }
}

impl DiscreteDistribution for BetaBinomial {
    fn pmf(&self, k: &usize) -> Probability { Probability::new_unchecked(self.pmf_raw(*k)) }
}

impl UnivariateMoments for BetaBinomial {
    fn mean(&self) -> f64 { self.params.n.0 as f64 * self.pi }

    fn variance(&self) -> f64 {
        let n = self.params.n.0 as f64;

        n * self.pi * (1.0 - self.pi) * (1.0 + (n - 1.0) * self.rho)
    }

    fn skewness(&self) -> f64 {
        let (n, alpha, beta) = get_params!(self);
        let n = n as f64;

        let ap1 = alpha + 1.0;
        let ap2 = ap1 + 1.0;

        // Compute bracket term:
        let b = n * n * ap1 * ap2 + 3.0 * n * ap1 * beta + beta * (beta - alpha);

        self.pi * self.rho * n * b / (ap2 + beta)
    }

    fn kurtosis(&self) -> f64 {
        let (n, alpha, beta) = get_params!(self);
        let n = n as f64;

        let apb = alpha + beta;

        let b1 = alpha
            / (n * self.pi * self.pi * self.rho * beta * (apb + 2.0) * (apb + 3.0) * (apb + n));
        let b2 = (apb - 1.0 + 6.0 * n) / alpha;
        let b3 = 3.0 * alpha * beta * (n - 2.0);
        let b4 = 6.0 * n * n;
        let b5 = -3.0 * self.pi * beta * n * (6.0 - n);
        let b6 = -18.0 * self.pi * beta * n * n / apb;

        b1 * (b2 + b3 + b4 + b5 + b6)
    }
}

impl fmt::Display for BetaBinomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (n, alpha, beta) = get_params!(self);

        write!(f, "BB({}, {}, {})", n, alpha, beta)
    }
}
