use crate::{consts::E, prelude::*};
use rand::Rng;
use spaces::real::Reals;
use std::fmt;

locscale_params! {
    Params {
        mu<f64>,
        b<f64>
    }
}

new_dist!(Laplace<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.b.0)
    };
}

impl Laplace {
    pub fn new(mu: f64, b: f64) -> Result<Laplace, failure::Error> {
        Params::new(mu, b).map(|p| Laplace(p))
    }

    pub fn new_unchecked(mu: f64, b: f64) -> Laplace { Laplace(Params::new_unchecked(mu, b)) }
}

impl Distribution for Laplace {
    type Support = Reals;
    type Params = Params;

    fn support(&self) -> Reals { Reals }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let (mu, b) = get_params!(self);

        Probability::new_unchecked((-((x - mu).abs() / b).abs()).exp() / 2.0 / b)
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let u: f64 = rand_distr::Uniform::new(-0.5, 0.5).sample(rng);
        let (mu, b) = get_params!(self);

        mu - b * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

impl ContinuousDistribution for Laplace {
    fn pdf(&self, x: &f64) -> f64 {
        use std::cmp::Ordering::*;

        let (mu, b) = get_params!(self);

        match x
            .partial_cmp(&mu)
            .expect("Invalid value provided for `mu`.")
        {
            Less | Equal => ((x - mu) / b).exp() / 2.0,
            Greater => 1.0 - ((mu - x) / b).exp() / 2.0,
        }
    }
}

impl UnivariateMoments for Laplace {
    fn mean(&self) -> f64 { self.0.mu.0 }

    fn variance(&self) -> f64 { 2.0 * self.0.b.0 * self.0.b.0 }

    fn skewness(&self) -> f64 { 0.0 }

    fn kurtosis(&self) -> f64 { 6.0 }

    fn excess_kurtosis(&self) -> f64 { 3.0 }
}

impl Quantiles for Laplace {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 { self.0.mu.0 }
}

impl Modes for Laplace {
    fn modes(&self) -> Vec<f64> { vec![self.0.mu.0] }
}

impl ShannonEntropy for Laplace {
    fn shannon_entropy(&self) -> f64 { (2.0 * self.0.b.0 * E).ln() }
}

impl fmt::Display for Laplace {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mu, b) = get_params!(self);

        write!(f, "Laplace({}, {})", mu, b)
    }
}
