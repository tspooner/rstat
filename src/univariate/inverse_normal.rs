use crate::{
    consts::{PI_2, PI_E_2},
    statistics::{Modes, ShannonEntropy, UvMoments},
    univariate::{normal, uniform::Uniform},
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::real::{PositiveReals, positive_reals};
use std::fmt;

pub use crate::params::Shape;

params! {
    #[derive(Copy)]
    Params {
        mu: Shape<f64>,
        lambda: Shape<f64>
    }
}

pub type InvGaussian = InvNormal;

#[derive(Debug, Clone, Copy)]
pub struct InvNormal {
    params: Params,
    std_norm: normal::Normal,
}

macro_rules! get_params {
    ($self:ident) => {
        ($self.params.mu.0, $self.params.lambda.0)
    };
}

impl InvNormal {
    pub fn new(mu: f64, lambda: f64) -> Result<InvNormal, failure::Error> {
        Params::new(mu, lambda).map(|p| InvNormal {
            params: p,
            std_norm: normal::Normal::standard(),
        })
    }

    pub fn new_unchecked(mu: f64, lambda: f64) -> InvNormal {
        InvNormal {
            params: Params::new_unchecked(mu, lambda),
            std_norm: normal::Normal::standard(),
        }
    }
}

impl From<Params> for InvNormal {
    fn from(params: Params) -> InvNormal {
        InvNormal {
            params,
            std_norm: normal::Normal::new_unchecked(0.0, 1.0),
        }
    }
}

impl Distribution for InvNormal {
    type Support = PositiveReals<f64>;
    type Params = Params;

    fn support(&self) -> PositiveReals<f64> { positive_reals() }

    fn params(&self) -> Params { self.params }

    fn cdf(&self, x: &f64) -> Probability {
        let (mu, lambda) = get_params!(self);

        let xom = x / mu;
        let lox_sqrt = (lambda / x).sqrt();
        let inner_term = lox_sqrt * (xom - 1.0);

        let term1 = self.std_norm.cdf(&inner_term).unwrap();
        let term2 = (2.0 * lambda / mu).exp() * self.std_norm.cdf(&(&inner_term)).unwrap();

        Probability::new_unchecked(term1 + term2)
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let (mu, lambda) = get_params!(self);

        let nu = self.std_norm.sample(rng);

        let y = nu * nu;
        let x = mu + mu * mu * y / 2.0 / lambda
            - mu / 2.0 / lambda * (4.0 * mu * lambda * y + mu * mu * y * y).sqrt();

        let z = Uniform::<f64>::new_unchecked(0.0, 1.0).sample(rng);

        if z < mu / (mu + x) {
            x
        } else {
            mu * mu / x
        }
    }
}

impl ContinuousDistribution for InvNormal {
    fn pdf(&self, x: &f64) -> f64 {
        let (mu, lambda) = get_params!(self);
        let z = (lambda / PI_2 / x / x / x).sqrt();

        let diff = x - mu;
        let exponent = (-lambda * diff * diff) / 2.0 / mu / mu / x;

        z * exponent.exp()
    }
}

impl Univariate for InvNormal {}

impl UvMoments for InvNormal {
    fn mean(&self) -> f64 { self.params.mu.0 }

    fn variance(&self) -> f64 {
        let (mu, lambda) = get_params!(self);

        mu * mu * mu / lambda
    }

    fn skewness(&self) -> f64 {
        let (mu, lambda) = get_params!(self);

        3.0 * (mu / lambda).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let (mu, lambda) = get_params!(self);

        16.0 * mu / lambda
    }
}

impl Modes for InvNormal {
    fn modes(&self) -> Vec<f64> {
        let (mu, lambda) = get_params!(self);

        let term1 = (1.0 + 9.0 * mu * mu / 4.0 / lambda / lambda).sqrt();
        let term2 = 3.0 * mu / 2.0 / lambda;

        vec![mu * (term1 - term2)]
    }
}

impl ShannonEntropy for InvNormal {
    fn shannon_entropy(&self) -> f64 { (PI_E_2 * self.variance()).ln() / 2.0 }
}

impl fmt::Display for InvNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mu, lambda) = get_params!(self);

        write!(f, "IG({}, {})", mu, lambda)
    }
}
