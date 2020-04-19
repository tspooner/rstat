use crate::{consts::PI_2, prelude::*, univariate::normal::Normal};
use ndarray::Array2;
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

pub use crate::univariate::normal::Params;

#[derive(Debug, Clone, Copy)]
pub struct LogNormal(Normal);

macro_rules! get_params {
    ($self:ident) => {
        (($self.0).0.mu.0, ($self.0).0.sigma.0)
    };
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> Result<LogNormal, failure::Error> {
        Params::new(mu, sigma).map(|p| LogNormal(Normal(p)))
    }

    pub fn new_unchecked(mu: f64, sigma: f64) -> LogNormal {
        LogNormal(Normal(Params::new_unchecked(mu, sigma)))
    }
}

impl From<Params> for LogNormal {
    fn from(params: Params) -> LogNormal { LogNormal(params.into()) }
}

impl From<Normal> for LogNormal {
    fn from(normal: Normal) -> LogNormal { LogNormal(normal) }
}

impl Distribution for LogNormal {
    type Support = PositiveReals;
    type Params = Params;

    fn support(&self) -> PositiveReals { PositiveReals }

    fn params(&self) -> Params { self.0.params() }

    fn into_params(self) -> Params { self.0.into_params() }

    fn cdf(&self, x: &f64) -> Probability {
        let ln_x = x.ln();

        self.0.cdf(&ln_x)
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        rand_distr::LogNormal::new(self.mean(), self.standard_deviation())
            .unwrap()
            .sample(rng)
    }
}

impl ContinuousDistribution for LogNormal {
    fn pdf(&self, x: &f64) -> f64 {
        let ln_x = x.ln();

        self.0.pdf(&ln_x)
    }
}

impl UnivariateMoments for LogNormal {
    fn mean(&self) -> f64 {
        let (mu, sigma) = get_params!(self);

        (mu + sigma * sigma / 2.0).exp()
    }

    fn variance(&self) -> f64 {
        let (mu, sigma) = get_params!(self);
        let sigma2 = sigma * sigma;

        (sigma2.exp() - 1.0) * (2.0 * mu + sigma2).exp()
    }

    fn skewness(&self) -> f64 {
        let sigma = (self.0).0.sigma.0;
        let sigma2 = sigma * sigma;

        (sigma2.exp() + 2.0) * (sigma2.exp() - 1.0).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let sigma = (self.0).0.sigma.0;
        let sigma2 = sigma * sigma;

        (4.0 * sigma2).exp() + 2.0 * (3.0 * sigma2).exp() + 3.0 * (2.0 * sigma2).exp() - 6.0
    }
}

impl Quantiles for LogNormal {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 { (self.0).0.mu.0.exp() }
}

impl Modes for LogNormal {
    fn modes(&self) -> Vec<f64> {
        let (mu, sigma) = get_params!(self);

        vec![(mu - sigma * sigma).exp()]
    }
}

impl ShannonEntropy for LogNormal {
    fn shannon_entropy(&self) -> f64 {
        let (mu, sigma) = get_params!(self);

        (sigma * (mu + 0.5).exp() * PI_2.sqrt()).log2()
    }
}

impl FisherInformation for LogNormal {
    fn fisher_information(&self) -> Array2<f64> {
        let sigma = (self.0).0.sigma.0;
        let one_over_sigma2 = 1.0 / sigma / sigma;

        unsafe {
            Array2::from_shape_vec_unchecked(
                (2, 2),
                vec![
                    one_over_sigma2,
                    0.0,
                    0.0,
                    one_over_sigma2 * one_over_sigma2 / 2.0,
                ],
            )
        }
    }
}

impl fmt::Display for LogNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LogNormal({}, {})", (self.0).0.mu.0, self.variance())
    }
}
