use crate::{
    consts::PI_2,
    statistics::{FisherInformation, Modes, Quantiles, ShannonEntropy, UvMoments},
    univariate::normal::Normal,
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::real::{PositiveReals, positive_reals};
use std::fmt;

pub use crate::univariate::normal::Params;

#[derive(Debug, Clone, Copy)]
pub struct LogNormal(Normal);

macro_rules! get_params {
    ($self:ident) => {
        (($self.0).0.mu.0, ($self.0).0.Sigma.0)
    };
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> Result<LogNormal, failure::Error> {
        Normal::new(mu, sigma).map(LogNormal)
    }

    pub fn new_unchecked(mu: f64, sigma: f64) -> LogNormal {
        LogNormal(Normal::new_unchecked(mu, sigma))
    }
}

impl From<Params> for LogNormal {
    fn from(params: Params) -> LogNormal { LogNormal(params.into()) }
}

impl From<Normal> for LogNormal {
    fn from(normal: Normal) -> LogNormal { LogNormal(normal) }
}

impl Distribution for LogNormal {
    type Support = PositiveReals<f64>;
    type Params = Params;

    fn support(&self) -> PositiveReals<f64> { positive_reals() }

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

impl Univariate for LogNormal {}

impl UvMoments for LogNormal {
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
        let sigma = (self.0).0.Sigma.0;
        let sigma2 = sigma * sigma;

        (sigma2.exp() + 2.0) * (sigma2.exp() - 1.0).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let sigma = (self.0).0.Sigma.0;
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

impl FisherInformation<2> for LogNormal {
    fn fisher_information(&self) -> [[f64; 2]; 2] {
        let sigma = (self.0).0.Sigma.0;
        let one_over_sigma2 = 1.0 / sigma / sigma;

        [
            [one_over_sigma2, 0.0],
            [0.0, one_over_sigma2 * one_over_sigma2 / 2.0]
        ]
    }
}

impl fmt::Display for LogNormal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LogNormal({}, {})", (self.0).0.mu.0, self.variance())
    }
}
