use crate::{
    consts::THREE_HALVES,
    statistics::{Modes, Quantiles, ShannonEntropy, UvMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::real::{PositiveReals, positive_reals};
use special_fun::FloatSpecial;
use std::fmt;

pub use crate::params::DOF;

params! {
    #[derive(Copy)]
    Params {
        k: DOF<usize>
    }
}

new_dist!(Chi<Params>);

macro_rules! get_k {
    ($self:ident) => {
        $self.0.k.0 as f64
    };
}

impl Chi {
    pub fn new(k: usize) -> Result<Chi, failure::Error> { Params::new(k).map(Chi) }

    pub fn new_unchecked(k: usize) -> Chi { Chi(Params::new_unchecked(k)) }
}

impl Distribution for Chi {
    type Support = PositiveReals<f64>;
    type Params = Params;

    fn support(&self) -> PositiveReals<f64> { positive_reals() }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        Probability::new_unchecked((get_k!(self) / 2.0).gammainc(x * x / 2.0))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for Chi {
    fn pdf(&self, x: &f64) -> f64 {
        let k = get_k!(self);
        let ko2 = k / 2.0;
        let norm = 2.0f64.powf(ko2 - 1.0) * ko2.gamma();

        x.powf(k - 1.0) * (-x * x / 2.0).exp() / norm
    }
}

impl Univariate for Chi {}

impl UvMoments for Chi {
    fn mean(&self) -> f64 {
        let k = get_k!(self);

        2.0f64.sqrt() * ((k + 1.0) / 2.0).gamma() / (k / 2.0).gamma()
    }

    fn variance(&self) -> f64 {
        let k = get_k!(self);
        let mu = self.mean();

        k - mu * mu
    }

    fn skewness(&self) -> f64 {
        let mu = self.mean();
        let var = self.variance();

        mu / var.powf(THREE_HALVES) * (1.0 - 2.0 * var)
    }

    fn excess_kurtosis(&self) -> f64 {
        let mu = self.mean();
        let var = self.variance();
        let std = var.sqrt();
        let skewness = self.skewness();

        2.0 / var * (1.0 - mu * std * skewness - var)
    }
}

impl Quantiles for Chi {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        let k = get_k!(self);

        (k * (1.0 - 2.0 / 9.0 / k).powi(3)).sqrt()
    }
}

impl Modes for Chi {
    fn modes(&self) -> Vec<f64> {
        let k = self.0.k.0;

        if k >= 1 {
            vec![((k - 1) as f64).sqrt()]
        } else {
            vec![]
        }
    }
}

impl ShannonEntropy for Chi {
    fn shannon_entropy(&self) -> f64 {
        let k = get_k!(self);
        let ko2 = k / 2.0;

        ko2.gamma().ln() + (k - 2.0f64.ln() - (k - 1.0) * ko2.digamma())
    }
}

impl fmt::Display for Chi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Chi({})", self.0.k.0) }
}
