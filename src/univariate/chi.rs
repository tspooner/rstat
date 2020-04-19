use crate::{consts::THREE_HALVES, prelude::*};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

pub use crate::params::DOF;

new_dist!(Chi<DOF<usize>>);

macro_rules! get_k {
    ($self:ident) => {
        ($self.0).0 as f64
    };
}

impl Chi {
    pub fn new(dof: usize) -> Result<Chi, failure::Error> { Ok(Chi(DOF::new(dof)?)) }

    pub fn new_unchecked(dof: usize) -> Chi { Chi(DOF(dof)) }

    #[inline(always)]
    pub fn k(&self) -> f64 { get_k!(self) }
}

impl Distribution for Chi {
    type Support = PositiveReals;
    type Params = DOF<usize>;

    fn support(&self) -> PositiveReals { PositiveReals }

    fn params(&self) -> DOF<usize> { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;

        Probability::new_unchecked((get_k!(self) / 2.0).gammainc(x * x / 2.0))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for Chi {
    fn pdf(&self, x: &f64) -> f64 {
        use special_fun::FloatSpecial;

        let k = get_k!(self);
        let ko2 = k / 2.0;
        let norm = 2.0f64.powf(ko2 - 1.0) * ko2.gamma();

        x.powf(k - 1.0) * (-x * x / 2.0).exp() / norm
    }
}

impl UnivariateMoments for Chi {
    fn mean(&self) -> f64 {
        use special_fun::FloatSpecial;

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
        let k = (self.0).0;

        if k >= 1 {
            vec![((k - 1) as f64).sqrt()]
        } else {
            vec![]
        }
    }
}

impl ShannonEntropy for Chi {
    fn shannon_entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let k = get_k!(self);
        let ko2 = k / 2.0;

        ko2.gamma().ln() + (k - 2.0f64.ln() - (k - 1.0) * ko2.digamma())
    }
}

impl fmt::Display for Chi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Chi({})", self.k()) }
}
