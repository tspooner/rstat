use crate::{
    consts::{PI_E_2, ONE_HALF, ONE_THIRD, ONE_TWELTH, ONE_TWENTY_FOURTH, NINETEEN_OVER_360},
    fitting::MLE,
    prelude::*,
};
use ndarray::Array2;
use rand::Rng;
use spaces::discrete::Naturals;
use std::fmt;
use super::factorial;

pub use crate::params::Rate;

new_dist!(Poisson<Rate<f64>>);

macro_rules! get_lambda {
    ($self:ident) => { ($self.0).0 }
}

impl Poisson {
    pub fn new(lambda: f64) -> Result<Poisson, failure::Error> {
        Ok(Poisson(Rate::new(lambda)?))
    }

    pub fn new_unchecked(lambda: f64) -> Poisson {
        Poisson(Rate(lambda))
    }
}

impl Distribution for Poisson {
    type Support = Naturals;
    type Params = Rate<f64>;

    fn support(&self) -> Naturals { Naturals }

    fn params(&self) -> Rate<f64> { self.0 }

    fn cdf(&self, _: &u64) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        use rand_distr::Distribution as _;

        rand_distr::Poisson::<f64>::new(get_lambda!(self)).unwrap().sample(rng)
    }
}

impl DiscreteDistribution for Poisson {
    fn pmf(&self, k: &u64) -> Probability {
        let l = get_lambda!(self);
        let p = l.powi(*k as i32) * l.exp() / factorial(*k) as f64;

        Probability::new_unchecked(p)
    }
}

impl UnivariateMoments for Poisson {
    fn mean(&self) -> f64 { get_lambda!(self) }

    fn variance(&self) -> f64 { get_lambda!(self) }

    fn skewness(&self) -> f64 { get_lambda!(self).powf(-ONE_HALF) }

    fn kurtosis(&self) -> f64 { 1.0 / get_lambda!(self) }
}

impl Quantiles for Poisson {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        let l = get_lambda!(self);

        (l + ONE_THIRD - 0.02 / l).floor()
    }
}

impl Modes for Poisson {
    fn modes(&self) -> Vec<u64> {
        vec![get_lambda!(self).floor() as u64]
    }
}

impl Entropy for Poisson {
    fn entropy(&self) -> f64 {
        let l = get_lambda!(self);

        (PI_E_2 * l).ln() / 2.0 - ONE_TWELTH / l -
            ONE_TWENTY_FOURTH / l / l - NINETEEN_OVER_360 / l / l / l
    }
}

impl FisherInformation for Poisson {
    fn fisher_information(&self) -> Array2<f64> {
        Array2::from_elem((1, 1), get_lambda!(self))
    }
}

impl Convolution<Poisson> for Poisson {
    type Output = Poisson;

    fn convolve(self, rv: Poisson) -> Result<Poisson, failure::Error> {
        Ok(Poisson(Rate(get_lambda!(self) + get_lambda!(rv))))
    }
}

impl MLE for Poisson {
    fn fit_mle(xs: &[u64]) -> Result<Self, failure::Error> {
        let n = xs.len() as f64;

        Poisson::new(xs.iter().fold(0.0, |acc, &x| acc + x as f64) as f64 / n)
    }
}

impl fmt::Display for Poisson {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Poi({})", get_lambda!(self))
    }
}
