use crate::{
    consts::PI,
    prelude::*,
};
use rand::Rng;
use spaces::real::Reals;
use std::fmt;

pub use crate::params::Loc;

new_dist!(StudentT<Loc<f64>>);

macro_rules! get_nu {
    ($self:ident) => { ($self.0).0 }
}

impl Distribution for StudentT {
    type Support = Reals;
    type Params = Loc<f64>;

    fn support(&self) -> Reals { Reals }

    fn params(&self) -> Loc<f64> { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;

        let nu = get_nu!(self);
        let np1o2 = (nu + 1.0) / 2.0;
        let hyp2f1 = 0.5f64.hyp2f1(np1o2, 3.0 / 2.0, -x * x / nu);

        Probability::new_unchecked(
            0.5 + x * np1o2.gamma() * hyp2f1 / (nu * PI).sqrt() * (nu / 2.0).gamma()
        )
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        rand_distr::StudentT::new(get_nu!(self)).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for StudentT {
    fn pdf(&self, x: &f64) -> f64 {
        use special_fun::FloatSpecial;

        let nu = get_nu!(self);
        let np1o2 = (nu + 1.0) / 2.0;
        let norm = np1o2.gamma() / (nu * PI).sqrt() / (nu / 2.0).gamma();

        norm * (1.0 + x * x / nu).powf(-np1o2)
    }
}

impl UnivariateMoments for StudentT {
    fn mean(&self) -> f64 {
        if get_nu!(self) <= 1.0 {
            undefined!("Mean is undefined for nu <= 1.");
        }

        0.0
    }

    fn variance(&self) -> f64 {
        let nu = get_nu!(self);

        if nu <= 1.0 {
            undefined!("Variance is undefined for nu <= 1.");
        } else if nu <= 2.0 {
            undefined!("Variance is infinite for 1 < nu <= 2.");
        }

        nu / (nu - 2.0)
    }

    fn skewness(&self) -> f64 {
        if get_nu!(self) <= 3.0 {
            undefined!("Skewness is undefined for nu <= 1.");
        }

        0.0
    }

    fn excess_kurtosis(&self) -> f64 {
        let nu = get_nu!(self);

        if nu <= 2.0 {
            undefined!("Kurtosis is undefined for nu <= 2.");
        } else if nu <= 4.0 {
            undefined!("Kurtosis is infinite for 2 < nu <= 4.");
        }

        6.0 / (nu - 4.0)
    }
}

impl Quantiles for StudentT {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 { 0.0 }
}

impl Modes for StudentT {
    fn modes(&self) -> Vec<f64> { vec![0.0] }
}

impl Entropy for StudentT {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let nu = get_nu!(self);
        let no2 = nu / 2.0;
        let np1o2 = (nu + 1.0) / 2.0;

        np1o2 * (np1o2.gamma() - no2.gamma()) + (nu.sqrt() * (no2.beta(0.5)))
    }
}

impl fmt::Display for StudentT {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "StudentT({})", get_nu!(self))
    }
}
