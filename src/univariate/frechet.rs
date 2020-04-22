use crate::{
    statistics::{Modes, Quantiles, ShannonEntropy, UnivariateMoments},
    univariate::uniform::Uniform,
    ContinuousDistribution,
    Distribution,
    Probability,
};
use rand::Rng;
use spaces::real::Interval;
use std::{f64::INFINITY, fmt};

pub use crate::params::Shape;

params! {
    Params {
        alpha: Shape<f64>
    }
}

new_dist!(Frechet<Params>);

macro_rules! get_alpha {
    ($self:ident) => {
        $self.0.alpha.0
    };
}

impl Frechet {
    pub fn new(alpha: f64) -> Result<Frechet, failure::Error> { Params::new(alpha).map(Frechet) }

    pub fn new_unchecked(alpha: f64) -> Frechet { Frechet(Params::new_unchecked(alpha)) }
}

impl Default for Frechet {
    fn default() -> Frechet { Frechet::new_unchecked(1.0) }
}

impl Distribution for Frechet {
    type Support = Interval;
    type Params = Params;

    fn support(&self) -> Interval { Interval::left_bounded(0.0) }

    fn params(&self) -> Params { Params::new_unchecked(get_alpha!(self)) }

    fn cdf(&self, x: &f64) -> Probability {
        Probability::new_unchecked((-(x.powf(-get_alpha!(self)))).exp())
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let u = Uniform::default();

        (-u.sample(rng).ln()).powf(1.0 / get_alpha!(self))
    }
}

impl ContinuousDistribution for Frechet {
    fn pdf(&self, x: &f64) -> f64 {
        let cdf = self.cdf(x).unwrap();
        let alpha = get_alpha!(self);

        alpha * x.powf(-1.0 - alpha) * cdf
    }
}

impl UnivariateMoments for Frechet {
    fn mean(&self) -> f64 {
        match get_alpha!(self) {
            alpha if alpha <= 1.0 => INFINITY,
            alpha => {
                use special_fun::FloatSpecial;

                (1.0 - 1.0 / alpha).gamma()
            },
        }
    }

    fn variance(&self) -> f64 {
        match get_alpha!(self) {
            alpha if alpha <= 2.0 => INFINITY,
            alpha => {
                use special_fun::FloatSpecial;

                let gamma_1m1oa = (1.0 - 1.0 / alpha).gamma();

                (1.0 - 2.0 / alpha).gamma() - gamma_1m1oa * gamma_1m1oa
            },
        }
    }

    fn skewness(&self) -> f64 {
        match get_alpha!(self) {
            alpha if alpha <= 3.0 => INFINITY,
            alpha => {
                use special_fun::FloatSpecial;

                let gamma_1m1oa = (1.0 - 1.0 / alpha).gamma();
                let gamma_1m2oa = (1.0 - 2.0 / alpha).gamma();

                let numerator = (1.0 - 3.0 / alpha).gamma() - 3.0 * gamma_1m2oa * gamma_1m1oa
                    + 2.0 * gamma_1m1oa * gamma_1m1oa * gamma_1m1oa;

                let denominator_inner = gamma_1m2oa - gamma_1m1oa * gamma_1m1oa;
                let denominator =
                    (denominator_inner * denominator_inner * denominator_inner).sqrt();

                (numerator / denominator).into()
            },
        }
    }

    fn excess_kurtosis(&self) -> f64 {
        match get_alpha!(self) {
            alpha if alpha <= 4.0 => INFINITY,
            alpha => {
                use special_fun::FloatSpecial;

                let gamma_1m1oa = (1.0 - 1.0 / alpha).gamma();
                let gamma_1m2oa = (1.0 - 2.0 / alpha).gamma();

                let numerator = (1.0 - 4.0 / alpha).gamma()
                    - 4.0 * (1.0 - 3.0 / alpha).gamma() * gamma_1m1oa
                    + 3.0 * gamma_1m2oa * gamma_1m2oa;
                let denominator_inner = gamma_1m2oa - gamma_1m1oa * gamma_1m1oa;
                let denominator = denominator_inner * denominator_inner;

                numerator / denominator - 6.0
            },
        }
    }
}

impl Quantiles for Frechet {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 { 1.0 / 2.0f64.ln().powf(1.0 / get_alpha!(self)) }
}

impl Modes for Frechet {
    fn modes(&self) -> Vec<f64> {
        let alpha = get_alpha!(self);

        vec![(alpha / (1.0 + alpha)).powf(1.0 / alpha)]
    }
}

impl ShannonEntropy for Frechet {
    fn shannon_entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let alpha = get_alpha!(self);
        let gamma = -(1.0f64.digamma());

        1.0 + gamma / alpha + gamma + (1.0 / alpha).ln()
    }
}

impl fmt::Display for Frechet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Frechet({})", get_alpha!(self))
    }
}
