use crate::{
    statistics::{Modes, ShannonEntropy, UvMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand::Rng;
use spaces::real::{PositiveReals, positive_reals};
use std::fmt;

shape_params! {
    #[derive(Copy)]
    Params<f64> {
        alpha,
        beta
    }
}

new_dist!(BetaPrime<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.alpha.0, $self.0.beta.0)
    };
}

impl BetaPrime {
    pub fn new(alpha: f64, beta: f64) -> Result<BetaPrime, failure::Error> {
        Params::new(alpha, beta).map(BetaPrime)
    }

    pub fn new_unchecked(alpha: f64, beta: f64) -> BetaPrime {
        BetaPrime(Params::new_unchecked(alpha, beta))
    }
}

impl Default for BetaPrime {
    fn default() -> BetaPrime { BetaPrime(Params::new_unchecked(1.0, 1.0)) }
}

impl Distribution for BetaPrime {
    type Support = PositiveReals<f64>;
    type Params = Params;

    fn support(&self) -> PositiveReals<f64> { positive_reals() }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;

        let (alpha, beta) = get_params!(self);

        Probability::new_unchecked((x / (1.0 + x)).betainc(alpha, beta))
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for BetaPrime {
    fn pdf(&self, x: &f64) -> f64 {
        use special_fun::FloatSpecial;

        let (a, b) = get_params!(self);

        let numerator = x.powf(a - 1.0) * (1.0 + x).powf(-a - b);
        let denominator = a.beta(b);

        numerator / denominator
    }
}

impl Univariate for BetaPrime {}

impl UvMoments for BetaPrime {
    fn mean(&self) -> f64 {
        let (a, b) = get_params!(self);

        if b <= 1.0 {
            undefined!("mean is undefined for values of beta <= 1.")
        }

        a / (b - 1.0)
    }

    fn variance(&self) -> f64 {
        let (a, b) = get_params!(self);

        if b <= 2.0 {
            undefined!("variance is undefined for values of beta <= 2.")
        }

        let bm1 = b - 1.0;

        a * (a + bm1) / (b - 2.0) / bm1 / bm1
    }

    fn skewness(&self) -> f64 {
        let (a, b) = get_params!(self);

        if b <= 3.0 {
            undefined!("skewness is undefined for values of beta <= 3.")
        }

        let bm1 = b - 1.0;

        2.0 * (2.0 * a + bm1) / (b - 3.0) * ((b - 2.0) / (a * (a + bm1))).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let (a, b) = get_params!(self);
        let bm1 = b - 1.0;

        let numerator = 6.0 * (a + bm1) * (5.0 * b - 11.0) + bm1 * bm1 * (b - 2.0);
        let denominator = a * (a + bm1) * (b - 3.0) * (b - 4.0);

        numerator / denominator
    }
}

impl Modes for BetaPrime {
    fn modes(&self) -> Vec<f64> {
        let (a, b) = get_params!(self);

        if a >= 1.0 {
            vec![(a - 1.0) / (b + 1.0)]
        } else {
            vec![0.0]
        }
    }
}

impl ShannonEntropy for BetaPrime {
    fn shannon_entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let (a, b) = get_params!(self);
        let apb = a + b;

        a.logbeta(b) - (a - 1.0) * a.digamma() - (b - 1.0) * b.digamma()
            + (apb - 2.0) * apb.digamma()
    }
}

impl fmt::Display for BetaPrime {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (alpha, beta) = get_params!(self);

        write!(f, "BetaPrime({}, {})", alpha, beta)
    }
}
