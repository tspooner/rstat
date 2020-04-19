use crate::{
    consts::{PI2, PI3, TWELVE_FIFTHS, TWENTY_SEVEN_FIFTHS},
    prelude::*,
};
use rand::Rng;
use spaces::real::Reals;
use std::fmt;

locscale_params! {
    Params {
        mu<f64>,
        beta<f64>
    }
}

new_dist!(Gumbel<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.beta.0)
    };
}

impl Gumbel {
    pub fn new(mu: f64, beta: f64) -> Result<Gumbel, failure::Error> {
        Params::new(mu, beta).map(|p| Gumbel(p))
    }

    pub fn new_unchecked(mu: f64, beta: f64) -> Gumbel { Gumbel(Params::new_unchecked(mu, beta)) }

    #[inline(always)]
    pub fn z(&self, x: f64) -> f64 {
        let (mu, beta) = get_params!(self);

        (x - mu) / beta
    }
}

impl Default for Gumbel {
    fn default() -> Gumbel { Gumbel(Params::new_unchecked(0.0, 1.0)) }
}

impl Distribution for Gumbel {
    type Support = Reals;
    type Params = Params;

    fn support(&self) -> Reals { Reals }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let z = self.z(*x);

        Probability::new_unchecked((-(-z).exp()).exp())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for Gumbel {
    fn pdf(&self, x: &f64) -> f64 {
        let z = self.z(*x);

        (-z - (-z).exp()).exp() / self.0.beta.0
    }
}

impl UnivariateMoments for Gumbel {
    fn mean(&self) -> f64 {
        use special_fun::FloatSpecial;

        let (mu, beta) = get_params!(self);

        mu + beta * -(1.0f64.digamma())
    }

    fn variance(&self) -> f64 {
        let beta = self.0.beta.0;

        PI2 * beta * beta / 6.0
    }

    fn skewness(&self) -> f64 {
        use special_fun::FloatSpecial;

        12.0 * 6.0f64.sqrt() * 3.0f64.riemann_zeta() / PI3
    }

    fn kurtosis(&self) -> f64 { TWENTY_SEVEN_FIFTHS }

    fn excess_kurtosis(&self) -> f64 { TWELVE_FIFTHS }
}

impl Quantiles for Gumbel {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        let (mu, beta) = get_params!(self);

        mu - beta * 2.0f64.ln().ln()
    }
}

impl Modes for Gumbel {
    fn modes(&self) -> Vec<f64> { vec![self.0.mu.0] }
}

impl ShannonEntropy for Gumbel {
    fn shannon_entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        self.0.beta.0.ln() + -(1.0f64.digamma()) + 1.0
    }
}

impl fmt::Display for Gumbel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mu, beta) = get_params!(self);

        write!(f, "Gumbel({}, {})", mu, beta)
    }
}
