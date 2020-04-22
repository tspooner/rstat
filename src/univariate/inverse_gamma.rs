use crate::{
    statistics::{Modes, ShannonEntropy, UnivariateMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

shape_params! {
    #[derive(Copy)]
    Params<f64> {
        alpha,
        beta
    }
}

new_dist!(InvGamma<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.alpha.0, $self.0.beta.0)
    };
}

impl InvGamma {
    pub fn new(alpha: f64, beta: f64) -> Result<InvGamma, failure::Error> {
        Params::new(alpha, beta).map(InvGamma)
    }

    pub fn new_unchecked(alpha: f64, beta: f64) -> InvGamma {
        InvGamma(Params::new_unchecked(alpha, beta))
    }
}

impl Default for InvGamma {
    fn default() -> InvGamma { InvGamma(Params::new_unchecked(1.0, 1.0)) }
}

impl Distribution for InvGamma {
    type Support = PositiveReals;
    type Params = Params;

    fn support(&self) -> PositiveReals { PositiveReals }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;

        let (alpha, beta) = get_params!(self);

        Probability::new_unchecked(alpha.gammainc(beta / x) / alpha.gamma())
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for InvGamma {
    fn pdf(&self, x: &f64) -> f64 {
        use special_fun::FloatSpecial;

        let (alpha, beta) = get_params!(self);

        beta.powf(alpha) * x.powf(-alpha - 1.0) * (-beta / x).exp() / alpha.gamma()
    }
}

impl UnivariateMoments for InvGamma {
    fn mean(&self) -> f64 {
        match self.0.alpha.0 {
            alpha if alpha <= 1.0 => undefined!("Mean is undefined for alpha <= 1."),
            alpha => self.0.beta.0 / (alpha - 1.0),
        }
    }

    fn variance(&self) -> f64 {
        match self.0.alpha.0 {
            alpha if alpha <= 2.0 => undefined!("Variance is undefined for alpha <= 2."),
            alpha => {
                let am1 = alpha - 1.0;
                let beta = self.0.beta.0;

                beta * beta / am1 / am1 / (alpha - 2.0)
            },
        }
    }

    fn skewness(&self) -> f64 {
        match self.0.alpha.0 {
            alpha if alpha <= 3.0 => undefined!("Skewness is undefined for alpha <= 3."),
            alpha => 4.0 * (alpha - 2.0).sqrt() / (alpha - 3.0),
        }
    }

    fn excess_kurtosis(&self) -> f64 {
        match self.0.alpha.0 {
            alpha if alpha <= 4.0 => undefined!("Kurtosis is undefined for alpha <= 4."),
            alpha => (30.0 * alpha - 66.0) / (alpha - 3.0) / (alpha - 4.0),
        }
    }
}

impl Modes for InvGamma {
    fn modes(&self) -> Vec<f64> {
        let (alpha, beta) = get_params!(self);

        vec![beta / (alpha + 1.0)]
    }
}

impl ShannonEntropy for InvGamma {
    fn shannon_entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let (alpha, beta) = get_params!(self);

        alpha + (beta * alpha.gamma()).ln() - (1.0 + alpha) * alpha.digamma()
    }
}

impl fmt::Display for InvGamma {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (alpha, beta) = get_params!(self);

        write!(f, "Inv-Gamma({}, {})", alpha, beta)
    }
}
