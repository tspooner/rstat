use crate::prelude::*;
use ndarray::Array2;
use rand::Rng;
use spaces::real::Interval;
use std::{f64::INFINITY, fmt};

locscale_params! {
    Params {
        x_m<f64>,
        alpha<f64>
    }
}

new_dist!(Pareto<Params>);

macro_rules! get_params {
    ($self:ident) => { ($self.0.x_m.0, $self.0.alpha.0) }
}

impl Pareto {
    pub fn new(x_m: f64, alpha: f64) -> Result<Pareto, failure::Error> {
        Params::new(x_m, alpha).map(|p| Pareto(p))
    }

    pub fn new_unchecked(x_m: f64, alpha: f64) -> Pareto {
        Pareto(Params::new_unchecked(x_m, alpha))
    }
}

impl Distribution for Pareto {
    type Support = Interval;
    type Params = Params;

    fn support(&self) -> Interval { Interval::left_bounded(self.0.x_m.0) }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let (x_m, alpha) = get_params!(self);

        Probability::new_unchecked(1.0 - (x_m / x).powf(alpha))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let (x_m, alpha) = get_params!(self);

        rand_distr::Pareto::<f64>::new(x_m, alpha).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for Pareto {
    fn pdf(&self, x: &f64) -> f64 {
        let x_m = self.0.x_m.0;

        if *x < x_m {
            0.0
        } else {
            let alpha = self.0.alpha.0;

            alpha * x_m.powf(alpha) / x.powf(alpha + 1.0)
        }
    }
}

impl UnivariateMoments for Pareto {
    fn mean(&self) -> f64 {
        match self.0.alpha.0 {
            alpha if alpha <= 1.0 => INFINITY,
            alpha => alpha * self.0.x_m.0 / (alpha - 1.0),
        }
    }

    fn variance(&self) -> f64 {
        match self.0.alpha.0 {
            alpha if alpha <= 2.0 => INFINITY,
            alpha => {
                let x_m = self.0.x_m.0;
                let am1 = alpha - 1.0;

                x_m * x_m * alpha / am1 / am1 / (alpha - 2.0)
            },
        }
    }

    fn skewness(&self) -> f64 {
        match self.0.alpha.0 {
            alpha if alpha <= 3.0 => undefined!("Variance is undefined for alpha <= 3."),
            alpha => 2.0 * (1.0 + alpha) / (alpha - 3.0) * ((alpha - 2.0) / alpha).sqrt(),
        }
    }

    fn excess_kurtosis(&self) -> f64 {
        match self.0.alpha.0 {
            alpha if alpha <= 4.0 => undefined!("Kurtosis is undefined for alpha <= 4."),
            alpha => {
                let a2 = alpha * alpha;
                let a3 = a2 * alpha;

                6.0 * (a3 + a2 - 6.0 * alpha - 2.0) / alpha / (alpha - 3.0) / (alpha - 4.0)
            },
        }
    }
}

impl Quantiles for Pareto {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        let (x_m, alpha) = get_params!(self);

        x_m * 2.0f64.powf(1.0 / alpha)
    }
}

impl Modes for Pareto {
    fn modes(&self) -> Vec<f64> {
        vec![self.0.x_m.0]
    }
}

impl Entropy for Pareto {
    fn entropy(&self) -> f64 {
        let (x_m, alpha) = get_params!(self);

        (x_m / alpha * (1.0 + 1.0 / alpha).exp()).ln()
    }
}

impl FisherInformation for Pareto {
    fn fisher_information(&self) -> Array2<f64> {
        let (x_m, alpha) = get_params!(self);
        let off_diag = -1.0 / x_m;

        unsafe {
            Array2::from_shape_vec_unchecked(
                (2, 2),
                vec![
                    alpha / x_m / x_m,
                    off_diag,
                    off_diag,
                    1.0 / alpha / alpha,
                ],
            )
        }
    }
}

impl fmt::Display for Pareto {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (x_m, alpha) = get_params!(self);

        write!(f, "Pareto({}, {})", x_m, alpha)
    }
}
