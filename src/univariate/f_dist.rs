use crate::{
    statistics::{Modes, ShannonEntropy, UnivariateMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

pub use crate::params::DOF;

params! {
    #[derive(Copy)]
    Params {
        d1: DOF<usize>,
        d2: DOF<usize>
    }
}

new_dist!(FDist<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.d1.0, $self.0.d2.0)
    };
}

impl FDist {
    pub fn new(d1: usize, d2: usize) -> Result<FDist, failure::Error> {
        Params::new(d1, d2).map(FDist)
    }

    pub fn new_unchecked(d1: usize, d2: usize) -> FDist { FDist(Params::new_unchecked(d1, d2)) }
}

impl Distribution for FDist {
    type Support = PositiveReals;
    type Params = Params;

    fn support(&self) -> PositiveReals { PositiveReals }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;

        let (d1, d2) = get_params!(self);
        let (d1, d2) = (d1 as f64, d2 as f64);

        let x = d1 * x / (d1 * x + d2);

        Probability::new_unchecked(x.betainc(d1 / 2.0, d2 / 2.0))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let (d1, d2) = get_params!(self);

        rand_distr::FisherF::new(d1 as f64, d2 as f64)
            .unwrap()
            .sample(rng)
    }
}

impl ContinuousDistribution for FDist {
    fn pdf(&self, x: &f64) -> f64 {
        use special_fun::FloatSpecial;

        let (d1, d2) = get_params!(self);
        let (d1, d2) = (d1 as f64, d2 as f64);

        let numerator = ((d1 * x).powf(d1) * d2.powf(d2) / (d1 * x + d2).powf(d1 + d2)).sqrt();
        let denominator = x * (d1 / 2.0).beta(d2 / 2.0);

        numerator / denominator
    }
}

impl UnivariateMoments for FDist {
    fn mean(&self) -> f64 {
        match self.0.d2.0 {
            d2 if d2 <= 2 => undefined!("Mean is undefined for values of d2 <= 2."),
            d2 => {
                let d2 = d2 as f64;

                d2 / (d2 - 2.0)
            },
        }
    }

    fn variance(&self) -> f64 {
        match self.0.d2.0 {
            d2 if d2 <= 4 => undefined!("Variance is undefined for values of d2 <= 4."),
            d2 => {
                let (d1, d2) = (self.0.d1.0 as f64, d2 as f64);
                let d2m2 = d2 - 2.0;

                2.0 * d2 * d2 * (d1 + d2m2) / d1 / d2m2 / d2m2 / (d2 - 4.0)
            },
        }
    }

    fn skewness(&self) -> f64 {
        match self.0.d2.0 {
            d2 if d2 <= 4 => undefined!("Skewness is undefined for values of d2 <= 6."),
            d2 => {
                let (d1, d2) = (self.0.d1.0 as f64, d2 as f64);

                let numerator = (2.0 * d1 + d2 - 2.0) * (8.0 * (d2 - 4.0)).sqrt();
                let denominator = (d2 - 6.0) * (d1 * (d1 + d2 - 2.0)).sqrt();

                numerator / denominator
            },
        }
    }

    fn excess_kurtosis(&self) -> f64 {
        match self.0.d2.0 {
            d2 if d2 <= 4 => undefined!("Kurtosis is undefined for values of d2 <= 8."),
            d2 => {
                let (d1, d2) = (self.0.d1.0 as f64, d2 as f64);
                let d2m2 = d2 - 2.0;

                let numerator =
                    12.0 * d1 * (5.0 * d2 - 22.0) * (d1 + d2m2) + (d2 - 4.0) * d2m2 * d2m2;
                let denominator = d1 * (d2 - 6.0) * (d2 - 8.0) * (d1 + d2m2);

                numerator / denominator
            },
        }
    }
}

impl Modes for FDist {
    fn modes(&self) -> Vec<f64> {
        match self.0.d1.0 {
            d1 if d1 <= 2 => undefined!("Mode is undefined for values of d1 <= 2."),
            d1 => {
                let (d1, d2) = (d1 as f64, self.0.d2.0 as f64);

                vec![(d1 - 2.0) / d1 * d2 / (d2 + 2.0)]
            },
        }
    }
}

impl ShannonEntropy for FDist {
    fn shannon_entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let (d1, d2) = get_params!(self);
        let (d1, d2) = (d1 as f64, d2 as f64);

        let d1o2 = d1 / 2.0;
        let d2o2 = d2 / 2.0;

        d1o2.gamma().ln() + d2o2.gamma().ln() - ((d1 + d2) / 2.0).gamma().ln()
            + (1.0 - d1o2) * (1.0 + d1o2).digamma()
            - (1.0 - d2o2) * (1.0 + d2o2).digamma()
            + ((d1 + d2) / 2.0) * ((d1 + d2) / 2.0).digamma()
            + (d1 / d2).ln()
    }
}

impl fmt::Display for FDist {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "F({}, {})", self.0.d1.0, self.0.d2.0)
    }
}
