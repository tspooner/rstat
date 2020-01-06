use crate::{prelude::*, validation::{Validator, Result}};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct FDist {
    pub d1: usize,
    pub d2: usize,
}

impl FDist {
    pub fn new(d1: usize, d2: usize) -> Result<FDist> {
        Validator
            .require_natural(d1)?
            .require_natural(d2)
            .map(|_| FDist::new_unchecked(d1, d2))
    }

    pub fn new_unchecked(d1: usize, d2: usize) -> FDist {
        FDist { d1, d2 }
    }
}

impl Into<rand_distr::FisherF<f64>> for FDist {
    fn into(self) -> rand_distr::FisherF<f64> {
        rand_distr::FisherF::new(self.d1 as f64, self.d2 as f64).unwrap()
    }
}

impl Into<rand_distr::FisherF<f64>> for &FDist {
    fn into(self) -> rand_distr::FisherF<f64> {
        rand_distr::FisherF::new(self.d1 as f64, self.d2 as f64).unwrap()
    }
}

impl Distribution for FDist {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        let d1 = self.d1 as f64;
        let d2 = self.d2 as f64;

        let x = d1 * x / (d1 * x + d2);

        Probability::new_unchecked(x.betainc(d1 / 2.0, d2 / 2.0))
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        let sampler: rand_distr::FisherF<f64> = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for FDist {
    fn pdf(&self, x: f64) -> f64 {
        use special_fun::FloatSpecial;

        let d1 = self.d1 as f64;
        let d2 = self.d2 as f64;

        let numerator = ((d1 * x).powf(d1) * d2.powf(d2) / (d1 * x + d2).powf(d1 + d2)).sqrt();
        let denominator = x * (d1 / 2.0).beta(d2 / 2.0);

        numerator / denominator
    }
}

impl UnivariateMoments for FDist {
    fn mean(&self) -> f64 {
        if self.d2 <= 2 {
            unimplemented!("Mean is undefined for values of d2 <= 2.")
        }

        let d2 = self.d2 as f64;

        d2 / (d2 - 2.0)
    }

    fn variance(&self) -> f64 {
        if self.d2 <= 4 {
            unimplemented!("Variance is undefined for values of d2 <= 4.")
        }

        let d1 = self.d1 as f64;
        let d2 = self.d2 as f64;

        let d2m2 = d2 - 2.0;

        2.0 * d2 * d2 * (d1 + d2m2) / d1 / d2m2 / d2m2 / (d2 - 4.0)
    }

    fn skewness(&self) -> f64 {
        if self.d2 <= 6 {
            unimplemented!("Skewness is undefined for values of d2 <= 6.")
        }

        let d1 = self.d1 as f64;
        let d2 = self.d2 as f64;

        let numerator = (2.0 * d1 + d2 - 2.0) * (8.0 * (d2 - 4.0)).sqrt();
        let denominator = (d2 - 6.0) * (d1 * (d1 + d2 - 2.0)).sqrt();

        numerator / denominator
    }

    fn excess_kurtosis(&self) -> f64 {
        if self.d2 <= 8 {
            unimplemented!("Kurtosis is undefined for values of d2 <= 8.")
        }

        let d1 = self.d1 as f64;
        let d2 = self.d2 as f64;

        let d2m2 = d2 - 2.0;

        let numerator = 12.0 * d1 * (5.0 * d2 - 22.0) * (d1 + d2m2) + (d2 - 4.0) * d2m2 * d2m2;
        let denominator = d1 * (d2 - 6.0) * (d2 - 8.0) * (d1 + d2m2);

        numerator / denominator
    }
}

impl Modes for FDist {
    fn modes(&self) -> Vec<f64> {
        if self.d1 <= 2 {
            unimplemented!("Mode is undefined for values of d1 <= 2.")
        }

        let d1 = self.d1 as f64;
        let d2 = self.d2 as f64;

        vec![(d1 - 2.0) / d1 * d2 / (d2 + 2.0)]
    }
}

impl Entropy for FDist {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let d1 = self.d1 as f64;
        let d2 = self.d2 as f64;

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
        write!(f, "F({}, {})", self.d1, self.d2)
    }
}
