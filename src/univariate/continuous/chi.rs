use crate::{
    consts::THREE_HALVES,
    prelude::*,
};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Chi {
    pub k: usize,
}

impl Chi {
    pub fn new(k: usize) -> Chi {
        Chi { k }
    }
}

impl Distribution for Chi {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        (self.k as f64 / 2.0).gammainc(x * x / 2.0).into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Chi {
    fn pdf(&self, x: f64) -> f64 {
        use special_fun::FloatSpecial;

        let k = self.k as f64;
        let ko2 = k / 2.0;
        let norm = 2.0f64.powf(ko2 - 1.0) * ko2.gamma();

        x.powf(k - 1.0) * (-x * x / 2.0).exp() / norm
    }
}

impl UnivariateMoments for Chi {
    fn mean(&self) -> f64 {
        use special_fun::FloatSpecial;

        let k = self.k as f64;

        2.0f64.sqrt() * ((k + 1.0) / 2.0).gamma() / (k / 2.0).gamma()
    }

    fn variance(&self) -> f64 {
        let mu = self.mean();

        self.k as f64 - mu * mu
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
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        let k = self.k as f64;

        (k * (1.0 - 2.0 / 9.0 / k).powi(3)).sqrt()
    }
}

impl Modes for Chi {
    fn modes(&self) -> Vec<f64> {
        if self.k >= 1 {
            vec![(self.k as f64 - 1.0).sqrt()]
        } else {
            vec![]
        }
    }
}

impl Entropy for Chi {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let k = self.k as f64;
        let ko2 = k / 2.0;

        ko2.gamma().ln() + (k - 2.0f64.ln() - (k - 1.0) * ko2.digamma())
    }
}

impl fmt::Display for Chi {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Chi({})", self.k)
    }
}
