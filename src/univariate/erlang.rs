use crate::{prelude::*, validation::{Validator, Result}};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Erlang {
    pub k: usize,
    pub lambda: f64,
}

impl Erlang {
    pub fn new(k: usize, lambda: f64) -> Result<Erlang> {
        Validator
            .require_gte(k, 1)?
            .require_positive_real(lambda)
            .map(|_| Erlang::new_unchecked(k, lambda))
    }

    pub fn new_unchecked(k: usize, lambda: f64) -> Erlang {
        Erlang { k, lambda }
    }

    pub fn mu(&self) -> f64 {
        1.0 / self.lambda
    }
}

impl Default for Erlang {
    fn default() -> Erlang {
        Erlang { k: 1, lambda: 1.0 }
    }
}

impl Distribution for Erlang {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        use special_fun::FloatSpecial;

        Probability::new_unchecked(
            (self.k as f64).gammainc(self.lambda * x) / (self.k as f64).factorial()
        )
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for Erlang {
    fn pdf(&self, x: f64) -> f64 {
        use special_fun::FloatSpecial;

        self.lambda.powi(self.k as i32) * x.powi(self.k as i32 - 1) * (-self.lambda * x).exp()
            / (self.k as f64).factorial()
    }
}

impl UnivariateMoments for Erlang {
    fn mean(&self) -> f64 {
        self.k as f64 / self.lambda
    }

    fn variance(&self) -> f64 {
        self.k as f64 / self.lambda / self.lambda
    }

    fn skewness(&self) -> f64 {
        2.0 / (self.k as f64).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        6.0 / self.k as f64
    }
}

impl Modes for Erlang {
    fn modes(&self) -> Vec<f64> {
        vec![(self.k - 1) as f64 / self.lambda]
    }
}

impl fmt::Display for Erlang {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Erlang({})", self.k)
    }
}
