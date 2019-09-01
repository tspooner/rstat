use crate::{
    consts::THREE_HALVES,
    core::*,
};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Weibull {
    pub lambda: f64,
    pub k: f64,
}

impl Weibull {
    pub fn new(lambda: f64, k: f64) -> Weibull {
        assert_positive_real!(lambda);
        assert_positive_real!(k);

        Weibull { lambda, k }
    }

    fn gamma_i(&self, num: f64) -> f64 {
        use special_fun::FloatSpecial;

        (1.0 + num / self.k).gamma()
    }
}

impl Default for Weibull {
    fn default() -> Weibull {
        Weibull {
            lambda: 1.0,
            k: 1.0,
        }
    }
}

impl Into<rand_distr::Weibull<f64>> for Weibull {
    fn into(self) -> rand_distr::Weibull<f64> {
        rand_distr::Weibull::new(self.lambda, self.k).unwrap()
    }
}

impl Into<rand_distr::Weibull<f64>> for &Weibull {
    fn into(self) -> rand_distr::Weibull<f64> {
        rand_distr::Weibull::new(self.lambda, self.k).unwrap()
    }
}

impl Distribution for Weibull {
    type Support = PositiveReals;

    fn support(&self) -> PositiveReals {
        PositiveReals
    }

    fn cdf(&self, x: f64) -> Probability {
        if x >= 0.0 {
            1.0 - (-(x / self.lambda).powf(self.k)).exp()
        } else {
            0.0
        }
        .into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        let sampler: rand_distr::Weibull<f64> = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for Weibull {
    fn pdf(&self, x: f64) -> f64 {
        if x >= 0.0 {
            let xol = x / self.lambda;

            self.k / self.lambda * xol.powf(self.k - 1.0) * (-xol.powf(self.k)).exp()
        } else {
            0.0
        }
    }
}

impl UnivariateMoments for Weibull {
    fn mean(&self) -> f64 {
        self.lambda * self.gamma_i(1.0)
    }

    fn variance(&self) -> f64 {
        let bracket_term1 = self.gamma_i(2.0);
        let bracket_term2 = self.gamma_i(1.0);

        let bracket_term = bracket_term1 - bracket_term2 * bracket_term2;

        self.lambda * self.lambda * bracket_term
    }

    fn skewness(&self) -> f64 {
        let mu = self.mean();
        let var = self.variance();

        let numerator = self.gamma_i(3.0) * self.lambda * self.lambda * self.lambda
            - 3.0 * mu * var
            - mu * mu * mu;
        let denominator = var.powf(THREE_HALVES);

        numerator / denominator
    }

    fn excess_kurtosis(&self) -> f64 {
        // Version 1:
        let gamma_1 = self.gamma_i(1.0);
        let gamma_2 = self.gamma_i(2.0);
        let gamma_1_sq = gamma_1 * gamma_1;

        let g2mg1sq = gamma_2 - gamma_1_sq;

        let numerator = 12.0 * gamma_1_sq * gamma_2
            - 6.0 * gamma_1_sq * gamma_1_sq
            - 3.0 * gamma_2 * gamma_2
            - 4.0 * gamma_1 * self.gamma_i(3.0)
            + self.gamma_i(4.0);
        let denominator = g2mg1sq * g2mg1sq;

        numerator / denominator

        // Version 2:
        // let gamma_4 = self.gamma_i(4.0);

        // let mu = self.mean();
        // let mu2 = mu * mu;
        // let var = self.variance();
        // let skewness = self.skewness();

        // let numerator = self.lambda * self.lambda * self.lambda * self.lambda * gamma_4 -
        // 4.0 * skewness * var.powf(THREE_HALVES) * mu -
        // 6.0 * mu2 * var - mu2 * mu2;
        // let denominator = var * var;

        // numerator / denominator
    }
}

impl Quantiles for Weibull {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.lambda * 2.0f64.ln().powf(1.0 / self.k)
    }
}

impl Modes for Weibull {
    fn modes(&self) -> Vec<f64> {
        vec![if self.k > 0.0 {
            0.0
        } else {
            self.lambda * ((self.k - 1.0) / self.k).powf(1.0 / self.k)
        }]
    }
}

impl Entropy for Weibull {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let gamma = -(1.0f64.digamma());

        gamma * (1.0 - 1.0 / self.k) + (self.lambda / self.k).ln() + 1.0
    }
}

impl fmt::Display for Weibull {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Weibull({}, {})", self.lambda, self.k)
    }
}
