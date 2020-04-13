use crate::{
    consts::THREE_HALVES,
    params::{Scale, Shape},
    prelude::*,
};
use rand::Rng;
use spaces::real::PositiveReals;
use std::fmt;

params! {
    Params {
        lambda: Scale<f64>,
        k: Shape<f64>
    }
}

new_dist!(Weibull<Params>);

macro_rules! get_params {
    ($self:ident) => { ($self.0.lambda.0, $self.0.k.0) }
}

impl Weibull {
    fn gamma_i(&self, num: f64) -> f64 {
        use special_fun::FloatSpecial;

        (1.0 + num / self.0.k.0).gamma()
    }
}

impl Distribution for Weibull {
    type Support = PositiveReals;
    type Params = Params;

    fn support(&self) -> PositiveReals { PositiveReals }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        if *x >= 0.0 {
            let (lambda, k) = get_params!(self);

            Probability::new_unchecked(1.0 - (-(x / lambda).powf(k)).exp())
        } else {
            Probability::zero()
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let (lambda, k) = get_params!(self);

        rand_distr::Weibull::new(lambda, k).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for Weibull {
    fn pdf(&self, x: &f64) -> f64 {
        if *x >= 0.0 {
            let (lambda, k) = get_params!(self);
            let xol = x / lambda;

            k / lambda * xol.powf(k - 1.0) * (-xol.powf(k)).exp()
        } else {
            0.0
        }
    }
}

impl UnivariateMoments for Weibull {
    fn mean(&self) -> f64 { self.0.lambda.0 * self.gamma_i(1.0) }

    fn variance(&self) -> f64 {
        let lambda = self.0.lambda.0;

        let bracket_term1 = self.gamma_i(2.0);
        let bracket_term2 = self.gamma_i(1.0);

        let bracket_term = bracket_term1 - bracket_term2 * bracket_term2;

        lambda * lambda * bracket_term
    }

    fn skewness(&self) -> f64 {
        let mu = self.mean();
        let var = self.variance();
        let lambda = self.0.lambda.0;

        let numerator =
            self.gamma_i(3.0) * lambda * lambda * lambda
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
        let (lambda, k) = get_params!(self);

        lambda * 2.0f64.ln().powf(1.0 / k)
    }
}

impl Modes for Weibull {
    fn modes(&self) -> Vec<f64> {
        let k = self.0.k.0;

        vec![if k > 0.0 {
            0.0
        } else {
            self.0.lambda.0 * ((k - 1.0) / k).powf(1.0 / k)
        }]
    }
}

impl Entropy for Weibull {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let (lambda, k) = get_params!(self);
        let gamma = -(1.0f64.digamma());

        gamma * (1.0 - 1.0 / k) + (lambda / k).ln() + 1.0
    }
}

impl fmt::Display for Weibull {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (lambda, k) = get_params!(self);

        write!(f, "Weibull({}, {})", lambda, k)
    }
}
