use crate::{
    consts::{ONE_THIRD, TWO_THIRDS},
    fitting::{Likelihood, Score},
    statistics::{Modes, Quantiles, ShannonEntropy, UvMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand;
use spaces::intervals::Closed;
use special_fun::FloatSpecial;
use std::fmt;

shape_params! {
    #[derive(Copy)]
    Params<f64> { alpha, beta }
}

pub struct Grad {
    pub alpha: f64,
    pub beta: f64,
}

new_dist!(Beta<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.alpha.0, $self.0.beta.0)
    };
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Result<Beta, failure::Error> {
        Params::new(alpha, beta).map(Beta)
    }

    pub fn new_unchecked(alpha: f64, beta: f64) -> Beta {
        Beta(Params::new_unchecked(alpha, beta))
    }
}

impl Default for Beta {
    fn default() -> Beta { Beta(Params::new_unchecked(1.0, 1.0)) }
}

impl Distribution for Beta {
    type Support = Closed<f64>;
    type Params = Params;

    fn support(&self) -> Closed<f64> { Closed::unit() }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let (alpha, beta) = get_params!(self);

        Probability::new_unchecked(x.betainc(alpha, beta))
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let (alpha, beta) = get_params!(self);

        rand_distr::Beta::new(alpha, beta).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for Beta {
    fn pdf(&self, x: &f64) -> f64 {
        let (a, b) = get_params!(self);

        let numerator = x.powf(a - 1.0) * (1.0 - x).powf(b - 1.0);
        let denominator = a.beta(b);

        numerator / denominator
    }
}

impl Univariate for Beta {}

impl UvMoments for Beta {
    fn mean(&self) -> f64 {
        let (a, b) = get_params!(self);

        1.0 / (1.0 + b / a)
    }

    fn variance(&self) -> f64 {
        let (a, b) = get_params!(self);
        let apb = a + b;

        a * b / (apb * apb * (apb + 1.0))
    }

    fn skewness(&self) -> f64 {
        let (a, b) = get_params!(self);
        let apb = a + b;

        2.0 * (b - a) * (apb + 1.0).sqrt() / (apb + 2.0) / (a * b).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        let (a, b) = get_params!(self);

        let apb = a + b;
        let asb = a - b;
        let amb = a * b;

        let apbp2 = apb + 2.0;

        3.0 * asb * asb * (apb + 1.0) - amb * apbp2 / amb / apbp2 / (apb + 3.0)
    }
}

impl Quantiles for Beta {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        let (a, b) = get_params!(self);

        if (a - b).abs() < 1e-7 {
            0.5
        } else if a > 1.0 && b > 1.0 {
            (a - ONE_THIRD) / (a + b - TWO_THIRDS)
        } else if (a - 1.0).abs() < 1e-7 {
            1.0 - 2.0f64.powf(-1.0 / b)
        } else if (b - 1.0).abs() < 1e-7 {
            2.0f64.powf(-1.0 / a)
        } else if (a - 3.0).abs() < 1e-7 && (b - 2.0).abs() < 1e-7 {
            0.6142724318676105
        } else if (a - 2.0).abs() < 1e-7 && (b - 3.0).abs() < 1e-7 {
            0.38572756813238945
        } else {
            undefined!()
        }
    }
}

impl Modes for Beta {
    fn modes(&self) -> Vec<f64> {
        let (a, b) = get_params!(self);

        if a > 1.0 && b > 1.0 {
            vec![(a - 1.0) / (a + b - 2.0)]
        } else if a < 1.0 && b < 1.0 {
            vec![0.0, 1.0]
        } else {
            vec![]
        }
    }
}

impl ShannonEntropy for Beta {
    fn shannon_entropy(&self) -> f64 {
        let (a, b) = get_params!(self);
        let (am1, bm1) = (a - 1.0, b - 1.0);

        a.logbeta(b) - am1 * a.digamma() - bm1 * b.digamma() + (am1 + bm1) * (a + b).digamma()
    }
}

impl Likelihood for Beta {
    fn log_likelihood(&self, samples: &[f64]) -> f64 {
        const JITTER: f64 = 1e-9;

        let n = samples.len() as f64;

        let (a, b) = get_params!(self);
        let (am1, bm1) = (a - 1.0, b - 1.0);

        samples
            .into_iter()
            .map(|x| am1 * x.max(JITTER).ln() + bm1 * (1.0 - x).max(JITTER).ln())
            .sum::<f64>()
            - n * a.logbeta(b)
    }
}

impl Score for Beta {
    type Grad = Grad;

    fn score(&self, samples: &[f64]) -> Grad {
        const JITTER: f64 = 1e-9;

        let (a, b) = get_params!(self);

        let apb_digamma = (a + b).digamma();
        let a_digamma = a.digamma();
        let b_digamma = b.digamma();

        let [grad_a, grad_b] = samples.into_iter().fold([0.0; 2], |acc, x| {
            [
                acc[0] + x.max(JITTER).ln() - a_digamma + apb_digamma,
                acc[1] + (1.0 - x).max(JITTER).ln() - b_digamma + apb_digamma,
            ]
        });

        Grad {
            alpha: grad_a,
            beta: grad_b,
        }
    }
}

impl fmt::Display for Beta {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (a, b) = get_params!(self);

        write!(f, "Beta({}, {})", a, b)
    }
}
