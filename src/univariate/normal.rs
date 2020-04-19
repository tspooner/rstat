use crate::{
    consts::{PI_2, PI_E_2},
    fitting::MLE,
    prelude::*,
};
use ndarray::Array2;
use rand::Rng;
use spaces::real::Reals;
use std::fmt;

locscale_params! {
    Params {
        mu<f64>,
        sigma<f64>
    }
}

new_dist!(Normal<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.sigma.0)
    };
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Result<Normal, failure::Error> {
        Params::new(mu, sigma).map(Normal)
    }

    pub fn new_unchecked(mu: f64, sigma: f64) -> Normal {
        Normal(Params::new_unchecked(mu, sigma))
    }

    pub fn standard() -> Normal { Normal(Params::new_unchecked(0.0, 1.0)) }

    #[inline(always)]
    pub fn z(&self, x: f64) -> f64 {
        let (mu, sigma) = get_params!(self);

        (x - mu) / sigma
    }

    #[inline(always)]
    pub fn precision(&self) -> f64 { 1.0 / self.0.sigma.0 / self.0.sigma.0 }

    #[inline(always)]
    pub fn width(&self) -> f64 { 2.0 * self.precision() }
}

impl Distribution for Normal {
    type Support = Reals;
    type Params = Params;

    fn support(&self) -> Reals { Reals }

    fn params(&self) -> Self::Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use std::num::FpCategory;
        use special_fun::FloatSpecial;

        Probability::new_unchecked(match x.classify() {
            FpCategory::Infinite => if x.is_sign_negative() { 0.0 } else { 1.0 },
            _ => self.z(*x).norm()
        })
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let (mu, sigma) = get_params!(self);

        rand_distr::Normal::new(mu, sigma).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for Normal {
    fn pdf(&self, x: &f64) -> f64 {
        let z = self.z(*x);
        let norm = PI_2.sqrt() * self.0.sigma.0;

        (-z * z / 2.0).exp() / norm
    }
}

impl UnivariateMoments for Normal {
    fn mean(&self) -> f64 { self.0.mu.0 }

    fn variance(&self) -> f64 { self.0.sigma.0 * self.0.sigma.0 }

    fn skewness(&self) -> f64 { 0.0 }

    fn kurtosis(&self) -> f64 { 0.0 }

    fn excess_kurtosis(&self) -> f64 { -3.0 }
}

impl Quantiles for Normal {
    fn quantile(&self, p: Probability) -> f64 {
        use special_fun::FloatSpecial;

        let (mu, sigma) = get_params!(self);

        mu + sigma * p.unwrap().norm_inv()
    }

    fn median(&self) -> f64 { self.0.mu.0 }
}

impl Modes for Normal {
    fn modes(&self) -> Vec<f64> { vec![self.0.mu.0] }
}

impl ShannonEntropy for Normal {
    fn shannon_entropy(&self) -> f64 { (PI_E_2 * self.variance()).ln() / 2.0 }
}

impl FisherInformation for Normal {
    fn fisher_information(&self) -> Array2<f64> {
        let precision = self.precision();

        unsafe {
            Array2::from_shape_vec_unchecked(
                (2, 2),
                vec![precision, 0.0, 0.0, precision * precision / 2.0],
            )
        }
    }
}

impl Convolution<Normal> for Normal {
    type Output = Normal;

    fn convolve(self, rv: Normal) -> Result<Normal, failure::Error> {
        let new_mu = self.0.mu.0 + rv.0.mu.0;
        let new_var = (self.variance() + rv.variance()).sqrt();

        Ok(Normal::new_unchecked(new_mu, new_var))
    }
}

impl MLE for Normal {
    fn fit_mle(xs: &[f64]) -> Result<Self, failure::Error> {
        let n = xs.len() as f64;

        let mean = xs.iter().fold(0.0, |acc, &x| acc + x) / n;
        let var = xs
            .into_iter()
            .map(|x| x - mean)
            .fold(0.0, |acc, r| acc + r * r)
            / (n - 1.0);

        Normal::new(mean, var.sqrt())
    }
}

impl fmt::Display for Normal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "N({}, {})", self.0.mu.0, self.variance())
    }
}
