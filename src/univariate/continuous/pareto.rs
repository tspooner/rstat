use crate::core::*;
use ndarray::Array2;
use rand::Rng;
use spaces::real::Interval;
use std::{f64::INFINITY, fmt};

#[derive(Debug, Clone, Copy)]
pub struct Pareto {
    pub x_m: f64,
    pub alpha: f64,
}

impl Pareto {
    pub fn new(x_m: f64, alpha: f64) -> Pareto {
        assert_positive_real!(x_m);
        assert_positive_real!(alpha);

        Pareto { x_m, alpha }
    }
}

impl Default for Pareto {
    fn default() -> Pareto {
        Pareto {
            x_m: 1.0,
            alpha: 1.0,
        }
    }
}

impl Into<rand_distr::Pareto<f64>> for Pareto {
    fn into(self) -> rand_distr::Pareto<f64> {
        rand_distr::Pareto::new(self.x_m, self.alpha).unwrap()
    }
}

impl Into<rand_distr::Pareto<f64>> for &Pareto {
    fn into(self) -> rand_distr::Pareto<f64> {
        rand_distr::Pareto::new(self.x_m, self.alpha).unwrap()
    }
}

impl Distribution for Pareto {
    type Support = Interval;

    fn support(&self) -> Interval {
        Interval::left_bounded(self.x_m)
    }

    fn cdf(&self, x: f64) -> Probability {
        (1.0 - (self.x_m / x).powf(self.alpha)).into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution;

        let sampler: rand_distr::Pareto<f64> = self.into();

        sampler.sample(rng)
    }
}

impl ContinuousDistribution for Pareto {
    fn pdf(&self, x: f64) -> f64 {
        if x < self.x_m {
            0.0
        } else {
            self.alpha * self.x_m.powf(self.alpha) / x.powf(self.alpha + 1.0)
        }
    }
}

impl UnivariateMoments for Pareto {
    fn mean(&self) -> f64 {
        if self.alpha <= 1.0 {
            INFINITY
        } else {
            self.alpha * self.x_m / (self.alpha - 1.0)
        }
    }

    fn variance(&self) -> f64 {
        if self.alpha <= 2.0 {
            INFINITY
        } else {
            let am1 = self.alpha - 1.0;

            self.x_m * self.x_m * self.alpha / am1 / am1 / (self.alpha - 2.0)
        }
    }

    fn skewness(&self) -> f64 {
        if self.alpha <= 3.0 {
            unimplemented!("Variance is undefined for alpha <= 3.")
        }

        2.0 * (1.0 + self.alpha) / (self.alpha - 3.0) * ((self.alpha - 2.0) / self.alpha).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        if self.alpha <= 4.0 {
            unimplemented!("Kurtosis is undefined for alpha <= 4.")
        }

        let a2 = self.alpha * self.alpha;
        let a3 = a2 * self.alpha;

        6.0 * (a3 + a2 - 6.0 * self.alpha - 2.0)
            / self.alpha
            / (self.alpha - 3.0)
            / (self.alpha - 4.0)
    }
}

impl Quantiles for Pareto {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        self.x_m * 2.0f64.powf(1.0 / self.alpha)
    }
}

impl Modes for Pareto {
    fn modes(&self) -> Vec<f64> {
        vec![self.x_m]
    }
}

impl Entropy for Pareto {
    fn entropy(&self) -> f64 {
        (self.x_m / self.alpha * (1.0 + 1.0 / self.alpha).exp()).ln()
    }
}

impl FisherInformation for Pareto {
    fn fisher_information(&self) -> Array2<f64> {
        let off_diag = -1.0 / self.x_m;

        unsafe {
            Array2::from_shape_vec_unchecked(
                (2, 2),
                vec![
                    self.alpha / self.x_m / self.x_m,
                    off_diag,
                    off_diag,
                    1.0 / self.alpha / self.alpha,
                ],
            )
        }
    }
}

impl fmt::Display for Pareto {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Pareto({}, {})", self.x_m, self.alpha)
    }
}
