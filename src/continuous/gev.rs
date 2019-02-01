use consts::{ONE_THIRD, PI2_OVER_6, PI3, TWELVE_FIFTHS};
use core::*;
use rand::Rng;
use spaces::continuous::Interval;
use special_fun::FloatSpecial;
use std::{f64::INFINITY, fmt};

#[derive(Debug, Clone, Copy)]
pub struct GeneralisedExtremeValue {
    pub mu: f64,
    pub sigma: f64,
    pub zeta: f64,
}

impl GeneralisedExtremeValue {
    pub fn new(mu: f64, sigma: f64, zeta: f64) -> GeneralisedExtremeValue {
        assert_positive_real!(sigma);

        GeneralisedExtremeValue { mu, sigma, zeta }
    }

    fn t_func(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;

        if (self.zeta - 0.0) < 1e-7 {
            (-z).exp()
        } else {
            (1.0 + self.zeta * z).powf(-1.0 / self.zeta)
        }
    }

    #[inline]
    fn g_func(&self, k: f64) -> f64 {
        (1.0 - k * self.zeta).gamma()
    }
}

impl Distribution for GeneralisedExtremeValue {
    type Support = Interval;

    fn support(&self) -> Interval {
        use std::cmp::Ordering::*;

        match self
            .zeta
            .partial_cmp(&0.0)
            .expect("Invalid value provided for `zeta`.")
        {
            Less => Interval::right_bounded(self.mu - self.sigma / self.zeta),
            Equal => Interval::unbounded(),
            Greater => Interval::left_bounded(self.mu - self.sigma / self.zeta),
        }
    }

    fn cdf(&self, x: f64) -> Probability {
        (-self.t_func(x)).exp().into()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 {
        unimplemented!()
    }
}

impl ContinuousDistribution for GeneralisedExtremeValue {
    fn pdf(&self, x: f64) -> Probability {
        (self.t_func(x).powf(self.zeta + 1.0) * (-self.t_func(x)).exp() / self.sigma).into()
    }
}

impl UnivariateMoments for GeneralisedExtremeValue {
    fn mean(&self) -> f64 {
        if self.zeta >= 1.0 {
            INFINITY
        } else if self.zeta.abs() < 1e-7 {
            self.mu - self.sigma * 1.0f64.digamma()
        } else {
            self.mu + self.sigma * (self.g_func(1.0) - 1.0) / self.zeta
        }
    }

    fn variance(&self) -> f64 {
        if self.zeta >= 0.5 {
            INFINITY
        } else if self.zeta.abs() < 1e-7 {
            self.sigma * self.sigma * PI2_OVER_6
        } else {
            let g1 = self.g_func(1.0);
            let g2 = self.g_func(2.0);

            self.sigma * self.sigma * (g2 - g1 * g1) / self.zeta / self.zeta
        }
    }

    fn skewness(&self) -> f64 {
        if self.zeta >= ONE_THIRD {
            INFINITY
        } else if self.zeta.abs() < 1e-7 {
            12.0 * 6.0f64.sqrt() * 3.0f64.riemann_zeta() / PI3
        } else {
            let g1 = self.g_func(1.0);
            let g2 = self.g_func(2.0);
            let g3 = self.g_func(3.0);

            let numerator = g3 - 3.0 * g2 * g1 + 2.0 * g1.powi(3);
            let denominator = (g2 - g1 * g1).powf(3.0 / 2.0);

            self.zeta.signum() * numerator / denominator
        }
    }

    fn excess_kurtosis(&self) -> f64 {
        if self.zeta >= 0.25 {
            INFINITY
        } else if self.zeta.abs() < 1e-7 {
            TWELVE_FIFTHS
        } else {
            let g1 = self.g_func(1.0);
            let g2 = self.g_func(2.0);
            let g3 = self.g_func(3.0);
            let g4 = self.g_func(4.0);

            let numerator =
                g4 - 4.0 * g3 * g1 - 3.0 * g2 * g2 + 12.0 * g2 * g1 * g1 - 6.0 * g1.powi(4);
            let denominator = (g2 - g1 * g1).powi(2);

            numerator / denominator
        }
    }
}

impl Quantiles for GeneralisedExtremeValue {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        if self.zeta.abs() < 1e-7 {
            self.mu - self.sigma * 2.0f64.ln().ln()
        } else {
            self.mu + self.sigma * (2.0f64.ln().powf(-self.zeta) - 1.0) / self.zeta
        }
    }
}

impl Modes for GeneralisedExtremeValue {
    fn modes(&self) -> Vec<f64> {
        vec![if self.zeta.abs() < 1e-7 {
            self.mu
        } else {
            self.mu + self.sigma * ((1.0 + self.zeta).powf(-self.zeta) - 1.0) / self.zeta
        }]
    }
}

impl Entropy for GeneralisedExtremeValue {
    fn entropy(&self) -> f64 {
        let euler = -1.0f64.digamma();

        self.sigma.ln() + euler * self.zeta + euler + 1.0
    }
}

impl fmt::Display for GeneralisedExtremeValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GEV({}, {}, {})", self.mu, self.sigma, self.zeta)
    }
}
