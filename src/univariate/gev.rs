use crate::{
    consts::{ONE_THIRD, PI2_OVER_6, PI3, TWELVE_FIFTHS},
    statistics::{Modes, Quantiles, ShannonEntropy, UnivariateMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
};
use rand::Rng;
use spaces::real::Interval;
use special_fun::FloatSpecial;
use std::{f64::INFINITY, fmt};

pub use crate::params::{Loc, Shape};

params! {
    #[derive(Copy)]
    Params {
        mu: Loc<f64>,
        sigma: Shape<f64>,
        zeta: Shape<f64>
    }
}

new_dist!(GeneralisedExtremeValue<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.mu.0, $self.0.sigma.0, $self.0.zeta.0)
    };
}

impl GeneralisedExtremeValue {
    pub fn new(mu: f64, sigma: f64, zeta: f64) -> Result<GeneralisedExtremeValue, failure::Error> {
        Params::new(mu, sigma, zeta).map(GeneralisedExtremeValue)
    }

    pub fn new_unchecked(mu: f64, sigma: f64, zeta: f64) -> GeneralisedExtremeValue {
        GeneralisedExtremeValue(Params::new_unchecked(mu, sigma, zeta))
    }
}

impl GeneralisedExtremeValue {
    #[inline]
    fn g_func(&self, k: f64) -> f64 { (1.0 - k * self.0.zeta.0).gamma() }

    fn t_func(&self, x: f64) -> f64 {
        let (mu, sigma, zeta) = get_params!(self);

        let z = (x - mu) / sigma;

        if (zeta - 0.0) < 1e-7 {
            (-z).exp()
        } else {
            (1.0 + zeta * z).powf(-1.0 / zeta)
        }
    }
}

impl Distribution for GeneralisedExtremeValue {
    type Support = Interval;
    type Params = Params;

    fn support(&self) -> Interval {
        use std::cmp::Ordering::*;

        let (mu, sigma, zeta) = get_params!(self);

        match zeta
            .partial_cmp(&0.0)
            .expect("Invalid value provided for `zeta`.")
        {
            Less => Interval::right_bounded(mu - sigma / zeta),
            Equal => Interval::unbounded(),
            Greater => Interval::left_bounded(mu - sigma / zeta),
        }
    }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability { Probability::new_unchecked((-self.t_func(*x)).exp()) }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> f64 { unimplemented!() }
}

impl ContinuousDistribution for GeneralisedExtremeValue {
    fn pdf(&self, x: &f64) -> f64 {
        let tx = self.t_func(*x);

        tx.powf(self.0.zeta.0 + 1.0) * (-tx).exp() / self.0.sigma.0
    }
}

impl UnivariateMoments for GeneralisedExtremeValue {
    fn mean(&self) -> f64 {
        let (mu, sigma, zeta) = get_params!(self);

        if zeta >= 1.0 {
            INFINITY
        } else if zeta.abs() < 1e-7 {
            mu - sigma * 1.0f64.digamma()
        } else {
            mu + sigma * (self.g_func(1.0) - 1.0) / zeta
        }
    }

    fn variance(&self) -> f64 {
        let sigma = self.0.sigma.0;
        let zeta = self.0.zeta.0;

        if zeta >= 0.5 {
            INFINITY
        } else if zeta.abs() < 1e-7 {
            sigma * sigma * PI2_OVER_6
        } else {
            let g1 = self.g_func(1.0);
            let g2 = self.g_func(2.0);

            sigma * sigma * (g2 - g1 * g1) / zeta / zeta
        }
    }

    fn skewness(&self) -> f64 {
        let zeta = self.0.zeta.0;

        if zeta >= ONE_THIRD {
            INFINITY
        } else if zeta.abs() < 1e-7 {
            12.0 * 6.0f64.sqrt() * 3.0f64.riemann_zeta() / PI3
        } else {
            let g1 = self.g_func(1.0);
            let g2 = self.g_func(2.0);
            let g3 = self.g_func(3.0);

            let numerator = g3 - 3.0 * g2 * g1 + 2.0 * g1.powi(3);
            let denominator = (g2 - g1 * g1).powf(3.0 / 2.0);

            zeta.signum() * numerator / denominator
        }
    }

    fn excess_kurtosis(&self) -> f64 {
        let zeta = self.0.zeta.0;

        if zeta >= 0.25 {
            INFINITY
        } else if zeta.abs() < 1e-7 {
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
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        let (mu, sigma, zeta) = get_params!(self);

        if zeta.abs() < 1e-7 {
            mu - sigma * 2.0f64.ln().ln()
        } else {
            mu + sigma * (2.0f64.ln().powf(-zeta) - 1.0) / zeta
        }
    }
}

impl Modes for GeneralisedExtremeValue {
    fn modes(&self) -> Vec<f64> {
        let (mu, sigma, zeta) = get_params!(self);

        vec![if zeta.abs() < 1e-7 {
            mu
        } else {
            mu + sigma * ((1.0 + zeta).powf(-zeta) - 1.0) / zeta
        }]
    }
}

impl ShannonEntropy for GeneralisedExtremeValue {
    fn shannon_entropy(&self) -> f64 {
        let euler = -1.0f64.digamma();

        self.0.sigma.0.ln() + euler * self.0.zeta.0 + euler + 1.0
    }
}

impl fmt::Display for GeneralisedExtremeValue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (mu, sigma, zeta) = get_params!(self);

        write!(f, "GEV({}, {}, {})", mu, sigma, zeta)
    }
}
