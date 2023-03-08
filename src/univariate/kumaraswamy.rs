use crate::{
    statistics::{Modes, Quantiles, ShannonEntropy, UvMoments},
    ContinuousDistribution,
    Distribution,
    Probability,
    Univariate,
};
use rand;
use spaces::intervals::Closed;
use std::fmt;

#[inline]
fn harmonic_n(n: usize) -> f64 { (1..=n).map(|i| 1.0 / i as f64).sum() }

shape_params! {
    #[derive(Copy)]
    Params<f64> { a, b }
}

new_dist!(Kumaraswamy<Params>);

macro_rules! get_params {
    ($self:ident) => {
        ($self.0.a.0, $self.0.b.0)
    };
}

impl Kumaraswamy {
    fn moment_n(&self, n: usize) -> f64 {
        use special_fun::FloatSpecial;

        let (a, b) = get_params!(self);

        b * (1.0 + n as f64 / a).beta(b)
    }
}

impl Default for Kumaraswamy {
    fn default() -> Kumaraswamy { Kumaraswamy(Params::new_unchecked(1.0, 1.0)) }
}

impl Distribution for Kumaraswamy {
    type Support = Closed<f64>;
    type Params = Params;

    fn support(&self) -> Closed<f64> { Closed::unit() }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let (a, b) = get_params!(self);

        Probability::new_unchecked(1.0 - (1.0 - x.powf(a)).powf(b))
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let x = crate::univariate::uniform::Uniform::<f64>::default().sample(rng);
        let (a, b) = get_params!(self);

        (1.0f64 - (1.0 - x).powf(1.0 / b)).powf(1.0 / a)
    }
}

impl ContinuousDistribution for Kumaraswamy {
    fn pdf(&self, x: &f64) -> f64 {
        let (a, b) = get_params!(self);

        a * b * x.powf(a - 1.0) * (1.0 - x.powf(a)).powf(b - 1.0)
    }
}

impl Univariate for Kumaraswamy {}

impl UvMoments for Kumaraswamy {
    fn mean(&self) -> f64 { self.moment_n(1) }

    fn variance(&self) -> f64 {
        let m1 = self.moment_n(1);

        self.moment_n(2) - m1 * m1
    }

    fn skewness(&self) -> f64 {
        let m3 = self.moment_n(3);
        let var = self.variance();
        let mean = self.mean();

        (m3 - 3.0 * mean * var - mean.powi(3)) / var.powf(3.0 / 2.0)
    }

    fn excess_kurtosis(&self) -> f64 { unimplemented!() }
}

impl Quantiles for Kumaraswamy {
    fn quantile(&self, _: Probability) -> f64 { unimplemented!() }

    fn median(&self) -> f64 {
        let (a, b) = get_params!(self);

        (1.0 - 2.0f64.powf(-1.0 / b)).powf(1.0 / a)
    }
}

impl Modes for Kumaraswamy {
    fn modes(&self) -> Vec<f64> {
        let (a, b) = get_params!(self);

        if (a > 1.0 && b >= 1.0) || (a >= 1.0 && b > 1.0) {
            vec![((a - 1.0) / (a * b - 1.0)).powf(1.0 / a)]
        } else {
            vec![]
        }
    }
}

impl ShannonEntropy for Kumaraswamy {
    fn shannon_entropy(&self) -> f64 {
        let (a, b) = get_params!(self);
        let hb = harmonic_n(b.floor() as usize);

        (1.0 - 1.0 / b) + (1.0 - 1.0 / a) * hb - (a * b).ln()
    }
}

impl fmt::Display for Kumaraswamy {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (a, b) = get_params!(self);

        write!(f, "Kumaraswamy({}, {})", a, b)
    }
}
