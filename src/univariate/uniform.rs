use crate::{
    consts::{NINE_FIFTHS, SIX_FIFTHS},
    params::{
        constraints::{All, Constraint, Positive, UnsatisfiedConstraintError},
        Param,
    },
    statistics::{Quantiles, ShannonEntropy, UnivariateMoments},
    ContinuousDistribution,
    DiscreteDistribution,
    Distribution,
    Probability,
};
use rand::Rng;
use spaces::{discrete::Interval as DiscreteInterval, real::Interval as RealInterval};
use std::fmt;

pub use crate::params::{Loc, Scale};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Params<T> {
    pub lb: Loc<T>,
    pub width: Scale<T>,
}

impl<T> Params<T>
where All<Positive>: Constraint<T>
{
    pub fn new(lb: T, width: T) -> Result<Params<T>, UnsatisfiedConstraintError<T>>
    where T: fmt::Debug {
        Ok(Params {
            lb: Loc::new(lb)?,
            width: Scale::new(width)?,
        })
    }
}

impl<T> Params<T> {
    pub fn new_unchecked(lb: T, width: T) -> Params<T> {
        Params {
            lb: Loc(lb),
            width: Scale(width),
        }
    }

    #[inline(always)]
    pub fn lb(&self) -> &Loc<T> { &self.lb }

    #[inline(always)]
    pub fn width(&self) -> &Scale<T> { &self.width }
}

macro_rules! get_params {
    ($self:ident) => {
        ($self.params.lb.0, $self.params.lb.0 + $self.params.width.0)
    };
}

#[derive(Debug, Clone, Copy)]
pub struct Uniform<T>
where
    Loc<T>: Param,
    Scale<T>: Param,
{
    params: Params<T>,
    prob: f64,
}

// Continuous:
impl Uniform<f64> {
    pub fn new(lb: f64, width: f64) -> Result<Uniform<f64>, UnsatisfiedConstraintError<f64>> {
        Params::new(lb, width).map(|p| Uniform {
            prob: 1.0 / p.width.0,
            params: p,
        })
    }

    pub fn new_unchecked(lb: f64, width: f64) -> Uniform<f64> {
        Params::new_unchecked(lb, width).into()
    }
}

impl From<Params<f64>> for Uniform<f64> {
    fn from(params: Params<f64>) -> Uniform<f64> {
        Uniform {
            prob: 1.0 / params.width.0,
            params,
        }
    }
}

impl Default for Uniform<f64> {
    fn default() -> Uniform<f64> {
        Uniform {
            params: Params::new_unchecked(0.0, 1.0),
            prob: 1.0,
        }
    }
}

impl Distribution for Uniform<f64> {
    type Support = RealInterval;
    type Params = Params<f64>;

    fn support(&self) -> RealInterval {
        let (lb, ub) = get_params!(self);

        RealInterval::bounded(lb, ub)
    }

    fn params(&self) -> Params<f64> { self.params }

    fn cdf(&self, x: &f64) -> Probability {
        let x = *x;
        let (lb, ub) = get_params!(self);

        if x < lb {
            Probability::zero()
        } else if x >= ub {
            Probability::one()
        } else {
            Probability::new_unchecked((x - lb) * self.prob)
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let (lb, ub) = get_params!(self);

        rand_distr::Uniform::new_inclusive(lb, ub).sample(rng)
    }
}

impl ContinuousDistribution for Uniform<f64> {
    fn pdf(&self, x: &f64) -> f64 {
        let x = *x;
        let (lb, ub) = get_params!(self);

        if x < lb || x > ub {
            0.0
        } else {
            self.prob
        }
    }
}

impl UnivariateMoments for Uniform<f64> {
    fn mean(&self) -> f64 { self.params.lb.0 + self.params.width.0 / 2.0 }

    fn variance(&self) -> f64 {
        let width = self.params.width.0;

        width * width / 12.0
    }

    fn skewness(&self) -> f64 { 0.0 }

    fn kurtosis(&self) -> f64 { NINE_FIFTHS }

    fn excess_kurtosis(&self) -> f64 { -SIX_FIFTHS }
}

impl Quantiles for Uniform<f64> {
    fn quantile(&self, p: Probability) -> f64 { self.params.lb.0 + p * self.params.width.0 }

    fn median(&self) -> f64 { self.params.lb.0 + self.params.width.0 / 2.0 }
}

impl ShannonEntropy for Uniform<f64> {
    fn shannon_entropy(&self) -> f64 { self.params.width.0.ln() }
}

impl fmt::Display for Uniform<f64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (lb, ub) = get_params!(self);

        write!(f, "U({}, {})", lb, ub)
    }
}

// Discrete:
impl Uniform<i64> {
    pub fn new(lb: i64, width: u64) -> Result<Uniform<i64>, UnsatisfiedConstraintError<i64>> {
        Params::new(lb, width as i64).map(|p| Uniform {
            prob: 1.0 / (p.width.0 + 1) as f64,
            params: p,
        })
    }

    pub fn new_unchecked(lb: i64, width: u64) -> Uniform<i64> {
        Params::new_unchecked(lb, width as i64).into()
    }
}

impl From<Params<i64>> for Uniform<i64> {
    fn from(params: Params<i64>) -> Uniform<i64> {
        Uniform {
            prob: 1.0 / (params.width.0 + 1) as f64,
            params,
        }
    }
}

impl Uniform<i64> {
    #[inline]
    pub fn span(&self) -> u64 { (self.params.width.0 + 1) as u64 }
}

impl Distribution for Uniform<i64> {
    type Support = DiscreteInterval;
    type Params = Params<i64>;

    fn support(&self) -> DiscreteInterval {
        let (lb, ub) = get_params!(self);

        DiscreteInterval::bounded(lb, ub)
    }

    fn params(&self) -> Params<i64> { self.params }

    fn cdf(&self, k: &i64) -> Probability {
        let k = *k;
        let (lb, ub) = get_params!(self);

        if k < lb {
            Probability::zero()
        } else if k >= ub {
            Probability::one()
        } else {
            Probability::new_unchecked((k - lb + 1) as f64 * self.prob)
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> i64 {
        use rand_distr::Distribution as _;

        let (lb, ub) = get_params!(self);

        rand_distr::Uniform::new_inclusive(lb, ub).sample(rng)
    }
}

impl DiscreteDistribution for Uniform<i64> {
    fn pmf(&self, x: &i64) -> Probability {
        let x = *x;
        let (lb, ub) = get_params!(self);

        if x < lb || x > ub {
            Probability::zero()
        } else {
            Probability::new_unchecked(self.prob)
        }
    }
}

impl UnivariateMoments for Uniform<i64> {
    fn mean(&self) -> f64 { self.params.lb.0 as f64 + self.params.width.0 as f64 / 2.0 }

    fn variance(&self) -> f64 {
        let n = self.span() as f64;

        (n * n - 1.0) / 12.0
    }

    fn skewness(&self) -> f64 { 0.0 }

    fn excess_kurtosis(&self) -> f64 {
        let n = self.span() as f64;
        let n2 = n * n;

        -SIX_FIFTHS * (n2 + 1.0) / (n2 - 1.0)
    }
}

impl Quantiles for Uniform<i64> {
    fn quantile(&self, p: Probability) -> f64 {
        let n = self.span() as f64;

        self.params.lb.0 as f64 + (p * n).floor()
    }

    fn median(&self) -> f64 { self.params.lb.0 as f64 + self.params.width.0 as f64 / 2.0 }
}

impl ShannonEntropy for Uniform<i64> {
    fn shannon_entropy(&self) -> f64 { ((self.params.width.0 + 1) as f64).ln() }
}

impl fmt::Display for Uniform<i64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (lb, ub) = get_params!(self);

        write!(f, "U{{{}, {}}}", lb, ub)
    }
}
