use crate::prelude::*;
use rand;
use spaces::real::PositiveReals;
use std::fmt;

pub use crate::params::DOF;

new_dist!(ChiSq<DOF<usize>>);

macro_rules! get_k {
    ($self:ident) => { ($self.0).0 as f64 }
}

impl ChiSq {
    pub fn new(dof: usize) -> Result<ChiSq, failure::Error> {
        Ok(ChiSq(DOF::new(dof)?))
    }

    pub fn new_unchecked(dof: usize) -> ChiSq {
        ChiSq(DOF(dof))
    }

    #[inline(always)]
    pub fn k(&self) -> f64 { get_k!(self) }
}

impl Distribution for ChiSq {
    type Support = PositiveReals;
    type Params = DOF<usize>;

    fn support(&self) -> PositiveReals { PositiveReals }

    fn params(&self) -> DOF<usize> { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        use special_fun::FloatSpecial;

        let k = get_k!(self);
        let ko2 = k / 2.0;

        Probability::new_unchecked(ko2.gammainc(x / 2.0) / ko2.gamma())
    }

    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        rand_distr::ChiSquared::new(get_k!(self)).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for ChiSq {
    fn pdf(&self, x: &f64) -> f64 {
        use special_fun::FloatSpecial;

        let k = get_k!(self);
        let ko2 = k / 2.0;
        let norm = 2.0f64.powf(ko2) * ko2.gamma();

        x.powf(ko2 - 1.0) * (-x / 2.0).exp() / norm
    }
}

impl UnivariateMoments for ChiSq {
    fn mean(&self) -> f64 { get_k!(self) }

    fn variance(&self) -> f64 {
        (2 * (self.0).0) as f64
    }

    fn skewness(&self) -> f64 {
        (8.0 / get_k!(self)).sqrt()
    }

    fn excess_kurtosis(&self) -> f64 {
        12.0 / get_k!(self)
    }
}

impl Quantiles for ChiSq {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        let k = get_k!(self);

        k * (1.0 - 2.0 / 9.0 / k).powi(3)
    }
}

impl Modes for ChiSq {
    fn modes(&self) -> Vec<f64> {
        vec![((self.0).0.max(2) - 2) as f64]
    }
}

impl Entropy for ChiSq {
    fn entropy(&self) -> f64 {
        use special_fun::FloatSpecial;

        let k = get_k!(self);
        let ko2 = k / 2.0;

        ko2 + (2.0 * ko2.gamma()).ln() + (1.0 - ko2) * ko2.digamma()
    }
}

impl Convolution<ChiSq> for ChiSq {
    type Output = ChiSq;

    fn convolve(self, rv: ChiSq) -> Result<ChiSq, failure::Error> {
        let k1 = (self.0).0;
        let k2 = (rv.0).0;

        assert_constraint!(k1 == k2)?;

        Ok(ChiSq::new_unchecked(k1 + k2))
    }
}

impl fmt::Display for ChiSq {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "ChiSq({})", self.k())
    }
}
