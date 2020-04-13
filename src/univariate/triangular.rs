use crate::{
    consts::{THREE_FIFTHS, THREE_HALVES, TWELVE_FIFTHS},
    params::{Loc, Scale},
    prelude::*,
};
use rand::Rng;
use spaces::real::Interval;
use std::fmt;

params! {
    Params {
        a: Loc<f64>,
        a2c: Scale<f64>,
        c2b: Scale<f64>
    }
}

impl Params {
    pub fn symmetric(a: f64, b: f64) -> Result<Params, failure::Error> {
        let d = (b - a) / 2.0;

        Params::new(a, d, d)
    }

    #[inline(always)]
    pub fn c(&self) -> f64 { self.a.0 + self.a2c.0 }

    #[inline(always)]
    pub fn b(&self) -> f64 { self.c() + self.c2b.0 }
}

new_dist!(Triangular<Params>);

macro_rules! get_params {
    ($self:ident) => { ($self.0.a.0, $self.0.b(), $self.0.c()) }
}

impl Triangular {
    pub fn new(a: f64, a2c: f64, c2b: f64) -> Result<Triangular, failure::Error> {
        Params::new(a, a2c, c2b).map(|p| Triangular(p))
    }

    pub fn new_unchecked(a: f64, a2c: f64, c2b: f64) -> Triangular {
        Triangular(Params::new_unchecked(a, a2c, c2b))
    }
}

impl Distribution for Triangular {
    type Support = Interval;
    type Params = Params;

    fn support(&self) -> Interval { Interval::bounded(self.0.a.0, self.0.b()) }

    fn params(&self) -> Params { self.0 }

    fn cdf(&self, x: &f64) -> Probability {
        let (a, b, c) = get_params!(self);

        match *x {
            x if x <= a => Probability::zero(),

            x if x <= c => Probability::new_unchecked(
                (x - a) * (x - a) / (b - a) / (c - a)
            ),

            x if x <= b => Probability::new_unchecked(
                1.0 - (b - x) * (b - x) / (b - a) / (b - c)
            ),

            _ => Probability::one(),
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        use rand_distr::Distribution as _;

        let (a, b, c) = get_params!(self);

        rand_distr::Triangular::new(a, b, c).unwrap().sample(rng)
    }
}

impl ContinuousDistribution for Triangular {
    fn pdf(&self, x: &f64) -> f64 {
        let (a, b, c) = get_params!(self);

        match *x {
            x if x <= a => 0.0,

            x if (x - c).abs() < 1e-7 => 2.0 / (b - a),

            x if x < c => 2.0 * (x - a) / (b - a) / (c - a),

            x if x <= b => 2.0 * (b - x) / (b - a) / (b - c),

            _ => 0.0,
        }
    }
}

impl UnivariateMoments for Triangular {
    fn mean(&self) -> f64 {
        let (a, b, c) = get_params!(self);

        (a + b + c) / 2.0
    }

    fn variance(&self) -> f64 {
        let (a, b, c) = get_params!(self);

        let sq_terms = a * a + b * b + c * c;
        let cross_terms = a * b + a * c + b * c;

        (sq_terms - cross_terms) / 18.0
    }

    fn skewness(&self) -> f64 {
        let (a, b, c) = get_params!(self);

        let sq_terms = a * a + b * b + c * c;
        let cross_terms = a * b + a * c + b * c;

        let numerator = 2.0f64.sqrt() * (a + b - 2.0 * c) * (2.0 * a - b - c) * (a - 2.0 * b + c);
        let denominator = 5.0 * (sq_terms - cross_terms).powf(THREE_HALVES);

        numerator / denominator
    }

    fn kurtosis(&self) -> f64 { TWELVE_FIFTHS }

    fn excess_kurtosis(&self) -> f64 { -THREE_FIFTHS }
}

impl Quantiles for Triangular {
    fn quantile(&self, _: Probability) -> f64 {
        unimplemented!()
    }

    fn median(&self) -> f64 {
        let (a, b, c) = get_params!(self);

        let midpoint = (a + b) / 2.0;

        if c >= midpoint {
            a + ((b - a) * (c - a) / 2.0).sqrt()
        } else {
            b - ((b - a) * (b - c) / 2.0).sqrt()
        }
    }
}

impl Modes for Triangular {
    fn modes(&self) -> Vec<f64> {
        vec![self.0.c()]
    }
}

impl Entropy for Triangular {
    fn entropy(&self) -> f64 {
        1.0 + ((self.0.b() - self.0.a.0) / 2.0).ln()
    }
}

impl fmt::Display for Triangular {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (a, b, c) = get_params!(self);

        write!(f, "Triangular({}, {}, {})", a, b, c)
    }
}
