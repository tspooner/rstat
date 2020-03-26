use failure::Fail;
use std::{
    fmt,
    ops::{Add, Sub, Mul, Div, Rem, Not},
    result::Result as _Result,
};

pub type Result<T> = _Result<T, ProbabilityError>;

#[derive(Debug, Fail)]
pub enum ProbabilityError {
    #[fail(display="Value {} doesn't lie in the range [0.0, 1.0].", _0)]
    InvalidProbability(f64),

    #[fail(display="{}", _0)]
    NumParseError(<f64 as num::Num>::FromStrRadixErr),
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(f64);

impl Probability {
    pub fn new(p: f64) -> Result<Self> {
        if p >= 0.0 && p <= 1.0 {
            Ok(Probability(p))
        } else {
            Err(ProbabilityError::InvalidProbability(p))
        }
    }

    pub fn new_unchecked(p: f64) -> Probability {
        Probability(p)
    }

    pub fn zero() -> Probability {
        Probability(0.0)
    }

    pub fn half() -> Probability {
        Probability(0.5)
    }

    pub fn one() -> Probability {
        Probability(1.0)
    }

    pub fn normalised(probs: Vec<Probability>) -> Vec<Probability> {
        let z: f64 = probs.iter().fold(0.0, |acc, p| acc + p.0);

        probs.into_iter().map(|p| {
            Probability(p.0 / z)
        }).collect()
    }

    pub fn unwrap(self) -> f64 { self.0 }

    pub fn ln(self) -> f64 { self.0.ln() }

    pub fn log2(self) -> f64 { self.0.log2() }

    pub fn powf(self, e: f64) -> f64 { self.0.powf(e) }

    pub fn powi(self, e: i32) -> f64 { self.0.powi(e) }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::convert::TryFrom<f64> for Probability {
    type Error = ProbabilityError;

    fn try_from(p: f64) -> Result<Self> {
        Probability::new(p)
    }
}

impl From<Probability> for f64 {
    fn from(p: Probability) -> f64 { p.unwrap() }
}

impl num::Num for Probability {
    type FromStrRadixErr = <f64 as num::Num>::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> _Result<Probability, Self::FromStrRadixErr> {
        f64::from_str_radix(str, radix).map(Probability)
    }
}

impl num::Zero for Probability {
    fn zero() -> Probability { Probability(0.0) }

    fn is_zero(&self) -> bool { self.0.is_zero() }
}

impl num::One for Probability {
    fn one() -> Probability { Probability(1.0) }
}

impl Add for Probability {
    type Output = Probability;

    fn add(self, other: Probability) -> Probability {
        Probability((self.0 + other.0).min(1.0))
    }
}

impl Sub for Probability {
    type Output = Probability;

    fn sub(self, other: Probability) -> Probability {
        Probability((self.0 - other.0).max(0.0))
    }
}

impl Mul for Probability {
    type Output = Probability;

    fn mul(self, other: Probability) -> Probability {
        Probability((self.0 * other.0).max(0.0).min(1.0))
    }
}

impl Div for Probability {
    type Output = Probability;

    fn div(self, other: Probability) -> Probability {
        Probability((self.0 / other.0).min(1.0).max(0.0))
    }
}

impl Rem for Probability {
    type Output = Probability;

    fn rem(self, other: Probability) -> Probability {
        Probability(self.0.rem(other.0))
    }
}

impl Not for Probability {
    type Output = Probability;

    fn not(self) -> Probability { Probability(1.0 - self.0) }
}

impl Add<f64> for Probability {
    type Output = f64;

    fn add(self, other: f64) -> f64 {
        self.0 + other
    }
}

impl Sub<f64> for Probability {
    type Output = f64;

    fn sub(self, other: f64) -> f64 {
        self.0 - other
    }
}

impl Mul<f64> for Probability {
    type Output = f64;

    fn mul(self, other: f64) -> f64 {
        self.0 * other
    }
}

impl Div<f64> for Probability {
    type Output = f64;

    fn div(self, other: f64) -> f64 {
        self.0 / other
    }
}

impl Rem<f64> for Probability {
    type Output = f64;

    fn rem(self, other: f64) -> f64 {
        self.0.rem(other)
    }
}
