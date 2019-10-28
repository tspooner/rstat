use std::{error::Error, fmt, ops::{Add, Sub, Mul, Div, Not}};

pub type ProbabilityResult<T> = Result<T, ProbabilityError>;

#[derive(Clone, Copy, Debug)]
pub enum ProbabilityError {
    InvalidProbability(f64),
}

impl ProbabilityError {
    #[inline(always)]
    pub fn check_bounded(p: f64) -> ProbabilityResult<f64> {
        if p >= 0.0 && p <= 1.0 {
            Ok(p)
        } else {
            Err(ProbabilityError::InvalidProbability(p))
        }
    }
}

impl fmt::Display for ProbabilityError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            ProbabilityError::InvalidProbability(p) =>
                f.write_str(&format!("InvalidProbability({})", p)),
        }
    }
}

impl Error for ProbabilityError {
    fn description(&self) -> &str {
        match *self {
            ProbabilityError::InvalidProbability(_) =>
                "Probabilities must lie in the range [0.0, 1.0].",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(f64);

impl Probability {
    pub fn new(p: f64) -> ProbabilityResult<Self> {
        ProbabilityError::check_bounded(p).map(|p| Probability(p))
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

    pub fn powf(self, e: f64) -> Probability {
        Probability(self.0.powf(e))
    }

    pub fn powi(self, e: i32) -> Probability {
        Probability(self.0.powi(e))
    }

    pub fn normalised(probs: Vec<Probability>) -> Vec<Probability> {
        let mut z: f64 = probs.iter().fold(0.0, |acc, p| acc + p.0);

        probs.into_iter().map(|p| {
            Probability(p.0 / z)
        }).collect()
    }
}

impl Probability {
    pub fn non_zero(&self) -> bool {
        self.0 > 1e-7
    }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::convert::TryFrom<f64> for Probability {
    type Error = ProbabilityError;

    fn try_from(p: f64) -> ProbabilityResult<Self> {
        Probability::new(p)
    }
}

impl From<Probability> for f64 {
    fn from(p: Probability) -> f64 {
        p.0
    }
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

impl Not for Probability {
    type Output = Probability;

    fn not(self) -> Probability { Probability(1.0 - self.0) }
}
