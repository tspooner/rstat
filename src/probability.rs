use failure::Fail;
use std::{fmt, iter::Sum, ops::{Add, Sub, Mul, Div, Rem, Not}};

// mod simplex;
// pub use self::simplex::{UnitSimplex, SimplexError, SimplexVector};

#[derive(Debug, Fail)]
pub enum ProbabilityError {
    #[fail(display="Value {} doesn't lie in the range [0.0, 1.0].", _0)]
    InvalidProbability(f64),
}

/// Type representing the probability an event.
///
/// This struct is just a wrapper around `f64` that strictly enforces that the probability lies in
/// \\([0, 1]\\).
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(pub(crate) f64);

impl Probability {
    /// Construct a probability, checking that the argument lies in \\([0, 1]\\).
    ///
    /// # Examples
    /// ```
    /// use rstat::Probability;
    ///
    /// // Invalid probabilities:
    /// assert!(Probability::new(1.5).is_err());
    /// assert!(Probability::new(-1.0).is_err());
    ///
    /// // Valid probabilities:
    /// assert!(Probability::new(0.0).is_ok());
    /// assert!(Probability::new(0.5).is_ok());
    /// assert!(Probability::new(1.0).is_ok());
    /// ```
    pub fn new(p: f64) -> Result<Self, ProbabilityError> {
        if p >= 0.0 && p <= 1.0 {
            Ok(Probability(p))
        } else {
            Err(ProbabilityError::InvalidProbability(p))
        }
    }

    /// Construct a probability without checking for validity.
    pub fn new_unchecked(p: f64) -> Probability {
        Probability(p)
    }

    /// Returns a new [Probability](struct.Probability.html) with value 0.
    pub fn zero() -> Probability {
        Probability(0.0)
    }

    /// Returns a new [Probability](struct.Probability.html) with value 0.5.
    pub fn half() -> Probability {
        Probability(0.5)
    }

    /// Returns a new [Probability](struct.Probability.html) with value 1.
    pub fn one() -> Probability {
        Probability(1.0)
    }

    /// Unwrap the probability and return the internal `f64`.
    ///
    /// # Examples
    /// ```
    /// use rstat::Probability;
    ///
    /// assert_eq!(Probability::half().unwrap(), 0.5);
    /// ```
    pub fn unwrap(self) -> f64 { self.0 }

    /// Returns true if the probability lies in \\([0, 1]\\).
    ///
    /// # Examples
    /// ```
    /// use rstat::Probability;
    ///
    /// assert!(Probability::new_unchecked(0.5).is_valid());
    /// assert!(!Probability::new_unchecked(10.0).is_valid());
    /// ```
    pub fn is_valid(&self) -> bool { self.0 <= 1.0 && self.0 >= 0.0 }

    /// Return the natural logarithm of the probability: \\(\ln{p}\\).
    ///
    /// # Examples
    /// ```
    /// use rstat::Probability;
    ///
    /// assert_eq!(Probability::one().ln(), 0.0);
    /// ```
    pub fn ln(self) -> f64 { self.0.ln() }

    /// Return the base-2 logarithm of the probability: \\(\log_2{p}\\).
    ///
    /// # Examples
    /// ```
    /// use rstat::Probability;
    ///
    /// assert_eq!(Probability::half().log2(), -1.0);
    /// ```
    pub fn log2(self) -> f64 { self.0.log2() }

    /// Return the probability raised to the power `e`: \\(p^e\\).
    ///
    /// # Examples
    /// ```
    /// use rstat::Probability;
    ///
    /// assert_eq!(Probability::half().powf(2.5), 0.1767766952966369);
    /// ```
    pub fn powf(self, e: f64) -> f64 { self.0.powf(e) }

    /// Return the probability raised to the power `e`: \\(p^e\\).
    ///
    /// # Examples
    /// ```
    /// use rstat::Probability;
    ///
    /// assert_eq!(Probability::half().powi(2), 0.25);
    /// ```
    pub fn powi(self, e: i32) -> f64 { self.0.powi(e) }
}

impl crate::params::Param for Probability {
    type Value = f64;

    fn value(&self) -> &f64 { &self.0 }

    fn constraints() -> crate::params::constraints::Constraints<Self::Value> {
        vec![Box::new(crate::params::constraints::Interval { lb: 0.0, ub: 1.0, })]
    }
}

impl std::convert::TryFrom<f64> for Probability {
    type Error = ProbabilityError;

    fn try_from(p: f64) -> Result<Self, ProbabilityError> {
        Probability::new(p)
    }
}

impl From<Probability> for f64 {
    fn from(p: Probability) -> f64 { p.unwrap() }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl PartialEq<f64> for Probability {
    fn eq(&self, other: &f64) -> bool {
        self.0.eq(other)
    }
}

impl Sum<Probability> for f64 {
    fn sum<I: Iterator<Item = Probability>>(iter: I) -> f64 {
        iter.map(|p| p.unwrap()).sum()
    }
}

impl<'a> Sum<&'a Probability> for f64 {
    fn sum<I: Iterator<Item = &'a Probability>>(iter: I) -> f64 {
        iter.map(|p| p.unwrap()).sum()
    }
}

macro_rules! impl_op {
    ($trait:tt::$op:ident($self:ident, $other:ident) $code:block -> $out:ty) => (
        // Prob | Prob
        impl $trait<Probability> for Probability {
            type Output = $out;

            #[inline]
            fn $op($self, $other: Probability) -> $out { $code }
        }

        impl<'a> $trait<&'a Probability> for Probability {
            type Output = $out;

            #[inline]
            fn $op($self, $other: &'a Probability) -> $out { $code }
        }

        impl<'a> $trait<Probability> for &'a Probability {
            type Output = $out;

            #[inline]
            fn $op($self, $other: Probability) -> $out { $code }
        }

        impl<'a, 'b> $trait<&'a Probability> for &'b Probability {
            type Output = $out;

            #[inline]
            fn $op($self, $other: &'a Probability) -> $out { $code }
        }

        // Prob | f64
        impl $trait<Probability> for f64 {
            type Output = f64;

            #[inline]
            fn $op($self, $other: Probability) -> f64 {
                $self.$op($other.0)
            }
        }

        impl<'a> $trait<&'a Probability> for f64 {
            type Output = f64;

            #[inline]
            fn $op($self, $other: &'a Probability) -> f64 {
                $self.$op($other.0)
            }
        }

        impl<'a> $trait<Probability> for &'a f64 {
            type Output = f64;

            #[inline]
            fn $op($self, $other: Probability) -> f64 {
                $self.$op($other.0)
            }
        }

        impl<'a, 'b> $trait<&'a Probability> for &'b f64 {
            type Output = f64;

            #[inline]
            fn $op($self, $other: &'a Probability) -> f64 {
                $self.$op($other.0)
            }
        }

        // f64 | Prob
        impl $trait<f64> for Probability {
            type Output = f64;

            #[inline]
            fn $op($self, $other: f64) -> f64 {
                ($self.0).$op($other)
            }
        }

        impl<'a> $trait<&'a f64> for Probability {
            type Output = f64;

            #[inline]
            fn $op($self, $other: &'a f64) -> f64 {
                ($self.0).$op($other)
            }
        }

        impl<'a> $trait<f64> for &'a Probability {
            type Output = f64;

            #[inline]
            fn $op($self, $other: f64) -> f64 {
                ($self.0).$op($other)
            }
        }

        impl<'a, 'b> $trait<&'a f64> for &'b Probability {
            type Output = f64;

            #[inline]
            fn $op($self, $other: &'a f64) -> f64 {
                ($self.0).$op($other)
            }
        }
    )
}

impl_op!(Add::add(self, other) {
    Probability::new(self.0 + other.0)
} -> Result<Probability, ProbabilityError>);

impl_op!(Sub::sub(self, other) {
    Probability::new(self.0 - other.0)
} -> Result<Probability, ProbabilityError>);

impl_op!(Mul::mul(self, other) {
    Probability::new_unchecked(self.0 * other.0)
} -> Probability);

impl_op!(Div::div(self, other) {
    Probability::new(self.0 / other.0)
} -> Result<Probability, ProbabilityError>);

impl_op!(Rem::rem(self, other) {
    Probability::new(self.0 % other.0)
} -> Result<Probability, ProbabilityError>);

impl Not for Probability {
    type Output = Probability;

    fn not(self) -> Probability {
        Probability::new_unchecked(1.0 - self.0)
    }
}

impl Not for &Probability {
    type Output = Probability;

    fn not(self) -> Probability {
        Probability::new_unchecked(1.0 - self.0)
    }
}
