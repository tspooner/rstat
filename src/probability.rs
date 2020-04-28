use crate::{
    params::{Param, constraints::{Constraints, Interval}},
    univariate::uniform::Uniform,
    Distribution,
};
use failure::Fail;
use rand::Rng;
use std::{
    fmt,
    iter::Sum,
    ops::{Add, Div, Mul, Not, Rem, Sub},
};

#[derive(Debug, Fail)]
#[fail(display = "Value {} doesn't lie in the range [0.0, 1.0].", _0)]
pub struct InvalidProbabilityError(f64);

/// Type representing the probability an event.
///
/// This struct is just a wrapper around `f64` that strictly enforces that the
/// probability lies in \\([0, 1]\\).
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Probability(pub(crate) f64);

impl Probability {
    /// Construct a probability, checking that the argument lies in \\([0,
    /// 1]\\).
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
    pub fn new(p: f64) -> Result<Self, InvalidProbabilityError> {
        if p >= 0.0 && p <= 1.0 {
            Ok(Probability(p))
        } else {
            Err(InvalidProbabilityError(p))
        }
    }

    /// Construct a probability without checking for validity.
    pub fn new_unchecked(p: f64) -> Probability { Probability(p) }

    /// Returns a new [Probability](struct.Probability.html) with value 0.
    pub fn zero() -> Probability { Probability(0.0) }

    /// Returns a new [Probability](struct.Probability.html) with value 0.5.
    pub fn half() -> Probability { Probability(0.5) }

    /// Returns a new [Probability](struct.Probability.html) with value 1.
    pub fn one() -> Probability { Probability(1.0) }

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

impl Param for Probability {
    type Value = f64;

    fn value(&self) -> &f64 { &self.0 }

    fn into_value(self) -> f64 { self.0 }

    fn constraints() -> Constraints<Self::Value> {
        vec![Box::new(Interval {
            lb: 0.0,
            ub: 1.0,
        })]
    }
}

impl std::convert::TryFrom<f64> for Probability {
    type Error = InvalidProbabilityError;

    fn try_from(p: f64) -> Result<Self, InvalidProbabilityError> { Probability::new(p) }
}

impl From<Probability> for f64 {
    fn from(p: Probability) -> f64 { p.unwrap() }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "{}", self.0) }
}

impl PartialEq<f64> for Probability {
    fn eq(&self, other: &f64) -> bool { self.0.eq(other) }
}

impl Sum<Probability> for f64 {
    fn sum<I: Iterator<Item = Probability>>(iter: I) -> f64 { iter.map(|p| p.unwrap()).sum() }
}

impl<'a> Sum<&'a Probability> for f64 {
    fn sum<I: Iterator<Item = &'a Probability>>(iter: I) -> f64 { iter.map(|p| p.unwrap()).sum() }
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
} -> Result<Probability, InvalidProbabilityError>);

impl_op!(Sub::sub(self, other) {
    Probability::new(self.0 - other.0)
} -> Result<Probability, InvalidProbabilityError>);

impl_op!(Mul::mul(self, other) {
    Probability::new_unchecked(self.0 * other.0)
} -> Probability);

impl_op!(Div::div(self, other) {
    Probability::new(self.0 / other.0)
} -> Result<Probability, InvalidProbabilityError>);

impl_op!(Rem::rem(self, other) {
    Probability::new(self.0 % other.0)
} -> Result<Probability, InvalidProbabilityError>);

impl Not for Probability {
    type Output = Probability;

    fn not(self) -> Probability { Probability::new_unchecked(1.0 - self.0) }
}

impl Not for &Probability {
    type Output = Probability;

    fn not(self) -> Probability { Probability::new_unchecked(1.0 - self.0) }
}

/// Utility for sampling from a unit \\(K\\)-simplex.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct UnitSimplex(usize);

impl UnitSimplex {
    /// Construct a \\(K = n + 1\\) probability simplex.
    pub fn new(n: usize) -> UnitSimplex { UnitSimplex(n + 1) }

    /// Compute the central point of the simplex \\(x_i = 1 / K\\).
    ///
    /// # Examples
    /// ```
    /// use rstat::UnitSimplex;
    ///
    /// let s = UnitSimplex::new(2).centre();
    ///
    /// for i in 0..3 {
    ///     assert_eq!(s[i], 1.0 / 3.0);
    /// }
    /// ```
    pub fn centre(&self) -> SimplexVector {
        let p = 1.0 / self.0 as f64;

        SimplexVector::new_unchecked(std::iter::repeat(p).take(self.0))
    }

    /// Draws a uniformly random point on the simplex.
    ///
    /// This algorithm works as follows:
    ///
    /// 1. Draw \\(K\\) independent points, \\(x_i \in [0, 1]\\), uniformly at
    /// random.
    /// 2. Apply the transformation \\(z_i = -\ln{x_i}\\).
    /// 3. Compute the sum \\(s = \sum_i x_i\\).
    /// 4. Return the vector of values \\(z_i / s\\).
    ///
    /// # Examples
    /// ```
    /// use rand::thread_rng;
    /// use rstat::UnitSimplex;
    ///
    /// let s = UnitSimplex::new(2).sample(&mut thread_rng());
    ///
    /// assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-7);
    /// ```
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SimplexVector {
        let mut sum = 0.0;

        let vs: Vec<f64> = Uniform::<f64>::new_unchecked(0.0, 1.0)
            .sample_iter(rng)
            .map(|s| -s.ln())
            .inspect(|v| sum += v)
            .take(self.0)
            .collect();

        SimplexVector(vs.into_iter().map(|v| v / sum).collect())
    }
}

#[derive(Clone, Debug, Fail)]
#[fail(display = "Probabilities {:?} do not sum to 1.", _0)]
pub struct InvalidSimplexError(Vec<f64>);

/// Probability vector constrainted to the [unit
/// simplex](struct.UnitSimplex.html).
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct SimplexVector(Vec<f64>);

impl SimplexVector {
    /// Construct a new probability vector on the [unit
    /// simplex](struct.UnitSimplex.html).
    pub fn new(ps: Vec<f64>) -> Result<SimplexVector, failure::Error> {
        std::convert::TryFrom::try_from(ps)
    }

    /// Construct a new probability vector without enforcing constraints.
    pub fn new_unchecked<I>(ps: I) -> SimplexVector
    where I: IntoIterator<Item = f64> {
        SimplexVector(ps.into_iter().collect())
    }

    /// Unwrap and return the inner `Vec<f64>` instance.
    pub fn unwrap(self) -> Vec<f64> { self.0 }

    /// Sample a probability-weighted random index from the vector.
    pub fn sample_index<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut cdf = self.0.iter().scan(0.0, |state, p| {
            *state += p;

            Some(*state)
        });
        let rval: f64 = rng.gen();

        cdf.position(|p| rval < p).unwrap_or(self.len() - 1)
    }
}

impl ::std::ops::Deref for SimplexVector {
    type Target = [f64];

    fn deref(&self) -> &[f64] { self.0.deref() }
}

impl std::convert::TryFrom<Vec<f64>> for SimplexVector {
    type Error = failure::Error;

    fn try_from(ps: Vec<f64>) -> Result<SimplexVector, failure::Error> {
        let mut z: f64 = 0.0;

        for p in ps.iter() {
            z += Probability::new(*p)?.0;
        }

        if (z - 1.0).abs() < 1e-5 {
            Ok(SimplexVector(ps))
        } else {
            Err(InvalidSimplexError(ps))?
        }
    }
}

impl Param for SimplexVector {
    type Value = Vec<f64>;

    fn value(&self) -> &Vec<f64> { &self.0 }

    fn into_value(self) -> Vec<f64> { self.0 }

    fn constraints() -> Constraints<Vec<f64>> { vec![] }
}

#[cfg(test)]
mod tests {
    use super::SimplexVector;
    use rand::thread_rng;
    use std::iter::{once, repeat};

    #[test]
    fn test_sample_index_degenerate() {
        let mut rng = thread_rng();

        let s = SimplexVector::new(vec![1.0]).unwrap();

        assert_eq!(s.sample_index(&mut rng), 0);
    }

    #[test]
    fn test_sample_index_degenerate_n() {
        let mut rng = thread_rng();

        fn make_simplex(idx: usize, n: usize) -> SimplexVector {
            SimplexVector::new_unchecked(
                repeat(0.0)
                    .take(idx)
                    .chain(once(1.0))
                    .chain(repeat(0.0).take(n - idx - 1)),
            )
        }

        assert_eq!(make_simplex(0, 5).sample_index(&mut rng), 0);
        assert_eq!(make_simplex(1, 5).sample_index(&mut rng), 1);
        assert_eq!(make_simplex(2, 5).sample_index(&mut rng), 2);
        assert_eq!(make_simplex(3, 5).sample_index(&mut rng), 3);
        assert_eq!(make_simplex(4, 5).sample_index(&mut rng), 4);
    }
}
