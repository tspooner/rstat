use crate::prelude::*;
use failure::{Backtrace, Fail};
use num::{zero, PrimInt, Zero};
use std::{
    fmt::{self, Debug, Display},
    ops::Deref,
};

#[macro_export]
macro_rules! assert_constraint {
    ($x:ident+) => {
        $crate::params::constraints::Constraint::check($crate::params::constraints::Positive, $x)
    };
    ($x:ident == $t:tt) => {
        $crate::params::constraints::Constraint::check($crate::params::constraints::Equal($t), $x)
    };
    ($x:ident < $t:tt) => {
        $crate::params::constraints::Constraint::check(
            $crate::params::constraints::LessThan($t),
            $x,
        )
    };
    ($x:ident <= $t:tt) => {
        $crate::params::constraints::Constraint::check(
            $crate::params::constraints::LessThanOrEqual($t),
            $x,
        )
    };
    ($x:tt > $t:tt) => {
        $crate::params::constraints::Constraint::check(
            $crate::params::constraints::GreaterThan($t),
            $x,
        )
    };
    ($x:ident >= $t:tt) => {
        $crate::params::constraints::Constraint::check(
            $crate::params::constraints::GreaterThanOrEqual($t),
            $x,
        )
    };
}

pub struct UnsatisfiedConstraintError<T> {
    pub value: T,
    pub target: Option<String>,
    pub constraint: Box<dyn Constraint<T>>,
}

impl<T: Debug + Send + Sync + 'static> Fail for UnsatisfiedConstraintError<T> {
    fn name(&self) -> Option<&str> { Some("UnsatisfiedConstraint") }

    fn cause(&self) -> Option<&dyn Fail> { None }

    fn backtrace(&self) -> Option<&Backtrace> { None }
}

impl<T> UnsatisfiedConstraintError<T> {
    pub fn new(value: T, constraint: Box<dyn Constraint<T>>) -> Self {
        UnsatisfiedConstraintError {
            value,
            target: None,
            constraint,
        }
    }

    pub fn with_target<S: ToString>(self, target: S) -> Self {
        UnsatisfiedConstraintError {
            value: self.value,
            target: Some(target.to_string()),
            constraint: self.constraint,
        }
    }
}

impl<T: Debug> Debug for UnsatisfiedConstraintError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        if let Some(ref target) = self.target {
            f.debug_struct("UnsatisfiedConstraintError")
                .field("value", &self.value)
                .field("target", target)
                .field("constraint", &self.constraint)
                .finish()
        } else {
            f.debug_struct("UnsatisfiedConstraintError")
                .field("value", &self.value)
                .field("constraint", &self.constraint)
                .finish()
        }
    }
}

impl<T: Debug> Display for UnsatisfiedConstraintError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::result::Result<(), std::fmt::Error> {
        if let Some(ref target) = self.target {
            write!(
                f,
                "Constraint {} on {:?} is unsatisfied for value {:?}.",
                self.constraint, target, self.value
            )
        } else {
            write!(
                f,
                "Constraint {} is unsatisfied for value {:?}.",
                self.constraint, self.value
            )
        }
    }
}

pub(crate) type Result<T> = std::result::Result<T, UnsatisfiedConstraintError<T>>;

pub trait Constraint<T>: Display + Debug + Send + Sync {
    fn is_satisfied_by(&self, value: &T) -> bool;

    fn check(self, value: T) -> Result<T>
    where
        T: Debug,
        Self: Sized + 'static,
    {
        if self.is_satisfied_by(&value) {
            Ok(value)
        } else {
            Err(UnsatisfiedConstraintError::new(value, Box::new(self)))
        }
    }
}

pub type Constraints<T> = Vec<Box<dyn Constraint<T>>>;

impl<T, C: Constraint<T> + ?Sized> Constraint<T> for Box<C> {
    fn is_satisfied_by(&self, value: &T) -> bool { self.deref().is_satisfied_by(value) }
}

macro_rules! impl_display {
    ($type:ty) => {
        impl Display for $type {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, stringify!($type)) }
        }
    };
}

macro_rules! impl_constraint {
    ($name:ident<$tc0:ident $(+ $tc:ident)*>; $self:ident, $value:ident, $impl:block) => {
        impl_display!($name);

        impl<T: $tc0 $(+$tc)*> Constraint<T> for $name {
            fn is_satisfied_by(&$self, $value: &T) -> bool { $impl }
        }
    };
    ($name:ident; $self:ident, $value:ident, $impl:block) => {
        impl_display!($name);

        impl<T> Constraint<T> for $name {
            fn is_satisfied_by(&$self, $value: &T) -> bool { $impl }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct All<C>(pub C);

impl<C: Display> Display for All<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "All({})", self.0) }
}

impl<C: Constraint<f64>> Constraint<f64> for All<C> {
    fn check(self, value: f64) -> Result<f64>
    where Self: Sized + 'static {
        self.0.check(value)
    }

    fn is_satisfied_by(&self, value: &f64) -> bool { self.0.is_satisfied_by(value) }
}

impl<C: Constraint<i64>> Constraint<i64> for All<C> {
    fn check(self, value: i64) -> Result<i64>
    where Self: Sized + 'static {
        self.0.check(value)
    }

    fn is_satisfied_by(&self, value: &i64) -> bool { self.0.is_satisfied_by(value) }
}

impl<C: Constraint<usize>> Constraint<usize> for All<C> {
    fn check(self, value: usize) -> Result<usize>
    where Self: Sized + 'static {
        self.0.check(value)
    }

    fn is_satisfied_by(&self, value: &usize) -> bool { self.0.is_satisfied_by(value) }
}

impl<T, C: Constraint<T>> Constraint<Vec<T>> for All<C> {
    fn check(self, value: Vec<T>) -> Result<Vec<T>>
    where
        Vec<T>: Debug,
        Self: Sized + 'static,
    {
        for v in value.iter() {
            if !self.0.is_satisfied_by(v) {
                return Err(UnsatisfiedConstraintError::new(value, Box::new(self)));
            }
        }

        Ok(value)
    }

    fn is_satisfied_by(&self, value: &Vec<T>) -> bool {
        value.iter().all(|v| self.0.is_satisfied_by(v))
    }
}

impl<T, C: Constraint<T>> Constraint<[T; 2]> for All<C> {
    fn check(self, value: [T; 2]) -> Result<[T; 2]>
    where
        [T; 2]: Debug,
        Self: Sized + 'static,
    {
        for v in value.iter() {
            if !self.0.is_satisfied_by(v) {
                return Err(UnsatisfiedConstraintError::new(value, Box::new(self)));
            }
        }

        Ok(value)
    }

    fn is_satisfied_by(&self, value: &[T; 2]) -> bool {
        value.iter().all(|v| self.0.is_satisfied_by(v))
    }
}

impl<T, C: Constraint<T>> Constraint<Vector<T>> for All<C> {
    fn check(self, value: Vector<T>) -> Result<Vector<T>>
    where
        Vector<T>: Debug,
        Self: Sized + 'static,
    {
        for v in value.iter() {
            if !self.0.is_satisfied_by(v) {
                return Err(UnsatisfiedConstraintError::new(value, Box::new(self)));
            }
        }

        Ok(value)
    }

    fn is_satisfied_by(&self, value: &Vector<T>) -> bool {
        value.iter().all(|v| self.0.is_satisfied_by(v))
    }
}

impl<T, C: Constraint<T>> Constraint<Matrix<T>> for All<C> {
    fn check(self, value: Matrix<T>) -> Result<Matrix<T>>
    where
        Matrix<T>: Debug,
        Self: Sized + 'static,
    {
        for v in value.iter() {
            if !self.0.is_satisfied_by(&v) {
                return Err(UnsatisfiedConstraintError::new(value, Box::new(self)));
            }
        }

        Ok(value)
    }

    fn is_satisfied_by(&self, value: &Matrix<T>) -> bool {
        value.iter().all(|v| self.0.is_satisfied_by(v))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Not<C>(pub C);

impl<C: Display> Display for Not<C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Not({})", self.0) }
}

impl<T, C: Constraint<T>> Constraint<T> for Not<C> {
    fn is_satisfied_by(&self, value: &T) -> bool { !self.0.is_satisfied_by(value) }
}

#[derive(Debug, Clone, Copy)]
pub struct Or<C1, C2>(pub (C1, C2));

impl<C1: Display, C2: Display> Display for Or<C1, C2> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Or({} || {})", (self.0).0, (self.0).1)
    }
}

impl<T, C1: Constraint<T>, C2: Constraint<T>> Constraint<T> for Or<C1, C2> {
    fn check(self, value: T) -> Result<T>
    where
        T: Debug,
        Self: Sized + 'static,
    {
        let (c1, c2) = self.0;

        c1.check(value).or_else(|uc| c2.check(uc.value))
    }

    fn is_satisfied_by(&self, value: &T) -> bool {
        (self.0).0.is_satisfied_by(value) || (self.0).1.is_satisfied_by(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct And<C1, C2>(pub (C1, C2));

impl<C1: Display, C2: Display> Display for And<C1, C2> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "And({} || {})", (self.0).0, (self.0).1)
    }
}

impl<T, C1: Constraint<T>, C2: Constraint<T>> Constraint<T> for And<C1, C2> {
    fn check(self, value: T) -> Result<T>
    where
        T: Debug,
        Self: Sized + 'static,
    {
        let (c1, c2) = self.0;

        c1.check(value).and_then(|value| c2.check(value))
    }

    fn is_satisfied_by(&self, value: &T) -> bool {
        (self.0).0.is_satisfied_by(value) && (self.0).1.is_satisfied_by(value)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Equal<T>(pub T);

impl<T: PartialEq + Debug> Display for Equal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Equal({:?})", self.0) }
}

impl<T: PartialEq + Send + Sync + Debug> Constraint<T> for Equal<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value == &self.0 }
}

#[derive(Debug, Clone, Copy)]
pub struct LessThan<T>(pub T);

impl<T: PartialOrd + Debug> Display for LessThan<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "LessThan({:?})", self.0) }
}

impl<T: PartialOrd + Send + Sync + Debug> Constraint<T> for LessThan<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value < &self.0 }
}

#[derive(Debug, Clone, Copy)]
pub struct Negative;

impl_constraint!(Negative<PartialOrd + Zero>; self, value, { value < &zero() });

#[derive(Debug, Clone, Copy)]
pub struct LessThanOrEqual<T>(pub T);

impl<T: PartialOrd + Debug> Display for LessThanOrEqual<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LessThanOrEqual({:?})", self.0)
    }
}

impl<T: PartialOrd + Send + Sync + Debug> Constraint<T> for LessThanOrEqual<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value <= &self.0 }
}

#[derive(Debug, Clone, Copy)]
pub struct NonPositive;

impl_constraint!(NonPositive<PartialOrd + Zero>; self, value, { value <= &zero() });

#[derive(Debug, Clone, Copy)]
pub struct GreaterThan<T>(pub T);

impl<T: PartialOrd + Debug> Display for GreaterThan<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "GreaterThan({:?})", self.0) }
}

impl<T: PartialOrd + Send + Sync + Debug> Constraint<T> for GreaterThan<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value > &self.0 }
}

#[derive(Debug, Clone, Copy)]
pub struct Positive;

impl_constraint!(Positive<PartialOrd + Zero>; self, value, { value > &zero() });

#[derive(Debug, Clone, Copy)]
pub struct GreaterThanOrEqual<T>(pub T);

impl<T: PartialOrd + Debug> Display for GreaterThanOrEqual<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GreaterThanOrEqual({:?})", self.0)
    }
}

impl<T: PartialOrd + Send + Sync + Debug> Constraint<T> for GreaterThanOrEqual<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value >= &self.0 }
}

#[derive(Debug, Clone, Copy)]
pub struct Interval<T> {
    pub lb: T,
    pub ub: T,
}

impl<T: PartialOrd + Debug> Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{:?}, {:?}]", self.lb, self.ub)
    }
}

impl<T: PartialOrd + Send + Sync + Debug> Constraint<T> for Interval<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value >= &self.lb && value <= &self.ub }
}

#[derive(Debug, Clone, Copy)]
pub struct NonNegative;

impl_constraint!(NonNegative<PartialOrd + Zero>; self, value, { value >= &zero() });

#[derive(Debug, Clone, Copy)]
pub struct Natural;

impl_constraint!(Natural<PrimInt>; self, value, { value > &zero() });

#[derive(Debug, Clone, Copy)]
pub struct Empty;

impl Display for Empty {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Empty") }
}

impl<T> Constraint<Vec<T>> for Empty {
    fn is_satisfied_by(&self, vec: &Vec<T>) -> bool { vec.len() == 0 }
}

impl<T> Constraint<Vector<T>> for Empty {
    fn is_satisfied_by(&self, vector: &Vector<T>) -> bool { vector.len() == 0 }
}

#[derive(Debug, Clone, Copy)]
pub struct Square;

impl Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Square") }
}

impl<T> Constraint<Matrix<T>> for Square {
    fn is_satisfied_by(&self, matrix: &Matrix<T>) -> bool { matrix.is_square() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_negative() {
        let c = NonNegative;

        assert_eq!(c.to_string(), "NonNegative");

        assert!(!c.is_satisfied_by(&-1.50));
        assert!(!c.is_satisfied_by(&-0.50));

        assert!(c.is_satisfied_by(&0.50));
        assert!(c.is_satisfied_by(&1.50));
    }
}
