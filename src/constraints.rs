use crate::prelude::*;
use failure::Fail;
use num::{Zero, zero, PrimInt};
use std::{fmt::{self, Debug, Display}, ops::Deref};

macro_rules! assert_constraint {
    ($x:ident+) => {
        $crate::constraints::Constraint::check(
            $crate::constraints::NonNegative, $x
        )
    };
    ($x:ident == $t:tt) => {
        $crate::constraints::Constraint::check(
            $crate::constraints::Equal($t), $x
        )
    };
    ($x:ident < $t:tt) => {
        $crate::constraints::Constraint::check(
            $crate::constraints::LessThan($t), $x
        )
    };
    ($x:ident <= $t:tt) => {
        $crate::constraints::Constraint::check(
            $crate::constraints::LessThanOrEqual($t), $x
        )
    };
    ($x:tt > $t:tt) => {
        $crate::constraints::Constraint::check(
            $crate::constraints::GreaterThan($t), $x
        )
    };
    ($x:ident >= $t:tt) => {
        $crate::constraints::Constraint::check(
            $crate::constraints::GreaterThanOrEqual($t), $x
        )
    };
}

#[derive(Debug, Fail)]
#[fail(display = "Constraint {} unsatisfied by value {}.", constraint, value)]
pub struct UnsatisfiedConstraint<T: Display + Debug + Send + Sync + 'static> {
    pub value: T,
    pub constraint: Box<dyn Constraint<T>>,
}

pub type Result<T> = std::result::Result<T, UnsatisfiedConstraint<T>>;

///
pub trait Constraint<T>: Display + Debug + Send + Sync {
    fn check(self, value: T) -> Result<T>
    where
        T: Display + Debug + Send + Sync + 'static,
        dyn Constraint<T>: Display + Debug + Send + Sync + 'static,
        Self: Sized + 'static,
    {
        if self.is_satisfied_by(&value) {
            Ok(value)
        } else {
            Err(UnsatisfiedConstraint {
                value,
                constraint: Box::new(self),
            })
        }
    }

    fn is_satisfied_by(&self, value: &T) -> bool;
}

impl<T, C: Constraint<T> + ?Sized> Constraint<T> for Box<C> {
    fn is_satisfied_by(&self, value: &T) -> bool {
        self.deref().is_satisfied_by(value)
    }
}

macro_rules! impl_display {
    ($type:ty) => {
        impl Display for $type {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, stringify!($type))
            }
        }
    }
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

///
#[derive(Debug, Clone, Copy)]
pub struct Or<C1, C2>((C1, C2));

impl<C1: Display, C2: Display> Display for Or<C1, C2> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Or({} || {})", (self.0).0, (self.0).1)
    }
}

impl<T, C1: Constraint<T>, C2: Constraint<T>> Constraint<T> for Or<C1, C2> {
    fn check(self, value: T) -> Result<T>
    where
        T: Display + Debug + Send + Sync + 'static,
        dyn Constraint<T>: Display + Debug + Send + Sync + 'static,
        Self: Sized + 'static,
    {
        let (c1, c2) = self.0;

        c1.check(value).or_else(|uc| c2.check(uc.value))
    }

    fn is_satisfied_by(&self, value: &T) -> bool {
        (self.0).0.is_satisfied_by(value) || (self.0).1.is_satisfied_by(value)
    }
}

///
#[derive(Debug, Clone, Copy)]
pub struct And<C1, C2>((C1, C2));

impl<C1: Display, C2: Display> Display for And<C1, C2> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "And({} || {})", (self.0).0, (self.0).1)
    }
}

impl<T, C1: Constraint<T>, C2: Constraint<T>> Constraint<T> for And<C1, C2> {
    fn check(self, value: T) -> Result<T>
    where
        T: Display + Debug + Send + Sync + 'static,
        dyn Constraint<T>: Display + Debug + Send + Sync + 'static,
        Self: Sized + 'static,
    {
        let (c1, c2) = self.0;

        c1.check(value).and_then(|value| c2.check(value))
    }

    fn is_satisfied_by(&self, value: &T) -> bool {
        (self.0).0.is_satisfied_by(value) && (self.0).1.is_satisfied_by(value)
    }
}

///
#[derive(Debug, Clone, Copy)]
pub struct Equal<T: PartialEq>(pub T);

impl<T: PartialEq + Display> Display for Equal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Equal({})", self.0)
    }
}

impl<T: PartialEq + Send + Sync + Debug + Display> Constraint<T> for Equal<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value == &self.0 }
}

///
#[derive(Debug, Clone, Copy)]
pub struct LessThan<T: PartialOrd>(pub T);

impl<T: PartialOrd + Display> Display for LessThan<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LessThan({})", self.0)
    }
}

impl<T: PartialOrd + Send + Sync + Debug + Display> Constraint<T> for LessThan<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value < &self.0 }
}

///
#[derive(Debug, Clone, Copy)]
pub struct Negative;

impl_constraint!(Negative<PartialOrd + Zero>; self, value, { value < &zero() });

///
#[derive(Debug, Clone, Copy)]
pub struct LessThanOrEqual<T: PartialOrd>(pub T);

impl<T: PartialOrd + Display> Display for LessThanOrEqual<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LessThanOrEqual({})", self.0)
    }
}

impl<T: PartialOrd + Send + Sync + Debug + Display> Constraint<T> for LessThanOrEqual<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value <= &self.0 }
}

///
#[derive(Debug, Clone, Copy)]
pub struct NonPositive;

impl_constraint!(NonPositive<PartialOrd + Zero>; self, value, { value <= &zero() });

///
#[derive(Debug, Clone, Copy)]
pub struct GreaterThan<T: PartialOrd>(pub T);

impl<T: PartialOrd + Display> Display for GreaterThan<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GreaterThan({})", self.0)
    }
}

impl<T: PartialOrd + Send + Sync + Debug + Display> Constraint<T> for GreaterThan<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value > &self.0 }
}

///
#[derive(Debug, Clone, Copy)]
pub struct Positive;

impl_constraint!(Positive<PartialOrd + Zero>; self, value, { value > &zero() });

///
#[derive(Debug, Clone, Copy)]
pub struct GreaterThanOrEqual<T: PartialOrd>(pub T);

impl<T: PartialOrd + Display> Display for GreaterThanOrEqual<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GreaterThanOrEqual({})", self.0)
    }
}

impl<T: PartialOrd + Send + Sync + Debug + Display> Constraint<T> for GreaterThanOrEqual<T> {
    fn is_satisfied_by(&self, value: &T) -> bool { value >= &self.0 }
}

///
#[derive(Debug, Clone, Copy)]
pub struct Interval<T: PartialOrd> {
    pub lb: T,
    pub ub: T,
}

impl<T: PartialOrd + Display> Display for Interval<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[{}, {}]", self.lb, self.ub)
    }
}

impl<T: PartialOrd + Send + Sync + Debug + Display> Constraint<T> for Interval<T> {
    fn is_satisfied_by(&self, value: &T) -> bool {
        value >= &self.lb && value <= &self.ub
    }
}

///
#[derive(Debug, Clone, Copy)]
pub struct NonNegative;

impl_constraint!(NonNegative<PartialOrd + Zero>; self, value, { value >= &zero() });

///
#[derive(Debug, Clone, Copy)]
pub struct Natural;

impl_constraint!(Natural<PrimInt>; self, value, { value > &zero() });

///
#[derive(Debug, Clone, Copy)]
pub struct Square;

impl Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Square")
    }
}

impl<T> Constraint<Matrix<T>> for Square {
    fn is_satisfied_by(&self, matrix: &Matrix<T>) -> bool {
        matrix.is_square()
    }
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
