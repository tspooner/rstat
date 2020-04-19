use super::{
    constraints::{self, All, Constraint, Interval, NonNegative, Positive},
    Param,
};
use crate::linalg::Vector;
use num::{Integer, One, Signed, Unsigned, Zero};

#[macro_export]
macro_rules! param {
    (@common $name:ident) => {
        #[derive(Debug, Clone, Copy)]
        #[cfg_attr(
            feature = "serde",
            derive(Serialize, Deserialize),
            serde(crate = "serde_crate")
        )]
        pub struct $name<T>(pub T);

        impl<T: std::fmt::Display> std::fmt::Display for $name<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
                self.0.fmt(f)
            }
        }
    };
    ($name:ident) => {
        param!(@common $name);

        impl<T: std::fmt::Debug> $name<T> {
            pub fn new(value: T) -> Result<
                Self,
                $crate::params::constraints::UnsatisfiedConstraintError<T>
            > {
                Ok($name(value))
            }
        }

        impl<T> Param for $name<T> {
            type Value = T;

            fn value(&self) -> &T { &self.0 }

            fn constraints() -> $crate::params::constraints::Constraints<T> { vec![] }
        }
    };
    ($name:ident s.t. $cst:ty { $cst_build:expr }) => {
        param!(@common $name);

        impl<T: std::fmt::Debug> $name<T>
        where $cst: Constraint<T>,
        {
            pub fn new(value: T) -> Result<
                Self,
                $crate::params::constraints::UnsatisfiedConstraintError<T>
            > {
                Ok($name($crate::params::constraints::Constraint::check($cst_build, value)?))
            }
        }

        impl<T> Param for $name<T>
        where $cst: Constraint<T>,
        {
            type Value = T;

            fn value(&self) -> &T { &self.0 }

            fn constraints() -> $crate::params::constraints::Constraints<T> {
                vec![Box::new($cst_build)]
            }
        }
    };
}

param!(Loc);

param!(Scale s.t. All<Positive> { All(Positive) });

param!(Rate s.t. All<Positive> { All(Positive) });

param!(Shape s.t. All<Positive> { All(Positive) });

param!(DOF s.t. All<Positive> { All(Positive) });

param!(Count s.t. All<NonNegative> { All(NonNegative) });

#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Corr<T>(pub T);

impl<T> Corr<T>
where
    T: std::fmt::Debug + num::One + std::ops::Neg<Output = T> + 'static,
    Interval<T>: Constraint<T>,
    All<Interval<T>>: Constraint<T>,
{
    pub fn new(value: T) -> Result<Self, constraints::UnsatisfiedConstraintError<T>> {
        Ok(Corr(
            All(Interval {
                lb: -T::one(),
                ub: T::one(),
            })
            .check(value)?,
        ))
    }
}

impl<T> Param for Corr<T>
where
    T: std::fmt::Debug + num::One + std::ops::Neg<Output = T> + 'static,
    Interval<T>: Constraint<T>,
    All<Interval<T>>: Constraint<T>,
{
    type Value = T;

    fn value(&self) -> &T { &self.0 }

    fn constraints() -> constraints::Constraints<Self::Value> {
        vec![Box::new(constraints::All(constraints::Interval {
            lb: -T::one(),
            ub: T::one(),
        }))]
    }
}

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Concentrations(pub Vector<f64>);

impl Concentrations {
    pub fn new(
        value: Vector<f64>,
    ) -> Result<Self, constraints::UnsatisfiedConstraintError<Vector<f64>>> {
        constraints::All(constraints::Positive)
            .check(value)
            .and_then(|value| constraints::Not(constraints::Empty).check(value))
            .map(Concentrations)
    }
}

impl Param for Concentrations {
    type Value = Vector<f64>;

    fn value(&self) -> &Vector<f64> { &self.0 }

    fn constraints() -> constraints::Constraints<Self::Value> {
        let c1 = constraints::Not(constraints::Empty);
        let c2 = constraints::All(constraints::Positive);

        vec![Box::new(c1), Box::new(c2)]
    }
}
