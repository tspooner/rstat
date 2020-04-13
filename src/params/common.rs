use crate::linalg::Vector;
use num::{Zero, Integer, Unsigned};
use super::{Param, constraints::{self, Constraint}};

#[macro_export]
macro_rules! param {
    (@impl_new $name:ident<$T:ident> s.t. [$($cst:expr),*]) => {
        pub fn new(value: $T) -> Result<
            Self,
            $crate::params::constraints::UnsatisfiedConstraintError<$T>
        > {
            $(let value = $crate::params::constraints::Constraint::check($cst, value)?;)*

            Ok($name(value))
        }
    };
    (@impl_param $name:ident<$T:ident> s.t. [$($cst:expr),*]) => {
        type Value = $T;

        fn value(&self) -> &$T { &self.0 }

        fn constraints() -> $crate::params::constraints::Constraints<$T> {
            vec![
                $(Box::new($cst)),*
            ]
        }
    };
    ($name:ident<$T:ident$(: $tc0:ident $(+ $tc:ident)*)?> s.t. [$($cst:expr),*]) => {
        #[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
        #[derive(Debug, Clone, Copy)]
        pub struct $name<$T$(: $tc0 $(+ $tc)*)?>(pub $T);

        impl<$T: std::fmt::Debug $(+ $tc0 $(+ $tc)*)?> $name<$T> {
            param!(@impl_new $name<$T> s.t. [$($cst),*]);
        }

        impl<$T $(: $tc0 $(+ $tc)*)?> Param for $name<$T> {
            param!(@impl_param $name<$T> s.t. [$($cst),*]);
        }

        impl<$T: std::fmt::Display $(+ $tc0 $(+ $tc)*)?> std::fmt::Display for $name<$T> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
                self.0.fmt(f)
            }
        }
    };
}

param!(Loc<T> s.t. []);

param!(Scale<T: Zero + PartialOrd> s.t. [constraints::Positive]);

param!(Rate<T: Zero + PartialOrd> s.t. [constraints::Positive]);

param!(Shape<T: Zero + PartialOrd> s.t. [constraints::Positive]);

param!(DOF<T: Integer + Unsigned> s.t. [constraints::Positive]);

param!(Count<T: Integer + Unsigned> s.t. [constraints::NonNegative]);

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Concentrations(pub Vector<f64>);

impl Concentrations {
    pub fn new(value: Vector<f64>) -> Result<
        Self,
        constraints::UnsatisfiedConstraintError<Vector<f64>>
    > {
        constraints::All(constraints::Positive).check(value)
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
