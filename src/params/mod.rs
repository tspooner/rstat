///////////////////////////////////////////////////////////////////////////////////////////////////
// Constraints
///////////////////////////////////////////////////////////////////////////////////////////////////
#[macro_use]
pub mod constraints;

use constraints::{All, NonNegative, Positive};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Params
///////////////////////////////////////////////////////////////////////////////////////////////////
pub trait Param {
    type Value;

    /// Returns a reference to the parameter's value.
    ///
    /// # Examples
    /// ```
    /// # use rstat::params::{Param, Loc};
    /// let loc = Loc::new(1.0)?;
    ///
    /// assert_eq!(loc.value(), &1.0);
    /// # Ok::<(), failure::Error>(())
    /// ```
    fn value(&self) -> &Self::Value;

    /// Converts the parameter into it's value.
    fn into_value(self) -> Self::Value;

    /// Returns the constraints associated with this parameter type.
    fn constraints() -> constraints::Constraints<Self::Value>;
}

macro_rules! impl_param {
    ($name:ident) => {
        impl<T> $crate::params::Param for $name<T> {
            type Value = T;

            fn value(&self) -> &T { &self.0 }

            fn into_value(self) -> T { self.0 }

            fn constraints() -> $crate::params::constraints::Constraints<T> { vec![] }
        }
    };
    ($name:ident s.t. $cst:ty { $cst_build:expr }) => {
        impl<T> $crate::params::Param for $name<T>
        where $cst: $crate::params::constraints::Constraint<T>
        {
            type Value = T;

            fn value(&self) -> &T { &self.0 }

            fn into_value(self) -> T { self.0 }

            fn constraints() -> $crate::params::constraints::Constraints<T> {
                vec![Box::new($cst_build)]
            }
        }
    };
}

macro_rules! param {
    (@struct $name:ident) => {
        #[derive(Debug, Clone, Copy)]
        #[cfg_attr(
            feature = "serde",
            derive(Serialize, Deserialize),
            serde(crate = "serde_crate")
        )]
        pub struct $name<T>(pub T);

        impl<T> std::ops::Deref for $name<T> {
            type Target = T;

            fn deref(&self) -> &T { &self.0 }
        }

        impl<T: std::fmt::Display> std::fmt::Display for $name<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
                self.0.fmt(f)
            }
        }
    };
    ($name:ident) => {
        param!(@struct $name);

        impl<T: std::fmt::Debug> $name<T> {
            pub fn new(value: T) -> Result<
                Self,
                $crate::params::constraints::UnsatisfiedConstraintError<T>
            > {
                Ok($name(value))
            }

            pub fn new_unchecked(value: T) -> $name<T> { $name(value) }
        }

        impl_param!($name);

    };
    ($name:ident s.t. $cst:ty { $cst_build:expr }) => {
        param!(@struct $name);

        impl<T: std::fmt::Debug> $name<T>
        where $cst: $crate::params::constraints::Constraint<T>,
        {
            pub fn new(value: T) -> Result<
                Self,
                $crate::params::constraints::UnsatisfiedConstraintError<T>
            > {
                Ok($name($crate::params::constraints::Constraint::check($cst_build, value)?))
            }

            pub fn new_unchecked(value: T) -> $name<T> { $name(value) }
        }

        impl_param!($name s.t. $cst { $cst_build });
    };
}

macro_rules! params {
    (@munch () -> {$(#[$attr:meta])* $name:ident $(($id:ident: $ty:ident<$($ity:ident),*>))*}) => {
        #[derive(Debug, Clone)]
        #[cfg_attr(
            feature = "serde",
            derive(Serialize, Deserialize),
            serde(crate = "serde_crate")
        )]
        $(#[$attr])*
        pub struct $name {
            $(pub $id: $ty<$($ity),*>),*
        }

        impl $name {
            pub fn new($($id: <$ty<$($ity),*> as $crate::params::Param>::Value),*) -> Result<$name, failure::Error> {
                Ok($name {
                    $($id: $ty::new($id)?),*
                })
            }

            pub fn new_unchecked($($id: <$ty<$($ity),*> as $crate::params::Param>::Value),*) -> $name {
                $name {
                    $($id: $ty::new_unchecked($id)),*
                }
            }

            // $(#[inline(always)] pub fn $id(&self) -> &$ty<$($ity),*> { &self.$id })*
        }
    };
    (@munch ($id:ident: $ty:ident<$($ity:ident),*>) -> {$($output:tt)*}) => {
        params!(@munch () -> {$($output)* ($id: $ty<$($ity),*>)});
    };
    (@munch ($id:ident: $ty:ident<$($ity:ident),*>, $($next:tt)*) -> {$($output:tt)*}) => {
        params!(@munch ($($next)*) -> {$($output)* ($id: $ty<$($ity),*>)});
    };
    ($(#[$attr:meta])* $name:ident { $($input:tt)*} ) => {
        params!(@munch ($($input)*) -> {$(#[$attr])* $name});
    }
}

macro_rules! locscale_params {
    ($(#[$attr:meta])* $name:ident { $loc:ident<$lty:tt>, $scale:ident<$sty:tt> }) => {
        pub use $crate::params::{Loc, Scale};

        params! {
            $(#[$attr])*
            $name {
                $loc: Loc<$lty>,
                $scale: Scale<$sty>
            }
        }
    }
}

macro_rules! shape_params {
    ($(#[$attr:meta])* $name:ident<$ity:ident> { $($id:ident),* }) => {
        pub use $crate::params::Shape;

        params! {
            $(#[$attr])*
            $name {
                $($id: Shape<$ity>),*
            }
        }
    }
}

param!(Loc);

param!(Scale s.t. All<Positive> { All(Positive) });

param!(Rate s.t. All<Positive> { All(Positive) });

param!(Shape s.t. All<Positive> { All(Positive) });

param!(DOF s.t. All<Positive> { All(Positive) });

param!(Count s.t. All<NonNegative> { All(NonNegative) });

mod corr;
pub use self::corr::Corr;

mod concentrations;
pub use self::concentrations::Concentrations;
