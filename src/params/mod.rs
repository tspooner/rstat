#[macro_use]
pub mod constraints;

pub trait Param {
    type Value;

    fn value(&self) -> &Self::Value;

    fn constraints() -> constraints::Constraints<Self::Value>;
}

#[macro_use]
mod common;
pub use self::common::*;

#[macro_export]
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
            pub fn new($($id: <$ty<$($ity),*> as Param>::Value),*) -> Result<$name, failure::Error> {
                Ok($name {
                    $($id: $ty::new($id)?),*
                })
            }

            pub fn new_unchecked($($id: <$ty<$($ity),*> as Param>::Value),*) -> $name {
                $name {
                    $($id: $ty($id)),*
                }
            }

            $(#[inline(always)] pub fn $id(&self) -> &$ty<$($ity),*> { &self.$id })*
        }

        impl Copy for $name
        where
            $($ty<$($ity),*>: Copy),*
        {}
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
    ($(#[$attr:meta])* $name:ident { $loc:ident<$lty:ident>, $scale:ident<$sty:ident> }) => {
        use $crate::params::{Loc as __Loc, Scale as __Scale};

        params! {
            $(#[$attr])*
            $name {
                $loc: __Loc<$lty>,
                $scale: __Scale<$sty>,
            }
        }
    }
}

macro_rules! shape_params {
    ($(#[$attr:meta])* $name:ident<$ity:ident> { $($id:ident),* }) => {
        use $crate::params::Shape as __Shape;

        params! {
            $(#[$attr])*
            $name {
                $($id: __Shape<$ity>),*
            }
        }
    }
}
