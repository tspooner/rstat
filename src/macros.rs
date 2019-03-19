#![macro_use]
#![allow(unused_macros)]

macro_rules! clip {
    ($lb:expr, $x:expr, $ub:expr) => {{
        $lb.max($ub.min($x))
    }};
}

macro_rules! import_all {
    ($module:ident) => {
        pub(crate) mod $module;
        pub use self::$module::*;
    };
}

macro_rules! assert_positive_real {
    ($var:expr) => {
        if $var <= 0.0f64 {
            panic!("$var must be a positive, finite real number.")
        }
    };
}

macro_rules! assert_bounded {
    ($lb:expr; $var:expr; $ub:expr) => {
        if $lb <= $var && $var <= $ub {
            panic!("$var must be a positive, finite real number.")
        }
    };
}

macro_rules! assert_natural {
    ($var:expr) => {
        if $var == 0usize {
            panic!("$var must be a positive, finite real number.")
        }
    };
}

macro_rules! assert_len {
    ($var:expr => $len:expr; $name:ident) => {
        if $var.len() == $len {
            panic!("$var must have the same length as $name.")
        }
    };
}

macro_rules! assert_dim {
    ($var:expr => $dim:expr; $name:ident) => {
        if $var.dim() == $dim {
            panic!("$var must have the same dimensions as $name.")
        }
    };
}

macro_rules! assert_square {
    ($var:expr) => {
        if !$var.is_square() {
            panic!("$var must be a square matrix.")
        }
    };
}
