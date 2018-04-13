#![macro_use]
#![allow(unused_macros)]

macro_rules! clip {
    ($lb:expr, $x:expr, $ub:expr) => {{
        $lb.max($ub.min($x))
    }};
}

macro_rules! import_all {
    ($module:ident) => {
        mod $module;
        pub use self::$module::*;
    };
}

macro_rules! assert_positive_real {
    ($var:ident) => {
        if $var <= 0.0f64 {
            panic!("$var must be a positive, finite real number.")
        }
    }
}

macro_rules! assert_natural {
    ($var:ident) => {
        if $var == 0usize {
            panic!("$var must be a positive, finite real number.")
        }
    }
}

macro_rules! assert_len {
    ($var:ident => $len:expr; $name:ident) => {
        if $var.len() == $len {
            panic!("$var must have length $name.")
        }
    }
}

macro_rules! assert_dim {
    ($var:ident => $dim:expr; $name:ident) => {
        if $var.dim() == $dim {
            panic!("$var must have dimensions $name.")
        }
    }
}
