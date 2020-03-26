//! Probability distributions and statistics in Rust with integrated fitting routines, convolution
//! support and mixtures.
// #![warn(missing_docs)]
extern crate failure;

extern crate rand;
extern crate rand_distr;

extern crate num;
extern crate spaces;
extern crate special_fun;

extern crate ndarray;
#[cfg(any(test, backend))]
extern crate ndarray_linalg;

mod consts;
mod linalg;

mod prelude;

mod probability;
pub use self::probability::{Probability, ProbabilityError};

mod simplex;
pub use self::simplex::Simplex;

mod distribution;
pub use self::distribution::*;

mod convolution;
pub use self::convolution::*;

pub mod fitting;
pub mod statistics;

#[macro_use]
pub mod constraints;

pub mod univariate;
pub mod multivariate;

// mod mixture;
// pub use self::mixture::Mixture;
