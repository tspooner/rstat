//! Probability distributions and statistics in Rust with integrated fitting routines, convolution
//! support and mixtures.
// #![warn(missing_docs)]

extern crate rand;
extern crate rand_distr;
extern crate spaces;
extern crate ndarray;
extern crate num_traits;
extern crate special_fun;

mod consts;
mod macros;
mod prelude;

mod probability;
pub use self::probability::*;

mod distribution;
pub use self::distribution::*;

mod convolution;
pub use self::convolution::*;

pub mod fitting;
pub mod statistics;
pub mod validation;

pub mod univariate;
pub mod multivariate;

mod mixture;
pub use self::mixture::Mixture;
