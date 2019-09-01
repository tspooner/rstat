extern crate rand;
extern crate rand_distr;
extern crate spaces;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate special_fun;

mod consts;
mod macros;

pub mod core;
pub mod univariate;
pub mod multivariate;

import_all!(mixture);

pub use self::core::Probability;
pub use self::core::distribution::*;
