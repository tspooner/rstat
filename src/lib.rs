extern crate rand;
extern crate spaces;
extern crate ndarray;
extern crate special_fun;

mod consts;
mod macros;

pub mod core;
pub mod univariate;
pub mod multivariate;

import_all!(mixture);

pub use self::core::{Distribution, Probability};
