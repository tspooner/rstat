extern crate rand;
extern crate spaces;
extern crate ndarray;
extern crate special_fun;

mod consts;
mod macros;

pub mod continuous;
pub mod core;
pub mod discrete;

import_all!(uniform);
import_all!(degenerate);

pub use self::core::{Distribution, Probability};
