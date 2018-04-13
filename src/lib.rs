extern crate ndarray;
extern crate rand;
extern crate spaces;
extern crate special_fun;

mod consts;
mod macros;

pub mod core;
pub mod continuous;
pub mod discrete;

pub use self::core::{
    Distribution,
    Probability
};
