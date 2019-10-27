extern crate rand;
extern crate rand_distr;
extern crate spaces;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate special_fun;

mod consts;
mod macros;

import_all!(probability);
import_all!(distribution);
import_all!(statistics);
import_all!(convolution);
import_all!(fitting);

pub mod prelude;
pub mod univariate;
pub mod multivariate;

// import_all!(mixture);
