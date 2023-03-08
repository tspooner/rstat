//! A collection of bivariate (i.e. pair output)
//! [distributions](trait.Distribution.html).
pub use crate::statistics::MvMoments as Moments;

// Continuous:
pub mod normal {
    pub use crate::normal::{
        BvNormal as Normal,
        BvNormalGrad as Grad,
        BvNormalParams as Params,
    };
}
