//! A collection of multivariate (i.e. multiple output)
//! [distributions](../trait.Distribution.html).
pub use crate::statistics::MvMoments as Moments;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Continuous
///////////////////////////////////////////////////////////////////////////////////////////////////
pub mod normal {
    pub use crate::normal::{
        MvNormal as Normal,
        MvNormalGrad as Grad,
        MvNormalParams as Params,

        DiagonalNormal, DiagonalNormalGrad, DiagonalNormalParams,
        IsotropicNormal, IsotropicNormalGrad, IsotropicNormalParams,
    };
}

pub mod lognormal;

pub mod dirichlet;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Discrete
///////////////////////////////////////////////////////////////////////////////////////////////////
pub mod multinomial;
