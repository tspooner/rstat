//! A collection of univariate (i.e. scalar output)
//! [distributions](trait.Distribution.html).
pub use crate::statistics::UvMoments as Moments;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Abstract
///////////////////////////////////////////////////////////////////////////////////////////////////
pub mod uniform;

pub mod degenerate;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Continuous
///////////////////////////////////////////////////////////////////////////////////////////////////
pub mod arcsine;

pub mod beta;

pub mod beta_prime;

pub mod cauchy;

pub mod chi;

pub mod chi_sq;

pub mod cosine;

pub mod erlang;

pub mod exponential;

pub mod f_dist;

pub mod folded_normal;

pub mod frechet;

pub mod gamma;

pub mod gev;

pub mod gpd;

pub mod gumbel;

pub mod inverse_gamma;

pub mod inverse_normal;

pub mod kumaraswamy;

pub mod laplace;

pub mod levy;

pub mod logistic;

pub mod lognormal;

pub mod normal {
    pub use crate::normal::{
        UvNormal as Normal,
        UvNormalGrad as Grad,
        UvNormalParams as Params,
    };
}

pub mod pareto;

pub mod rayleigh;

pub mod student_t;

pub mod triangular;

pub mod weibull;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Discrete
///////////////////////////////////////////////////////////////////////////////////////////////////
pub mod bernoulli;

pub mod beta_binomial;

pub mod binomial;

pub mod categorical;

pub mod geometric;

pub mod poisson;
