//! A collection of traits for generalised construction of distributions.
use crate::{
    linalg::{Vector, Matrix},
    normal::{MvNormal, DiagonalNormal, IsotropicNormal, BvNormal, PairedNormal, UvNormal},
    ContinuousDistribution,
};

#[derive(Debug, Clone, Copy)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Builder;

///////////////////////////////////////////////////////////////////////////////////////////////////
// Building Normal distributions
///////////////////////////////////////////////////////////////////////////////////////////////////
pub trait BuildNormal<M, S> {
    type Normal: ContinuousDistribution;

    fn build(mu: M, sigma: S) -> Result<Self::Normal, failure::Error>;

    fn build_unchecked(mu: M, sigma: S) -> Self::Normal;
}

macro_rules! impl_build_normal {
    (BuildNormal<$m:ty, $s:ty, Normal = $n:ty> with $build:ident and $build_unchecked:ident) => {
        impl BuildNormal<$m, $s> for Builder {
            type Normal = $n;

            fn build(mu: $m, sigma: $s) -> Result<$n, failure::Error> {
                <$n>::$build(mu, sigma)
            }

            fn build_unchecked(mu: $m, sigma: $s) -> $n {
                <$n>::$build_unchecked(mu, sigma)
            }
        }
    };
}

impl_build_normal!(BuildNormal<f64, f64, Normal = UvNormal> with new and new_unchecked);

impl BuildNormal<[f64; 2], ([f64; 2], f64)> for Builder {
    type Normal = BvNormal;

    fn build(mu: [f64; 2], sigma: ([f64; 2], f64)) -> Result<BvNormal, failure::Error> {
        BvNormal::new(mu, sigma.0, sigma.1)
    }

    fn build_unchecked(mu: [f64; 2], sigma: ([f64; 2], f64)) -> BvNormal {
        BvNormal::new_unchecked(mu, sigma.0, sigma.1)
    }
}

impl_build_normal!(
    BuildNormal<[f64; 2], [f64; 2], Normal = PairedNormal>
    with new and new_unchecked
);

impl_build_normal!(
    BuildNormal<[f64; 2], f64, Normal = BvNormal>
    with isotropic and isotropic_unchecked
);

impl_build_normal!(
    BuildNormal<Vec<f64>, f64, Normal = IsotropicNormal>
    with isotropic and isotropic_unchecked
);

impl_build_normal!(
    BuildNormal<Vector<f64>, f64, Normal = IsotropicNormal>
    with isotropic and isotropic_unchecked
);

impl_build_normal!(
    BuildNormal<Vec<f64>, Vec<f64>, Normal = DiagonalNormal>
    with diagonal and diagonal_unchecked
);

impl_build_normal!(
    BuildNormal<Vector<f64>, Vector<f64>, Normal = DiagonalNormal>
    with diagonal and diagonal_unchecked
);

impl_build_normal!(
    BuildNormal<Vec<f64>, Matrix<f64>, Normal = MvNormal>
    with new and new_unchecked
);

impl_build_normal!(
    BuildNormal<Vector<f64>, Matrix<f64>, Normal = MvNormal>
    with new and new_unchecked
);
