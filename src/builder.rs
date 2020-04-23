//! A collection of traits for generalised construction of distributions.
use crate::{
    univariate::normal::Normal as UvNormal,
    bivariate::normal::Normal as BvNormal,
    multivariate::normal as mv_normal,
    linalg::{Vector, Matrix},
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

    fn build(mean: M, stddev: S) -> Result<Self::Normal, failure::Error>;

    fn build_unchecked(mean: M, stddev: S) -> Self::Normal;
}

macro_rules! impl_build_normal {
    (BuildNormal<$m:ty, $s:ty, Normal = $n:ty> with $build:ident and $build_unchecked:ident) => {
        impl BuildNormal<$m, $s> for Builder {
            type Normal = $n;

            fn build(mean: $m, stddev: $s) -> Result<$n, failure::Error> {
                <$n>::$build(mean, stddev)
            }

            fn build_unchecked(mean: $m, stddev: $s) -> $n {
                <$n>::$build_unchecked(mean, stddev)
            }
        }
    };
}

impl_build_normal!(BuildNormal<f64, f64, Normal = UvNormal> with new and new_unchecked);

impl_build_normal!(
    BuildNormal<[f64; 2], [f64; 2], Normal = BvNormal>
    with independent and independent_unchecked
);

impl_build_normal!(
    BuildNormal<[f64; 2], f64, Normal = BvNormal>
    with isotropic and isotropic_unchecked
);

impl_build_normal!(
    BuildNormal<Vec<f64>, f64, Normal = mv_normal::IsotropicNormal>
    with isotropic and isotropic_unchecked
);

impl_build_normal!(
    BuildNormal<Vector<f64>, f64, Normal = mv_normal::IsotropicNormal>
    with isotropic and isotropic_unchecked
);

impl_build_normal!(
    BuildNormal<Vec<f64>, Vec<f64>, Normal = mv_normal::DiagonalNormal>
    with diagonal and diagonal_unchecked
);

impl_build_normal!(
    BuildNormal<Vector<f64>, Vector<f64>, Normal = mv_normal::DiagonalNormal>
    with diagonal and diagonal_unchecked
);

impl_build_normal!(
    BuildNormal<Vec<f64>, Matrix<f64>, Normal = mv_normal::Normal>
    with new and new_unchecked
);

impl_build_normal!(
    BuildNormal<Vector<f64>, Matrix<f64>, Normal = mv_normal::Normal>
    with new and new_unchecked
);
