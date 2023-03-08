use crate::{
    statistics::UvMoments,
    ContinuousDistribution,
    Distribution,
    Probability,
    SimplexVector,
    Univariate,
};
use failure::{Error, Fail};
use rand::Rng;
use spaces::{Space, ops::{Union, Closure}};

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Params<const N: usize, CP> {
    pub ps: SimplexVector<N>,
    pub component_params: Vec<CP>,
}

#[derive(Clone, Debug, Fail)]
pub enum MixtureError {
    #[fail(display = "Incompatible parameters.")]
    IncompatibleParameters,
}

/// Probability [distribution](struct.Distribution.html) derived from a linear
/// sum of random variables.
///
/// # Examples
/// ```
/// # use failure::Error;
/// use rstat::{
///     Probability, Mixture,
///     univariate::{normal::Normal, Moments}
/// };
///
/// let gmm = Mixture::new(
///     [0.2, 0.5, 0.3],
///     [
///         Normal::new(-2.0, 1.2_f64.powi(2))?,
///         Normal::new(0.0, 1.0_f64.powi(2))?,
///         Normal::new(3.0, 2.5_f64.powi(2))?,
///     ],
/// )?;
///
/// assert!((gmm.mean() - 0.5).abs() < 1e-7);
/// assert!((gmm.variance() - 5.913).abs() < 1e-7);
/// assert!((gmm.standard_deviation() - gmm.variance().sqrt()).abs() < 1e-7);
///
/// # Ok::<(), failure::Error>(())
/// ```
#[derive(Debug, Clone)]
pub struct Mixture<const N: usize, C: Distribution> {
    /// The weights of the linear sum.
    pub weights: SimplexVector<N>,

    /// The [distribution](trait.Distribution.html) components of the linear
    /// sum.
    pub components: [C; N],

    support: C::Support,
}

impl<const N: usize, C: Distribution> Mixture<N, C>
where
    C::Support: Union<C::Support>,
    <C::Support as Union<C::Support>>::Output: Closure<Output = C::Support>,
{
    pub fn new(weights: [f64; N], components: [C; N]) -> Result<Mixture<N, C>, Error> {
        if components.len() != weights.len() {
            Err(MixtureError::IncompatibleParameters)?
        } else {
            Ok(Mixture {
                support: Self::compute_support(&components),

                weights: SimplexVector::new(weights)?,
                components,
            })
        }
    }

    pub fn new_unchecked(weights: [f64; N], components: [C; N]) -> Mixture<N, C> {
        Mixture {
            support: Self::compute_support(&components),

            weights: SimplexVector::new_unchecked(weights),
            components,
        }
    }

    pub fn n_components() -> usize { N }

    fn compute_support(components: &[C; N]) -> C::Support {
        components
            .iter()
            .skip(1)
            .fold(components[0].support(), |acc, c| acc.union_closure(c.support()))
    }
}

impl<const N: usize, C: Distribution> From<Params<N, C::Params>> for Mixture<N, C>
where
    C::Support: Union<C::Support>,
    <C::Support as Union<C::Support>>::Output: Closure<Output = C::Support>,
{
    fn from(params: Params<N, C::Params>) -> Mixture<N, C> {
        let mut components: [std::mem::MaybeUninit<C>; N] = unsafe {
            std::mem::MaybeUninit::uninit().assume_init()
        };

        params.component_params.into_iter().map(|c| c.into()).enumerate().for_each(|(i, c)| {
            components[i].write(c);
        });

        let components: [C; N] = unsafe { (&components as *const _ as *const [C; N]).read() };

        Mixture {
            weights: params.ps,
            support: Self::compute_support(&components),
            components,
        }
    }
}

impl<const N: usize, C: Distribution> Distribution for Mixture<N, C>
where
    C::Support: Union<C::Support> + Clone,
    <C::Support as Union<C::Support>>::Output: Closure<Output = C::Support>,
{
    type Support = C::Support;
    type Params = Params<N, C::Params>;

    fn support(&self) -> Self::Support { self.support.clone() }

    fn params(&self) -> Params<N, C::Params> {
        Params {
            ps: self.weights.clone(),
            component_params: self.components.iter().map(|c| c.params()).collect(),
        }
    }

    fn cdf(&self, x: &<Self::Support as Space>::Value) -> Probability {
        let p = self
            .components
            .iter()
            .zip(self.weights.iter())
            .map(|(c, p)| p * c.cdf(x))
            .sum();

        Probability::new_unchecked(p)
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self::Support as Space>::Value {
        self.components[self.weights.sample_index(rng)].sample(rng)
    }
}

impl<const N: usize, C: Distribution> Univariate for Mixture<N, C>
where
    C::Support: Union<C::Support> + Clone,
    <C::Support as Union<C::Support>>::Output: Closure<Output = C::Support>,
{}

impl<const N: usize, C: ContinuousDistribution> ContinuousDistribution for Mixture<N, C>
where
    C::Support: Union<C::Support> + Clone,
    <C::Support as Union<C::Support>>::Output: Closure<Output = C::Support>,
{
    fn pdf(&self, x: &<Self::Support as Space>::Value) -> f64 {
        self.components
            .iter()
            .zip(self.weights.iter())
            .map(|(c, p)| p * c.pdf(x))
            .sum()
    }
}

impl<const N: usize, C: UvMoments> UvMoments for Mixture<N, C>
where
    C::Support: Union<C::Support> + Clone,
    <C::Support as Union<C::Support>>::Output: Closure<Output = C::Support>,
{
    fn mean(&self) -> f64 {
        self.components
            .iter()
            .zip(self.weights.iter())
            .fold(0.0, |acc, (c, &w)| acc + w * c.mean())
    }

    fn variance(&self) -> f64 {
        let (mean, var_term) =
            self.components
                .iter()
                .zip(self.weights.iter())
                .fold((0.0, 0.0), |acc, (c, &p)| {
                    let c_mean = c.mean();
                    let c_variance = c.variance();

                    (
                        acc.0 + p * c_mean,
                        acc.1 + p * (c_mean * c_mean + c_variance),
                    )
                });

        var_term - mean * mean
    }

    fn skewness(&self) -> f64 { todo!() }

    fn kurtosis(&self) -> f64 { todo!() }
}

#[cfg(test)]
mod tests {
    use crate::univariate::uniform::Uniform;
    use super::*;
    use spaces::intervals::Interval;
    use failure::Error;

    #[test]
    fn test_uniform_pair_mixture() -> Result<(), Error> {
        let uniform = Uniform::<f64>::new(0.0, 2.0)?;
        let mixture = Mixture::new(
            [0.5, 0.5],
            [
                Uniform::<f64>::new(0.0, 1.0)?,
                Uniform::<f64>::new(1.0, 1.0)?,
            ],
        )?;

        assert_eq!(uniform.support(), Interval::closed_unchecked(0.0, 2.0));

        assert!((uniform.mean() - mixture.mean()).abs() < 1e-7);
        assert!((uniform.variance() - mixture.variance()).abs() < 1e-7);

        Ok(())
    }
}
