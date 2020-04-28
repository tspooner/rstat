use crate::{
    statistics::UnivariateMoments,
    ContinuousDistribution,
    Distribution,
    Probability,
    SimplexVector,
};
use failure::{Error, Fail};
use rand::Rng;
use spaces::{Space, Union};

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Params<CP> {
    pub ps: SimplexVector,
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
///     vec![0.2, 0.5, 0.3],
///     vec![
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
pub struct Mixture<C: Distribution> {
    /// The weights of the linear sum.
    pub weights: SimplexVector,

    /// The [distribution](trait.Distribution.html) components of the linear
    /// sum.
    pub components: Vec<C>,

    support: C::Support,
}

impl<C: Distribution> Mixture<C> {
    pub fn new(weights: Vec<f64>, components: Vec<C>) -> Result<Mixture<C>, Error>
    where C::Support: Union {
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

    pub fn new_unchecked(weights: Vec<f64>, components: Vec<C>) -> Mixture<C>
    where C::Support: Union {
        Mixture {
            support: Self::compute_support(&components),

            weights: SimplexVector::new_unchecked(weights),
            components,
        }
    }

    pub fn n_components(&self) -> usize { self.components.len() }

    fn compute_support(components: &[C]) -> C::Support
    where C::Support: Union {
        components
            .iter()
            .skip(1)
            .fold(components[0].support(), |acc, c| acc.union(&c.support()))
    }
}

impl<C: Distribution> From<Params<C::Params>> for Mixture<C>
where C::Support: Union
{
    fn from(params: Params<C::Params>) -> Mixture<C> {
        let components: Vec<C> = params
            .component_params
            .into_iter()
            .map(|cp| cp.into())
            .collect();

        Mixture {
            weights: params.ps,
            support: Self::compute_support(&components),
            components,
        }
    }
}

impl<C: Distribution> Distribution for Mixture<C>
where C::Support: Union + Clone
{
    type Support = C::Support;
    type Params = Params<C::Params>;

    fn support(&self) -> Self::Support { self.support.clone() }

    fn params(&self) -> Params<C::Params> {
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

impl<C: ContinuousDistribution> ContinuousDistribution for Mixture<C>
where C::Support: Union + Clone
{
    fn pdf(&self, x: &<Self::Support as Space>::Value) -> f64 {
        self.components
            .iter()
            .zip(self.weights.iter())
            .map(|(c, p)| p * c.pdf(x))
            .sum()
    }
}

impl<C: UnivariateMoments> UnivariateMoments for Mixture<C>
where C::Support: Union + Clone
{
    fn mean(&self) -> f64 {
        self.components
            .iter()
            .zip(self.weights.iter())
            .fold(0.0, |acc, (c, &p)| acc + p * c.mean())
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
    use super::Mixture;
    use crate::{statistics::UnivariateMoments, univariate::uniform::Uniform};
    use failure::Error;

    #[test]
    fn test_uniform_pair_mixture() -> Result<(), Error> {
        let uniform = Uniform::<f64>::new(0.0, 2.0)?;
        let mixture = Mixture::new(
            vec![0.5, 0.5],
            vec![
                Uniform::<f64>::new(0.0, 1.0)?,
                Uniform::<f64>::new(1.0, 1.0)?,
            ],
        )?;

        assert!((uniform.mean() - mixture.mean()).abs() < 1e-7);
        assert!((uniform.variance() - mixture.variance()).abs() < 1e-7);

        Ok(())
    }
}
