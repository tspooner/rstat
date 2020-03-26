use crate::{prelude::*, univariate::Categorical};
use rand::Rng;
use spaces::{Space, Union};

#[derive(Debug, Clone)]
pub struct Mixture<C: Distribution> {
    pub dist: Categorical,
    pub components: Vec<C>,

    support: C::Support,
}

impl<C: Distribution> Mixture<C>
where
    C::Support: Union,
{
    pub fn new<T: Into<Categorical>>(prior: T, components: Vec<C>) -> Mixture<C> {
        let prior: Categorical = prior.into();

        if components.len() != prior.n_categories() {
            panic!("Number of components must match the number of assigned probabilities.")
        }

        let support = Self::compute_support(&components);

        Mixture {
            dist: prior.into(),
            components,

            support,
        }
    }

    pub fn homogeneous(components: Vec<C>) -> Mixture<C> {
        let n_components = components.len();

        Mixture::new(Categorical::equiprobable(n_components), components)
    }

    fn compute_support(components: &[C]) -> C::Support {
        components.iter().skip(1).fold(components[0].support(), |acc, c| acc.union(&c.support()))
    }
}

impl<C: Distribution> Mixture<C> {
    pub fn n_components(&self) -> usize {
        self.components.len()
    }
}

impl<C: Distribution> Distribution for Mixture<C>
where
    C::Support: Clone,
{
    type Support = C::Support;

    fn support(&self) -> Self::Support {
        self.support.clone()
    }

    fn cdf(&self, x: <Self::Support as Space>::Value) -> Probability {
        self.components.iter().zip(self.dist.ps.iter())
            .fold(Probability::zero(), |acc, (c, &p)| acc + p * c.cdf(x.clone()))
            .into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self::Support as Space>::Value {
        self.components[self.dist.sample(rng)].sample(rng)
    }
}

impl<C: ContinuousDistribution> ContinuousDistribution for Mixture<C>
where
    C::Support: Clone,
{
    fn pdf(&self, x: <Self::Support as Space>::Value) -> f64 {
        self.components.iter()
            .zip(self.dist.ps.iter())
            .fold(0.0, |acc, (c, &p)| acc + p * c.pdf(x.clone()))
    }
}

impl<C: UnivariateMoments> UnivariateMoments for Mixture<C>
where
    C::Support: Clone,
{
    fn mean(&self) -> f64 {
        self.components.iter()
            .zip(self.dist.ps.iter())
            .fold(0.0, |acc, (c, &p)| acc + p * c.mean())
    }

    fn variance(&self) -> f64 {
        let (mean, var_term) = self.components.iter()
            .zip(self.dist.ps.iter())
            .fold((0.0, 0.0), |acc, (c, &p)| {
                let c_mean = c.mean();
                let c_variance = c.variance();

                (acc.0 + p * c_mean, acc.1 + p * (c_mean * c_mean + c_variance))
            });

        var_term - mean * mean
    }

    fn skewness(&self) -> f64 {
        unimplemented!()
    }

    fn kurtosis(&self) -> f64 {
        unimplemented!()
    }
}

#[cfg(test)]
mod tests {
    use crate::{prelude::*, univariate::{Normal, Uniform, Categorical}, validation::Result};
    use super::Mixture;

    #[test]
    fn test_uniform_pair_mixture() -> Result<()> {
        let uniform = Uniform::<f64>::new(0.0, 2.0)?;
        let mixture = Mixture::homogeneous(vec![
            Uniform::<f64>::new(0.0, 1.0)?,
            Uniform::<f64>::new(1.0, 2.0)?,
        ]);

        assert!((uniform.mean() - mixture.mean()).abs() < 1e-7);
        assert!((uniform.variance() - mixture.variance()).abs() < 1e-7);

        Ok(())
    }

    #[test]
    fn test_gmm() -> Result<()> {
        let gmm = Mixture::new(
            Categorical::new(vec![0.2, 0.5, 0.3])?,
            vec![Normal::new(-2.0, 1.2)?, Normal::new(0.0, 1.0)?, Normal::new(3.0, 2.5)?],
        );

        println!("{:?}", gmm.variance());

        assert!((gmm.mean() - 0.5).abs() < 1e-7);
        assert!((gmm.variance() - 5.913).abs() < 1e-7);
        assert!((gmm.standard_deviation() - gmm.variance().sqrt()).abs() < 1e-7);

        Ok(())
    }
}
