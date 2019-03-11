use crate::{
    core::*,
    univariate::discrete::Categorical,
};
use rand::Rng;
use spaces::{Space, Enclose};

#[derive(Debug, Clone)]
pub struct Mixture<C: Distribution> {
    pub dist: Categorical,
    pub components: Vec<C>,

    support: C::Support,
}

impl<C: Distribution> Mixture<C>
where
    C::Support: Enclose,
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
        components.iter().skip(1)
            .fold(components[0].support(), |acc, c| acc.enclose(&c.support()))
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
            .fold(Probability::zero(), |acc, (c, p)| acc + *p * c.cdf(x.clone()))
            .into()
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> <Self::Support as Space>::Value {
        self.components[self.dist.sample(rng)].sample(rng)
    }
}

impl<C: ContinuousDistribution> ContinuousDistribution for Mixture<C>
where
    C::Support: Clone,
    <C::Support as Space>::Value: Clone,
{
    fn pdf(&self, x: <Self::Support as Space>::Value) -> Probability {
        Probability::new_unchecked(self.components.iter()
            .zip(self.dist.ps.iter())
            .filter_map(|(c, &p)| if p.non_zero() {
                Some(f64::from(p * c.pdf(x.clone())))
            } else {
                None
            })
            .sum()
        )
    }
}

impl<C: UnivariateMoments> UnivariateMoments for Mixture<C>
where
    C::Support: Clone,
{
    fn mean(&self) -> f64 {
        self.components.iter()
            .zip(self.dist.ps.iter())
            .filter_map(|(c, &p)| if p.non_zero() {
                Some(f64::from(p) * c.mean())
            } else {
                None
            })
            .sum()
    }

    fn variance(&self) -> f64 {
        let mean = self.mean();
        let var_term: f64 = self.components.iter()
            .zip(self.dist.ps.iter())
            .filter_map(|(c, &p)| if p.non_zero() {
                let c_mean = c.mean();
                let c_variance = c.variance();

                Some(f64::from(p) * (c_mean * c_mean + c_variance))
            } else {
                None
            })
            .sum();

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
    use crate::{
        core::UnivariateMoments,
        univariate::continuous::{Normal, Uniform},
    };
    use super::Mixture;

    #[test]
    fn test_uniform_pair_mixture() {
        let uniform = Uniform::new(0.0, 2.0);
        let mixture = Mixture::homogeneous(vec![
            Uniform::new(0.0, 1.0),
            Uniform::new(1.0, 2.0),
        ]);

        assert!((uniform.mean() - mixture.mean()).abs() < 1e-7);
        assert!((uniform.variance() - mixture.variance()).abs() < 1e-7);
    }

    #[test]
    fn test_gmm() {
        let gmm = Mixture::new(
            vec![0.2, 0.5, 0.3],
            vec![Normal::new(-2.0, 1.2), Normal::new(0.0, 1.0), Normal::new(3.0, 2.5)],
        );

        assert!((gmm.mean() - 0.5).abs() < 1e-7);
        assert!((gmm.variance() - 5.913).abs() < 1e-7);
        assert!((gmm.standard_deviation() - gmm.variance().sqrt()).abs() < 1e-7);
    }
}
