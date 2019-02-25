use crate::{
    core::*,
    discrete::Categorical,
};
use rand::Rng;
use spaces::{Space, Vector};

#[derive(Debug, Clone)]
pub struct Mixture<C> {
    pub dist: Categorical,
    pub components: Vec<C>,
}

impl<C> Mixture<C> {
    pub fn new<T: Into<Categorical>>(prior: T, components: Vec<C>) -> Mixture<C> {
        let prior: Categorical = prior.into();

        if components.len() != prior.n_categories() {
            panic!("Number of components must match the number of assigned probabilities.")
        }

        Mixture {
            dist: prior.into(),
            components,
        }
    }

    pub fn homogeneous(components: Vec<C>) -> Mixture<C> {
        let n_components = components.len();

        Mixture::new(Categorical::equiprobable(n_components), components)
    }
}

impl<C> Mixture<C> {
    pub fn n_components(&self) -> usize {
        self.components.len()
    }
}

impl<C: Distribution> Distribution for Mixture<C> {
    type Support = C::Support;

    fn support(&self) -> Self::Support {
        unimplemented!()
    }

    fn cdf(&self, _: <Self::Support as Space>::Value) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> <Self::Support as Space>::Value {
        unimplemented!()
    }
}

impl<C: ContinuousDistribution> ContinuousDistribution for Mixture<C>
where
    <Self::Support as Space>::Value: Clone
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

impl<C: UnivariateMoments> UnivariateMoments for Mixture<C> {
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
        Uniform,
        core::UnivariateMoments,
    };
    use super::Mixture;

    #[test]
    fn test_uniform_pair_mixture() {
        let uniform = Uniform::<f64>::new(0.0, 2.0);
        let mixture = Mixture::homogeneous(vec![
            Uniform::<f64>::new(0.0, 1.0),
            Uniform::<f64>::new(1.0, 2.0),
        ]);

        assert!((uniform.mean() - mixture.mean()).abs() < 1e-7);
        assert!((uniform.variance() - mixture.variance()).abs() < 1e-7);
    }
}
