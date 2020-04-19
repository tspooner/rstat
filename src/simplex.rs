use crate::{
    params::{constraints::Constraints, Param},
    univariate::uniform::Uniform,
    Distribution,
    Probability,
};
use failure::Fail;
use rand::Rng;
use std::ops;

/// Utility for sampling from a unit \\(K\\)-simplex.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct UnitSimplex(usize);

impl UnitSimplex {
    /// Construct a \\(K = n + 1\\) probability simplex.
    pub fn new(n: usize) -> UnitSimplex { UnitSimplex(n + 1) }

    /// Compute the central point of the simplex \\(x_i = 1 / K\\).
    ///
    /// # Examples
    /// ```
    /// use rstat::UnitSimplex;
    ///
    /// let s = UnitSimplex::new(2).centre();
    ///
    /// for i in 0..3 {
    ///     assert_eq!(s[i], 1.0 / 3.0);
    /// }
    /// ```
    pub fn centre(&self) -> SimplexVector {
        let p = 1.0 / self.0 as f64;

        SimplexVector::new_unchecked(std::iter::repeat(p).take(self.0))
    }

    /// Draws a uniformly random point on the simplex.
    ///
    /// This algorithm works as follows:
    ///
    /// 1. Draw \\(K\\) independent points, \\(x_i \in [0, 1]\\), uniformly at
    /// random. 2. Apply the transformation \\(z_i = -\ln{x_i}\\).
    /// 3. Compute the sum \\(s = \sum_i x_i\\).
    /// 4. Return the vector of values \\(z_i / s\\).
    ///
    /// # Examples
    /// ```
    /// use rand::thread_rng;
    /// use rstat::UnitSimplex;
    ///
    /// let s = UnitSimplex::new(2).sample(&mut thread_rng());
    ///
    /// assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-7);
    /// ```
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> SimplexVector {
        let mut sum = 0.0;

        let vs: Vec<f64> = Uniform::<f64>::new_unchecked(0.0, 1.0)
            .sample_iter(rng)
            .map(|s| -s.ln())
            .inspect(|v| sum += v)
            .take(self.0)
            .collect();

        SimplexVector(vs.into_iter().map(|v| v / sum).collect())
    }
}

#[derive(Clone, Debug, Fail)]
pub enum SimplexError {
    #[fail(display = "Probabilities in SimplexVector must sum to 1.")]
    Unnormalised,
}

/// Probability vector constrainted to the [unit
/// simplex](struct.UnitSimplex.html).
#[derive(Clone, Debug)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct SimplexVector(Vec<f64>);

impl SimplexVector {
    /// Construct a new probability vector on the [unit
    /// simplex](struct.UnitSimplex.html).
    pub fn new(ps: Vec<f64>) -> Result<SimplexVector, failure::Error> {
        std::convert::TryFrom::try_from(ps)
    }

    /// Construct a new probability vector without enforcing constraints.
    pub fn new_unchecked<I>(ps: I) -> SimplexVector
    where I: IntoIterator<Item = f64> {
        SimplexVector(ps.into_iter().collect())
    }

    /// Sample a probability-weighted random index from the vector.
    pub fn sample_index<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut cdf = self.0.iter().scan(0.0, |state, p| {
            *state += p;

            Some(*state)
        });
        let rval: f64 = rng.gen();

        cdf.position(|p| rval < p).unwrap_or(self.len() - 1)
    }
}

impl ops::Deref for SimplexVector {
    type Target = [f64];

    fn deref(&self) -> &[f64] { self.0.deref() }
}

impl std::convert::TryFrom<Vec<f64>> for SimplexVector {
    type Error = failure::Error;

    fn try_from(ps: Vec<f64>) -> Result<SimplexVector, failure::Error> {
        let mut z: f64 = 0.0;

        for p in ps.iter() {
            z += Probability::new(*p)?.0;
        }

        if (z - 1.0).abs() < 1e-5 {
            Ok(SimplexVector(ps))
        } else {
            Err(SimplexError::Unnormalised)?
        }
    }
}

impl Param for SimplexVector {
    type Value = Vec<f64>;

    fn value(&self) -> &Vec<f64> { &self.0 }

    fn constraints() -> Constraints<Vec<f64>> { vec![] }
}

#[cfg(test)]
mod tests {
    use super::{Probability, SimplexVector};
    use rand::thread_rng;
    use std::iter::{once, repeat};

    #[test]
    fn test_sample_index_degenerate() {
        let mut rng = thread_rng();

        let s = SimplexVector::new(vec![1.0]).unwrap();

        assert_eq!(s.sample_index(&mut rng), 0);
    }

    #[test]
    fn test_sample_index_degenerate_n() {
        let mut rng = thread_rng();

        fn make_simplex(idx: usize, n: usize) -> SimplexVector {
            SimplexVector::new_unchecked(
                repeat(0.0)
                    .take(idx)
                    .chain(once(1.0))
                    .chain(repeat(0.0).take(n - idx - 1)),
            )
        }

        assert_eq!(make_simplex(0, 5).sample_index(&mut rng), 0);
        assert_eq!(make_simplex(1, 5).sample_index(&mut rng), 1);
        assert_eq!(make_simplex(2, 5).sample_index(&mut rng), 2);
        assert_eq!(make_simplex(3, 5).sample_index(&mut rng), 3);
        assert_eq!(make_simplex(4, 5).sample_index(&mut rng), 4);
    }
}
