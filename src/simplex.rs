use crate::Probability;
use failure::Fail;
use std::{ops, result::Result as _Result};
use rand::Rng;

pub type Result<T> = _Result<T, SimplexError>;

#[derive(Clone, Debug, Fail)]
pub enum SimplexError {
    #[fail(display="Probabilities must sum to 1.")]
    Unnormalised,
}

#[derive(Clone, Debug)]
pub struct Simplex(Vec<Probability>);

impl Simplex {
    pub fn new(ps: &[Probability]) -> Result<Simplex> {
        let z: f64 = ps.iter().map(|p| p.unwrap()).sum();

        if (z - 1.0).abs() < 1e-5 {
            Ok(Simplex(Vec::from(ps)))
        } else {
            Err(SimplexError::Unnormalised)
        }
    }

    pub fn normalised(ps: &[f64]) -> Simplex {
        let z: f64 = ps.iter().sum();

        Simplex(ps.iter().map(|p| Probability::new_unchecked(p / z)).collect())
    }

    pub fn equiprobable(n: usize) -> Simplex {
        Simplex(vec![Probability::new_unchecked(1.0 / n as f64); n])
    }

    pub fn new_unchecked<I: IntoIterator<Item = Probability>>(ps: I) -> Simplex {
        Simplex(ps.into_iter().collect())
    }

    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        let mut cdf = self.0.iter().scan(0.0, |state, p| {
            *state += p.unwrap();

            Some(*state)
        });
        let rval: f64 = rng.gen();

        cdf.position(|p| rval < p).unwrap_or(self.len() - 1)
    }
}

impl ops::Deref for Simplex {
    type Target = [Probability];

    fn deref(&self) -> &[Probability] { self.0.deref() }
}

#[cfg(test)]
mod tests {
    use rand::thread_rng;
    use std::iter::{repeat, once};
    use super::{Probability, Simplex};

    #[test]
    fn test_sample_degenerate() {
        let mut rng = thread_rng();

        let s = Simplex::new(&[Probability::one()]).unwrap();

        assert_eq!(s.sample(&mut rng), 0);
    }

    #[test]
    fn test_sample_degenerate_n() {
        let mut rng = thread_rng();

        fn make_simplex(idx: usize, n: usize) -> Simplex {
            Simplex::new_unchecked(
                repeat(Probability::zero()).take(idx)
                    .chain(once(Probability::one()))
                    .chain(repeat(Probability::zero()).take(n - idx - 1))
            )
        }

        assert_eq!(make_simplex(0, 5).sample(&mut rng), 0);
        assert_eq!(make_simplex(1, 5).sample(&mut rng), 1);
        assert_eq!(make_simplex(2, 5).sample(&mut rng), 2);
        assert_eq!(make_simplex(3, 5).sample(&mut rng), 3);
        assert_eq!(make_simplex(4, 5).sample(&mut rng), 4);
    }
}
