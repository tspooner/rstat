use crate::{statistics::Modes, DiscreteDistribution, Distribution, Probability, SimplexVector};
use rand::Rng;
use spaces::intervals::Closed;
use std::fmt;

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Params<const N: usize> {
    pub ps: SimplexVector<N>,
}

impl<const N: usize> Params<N> {
    pub fn new(ps: [f64; N]) -> Result<Params<N>, failure::Error> {
        Ok(Params {
            ps: SimplexVector::new(ps)?
        })
    }

    pub fn new_unchecked(ps: [f64; N]) -> Params<N> {
        Params {
            ps: SimplexVector::new_unchecked(ps),
        }
    }
}

pub type Multinoulli<const N: usize> = Categorical<N>;

#[derive(Debug, Clone)]
pub struct Categorical<const N: usize>(Params<N>);

impl<const N: usize> Categorical<N> {
    pub fn new(ps: [f64; N]) -> Result<Categorical<N>, failure::Error> {
        Params::new(ps).map(Categorical)
    }

    pub fn new_unchecked(ps: [f64; N]) -> Categorical<N> { Params::new_unchecked(ps).into() }

    pub const fn n_categories() -> usize { N }
}

impl<const N: usize> From<Params<N>> for Categorical<N> {
    fn from(params: Params<N>) -> Categorical<N> { Categorical(params) }
}

impl<const N: usize> Distribution for Categorical<N> {
    type Support = Closed<usize>;
    type Params = Params<N>;

    fn support(&self) -> Closed<usize> { Closed::closed_unchecked(0, N + 1) }

    fn params(&self) -> Params<N> { self.0.clone() }

    fn cdf(&self, x: &usize) -> Probability {
        Probability::new_unchecked(self.0.ps.iter().take(*x).sum())
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize { self.0.ps.sample_index(rng) }
}

impl<const N: usize> DiscreteDistribution for Categorical<N> {
    fn pmf(&self, i: &usize) -> Probability {
        let i = *i;

        if i >= N {
            panic!("Index must lie in the support: i < k.")
        }

        Probability::new_unchecked(self.0.ps[i])
    }
}

impl<const N: usize> Modes for Categorical<N> {
    fn modes(&self) -> Vec<usize> {
        self.0
            .ps
            .iter()
            .enumerate()
            .skip(1)
            .fold((vec![0], self.0.ps[0]), |(mut modes, pmax), (j, p)| {
                use std::cmp::Ordering::*;

                match p.partial_cmp(&pmax) {
                    Some(Less) => (modes, pmax),
                    Some(Equal) => {
                        modes.push(j);

                        (modes, pmax)
                    },
                    Some(Greater) => {
                        modes.truncate(1);
                        modes[0] = j;

                        (modes, *p)
                    },
                    None => unreachable!(),
                }
            })
            .0
    }
}

impl<const N: usize> fmt::Display for Categorical<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Cat({:?})", self.0.ps) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unimodal() {
        fn check<const N: usize>(ps: [f64; N], idx: usize) {
            let d = Categorical::new(ps).unwrap();
            let ms = d.modes();

            assert!(ms.len() == 1);
            assert!(ms[0] == idx);
        }

        check([0.5, 0.25, 0.25], 0);
        check([0.25, 0.5, 0.25], 1);
        check([0.25, 0.25, 0.5], 2);
    }

    #[test]
    fn test_bimodal() {
        let d = Categorical::new([0.4, 0.4, 0.1, 0.1]).unwrap();
        let ms = d.modes();

        assert!(ms.len() == 2);
        assert!(ms[0] == 0 || ms[0] == 1);
        assert!(ms[1] == 0 || ms[1] == 1);
    }
}
