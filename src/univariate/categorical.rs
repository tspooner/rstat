use crate::prelude::*;
use rand::Rng;
use spaces::discrete::Ordinal;
use std::fmt;

pub type Multinoulli = Categorical;

#[derive(Debug, Clone)]
pub struct Categorical {
    pub ps: SimplexVector,
}

impl Categorical {
    pub fn new(ps: SimplexVector) -> Categorical {
        Categorical { ps, }
    }

    pub fn n_categories(&self) -> usize {
        self.ps.len()
    }
}

impl From<SimplexVector> for Categorical {
    fn from(ps: SimplexVector) -> Categorical {
        Categorical::new(ps)
    }
}

impl Distribution for Categorical {
    type Support = Ordinal;
    type Params = SimplexVector;

    fn support(&self) -> Ordinal { Ordinal::new(self.ps.len() as usize) }

    fn params(&self) -> SimplexVector { self.ps.clone() }

    fn cdf(&self, x: &usize) -> Probability {
        Probability::new_unchecked(self.ps.iter().take(*x).sum())
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize {
        self.ps.sample_index(rng)
    }
}

impl DiscreteDistribution for Categorical {
    fn pmf(&self, i: &usize) -> Probability {
        let i = *i;

        if i > self.ps.len() {
            panic!("Index must lie in the support: i < k.")
        }

        Probability::new_unchecked(self.ps[i])
    }
}

impl Modes for Categorical {
    fn modes(&self) -> Vec<usize> {
        self.ps.iter().enumerate().fold((vec![0], self.ps[0]), |(mut modes, pmax), (j, p)| {
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
        }).0
    }
}

impl fmt::Display for Categorical {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Cat({:?})", self.ps)
    }
}
