use crate::{statistics::Modes, DiscreteDistribution, Distribution, Probability, SimplexVector};
use rand::Rng;
use spaces::discrete::Ordinal;
use std::fmt;

params! {
    Params {
        ps: SimplexVector<>
    }
}

pub type Multinoulli = Categorical;

#[derive(Debug, Clone)]
pub struct Categorical(Params);

impl Categorical {
    pub fn new(ps: Vec<f64>) -> Result<Categorical, failure::Error> {
        Params::new(ps).map(Categorical)
    }

    pub fn new_unchecked(ps: Vec<f64>) -> Categorical { Params::new_unchecked(ps).into() }

    pub fn n_categories(&self) -> usize { self.0.ps.len() }
}

impl From<Params> for Categorical {
    fn from(params: Params) -> Categorical { Categorical(params) }
}

impl Distribution for Categorical {
    type Support = Ordinal;
    type Params = Params;

    fn support(&self) -> Ordinal { Ordinal::new(self.0.ps.len() as usize) }

    fn params(&self) -> Params { self.0.clone() }

    fn cdf(&self, x: &usize) -> Probability {
        Probability::new_unchecked(self.0.ps.iter().take(*x).sum())
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> usize { self.0.ps.sample_index(rng) }
}

impl DiscreteDistribution for Categorical {
    fn pmf(&self, i: &usize) -> Probability {
        let i = *i;

        if i > self.0.ps.len() {
            panic!("Index must lie in the support: i < k.")
        }

        Probability::new_unchecked(self.0.ps[i])
    }
}

impl Modes for Categorical {
    fn modes(&self) -> Vec<usize> {
        self.0
            .ps
            .iter()
            .enumerate()
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

impl fmt::Display for Categorical {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Cat({:?})", self.0.ps) }
}
