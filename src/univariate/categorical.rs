use crate::prelude::*;
use rand::Rng;
use spaces::discrete::Ordinal;
use std::fmt;

pub type Multinoulli = Categorical;

#[derive(Debug, Clone)]
pub struct Categorical {
    pub ps: Vec<Probability>,
}

impl Categorical {
    pub fn new<P: Into<Probability>>(ps: Vec<P>) -> Categorical {
        Categorical {
            ps: Probability::normalised(ps)
        }
    }

    pub fn equiprobable(n: usize) -> Categorical {
        Categorical::new(vec![1.0 / n as f64; n])
    }

    pub fn n_categories(&self) -> usize {
        self.ps.len()
    }
}

impl<P: Into<Probability>> From<Vec<P>> for Categorical {
    fn from(ps: Vec<P>) -> Categorical {
        Categorical::new(ps)
    }
}

impl Distribution for Categorical {
    type Support = Ordinal;

    fn support(&self) -> Ordinal { Ordinal::new(self.ps.len() as usize) }

    fn cdf(&self, _: usize) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> usize {
        unimplemented!()
    }
}

impl DiscreteDistribution for Categorical {
    fn pmf(&self, i: usize) -> Probability {
        if i > self.ps.len() {
            panic!("Index must lie in the support: i < k.")
        }

        self.ps[i]
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
