use crate::{params::Count, prelude::*};
use failure::Error;
use ndarray::{Array1, Array2};
use rand::Rng;
use spaces::{ProductSpace, discrete::Ordinal};
use std::fmt;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct Params {
    pub n: Count<usize>,
    pub ps: SimplexVector,
}

impl Params {
    pub fn new(n: usize, ps: Vec<f64>) -> Result<Params, failure::Error> {
        Ok(Params {
            n: Count::new(n)?,
            ps: SimplexVector::new(ps)?,
        })
    }

    pub fn new_unchecked(n: usize, ps: Vec<f64>) -> Params {
        Params {
            n: Count(n),
            ps: SimplexVector::new_unchecked(ps),
        }
    }

    pub fn n(&self) -> &Count<usize> { &self.n }

    pub fn ps(&self) -> &SimplexVector { &self.ps }
}

#[derive(Debug, Clone)]
pub struct Multinomial(Params);

impl Multinomial {
    pub fn new(n: usize, ps: Vec<f64>) -> Result<Multinomial, Error> {
        let params = Params::new(n, ps)?;

        Ok(Multinomial(params))
    }


    pub fn new_unchecked(n: usize, ps: Vec<f64>) -> Multinomial {
        Multinomial(Params::new_unchecked(n, ps))
    }
}

impl Multinomial {
    #[inline]
    pub fn n_categories(&self) -> usize {
        self.0.ps.len()
    }
}

impl From<Params> for Multinomial {
    fn from(params: Params) -> Multinomial {
        Multinomial(params)
    }
}

impl Distribution for Multinomial {
    type Support = ProductSpace<Ordinal>;
    type Params = Params;

    fn support(&self) -> ProductSpace<Ordinal> {
        std::iter::repeat(Ordinal::new(self.0.n.0))
            .take(self.0.ps.len())
            .collect()
    }

    fn params(&self) -> Params { self.0.clone() }

    fn cdf(&self, _: &Vec<usize>) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> Vec<usize> {
        unimplemented!()
    }
}

impl DiscreteDistribution for Multinomial {
    fn pmf(&self, xs: &Vec<usize>) -> Probability {
        match xs.iter().fold(0, |acc, x| acc + x) {
            0 => Probability::zero(),
            _ => Probability::new(self.log_pmf(xs).exp()).unwrap(),
        }
    }

    fn log_pmf(&self, xs: &Vec<usize>) -> f64 {
        let xn = xs.len();

        #[allow(unused_parens)]
        let _ = assert_constraint!(xn == (self.0.ps.len())).unwrap();

        if xs.iter().fold(0, |acc, v| acc + *v) == self.0.n.0 {
            panic!("Total number of trials must be equal to n.")
        }

        use special_fun::FloatSpecial;

        let term_1 = (self.0.n.0 as f64 + 1.0).loggamma();
        let term_2 = xs.iter().zip(self.0.ps.iter()).fold(0.0, |acc, (&x, &p)| {
            let x_f64 = x as f64;
            let xlogy = if x == 0 {
                0.0
            } else {
                x_f64 * p.ln()
            };

            acc + xlogy - (x_f64 + 1.0).loggamma()
        });

        (term_1 + term_2).into()
    }
}

impl MultivariateMoments for Multinomial {
    fn mean(&self) -> Array1<f64> {
        self.0.ps.iter().map(|&p| p * self.0.n.0 as f64).collect()
    }

    fn variance(&self) -> Array1<f64> {
        self.0.ps.iter().map(|&p| {
            (p * (1.0 - p)) * self.0.n.0 as f64
        }).collect()
    }

    fn covariance(&self) -> Array2<f64> {
        let n = self.0.n.0 as f64;
        let d = self.0.ps.len();

        Array2::from_shape_fn((d, d), |(i, j)| {
            if i == j {
                let p = self.0.ps[i];

                (p * (1.0 - p)) * n
            } else {
                let pi = self.0.ps[i];
                let pj = self.0.ps[j];

                -pi * pj * n
            }
        })
    }

    fn correlation(&self) -> Array2<f64> {
        let d = self.0.ps.len();

        Array2::from_shape_fn((d, d), |(i, j)| {
            let pi = self.0.ps[i];
            let pj = self.0.ps[j];

            -(pi * pj / (1.0 - pi) / (1.0 - pj)).sqrt()
        })
    }
}

impl fmt::Display for Multinomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mult({}; {:?})", self.0.n.0, self.0.ps)
    }
}
