use crate::prelude::*;
use failure::Error;
use ndarray::{Array1, Array2};
use rand::Rng;
use spaces::{ProductSpace, discrete::Ordinal};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Multinomial {
    pub n: usize,
    pub ps: Simplex,
}

impl Multinomial {
    pub fn new(n: usize, ps: Simplex) -> Result<Multinomial, Error> {
        let n = assert_constraint!(n > 0)?;

        Ok(Multinomial::new_unchecked(n, ps))
    }


    pub fn new_unchecked(n: usize, ps: Simplex) -> Multinomial {
        Multinomial { n, ps, }
    }
}

impl Multinomial {
    #[inline]
    pub fn n_categories(&self) -> usize {
        self.ps.len()
    }
}

impl Distribution for Multinomial {
    type Support = ProductSpace<Ordinal>;

    fn support(&self) -> ProductSpace<Ordinal> {
        ProductSpace::new(vec![Ordinal::new(self.n); self.ps.len()])
    }

    fn cdf(&self, _: Vec<usize>) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> Vec<usize> {
        unimplemented!()
    }
}

impl DiscreteDistribution for Multinomial {
    fn pmf(&self, xs: Vec<usize>) -> Probability {
        match xs.iter().fold(0, |acc, x| acc + *x) {
            0 => Probability::zero(),
            _ => Probability::new(self.logpmf(xs).exp()).unwrap(),
        }
    }

    fn logpmf(&self, xs: Vec<usize>) -> f64 {
        let xn = xs.len();

        #[allow(unused_parens)]
        let _ = assert_constraint!(xn == (self.ps.len())).unwrap();

        if xs.iter().fold(0, |acc, v| acc + *v) == self.n {
            panic!("Total number of trials must be equal to n.")
        }

        use special_fun::FloatSpecial;

        let term_1 = (self.n as f64 + 1.0).loggamma();
        let term_2 = xs.iter().zip(self.ps.iter()).fold(0.0, |acc, (&x, &p)| {
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
        self.ps.iter().map(|&p| p * self.n as f64).collect()
    }

    fn variance(&self) -> Array1<f64> {
        self.ps.iter().map(|&p| {
            (p * !p) * self.n as f64
        }).collect()
    }

    fn covariance(&self) -> Array2<f64> {
        let n = self.n as f64;
        let d = self.ps.len();

        Array2::from_shape_fn((d, d), |(i, j)| {
            if i == j {
                let p = self.ps[i];

                (p * !p) * n
            } else {
                let pi = self.ps[i].unwrap();
                let pj = self.ps[j].unwrap();

                -pi * pj * n
            }
        })
    }

    fn correlation(&self) -> Array2<f64> {
        let d = self.ps.len();

        Array2::from_shape_fn((d, d), |(i, j)| {
            let pi = self.ps[i].unwrap();
            let pj = self.ps[j].unwrap();

            -(pi * pj / (1.0 - pi) / (1.0 - pj)).sqrt()
        })
    }
}

impl fmt::Display for Multinomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mult({}; {:?})", self.n, self.ps)
    }
}
