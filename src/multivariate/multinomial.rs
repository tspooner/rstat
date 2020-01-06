use crate::{prelude::*, validation::{UnsatisfiedConstraint}};
use ndarray::{Array1, Array2};
use rand::Rng;
use spaces::{ProductSpace, discrete::Ordinal};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Multinomial {
    pub n: usize,
    pub ps: Vec<Probability>,
}

impl Multinomial {
    pub fn new<P: std::convert::TryInto<Probability>>(n: usize, ps: Vec<P>)
        -> Result<Multinomial, UnsatisfiedConstraint>
    where
        <P as std::convert::TryInto<Probability>>::Error: Into<UnsatisfiedConstraint>,
    {
        ps.into_iter()
            .map(|p| p.try_into().map_err(|e| e.into()))
            .collect::<Result<Vec<Probability>, UnsatisfiedConstraint>>()
            .map(Probability::normalised)
            .map(|ps| Multinomial::new_unchecked(n, ps))
    }


    pub fn new_unchecked(n: usize, ps: Vec<Probability>) -> Multinomial {
        Multinomial { n, ps, }
    }

    pub fn equiprobable(n: usize, k: usize) -> Multinomial {
        let p = Probability::new_unchecked(1.0 / k as f64);

        Multinomial::new_unchecked(n, vec![p; k])
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
        assert_len!(xs => self.ps.len(); K);

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
                x_f64 * f64::from(p).ln()
            };

            acc + xlogy - (x_f64 + 1.0).loggamma()
        });

        (term_1 + term_2).into()
    }
}

impl MultivariateMoments for Multinomial {
    fn mean(&self) -> Array1<f64> {
        self.ps.iter().map(|&p| f64::from(p) * self.n as f64).collect()
    }

    fn variance(&self) -> Array1<f64> {
        self.ps.iter().map(|&p| {
            f64::from(p * !p) * self.n as f64
        }).collect()
    }

    fn covariance(&self) -> Array2<f64> {
        let n = self.n as f64;
        let d = self.ps.len();

        Array2::from_shape_fn((d, d), |(i, j)| {
            if i == j {
                let p = self.ps[i];

                f64::from(p * !p) * n
            } else {
                -f64::from(self.ps[i] * self.ps[j]) * n
            }
        })
    }

    fn correlation(&self) -> Array2<f64> {
        let d = self.ps.len();

        Array2::from_shape_fn((d, d), |(i, j)| {
            let pi = f64::from(self.ps[i]);
            let pj = f64::from(self.ps[j]);

            -(pi * pj / (1.0 - pi) / (1.0 - pj)).sqrt()
        })
    }
}

impl fmt::Display for Multinomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mult({}; {:?})", self.n, self.ps)
    }
}
