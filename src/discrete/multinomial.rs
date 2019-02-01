use core::*;
use rand::Rng;
use spaces::{Vector, Matrix, discrete::Ordinal, product::LinearSpace};
use std::fmt;

#[derive(Debug, Clone)]
pub struct Multinomial {
    pub n: usize,
    pub ps: Vector<Probability>,
}

impl Multinomial {
    pub fn new<P: Into<Probability>>(n: usize, ps: Vec<P>) -> Multinomial {
        let ps = Vector::from_vec(Probability::normalised(ps));

        Multinomial { n, ps }
    }

    pub fn equiprobable(n: usize, k: usize) -> Multinomial {
        let p = 1.0 / k as f64;

        Multinomial::new(n, vec![Probability(p); k])
    }
}

impl Multinomial {
    #[inline]
    pub fn n_categories(&self) -> usize {
        self.ps.len()
    }
}

impl Distribution for Multinomial {
    type Support = LinearSpace<Ordinal>;

    fn support(&self) -> LinearSpace<Ordinal> {
        LinearSpace::new(vec![Ordinal::new(self.n); self.ps.len()])
    }

    fn cdf(&self, _: Vector<usize>) -> Probability {
        unimplemented!()
    }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> Vector<usize> {
        unimplemented!()
    }
}

impl DiscreteDistribution for Multinomial {
    fn pmf(&self, xs: Vector<usize>) -> Probability {
        match xs.iter().fold(0, |acc, x| acc + *x) {
            0 => Probability(0.0),
            _ => Probability(self.logpmf(xs).exp()),
        }
    }

    fn logpmf(&self, xs: Vector<usize>) -> f64 {
        assert_len!(xs => self.ps.len(); K);

        if xs.iter().fold(0, |acc, v| acc + *v) == self.n {
            panic!("Total number of trials must be equal to n.")
        }

        use special_fun::FloatSpecial;

        let term_1 = (self.n as f64 + 1.0).loggamma();
        let term_2 = xs.iter().zip(self.ps.iter()).fold(0.0, |acc, (&x, p)| {
            let x_f64 = x as f64;
            let xlogy = if x == 0 {
                0.0
            } else {
                x_f64 * p.0.ln()
            };

            acc + xlogy - (x_f64 + 1.0).loggamma()
        });

        (term_1 + term_2).into()
    }
}

impl MultivariateMoments for Multinomial {
    fn mean(&self) -> Vector<f64> {
        self.ps.map(|p| p.0 * self.n as f64)
    }

    fn variance(&self) -> Vector<f64> {
        self.ps.map(|p| p.0 * (p.0 - 1.0) * self.n as f64)
    }

    fn covariance(&self) -> Matrix<f64> {
        let n = self.n as f64;
        let d = self.ps.len();

        Matrix::from_shape_fn((d, d), |(i, j)| {
            if i == j {
                let p = self.ps[i];

                p.0 * (1.0 - p.0) * n
            } else {
                -(self.ps[i] * self.ps[j]).0 * n
            }
        })
    }

    fn correlation(&self) -> Matrix<f64> {
        let d = self.ps.len();

        Matrix::from_shape_fn((d, d), |(i, j)| {
            let pi = self.ps[i].0;
            let pj = self.ps[j].0;

            -(pi * pj / (1.0 - pi) / (1.0 - pj)).sqrt()
        })
    }
}

impl fmt::Display for Multinomial {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mult({}; {})", self.n, self.ps)
    }
}
