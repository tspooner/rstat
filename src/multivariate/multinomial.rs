use crate::{
    statistics::MvMoments,
    DiscreteDistribution,
    Distribution,
    Probability,
    Multivariate,
    SimplexVector,
};
use failure::Error;
use rand::Rng;
use spaces::intervals::Closed;
use std::fmt;

pub use crate::params::Count;

#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Params<const N: usize> {
    pub n: Count<usize>,
    pub ps: SimplexVector<N>,
}

impl<const N: usize> Params<N> {
    pub fn new(n: usize, ps: [f64; N]) -> Result<Params<N>, failure::Error> {
        Ok(Params {
            n: Count::new(n)?,
            ps: SimplexVector::new(ps)?,
        })
    }

    pub fn new_unchecked(n: usize, ps: [f64; N]) -> Params<N> {
        Params {
            n: Count(n),
            ps: SimplexVector::new_unchecked(ps),
        }
    }

    pub fn n(&self) -> &Count<usize> { &self.n }

    pub fn ps(&self) -> &SimplexVector<N> { &self.ps }
}

#[derive(Debug, Clone)]
pub struct Multinomial<const N: usize>(Params<N>);

impl<const N: usize> Multinomial<N> {
    pub fn new(n: usize, ps: [f64; N]) -> Result<Multinomial<N>, Error> {
        Params::new(n, ps).map(Multinomial)
    }

    pub fn new_unchecked(n: usize, ps: [f64; N]) -> Multinomial<N> {
        Multinomial(Params::new_unchecked(n, ps))
    }

    pub const fn n_categories() -> usize { N }
}

impl<const N: usize> From<Params<N>> for Multinomial<N> {
    fn from(params: Params<N>) -> Multinomial<N> { Multinomial(params) }
}

impl<const N: usize> Distribution for Multinomial<N> {
    type Support = [Closed<usize>; N];
    type Params = Params<N>;

    fn support(&self) -> [Closed<usize>; N] {
        let mut support: [std::mem::MaybeUninit<Closed<usize>>; N] = unsafe {
            std::mem::MaybeUninit::uninit().assume_init()
        };

        for i in 0..N {
            let s = Closed::closed_unchecked(0, self.0.n.0);

            support[i].write(s);
        }

        unsafe { (&support as *const _ as *const [Closed<usize>; N]).read() }
    }

    fn params(&self) -> Params<N> { self.0.clone() }

    fn cdf(&self, _: &[usize; N]) -> Probability { unimplemented!() }

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> [usize; N] { unimplemented!() }
}

impl<const N: usize> DiscreteDistribution for Multinomial<N> {
    fn pmf(&self, xs: &[usize; N]) -> Probability {
        match xs.iter().fold(0, |acc, x| acc + x) {
            0 => Probability::zero(),
            _ => Probability::new(self.log_pmf(xs).exp()).unwrap(),
        }
    }

    fn log_pmf(&self, xs: &[usize; N]) -> f64 {
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
            let xlogy = if x == 0 { 0.0 } else { x_f64 * p.ln() };

            acc + xlogy - (x_f64 + 1.0).loggamma()
        });

        (term_1 + term_2).into()
    }
}

impl<const N: usize> Multivariate<N> for Multinomial<N> {}

impl<const N: usize> MvMoments<N> for Multinomial<N> {
    fn mean(&self) -> [f64; N] { self.0.ps.map(|p| p * self.0.n.0 as f64) }

    fn variance(&self) -> [f64; N] { self.0.ps.map(|p| (p * (1.0 - p)) * self.0.n.0 as f64) }

    fn covariance(&self) -> [[f64; N]; N] {
        let n = self.0.n.0 as f64;

        let mut cov = [[0.0; N]; N];

        for i in 0..N {
            for j in 0..i { cov[i][j] = -n * self.0.ps[i] * self.0.ps[j]; }
            for j in (i + 1)..N { cov[i][j] = -n * self.0.ps[i] * self.0.ps[j]; }

            cov[i][i] = n * self.0.ps[i] * (1.0 - self.0.ps[i]);
        }

        cov
    }

    fn correlation(&self) -> [[f64; N]; N] {
        let mut corr = [[1.0; N]; N];

        macro_rules! update_index {
            ($i:ident, $j:ident) => {
                let pi = self.0.ps[$i];
                let pj = self.0.ps[$j];

                corr[$i][$j] = -(pi * pj / (1.0 - pi) / (1.0 - pj)).sqrt();
            }
        }

        for i in 0..N {
            for j in 0..i { update_index!(i, j); }
            for j in (i + 1)..N { update_index!(i, j); }
        }

        corr
    }
}

impl<const N: usize> fmt::Display for Multinomial<N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Mult({}; {:?})", self.0.n.0, self.0.ps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_support() {
        let d = Multinomial::new(2, [0.5, 0.5]).unwrap();
        let s = d.support();

        assert_eq!(s.len(), 2);

        assert_eq!(s[0].left.0, 0);
        assert_eq!(s[0].right.0, 2);

        assert_eq!(s[1].left.0, 0);
        assert_eq!(s[1].right.0, 2);
    }

    #[test]
    fn test_cov() {
        let d = Multinomial::new(5, [0.25, 0.75]).unwrap();
        let cov = d.covariance();

        assert_eq!(cov[0][0], 0.9375);
        assert_eq!(cov[0][1], -0.9375);
        assert_eq!(cov[1][0], -0.9375);
        assert_eq!(cov[1][1], 0.9375);
    }

    #[test]
    fn test_corr() {
        let d = Multinomial::new(5, [0.25, 0.75]).unwrap();
        let cov = d.correlation();

        assert_eq!(cov[0][0], 1.0);
        assert_eq!(cov[0][1], -1.0);
        assert_eq!(cov[1][0], -1.0);
        assert_eq!(cov[1][1], 1.0);
    }
}
